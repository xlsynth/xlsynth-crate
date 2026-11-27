// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use super::{BodyChecker, K, MAX_ELABORATION_DEPTH, StructTable, unknown_declaration};
use crate::ast::{Expr, Type};
use crate::builtin::BuiltinOp;

/// The packed width and work needed to visit members and emit their
/// projections.
#[derive(Clone, Copy, Default)]
struct PackedLayout {
    width: usize,
    work: usize,
}

impl PackedLayout {
    /// Check nominal eligibility at each nested struct. Concrete type checking
    /// already bounds the expanded shape, including array repetitions and bit
    /// widths, by MAX_TYPE_SIZE. Each member adds at most two projections.
    fn for_type(
        ty: &Type,
        structs: &StructTable<'_>,
        records: &mut BTreeMap<usize, Self>,
    ) -> Result<Self, String> {
        match ty {
            Type::Bool => Ok(Self { width: 1, work: 0 }),
            Type::Bits { width, .. } | Type::Enum { width, .. } => Ok(Self {
                width: *width,
                work: 0,
            }),
            Type::Array { size: 0, .. } => Ok(Self::default()),
            Type::Array { element, size } => {
                let element = Self::for_type(element, structs, records)?;
                Ok(Self {
                    width: size * element.width,
                    // Empty elements need no projections. Otherwise every
                    // element needs an index literal and an array_index.
                    work: if element.width == 0 {
                        0
                    } else {
                        size * (element.work + 2)
                    },
                })
            }
            Type::Tuple(fields) => Self::for_fields(fields.iter(), structs, records),
            Type::Named(name) => {
                let &(index, _) = structs
                    .indices
                    .get(name)
                    .ok_or_else(|| unknown_declaration("struct", name))?;
                if let Some(layout) = records.get(&index) {
                    return Ok(*layout);
                }
                let def = &structs.definitions[index];
                if !def.derives_to_bits {
                    return Err(format!("struct '{name}' does not derive ToBits"));
                }
                if def.fields.iter().any(|field| field.name == "to_bits") {
                    return Err(format!(
                        "field '{name}.to_bits' conflicts with the derived to_bits() method"
                    ));
                }
                let layout = Self::for_fields(def.fields.iter().map(|f| &f.ty), structs, records)?;
                records.insert(index, layout);
                Ok(layout)
            }
            _ => Err(format!("cannot pack unresolved type '{ty}' with ToBits")),
        }
    }

    fn for_fields<'a>(
        fields: impl Iterator<Item = &'a Type>,
        structs: &StructTable<'_>,
        records: &mut BTreeMap<usize, Self>,
    ) -> Result<Self, String> {
        let mut layout = Self::default();
        for field in fields {
            let field = Self::for_type(field, structs, records)?;
            layout.width += field.width;
            // Lowering inspects every field, including fields with no bits.
            layout.work += field.work + 1;
        }
        Ok(layout)
    }
}

impl BodyChecker<'_, '_> {
    /// Resolve the derived receiver method after concrete struct
    /// specialization.
    pub(super) fn derived_to_bits(&mut self, receiver: &Expr) -> Result<(Type, K), String> {
        let value = self.expr(receiver, None)?;
        if !matches!(value.ty, Type::Named(_)) {
            return Err(format!(
                "to_bits() requires a struct receiver, got {}",
                value.ty
            ));
        }
        // Both the eligibility check and IR type inspection walk the concrete
        // layout, even when no field contributes a bit.
        let nodes = self
            .structs
            .type_shape(&value.ty, MAX_ELABORATION_DEPTH)?
            .ir_nodes;
        self.charge_expansion("to_bits()", nodes)?;
        let mut records = BTreeMap::new();
        let layout = PackedLayout::for_type(&value.ty, self.structs, &mut records)?;
        self.charge_expansion("to_bits()", layout.work + 1)?;
        for (index, record) in records {
            self.resolver
                .check_to_bits_derive(&self.structs.definitions[index].name, record.width)?;
        }
        let work = nodes + layout.work + 1;
        Ok((
            Type::unsigned(layout.width),
            K::Builtin {
                op: BuiltinOp::ToBits { work },
                args: vec![value],
            },
        ))
    }
}
