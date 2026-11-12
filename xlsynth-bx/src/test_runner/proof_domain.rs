// SPDX-License-Identifier: Apache-2.0

//! Restrict symbolic parameters to the same values as concrete QuickCheck.

use xlsynth_pir::IrValue;
use xlsynth_pir::ir_builder::{BValue, BuilderError, FnBuilder};

use super::quickcheck::Stimulus;
use crate::ast::Param;
use crate::check::MAX_TYPE_SIZE;

pub(super) fn legal_stimulus(
    builder: &mut FnBuilder,
    params: &[Param],
    values: &[BValue],
    domains: &[Stimulus],
) -> Result<BValue, String> {
    let mut predicates = Vec::new();
    let mut budget = MAX_TYPE_SIZE;
    for ((param, &value), domain) in params.iter().zip(values).zip(domains) {
        domain
            .emit_membership(builder, &[value], &mut predicates, &mut budget)
            .map_err(|e| format!("QuickCheck parameter '{}': {e}", param.name))?;
    }
    match predicates.as_slice() {
        [] => builder.literal(IrValue::bool(true)),
        [one] => Ok(*one),
        many => builder
            .concat(many)
            .and_then(|bits| builder.and_reduce(bits)),
    }
    .map_err(|e| e.to_string())
}

impl Stimulus {
    /// The register portion of a model is arbitrary. Check only stimulus enums.
    pub(super) fn admits(&self, value: &IrValue) -> bool {
        if !self.contains_enum() {
            return true;
        }
        match (self, value) {
            (Self::Enum(members), IrValue::Bits(bits)) => members.contains(bits),
            (Self::Tuple(fields), IrValue::Tuple(values)) => {
                fields.len() == values.len()
                    && fields
                        .iter()
                        .zip(values.iter())
                        .all(|(domain, v)| domain.admits(v))
            }
            (Self::Array { size, element, .. }, IrValue::Array(array)) => {
                array.elements().len() == *size
                    && array.elements().iter().all(|v| element.admits(v))
            }
            _ => false,
        }
    }

    fn contains_enum(&self) -> bool {
        match self {
            Self::Uniform(_) => false,
            Self::Enum(_) => true,
            Self::Tuple(fields) => fields.iter().any(Self::contains_enum),
            Self::Array { element, .. } => element.contains_enum(),
        }
    }

    /// Traverse the declared type once for a whole batch of array elements.
    /// Only project fields and elements whose type actually constrains a value.
    fn emit_membership(
        &self,
        builder: &mut FnBuilder,
        values: &[BValue],
        predicates: &mut Vec<BValue>,
        budget: &mut usize,
    ) -> Result<(), BuilderError> {
        match self {
            Self::Uniform(_) => { /* Every bit pattern is legal. */ }
            Self::Enum(members) => {
                charge_domain(
                    budget,
                    members
                        .len()
                        .saturating_add(2)
                        .saturating_mul(values.len())
                        .saturating_add(members.len()),
                )?;
                let literals = members
                    .iter()
                    .map(|m| builder.literal(IrValue::Bits(m.clone())))
                    .collect::<Result<Vec<_>, _>>()?;
                for &value in values {
                    let matches = literals
                        .iter()
                        .map(|&literal| builder.eq(value, literal))
                        .collect::<Result<Vec<_>, _>>()?;
                    let valid = match matches.as_slice() {
                        [one] => *one,
                        many => {
                            let bits = builder.concat(many)?;
                            builder.or_reduce(bits)?
                        }
                    };
                    predicates.push(valid);
                }
            }
            Self::Tuple(fields) => {
                for (i, field) in fields
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| field.contains_enum())
                {
                    charge_domain(budget, values.len())?;
                    let projected = values
                        .iter()
                        .map(|&v| builder.tuple_index(v, i))
                        .collect::<Result<Vec<_>, _>>()?;
                    field.emit_membership(builder, &projected, predicates, budget)?;
                }
            }
            Self::Array { size, element, .. } if element.contains_enum() => {
                let count = size.saturating_mul(values.len());
                charge_domain(budget, count.saturating_add(*size))?;
                let mut projected = Vec::with_capacity(count);
                for i in 0..*size {
                    // A constrained array fits the expansion budget above.
                    let index = builder.literal(IrValue::u32(i as u32))?;
                    for &v in values {
                        projected.push(builder.array_index(v, index)?);
                    }
                }
                element.emit_membership(builder, &projected, predicates, budget)?;
            }
            Self::Array { .. } => { /* No enum restrictions anywhere in the element. */ }
        }
        Ok(())
    }
}

fn charge_domain(budget: &mut usize, count: usize) -> Result<(), BuilderError> {
    *budget = budget.checked_sub(count).ok_or_else(|| {
        BuilderError::InvalidOperation("enum domain exceeds the proof expansion limit".into())
    })?;
    Ok(())
}
