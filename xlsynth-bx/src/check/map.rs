// SPDX-License-Identifier: Apache-2.0

//! Typecheck the unary callback of the DSLX map builtin in its lexical scope.

use super::{
    BodyChecker, BodyKind, K, MAX_ELABORATION_DEPTH, MapScope, Param, Type, TypedExpr, TypedMap,
    compatible,
};
use crate::ast::{ForIterable, MapExpr};
use xlsynth_pir::{IrBits, IrValue};

impl BodyChecker<'_, '_> {
    /// Check the array in its caller's scope and the body as a pure function
    /// of one element. The result array keeps the input length.
    pub(super) fn map_lambda(&mut self, map: &MapExpr) -> Result<(Type, K), String> {
        self.check_map_names(map)?;
        let annotation = map
            .parameter_type
            .as_ref()
            .map(|ty| self.source_type(ty))
            .transpose()?;
        // The pinned DSLX evaluator materializes the operand before applying
        // the callback. Its parameter annotation cannot widen those elements.
        let array = match &map.iterable {
            ForIterable::Array(expr) => self.with_prepared_value(expr, |checker, expr| {
                checker.elaborate_result(expr, None, false)
            })?,
            ForIterable::Range {
                start,
                limit,
                inclusive,
            } => self.with_scoped_results(&[start, limit], |checker, bounds| {
                let range = checker.static_range(bounds[0], bounds[1], None, *inclusive, "map")?;
                checker.map_range_array(range)
            })?,
        };
        let Type::Array { element, size } = &array.ty else {
            return Err(format!("map requires an array input, got {}", array.ty));
        };
        if *size == 0 {
            return Err("map requires a nonempty array".into());
        }
        if let Some(ty) = &annotation
            && !compatible(ty, element)
        {
            return Err(format!("map lambda parameter expects {ty}, got {element}"));
        }
        if self.replaying_body() {
            // A generated function has an IR parameter even when the lambda
            // ignores its argument.
            let nodes = self
                .structs
                .type_shape(element, MAX_ELABORATION_DEPTH)?
                .ir_nodes;
            self.charge_expansion("map lambda", nodes)?;
        }
        let id = self.next_map;
        self.next_map += 1;
        let parameter = Param {
            name: format!("$map_element_{id}"),
            ty: element.as_ref().clone(),
        };
        let start = self.statements.len();
        // Lambda-local bindings have function-local inference even when the
        // callback captures a loop element or accumulator.
        let outer_kind = std::mem::replace(&mut self.kind, BodyKind::Function);
        let outer_map = self.map_scope.replace(MapScope {
            id,
            outer_iterations: self.iteration_path.len(),
        });
        let outer_counted = std::mem::take(&mut self.in_counted_body);
        let mut saved = Vec::new();
        self.bind_scoped_value(
            &map.parameter,
            TypedExpr {
                ty: parameter.ty.clone(),
                kind: K::Name(parameter.name.clone()),
            },
            None,
            &mut saved,
        );
        let output = (|| {
            let result = map
                .result_type
                .as_ref()
                .map(|ty| self.source_type(ty))
                .transpose()?;
            self.scoped_result(&map.body, result.as_ref(), false)
        })();
        self.leave_scope(saved);
        self.in_counted_body = outer_counted;
        self.map_scope = outer_map;
        self.kind = outer_kind;
        let bindings = self.take_pure_bindings(start);
        let output = output.map_err(|e| format!("map lambda body: {e}"))?;
        let ty = Type::Array {
            element: Box::new(output.ty.clone()),
            size: *size,
        };
        let shape = self.structs.type_shape(&ty, MAX_ELABORATION_DEPTH)?;
        if self.replaying_body() {
            self.charge_expansion("map lambda", shape.ir_nodes)?;
        }
        Ok((
            ty,
            K::Map(Box::new(TypedMap::new(array, parameter, bindings, output))),
        ))
    }

    /// DSLX ranges used as map operands are constant arrays. Check the
    /// expanded layout before allocating their elements.
    fn map_range_array(&mut self, range: super::unroll::StaticRange) -> Result<TypedExpr, String> {
        if range.count == 0 {
            return Err("map requires a nonempty array".into());
        }
        let ty = Type::Array {
            element: Box::new(range.ty),
            size: range.count,
        };
        let shape = self.structs.type_shape(&ty, MAX_ELABORATION_DEPTH)?;
        self.charge_expansion(
            "map range",
            range.count.saturating_add(shape.size.div_ceil(64)),
        )?;
        let mut index = range.first;
        let one = IrBits::from_lsb_fn(index.get_bit_count(), |i| i == 0);
        let mut elements = Vec::with_capacity(range.count);
        for _ in 0..range.count {
            let next = index.add(&one);
            elements.push(IrValue::Bits(index));
            index = next;
        }
        let value = IrValue::make_array(&elements).map_err(|e| e.to_string())?;
        Ok(TypedExpr {
            ty,
            kind: K::Literal(value),
        })
    }
}
