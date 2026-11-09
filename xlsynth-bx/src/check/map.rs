// SPDX-License-Identifier: Apache-2.0

//! Typecheck the unary callback of the DSLX map builtin in its lexical scope.

use super::{
    BodyChecker, BodyKind, CheckedBinding, K, MAX_ELABORATION_DEPTH, MapScope, Param, Type,
    TypedExpr, TypedMap, compatible,
};
use crate::ast::{Expr, ForIterable, MapExpr};
use xlsynth_pir::{IrBits, IrValue};

/// The callback spelling determines how to construct the map's pure body.
enum MapCallback<'e> {
    Lambda(&'e MapExpr),
    Named(&'e str),
}

impl BodyChecker<'_, '_> {
    /// Check the array in its caller's scope and the body as a pure function
    /// of one element. The result array keeps the input length.
    pub(super) fn map_lambda(&mut self, map: &MapExpr) -> Result<(Type, K), String> {
        self.check_map_names(map)?;
        let array = self.map_iterable(&map.iterable)?;
        self.map(array, MapCallback::Lambda(map))
    }

    pub(super) fn named_array_map(&mut self, args: &[Expr]) -> Result<(Type, K), String> {
        let [array, callback] = args else {
            return Err(format!(
                "map expects an array and a callback, got {} arguments",
                args.len()
            ));
        };
        let Expr::Name(name) = callback else {
            return Err("map callback must be a named function or an inline lambda".into());
        };
        let array = self.map_array_operand(array)?;
        self.map(array, MapCallback::Named(name))
    }

    pub(super) fn named_range_map(
        &mut self,
        target: &str,
        iterable: &ForIterable,
        callback: &str,
    ) -> Result<(Type, K), String> {
        if !self.is_builtin_map(target, |target| self.names.contains(target))? {
            return Err("a range operand requires the built-in map".into());
        }
        let array = self.map_iterable(iterable)?;
        self.map(array, MapCallback::Named(callback))
    }

    fn map_array_operand(&mut self, expr: &Expr) -> Result<TypedExpr, String> {
        self.with_prepared_value(expr, |checker, expr| {
            checker.elaborate_result(expr, None, false)
        })
    }

    fn map_iterable(&mut self, iterable: &ForIterable) -> Result<TypedExpr, String> {
        // The pinned DSLX evaluator materializes the operand before applying
        // the callback. Its parameter type cannot widen those elements.
        match iterable {
            ForIterable::Array(expr) => self.map_array_operand(expr),
            ForIterable::Range {
                start,
                limit,
                inclusive,
            } => self.with_scoped_results(&[start, limit], |checker, bounds| {
                let range = checker.static_range(bounds[0], bounds[1], None, *inclusive, "map")?;
                checker.map_range_array(range)
            }),
        }
    }

    fn map(&mut self, array: TypedExpr, callback: MapCallback<'_>) -> Result<(Type, K), String> {
        let Type::Array { element, size } = &array.ty else {
            return Err(format!("map requires an array input, got {}", array.ty));
        };
        if *size == 0 {
            return Err("map requires a nonempty array".into());
        }
        let budget = match callback {
            MapCallback::Lambda(_) => "map lambda",
            MapCallback::Named(_) => "map",
        };
        if self.replaying_body() {
            // Account for the callback parameter even if the body ignores it.
            let nodes = self
                .structs
                .type_shape(element, MAX_ELABORATION_DEPTH)?
                .ir_nodes;
            self.charge_expansion(budget, nodes)?;
        }
        let id = self.next_map;
        self.next_map += 1;
        let parameter = Param {
            name: format!("$map_element_{id}"),
            ty: element.as_ref().clone(),
        };
        let (bindings, output) = match callback {
            MapCallback::Lambda(map) => self.map_lambda_body(map, &parameter, id)?,
            MapCallback::Named(name) => {
                let argument = TypedExpr {
                    ty: parameter.ty.clone(),
                    kind: K::Name(parameter.name.clone()),
                };
                (Vec::new(), self.map_element_call(name, argument)?)
            }
        };
        let ty = Type::Array {
            element: Box::new(output.ty.clone()),
            size: *size,
        };
        let shape = self.structs.type_shape(&ty, MAX_ELABORATION_DEPTH)?;
        if self.replaying_body() {
            self.charge_expansion(budget, shape.ir_nodes)?;
        }
        Ok((
            ty,
            K::Map(Box::new(TypedMap::new(array, parameter, bindings, output))),
        ))
    }

    fn map_lambda_body(
        &mut self,
        map: &MapExpr,
        parameter: &Param,
        id: usize,
    ) -> Result<(Vec<CheckedBinding>, TypedExpr), String> {
        if let Some(ty) = &map.parameter_type {
            let ty = self.source_type(ty)?;
            if !compatible(&ty, &parameter.ty) {
                return Err(format!(
                    "map lambda parameter expects {ty}, got {}",
                    parameter.ty
                ));
            }
        }
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
        Ok((bindings, output))
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
