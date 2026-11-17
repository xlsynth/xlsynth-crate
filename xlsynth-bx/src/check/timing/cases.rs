// SPDX-License-Identifier: Apache-2.0

//! Evaluate a stage helper for every legal combination of its control inputs.

use std::collections::{BTreeMap, HashSet};
use xlsynth_pir::IrValue;

use super::{Join, Stage, StageChecker, TimedValue};
use crate::ast::{BinaryOp, Type};
use crate::check::{MAX_TYPE_SIZE, compatible};
use crate::typed::{
    AvailableFunctions, CheckedBlock, CheckedEnum, ConstantBinding, ConstantFunctions, ConstantUse,
    TypedExpr, TypedExprKind as K, bits,
};

const MAX_CASES: usize = 64;

/// One set of control values and the output stage promised for that case.
pub(super) struct Case {
    pub inputs: BTreeMap<String, IrValue>,
    pub stage: u64,
    pub label: String,
}

/// Enumerate each distinct hardware value, keeping enum spellings for errors.
fn domain(
    ty: &Type,
    enums: &BTreeMap<String, CheckedEnum>,
) -> Result<Vec<(String, IrValue)>, String> {
    match ty {
        Type::Bool
        | Type::Bits {
            signed: false,
            width: 1,
        } => Ok([false, true]
            .into_iter()
            .map(|value| (value.to_string(), IrValue::bool(value)))
            .collect()),
        Type::Enum { name, .. } => {
            let def = &enums[name];
            if def.value_count() > MAX_CASES {
                return Err(format!("stage input domain exceeds {MAX_CASES} cases"));
            }
            let mut seen = HashSet::new();
            Ok(def
                .members
                .iter()
                .filter(|(_, bits)| seen.insert(*bits))
                .map(|(member, bits)| (format!("{name}::{member}"), IrValue::Bits(bits.clone())))
                .collect())
        }
        Type::Bits {
            signed: false,
            width,
        } if *width <= MAX_CASES.ilog2() as usize => (0..1u64 << width)
            .map(|value| {
                Ok((
                    value.to_string(),
                    IrValue::make_ubits(*width, value).map_err(|e| e.to_string())?,
                ))
            })
            .collect(),
        Type::Bits { signed: false, .. } => {
            Err(format!("stage input domain exceeds {MAX_CASES} cases"))
        }
        _ => Err(format!(
            "stage input must be bool, an enum or small unsigned bits, got {ty}"
        )),
    }
}

/// Resolve and evaluate the ordinary pure BX helper once per control case.
pub(super) fn enumerate(
    block: &CheckedBlock,
    name: &str,
    inputs: &[String],
    functions: &AvailableFunctions,
    enums: &BTreeMap<String, CheckedEnum>,
) -> Result<Vec<Case>, String> {
    let function = functions
        .borrow()
        .get(name)
        .cloned()
        .ok_or_else(|| format!("stage helper '{name}' must be a concrete pure BX function"))?;
    if inputs.is_empty() || inputs.len() != function.params.len() {
        return Err(format!(
            "stage helper '{name}' requires {} arguments and at least one control input",
            function.params.len()
        ));
    }
    if !matches!(bits(&function.result), Some((false, _))) {
        return Err(format!(
            "stage helper '{name}' must return an unsigned integer"
        ));
    }
    let mut count = 1;
    let mut domains = BTreeMap::new();
    let mut args = Vec::new();
    for (input, param) in inputs.iter().zip(&function.params) {
        let port = block
            .params
            .iter()
            .find(|p| p.name == *input)
            .ok_or_else(|| format!("unknown stage input '{input}'"))?;
        if !compatible(&port.ty, &param.ty) {
            return Err(format!(
                "stage helper '{name}' expects {} for '{input}', got {}",
                param.ty, port.ty
            ));
        }
        if !domains.contains_key(input) {
            let values = domain(&port.ty, enums)?;
            if values.is_empty() || count * values.len() > MAX_CASES {
                return Err(format!(
                    "stage input domain must contain 1 to {MAX_CASES} cases"
                ));
            }
            count *= values.len();
            domains.insert(input, values);
        }
        args.push(TypedExpr {
            ty: port.ty.clone(),
            kind: K::Name(input.clone()),
        });
    }
    let call = TypedExpr {
        ty: function.result.clone(),
        kind: K::Call {
            name: name.to_owned(),
            args,
        },
    };
    let mut evaluator = ConstantFunctions::new(functions.clone());
    let mut remaining = MAX_TYPE_SIZE;
    let mut cases = Vec::with_capacity(count);
    for index in 0..count {
        let mut index = index;
        let mut values = BTreeMap::new();
        let mut names = BTreeMap::new();
        let mut labels = Vec::new();
        for (input, domain) in &domains {
            let (label, value) = &domain[index % domain.len()];
            index /= domain.len();
            values.insert((*input).clone(), value.clone());
            names.insert((*input).clone(), ConstantBinding::Known(value.clone()));
            labels.push(format!("{input}={label}"));
        }
        let stage = call
            .fold(
                &names,
                &mut remaining,
                true,
                ConstantUse::Bx,
                Some(&mut evaluator),
                enums,
            )
            .map_err(|e| format!("stage helper evaluation failed: {e:?}"))?
            .ok_or("stage helper did not evaluate to a constant")?
            .value
            .to_u64()
            .map_err(|_| "stage helper result does not fit in u64")?;
        cases.push(Case {
            inputs: values,
            stage,
            label: labels.join(", "),
        });
    }
    Ok(cases)
}

impl StageChecker<'_> {
    /// Evaluate BX's ordinary constexpr rules on known operands. Bits are
    /// combined only after the stage rule has established that they line up.
    pub(super) fn fold_operation(
        &self,
        expr: &TypedExpr,
        args: &[&TimedValue],
    ) -> Result<Option<IrValue>, String> {
        if self.case.is_none() || args.iter().any(|arg| arg.known.is_none()) {
            return Ok(None);
        }
        let literal = |original: &TypedExpr, arg: &TimedValue| {
            Box::new(TypedExpr {
                ty: original.ty.clone(),
                kind: K::Literal(arg.known.as_ref().expect("known operand").clone()),
            })
        };
        let kind = match &expr.kind {
            K::Unary { op, value } => K::Unary {
                op: *op,
                value: literal(value, args[0]),
            },
            K::Cast { value } => K::Cast {
                value: literal(value, args[0]),
            },
            K::Binary { op, left, right } => K::Binary {
                op: *op,
                left: literal(left, args[0]),
                right: literal(right, args[1]),
            },
            _ => unreachable!("only unary and binary operations are folded here"),
        };
        let mut remaining = self.remaining.get();
        let result = TypedExpr {
            ty: expr.ty.clone(),
            kind,
        }
        .fold(
            &BTreeMap::new(),
            &mut remaining,
            true,
            ConstantUse::Bx,
            None,
            &self.structs.enums,
        );
        self.remaining.set(remaining);
        result
            .map(|fold| fold.map(|fold| fold.value))
            .map_err(|error| format!("stage case evaluation failed: {error:?}"))
    }

    /// A Boolean true absorbs OR, and false absorbs AND. The deciding input
    /// must belong to this case's promised stage; a value known for an earlier
    /// stage says nothing about the opcode currently at the output.
    pub(super) fn binary(
        &self,
        expr: &TypedExpr,
        left: TimedValue,
        right: TimedValue,
    ) -> Result<TimedValue, String> {
        let K::Binary { op, .. } = &expr.kind else {
            unreachable!()
        };
        if let Some(case) = &self.case {
            let absorbing = match op {
                BinaryOp::Or | BinaryOp::LogicalOr => Some(true),
                BinaryOp::And | BinaryOp::LogicalAnd => Some(false),
                _ => None,
            };
            if let Some(bit) = absorbing {
                for value in [&left, &right] {
                    if matches!(value.stage, Stage::Independent)
                        || value.stage == Stage::At(case.stage)
                    {
                        if value.boolean() == Some(bit) {
                            return Ok(value.clone());
                        }
                    }
                }
            }
        }
        let stage = Join::Operator(*op).apply(left.stage, right.stage)?;
        let known = self.fold_operation(expr, &[&left, &right])?;
        Ok(TimedValue { stage, known })
    }
}
