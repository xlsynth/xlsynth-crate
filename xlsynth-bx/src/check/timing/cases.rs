// SPDX-License-Identifier: Apache-2.0

//! Evaluate a stage helper for every hardware encoding of its control inputs.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use xlsynth_pir::{IrBits, IrValue};

use super::{Join, Stage, TimedValue};
use crate::ast::{BinaryOp, Type};
use crate::check::{MAX_TYPE_SIZE, compatible};
use crate::typed::{
    AvailableFunctions, CheckedBlock, CheckedEnum, CheckedStmt, ConstantBinding, ConstantCallees,
    ConstantFunctions, ConstantUse, TypedExpr, TypedExprKind as K, bits,
};

const MAX_CASES: usize = 64;

/// One set of control values and the output stage promised for that case.
pub(super) struct Case<'a> {
    pub inputs: BTreeMap<String, IrValue>,
    pub stage: u64,
    pub label: String,
    enums: &'a BTreeMap<String, CheckedEnum>,
    /// Reset can override the normal write, even when its enable is true.
    reset_registers: Rc<BTreeSet<&'a str>>,
    remaining: Cell<usize>,
}

/// Enumerate every hardware encoding, keeping enum spellings for diagnostics.
fn domain(
    ty: &Type,
    enums: &BTreeMap<String, CheckedEnum>,
) -> Result<Vec<(String, IrValue)>, String> {
    let (width, enumeration) = match ty {
        Type::Bool
        | Type::Bits {
            signed: false,
            width: 1,
        } => {
            return Ok([false, true]
                .into_iter()
                .map(|value| (value.to_string(), IrValue::bool(value)))
                .collect());
        }
        Type::Enum { name, width, .. } => (*width, Some(&enums[name])),
        Type::Bits {
            signed: false,
            width,
        } => (*width, None),
        _ => {
            return Err(format!(
                "stage input must be bool, an enum or small unsigned bits, got {ty}"
            ));
        }
    };
    if width > MAX_CASES.ilog2() as usize {
        return Err(format!("stage input domain exceeds {MAX_CASES} cases"));
    }

    // Enum ports are bits in RTL, and runtime casts can produce unnamed
    // encodings. Check those too; checking only declared members is unsound.
    let mut members = HashMap::new();
    if let Some(def) = enumeration {
        for (name, bits) in &def.members {
            members.entry(bits.clone()).or_insert(name);
        }
    }
    (0..1u64 << width)
        .map(|raw| {
            let bits = IrBits::make_ubits(width, raw).map_err(|e| e.to_string())?;
            let label = match enumeration {
                Some(def) => members.get(&bits).map_or_else(
                    || format!("{}({raw})", def.name),
                    |member| format!("{}::{member}", def.name),
                ),
                None => raw.to_string(),
            };
            Ok((label, IrValue::Bits(bits)))
        })
        .collect()
}

/// Resolve and evaluate the ordinary pure BX helper once per control case.
pub(super) fn enumerate<'a>(
    block: &'a CheckedBlock,
    name: &str,
    inputs: &[String],
    functions: &AvailableFunctions,
    enums: &'a BTreeMap<String, CheckedEnum>,
) -> Result<Vec<Case<'a>>, String> {
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
    // Every case checks the same hardware; index resettable registers once.
    let reset_registers: Rc<BTreeSet<_>> = Rc::new(
        block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                CheckedStmt::Reg {
                    name,
                    reset: Some(_),
                    ..
                } => Some(name.as_str()),
                _ => None,
            })
            .collect(),
    );
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
                ConstantCallees {
                    local: Some(&mut evaluator),
                    imports: None,
                },
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
            enums,
            reset_registers: Rc::clone(&reset_registers),
            remaining: Cell::new(MAX_TYPE_SIZE),
        });
    }
    Ok(cases)
}

impl Case<'_> {
    /// Apply the stage rule and retain bits only when the hardware guarantees
    /// them in this case. A held or resettable register has no exact value.
    pub(super) fn join(
        &self,
        rule: Join<'_>,
        left: TimedValue,
        right: TimedValue,
    ) -> Result<TimedValue, String> {
        if matches!(rule, Join::Select(_)) && right.known.is_some() {
            // Both arms are the same known value, so the selector is
            // irrelevant.
            return Ok(right);
        }
        let stage = rule.apply(left.stage, right.stage)?;
        let known = match rule {
            Join::Register(name)
                if right.boolean() == Some(true) && !self.reset_registers.contains(name) =>
            {
                left.known
            }
            Join::Guards => match (left.boolean(), right.boolean()) {
                (Some(a), Some(b)) => Some(IrValue::bool(a && b)),
                _ => None,
            },
            Join::Branches(_) if left.known.is_some() && left.known == right.known => left.known,
            Join::Selected(_) | Join::Select(_) => right.known,
            _ => None,
        };
        Ok(TimedValue { stage, known })
    }

    /// Use BX's constexpr evaluator on the already-known operands. It accepts
    /// expression trees, so replace only this operation's operands with
    /// literals; never walk its original dependencies a second time.
    pub(super) fn fold_operation(
        &self,
        expr: &TypedExpr,
        args: &[&TimedValue],
    ) -> Result<Option<IrValue>, String> {
        let mut args = args.iter().map(|arg| arg.known.as_ref());
        let Some(first) = args.next().flatten() else {
            return Ok(None);
        };
        let literal = |original: &TypedExpr, value: &IrValue| {
            Box::new(TypedExpr {
                ty: original.ty.clone(),
                kind: K::Literal(value.clone()),
            })
        };
        let kind = match &expr.kind {
            K::Unary { op, value } => K::Unary {
                op: *op,
                value: literal(value, first),
            },
            K::Cast { value } => K::Cast {
                value: literal(value, first),
            },
            K::Binary { op, left, right } => {
                let Some(second) = args.next().flatten() else {
                    return Ok(None);
                };
                K::Binary {
                    op: *op,
                    left: literal(left, first),
                    right: literal(right, second),
                }
            }
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
            ConstantCallees::default(),
            self.enums,
        );
        self.remaining.set(remaining);
        result
            .map(|fold| fold.map(|fold| fold.value))
            .map_err(|error| format!("stage case evaluation failed: {error:?}"))
    }

    /// A Boolean true absorbs OR, and false absorbs AND, at the stage where
    /// that bit is known. A later register or child can advance that stage.
    pub(super) fn binary(
        &self,
        expr: &TypedExpr,
        left: TimedValue,
        right: TimedValue,
    ) -> Result<TimedValue, String> {
        let K::Binary { op, .. } = &expr.kind else {
            unreachable!()
        };
        let absorbing = match op {
            BinaryOp::Or | BinaryOp::LogicalOr => Some(true),
            BinaryOp::And | BinaryOp::LogicalAnd => Some(false),
            _ => None,
        };
        if let Some(bit) = absorbing {
            // Prefer the promised stage when both operands decide the result
            // at different stages; otherwise retain the intermediate stage.
            let mut intermediate = None;
            for value in [&left, &right] {
                if value.boolean() != Some(bit) {
                    continue;
                }
                if value.stage == Stage::Independent || value.stage == Stage::At(self.stage) {
                    return Ok(value.clone());
                }
                if matches!(value.stage, Stage::At(_)) && intermediate.is_none() {
                    intermediate = Some(value);
                }
            }
            if let Some(value) = intermediate {
                return Ok(value.clone());
            }
        }
        let stage = Join::Operator(*op).apply(left.stage, right.stage)?;
        let known = self.fold_operation(expr, &[&left, &right])?;
        Ok(TimedValue { stage, known })
    }
}
