// SPDX-License-Identifier: Apache-2.0

//! Check one issue clock at a time. Only that clock's payload and controls
//! may affect the result at its declared deadline. Other clocks and the
//! initial register contents are arbitrary.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use xlsynth_pir::IrValue;

use crate::ast::Type;
use crate::check::{MAX_TYPE_SIZE, compatible};
use crate::typed::{
    AvailableFunctions, CheckedBlock, CheckedEnum, CheckedStmt, ConstantBinding, ConstantFunctions,
    ConstantUse, TypedExpr, TypedExprKind as K, bits, is_constexpr_only_name,
};

const MAX_CASES: usize = 64;
const MAX_LATENCY: u64 = 64;

/// Facts about a signal for one particular issue case at one particular clock.
#[derive(Clone, PartialEq)]
enum Signal {
    Known(IrValue),
    /// May vary with the payload or other data inputs on the issue clock.
    Issued,
    /// Could depend on another clock or on the initial register contents.
    Unknown,
}

impl Signal {
    fn boolean(&self) -> Option<bool> {
        match self {
            Self::Known(value) => value.to_bool().ok(),
            _ => None,
        }
    }

    /// An unknown select is harmless if both arms give the same known value.
    /// Two issue-dependent arms can still differ, so their select matters.
    fn select(condition: &Self, yes: Self, no: Self) -> Self {
        match condition.boolean() {
            Some(true) => yes,
            Some(false) => no,
            None if matches!(yes, Self::Known(_)) && yes == no => yes,
            None if *condition != Self::Unknown && yes != Self::Unknown && no != Self::Unknown => {
                Self::Issued
            }
            None => Self::Unknown,
        }
    }
}

/// Lexical values at this clock. The constexpr evaluator reads exact values.
/// A data input sampled at issue is safe to use even when its bits are unknown.
#[derive(Default)]
struct Frame {
    known: BTreeMap<String, ConstantBinding>,
    issued: BTreeSet<String>,
}

impl Frame {
    fn read(&self, name: &str) -> Signal {
        if let Some(ConstantBinding::Known(value)) = self.known.get(name) {
            Signal::Known(value.clone())
        } else if self.issued.contains(name) {
            Signal::Issued
        } else {
            Signal::Unknown
        }
    }

    fn bind(&mut self, name: &str, signal: Signal) {
        if signal == Signal::Unknown {
            self.issued.remove(name);
        } else {
            self.issued.insert(name.to_owned());
        }
        if let Signal::Known(value) = signal {
            self.known
                .insert(name.to_owned(), ConstantBinding::Known(value));
        } else {
            self.known.remove(name);
        }
    }
}

/// Shared constexpr cache and bounded work for all cases of one annotation.
struct Proof<'a> {
    enums: &'a BTreeMap<String, CheckedEnum>,
    functions: ConstantFunctions,
    remaining: usize,
}

impl Proof<'_> {
    fn charge(&mut self, amount: usize) -> Result<(), String> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or("issue timing work limit exceeded")?;
        Ok(())
    }

    fn constant(&mut self, expr: &TypedExpr, frame: &Frame) -> Result<Option<IrValue>, String> {
        expr.fold(
            &frame.known,
            &mut self.remaining,
            true,
            ConstantUse::Bx,
            Some(&mut self.functions),
            self.enums,
        )
        .map(|result| result.map(|fold| fold.value))
        .map_err(|error| format!("issue timing constant evaluation failed: {error:?}"))
    }

    /// Pick a known mux arm before collecting its dependencies. In particular,
    /// an opcode carried through a register stays known for this issue case.
    fn expr(&mut self, expr: &TypedExpr, frame: &Frame) -> Result<Signal, String> {
        self.charge(1)?;
        match &expr.kind {
            K::Name(name) => return Ok(frame.read(name)),
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.expr(condition, frame)?;
                return match condition.boolean() {
                    Some(true) => self.expr(then_value, frame),
                    Some(false) => self.expr(else_value, frame),
                    None => Ok(Signal::select(
                        &condition,
                        self.expr(then_value, frame)?,
                        self.expr(else_value, frame)?,
                    )),
                };
            }
            K::Match { subject, arms } => {
                if let Signal::Known(value) = self.expr(subject, frame)? {
                    for arm in arms {
                        for pattern in &arm.patterns {
                            if pattern.matches_value(&value, &subject.ty)? {
                                return self.expr(&arm.value, frame);
                            }
                        }
                    }
                    return Err("latency datapath match has no arm for the selected value".into());
                }
            }
            _ => { /* Other pure expressions use the regular BX evaluator below. */ }
        }
        Ok(if let Some(value) = self.constant(expr, frame)? {
            Signal::Known(value)
        } else if expr.has_constant_inputs(&frame.issued) {
            Signal::Issued
        } else {
            Signal::Unknown
        })
    }

    /// Writes all read the current frame and become visible together at the
    /// next edge. When a guard is false the register keeps its old value.
    fn statements<'a>(
        &mut self,
        stmts: &'a [CheckedStmt],
        frame: &mut Frame,
        gate: &Signal,
        next: &mut BTreeMap<&'a str, Signal>,
        locals: &mut Vec<(&'a str, Signal)>,
    ) -> Result<(), String> {
        for stmt in stmts {
            self.charge(1)?;
            match stmt {
                CheckedStmt::Reg { .. } => { /* An initial register read is unknown. */ }
                // Assertion-only bindings have no signal in the datapath.
                CheckedStmt::Let { name, .. } if is_constexpr_only_name(name) => continue,
                CheckedStmt::Let { name, value } => {
                    let signal = self.expr(value, frame)?;
                    locals.push((name, frame.read(name)));
                    frame.bind(name, signal);
                }
                CheckedStmt::Write { name, value } => {
                    let data = self.expr(value, frame)?;
                    next.insert(name, Signal::select(gate, data, frame.read(name)));
                }
                CheckedStmt::If { condition, body } => {
                    let condition = self.expr(condition, frame)?;
                    let enabled = Signal::select(gate, condition, Signal::Known(IrValue::bool(false)));
                    if enabled.boolean() != Some(false) {
                        let start = locals.len();
                        self.statements(body, frame, &enabled, next, locals)?;
                        for (name, old) in locals.drain(start..).rev() {
                            frame.bind(name, old);
                        }
                    }
                }
                _ => return Err("issue timing currently requires local registers and lets; child instances and deferred wires are unsupported".into()),
            }
        }
        Ok(())
    }

    fn check_case(
        &mut self,
        block: &CheckedBlock,
        mut frame: Frame,
        latency: &TypedExpr,
    ) -> Result<(), String> {
        let deadline = self
            .constant(latency, &frame)?
            .ok_or("latency helper did not evaluate to a constant")?
            .to_u64()
            .map_err(|_| "latency does not fit in u64")?;
        if deadline > MAX_LATENCY {
            return Err(format!(
                "latency +{deadline} exceeds the {MAX_LATENCY}-clock proof limit"
            ));
        }
        let mut registers: BTreeMap<&str, Signal> = BTreeMap::new();
        for clock in 0..=deadline {
            // On later clocks only the registers remember anything about issue.
            if clock > 0 {
                frame = Frame::default();
            }
            self.charge(registers.len() + 1)?;
            for (name, value) in &registers {
                frame.bind(name, value.clone());
            }
            self.statements(
                &block.statements,
                &mut frame,
                &Signal::Known(IrValue::bool(true)),
                &mut registers,
                &mut Vec::new(),
            )?;
        }
        if self.expr(&block.output, &frame)? == Signal::Unknown {
            return Err(format!(
                "cannot prove the output at +{deadline} depends only on this issue; it may read another clock or old register contents"
            ));
        }
        Ok(())
    }
}

/// Enumerate legal hardware values in stable order and retain source names
/// for a failing case. Aliases of an enum member denote the same input bits.
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
            .map(|b| (b.to_string(), IrValue::bool(b)))
            .collect()),
        Type::Enum { name, .. } => {
            let def = &enums[name];
            if def.value_count() > MAX_CASES {
                return Err(format!("latency input domain exceeds {MAX_CASES} cases"));
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
            .map(|n| {
                Ok((
                    n.to_string(),
                    IrValue::make_ubits(*width, n).map_err(|e| e.to_string())?,
                ))
            })
            .collect(),
        Type::Bits { signed: false, .. } => {
            Err(format!("latency input domain exceeds {MAX_CASES} cases"))
        }
        _ => Err(format!(
            "latency input must be bool, an enum or small unsigned bits, got {ty}"
        )),
    }
}

/// Evaluate the user's ordinary BX helper on each issue-input combination,
/// then check the already typed block at the resulting deadline.
pub(super) fn check(
    block: &CheckedBlock,
    name: &str,
    inputs: &[String],
    functions: &AvailableFunctions,
    enums: &BTreeMap<String, CheckedEnum>,
) -> Result<(), String> {
    if block.reset.is_some() || block.input_cycles.values().any(|&stage| stage != 0) {
        return Err(
            "issue timing currently requires a reset-free block with stage-0 inputs".into(),
        );
    }
    let function =
        functions.borrow().get(name).cloned().ok_or_else(|| {
            format!("latency helper '{name}' must be a concrete pure BX function")
        })?;
    if inputs.is_empty() || inputs.len() != function.params.len() {
        return Err(format!(
            "latency helper '{name}' requires {} arguments and at least one issue input",
            function.params.len()
        ));
    }
    if !matches!(bits(&function.result), Some((false, _))) {
        return Err(format!(
            "latency helper '{name}' must return an unsigned integer"
        ));
    }
    let mut cases = 1;
    let mut domains = BTreeMap::new();
    let mut args = Vec::new();
    for (input, param) in inputs.iter().zip(&function.params) {
        let port = block
            .params
            .iter()
            .find(|p| p.name == *input)
            .ok_or_else(|| format!("unknown issue input '{input}'"))?;
        if !compatible(&port.ty, &param.ty) {
            return Err(format!(
                "latency helper '{name}' expects {} for '{input}', got {}",
                param.ty, port.ty
            ));
        }
        if !domains.contains_key(input) {
            let values = domain(&port.ty, enums)?;
            if values.is_empty() || cases * values.len() > MAX_CASES {
                return Err(format!(
                    "latency input domain must contain 1 to {MAX_CASES} cases"
                ));
            }
            cases *= values.len();
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
    let mut proof = Proof {
        enums,
        functions: ConstantFunctions::new(functions.clone()),
        remaining: MAX_TYPE_SIZE,
    };
    for case in 0..cases {
        let mut frame = Frame::default();
        frame
            .issued
            .extend(block.params.iter().map(|p| p.name.clone()));
        let mut index = case;
        let mut labels = Vec::new();
        for (input, values) in &domains {
            let (label, value) = &values[index % values.len()];
            index /= values.len();
            frame.bind(input, Signal::Known(value.clone()));
            labels.push(format!("{input}={label}"));
        }
        proof
            .check_case(block, frame, &call)
            .map_err(|e| format!("issue ({}): {e}", labels.join(", ")))?;
    }
    Ok(())
}
