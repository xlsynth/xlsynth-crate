// SPDX-License-Identifier: Apache-2.0

//! Word-level symbolic execution of a clocked QuickCheck's fixed schedule.

use std::collections::HashMap;

use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use xlsynth_pir::block_transition::BlockTransition;
use xlsynth_pir::ir::{Fn, InstantiationKind, NodePayload, Package, PackageMember, Type};
use xlsynth_pir::ir_builder::{BValue, BuilderError, FnBuilder, InvokeSignature};
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::{IrBits, IrValue};

use super::quickcheck::Stimulus;
use super::{TestAction, TestIterable, TestProgram, TestStep, input_name};
use crate::ast::SourcePosition;

// Bound hierarchy flattening, then charge every unrolled step and iteration
// plus the interface and transitively expanded body of each transition call.
const MAX_PROOF_WORK: usize = 1 << 20;

/// Check the nodes and registers that the selected hierarchy would copy before
/// the transition builder flattens any block instances.
pub(super) fn check_block_expansion(package: &Package, entry: &str) -> Result<(), String> {
    let blocks: HashMap<_, _> = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Block(block) => Some((block.name.as_str(), block)),
            PackageMember::Function(_) => None,
        })
        .collect();
    let mut costs = ExpansionCosts::default();
    let mut pending = vec![entry];
    while let Some(name) = pending.pop() {
        if costs.local.contains_key(name) {
            continue;
        }
        let block = blocks
            .get(name)
            .ok_or_else(|| format!("missing block '{name}'"))?;
        costs.local.insert(
            name,
            block.nodes.len().saturating_add(block.registers.len()),
        );
        for instance in &block.instantiations {
            if instance.kind != InstantiationKind::Block {
                return Err(format!(
                    "cannot inline external instance '{}'",
                    instance.name
                ));
            }
            let child = instance.block.as_str();
            costs.add_use(child, name, 1);
            pending.push(child);
        }
    }
    if costs.total(entry)? > MAX_PROOF_WORK {
        return Err(format!(
            "block hierarchy exceeds the symbolic unrolling limit of {MAX_PROOF_WORK} work units before flattening"
        ));
    }
    Ok(())
}

/// Build a property over the stimulus parameters followed by the initial
/// registers.
pub(super) fn lower_property(
    program: &TestProgram,
    transition: &BlockTransition,
    domains: &[Stimulus],
    transition_package: &Package,
) -> Result<Fn, String> {
    let mut work_left = MAX_PROOF_WORK;
    check_schedule(&program.steps, 1, false, &mut work_left).map_err(|e| e.to_string())?;
    let mut name = format!("bx_prove_{}", program.harness);
    while transition_package
        .members
        .iter()
        .any(|m| m.graph().name == name)
    {
        name.push('_');
    }
    let mut builder = FnBuilder::new(&name);
    let params = program
        .quickcheck_params
        .as_deref()
        .ok_or("expected a block QuickCheck")?;
    let input_indices: HashMap<_, _> = transition
        .inputs
        .iter()
        .enumerate()
        .map(|(i, port)| (port.name.as_str(), i))
        .collect();
    let reset = transition
        .reset
        .as_ref()
        .ok_or("missing test harness reset")?;
    let reset_index = *input_indices
        .get(reset.name.as_str())
        .ok_or("missing reset input")?;
    // Test-port numbers (including snapshot and loop bindings) are contiguous.
    // Their declaration order in a flattened hardware interface is independent.
    let port_indices = (0..transition.inputs.len() - 1)
        .map(|i| {
            let name = input_name(i);
            input_indices
                .get(name.as_str())
                .copied()
                .ok_or_else(|| format!("missing harness port '{name}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let formal_values = params
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let port = port_indices.get(i).ok_or_else(|| {
                BuilderError::InvalidOperation("missing QuickCheck parameter port".into())
            })?;
            builder.param(
                &format!("bx_qc_arg_{i}"),
                transition.inputs[*port].ty.clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let registers = transition
        .registers
        .iter()
        .enumerate()
        .map(|(i, reg)| builder.param(&format!("bx_qc_state_{i}"), reg.ty.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let legal = super::proof_domain::legal_stimulus(&mut builder, params, &formal_values, domains)?;
    let inactive_reset = builder
        .literal(IrValue::bool(reset.active_low))
        .map_err(|e| e.to_string())?;
    let mut drives = vec![None; transition.inputs.len()];
    drives[reset_index] = Some(inactive_reset);
    for (index, value) in port_indices.iter().zip(formal_values) {
        drives[*index] = Some(value);
    }
    let mut frame = drives
        .into_iter()
        .zip(&transition.inputs)
        .map(|(value, input)| match value {
            Some(value) => Ok(value),
            None => builder.literal(zero_ir_value_for_type(&input.ty)),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    frame.extend(registers);
    let transition_work = expanded_node_visits(transition_package, &transition.function.name)?
        .saturating_add(frame.len())
        .saturating_add(transition.outputs.len())
        .saturating_add(transition.registers.len());
    let mut execution = SymbolicExecution {
        builder,
        frame,
        port_indices,
        inactive_reset,
        reset_index,
        active_low: reset.active_low,
        input_count: transition.inputs.len(),
        output_count: transition.outputs.len(),
        signature: InvokeSignature::from_function(&transition.function)
            .map_err(|e| e.to_string())?,
        settled: None,
        predicates: Vec::new(),
        transition_work,
        work_left,
    };
    execution
        .execute(&program.steps, None)
        .map_err(|e| e.to_string())?;
    execution
        .finish(legal, transition_package)
        .map_err(|e| e.to_string())
}

/// Count solver visits to each function once, including repeated visits through
/// invokes, maps, and counted loops. The package contains the transition and
/// its reachable callees, in arbitrary declaration order.
fn expanded_node_visits(package: &Package, entry: &str) -> Result<usize, String> {
    let mut costs = ExpansionCosts::default();
    for member in &package.members {
        let PackageMember::Function(f) = member else {
            return Err("expected a function-only transition package".into());
        };
        let caller = f.name.as_str();
        costs.local.insert(caller, f.nodes.len());
        for node in &f.nodes {
            let (callee, count) = match &node.payload {
                NodePayload::Invoke { to_apply, .. } => (to_apply, 1),
                NodePayload::CountedFor {
                    body, trip_count, ..
                } => (body, *trip_count),
                NodePayload::Map { array, to_apply } => {
                    let Type::Array(ty) = f.get_node_ty(*array) else {
                        return Err(format!("expected array operand for map in '{caller}'"));
                    };
                    (to_apply, ty.element_count)
                }
                _ => continue,
            };
            costs.add_use(callee, caller, count);
        }
    }
    costs.total(entry)
}

/// Expansion work per definition, with edge weights counting how often a
/// parent instantiates a child block or invokes a child function.
#[derive(Default)]
struct ExpansionCosts<'a> {
    parents: DiGraphMap<&'a str, usize>,
    local: HashMap<&'a str, usize>,
}

impl<'a> ExpansionCosts<'a> {
    fn add_use(&mut self, child: &'a str, parent: &'a str, count: usize) {
        if let Some(total) = self.parents.edge_weight_mut(child, parent) {
            *total = total.saturating_add(count);
        } else {
            self.parents.add_edge(child, parent, count);
        }
    }

    /// Fold children into parents once in dependency order, even when the
    /// expanded hierarchy contains exponentially many copies.
    fn total(mut self, entry: &str) -> Result<usize, String> {
        let order = toposort(&self.parents, None)
            .map_err(|e| format!("recursive proof dependency '{}'", e.node_id()))?;
        for child in order {
            let cost = *self
                .local
                .get(child)
                .ok_or_else(|| format!("missing proof dependency '{child}'"))?;
            for (_, parent, count) in self.parents.edges(child) {
                let total = self
                    .local
                    .get_mut(parent)
                    .expect("edge target is a declared parent");
                *total = total.saturating_add(count.saturating_mul(cost));
            }
        }
        self.local
            .get(entry)
            .copied()
            .ok_or_else(|| format!("missing proof entry '{entry}'"))
    }
}

/// Check the fixed schedule and bound nested loop expansion before building IR.
fn check_schedule(
    steps: &[TestStep],
    executions: usize,
    conditional: bool,
    work_left: &mut usize,
) -> Result<(), BuilderError> {
    for step in steps {
        charge(work_left, executions, step.position)?;
        match &step.action {
            TestAction::Tick(_) if conditional => {
                return Err(BuilderError::InvalidOperation(format!(
                    "{}:{}: conditional tick! in a block QuickCheck proof; place tick! outside the if and condition the values driven into the block",
                    step.position.line, step.position.column
                )));
            }
            TestAction::If {
                then_body,
                else_body,
                ..
            } => {
                check_schedule(then_body, executions, true, work_left)?;
                check_schedule(else_body, executions, true, work_left)?;
            }
            TestAction::For { count, body, .. } => {
                let iterations = executions.saturating_mul(*count);
                charge(work_left, iterations, step.position)?;
                check_schedule(body, iterations, conditional, work_left)?;
            }
            _ => { /* Other actions do not change the number of clock edges. */ }
        }
    }
    Ok(())
}

fn charge(
    work_left: &mut usize,
    units: usize,
    position: SourcePosition,
) -> Result<(), BuilderError> {
    *work_left = work_left.checked_sub(units).ok_or_else(|| {
        BuilderError::InvalidOperation(format!(
            "{}:{}: block QuickCheck proof exceeds the symbolic unrolling limit of {MAX_PROOF_WORK} work units",
            position.line, position.column
        ))
    })?;
    Ok(())
}

/// The current test-port drives and registers are SSA values in the property
/// function.
struct SymbolicExecution {
    builder: FnBuilder,
    frame: Vec<BValue>,
    port_indices: Vec<usize>,
    inactive_reset: BValue,
    reset_index: usize,
    active_low: bool,
    input_count: usize,
    output_count: usize,
    signature: InvokeSignature,
    settled: Option<BValue>,
    predicates: Vec<BValue>,
    transition_work: usize,
    work_left: usize,
}

impl SymbolicExecution {
    /// Reuse the same observation for all samples of an unchanged frame.
    fn settle(&mut self, position: SourcePosition) -> Result<BValue, BuilderError> {
        if let Some(value) = self.settled {
            return Ok(value);
        }
        charge(&mut self.work_left, self.transition_work, position)?;
        let value = self
            .builder
            .invoke_with_signature(&self.signature, &self.frame)?;
        self.settled = Some(value);
        Ok(value)
    }

    /// An action can sample individual outputs or one tuple captured on entry.
    fn operand(
        &mut self,
        index: usize,
        group: Option<(usize, BValue)>,
        position: SourcePosition,
    ) -> Result<BValue, BuilderError> {
        if let Some((start, tuple)) = group {
            let field = index.checked_sub(start).ok_or_else(|| {
                BuilderError::InvalidOperation("invalid test operand group".into())
            })?;
            self.builder.tuple_index(tuple, field)
        } else {
            let tuple = self.settle(position)?;
            self.builder.tuple_index(tuple, index)
        }
    }

    fn set(&mut self, index: usize, value: BValue) {
        if self.frame[index] != value {
            self.frame[index] = value;
            self.settled = None;
        }
    }

    /// A procedural drive persists on exactly the paths that execute it.
    fn drive(
        &mut self,
        port: usize,
        value: BValue,
        guard: Option<BValue>,
    ) -> Result<(), BuilderError> {
        let index = self.port_indices[port];
        let old = self.frame[index];
        if value != old {
            let new = match guard {
                Some(guard) => self.builder.select(guard, &[old, value], None)?,
                None => value,
            };
            self.set(index, new);
        }
        Ok(())
    }

    /// Pulse physical reset for this edge and capture every register
    /// simultaneously.
    fn tick(
        &mut self,
        asserted: Option<BValue>,
        position: SourcePosition,
    ) -> Result<(), BuilderError> {
        let reset = match asserted {
            Some(value) if self.active_low => self.builder.not(value)?,
            Some(value) => value,
            None => self.inactive_reset,
        };
        self.set(self.reset_index, reset);
        let sampled = self.settle(position)?;
        for index in self.input_count..self.frame.len() {
            let next = self
                .builder
                .tuple_index(sampled, self.output_count + (index - self.input_count))?;
            self.set(index, next);
        }
        self.set(self.reset_index, self.inactive_reset);
        Ok(())
    }

    fn assert(&mut self, condition: BValue, guard: Option<BValue>) -> Result<(), BuilderError> {
        let holds = match guard {
            Some(guard) => {
                let skipped = self.builder.not(guard)?;
                self.builder.or(skipped, condition)?
            }
            None => condition,
        };
        self.predicates.push(holds);
        Ok(())
    }

    /// Both arms use complementary guards derived before either arm changes the
    /// frame.
    fn execute(&mut self, steps: &[TestStep], guard: Option<BValue>) -> Result<(), BuilderError> {
        for step in steps {
            let pos = step.position;
            let group = step
                .group
                .map(|index| self.operand(index, None, pos).map(|tuple| (index, tuple)))
                .transpose()?;
            match &step.action {
                TestAction::Evaluate(index) => {
                    self.operand(*index, group, pos)?;
                }
                TestAction::Drive { port, value } => {
                    let value = self.operand(*value, group, pos)?;
                    self.drive(*port, value, guard)?;
                }
                TestAction::Tick(reset) => {
                    let reset = reset.map(|i| self.operand(i, group, pos)).transpose()?;
                    self.tick(reset, pos)?;
                }
                TestAction::Assert { predicate, .. } => {
                    let holds = self.operand(*predicate, group, pos)?;
                    self.assert(holds, guard)?;
                }
                TestAction::AssertEq { actual, expected } => {
                    let actual = self.operand(*actual, group, pos)?;
                    let expected = self.operand(*expected, group, pos)?;
                    let equal = self.builder.eq(actual, expected)?;
                    self.assert(equal, guard)?;
                }
                TestAction::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let condition = self.operand(*condition, group, pos)?;
                    let mut then_guard = condition;
                    let mut else_guard = self.builder.not(condition)?;
                    if let Some(parent) = guard {
                        then_guard = self.builder.and(parent, then_guard)?;
                        else_guard = self.builder.and(parent, else_guard)?;
                    }
                    self.execute(then_body, Some(then_guard))?;
                    self.execute(else_body, Some(else_guard))?;
                }
                TestAction::For {
                    port,
                    iterable,
                    count,
                    body,
                } => match iterable {
                    TestIterable::Array(index) => {
                        let array = self.operand(*index, group, pos)?;
                        for ordinal in 0..*count {
                            let i = self.builder.literal(IrValue::u32(ordinal as u32))?;
                            let element = self.builder.array_index(array, i)?;
                            self.drive(*port, element, guard)?;
                            self.execute(body, guard)?;
                        }
                    }
                    TestIterable::Range(start) => {
                        for ordinal in 0..*count {
                            let offset = IrBits::make_ubits(start.get_bit_count(), ordinal as u64)
                                .map_err(|e| BuilderError::InvalidOperation(e.to_string()))?;
                            let element =
                                self.builder.literal(IrValue::Bits(start.add(&offset)))?;
                            self.drive(*port, element, guard)?;
                            self.execute(body, guard)?;
                        }
                    }
                },
            }
        }
        Ok(())
    }

    /// Conjoin the path-guarded assertions with logarithmic expression depth.
    fn finish(mut self, legal: BValue, package: &Package) -> Result<Fn, BuilderError> {
        if self.predicates.is_empty() {
            return Err(BuilderError::InvalidOperation(
                "block QuickCheck is missing its final property".into(),
            ));
        }
        while self.predicates.len() > 1 {
            self.predicates = self
                .predicates
                .chunks(2)
                .map(|pair| match pair {
                    [a, b] => self.builder.and(*a, *b),
                    [a] => Ok(*a),
                    _ => unreachable!("chunks are nonempty"),
                })
                .collect::<Result<_, _>>()?;
        }
        let outside_domain = self.builder.not(legal)?;
        let result = self.builder.or(outside_domain, self.predicates[0])?;
        self.builder.build_in_package(result, package)
    }
}

#[cfg(test)]
mod tests {
    use super::expanded_node_visits;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn counts_shared_callees_once_per_invoke_and_array_element() {
        let mut package = Parser::new(
            r#"package p
fn element(x: bits[8] id=1) -> bits[8] {
  ret inverted: bits[8] = not(x, id=2)
}
fn twice(x: bits[8] id=3) -> bits[8] {
  first: bits[8] = invoke(x, to_apply=element, id=4)
  ret second: bits[8] = invoke(first, to_apply=element, id=5)
}
top fn lanes(xs: bits[8][3] id=6) -> bits[8][3] {
  ret mapped: bits[8][3] = map(xs, to_apply=twice, id=7)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        package.members.reverse();
        let element = package.get_fn("element").unwrap().nodes.len();
        let twice = package.get_fn("twice").unwrap().nodes.len() + 2 * element;
        let lanes = package.get_fn("lanes").unwrap().nodes.len() + 3 * twice;
        assert_eq!(expanded_node_visits(&package, "lanes").unwrap(), lanes);
    }
}
