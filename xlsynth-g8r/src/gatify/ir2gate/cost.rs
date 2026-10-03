// SPDX-License-Identifier: Apache-2.0

//! Bounded local cost evaluation using the ordinary g8r node lowering.

use super::{GateEnv, GateOrVec, GatifyOptions, gatify_node};
use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand};
use crate::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort_with_budget,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::dce::get_dead_nodes;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_cost::{IrCost, IrCostEvaluator};
use xlsynth_pir::ir_utils::{get_topological, is_observable_effect_root, operands};

const DEFAULT_WORK_BUDGET: usize = 16_384;
const MAX_TYPE_DEPTH: usize = 64;

/// Estimates bitwise regions using g8r's existing lowering and gate builder.
///
/// `area` is the number of reachable AIG AND nodes and `delay` is their
/// Graph LE delay. Unsupported operations form boundaries: their operands
/// remain roots and their results become fresh inputs. Their own costs are
/// omitted.
///
/// The graph is pruned to its retained roots before measuring fanout. No
/// preparation, libxls optimization, range information, or gate rewriting is
/// used. The shared shift lowering may still choose its stage order using
/// the available AIG control depths.
#[derive(Clone, Debug)]
pub struct GateBuilderCostEvaluator {
    options: GateBuilderOptions,
    graph_le_options: GraphLogicalEffortOptions,
    work_budget: usize,
}

impl Default for GateBuilderCostEvaluator {
    fn default() -> Self {
        let defaults = CanonicalG8rOptions::default();
        Self::new(
            GateBuilderOptions {
                fold: defaults.fold,
                hash: defaults.hash,
            },
            GraphLogicalEffortOptions {
                beta1: defaults.graph_logical_effort_beta1,
                beta2: defaults.graph_logical_effort_beta2,
            },
            DEFAULT_WORK_BUDGET,
        )
    }
}

impl GateBuilderCostEvaluator {
    /// Uses the caller's gate folding and sharing settings with this model.
    pub fn with_gate_builder_options(mut self, options: GateBuilderOptions) -> Self {
        self.options = options;
        self
    }

    /// Configures folding, sharing, Graph LE, and work limits. Construction
    /// preflight accounts for graph/type traversal, copied bits, prospective
    /// gates, and the shared shifter's trial builds and cone scans. Graph LE
    /// analysis gets a separate budget of the same size; either limit returns
    /// `Ok(None)` on exhaustion.
    /// Estimation requires finite, nonnegative Graph LE coefficients with at
    /// least one positive coefficient.
    pub fn new(
        options: GateBuilderOptions,
        graph_le_options: GraphLogicalEffortOptions,
        work_budget: usize,
    ) -> Self {
        Self {
            options,
            graph_le_options,
            work_budget,
        }
    }

    /// Lowers the preflighted regions and measures only their retained roots.
    fn estimate_impl(&self, f: &ir::Fn) -> Result<IrCost, EstimateError> {
        let GraphLogicalEffortOptions { beta1, beta2 } = self.graph_le_options;
        if !beta1.is_finite()
            || !beta2.is_finite()
            || beta1 < 0.0
            || beta2 < 0.0
            || (beta1 == 0.0 && beta2 == 0.0)
        {
            return Err(EstimateError::Lowering(
                "local cost: Graph LE coefficients must be finite and nonnegative, with at least one positive".to_string(),
            ));
        }
        let plan = preflight(f, self.work_budget)?;
        let mut builder = GateBuilder::new("ir_cost".to_string(), self.options);
        let mut env = GateEnv::new(f);
        let options = GatifyOptions {
            fold: self.options.fold,
            hash: self.options.hash,
            ..GatifyOptions::all_opts_disabled()
        };
        let mut roots = Vec::new();
        for nr in plan.order {
            if !plan.live[nr.index] {
                continue;
            }
            let node = f.get_node(nr);
            if supported(&node.payload) {
                lower_supported(f, nr, &plan.widths, &mut builder, &mut env, &options)?;
            } else {
                // Preserve logic consumed by an opaque operation, including
                // shared controls that no longer feed a supported output.
                for arg in operands(&node.payload) {
                    append_root(&env, arg, &mut roots)?;
                }
                let bits =
                    builder.add_input(format!("boundary_{}", nr.index), plan.widths[nr.index]);
                env.add(nr, GateOrVec::BitVector(bits));
            }
            if is_observable_effect_root(&node.payload) {
                append_root(&env, nr, &mut roots)?;
            }
        }
        append_root(&env, plan.ret, &mut roots)?;
        builder.add_output(
            "roots".to_string(),
            AigBitVector::from_lsb_is_index_0(&roots),
        );
        // Graph LE visits every stored gate; discarded bits must not inflate
        // fanout or contribute paths outside the retained regions.
        let gate_fn = dce(&builder.build());
        let area = gate_fn
            .gates
            .iter()
            .filter(|node| matches!(node, AigNode::And2 { .. }))
            .count();
        let delay = if area == 0 {
            // The shared analyzer uses -1 when there are no logic gates.
            0.0
        } else {
            analyze_graph_logical_effort_with_budget(
                &gate_fn,
                &self.graph_le_options,
                self.work_budget,
            )
            .ok_or(EstimateError::Unavailable)?
            .delay
        };
        if !delay.is_finite() || delay < 0.0 {
            return Err(EstimateError::Lowering(
                "local cost: Graph LE did not produce a finite nonnegative delay".to_string(),
            ));
        }
        Ok(IrCost { area, delay })
    }
}

impl IrCostEvaluator for GateBuilderCostEvaluator {
    fn estimate(&mut self, f: &ir::Fn) -> Result<Option<IrCost>, String> {
        match self.estimate_impl(f) {
            Ok(cost) => Ok(Some(cost)),
            Err(EstimateError::Unavailable) => Ok(None),
            Err(EstimateError::Lowering(error)) => Err(error),
        }
    }
}

enum EstimateError {
    Unavailable,
    Lowering(String),
}

impl From<String> for EstimateError {
    fn from(value: String) -> Self {
        Self::Lowering(value)
    }
}

struct Budget {
    remaining: usize,
}

impl Budget {
    fn spend(&mut self, work: usize) -> Result<(), EstimateError> {
        self.remaining = self
            .remaining
            .checked_sub(work)
            .ok_or(EstimateError::Unavailable)?;
        Ok(())
    }
}

struct Plan {
    order: Vec<NodeRef>,
    live: Vec<bool>,
    widths: Vec<usize>,
    ret: NodeRef,
}

fn add(a: usize, b: usize) -> Result<usize, EstimateError> {
    a.checked_add(b).ok_or(EstimateError::Unavailable)
}

fn mul(a: usize, b: usize) -> Result<usize, EstimateError> {
    a.checked_mul(b).ok_or(EstimateError::Unavailable)
}

/// Counts operand slots without first cloning potentially large operand lists.
fn operand_count(payload: &NodePayload) -> Result<usize, EstimateError> {
    Ok(match payload {
        NodePayload::Tuple(args)
        | NodePayload::Array(args)
        | NodePayload::ArrayConcat(args)
        | NodePayload::AfterAll(args)
        | NodePayload::Nary(_, args)
        | NodePayload::Invoke { operands: args, .. } => args.len(),
        NodePayload::ArrayUpdate { indices, .. } => add(2, indices.len())?,
        NodePayload::ArrayIndex { indices, .. } => add(1, indices.len())?,
        NodePayload::ExtNaryAdd { terms, .. } => terms.len(),
        NodePayload::Trace { operands, .. } => add(2, operands.len())?,
        NodePayload::PrioritySel { cases, default, .. }
        | NodePayload::Sel { cases, default, .. } => {
            add(add(1, cases.len())?, usize::from(default.is_some()))?
        }
        NodePayload::OneHotSel { cases, .. } => add(1, cases.len())?,
        NodePayload::CountedFor { invariant_args, .. } => add(1, invariant_args.len())?,
        // All remaining payloads have at most three operands, so this small
        // allocation is bounded independently of the input's size.
        _ => operands(payload).len(),
    })
}

/// Bounds type recursion and expanded aggregate traversal, including zero-bit
/// arrays and repeated bit copies at every level of literal flattening.
fn type_work(ty: &ir::Type, budget: &mut Budget, depth: usize) -> Result<usize, EstimateError> {
    budget.spend(1)?;
    if depth > MAX_TYPE_DEPTH {
        return Err(EstimateError::Unavailable);
    }
    let work = match ty {
        ir::Type::Bits(_) | ir::Type::Token => 1,
        ir::Type::Tuple(types) => {
            let mut work = 1;
            for ty in types {
                work = add(work, type_work(ty, budget, depth + 1)?)?;
            }
            work
        }
        ir::Type::Array(array) => add(
            1,
            mul(
                array.element_count,
                type_work(&array.element_type, budget, depth + 1)?,
            )?,
        )?,
    };
    let work = add(
        work,
        ty.checked_bit_count().ok_or(EstimateError::Unavailable)?,
    )?;
    if work > budget.remaining {
        return Err(EstimateError::Unavailable);
    }
    Ok(work)
}

fn supported(payload: &NodePayload) -> bool {
    matches!(
        payload,
        NodePayload::Literal(_)
            | NodePayload::Unop(
                Unop::Identity
                    | Unop::Not
                    | Unop::Reverse
                    | Unop::AndReduce
                    | Unop::OrReduce
                    | Unop::XorReduce,
                _
            )
            | NodePayload::Nary(..)
            | NodePayload::Tuple(_)
            | NodePayload::Array(_)
            | NodePayload::ArrayConcat(_)
            | NodePayload::TupleIndex { .. }
            | NodePayload::BitSlice { .. }
            | NodePayload::ZeroExt { .. }
            | NodePayload::SignExt { .. }
            | NodePayload::Binop(Binop::Shll | Binop::Shrl, ..)
            | NodePayload::Sel { .. }
            | NodePayload::PrioritySel { .. }
    )
}

/// Bounds the graph and all lowering work before constructing a GateBuilder.
fn preflight(f: &ir::Fn, limit: usize) -> Result<Plan, EstimateError> {
    let mut budget = Budget { remaining: limit };
    budget.spend(f.nodes.len())?;
    let ret = f.ret_node_ref.expect("valid function has a return node");
    for node in &f.nodes {
        budget.spend(operand_count(&node.payload)?)?;
    }
    // The graph helpers allocate operand lists, now bounded by the budget.
    let mut live = vec![true; f.nodes.len()];
    for nr in get_dead_nodes(f) {
        live[nr.index] = false;
    }
    let order = get_topological(f);
    let mut widths = vec![0; f.nodes.len()];
    let mut type_work_by_node = vec![0; f.nodes.len()];
    for nr in &order {
        if !live[nr.index] {
            continue;
        }
        let node = f.get_node(*nr);
        let value_work = type_work(&node.ty, &mut budget, 0)?;
        budget.spend(value_work)?;
        type_work_by_node[nr.index] = value_work;
        let width = node
            .ty
            .checked_bit_count()
            .ok_or(EstimateError::Unavailable)?;
        budget.spend(width)?;
        widths[nr.index] = width;
    }
    let mut prefix_size = 1;
    let mut root_count = widths[ret.index];
    for nr in &order {
        if !live[nr.index] {
            continue;
        }
        let node = f.get_node(*nr);
        if let NodePayload::TupleIndex { tuple, .. } = &node.payload {
            // Computing an offset walks sibling types even when they contain
            // no bits. Charge that traversal separately for each projection.
            budget.spend(type_work_by_node[tuple.index])?;
        }
        for arg in operands(&node.payload) {
            budget.spend(widths[arg.index])?;
            if !supported(&node.payload) {
                root_count = add(root_count, widths[arg.index])?;
            }
        }
        if is_observable_effect_root(&node.payload) {
            root_count = add(root_count, widths[nr.index])?;
        }
        let gate_bound = lowering_work(node, &widths, prefix_size, &mut budget)?;
        // Tags can accumulate on earlier gates after folding; include their
        // count when bounding checkpoint copies during later shift trials.
        prefix_size = add(prefix_size, add(gate_bound, widths[nr.index])?)?;
    }
    budget.spend(add(prefix_size, root_count)?)?;
    Ok(Plan {
        order,
        live,
        widths,
        ret,
    })
}

/// Charges worst-case primitive gate work, including discarded shift trials.
fn lowering_work(
    node: &ir::Node,
    widths: &[usize],
    prefix_size: usize,
    budget: &mut Budget,
) -> Result<usize, EstimateError> {
    let width = node
        .ty
        .checked_bit_count()
        .ok_or(EstimateError::Unavailable)?;
    let gate_bound = match &node.payload {
        NodePayload::Nary(NaryOp::Concat, _) => 0,
        NodePayload::Nary(op, args) => mul(
            mul(width, args.len().saturating_sub(1))?,
            if *op == NaryOp::Xor { 3 } else { 1 },
        )?,
        NodePayload::Unop(op @ (Unop::AndReduce | Unop::OrReduce | Unop::XorReduce), arg) => mul(
            widths[arg.index].saturating_sub(1),
            if *op == Unop::XorReduce { 3 } else { 1 },
        )?,
        NodePayload::Sel {
            selector,
            cases,
            default,
        } if width != 0 && !cases.is_empty() => {
            if widths[selector.index] == 0 {
                0
            } else if cases.len() == 2 && default.is_none() {
                mul(3, width)?
            } else {
                // Decoding, optional OOB reduction, per-case masks, and ORs.
                add(
                    mul(cases.len(), widths[selector.index])?,
                    mul(
                        mul(2, width)?,
                        add(cases.len(), usize::from(default.is_some()))?,
                    )?,
                )?
            }
        }
        NodePayload::PrioritySel { cases, .. } if width != 0 => {
            mul(add(mul(3, width)?, 2)?, add(cases.len(), 1)?)?
        }
        NodePayload::Binop(Binop::Shll | Binop::Shrl, _, amount) if width != 0 => {
            let natural_bits = (usize::BITS - (width - 1).leading_zeros()) as usize;
            let amount_width = widths[amount.index];
            let stages = natural_bits.min(amount_width);
            let oversize = usize::from(amount_width > natural_bits);
            let build = mul(mul(3, width)?, add(stages, oversize)?)?;
            let overflow_reduction = amount_width.saturating_sub(natural_bits).saturating_sub(1);
            if stages > 1 {
                // At most three builds and three cone scans. Each scan can
                // reach earlier nodes and checkpoint copies can retain tags.
                budget.spend(add(mul(3, prefix_size)?, mul(4, build)?)?)?;
            } else if stages == 1 {
                // Even one control may require traversing its complete cone.
                budget.spend(prefix_size)?;
            }
            add(build, overflow_reduction)?
        }
        payload if supported(payload) => 0,
        _ => width, // Opaque outputs are independent primary inputs.
    };
    budget.spend(gate_bound)?;
    Ok(gate_bound)
}

/// Handles empty values that the general lowering expects preparation to
/// remove, then delegates all normal Boolean, mux, and shift structures.
fn lower_supported(
    f: &ir::Fn,
    nr: NodeRef,
    widths: &[usize],
    builder: &mut GateBuilder,
    env: &mut GateEnv,
    options: &GatifyOptions,
) -> Result<(), EstimateError> {
    let node = f.get_node(nr);
    let width = widths[nr.index];
    let direct = if width == 0 {
        Some(AigBitVector::zeros(0))
    } else {
        match &node.payload {
            NodePayload::Sel { cases, default, .. } if cases.is_empty() => {
                Some(env.get_bit_vector(default.expect("valid default-only select has a default"))?)
            }
            NodePayload::Sel {
                selector, cases, ..
            } if widths[selector.index] == 0 => Some(env.get_bit_vector(cases[0])?),
            NodePayload::Unop(op @ (Unop::AndReduce | Unop::OrReduce | Unop::XorReduce), arg)
                if widths[arg.index] == 0 =>
            {
                Some(AigBitVector::from_bit(if *op == Unop::AndReduce {
                    builder.get_true()
                } else {
                    builder.get_false()
                }))
            }
            _ => None, // Ordinary supported nodes use the shared lowering.
        }
    };
    if let Some(bits) = direct {
        env.add(nr, GateOrVec::BitVector(bits));
    } else {
        gatify_node(f, nr, node, builder, env, options)?;
    }
    Ok(())
}

fn append_root(
    env: &GateEnv,
    nr: NodeRef,
    roots: &mut Vec<AigOperand>,
) -> Result<(), EstimateError> {
    roots.extend(env.get_bit_vector(nr)?.iter_lsb_to_msb().copied());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    fn parse(text: &str) -> ir::Fn {
        Parser::new(text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    fn estimate(f: &ir::Fn) -> IrCost {
        GateBuilderCostEvaluator::default()
            .estimate(f)
            .unwrap()
            .unwrap()
    }

    fn assert_cost(cost: IrCost, ands: usize, delay: f64) {
        assert_eq!(cost.area, ands);
        assert!(
            (cost.delay - delay).abs() < 1.0e-9,
            "expected Graph LE {delay}, got {}",
            cost.delay
        );
    }

    #[test]
    fn sharing_and_projection_count_only_reachable_ands() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, y: bits[4] id=2) -> (bits[2], bits[2]) {
  a: bits[4] = and(x, y, id=3)
  b: bits[4] = and(y, x, id=4)
  lo_a: bits[2] = bit_slice(a, start=0, width=2, id=5)
  lo_b: bits[2] = bit_slice(b, start=0, width=2, id=6)
  ret out: (bits[2], bits[2]) = tuple(lo_a, lo_b, id=7)
}"#,
        );
        assert_cost(estimate(&f), 2, 10.0 / 3.0);
        let mut unshared = GateBuilderCostEvaluator {
            options: GateBuilderOptions {
                fold: true,
                hash: false,
            },
            ..Default::default()
        };
        // Without sharing, each input bit drives two gates instead of one.
        assert_cost(unshared.estimate(&f).unwrap().unwrap(), 4, 14.0 / 3.0);
    }

    #[test]
    fn folding_is_configurable_and_delay_accounts_for_fanout() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> bits[4] {
  ret out: bits[4] = sel(p, cases=[x, x], id=3)
}"#,
        );
        assert_cost(estimate(&f), 0, 0.0);
        let mut raw = GateBuilderCostEvaluator {
            options: GateBuilderOptions::no_opt(),
            ..Default::default()
        };
        // Eight selector loads followed by one load, over two NAND stages.
        assert_cost(
            raw.estimate(&f).unwrap().unwrap(),
            12,
            4.0 + (8.0 / 3.0) * 8.0_f64.sqrt(),
        );
    }

    #[test]
    fn opaque_consumer_preserves_its_inputs_and_resets_depth() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, y: bits[4] id=2, p: bits[1] id=3) -> bits[4] {
  selected: bits[4] = sel(p, cases=[x, y], id=4)
  sum: bits[4] = add(selected, x, id=5)
  ret out: bits[4] = and(sum, y, id=6)
}"#,
        );
        assert_cost(estimate(&f), 16, 4.0 + (8.0 / 3.0) * 8.0_f64.sqrt());
    }

    #[test]
    fn equal_opaque_operations_produce_independent_inputs() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, y: bits[4] id=2) -> bits[4] {
  a: bits[4] = add(x, y, id=3)
  b: bits[4] = add(x, y, id=4)
  ret out: bits[4] = xor(a, b, id=5)
}"#,
        );
        assert_cost(estimate(&f), 12, 4.0 + (8.0 / 3.0) * 2.0_f64.sqrt());
    }

    #[test]
    fn selection_does_not_look_through_an_opaque_increment() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> bits[4] {
  one: bits[4] = literal(value=1, id=3)
  next: bits[4] = add(x, one, id=4)
  ret out: bits[4] = sel(p, cases=[x, next], id=5)
}"#,
        );
        assert_cost(estimate(&f), 12, 4.0 + (8.0 / 3.0) * 8.0_f64.sqrt());
    }

    #[test]
    fn wide_shift_amounts_preserve_the_overshift_rule() {
        for op in ["shll", "shrl"] {
            let f = parse(&format!(
                r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> bits[4] {{
  zero: bits[65] = literal(value=0, id=3)
  huge: bits[65] = literal(value=18446744073709551616, id=4)
  amount: bits[65] = sel(p, cases=[zero, huge], id=5)
  ret out: bits[4] = {op}(x, amount, id=6)
}}"#,
            ));
            assert_cost(estimate(&f), 4, 22.0 / 3.0);
        }
    }

    #[test]
    fn default_only_and_zero_selector_selections_preserve_their_data() {
        let f = parse(
            r#"package test
top fn main(x: bits[2] id=1, y: bits[2] id=2, s: bits[3] id=3, empty: bits[0] id=4) -> (bits[2], bits[2]) {
  masked: bits[2] = and(x, y, id=5)
  default_only: bits[2] = sel(s, cases=[], default=masked, id=6)
  singleton: bits[2] = sel(empty, cases=[masked], id=7)
  ret out: (bits[2], bits[2]) = tuple(default_only, singleton, id=8)
}"#,
        );
        assert_cost(estimate(&f), 2, 10.0 / 3.0);
        let mut raw = GateBuilderCostEvaluator {
            options: GateBuilderOptions::no_opt(),
            ..Default::default()
        };
        assert_cost(raw.estimate(&f).unwrap().unwrap(), 2, 10.0 / 3.0);
    }

    #[test]
    fn zero_width_data_and_empty_reductions_need_no_gates() {
        let f = parse(
            r#"package test
top fn main(x: bits[0] id=1, amount: bits[65] id=2, p: bits[1] id=3) -> (bits[0], bits[1], bits[1], bits[1]) {
  shifted: bits[0] = shll(x, amount, id=4)
  selected: bits[0] = sel(p, cases=[x, shifted], id=5)
  all: bits[1] = and_reduce(x, id=6)
  any: bits[1] = or_reduce(x, id=7)
  parity: bits[1] = xor_reduce(x, id=8)
  ret out: (bits[0], bits[1], bits[1], bits[1]) = tuple(selected, all, any, parity, id=9)
}"#,
        );
        for options in [GateBuilderOptions::opt(), GateBuilderOptions::no_opt()] {
            let mut evaluator = GateBuilderCostEvaluator {
                options,
                ..Default::default()
            };
            assert_cost(evaluator.estimate(&f).unwrap().unwrap(), 0, 0.0);
        }
    }

    #[test]
    fn repeated_wide_cases_are_rejected_before_gate_allocation() {
        let cases = vec!["x"; 1_024].join(", ");
        let f = parse(&format!(
            r#"package test
top fn main(x: bits[1024] id=1, s: bits[10] id=2) -> bits[1024] {{
  ret out: bits[1024] = sel(s, cases=[{cases}], id=3)
}}"#,
        ));
        assert!(matches!(
            preflight(&f, DEFAULT_WORK_BUDGET),
            Err(EstimateError::Unavailable)
        ));
        assert_eq!(
            GateBuilderCostEvaluator::default().estimate(&f).unwrap(),
            None
        );
    }

    #[test]
    fn huge_type_only_widths_and_zero_bit_arrays_are_bounded() {
        let original = parse(
            r#"package test
top fn main(x: bits[1] id=1) -> bits[1] {
  ret out: bits[1] = identity(x, id=2)
}"#,
        );
        for ty in [
            ir::Type::Bits(usize::MAX),
            ir::Type::new_array(ir::Type::Bits(usize::MAX), 2),
            ir::Type::new_array(ir::Type::Bits(0), usize::MAX),
            (0..32).fold(ir::Type::Bits(1024), |member, _| {
                ir::Type::Tuple(vec![Box::new(member)])
            }),
        ] {
            let mut f = original.clone();
            let input = f.params[0];
            let ret = f.ret_node_ref.unwrap();
            f.get_node_mut(input).ty = ty.clone();
            f.get_node_mut(ret).ty = ty.clone();
            f.ret_ty = ty;
            assert_eq!(
                GateBuilderCostEvaluator::default().estimate(&f).unwrap(),
                None
            );
        }
    }

    #[test]
    fn repeated_tuple_projections_charge_zero_width_siblings() {
        let mut members = vec!["bits[0]"; 4_097];
        members[0] = "bits[1]";
        let projections = (0..500)
            .map(|i| format!("  p{i}: bits[1] = tuple_index(x, index=0, id={})", i + 2))
            .collect::<Vec<_>>()
            .join("\n");
        let names = (0..500)
            .map(|i| format!("p{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let f = parse(&format!(
            r#"package test
top fn main(x: ({}) id=1) -> bits[500] {{
{projections}
  ret out: bits[500] = concat({names}, id=502)
}}"#,
            members.join(", ")
        ));
        assert!(matches!(
            preflight(&f, DEFAULT_WORK_BUDGET),
            Err(EstimateError::Unavailable)
        ));
        assert_eq!(
            GateBuilderCostEvaluator::default().estimate(&f).unwrap(),
            None
        );
    }

    #[test]
    fn evaluator_budget_is_per_call() {
        let f = parse(
            r#"package test
top fn main(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {
  ret out: bits[1] = and(x, y, id=3)
}"#,
        );
        let mut evaluator = GateBuilderCostEvaluator::default();
        assert_cost(evaluator.estimate(&f).unwrap().unwrap(), 1, 10.0 / 3.0);
        assert_cost(evaluator.estimate(&f).unwrap().unwrap(), 1, 10.0 / 3.0);
        evaluator.work_budget = 0;
        assert_eq!(evaluator.estimate(&f).unwrap(), None);
    }

    #[test]
    fn discarded_bits_do_not_inflate_fanout() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> bits[1] {
  mask: bits[4] = sign_ext(p, new_bit_count=4, id=3)
  masked: bits[4] = and(x, mask, id=4)
  ret out: bits[1] = bit_slice(masked, start=0, width=1, id=5)
}"#,
        );
        // Only one of the four constructed loads of p reaches the output.
        assert_cost(estimate(&f), 1, 10.0 / 3.0);
    }

    #[test]
    fn graph_le_uses_configured_load_coefficients() {
        let f = parse(
            r#"package test
top fn main(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {
  ret out: bits[1] = and(x, y, id=3)
}"#,
        );
        let mut evaluator = GateBuilderCostEvaluator::new(
            GateBuilderOptions::opt(),
            GraphLogicalEffortOptions {
                beta1: 2.0,
                beta2: 0.5,
            },
            DEFAULT_WORK_BUDGET,
        );
        assert_cost(evaluator.estimate(&f).unwrap().unwrap(), 1, 16.0 / 3.0);
    }

    #[test]
    fn invalid_load_coefficients_cannot_hide_a_path() {
        let f = parse(
            r#"package test
top fn main(x: bits[1] id=1, y: bits[1] id=2, z: bits[1] id=3) -> bits[2] {
  a: bits[1] = and(x, y, id=4)
  b: bits[1] = and(x, z, id=5)
  ret out: bits[2] = concat(a, b, id=6)
}"#,
        );
        // A negative quadratic coefficient can give x a negative load while
        // the other inputs still produce finite delays, silently hiding x.
        for (beta1, beta2) in [
            (1.0, -0.75),
            (-0.75, 1.0),
            (f64::NAN, 0.0),
            (1.0, f64::INFINITY),
            (0.0, 0.0),
        ] {
            let mut evaluator = GateBuilderCostEvaluator::new(
                GateBuilderOptions::opt(),
                GraphLogicalEffortOptions { beta1, beta2 },
                DEFAULT_WORK_BUDGET,
            );
            assert!(evaluator.estimate(&f).is_err());
        }
    }

    #[test]
    fn graph_le_budget_can_decline_a_small_reconvergent_graph() {
        let mut text = String::from(
            r#"package test
top fn main(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {
"#,
        );
        let (mut left, mut right) = ("x".to_string(), "y".to_string());
        for i in 0..999 {
            let ret = if i == 998 { "ret " } else { "" };
            text.push_str(&format!(
                "  {ret}g{i}: bits[1] = and({left}, {right}, id={})\n",
                i + 3
            ));
            left = right;
            right = format!("g{i}");
        }
        text.push('}');
        let f = parse(&text);
        assert!(preflight(&f, DEFAULT_WORK_BUDGET).is_ok());
        let mut evaluator = GateBuilderCostEvaluator::new(
            GateBuilderOptions::no_opt(),
            GraphLogicalEffortOptions {
                beta1: 2.0,
                beta2: 0.0,
            },
            DEFAULT_WORK_BUDGET,
        );
        assert_eq!(evaluator.estimate(&f).unwrap(), None);
    }
}
