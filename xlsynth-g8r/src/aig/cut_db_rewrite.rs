// SPDX-License-Identifier: Apache-2.0

//! Cut-db–driven AIG rewrite pass.
//!
//! Current goal: reduce global output depth on critical paths, then recover
//! area elsewhere under the achieved depth bound, using the precomputed 4-input
//! cut database plus a bounded large-cone refactor.
//!
//! The depth phase accepts local depth reductions rooted on the current global
//! critical path. The area phase is MFFC-aware: it only credits nodes that
//! become dead under the chosen cut boundary, and it accepts rewrites only when
//! live AND count decreases without increasing the global output depth.
//!
//! The rewrite pipeline is small-cut delay, large-cone delay, small-cut area,
//! then large-cone area. `small_cut` and `large_cone` own candidate
//! construction; this module owns acceptance policy, exact costing,
//! materialization, and dynamic depth/hash bookkeeping.

use std::collections::{BTreeSet, VecDeque};
use std::time::Instant;

use smallvec::SmallVec;

use crate::aig::dce::dce;
use crate::aig::dynamic_depth::DynamicDepthState;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn, PirNodeIds};
use crate::aig::get_summary_stats::get_aig_stats;
use crate::cut_db::fragment::{GateFnFragment, Lit};
use crate::cut_db::loader::CutDb;
#[cfg(test)]
use crate::cut_db::tt16::TruthTable16;

mod cut_replacement_cost;
mod large_cone;
mod small_cut;

use self::cut_replacement_cost::{gate_count_diff_for_replacement, materialize_replacement};
use self::large_cone::{
    FactoredExpr, FactoredExprNode, SopCoverMemo, SopReplacement, SopVariantKind,
    construct_large_cone_candidate_replacements_for_root, sop_depth_from_inputs,
};
use self::small_cut::{CutEnumerationStats, CutEnumerator, choose_candidate_replacements_for_root};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CutDbRewriteMode {
    /// Run delay-focused phases first, then area phases under the achieved
    /// delay bound.
    Delay,
    /// Skip delay-focused phases and run area phases without increasing delay.
    Balanced,
    /// Skip delay-focused phases and allow area phases to increase delay.
    Area,
}

impl CutDbRewriteMode {
    pub const DEFAULT_CLI_VALUE: &'static str = "delay";
    pub const CLI_VALUES: &'static [&'static str] = &["delay", "balanced", "area"];

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "delay" => Some(Self::Delay),
            "balanced" => Some(Self::Balanced),
            "area" => Some(Self::Area),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Delay => "delay",
            Self::Balanced => "balanced",
            Self::Area => "area",
        }
    }

    fn enables_depth_rewrite(self) -> bool {
        matches!(self, Self::Delay)
    }

    fn allows_area_depth_increase(self) -> bool {
        matches!(self, Self::Area)
    }
}

impl Default for CutDbRewriteMode {
    fn default() -> Self {
        Self::Delay
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RewriteOptions {
    /// Maximum number of cuts retained at each node during enumeration.
    ///
    /// This bounds local search breadth. If set to `0`, no per-node cut limit
    /// is applied, which can be expensive on large graphs.
    pub max_cuts_per_node: usize,
    /// Maximum number of outer rewrite iterations to run.
    ///
    /// An iteration is one global recompute round for each rewrite phase. If
    /// set to `0`, each phase runs until no improving rewrite is found.
    pub max_iterations: usize,
    /// If true, check accepted area rewrite cost deltas against an
    /// independently DCE-cleaned copy of the post-rewrite graph.
    pub verify_area_costing: bool,
    /// If true, check accepted delay rewrite deltas against an independently
    /// DCE-cleaned copy of the post-rewrite graph.
    pub verify_delay_costing: bool,
    /// If true, run the large-cone depth/area refactor phases after the
    /// 4-input cut-db phases.
    pub enable_large_cone_rewrite: bool,
    /// Cut-db QoR policy: delay, balanced, or area.
    pub mode: CutDbRewriteMode,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            // Keep the pass lightweight by default; callers that want a more
            // exhaustive search can increase these.
            max_cuts_per_node: 64,
            // Run to convergence by default; the rewrite criterion is monotone so
            // the pass will reach a fixed point.
            //
            // Callers that need a hard time/effort bound (e.g. fuzz targets) can
            // set `max_iterations` to a small positive value.
            max_iterations: 0,
            verify_area_costing: false,
            verify_delay_costing: false,
            enable_large_cone_rewrite: true,
            mode: CutDbRewriteMode::Delay,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum ReplacementImpl {
    Fragment {
        frag: GateFnFragment,
        input_depths: [u16; 4],
    },
    Sop(SopReplacement),
}

#[derive(Debug, Clone)]
pub(super) struct Replacement {
    pub(super) root: AigRef,
    pub(super) leaf_ops: Vec<AigOperand>,
    pub(super) implementation: ReplacementImpl,
    pub(super) score_depth: usize,
    pub(super) score_ands: usize,
    pub(super) raw_score_ands: usize,
    pub(super) structural_hash_only_area_win: bool,
}

#[derive(Debug, Clone)]
struct AreaCandidate {
    replacement: Replacement,
    mffc_nodes: BTreeSet<AigRef>,
    area_gain: usize,
    before_live_and_count: usize,
    after_live_and_count: usize,
    new_root_depth: usize,
    slack_consumed: usize,
}

#[derive(Debug, Clone)]
struct LargeConeCandidate {
    replacement: Replacement,
    mffc_nodes: BTreeSet<AigRef>,
    new_root_depth: usize,
    raw_and_count: usize,
    variant_order: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct LargeConeCandidateStats {
    roots_total: usize,
    roots_visited: usize,
    skipped_unrequired_depth: usize,
    rejected_no_cone: usize,
    rejected_small_cone: usize,
    rejected_empty_mffc: usize,
    rejected_sop_failed: usize,
    rejected_no_area_gain: usize,
    rejected_depth: usize,
    cones_built: usize,
    cone_leaves_sum: usize,
    cone_leaves_max: usize,
    cone_internal_sum: usize,
    cone_internal_max: usize,
    cone_mffc_sum: usize,
    cone_mffc_max: usize,
    sop_variants_sum: usize,
    sop_variants_max: usize,
    viable_candidates: usize,
    viable_structural_hash_only_area_win: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct AreaCandidateStats {
    roots_total: usize,
    roots_visited: usize,
    skipped_unrequired_depth: usize,
    skipped_identity_cut: usize,
    skipped_large_cut: usize,
    rejected_empty_mffc: usize,
    rejected_no_area_gain: usize,
    rejected_depth: usize,
    viable_candidates: usize,
    viable_structural_hash_only_area_win: usize,
    rejected_equal_area: usize,
    rejected_area_increase: usize,
    mffc_rejected_no_area_gain: usize,
    mffc_rejected_equal_area: usize,
    mffc_rejected_area_increase: usize,
    mffc_rejected_depth: usize,
    mffc_viable_candidates: usize,
    non_mffc_rejected_no_area_gain: usize,
    non_mffc_rejected_equal_area: usize,
    non_mffc_rejected_area_increase: usize,
    non_mffc_rejected_depth: usize,
    non_mffc_viable_candidates: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct AreaRewriteSelectionStats {
    total_candidates: usize,
    selected: usize,
    selected_structural_hash_only_area_win: usize,
    selected_raw_score_ands: usize,
    selected_hash_score_ands: usize,
    selected_sop_flat: usize,
    selected_sop_arrival_balanced: usize,
    selected_sop_factored: usize,
}

fn selected_sop_variant_kind(replacement: &Replacement) -> Option<SopVariantKind> {
    match &replacement.implementation {
        ReplacementImpl::Fragment { .. } => None,
        ReplacementImpl::Sop(sop) => Some(sop.kind),
    }
}

fn bump_selected_sop_variant(
    kind: Option<SopVariantKind>,
    flat: &mut usize,
    arrival_balanced: &mut usize,
    factored: &mut usize,
) {
    match kind {
        Some(SopVariantKind::Flat) => *flat += 1,
        Some(SopVariantKind::ArrivalBalanced) => *arrival_balanced += 1,
        Some(SopVariantKind::Factored) => *factored += 1,
        None => {}
    }
}

fn and_node_count(g: &GateFn) -> usize {
    g.gates
        .iter()
        .filter(|node| matches!(node, AigNode::And2 { .. }))
        .count()
}

fn verify_area_cost_delta(
    context: &str,
    before_g: &GateFn,
    after_g: &GateFn,
    before_live_ands: usize,
    after_live_ands: usize,
) {
    verify_expected_area_cost_delta(
        context,
        before_g,
        after_g,
        before_live_ands as isize - after_live_ands as isize,
    );
}

fn verify_expected_area_cost_delta(
    context: &str,
    before_g: &GateFn,
    after_g: &GateFn,
    expected_delta: isize,
) {
    let cleaned_before = dce(before_g);
    let cleaned_after = dce(after_g);
    let checked_before_live_ands = and_node_count(&cleaned_before);
    let checked_after_live_ands = and_node_count(&cleaned_after);
    let checked_delta = checked_before_live_ands as isize - checked_after_live_ands as isize;

    assert_eq!(
        checked_delta, expected_delta,
        "cut-db area cost verification failed in {context}: expected live AND delta {expected_delta}, DCE-cleaned graph delta {checked_delta}"
    );
}

fn verify_live_area_cost_delta_exact(
    context: &str,
    before_live_ands: usize,
    after_live_ands: usize,
    expected_delta: usize,
) {
    let actual_delta = before_live_ands.saturating_sub(after_live_ands);
    assert_eq!(
        actual_delta, expected_delta,
        "cut-db area cost verification failed in {context}: expected live AND delta {expected_delta}, live graph delta {actual_delta}"
    );
}

fn independent_output_node_depth(g: &GateFn) -> usize {
    get_aig_stats(&dce(g)).max_depth
}

fn verify_delay_not_increased(
    context: &str,
    before_g: &GateFn,
    after_g: &GateFn,
    expected_before_depth: usize,
) {
    let checked_before_depth = independent_output_node_depth(before_g);
    let checked_after_depth = independent_output_node_depth(after_g);
    assert_eq!(
        checked_before_depth, expected_before_depth,
        "cut-db delay verification failed in {context}: expected before depth {expected_before_depth}, DCE-cleaned graph before depth {checked_before_depth}"
    );
    assert!(
        checked_after_depth <= checked_before_depth,
        "cut-db delay verification failed in {context}: area rewrite increased DCE-cleaned output depth from {checked_before_depth} to {checked_after_depth}"
    );
}

fn replacement_depth_from_inputs(
    leaf_ops: &[AigOperand],
    implementation: &ReplacementImpl,
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
) -> usize {
    match implementation {
        ReplacementImpl::Fragment { input_depths, .. } => {
            let mut depth = 0usize;
            for (i, leaf) in leaf_ops.iter().enumerate() {
                let cand_depth = live_forward_depth(depth_state, structural_hash_state, leaf.node)
                    + input_depths[i] as usize;
                depth = depth.max(cand_depth);
            }
            depth
        }
        ReplacementImpl::Sop(sop) => {
            sop_depth_from_inputs(leaf_ops, sop, structural_hash_state, depth_state)
        }
    }
}

fn live_forward_depth(
    depth_state: &DynamicDepthState,
    structural_hash_state: &DynamicStructuralHash,
    node: AigRef,
) -> usize {
    depth_state
        .forward_depth(structural_hash_state, node)
        .expect("live cut-db node should have a forward depth")
}

fn dense_use_counts_from_dynamic_hash(state: &DynamicStructuralHash) -> Vec<usize> {
    (0..state.gate_fn().gates.len())
        .map(|id| state.use_count(AigRef { id }))
        .collect()
}

fn dynamic_false() -> AigOperand {
    AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    }
}

fn dynamic_true() -> AigOperand {
    dynamic_false().negate()
}

fn dynamic_is_known_false(op: AigOperand) -> bool {
    op.node.id == 0 && !op.negated
}

fn dynamic_is_known_true(op: AigOperand) -> bool {
    op.node.id == 0 && op.negated
}

fn dynamic_add_or_binary(
    state: &mut DynamicStructuralHash,
    lhs: AigOperand,
    rhs: AigOperand,
    pir_node_ids: &[u32],
    excluded: &BTreeSet<AigRef>,
) -> Result<AigOperand, String> {
    if dynamic_is_known_true(lhs) || dynamic_is_known_true(rhs) {
        state.add_pir_node_ids(dynamic_false().node, pir_node_ids)?;
        return Ok(dynamic_true());
    }
    if dynamic_is_known_false(lhs) && dynamic_is_known_false(rhs) {
        state.add_pir_node_ids(dynamic_false().node, pir_node_ids)?;
        return Ok(dynamic_false());
    }
    if dynamic_is_known_false(lhs) {
        state.add_pir_node_ids(rhs.node, pir_node_ids)?;
        return Ok(rhs);
    }
    if dynamic_is_known_false(rhs) {
        state.add_pir_node_ids(lhs.node, pir_node_ids)?;
        return Ok(lhs);
    }
    Ok(state
        .add_and_with_pir_node_ids_excluding(lhs.negate(), rhs.negate(), pir_node_ids, excluded)?
        .negate())
}

fn instantiate_fragment_dynamic(
    state: &mut DynamicStructuralHash,
    frag: &GateFnFragment,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
    excluded: &BTreeSet<AigRef>,
) -> Result<AigOperand, String> {
    let mut ops: Vec<AigOperand> = Vec::with_capacity(5 + frag.nodes.len());
    for i in 0..4usize {
        if i < leaf_ops.len() {
            ops.push(leaf_ops[i]);
        } else {
            ops.push(dynamic_false());
        }
    }
    ops.push(dynamic_false());

    let op_from_lit = |lit: Lit, ops: &[AigOperand]| -> AigOperand {
        let mut op = ops[lit.id as usize];
        if lit.negated {
            op = op.negate();
        }
        op
    };

    for node in &frag.nodes {
        let op = match *node {
            crate::cut_db::fragment::FragmentNode::And2 { a, b } => {
                let a_op = op_from_lit(a, &ops);
                let b_op = op_from_lit(b, &ops);
                state.add_and_with_pir_node_ids_excluding(a_op, b_op, pir_node_ids, excluded)?
            }
        };
        ops.push(op);
    }

    let output = op_from_lit(frag.output, &ops);
    state.add_pir_node_ids(output.node, pir_node_ids)?;
    Ok(output)
}

fn instantiate_sop_dynamic(
    state: &mut DynamicStructuralHash,
    sop: &SopReplacement,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
    excluded: &BTreeSet<AigRef>,
) -> Result<AigOperand, String> {
    let mut output =
        instantiate_factored_expr_dynamic(state, &sop.factored, leaf_ops, pir_node_ids, excluded)?;
    if sop.output_negated {
        output = output.negate();
    }
    state.add_pir_node_ids(output.node, pir_node_ids)?;
    Ok(output)
}

fn instantiate_factored_expr_dynamic(
    state: &mut DynamicStructuralHash,
    expr: &FactoredExpr,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
    excluded: &BTreeSet<AigRef>,
) -> Result<AigOperand, String> {
    let mut ops = Vec::with_capacity(expr.nodes().len());
    for node in expr.nodes().iter().copied() {
        let op = match node {
            FactoredExprNode::Const(false) => dynamic_false(),
            FactoredExprNode::Const(true) => dynamic_true(),
            FactoredExprNode::Lit { var, negated } => {
                let mut op = leaf_ops[var];
                if negated {
                    op = op.negate();
                }
                op
            }
            FactoredExprNode::And { lhs, rhs } => state.add_and_with_pir_node_ids_excluding(
                ops[lhs.0],
                ops[rhs.0],
                pir_node_ids,
                excluded,
            )?,
            FactoredExprNode::Or { lhs, rhs } => {
                dynamic_add_or_binary(state, ops[lhs.0], ops[rhs.0], pir_node_ids, excluded)?
            }
        };
        state.add_pir_node_ids(op.node, pir_node_ids)?;
        ops.push(op);
    }
    Ok(ops[expr.root().0])
}

fn instantiate_replacement_dynamic(
    state: &mut DynamicStructuralHash,
    implementation: &ReplacementImpl,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
    excluded: &BTreeSet<AigRef>,
) -> Result<AigOperand, String> {
    match implementation {
        ReplacementImpl::Fragment { frag, .. } => {
            instantiate_fragment_dynamic(state, frag, leaf_ops, pir_node_ids, excluded)
        }
        ReplacementImpl::Sop(sop) => {
            instantiate_sop_dynamic(state, sop, leaf_ops, pir_node_ids, excluded)
        }
    }
}

fn output_path_len(g: &GateFn, max_output_node_depth: usize) -> usize {
    let has_output_bit = g.outputs.iter().any(|output| output.get_bit_count() != 0);
    if has_output_bit {
        max_output_node_depth + 1
    } else {
        0
    }
}

/// Collects AND nodes on paths that reach the maximum output node depth.
///
/// `max_output_node_depth` is measured in AIG edges from an input/literal to an
/// output operand node. It is therefore one less than `deepest_path.len()` for
/// a non-empty path.
fn collect_critical_roots(
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    max_output_node_depth: usize,
) -> Vec<AigRef> {
    if max_output_node_depth == 0 {
        return Vec::new();
    }

    let g = structural_hash_state.gate_fn();
    let mut roots = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut worklist: Vec<AigRef> = g
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .filter_map(|op| {
            if depth_state.forward_depth(structural_hash_state, op.node)
                == Some(max_output_node_depth)
            {
                Some(op.node)
            } else {
                None
            }
        })
        .collect();

    while let Some(node) = worklist.pop() {
        if !visited.insert(node) {
            continue;
        }
        let AigNode::And2 { a, b, .. } = &g.gates[node.id] else {
            continue;
        };
        roots.insert(node);
        let node_depth = live_forward_depth(depth_state, structural_hash_state, node);
        for child in [a.node, b.node] {
            let Some(child_depth) = depth_state.forward_depth(structural_hash_state, child) else {
                continue;
            };
            if child_depth + 1 == node_depth {
                worklist.push(child);
            }
        }
    }

    let mut roots: Vec<AigRef> = roots.into_iter().collect();
    roots.sort_by(|a, b| {
        live_forward_depth(depth_state, structural_hash_state, *b)
            .cmp(&live_forward_depth(depth_state, structural_hash_state, *a))
            .then_with(|| a.id.cmp(&b.id))
    });
    roots
}

fn collect_live_fanout_cone(state: &DynamicStructuralHash, root: AigRef) -> BTreeSet<AigRef> {
    let mut seen = BTreeSet::new();
    let mut stack = state.fanout_nodes(root);
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        stack.extend(state.fanout_nodes(node));
    }
    seen
}

fn collect_new_nodes_and_fanouts(
    state: &DynamicStructuralHash,
    first_new_id: usize,
    after_gate_len: usize,
) -> BTreeSet<AigRef> {
    let mut roots = BTreeSet::new();
    for id in first_new_id..after_gate_len {
        let node = AigRef { id };
        if !state.is_live(node) {
            continue;
        }
        roots.insert(node);
        roots.extend(state.fanout_nodes(node));
    }
    roots
}

fn add_depth_dirty_node_and_fanins(nodes: &mut BTreeSet<AigRef>, g: &GateFn, node: AigRef) {
    if node.id >= g.gates.len() {
        return;
    }
    nodes.insert(node);
    if let AigNode::And2 { a, b, .. } = g.gates[node.id] {
        nodes.insert(a.node);
        nodes.insert(b.node);
    }
}

fn collect_replacement_depth_changed_nodes(
    state: &DynamicStructuralHash,
    first_new_id: usize,
    mut dirty_nodes: BTreeSet<AigRef>,
) -> Vec<AigRef> {
    let g = state.gate_fn();

    for id in first_new_id..g.gates.len() {
        add_depth_dirty_node_and_fanins(&mut dirty_nodes, g, AigRef { id });
    }

    for output in &g.outputs {
        for op in output.bit_vector.iter_lsb_to_msb() {
            dirty_nodes.insert(op.node);
        }
    }

    let seeds: Vec<AigRef> = dirty_nodes.iter().copied().collect();
    for node in seeds {
        add_depth_dirty_node_and_fanins(&mut dirty_nodes, g, node);
    }

    dirty_nodes.into_iter().collect()
}

fn collect_mffc_nodes_under_cut(
    state: &DynamicStructuralHash,
    root: AigRef,
    leaf_ops: &[AigOperand],
) -> BTreeSet<AigRef> {
    struct SparseRefCounts<'a> {
        state: &'a DynamicStructuralHash,
        overrides: SmallVec<[(AigRef, usize); 16]>,
    }

    impl<'a> SparseRefCounts<'a> {
        fn new(state: &'a DynamicStructuralHash) -> Self {
            Self {
                state,
                overrides: SmallVec::new(),
            }
        }

        fn get(&self, node: AigRef) -> usize {
            self.overrides
                .iter()
                .find_map(|(candidate, value)| (*candidate == node).then_some(*value))
                .unwrap_or_else(|| self.state.use_count(node))
        }

        fn set(&mut self, node: AigRef, value: usize) {
            if let Some((_, existing)) = self
                .overrides
                .iter_mut()
                .find(|(candidate, _)| *candidate == node)
            {
                *existing = value;
                return;
            }
            self.overrides.push((node, value));
        }

        fn decrement(&mut self, node: AigRef) -> Option<usize> {
            let refs = self.get(node);
            if refs == 0 {
                return None;
            }
            let updated = refs - 1;
            self.set(node, updated);
            Some(updated)
        }
    }

    fn deref_nodes(
        g: &GateFn,
        root: AigRef,
        boundary: &BTreeSet<AigRef>,
        refs: &mut SparseRefCounts<'_>,
        removable: &mut BTreeSet<AigRef>,
    ) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if boundary.contains(&node) {
                continue;
            }
            let Some(updated_refs) = refs.decrement(node) else {
                continue;
            };
            if updated_refs != 0 {
                continue;
            }
            let AigNode::And2 { a, b, .. } = &g.gates[node.id] else {
                continue;
            };
            removable.insert(node);
            stack.push(a.node);
            stack.push(b.node);
        }
    }

    let g = state.gate_fn();
    let boundary: BTreeSet<AigRef> = leaf_ops.iter().map(|op| op.node).collect();
    let mut removable = BTreeSet::new();
    if boundary.contains(&root) || state.use_count(root) == 0 {
        return removable;
    }

    let mut refs = SparseRefCounts::new(state);
    // Replacing `root` rewires all of its fanouts to the replacement, so model
    // the dereference as a single use regardless of current fanout count.
    refs.set(root, 1);
    deref_nodes(g, root, &boundary, &mut refs, &mut removable);
    removable
}

fn collect_internal_and_nodes_under_cut(
    g: &GateFn,
    root: AigRef,
    leaf_ops: &[AigOperand],
) -> BTreeSet<AigRef> {
    fn collect_internal_nodes(
        g: &GateFn,
        root: AigRef,
        boundary: &BTreeSet<AigRef>,
        visited: &mut BTreeSet<AigRef>,
    ) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if boundary.contains(&node) || !visited.insert(node) {
                continue;
            }
            let AigNode::And2 { a, b, .. } = &g.gates[node.id] else {
                visited.remove(&node);
                continue;
            };
            stack.push(a.node);
            stack.push(b.node);
        }
    }

    let boundary: BTreeSet<AigRef> = leaf_ops.iter().map(|op| op.node).collect();
    let mut internal_nodes = BTreeSet::new();
    collect_internal_nodes(g, root, &boundary, &mut internal_nodes);
    internal_nodes
}

struct RootAreaCandidates {
    candidates: Vec<AreaCandidate>,
    candidates_considered: usize,
}

struct ConstructedLargeConeCandidates {
    candidates: Vec<LargeConeCandidate>,
    candidates_considered: usize,
}

fn finite_backward_depth(backward_depths: &[usize], node: AigRef) -> Option<usize> {
    let backward_depth = backward_depths.get(node.id).copied()?;
    (backward_depth != usize::MAX).then_some(backward_depth)
}

fn replacement_preserves_output_depth(
    new_root_depth: usize,
    root_backward_depth: usize,
    output_depth_cap: usize,
) -> bool {
    new_root_depth.saturating_add(root_backward_depth) <= output_depth_cap
}

fn choose_area_candidate_replacements_for_root(
    g: &GateFn,
    root: AigRef,
    cut_enumerator: &mut CutEnumerator,
    output_depth_cap: usize,
    allow_depth_increase: bool,
    db: &CutDb,
    candidate_evals: &mut usize,
    stats: &mut AreaCandidateStats,
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
) -> RootAreaCandidates {
    let mut cands = Vec::new();
    let mut candidates_considered = 0usize;
    let root_cuts = cut_enumerator.cuts_for_root(g, root);
    let Some(root_backward_depth) = depth_state.backward_depth(structural_hash_state, root) else {
        return RootAreaCandidates {
            candidates: cands,
            candidates_considered,
        };
    };

    for cut in &root_cuts {
        let cut_leaves = cut.leaves.as_slice();
        if cut_leaves.len() == 1 && cut_leaves[0].node == root && !cut_leaves[0].negated {
            stats.skipped_identity_cut += 1;
            continue;
        }
        if cut.leaves.len() > 4 {
            stats.skipped_large_cut += 1;
            continue;
        }

        let mffc_nodes = collect_mffc_nodes_under_cut(structural_hash_state, root, cut_leaves);
        if mffc_nodes.is_empty() {
            stats.rejected_empty_mffc += 1;
            continue;
        }
        let internal_and_nodes = collect_internal_and_nodes_under_cut(g, root, cut_leaves);
        let cut_is_mffc = internal_and_nodes == mffc_nodes;

        let (xform, pareto) = db.lookup(cut.tt.0);
        candidates_considered += pareto.len();
        for p in pareto {
            *candidate_evals += 1;

            let frag = p.frag.apply_npn(xform);
            let input_depths = frag.input_depths();
            let implementation = ReplacementImpl::Fragment { frag, input_depths };
            let new_root_depth = replacement_depth_from_inputs(
                cut_leaves,
                &implementation,
                structural_hash_state,
                depth_state,
            );
            if !allow_depth_increase
                && !replacement_preserves_output_depth(
                    new_root_depth,
                    root_backward_depth,
                    output_depth_cap,
                )
            {
                stats.rejected_depth += 1;
                if cut_is_mffc {
                    stats.mffc_rejected_depth += 1;
                } else {
                    stats.non_mffc_rejected_depth += 1;
                }
                continue;
            }
            let raw_and_count = p.ands as usize;
            let mut replacement = Replacement {
                root,
                leaf_ops: cut.leaves.to_vec(),
                implementation,
                score_depth: new_root_depth,
                score_ands: raw_and_count,
                raw_score_ands: raw_and_count,
                structural_hash_only_area_win: false,
            };
            let Some(cost) =
                (match gate_count_diff_for_replacement(structural_hash_state, &replacement) {
                    Ok(cost) => cost,
                    Err(e) => {
                        log::debug!(
                            "cut-db area exact cost rejected candidate root={:?}: {}",
                            replacement.root,
                            e
                        );
                        continue;
                    }
                })
            else {
                continue;
            };

            if cost.live_and_delta <= 0 {
                stats.rejected_no_area_gain += 1;
                if cost.live_and_delta == 0 {
                    stats.rejected_equal_area += 1;
                } else {
                    stats.rejected_area_increase += 1;
                }
                if cut_is_mffc {
                    stats.mffc_rejected_no_area_gain += 1;
                    if cost.live_and_delta == 0 {
                        stats.mffc_rejected_equal_area += 1;
                    } else {
                        stats.mffc_rejected_area_increase += 1;
                    }
                } else {
                    stats.non_mffc_rejected_no_area_gain += 1;
                    if cost.live_and_delta == 0 {
                        stats.non_mffc_rejected_equal_area += 1;
                    } else {
                        stats.non_mffc_rejected_area_increase += 1;
                    }
                }
                continue;
            }

            let area_gain = cost.live_and_delta as usize;
            if cut_is_mffc {
                stats.mffc_viable_candidates += 1;
            } else {
                stats.non_mffc_viable_candidates += 1;
            }
            let structural_hash_only_area_win = mffc_nodes.len() <= raw_and_count && area_gain > 0;
            if structural_hash_only_area_win {
                stats.viable_structural_hash_only_area_win += 1;
            }
            replacement.score_ands = mffc_nodes.len().saturating_sub(area_gain);
            replacement.structural_hash_only_area_win = structural_hash_only_area_win;
            let slack_consumed = new_root_depth.saturating_sub(live_forward_depth(
                depth_state,
                structural_hash_state,
                root,
            ));
            cands.push(AreaCandidate {
                replacement,
                mffc_nodes: mffc_nodes.clone(),
                area_gain,
                before_live_and_count: cost.before_live_ands,
                after_live_and_count: cost.after_live_ands,
                new_root_depth,
                slack_consumed,
            });
        }
    }

    cands.sort_by(|a, b| {
        b.area_gain
            .cmp(&a.area_gain)
            .then_with(|| a.slack_consumed.cmp(&b.slack_consumed))
            .then_with(|| a.new_root_depth.cmp(&b.new_root_depth))
            .then_with(|| b.mffc_nodes.len().cmp(&a.mffc_nodes.len()))
            .then_with(|| a.replacement.root.id.cmp(&b.replacement.root.id))
            .then_with(|| a.replacement.leaf_ops.cmp(&b.replacement.leaf_ops))
    });

    RootAreaCandidates {
        candidates: cands,
        candidates_considered,
    }
}

/// Applies depth and exact area-cost acceptance to already-constructed
/// large-cone candidates.
fn cost_large_cone_area_candidates_for_root(
    constructed: ConstructedLargeConeCandidates,
    root_depth: usize,
    root_backward_depth: usize,
    output_depth_cap: usize,
    allow_depth_increase: bool,
    stats: &mut LargeConeCandidateStats,
    structural_hash_state: &DynamicStructuralHash,
) -> RootAreaCandidates {
    let mut cands = Vec::new();

    for cand in constructed.candidates {
        let LargeConeCandidate {
            mut replacement,
            mffc_nodes,
            new_root_depth,
            raw_and_count,
            variant_order: _,
        } = cand;
        if !allow_depth_increase
            && !replacement_preserves_output_depth(
                new_root_depth,
                root_backward_depth,
                output_depth_cap,
            )
        {
            stats.rejected_depth += 1;
            continue;
        }

        let Some(cost) = (match gate_count_diff_for_replacement(structural_hash_state, &replacement)
        {
            Ok(cost) => cost,
            Err(e) => {
                log::debug!(
                    "cut-db large-cone exact cost rejected candidate root={:?}: {}",
                    replacement.root,
                    e
                );
                continue;
            }
        }) else {
            continue;
        };
        if cost.live_and_delta <= 0 {
            stats.rejected_no_area_gain += 1;
            continue;
        }

        let area_gain = cost.live_and_delta as usize;
        let structural_hash_only_area_win = mffc_nodes.len() <= raw_and_count && area_gain > 0;
        if structural_hash_only_area_win {
            stats.viable_structural_hash_only_area_win += 1;
        }
        replacement.score_ands = mffc_nodes.len().saturating_sub(area_gain);
        replacement.structural_hash_only_area_win = structural_hash_only_area_win;
        let slack_consumed = new_root_depth.saturating_sub(root_depth);
        cands.push(AreaCandidate {
            replacement,
            mffc_nodes,
            area_gain,
            before_live_and_count: cost.before_live_ands,
            after_live_and_count: cost.after_live_ands,
            new_root_depth,
            slack_consumed,
        });
    }

    cands.sort_by(|a, b| {
        b.area_gain
            .cmp(&a.area_gain)
            .then_with(|| a.slack_consumed.cmp(&b.slack_consumed))
            .then_with(|| a.new_root_depth.cmp(&b.new_root_depth))
            .then_with(|| b.mffc_nodes.len().cmp(&a.mffc_nodes.len()))
            .then_with(|| a.replacement.root.id.cmp(&b.replacement.root.id))
            .then_with(|| a.replacement.leaf_ops.cmp(&b.replacement.leaf_ops))
    });

    RootAreaCandidates {
        candidates: cands,
        candidates_considered: constructed.candidates_considered,
    }
}

fn filter_large_cone_depth_candidates_for_root(
    constructed: ConstructedLargeConeCandidates,
    root_depth: usize,
    stats: &mut LargeConeCandidateStats,
) -> ConstructedLargeConeCandidates {
    let mut candidates = Vec::new();

    for cand in constructed.candidates {
        if cand.new_root_depth >= root_depth {
            stats.rejected_depth += 1;
            continue;
        }
        candidates.push(cand);
    }

    ConstructedLargeConeCandidates {
        candidates,
        candidates_considered: constructed.candidates_considered,
    }
}

fn replacement_pir_node_ids(g: &GateFn, repl: &Replacement) -> PirNodeIds {
    let mut replacement_pir_node_ids = PirNodeIds::new();
    replacement_pir_node_ids.extend(g.gates[repl.root.id].get_pir_node_ids().iter().copied());
    for leaf in &repl.leaf_ops {
        for pir_node_id in g.gates[leaf.node.id].get_pir_node_ids() {
            match replacement_pir_node_ids.binary_search(pir_node_id) {
                Ok(_) => {}
                Err(index) => replacement_pir_node_ids.insert(index, *pir_node_id),
            }
        }
    }
    replacement_pir_node_ids
}

pub(super) fn apply_replacement_to_dynamic_hash(
    state: &mut DynamicStructuralHash,
    repl: &Replacement,
) -> Result<Option<AigOperand>, String> {
    if !state.is_live(repl.root) {
        return Ok(None);
    }
    let pir_node_ids = replacement_pir_node_ids(state.gate_fn(), repl);
    let excluded = BTreeSet::from([repl.root]);
    let new_op = instantiate_replacement_dynamic(
        state,
        &repl.implementation,
        &repl.leaf_ops,
        pir_node_ids.as_slice(),
        &excluded,
    )?;
    state.replace_node_with_operand(repl.root, new_op)?;
    Ok(Some(new_op))
}

pub(super) fn cleanup_dangling_new_dynamic_hash_nodes(
    state: &mut DynamicStructuralHash,
    first_new_id: usize,
) -> Result<(), String> {
    for id in (first_new_id..state.gate_fn().gates.len()).rev() {
        let node = AigRef { id };
        if state.is_live(node)
            && matches!(state.gate_fn().gates[id], AigNode::And2 { .. })
            && state.fanout_count(node) == 0
            && state.output_use_count(node) == 0
        {
            state.delete_node(node)?;
        }
    }
    Ok(())
}

fn enqueue_live_and_roots<I>(
    structural_hash_state: &DynamicStructuralHash,
    roots: I,
    pending_roots: &mut BTreeSet<AigRef>,
    root_queue: &mut VecDeque<AigRef>,
    roots_total: &mut usize,
) where
    I: IntoIterator<Item = AigRef>,
{
    for node in roots {
        if structural_hash_state.is_live(node)
            && matches!(
                structural_hash_state.gate_fn().gates.get(node.id),
                Some(AigNode::And2 { .. })
            )
            && pending_roots.insert(node)
        {
            *roots_total += 1;
            root_queue.push_back(node);
        }
    }
}

fn is_live_critical_and_root(
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    max_output_node_depth: usize,
    node: AigRef,
) -> bool {
    if !structural_hash_state.is_live(node)
        || !matches!(
            structural_hash_state.gate_fn().gates.get(node.id),
            Some(AigNode::And2 { .. })
        )
    {
        return false;
    }

    let Some(forward_depth) = depth_state.forward_depth(structural_hash_state, node) else {
        return false;
    };
    let Some(backward_depth) = depth_state.backward_depth(structural_hash_state, node) else {
        return false;
    };

    forward_depth.checked_add(backward_depth) == Some(max_output_node_depth)
}

struct CriticalRootTracker {
    roots: BTreeSet<AigRef>,
    max_output_node_depth: usize,
}

impl CriticalRootTracker {
    fn new(
        structural_hash_state: &DynamicStructuralHash,
        depth_state: &DynamicDepthState,
        max_output_node_depth: usize,
    ) -> Self {
        Self {
            roots: collect_critical_roots(
                structural_hash_state,
                depth_state,
                max_output_node_depth,
            )
            .into_iter()
            .collect(),
            max_output_node_depth,
        }
    }

    fn len(&self) -> usize {
        self.roots.len()
    }

    fn iter(&self) -> impl Iterator<Item = AigRef> + '_ {
        self.roots.iter().copied()
    }

    fn refresh_after_depth_update(
        &mut self,
        structural_hash_state: &DynamicStructuralHash,
        depth_state: &DynamicDepthState,
        max_output_node_depth: usize,
        affected_nodes: &[AigRef],
    ) {
        if max_output_node_depth != self.max_output_node_depth {
            *self = Self::new(structural_hash_state, depth_state, max_output_node_depth);
            return;
        }

        for node in affected_nodes {
            if is_live_critical_and_root(
                structural_hash_state,
                depth_state,
                max_output_node_depth,
                *node,
            ) {
                self.roots.insert(*node);
            } else {
                self.roots.remove(node);
            }
        }
    }

    fn enqueue_all(
        &self,
        structural_hash_state: &DynamicStructuralHash,
        depth_state: &DynamicDepthState,
        pending_roots: &mut BTreeSet<AigRef>,
        root_queue: &mut VecDeque<AigRef>,
    ) -> usize {
        let mut roots: Vec<AigRef> = self.iter().collect();
        roots.sort_by(|a, b| {
            live_forward_depth(depth_state, structural_hash_state, *b)
                .cmp(&live_forward_depth(depth_state, structural_hash_state, *a))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut added = 0usize;
        for root in roots {
            if pending_roots.insert(root) {
                added += 1;
                root_queue.push_back(root);
            }
        }
        added
    }
}

fn rewrite_gatefn_depth_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();

    // Walk critical-path roots from the current graph once. Accepted
    // replacements are materialized immediately into the dynamic graph state,
    // and all current critical-path roots are requeued before the round
    // continues.
    {
        let t_iter0 = Instant::now();
        let t0 = Instant::now();
        let mut structural_hash_state = DynamicStructuralHash::new(cur.clone())
            .expect("dynamic local strash should initialize from cut-db input graph");
        let t_use_count_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let mut depth_state = DynamicDepthState::new(&structural_hash_state)
            .expect("dynamic depth state should initialize from cut-db input graph");
        let t_depth_ms = t1.elapsed().as_millis();
        let mut cur_output_node_depth = depth_state
            .max_output_node_depth(&structural_hash_state)
            .expect("dynamic depth state should compute output depth");
        let cur_path_len = output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);

        log::info!(
            "cut-db depth round: live_nodes={} path_len={} output_node_depth={}",
            structural_hash_state.live_nodes().len(),
            cur_path_len,
            cur_output_node_depth
        );

        let mut critical_roots =
            CriticalRootTracker::new(&structural_hash_state, &depth_state, cur_output_node_depth);
        let initial_critical_roots = critical_roots.len();
        let mut root_queue: VecDeque<AigRef> = critical_roots.iter().collect();
        let mut pending_roots = critical_roots.roots.clone();

        let mut after_path_len = cur_path_len;
        let mut candidates_considered: usize = 0;
        let mut candidate_evals: usize = 0;
        let mut accepted_count = 0usize;
        let mut roots_visited = 0usize;
        let mut cut_enumerator =
            CutEnumerator::new(structural_hash_state.gate_fn(), opts.max_cuts_per_node);
        let t_phase = Instant::now();
        while let Some(root) = root_queue.pop_front() {
            pending_roots.remove(&root);
            roots_visited += 1;
            if !is_live_critical_and_root(
                &structural_hash_state,
                &depth_state,
                cur_output_node_depth,
                root,
            ) {
                continue;
            }
            let root_cuts = cut_enumerator.cuts_for_root(structural_hash_state.gate_fn(), root);
            let root_depth = live_forward_depth(&depth_state, &structural_hash_state, root);
            let cands = choose_candidate_replacements_for_root(
                root,
                &root_cuts,
                &structural_hash_state,
                &depth_state,
                db,
            );
            candidates_considered += cands.len();
            for cand in cands {
                candidate_evals += 1;
                if cand.score_depth >= root_depth {
                    continue;
                }

                let before_g_for_verify = if opts.verify_delay_costing {
                    Some(structural_hash_state.gate_fn().clone())
                } else {
                    None
                };
                let mut invalidated_cut_nodes =
                    collect_live_fanout_cone(&structural_hash_state, cand.root);
                invalidated_cut_nodes.insert(cand.root);
                for leaf in &cand.leaf_ops {
                    invalidated_cut_nodes.insert(leaf.node);
                }
                let materialized = match materialize_replacement(&mut structural_hash_state, &cand)
                {
                    Ok(Some(materialized)) => materialized,
                    Ok(None) => {
                        panic!(
                            "cut-db depth virtual-accepted candidate did not materialize root={:?}",
                            cand.root
                        );
                    }
                    Err(e) => {
                        panic!(
                            "cut-db depth virtual-accepted candidate failed to materialize root={:?}: {}",
                            cand.root, e
                        );
                    }
                };
                invalidated_cut_nodes.insert(materialized.replacement_op.node);

                let changed_nodes = collect_replacement_depth_changed_nodes(
                    &structural_hash_state,
                    materialized.first_new_id,
                    invalidated_cut_nodes.clone(),
                );
                let affected_depth_nodes = depth_state
                    .refresh_from_changed_nodes_collecting_affected(
                        &structural_hash_state,
                        &changed_nodes,
                    )
                    .expect(
                        "incremental dynamic depth update should succeed for cut-db output graph",
                    );
                let after_output_node_depth = depth_state
                    .max_output_node_depth(&structural_hash_state)
                    .expect("dynamic depth state should compute trial output depth");
                assert!(
                    after_output_node_depth <= cur_output_node_depth,
                    "cut-db depth local rewrite increased output depth from {cur_output_node_depth} to {after_output_node_depth}"
                );
                let replacement_depth = depth_state
                    .forward_depth(&structural_hash_state, materialized.replacement_op.node)
                    .expect("dynamic depth state should compute replacement root depth");
                assert!(
                    replacement_depth < root_depth,
                    "cut-db depth local rewrite did not shorten root {:?}: old root depth {}, replacement depth {}",
                    cand.root,
                    root_depth,
                    replacement_depth
                );

                if opts.verify_delay_costing {
                    verify_delay_not_increased(
                        "depth rewrite",
                        before_g_for_verify.as_ref().unwrap(),
                        structural_hash_state.gate_fn(),
                        cur_output_node_depth,
                    );
                }

                cur_output_node_depth = after_output_node_depth;
                critical_roots.refresh_after_depth_update(
                    &structural_hash_state,
                    &depth_state,
                    cur_output_node_depth,
                    &affected_depth_nodes,
                );
                after_path_len =
                    output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);
                accepted_count += 1;

                cut_enumerator.sync_len(structural_hash_state.gate_fn());
                cut_enumerator.invalidate_nodes(&invalidated_cut_nodes);

                critical_roots.enqueue_all(
                    &structural_hash_state,
                    &depth_state,
                    &mut pending_roots,
                    &mut root_queue,
                );
                break;
            }
        }
        let cut_stats = cut_enumerator.stats();
        let t_phase_ms = t_phase.elapsed().as_millis();

        if accepted_count != 0 {
            let new_g = dce(structural_hash_state.gate_fn());
            if opts.verify_delay_costing {
                let checked_depth = independent_output_node_depth(&new_g);
                assert_eq!(
                    checked_depth, cur_output_node_depth,
                    "cut-db depth final DCE changed output depth from tracked {cur_output_node_depth} to checked {checked_depth}"
                );
            }
            cur = new_g;
        }

        log::debug!(
            "cut-db depth round timings: use_count_ms={} depth_ms={} cuts_ms={} phase_ms={} critical_roots={} roots_visited={} cands_considered={} candidate_evals={} accepted={} before_path_len={} after_path_len={} round_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            cut_stats.elapsed_ms,
            t_phase_ms,
            initial_critical_roots,
            roots_visited,
            candidates_considered,
            candidate_evals,
            accepted_count,
            cur_path_len,
            after_path_len,
            t_iter0.elapsed().as_millis()
        );

        if accepted_count == 0 {
            log::info!("cut-db depth rewrite: no depth-improving candidate found; stopping");
        } else {
            log::info!(
                "cut-db depth rewrite: completed single dynamic depth round with {} accepted rewrites",
                accepted_count
            );
        }
    }

    cur
}

fn rewrite_gatefn_area_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();
    let mut iter: usize = 0;

    loop {
        if opts.max_iterations != 0 && iter >= opts.max_iterations {
            log::info!(
                "cut-db area rewrite: reached max_iterations={}; stopping",
                opts.max_iterations
            );
            break;
        }
        iter += 1;

        let t_iter0 = Instant::now();
        let t0 = Instant::now();
        let mut structural_hash_state = DynamicStructuralHash::new(cur.clone())
            .expect("dynamic local strash should initialize from cut-db input graph");
        let t_use_count_ms = t0.elapsed().as_millis();
        let live_nodes = structural_hash_state.live_nodes();
        let live_and_count = structural_hash_state.live_and_count();

        let t1 = Instant::now();
        let mut depth_state = DynamicDepthState::new(&structural_hash_state)
            .expect("dynamic depth state should initialize from cut-db input graph");
        let t_depth_ms = t1.elapsed().as_millis();
        let cur_output_node_depth = depth_state
            .max_output_node_depth(&structural_hash_state)
            .expect("dynamic depth state should compute output depth");
        let cur_path_len = output_path_len(&cur, cur_output_node_depth);

        log::info!(
            "cut-db area round: live_nodes={} live_ands={} path_len={} output_node_depth={}",
            live_nodes.len(),
            live_and_count,
            cur_path_len,
            cur_output_node_depth
        );

        let t_phase = Instant::now();
        let mut roots = structural_hash_state.live_and_nodes();
        roots.sort_by(|a, b| {
            live_forward_depth(&depth_state, &structural_hash_state, *b)
                .cmp(&live_forward_depth(
                    &depth_state,
                    &structural_hash_state,
                    *a,
                ))
                .then_with(|| a.id.cmp(&b.id))
        });
        let roots_total = roots.len();
        let mut candidate_stats = AreaCandidateStats {
            roots_total,
            ..AreaCandidateStats::default()
        };
        let mut rewrite_stats = AreaRewriteSelectionStats::default();
        let mut cut_stats = CutEnumerationStats::default();
        let mut cut_enumerator =
            CutEnumerator::new(structural_hash_state.gate_fn(), opts.max_cuts_per_node);
        let mut pending_roots: BTreeSet<AigRef> = roots.iter().copied().collect();
        let mut root_queue: VecDeque<AigRef> = roots.into();
        let mut candidates_considered = 0usize;
        let mut candidate_evals = 0usize;
        let mut area_candidate_count = 0usize;
        let mut accepted_count = 0usize;

        while !root_queue.is_empty() {
            let root = root_queue
                .pop_front()
                .expect("root queue checked non-empty above");
            pending_roots.remove(&root);
            candidate_stats.roots_visited += 1;
            if !structural_hash_state.is_live(root) {
                continue;
            }
            if !matches!(
                structural_hash_state.gate_fn().gates[root.id],
                AigNode::And2 { .. }
            ) {
                continue;
            }
            if depth_state
                .backward_depth(&structural_hash_state, root)
                .is_none()
            {
                candidate_stats.skipped_unrequired_depth += 1;
                continue;
            }

            let root_result = {
                choose_area_candidate_replacements_for_root(
                    structural_hash_state.gate_fn(),
                    root,
                    &mut cut_enumerator,
                    cur_output_node_depth,
                    opts.mode.allows_area_depth_increase(),
                    db,
                    &mut candidate_evals,
                    &mut candidate_stats,
                    &structural_hash_state,
                    &depth_state,
                )
            };
            candidates_considered += root_result.candidates_considered;
            area_candidate_count += root_result.candidates.len();
            let mut root_candidates = root_result.candidates;
            root_candidates.sort_by(|a, b| {
                b.area_gain
                    .cmp(&a.area_gain)
                    .then_with(|| a.slack_consumed.cmp(&b.slack_consumed))
                    .then_with(|| a.new_root_depth.cmp(&b.new_root_depth))
                    .then_with(|| b.mffc_nodes.len().cmp(&a.mffc_nodes.len()))
                    .then_with(|| a.replacement.root.id.cmp(&b.replacement.root.id))
                    .then_with(|| a.replacement.leaf_ops.cmp(&b.replacement.leaf_ops))
            });

            if let Some(cand) = root_candidates.into_iter().next() {
                let first_new_id = structural_hash_state.gate_fn().gates.len();
                debug_assert_eq!(
                    structural_hash_state.live_and_count(),
                    cand.before_live_and_count,
                    "cached exact area cost should be evaluated against the current graph"
                );

                if opts.verify_area_costing {
                    verify_live_area_cost_delta_exact(
                        "area rewrite",
                        cand.before_live_and_count,
                        cand.after_live_and_count,
                        cand.area_gain,
                    );
                }
                if cand.replacement.structural_hash_only_area_win {
                    rewrite_stats.selected_structural_hash_only_area_win += 1;
                }
                rewrite_stats.selected_raw_score_ands += cand.replacement.raw_score_ands;
                rewrite_stats.selected_hash_score_ands += cand.replacement.score_ands;

                let fanout_cone_nodes =
                    collect_live_fanout_cone(&structural_hash_state, cand.replacement.root);
                let mut invalidated_cut_nodes = fanout_cone_nodes.clone();
                invalidated_cut_nodes.insert(cand.replacement.root);
                for leaf in &cand.replacement.leaf_ops {
                    invalidated_cut_nodes.insert(leaf.node);
                }
                invalidated_cut_nodes.extend(cand.mffc_nodes.iter().copied());
                let replacement = cand.replacement.clone();
                let materialized =
                    materialize_replacement(&mut structural_hash_state, &replacement)
                        .expect("exact replacement cost should materialize successfully")
                        .expect("exact replacement cost accepted a non-materialized replacement");
                invalidated_cut_nodes.insert(materialized.replacement_op.node);
                assert_eq!(
                    structural_hash_state.live_and_count(),
                    cand.after_live_and_count,
                    "exact replacement cost did not match materialized live AND count"
                );
                let after_gate_len = structural_hash_state.gate_fn().gates.len();
                let new_nodes_and_fanouts = collect_new_nodes_and_fanouts(
                    &structural_hash_state,
                    first_new_id,
                    after_gate_len,
                );
                let changed_nodes = collect_replacement_depth_changed_nodes(
                    &structural_hash_state,
                    materialized.first_new_id,
                    invalidated_cut_nodes.clone(),
                );
                depth_state
                    .refresh_from_changed_nodes(&structural_hash_state, &changed_nodes)
                    .expect(
                        "incremental dynamic depth update should succeed for cut-db output graph",
                    );
                let after_output_node_depth = depth_state
                    .max_output_node_depth(&structural_hash_state)
                    .expect("dynamic depth state should compute output depth");
                if !opts.mode.allows_area_depth_increase() {
                    assert!(
                        after_output_node_depth <= cur_output_node_depth,
                        "cut-db area rewrite unexpectedly increased live output depth from {cur_output_node_depth} to {after_output_node_depth}"
                    );
                }
                cut_enumerator.sync_len(structural_hash_state.gate_fn());
                cut_enumerator.invalidate_nodes(&invalidated_cut_nodes);
                accepted_count += 1;

                enqueue_live_and_roots(
                    &structural_hash_state,
                    fanout_cone_nodes
                        .iter()
                        .copied()
                        .chain(new_nodes_and_fanouts.iter().copied()),
                    &mut pending_roots,
                    &mut root_queue,
                    &mut candidate_stats.roots_total,
                );
            }
        }
        cut_stats.add(cut_enumerator.stats());
        candidate_stats.viable_candidates = area_candidate_count;
        rewrite_stats.total_candidates = area_candidate_count;
        rewrite_stats.selected = accepted_count;
        let t_phase_ms = t_phase.elapsed().as_millis();

        let rebuild_ms = 0;
        let mut after_live_and_count = live_and_count;
        let mut after_path_len = cur_path_len;
        let mut round_materialized = false;
        if accepted_count != 0 {
            after_live_and_count = structural_hash_state.live_and_count();
            let after_output_node_depth = depth_state
                .max_output_node_depth(&structural_hash_state)
                .expect("dynamic depth state should compute final output depth");
            let new_g = dce(structural_hash_state.gate_fn());
            let dce_after_live_and_count = and_node_count(&new_g);
            after_path_len = output_path_len(&new_g, after_output_node_depth);

            if opts.verify_delay_costing && !opts.mode.allows_area_depth_increase() {
                verify_delay_not_increased(
                    "area rewrite final",
                    &cur,
                    &new_g,
                    cur_output_node_depth,
                );
            }
            if opts.verify_area_costing {
                verify_area_cost_delta(
                    "area rewrite final",
                    &cur,
                    &new_g,
                    live_and_count,
                    dce_after_live_and_count,
                );
            }

            if dce_after_live_and_count < live_and_count
                && (opts.mode.allows_area_depth_increase()
                    || after_output_node_depth <= cur_output_node_depth)
                && dce_after_live_and_count == after_live_and_count
            {
                after_live_and_count = dce_after_live_and_count;
                cur = new_g;
                round_materialized = true;
            } else {
                log::info!(
                    "cut-db area rewrite: discarded non-improving replacement round; before_ands={} after_ands={} dce_after_ands={} before_depth={} after_depth={} replacements={}",
                    live_and_count,
                    after_live_and_count,
                    dce_after_live_and_count,
                    cur_output_node_depth,
                    after_output_node_depth,
                    accepted_count
                );
                break;
            }
        }

        log::debug!(
            "cut-db area round timings: use_count_ms={} depth_ms={} cuts_ms={} phase_ms={} rebuild_ms={} area_candidates={} cands_considered={} candidate_evals={} accepted={} before_ands={} after_ands={} before_path_len={} after_path_len={} round_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            cut_stats.elapsed_ms,
            t_phase_ms,
            rebuild_ms,
            area_candidate_count,
            candidates_considered,
            candidate_evals,
            accepted_count,
            live_and_count,
            after_live_and_count,
            cur_path_len,
            after_path_len,
            t_iter0.elapsed().as_millis()
        );
        log::debug!(
            "cut-db area rejection stats: roots_total={} roots_visited={} skipped_unrequired_depth={} skipped_identity_cut={} skipped_large_cut={} rejected_empty_mffc={} rejected_no_area_gain={} rejected_equal_area={} rejected_area_increase={} rejected_depth={} viable_candidates={} viable_structural_hash_only_area_win={} mffc_rejected_no_area_gain={} mffc_rejected_equal_area={} mffc_rejected_area_increase={} mffc_rejected_depth={} mffc_viable_candidates={} non_mffc_rejected_no_area_gain={} non_mffc_rejected_equal_area={} non_mffc_rejected_area_increase={} non_mffc_rejected_depth={} non_mffc_viable_candidates={} round_materialized={} round_total_candidates={} round_selected={} round_selected_structural_hash_only_area_win={} round_selected_raw_score_ands={} round_selected_hash_score_ands={}",
            candidate_stats.roots_total,
            candidate_stats.roots_visited,
            candidate_stats.skipped_unrequired_depth,
            candidate_stats.skipped_identity_cut,
            candidate_stats.skipped_large_cut,
            candidate_stats.rejected_empty_mffc,
            candidate_stats.rejected_no_area_gain,
            candidate_stats.rejected_equal_area,
            candidate_stats.rejected_area_increase,
            candidate_stats.rejected_depth,
            candidate_stats.viable_candidates,
            candidate_stats.viable_structural_hash_only_area_win,
            candidate_stats.mffc_rejected_no_area_gain,
            candidate_stats.mffc_rejected_equal_area,
            candidate_stats.mffc_rejected_area_increase,
            candidate_stats.mffc_rejected_depth,
            candidate_stats.mffc_viable_candidates,
            candidate_stats.non_mffc_rejected_no_area_gain,
            candidate_stats.non_mffc_rejected_equal_area,
            candidate_stats.non_mffc_rejected_area_increase,
            candidate_stats.non_mffc_rejected_depth,
            candidate_stats.non_mffc_viable_candidates,
            round_materialized,
            rewrite_stats.total_candidates,
            rewrite_stats.selected,
            rewrite_stats.selected_structural_hash_only_area_win,
            rewrite_stats.selected_raw_score_ands,
            rewrite_stats.selected_hash_score_ands
        );

        if accepted_count == 0 {
            log::info!("cut-db area rewrite: no area-improving candidate found; stopping");
            break;
        }
    }

    cur
}

fn rewrite_gatefn_large_cone_depth_refactor(g: &GateFn, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();
    let mut sop_cover_memo = SopCoverMemo::default();

    {
        let t_iter0 = Instant::now();
        let t0 = Instant::now();
        let mut structural_hash_state = DynamicStructuralHash::new(cur.clone())
            .expect("dynamic local strash should initialize from cut-db input graph");
        let mut dense_counts = dense_use_counts_from_dynamic_hash(&structural_hash_state);
        let t_use_count_ms = t0.elapsed().as_millis();

        let t1 = Instant::now();
        let mut depth_state = DynamicDepthState::new(&structural_hash_state)
            .expect("dynamic depth state should initialize from cut-db input graph");
        let t_depth_ms = t1.elapsed().as_millis();
        let mut depths = depth_state.forward_depths().to_vec();
        let mut cur_output_node_depth = depth_state
            .max_output_node_depth(&structural_hash_state)
            .expect("dynamic depth state should compute output depth");
        let cur_path_len = output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);

        log::info!(
            "cut-db large-cone depth round: live_nodes={} live_ands={} path_len={} output_node_depth={}",
            structural_hash_state.live_nodes().len(),
            structural_hash_state.live_and_count(),
            cur_path_len,
            cur_output_node_depth
        );

        let mut critical_roots =
            CriticalRootTracker::new(&structural_hash_state, &depth_state, cur_output_node_depth);
        let initial_critical_roots = critical_roots.len();
        let mut root_queue: VecDeque<AigRef> = critical_roots.iter().collect();
        let mut pending_roots = critical_roots.roots.clone();

        let t_phase = Instant::now();
        let mut candidate_stats = LargeConeCandidateStats {
            roots_total: initial_critical_roots,
            ..LargeConeCandidateStats::default()
        };
        let mut candidates_considered = 0usize;
        let mut candidate_evals = 0usize;
        let mut large_candidate_count = 0usize;
        let mut accepted_count = 0usize;
        let mut selected_raw_score_ands = 0usize;
        let mut selected_sop_flat = 0usize;
        let mut selected_sop_arrival_balanced = 0usize;
        let mut selected_sop_factored = 0usize;
        let mut after_path_len = cur_path_len;
        let before_live_and_count = structural_hash_state.live_and_count();
        while let Some(root) = root_queue.pop_front() {
            pending_roots.remove(&root);
            candidate_stats.roots_visited += 1;
            if !is_live_critical_and_root(
                &structural_hash_state,
                &depth_state,
                cur_output_node_depth,
                root,
            ) {
                continue;
            }
            let root_depth = live_forward_depth(&depth_state, &structural_hash_state, root);

            let constructed = construct_large_cone_candidate_replacements_for_root(
                structural_hash_state.gate_fn(),
                root,
                &depths,
                &dense_counts,
                &mut candidate_evals,
                &mut candidate_stats,
                &structural_hash_state,
                &mut sop_cover_memo,
            );
            large_candidate_count += constructed.candidates.len();
            let root_result = filter_large_cone_depth_candidates_for_root(
                constructed,
                root_depth,
                &mut candidate_stats,
            );
            candidates_considered += root_result.candidates_considered;

            for cand in root_result.candidates {
                let before_g_for_verify = if opts.verify_delay_costing {
                    Some(structural_hash_state.gate_fn().clone())
                } else {
                    None
                };
                let mut invalidated_cut_nodes =
                    collect_live_fanout_cone(&structural_hash_state, cand.replacement.root);
                invalidated_cut_nodes.insert(cand.replacement.root);
                for leaf in &cand.replacement.leaf_ops {
                    invalidated_cut_nodes.insert(leaf.node);
                }
                invalidated_cut_nodes.extend(cand.mffc_nodes.iter().copied());
                let materialized = match materialize_replacement(
                    &mut structural_hash_state,
                    &cand.replacement,
                ) {
                    Ok(Some(materialized)) => materialized,
                    Ok(None) => {
                        panic!(
                            "cut-db large-cone depth virtual-accepted candidate did not materialize root={:?}",
                            cand.replacement.root
                        );
                    }
                    Err(e) => {
                        panic!(
                            "cut-db large-cone depth virtual-accepted candidate failed to materialize root={:?}: {}",
                            cand.replacement.root, e
                        );
                    }
                };
                invalidated_cut_nodes.insert(materialized.replacement_op.node);

                let changed_nodes = collect_replacement_depth_changed_nodes(
                    &structural_hash_state,
                    materialized.first_new_id,
                    invalidated_cut_nodes,
                );
                let affected_depth_nodes = depth_state
                    .refresh_from_changed_nodes_collecting_affected(
                        &structural_hash_state,
                        &changed_nodes,
                    )
                    .expect(
                        "incremental dynamic depth update should succeed for cut-db output graph",
                    );
                let after_output_node_depth = depth_state
                    .max_output_node_depth(&structural_hash_state)
                    .expect("dynamic depth state should compute trial output depth");
                assert!(
                    after_output_node_depth <= cur_output_node_depth,
                    "cut-db large-cone depth local rewrite increased output depth from {cur_output_node_depth} to {after_output_node_depth}"
                );
                let replacement_depth = depth_state
                    .forward_depth(&structural_hash_state, materialized.replacement_op.node)
                    .expect("dynamic depth state should compute replacement root depth");
                assert!(
                    replacement_depth < root_depth,
                    "cut-db large-cone depth local rewrite did not shorten root {:?}: old root depth {}, replacement depth {}",
                    cand.replacement.root,
                    root_depth,
                    replacement_depth
                );

                if opts.verify_delay_costing {
                    verify_delay_not_increased(
                        "large-cone depth rewrite",
                        before_g_for_verify.as_ref().unwrap(),
                        structural_hash_state.gate_fn(),
                        cur_output_node_depth,
                    );
                }

                depths = depth_state.forward_depths().to_vec();
                cur_output_node_depth = after_output_node_depth;
                critical_roots.refresh_after_depth_update(
                    &structural_hash_state,
                    &depth_state,
                    cur_output_node_depth,
                    &affected_depth_nodes,
                );
                dense_counts = dense_use_counts_from_dynamic_hash(&structural_hash_state);
                after_path_len =
                    output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);
                accepted_count += 1;
                selected_raw_score_ands += cand.raw_and_count;
                bump_selected_sop_variant(
                    selected_sop_variant_kind(&cand.replacement),
                    &mut selected_sop_flat,
                    &mut selected_sop_arrival_balanced,
                    &mut selected_sop_factored,
                );

                candidate_stats.roots_total += critical_roots.enqueue_all(
                    &structural_hash_state,
                    &depth_state,
                    &mut pending_roots,
                    &mut root_queue,
                );
                break;
            }
        }
        candidate_stats.viable_candidates = large_candidate_count;
        let t_phase_ms = t_phase.elapsed().as_millis();

        let mut after_live_and_count = before_live_and_count;
        if accepted_count != 0 {
            after_live_and_count = structural_hash_state.live_and_count();
            let new_g = dce(structural_hash_state.gate_fn());
            if opts.verify_delay_costing {
                let checked_depth = independent_output_node_depth(&new_g);
                assert_eq!(
                    checked_depth, cur_output_node_depth,
                    "cut-db large-cone depth final DCE changed output depth from tracked {cur_output_node_depth} to checked {checked_depth}"
                );
            }
            cur = new_g;
        }

        log::debug!(
            "cut-db large-cone depth round timings: use_count_ms={} depth_ms={} phase_ms={} critical_roots={} roots_visited={} cands_considered={} candidate_evals={} accepted={} before_ands={} after_ands={} before_path_len={} after_path_len={} round_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            t_phase_ms,
            initial_critical_roots,
            candidate_stats.roots_visited,
            candidates_considered,
            candidate_evals,
            accepted_count,
            before_live_and_count,
            after_live_and_count,
            cur_path_len,
            after_path_len,
            t_iter0.elapsed().as_millis()
        );
        log::debug!(
            "cut-db large-cone depth rejection stats: roots_total={} roots_visited={} rejected_no_cone={} rejected_small_cone={} rejected_empty_mffc={} rejected_sop_failed={} rejected_depth={} cones_built={} cone_leaves_sum={} cone_leaves_max={} cone_internal_sum={} cone_internal_max={} cone_mffc_sum={} cone_mffc_max={} sop_variants_sum={} sop_variants_max={} viable_candidates={} round_selected={} round_selected_raw_score_ands={} round_selected_sop_flat={} round_selected_sop_arrival_balanced={} round_selected_sop_factored={}",
            candidate_stats.roots_total,
            candidate_stats.roots_visited,
            candidate_stats.rejected_no_cone,
            candidate_stats.rejected_small_cone,
            candidate_stats.rejected_empty_mffc,
            candidate_stats.rejected_sop_failed,
            candidate_stats.rejected_depth,
            candidate_stats.cones_built,
            candidate_stats.cone_leaves_sum,
            candidate_stats.cone_leaves_max,
            candidate_stats.cone_internal_sum,
            candidate_stats.cone_internal_max,
            candidate_stats.cone_mffc_sum,
            candidate_stats.cone_mffc_max,
            candidate_stats.sop_variants_sum,
            candidate_stats.sop_variants_max,
            candidate_stats.viable_candidates,
            accepted_count,
            selected_raw_score_ands,
            selected_sop_flat,
            selected_sop_arrival_balanced,
            selected_sop_factored
        );

        if accepted_count == 0 {
            log::info!(
                "cut-db large-cone depth rewrite: no depth-improving candidate found; stopping"
            );
        } else {
            log::info!(
                "cut-db large-cone depth rewrite: completed single dynamic depth round with {} accepted rewrites",
                accepted_count
            );
        }
    }

    cur
}

fn rewrite_gatefn_large_cone_refactor(g: &GateFn, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();
    let mut iter: usize = 0;
    let mut sop_cover_memo = SopCoverMemo::default();

    loop {
        if opts.max_iterations != 0 && iter >= opts.max_iterations {
            log::info!(
                "cut-db large-cone refactor: reached max_iterations={}; stopping",
                opts.max_iterations
            );
            break;
        }
        iter += 1;

        let t_iter0 = Instant::now();
        let t0 = Instant::now();
        let mut structural_hash_state = DynamicStructuralHash::new(cur.clone())
            .expect("dynamic local strash should initialize from cut-db input graph");
        let mut dense_counts = dense_use_counts_from_dynamic_hash(&structural_hash_state);
        let t_use_count_ms = t0.elapsed().as_millis();
        let live_nodes = structural_hash_state.live_nodes();
        let live_and_count = structural_hash_state.live_and_count();

        let t1 = Instant::now();
        let mut depth_state = DynamicDepthState::new(&structural_hash_state)
            .expect("dynamic depth state should initialize from cut-db input graph");
        let t_depth_ms = t1.elapsed().as_millis();
        let mut depths = depth_state.forward_depths().to_vec();
        let mut backward_depths = depth_state.backward_depths().to_vec();
        let cur_output_node_depth = depth_state
            .max_output_node_depth(&structural_hash_state)
            .expect("dynamic depth state should compute output depth");
        let cur_path_len = output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);

        log::info!(
            "cut-db large-cone round: live_nodes={} live_ands={} path_len={} output_node_depth={}",
            live_nodes.len(),
            live_and_count,
            cur_path_len,
            cur_output_node_depth
        );

        let t_phase = Instant::now();
        let mut roots = structural_hash_state.live_and_nodes();
        roots.sort_by(|a, b| {
            depths[b.id]
                .cmp(&depths[a.id])
                .then_with(|| a.id.cmp(&b.id))
        });
        let roots_total = roots.len();
        let mut candidate_stats = LargeConeCandidateStats {
            roots_total,
            ..LargeConeCandidateStats::default()
        };
        let mut rewrite_stats = AreaRewriteSelectionStats::default();
        let mut pending_roots: BTreeSet<AigRef> = roots.iter().copied().collect();
        let mut root_queue: VecDeque<AigRef> = roots.into();
        let mut candidates_considered = 0usize;
        let mut candidate_evals = 0usize;
        let mut area_candidate_count = 0usize;
        let mut accepted_count = 0usize;

        while !root_queue.is_empty() {
            let root = root_queue
                .pop_front()
                .expect("root queue checked non-empty above");
            pending_roots.remove(&root);
            candidate_stats.roots_visited += 1;
            if !structural_hash_state.is_live(root) {
                continue;
            }
            if !matches!(
                structural_hash_state.gate_fn().gates[root.id],
                AigNode::And2 { .. }
            ) {
                continue;
            }
            let Some(root_backward_depth) = finite_backward_depth(&backward_depths, root) else {
                candidate_stats.skipped_unrequired_depth += 1;
                continue;
            };

            let constructed = construct_large_cone_candidate_replacements_for_root(
                structural_hash_state.gate_fn(),
                root,
                &depths,
                &dense_counts,
                &mut candidate_evals,
                &mut candidate_stats,
                &structural_hash_state,
                &mut sop_cover_memo,
            );
            let root_result = cost_large_cone_area_candidates_for_root(
                constructed,
                depths[root.id],
                root_backward_depth,
                cur_output_node_depth,
                opts.mode.allows_area_depth_increase(),
                &mut candidate_stats,
                &structural_hash_state,
            );
            candidates_considered += root_result.candidates_considered;
            area_candidate_count += root_result.candidates.len();
            let mut root_candidates = root_result.candidates;
            root_candidates.sort_by(|a, b| {
                b.area_gain
                    .cmp(&a.area_gain)
                    .then_with(|| a.slack_consumed.cmp(&b.slack_consumed))
                    .then_with(|| a.new_root_depth.cmp(&b.new_root_depth))
                    .then_with(|| b.mffc_nodes.len().cmp(&a.mffc_nodes.len()))
                    .then_with(|| a.replacement.root.id.cmp(&b.replacement.root.id))
                    .then_with(|| a.replacement.leaf_ops.cmp(&b.replacement.leaf_ops))
            });

            for cand in root_candidates {
                debug_assert_eq!(
                    structural_hash_state.live_and_count(),
                    cand.before_live_and_count,
                    "cached exact large-cone area cost should be evaluated against the current graph"
                );
                let candidate_after_live_and_count = cand.after_live_and_count;

                if opts.verify_area_costing {
                    verify_live_area_cost_delta_exact(
                        "large-cone area rewrite",
                        cand.before_live_and_count,
                        candidate_after_live_and_count,
                        cand.area_gain,
                    );
                }
                if cand.replacement.structural_hash_only_area_win {
                    rewrite_stats.selected_structural_hash_only_area_win += 1;
                }
                rewrite_stats.selected_raw_score_ands += cand.replacement.raw_score_ands;
                rewrite_stats.selected_hash_score_ands += cand.replacement.score_ands;
                bump_selected_sop_variant(
                    selected_sop_variant_kind(&cand.replacement),
                    &mut rewrite_stats.selected_sop_flat,
                    &mut rewrite_stats.selected_sop_arrival_balanced,
                    &mut rewrite_stats.selected_sop_factored,
                );

                let fanout_cone_nodes =
                    collect_live_fanout_cone(&structural_hash_state, cand.replacement.root);
                let mut invalidated_nodes = fanout_cone_nodes.clone();
                invalidated_nodes.insert(cand.replacement.root);
                for leaf in &cand.replacement.leaf_ops {
                    invalidated_nodes.insert(leaf.node);
                }
                invalidated_nodes.extend(cand.mffc_nodes.iter().copied());
                let replacement = cand.replacement.clone();
                let materialized =
                    materialize_replacement(&mut structural_hash_state, &replacement)
                        .expect("exact replacement cost should materialize successfully")
                        .expect("exact replacement cost accepted a non-materialized replacement");
                invalidated_nodes.insert(materialized.replacement_op.node);
                assert_eq!(
                    structural_hash_state.live_and_count(),
                    candidate_after_live_and_count,
                    "exact replacement cost did not match materialized live AND count"
                );
                let after_gate_len = structural_hash_state.gate_fn().gates.len();
                let new_nodes_and_fanouts = collect_new_nodes_and_fanouts(
                    &structural_hash_state,
                    materialized.first_new_id,
                    after_gate_len,
                );
                let changed_nodes = collect_replacement_depth_changed_nodes(
                    &structural_hash_state,
                    materialized.first_new_id,
                    invalidated_nodes.clone(),
                );
                depth_state
                    .refresh_from_changed_nodes(&structural_hash_state, &changed_nodes)
                    .expect(
                        "incremental dynamic depth update should succeed for cut-db output graph",
                    );
                let after_output_node_depth = depth_state
                    .max_output_node_depth(&structural_hash_state)
                    .expect("dynamic depth state should compute output depth");
                if !opts.mode.allows_area_depth_increase() {
                    assert!(
                        after_output_node_depth <= cur_output_node_depth,
                        "cut-db large-cone rewrite unexpectedly increased live output depth from {cur_output_node_depth} to {after_output_node_depth}"
                    );
                }
                depths = depth_state.forward_depths().to_vec();
                backward_depths = depth_state.backward_depths().to_vec();
                dense_counts = dense_use_counts_from_dynamic_hash(&structural_hash_state);
                accepted_count += 1;

                enqueue_live_and_roots(
                    &structural_hash_state,
                    fanout_cone_nodes
                        .iter()
                        .copied()
                        .chain(new_nodes_and_fanouts.iter().copied()),
                    &mut pending_roots,
                    &mut root_queue,
                    &mut candidate_stats.roots_total,
                );
                break;
            }
        }
        candidate_stats.viable_candidates = area_candidate_count;
        rewrite_stats.total_candidates = area_candidate_count;
        rewrite_stats.selected = accepted_count;
        let t_phase_ms = t_phase.elapsed().as_millis();

        let rebuild_ms = 0;
        let mut after_live_and_count = live_and_count;
        let mut after_path_len = cur_path_len;
        let mut round_materialized = false;
        if accepted_count != 0 {
            after_live_and_count = structural_hash_state.live_and_count();
            let after_output_node_depth = depth_state
                .max_output_node_depth(&structural_hash_state)
                .expect("dynamic depth state should compute final output depth");
            let new_g = dce(structural_hash_state.gate_fn());
            let dce_after_live_and_count = and_node_count(&new_g);
            after_path_len = output_path_len(&new_g, after_output_node_depth);

            if opts.verify_delay_costing && !opts.mode.allows_area_depth_increase() {
                verify_delay_not_increased(
                    "large-cone area rewrite final",
                    &cur,
                    &new_g,
                    cur_output_node_depth,
                );
            }
            if opts.verify_area_costing {
                verify_area_cost_delta(
                    "large-cone area rewrite final",
                    &cur,
                    &new_g,
                    live_and_count,
                    dce_after_live_and_count,
                );
            }
            if dce_after_live_and_count < live_and_count
                && (opts.mode.allows_area_depth_increase()
                    || after_output_node_depth <= cur_output_node_depth)
                && dce_after_live_and_count == after_live_and_count
            {
                after_live_and_count = dce_after_live_and_count;
                cur = new_g;
                round_materialized = true;
            } else {
                log::info!(
                    "cut-db large-cone refactor: discarded non-improving replacement round; before_ands={} after_ands={} dce_after_ands={} before_depth={} after_depth={} replacements={}",
                    live_and_count,
                    after_live_and_count,
                    dce_after_live_and_count,
                    cur_output_node_depth,
                    after_output_node_depth,
                    accepted_count
                );
                break;
            }
        }

        log::debug!(
            "cut-db large-cone round timings: use_count_ms={} depth_ms={} phase_ms={} rebuild_ms={} area_candidates={} cands_considered={} candidate_evals={} accepted={} before_ands={} after_ands={} before_path_len={} after_path_len={} round_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            t_phase_ms,
            rebuild_ms,
            area_candidate_count,
            candidates_considered,
            candidate_evals,
            accepted_count,
            live_and_count,
            after_live_and_count,
            cur_path_len,
            after_path_len,
            t_iter0.elapsed().as_millis()
        );
        log::debug!(
            "cut-db large-cone rejection stats: roots_total={} roots_visited={} skipped_unrequired_depth={} rejected_no_cone={} rejected_small_cone={} rejected_empty_mffc={} rejected_sop_failed={} rejected_no_area_gain={} rejected_depth={} cones_built={} cone_leaves_sum={} cone_leaves_max={} cone_internal_sum={} cone_internal_max={} cone_mffc_sum={} cone_mffc_max={} sop_variants_sum={} sop_variants_max={} viable_candidates={} viable_structural_hash_only_area_win={} round_materialized={} round_total_candidates={} round_selected={} round_selected_structural_hash_only_area_win={} round_selected_raw_score_ands={} round_selected_hash_score_ands={} round_selected_sop_flat={} round_selected_sop_arrival_balanced={} round_selected_sop_factored={}",
            candidate_stats.roots_total,
            candidate_stats.roots_visited,
            candidate_stats.skipped_unrequired_depth,
            candidate_stats.rejected_no_cone,
            candidate_stats.rejected_small_cone,
            candidate_stats.rejected_empty_mffc,
            candidate_stats.rejected_sop_failed,
            candidate_stats.rejected_no_area_gain,
            candidate_stats.rejected_depth,
            candidate_stats.cones_built,
            candidate_stats.cone_leaves_sum,
            candidate_stats.cone_leaves_max,
            candidate_stats.cone_internal_sum,
            candidate_stats.cone_internal_max,
            candidate_stats.cone_mffc_sum,
            candidate_stats.cone_mffc_max,
            candidate_stats.sop_variants_sum,
            candidate_stats.sop_variants_max,
            candidate_stats.viable_candidates,
            candidate_stats.viable_structural_hash_only_area_win,
            round_materialized,
            rewrite_stats.total_candidates,
            rewrite_stats.selected,
            rewrite_stats.selected_structural_hash_only_area_win,
            rewrite_stats.selected_raw_score_ands,
            rewrite_stats.selected_hash_score_ands,
            rewrite_stats.selected_sop_flat,
            rewrite_stats.selected_sop_arrival_balanced,
            rewrite_stats.selected_sop_factored
        );

        if accepted_count == 0 {
            log::info!("cut-db large-cone refactor: no area-improving candidate found; stopping");
            break;
        }
    }

    cur
}

/// Performs iterative depth and area rewriting with small cuts before large
/// cones.
pub fn rewrite_gatefn_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let cleaned_input = dce(g);
    let depth_rewritten = if opts.mode.enables_depth_rewrite() {
        rewrite_gatefn_depth_with_cut_db(&cleaned_input, db, opts)
    } else {
        cleaned_input
    };
    if !opts.enable_large_cone_rewrite {
        return rewrite_gatefn_area_with_cut_db(&depth_rewritten, db, opts);
    }
    let large_depth_rewritten = if opts.mode.enables_depth_rewrite() {
        rewrite_gatefn_large_cone_depth_refactor(&depth_rewritten, opts)
    } else {
        depth_rewritten
    };
    let area_rewritten = rewrite_gatefn_area_with_cut_db(&large_depth_rewritten, db, opts);
    rewrite_gatefn_large_cone_refactor(&area_rewritten, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::get_summary_stats::get_summary_stats;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::cut_db::fragment::{FIRST_NODE_ID, FragmentNode};
    use crate::cut_db::npn::canon_tt16;
    use crate::cut_db::pareto::ParetoPoint;
    use crate::cut_db::serdes::CanonEntry;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::use_count::get_id_to_use_count;
    use xlsynth::IrBits;

    fn live_and_count_from_use_counts(
        g: &GateFn,
        id_to_use_count: &std::collections::HashMap<AigRef, usize>,
    ) -> usize {
        id_to_use_count
            .keys()
            .filter(|node| matches!(g.gates[node.id], AigNode::And2 { .. }))
            .count()
    }

    fn make_balanced_and4_frag() -> GateFnFragment {
        GateFnFragment {
            nodes: vec![
                FragmentNode::And2 {
                    a: Lit::new(0, false),
                    b: Lit::new(1, false),
                },
                FragmentNode::And2 {
                    a: Lit::new(2, false),
                    b: Lit::new(3, false),
                },
                FragmentNode::And2 {
                    a: Lit::new(FIRST_NODE_ID, false),
                    b: Lit::new(FIRST_NODE_ID + 1, false),
                },
            ],
            output: Lit::new(FIRST_NODE_ID + 2, false),
        }
    }

    fn make_and2_frag() -> GateFnFragment {
        GateFnFragment {
            nodes: vec![FragmentNode::And2 {
                a: Lit::new(0, false),
                b: Lit::new(1, false),
            }],
            output: Lit::new(FIRST_NODE_ID, false),
        }
    }

    fn make_linear_and3_frag() -> GateFnFragment {
        GateFnFragment {
            nodes: vec![
                FragmentNode::And2 {
                    a: Lit::new(0, false),
                    b: Lit::new(1, false),
                },
                FragmentNode::And2 {
                    a: Lit::new(FIRST_NODE_ID, false),
                    b: Lit::new(2, false),
                },
            ],
            output: Lit::new(FIRST_NODE_ID + 1, false),
        }
    }

    fn make_single_entry_db(tt: TruthTable16, frag: GateFnFragment) -> CutDb {
        let (canon_tt, xform) = canon_tt16(tt);
        let canon_frag = frag.apply_npn(xform.inverse());
        assert_eq!(canon_frag.eval_tt16(), canon_tt);

        let canon_entries = vec![
            CanonEntry {
                canon_tt: TruthTable16(0),
                pareto: Vec::new(),
            },
            CanonEntry {
                canon_tt,
                pareto: vec![ParetoPoint {
                    tt: canon_tt,
                    ands: frag.and_count(),
                    depth: frag.depth(),
                    frag: canon_frag,
                }],
            },
        ];
        let mut dense = vec![
            crate::cut_db::loader::DenseInfo {
                canon_index: 0,
                xform: crate::cut_db::npn::NpnTransform::identity(),
            };
            65536
        ];
        dense[tt.0 as usize] = crate::cut_db::loader::DenseInfo {
            canon_index: 1,
            xform,
        };
        CutDb::from_raw_for_test(canon_entries, dense)
    }

    fn eval_single_output_bit(g: &GateFn, inputs: &[bool]) -> bool {
        let inputs: Vec<IrBits> = inputs
            .iter()
            .map(|bit| IrBits::make_ubits(1, if *bit { 1 } else { 0 }).unwrap())
            .collect();
        gate_sim::eval(g, &inputs, Collect::None).outputs[0]
            .get_bit(0)
            .unwrap()
    }

    fn add_linear_and(gb: &mut GateBuilder, args: &[AigOperand]) -> AigOperand {
        let mut out = args[0];
        for arg in &args[1..] {
            out = gb.add_and_binary(out, *arg);
        }
        out
    }

    #[test]
    fn test_cut_db_rewrite_reduces_unbalanced_and4_depth() {
        // Function: a & b & c & d. Truth table is 1 only at assignment 0b1111 => bit
        // 15.
        let and4_tt = TruthTable16(0x8000);
        let frag = make_balanced_and4_frag();
        assert_eq!(frag.eval_tt16(), and4_tt);

        let (canon_tt, xform) = canon_tt16(and4_tt);
        let canon_frag = frag.apply_npn(xform.inverse());
        assert_eq!(canon_frag.eval_tt16(), canon_tt);

        let entry = CanonEntry {
            canon_tt,
            pareto: vec![ParetoPoint {
                tt: canon_tt,
                ands: 3,
                depth: 2,
                frag: canon_frag,
            }],
        };
        let canon_entries = vec![
            CanonEntry {
                canon_tt: TruthTable16(0),
                pareto: Vec::new(),
            },
            entry,
        ];
        let mut dense = vec![
            crate::cut_db::loader::DenseInfo {
                canon_index: 0,
                xform: crate::cut_db::npn::NpnTransform::identity(),
            };
            65536
        ];
        dense[and4_tt.0 as usize] = crate::cut_db::loader::DenseInfo {
            canon_index: 1,
            xform,
        };
        let db = CutDb::from_raw_for_test(canon_entries, dense);

        // Build a deep (unbalanced) AND4 to give the rewriter something to improve.
        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);
        let ab = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let abc = gb.add_and_binary(ab, *c.get_lsb(0));
        let abcd = gb.add_and_binary(abc, *d.get_lsb(0));
        gb.add_output("o".to_string(), crate::aig::AigBitVector::from_bit(abcd));
        let g = gb.build();

        let hash_state = DynamicStructuralHash::new(g.clone()).unwrap();
        let depth_state = DynamicDepthState::new(&hash_state).unwrap();
        let output_node_depth = depth_state
            .max_output_node_depth(&hash_state)
            .expect("dynamic depth state should compute output depth");
        assert_eq!(
            output_path_len(&g, output_node_depth),
            output_node_depth + 1
        );
        let critical_roots = collect_critical_roots(&hash_state, &depth_state, output_node_depth);
        assert!(
            critical_roots.contains(&abcd.node),
            "output AND should be on the critical path roots"
        );

        let before = get_summary_stats(&g);
        let rewritten = rewrite_gatefn_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
                verify_area_costing: true,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );
        let after = get_summary_stats(&rewritten);

        assert!(
            after.deepest_path < before.deepest_path,
            "global path length should decrease: before={} after={}",
            before.deepest_path,
            after.deepest_path
        );
    }

    #[test]
    fn test_cut_db_rewrite_preserves_non_empty_provenance() {
        let and4_tt = TruthTable16(0x8000);
        let frag = make_balanced_and4_frag();
        let (canon_tt, xform) = canon_tt16(and4_tt);
        let canon_frag = frag.apply_npn(xform.inverse());
        let entry = CanonEntry {
            canon_tt,
            pareto: vec![ParetoPoint {
                tt: canon_tt,
                ands: 3,
                depth: 2,
                frag: canon_frag,
            }],
        };
        let canon_entries = vec![
            CanonEntry {
                canon_tt: TruthTable16(0),
                pareto: Vec::new(),
            },
            entry,
        ];
        let mut dense = vec![
            crate::cut_db::loader::DenseInfo {
                canon_index: 0,
                xform: crate::cut_db::npn::NpnTransform::identity(),
            };
            65536
        ];
        dense[and4_tt.0 as usize] = crate::cut_db::loader::DenseInfo {
            canon_index: 1,
            xform,
        };
        let db = CutDb::from_raw_for_test(canon_entries, dense);

        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);
        gb.add_pir_node_id(a.get_lsb(0).node, 1);
        gb.add_pir_node_id(b.get_lsb(0).node, 2);
        gb.add_pir_node_id(c.get_lsb(0).node, 3);
        gb.add_pir_node_id(d.get_lsb(0).node, 4);
        gb.set_current_pir_node_id(Some(10));
        let ab = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        gb.set_current_pir_node_id(Some(11));
        let abc = gb.add_and_binary(ab, *c.get_lsb(0));
        gb.set_current_pir_node_id(Some(12));
        let abcd = gb.add_and_binary(abc, *d.get_lsb(0));
        gb.set_current_pir_node_id(None);
        gb.add_output("o".to_string(), crate::aig::AigBitVector::from_bit(abcd));
        let g = gb.build();

        let rewritten = rewrite_gatefn_area_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
                verify_area_costing: true,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );

        for node in &rewritten.gates {
            if let AigNode::And2 { .. } = node {
                assert!(
                    !node.get_pir_node_ids().is_empty(),
                    "rewritten AND node should retain non-empty provenance: {:?}",
                    node
                );
            }
        }
    }

    #[test]
    fn test_cut_db_area_rewrite_reduces_noncritical_mffc_area() {
        let and2_tt = TruthTable16::var(0).and(TruthTable16::var(1));
        let db = make_single_entry_db(and2_tt, make_and2_frag());

        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);
        let e = gb.add_input("e".to_string(), 1);
        let f = gb.add_input("f".to_string(), 1);

        // Non-critical redundant cone: (a & b) & (a & b), intentionally built
        // without structural hashing so the area phase has an MFFC to recover.
        let ab0 = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let ab1 = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let redundant_ab = gb.add_and_binary(ab0, ab1);
        gb.add_output(
            "area".to_string(),
            crate::aig::AigBitVector::from_bit(redundant_ab),
        );

        // Separate deeper cone keeps the global critical depth unchanged.
        let cd = gb.add_and_binary(*c.get_lsb(0), *d.get_lsb(0));
        let cde = gb.add_and_binary(cd, *e.get_lsb(0));
        let cdef = gb.add_and_binary(cde, *f.get_lsb(0));
        gb.add_output(
            "critical".to_string(),
            crate::aig::AigBitVector::from_bit(cdef),
        );

        let g = gb.build();
        let before_counts = get_id_to_use_count(&g);
        let before_area = live_and_count_from_use_counts(&g, &before_counts);
        let before_stats = get_summary_stats(&g);

        let rewritten = rewrite_gatefn_area_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
                verify_area_costing: true,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );

        let after_counts = get_id_to_use_count(&rewritten);
        let after_area = live_and_count_from_use_counts(&rewritten, &after_counts);
        let after_stats = get_summary_stats(&rewritten);

        assert!(
            after_area < before_area,
            "area phase should reduce live ANDs: before={} after={}",
            before_area,
            after_area
        );
        assert_eq!(
            after_stats.deepest_path, before_stats.deepest_path,
            "area phase should preserve the current global depth"
        );
    }

    #[test]
    fn test_area_rewrite_counts_structural_hash_reuse() {
        let and3_tt = TruthTable16::var(0)
            .and(TruthTable16::var(1))
            .and(TruthTable16::var(2));
        let db = make_single_entry_db(and3_tt, make_linear_and3_frag());

        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let a = *a.get_lsb(0);
        let b = *b.get_lsb(0);
        let c = *c.get_lsb(0);

        // Keep one copy of a&b live outside the candidate MFFC. Rewriting the
        // duplicate (a&b)&c should credit that existing node instead of judging
        // only by the raw two-AND fragment size.
        let shared_ab = gb.add_and_binary(a, b);
        let duplicate_ab = gb.add_and_binary(a, b);
        let duplicate_and3 = gb.add_and_binary(duplicate_ab, c);
        gb.add_output(
            "and3".to_string(),
            crate::aig::AigBitVector::from_bit(duplicate_and3),
        );
        gb.add_output(
            "shared".to_string(),
            crate::aig::AigBitVector::from_bit(shared_ab),
        );
        let g = gb.build();

        let before_counts = get_id_to_use_count(&g);
        let before_area = live_and_count_from_use_counts(&g, &before_counts);
        let before_stats = get_summary_stats(&g);

        let rewritten = rewrite_gatefn_area_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
                verify_area_costing: true,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );

        let after_counts = get_id_to_use_count(&rewritten);
        let after_area = live_and_count_from_use_counts(&rewritten, &after_counts);
        let after_stats = get_summary_stats(&rewritten);

        assert!(
            after_area < before_area,
            "area phase should accept structural-hash area gains: before={} after={}",
            before_area,
            after_area
        );
        assert_eq!(
            after_stats.deepest_path, before_stats.deepest_path,
            "structural-hash area rewrite should preserve global depth"
        );
    }

    #[test]
    fn test_large_cone_refactor_reduces_duplicate_and5_mffc() {
        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);
        let e = gb.add_input("e".to_string(), 1);
        let inputs = [
            *a.get_lsb(0),
            *b.get_lsb(0),
            *c.get_lsb(0),
            *d.get_lsb(0),
            *e.get_lsb(0),
        ];

        let and5_0 = add_linear_and(&mut gb, &inputs);
        let and5_1 = add_linear_and(&mut gb, &inputs);
        let redundant = gb.add_and_binary(and5_0, and5_1);
        gb.add_output(
            "o".to_string(),
            crate::aig::AigBitVector::from_bit(redundant),
        );
        let g = gb.build();

        let before_counts = get_id_to_use_count(&g);
        let before_area = live_and_count_from_use_counts(&g, &before_counts);
        let before_stats = get_summary_stats(&g);

        let rewritten = rewrite_gatefn_large_cone_refactor(
            &g,
            RewriteOptions {
                max_iterations: 8,
                verify_area_costing: true,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );

        let after_counts = get_id_to_use_count(&rewritten);
        let after_area = live_and_count_from_use_counts(&rewritten, &after_counts);
        let after_stats = get_summary_stats(&rewritten);

        assert!(
            after_area < before_area,
            "large-cone refactor should reduce live ANDs: before={} after={}",
            before_area,
            after_area
        );
        assert!(
            after_stats.deepest_path <= before_stats.deepest_path,
            "large-cone refactor should not increase depth: before={} after={}",
            before_stats.deepest_path,
            after_stats.deepest_path
        );

        for assignment in 0..32 {
            let sample: Vec<bool> = (0..5).map(|i| ((assignment >> i) & 1) != 0).collect();
            assert_eq!(
                eval_single_output_bit(&g, &sample),
                eval_single_output_bit(&rewritten, &sample),
                "assignment={:05b}",
                assignment
            );
        }
    }

    #[test]
    fn test_large_cone_depth_refactor_reduces_linear_and5_depth() {
        let mut gb = GateBuilder::new("t".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);
        let e = gb.add_input("e".to_string(), 1);
        let inputs = [
            *a.get_lsb(0),
            *b.get_lsb(0),
            *c.get_lsb(0),
            *d.get_lsb(0),
            *e.get_lsb(0),
        ];

        let and5 = add_linear_and(&mut gb, &inputs);
        gb.add_output("o".to_string(), crate::aig::AigBitVector::from_bit(and5));
        let g = gb.build();

        let before_stats = get_summary_stats(&g);
        let rewritten = rewrite_gatefn_large_cone_depth_refactor(
            &g,
            RewriteOptions {
                max_iterations: 8,
                verify_delay_costing: true,
                ..RewriteOptions::default()
            },
        );
        let after_stats = get_summary_stats(&rewritten);

        assert!(
            after_stats.deepest_path < before_stats.deepest_path,
            "large-cone depth refactor should reduce global depth: before={} after={}",
            before_stats.deepest_path,
            after_stats.deepest_path
        );

        for assignment in 0..32 {
            let sample: Vec<bool> = (0..5).map(|i| ((assignment >> i) & 1) != 0).collect();
            assert_eq!(
                eval_single_output_bit(&g, &sample),
                eval_single_output_bit(&rewritten, &sample),
                "assignment={:05b}",
                assignment
            );
        }
    }
}
