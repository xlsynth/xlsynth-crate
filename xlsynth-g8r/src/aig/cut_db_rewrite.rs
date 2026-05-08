// SPDX-License-Identifier: Apache-2.0

//! Cut-db–driven AIG rewrite pass.
//!
//! Current goal: reduce global output depth on critical paths, then recover
//! area elsewhere under the achieved depth bound, using the precomputed 4-input
//! cut database.
//!
//! The depth phase accepts only rewrites that reduce the global depth. The area
//! phase is MFFC-aware: it only credits nodes that become dead under the chosen
//! cut boundary, and it accepts rewrites only when live AND count decreases
//! without increasing the global output depth.

use std::collections::{BTreeSet, VecDeque};
use std::time::Instant;

use smallvec::SmallVec;

use crate::aig::cut_replacement_cost::{
    gate_count_diff_for_replacement, materialize_replacement, output_depth_after_replacement,
};
use crate::aig::dce::dce;
use crate::aig::dynamic_depth::DynamicDepthState;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn, PirNodeIds};
use crate::aig::get_summary_stats::get_aig_stats;
use crate::cut_db::fragment::{GateFnFragment, Lit};
use crate::cut_db::loader::CutDb;
use crate::cut_db::tt16::TruthTable16;

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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Cut {
    leaves: Vec<AigOperand>, // sorted
    tt: TruthTable16,
}

fn push_unique_cut(cuts: &mut Vec<Cut>, cut: Cut) -> bool {
    if cuts.iter().any(|existing| existing == &cut) {
        false
    } else {
        cuts.push(cut);
        true
    }
}

fn union_cut_leaves(lhs: &[AigOperand], rhs: &[AigOperand]) -> Option<Vec<AigOperand>> {
    let mut leaves: SmallVec<[AigOperand; 8]> = SmallVec::new();
    leaves.extend(lhs.iter().copied());
    leaves.extend(rhs.iter().copied());
    leaves.sort();
    leaves.dedup();
    if leaves.len() > 4 {
        None
    } else {
        Some(leaves.into_iter().collect())
    }
}

fn sort_and_prune_cuts(mut cuts: Vec<Cut>, max_cuts_per_node: usize) -> Vec<Cut> {
    cuts.sort();
    cuts.dedup();
    if max_cuts_per_node != 0 && cuts.len() > max_cuts_per_node {
        cuts.truncate(max_cuts_per_node);
    }
    cuts
}

fn negate_tt(tt: TruthTable16) -> TruthTable16 {
    tt.not()
}

/// Remaps `tt` from `old_leaves` order into the larger `union_leaves` order.
fn embed_tt_into_union(
    tt: TruthTable16,
    old_leaves: &[AigOperand],
    union_leaves: &[AigOperand],
) -> TruthTable16 {
    if old_leaves.is_empty() {
        return tt;
    }
    let mut map: [usize; 4] = [0; 4];
    for (i, leaf) in old_leaves.iter().enumerate() {
        let j = union_leaves
            .binary_search(leaf)
            .expect("old leaf must appear in union");
        map[i] = j;
    }
    let old_len = old_leaves.len();
    let mut out = TruthTable16::const0();
    for assign in 0u8..16 {
        let mut old_assign: u8 = 0;
        for i in 0..old_len {
            let bit = ((assign >> map[i]) & 1) != 0;
            old_assign |= (bit as u8) << i;
        }
        let bit = tt.get_bit(old_assign);
        out.set_bit(assign, bit);
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
struct CutEnumerationStats {
    computed_nodes: usize,
    total_cuts: usize,
    truncated_nodes: usize,
    elapsed_ms: u128,
}

impl CutEnumerationStats {
    fn add(&mut self, other: CutEnumerationStats) {
        self.computed_nodes += other.computed_nodes;
        self.total_cuts += other.total_cuts;
        self.truncated_nodes += other.truncated_nodes;
        self.elapsed_ms += other.elapsed_ms;
    }
}

/// Enumerates and prunes 4-feasible cuts for one immutable graph.
///
/// The memo is shared across roots in a rewrite sweep. Callers invalidate the
/// rewritten fanout cone after each committed replacement so unrelated
/// descendant cut sets remain cached.
struct CutEnumerator {
    max_cuts_per_node: usize,
    memo: Vec<Option<Vec<Cut>>>,
    stats: CutEnumerationStats,
}

impl CutEnumerator {
    fn new(g: &GateFn, max_cuts_per_node: usize) -> Self {
        Self {
            max_cuts_per_node,
            memo: vec![None; g.gates.len()],
            stats: CutEnumerationStats::default(),
        }
    }

    fn stats(&self) -> CutEnumerationStats {
        self.stats
    }

    fn sync_len(&mut self, g: &GateFn) {
        if self.memo.len() < g.gates.len() {
            self.memo.resize(g.gates.len(), None);
        }
    }

    fn invalidate_node(&mut self, node: AigRef) {
        if let Some(entry) = self.memo.get_mut(node.id) {
            *entry = None;
        }
    }

    fn invalidate_nodes(&mut self, nodes: &BTreeSet<AigRef>) {
        for node in nodes {
            self.invalidate_node(*node);
        }
    }

    fn cuts_for_root(&mut self, g: &GateFn, root: AigRef) -> Vec<Cut> {
        let t0 = Instant::now();
        let cuts = self.cuts_for_node(g, root);
        self.stats.elapsed_ms += t0.elapsed().as_millis();
        cuts
    }

    fn cuts_for_operand(&mut self, g: &GateFn, op: AigOperand) -> Vec<Cut> {
        let mut cuts = self.cuts_for_node(g, op.node);
        if op.negated {
            for c in &mut cuts {
                c.tt = negate_tt(c.tt);
            }
        }
        cuts
    }

    fn cuts_for_node(&mut self, g: &GateFn, r: AigRef) -> Vec<Cut> {
        if let Some(cuts) = &self.memo[r.id] {
            return cuts.clone();
        }

        let mut cuts: Vec<Cut> = Vec::new();

        // Trivial self-cut: allow this node to be used as a leaf for its fanout.
        push_unique_cut(
            &mut cuts,
            Cut {
                leaves: vec![AigOperand {
                    node: r,
                    negated: false,
                }],
                tt: TruthTable16::var(0),
            },
        );

        let node_kind = match &g.gates[r.id] {
            AigNode::Input { .. } => None,
            AigNode::Literal { value: v, .. } => {
                // Constant cut with no leaves.
                push_unique_cut(
                    &mut cuts,
                    Cut {
                        leaves: Vec::new(),
                        tt: if *v {
                            TruthTable16::const1()
                        } else {
                            TruthTable16::const0()
                        },
                    },
                );
                None
            }
            AigNode::And2 { a, b, .. } => Some((*a, *b)),
        };

        if let Some((a, b)) = node_kind {
            let a_cuts = self.cuts_for_operand(g, a);
            let b_cuts = self.cuts_for_operand(g, b);
            'pairs: for ca in &a_cuts {
                for cb in &b_cuts {
                    let Some(union_leaves) = union_cut_leaves(&ca.leaves, &cb.leaves) else {
                        continue;
                    };
                    let ca_tt = embed_tt_into_union(ca.tt, &ca.leaves, &union_leaves);
                    let cb_tt = embed_tt_into_union(cb.tt, &cb.leaves, &union_leaves);
                    let tt = ca_tt.and(cb_tt);
                    push_unique_cut(
                        &mut cuts,
                        Cut {
                            leaves: union_leaves,
                            tt,
                        },
                    );
                    if self.max_cuts_per_node != 0 && cuts.len() >= self.max_cuts_per_node {
                        self.stats.truncated_nodes += 1;
                        break 'pairs;
                    }
                }
            }
        }

        let v = sort_and_prune_cuts(cuts, self.max_cuts_per_node);
        self.stats.computed_nodes += 1;
        self.stats.total_cuts += v.len();
        self.memo[r.id] = Some(v.clone());
        v
    }
}

#[derive(Debug, Clone)]
pub(super) enum ReplacementImpl {
    Fragment {
        frag: GateFnFragment,
        input_depths: [u16; 4],
    },
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

fn verify_expected_delay_reduction(
    context: &str,
    before_g: &GateFn,
    after_g: &GateFn,
    expected_before_depth: usize,
    expected_after_depth: usize,
) {
    let checked_before_depth = independent_output_node_depth(before_g);
    let checked_after_depth = independent_output_node_depth(after_g);
    assert_eq!(
        checked_before_depth, expected_before_depth,
        "cut-db delay verification failed in {context}: expected before depth {expected_before_depth}, DCE-cleaned graph before depth {checked_before_depth}"
    );
    assert_eq!(
        checked_after_depth, expected_after_depth,
        "cut-db delay verification failed in {context}: expected after depth {expected_after_depth}, DCE-cleaned graph after depth {checked_after_depth}"
    );
    let expected_delta = expected_before_depth.saturating_sub(expected_after_depth);
    let checked_delta = checked_before_depth.saturating_sub(checked_after_depth);
    assert_eq!(
        checked_delta, expected_delta,
        "cut-db delay verification failed in {context}: expected depth delta {expected_delta}, DCE-cleaned graph delta {checked_delta}"
    );
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

fn dynamic_false() -> AigOperand {
    AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    }
}

fn instantiate_fragment_dynamic(
    state: &mut DynamicStructuralHash,
    frag: &GateFnFragment,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
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
                state.add_and_with_pir_node_ids(a_op, b_op, pir_node_ids)?
            }
        };
        ops.push(op);
    }

    let output = op_from_lit(frag.output, &ops);
    state.add_pir_node_ids(output.node, pir_node_ids)?;
    Ok(output)
}

fn instantiate_replacement_dynamic(
    state: &mut DynamicStructuralHash,
    implementation: &ReplacementImpl,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
) -> Result<AigOperand, String> {
    match implementation {
        ReplacementImpl::Fragment { frag, .. } => {
            instantiate_fragment_dynamic(state, frag, leaf_ops, pir_node_ids)
        }
    }
}

/// Picks a small, deterministically ordered set of candidate replacements for
/// `root` using `db` and the current depth map.
fn choose_candidate_replacements_for_root(
    root: AigRef,
    root_cuts: &[Cut],
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    db: &CutDb,
) -> Vec<Replacement> {
    let mut cands: Vec<Replacement> = Vec::new();

    for cut in root_cuts {
        // Skip trivial self-cut.
        if cut.leaves.len() == 1 && cut.leaves[0].node == root && !cut.leaves[0].negated {
            continue;
        }
        if cut.leaves.len() > 4 {
            continue;
        }

        let (xform, pareto) = db.lookup(cut.tt.0);
        for p in pareto {
            let frag = p.frag.apply_npn(xform);
            let input_depths = frag.input_depths();
            let mut new_depth_at_root: usize = 0;
            for (i, leaf) in cut.leaves.iter().enumerate() {
                let leaf_depth = live_forward_depth(depth_state, structural_hash_state, leaf.node);
                let cand = leaf_depth + (input_depths[i] as usize);
                new_depth_at_root = core::cmp::max(new_depth_at_root, cand);
            }

            let score_depth = new_depth_at_root;
            let score_ands = p.ands;

            cands.push(Replacement {
                root,
                leaf_ops: cut.leaves.clone(),
                implementation: ReplacementImpl::Fragment { frag, input_depths },
                score_depth,
                score_ands: score_ands as usize,
                raw_score_ands: score_ands as usize,
                structural_hash_only_area_win: false,
            });
        }
    }

    // Deterministic ordering + cap for performance. `sort_by` is stable, so
    // equal-score candidates keep the deterministic generation order.
    cands.sort_by(|a, b| {
        (a.score_depth, a.score_ands, a.root.id, &a.leaf_ops).cmp(&(
            b.score_depth,
            b.score_ands,
            b.root.id,
            &b.leaf_ops,
        ))
    });
    cands.truncate(16);
    cands
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
        if cut.leaves.len() == 1 && cut.leaves[0].node == root && !cut.leaves[0].negated {
            stats.skipped_identity_cut += 1;
            continue;
        }
        if cut.leaves.len() > 4 {
            stats.skipped_large_cut += 1;
            continue;
        }

        let mffc_nodes = collect_mffc_nodes_under_cut(structural_hash_state, root, &cut.leaves);
        if mffc_nodes.is_empty() {
            stats.rejected_empty_mffc += 1;
            continue;
        }
        let internal_and_nodes = collect_internal_and_nodes_under_cut(g, root, &cut.leaves);
        let cut_is_mffc = internal_and_nodes == mffc_nodes;

        let (xform, pareto) = db.lookup(cut.tt.0);
        candidates_considered += pareto.len();
        for p in pareto {
            *candidate_evals += 1;

            let frag = p.frag.apply_npn(xform);
            let input_depths = frag.input_depths();
            let implementation = ReplacementImpl::Fragment { frag, input_depths };
            let new_root_depth = replacement_depth_from_inputs(
                &cut.leaves,
                &implementation,
                structural_hash_state,
                depth_state,
            );
            if !replacement_preserves_output_depth(
                new_root_depth,
                root_backward_depth,
                output_depth_cap,
            ) {
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
                leaf_ops: cut.leaves.clone(),
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
    let new_op = instantiate_replacement_dynamic(
        state,
        &repl.implementation,
        &repl.leaf_ops,
        pir_node_ids.as_slice(),
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

fn rewrite_gatefn_depth_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();

    // In each global recompute round, walk critical-path roots from the current
    // graph. Accepted replacements are materialized immediately into the
    // dynamic graph state, and newly critical roots are queued before the round
    // continues.
    let mut iter: usize = 0;
    loop {
        if opts.max_iterations != 0 && iter >= opts.max_iterations {
            log::info!(
                "cut-db depth rewrite: reached max_iterations={}; stopping",
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

        let mut critical_roots: BTreeSet<AigRef> =
            collect_critical_roots(&structural_hash_state, &depth_state, cur_output_node_depth)
                .into_iter()
                .collect();
        let initial_critical_roots = critical_roots.len();
        let mut root_queue: VecDeque<AigRef> = critical_roots.iter().copied().collect();
        let mut pending_roots = critical_roots.clone();

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
            if !critical_roots.contains(&root) || !structural_hash_state.is_live(root) {
                continue;
            }
            if !matches!(
                structural_hash_state.gate_fn().gates[root.id],
                AigNode::And2 { .. }
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

                let trial_output_node_depth = match output_depth_after_replacement(
                    &structural_hash_state,
                    &depth_state,
                    &cand,
                ) {
                    Ok(Some(depth)) => depth,
                    Ok(None) => continue,
                    Err(e) => {
                        log::debug!(
                            "cut-db depth exact virtual trial rejected candidate root={:?}: {}",
                            cand.root,
                            e
                        );
                        continue;
                    }
                };
                if trial_output_node_depth >= cur_output_node_depth {
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
                depth_state
                    .refresh_from_changed_nodes(&structural_hash_state, &changed_nodes)
                    .expect(
                        "incremental dynamic depth update should succeed for cut-db output graph",
                    );
                let after_output_node_depth = depth_state
                    .max_output_node_depth(&structural_hash_state)
                    .expect("dynamic depth state should compute trial output depth");
                assert_eq!(
                    after_output_node_depth, trial_output_node_depth,
                    "cut-db depth virtual output depth did not match materialized output depth"
                );

                if opts.verify_delay_costing {
                    verify_expected_delay_reduction(
                        "depth rewrite",
                        before_g_for_verify.as_ref().unwrap(),
                        structural_hash_state.gate_fn(),
                        cur_output_node_depth,
                        after_output_node_depth,
                    );
                }

                cur_output_node_depth = after_output_node_depth;
                after_path_len =
                    output_path_len(structural_hash_state.gate_fn(), cur_output_node_depth);
                accepted_count += 1;

                cut_enumerator.sync_len(structural_hash_state.gate_fn());
                cut_enumerator.invalidate_nodes(&invalidated_cut_nodes);

                critical_roots = collect_critical_roots(
                    &structural_hash_state,
                    &depth_state,
                    cur_output_node_depth,
                )
                .into_iter()
                .collect();
                for critical_root in critical_roots.iter().copied() {
                    if pending_roots.insert(critical_root) {
                        root_queue.push_back(critical_root);
                    }
                }
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
            break;
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

                let mut invalidated_cut_nodes =
                    collect_live_fanout_cone(&structural_hash_state, cand.replacement.root);
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
                assert!(
                    after_output_node_depth <= cur_output_node_depth,
                    "cut-db area rewrite unexpectedly increased live output depth from {cur_output_node_depth} to {after_output_node_depth}"
                );
                cut_enumerator.sync_len(structural_hash_state.gate_fn());
                cut_enumerator.invalidate_nodes(&invalidated_cut_nodes);
                accepted_count += 1;

                for id in first_new_id..after_gate_len {
                    let node = AigRef { id };
                    if structural_hash_state.is_live(node)
                        && matches!(
                            structural_hash_state.gate_fn().gates[id],
                            AigNode::And2 { .. }
                        )
                        && pending_roots.insert(node)
                    {
                        candidate_stats.roots_total += 1;
                        root_queue.push_back(node);
                    }
                }
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

            if opts.verify_delay_costing {
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
                && after_output_node_depth <= cur_output_node_depth
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

/// Performs iterative depth and area rewriting with 4-input cuts.
pub fn rewrite_gatefn_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let cleaned_input = dce(g);
    let depth_rewritten = rewrite_gatefn_depth_with_cut_db(&cleaned_input, db, opts);
    rewrite_gatefn_area_with_cut_db(&depth_rewritten, db, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::get_summary_stats::get_summary_stats;
    use crate::cut_db::fragment::{FIRST_NODE_ID, FragmentNode};
    use crate::cut_db::npn::canon_tt16;
    use crate::cut_db::pareto::ParetoPoint;
    use crate::cut_db::serdes::CanonEntry;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::use_count::get_id_to_use_count;

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
}
