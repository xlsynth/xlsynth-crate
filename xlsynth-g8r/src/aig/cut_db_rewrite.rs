// SPDX-License-Identifier: Apache-2.0

//! Cut-db–driven AIG rewrite pass (4-leaf cuts).
//!
//! Current goal: aggressively reduce global output depth using the precomputed
//! 4-input cut database.
//!
//! We still use AND-count as a *secondary tie-breaker for candidate ordering*,
//! but the acceptance criterion in `rewrite_gatefn_with_cut_db` is currently
//! depth-only (we accept only rewrites that reduce the global depth). Area/AND
//! recovery is intentionally left to other passes.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use blake3::Hasher;

use crate::aig::dce::dce;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn, PirNodeIds};
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::topo::topo_sort_refs;
use crate::cut_db::fragment::{GateFnFragment, Lit};
use crate::cut_db::loader::CutDb;
use crate::cut_db::tt16::TruthTable16;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::use_count::get_id_to_use_count;

#[derive(Debug, Clone, Copy)]
pub struct RewriteOptions {
    /// Maximum number of cuts retained at each node during enumeration.
    ///
    /// This bounds local search breadth. If set to `0`, no per-node cut limit
    /// is applied, which can be expensive on large graphs.
    pub max_cuts_per_node: usize,
    /// Maximum number of outer rewrite iterations to run.
    ///
    /// An iteration is one global recompute round: compute use counts/depths,
    /// enumerate cuts, accept a batch of virtual depth-improving replacements,
    /// then materialize the batch with one rebuild/DCE. If set to `0`, this
    /// outer loop is unbounded and stops only when no improving rewrite is
    /// found or another configured cap prevents further progress.
    pub max_iterations: usize,
    /// Maximum cheap candidate depth evaluations per global recompute round.
    ///
    /// If set to `0`, candidate evaluation is unbounded.
    pub max_candidate_evals_per_round: usize,
    /// Maximum accepted replacements per global recompute round.
    ///
    /// Accepted replacements are validated in the virtual depth model, then
    /// materialized together. Larger batches amortize rebuild/DCE work; smaller
    /// batches recompute global state more often and can take a different QoR
    /// trajectory. If set to `0`, accepted replacements are unbounded.
    pub max_rewrites_per_round: usize,
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
            max_candidate_evals_per_round: 4096,
            max_rewrites_per_round: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Cut {
    leaves: Vec<AigOperand>, // sorted
    tt: TruthTable16,
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

/// Returns the precomputed cuts for `op`, flipping truth tables if `op` is
/// negated.
fn cuts_for_operand(cuts_by_node: &[Vec<Cut>], op: AigOperand) -> Vec<Cut> {
    let mut v = cuts_by_node[op.node.id].clone();
    if op.negated {
        for c in &mut v {
            c.tt = negate_tt(c.tt);
        }
    }
    v
}

/// Enumerates and prunes all 4-feasible cuts (with truth tables) for every node
/// in `g`.
fn compute_cuts(g: &GateFn, max_cuts_per_node: usize) -> Vec<Vec<Cut>> {
    let t0 = Instant::now();
    let topo = topo_sort_refs(&g.gates);
    let mut cuts_by_node: Vec<Vec<Cut>> = vec![Vec::new(); g.gates.len()];
    let mut total_cuts: usize = 0;
    let mut truncated_nodes: usize = 0;

    for r in topo {
        let node = &g.gates[r.id];
        // Use an ordered set so we dedup as we go and can early-stop
        // deterministically. This avoids building huge intermediate vectors for
        // nodes with many possible cut combinations.
        let mut cuts: BTreeSet<Cut> = BTreeSet::new();

        // Trivial self-cut: allow this node to be used as a leaf for its fanout.
        cuts.insert(Cut {
            leaves: vec![AigOperand {
                node: r,
                negated: false,
            }],
            tt: TruthTable16::var(0),
        });

        match node {
            AigNode::Input { .. } => {
                // Inputs already covered by self-cut.
            }
            AigNode::Literal { value: v, .. } => {
                // Constant cut with no leaves.
                cuts.insert(Cut {
                    leaves: Vec::new(),
                    tt: if *v {
                        TruthTable16::const1()
                    } else {
                        TruthTable16::const0()
                    },
                });
            }
            AigNode::And2 { a, b, .. } => {
                let a_cuts = cuts_for_operand(&cuts_by_node, *a);
                let b_cuts = cuts_for_operand(&cuts_by_node, *b);
                'pairs: for ca in &a_cuts {
                    for cb in &b_cuts {
                        let mut set: BTreeSet<AigOperand> = BTreeSet::new();
                        set.extend(ca.leaves.iter().copied());
                        set.extend(cb.leaves.iter().copied());
                        if set.len() > 4 {
                            continue;
                        }
                        let union_leaves: Vec<AigOperand> = set.into_iter().collect();
                        let ca_tt = embed_tt_into_union(ca.tt, &ca.leaves, &union_leaves);
                        let cb_tt = embed_tt_into_union(cb.tt, &cb.leaves, &union_leaves);
                        let tt = ca_tt.and(cb_tt);
                        cuts.insert(Cut {
                            leaves: union_leaves,
                            tt,
                        });
                        if max_cuts_per_node != 0 && cuts.len() >= max_cuts_per_node {
                            truncated_nodes += 1;
                            break 'pairs;
                        }
                    }
                }
            }
        }

        // Deterministic pruning: already sorted/deduped by BTreeSet.
        let v: Vec<Cut> = if max_cuts_per_node == 0 {
            cuts.into_iter().collect()
        } else {
            cuts.into_iter().take(max_cuts_per_node).collect()
        };
        total_cuts += v.len();
        cuts_by_node[r.id] = v;
    }

    log::debug!(
        "compute_cuts: nodes={} max_cuts_per_node={} total_cuts={} truncated_nodes={} elapsed_ms={}",
        g.gates.len(),
        max_cuts_per_node,
        total_cuts,
        truncated_nodes,
        t0.elapsed().as_millis()
    );
    cuts_by_node
}

/// Instantiates a `GateFnFragment` into `gb`, wiring its leaves from `leaf_ops`
/// (missing leaves default to 0).
fn instantiate_fragment(
    gb: &mut GateBuilder,
    frag: &GateFnFragment,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
) -> AigOperand {
    let mut ops: Vec<AigOperand> = Vec::with_capacity(5 + frag.nodes.len());
    for i in 0..4usize {
        if i < leaf_ops.len() {
            ops.push(leaf_ops[i]);
        } else {
            ops.push(gb.get_false());
        }
    }
    ops.push(gb.get_false()); // const0

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
                let op = gb.add_and_binary(a_op, b_op);
                gb.add_pir_node_ids(op.node, pir_node_ids);
                op
            }
        };
        ops.push(op);
    }

    let output = op_from_lit(frag.output, &ops);
    gb.add_pir_node_ids(output.node, pir_node_ids);
    output
}

#[derive(Debug, Clone)]
struct Replacement {
    root: AigRef,
    leaf_ops: Vec<AigOperand>,
    frag: GateFnFragment,
    frag_key: [u8; 32],
    score_depth: usize,
    score_ands: u16,
}

fn stable_fragment_key(frag: &GateFnFragment) -> [u8; 32] {
    // Deterministic, toolchain-stable tie-break key (BLAKE3 over a fixed byte
    // encoding). This is only used for ordering; it is not security-sensitive.
    let mut hasher = Hasher::new();
    hasher.update(b"GateFnFragment");
    hasher.update(&(frag.nodes.len() as u64).to_le_bytes());
    for n in &frag.nodes {
        match *n {
            crate::cut_db::fragment::FragmentNode::And2 { a, b } => {
                hasher.update(&[0]); // tag: And2
                hasher.update(&lit_to_bytes(a));
                hasher.update(&lit_to_bytes(b));
            }
        }
    }
    hasher.update(&[1]); // tag: output
    hasher.update(&lit_to_bytes(frag.output));
    *hasher.finalize().as_bytes()
}

fn lit_to_bytes(lit: Lit) -> [u8; 9] {
    let mut out = [0u8; 9];
    out[0..8].copy_from_slice(&(lit.id as u64).to_le_bytes());
    out[8] = u8::from(lit.negated);
    out
}

/// Picks a small, deterministically ordered set of candidate replacements for
/// `root` using `db` and the current depth map.
fn choose_candidate_replacements_for_root(
    root: AigRef,
    cuts_by_node: &[Vec<Cut>],
    depths: &[usize],
    db: &CutDb,
) -> Vec<Replacement> {
    let mut cands: Vec<Replacement> = Vec::new();

    for cut in &cuts_by_node[root.id] {
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
            let frag_key = stable_fragment_key(&frag);
            let input_depths = frag.input_depths();
            let mut new_depth_at_root: usize = 0;
            for (i, leaf) in cut.leaves.iter().enumerate() {
                let leaf_depth = depths[leaf.node.id];
                let cand = leaf_depth + (input_depths[i] as usize);
                new_depth_at_root = core::cmp::max(new_depth_at_root, cand);
            }

            let score_depth = new_depth_at_root;
            let score_ands = p.ands;

            cands.push(Replacement {
                root,
                leaf_ops: cut.leaves.clone(),
                frag,
                frag_key,
                score_depth,
                score_ands,
            });
        }
    }

    // Deterministic ordering + cap for performance.
    cands.sort_by(|a, b| {
        (
            a.score_depth,
            a.score_ands,
            a.root.id,
            &a.leaf_ops,
            a.frag_key,
        )
            .cmp(&(
                b.score_depth,
                b.score_ands,
                b.root.id,
                &b.leaf_ops,
                b.frag_key,
            ))
    });
    cands.truncate(16);
    cands
}

#[derive(Debug, Clone)]
enum DepthFormula {
    Original,
    Replacement {
        leaves: Vec<AigOperand>,
        input_depths: [u16; 4],
    },
}

#[derive(Debug)]
struct DepthChange {
    node: AigRef,
    old_depth: usize,
    new_depth: usize,
}

#[derive(Debug)]
struct FormulaChange {
    root: AigRef,
    old_formula: DepthFormula,
    old_deps: Vec<AigRef>,
    new_deps: Vec<AigRef>,
}

struct VirtualDepthState<'a> {
    g: &'a GateFn,
    formulas: Vec<DepthFormula>,
    fanouts: Vec<Vec<AigRef>>,
    output_use_counts: Vec<usize>,
    output_depth_counts: BTreeMap<usize, usize>,
    depths: Vec<usize>,
}

fn push_unique(xs: &mut Vec<AigRef>, x: AigRef) {
    if !xs.contains(&x) {
        xs.push(x);
    }
}

impl<'a> VirtualDepthState<'a> {
    fn new(
        g: &'a GateFn,
        depth_map: &std::collections::HashMap<AigRef, usize>,
    ) -> VirtualDepthState<'a> {
        let mut fanouts: Vec<Vec<AigRef>> = vec![Vec::new(); g.gates.len()];
        for (id, node) in g.gates.iter().enumerate() {
            if let AigNode::And2 { a, b, .. } = node {
                let parent = AigRef { id };
                push_unique(&mut fanouts[a.node.id], parent);
                push_unique(&mut fanouts[b.node.id], parent);
            }
        }

        let mut depths = vec![0usize; g.gates.len()];
        for (node, depth) in depth_map {
            depths[node.id] = *depth;
        }

        let mut output_use_counts = vec![0usize; g.gates.len()];
        let mut output_depth_counts = BTreeMap::new();
        for output in &g.outputs {
            for op in output.bit_vector.iter_lsb_to_msb() {
                output_use_counts[op.node.id] += 1;
                *output_depth_counts.entry(depths[op.node.id]).or_insert(0) += 1;
            }
        }

        Self {
            g,
            formulas: vec![DepthFormula::Original; g.gates.len()],
            fanouts,
            output_use_counts,
            output_depth_counts,
            depths,
        }
    }

    fn max_output_node_depth(&self) -> usize {
        self.output_depth_counts
            .keys()
            .next_back()
            .copied()
            .unwrap_or(0)
    }

    fn deps_for_formula(&self, node: AigRef, formula: &DepthFormula) -> Vec<AigRef> {
        match formula {
            DepthFormula::Original => match &self.g.gates[node.id] {
                AigNode::And2 { a, b, .. } => {
                    let mut deps = Vec::with_capacity(2);
                    push_unique(&mut deps, a.node);
                    push_unique(&mut deps, b.node);
                    deps
                }
                AigNode::Input { .. } | AigNode::Literal { .. } => Vec::new(),
            },
            DepthFormula::Replacement { leaves, .. } => {
                let mut deps = Vec::with_capacity(leaves.len());
                for leaf in leaves {
                    push_unique(&mut deps, leaf.node);
                }
                deps
            }
        }
    }

    fn replacement_depth(&self, cand: &Replacement) -> usize {
        let input_depths = cand.frag.input_depths();
        let mut depth = 0usize;
        for (i, leaf) in cand.leaf_ops.iter().enumerate() {
            let cand_depth = self.depths[leaf.node.id] + input_depths[i] as usize;
            depth = depth.max(cand_depth);
        }
        depth
    }

    fn recompute_depth(&self, node: AigRef) -> usize {
        match &self.formulas[node.id] {
            DepthFormula::Original => match &self.g.gates[node.id] {
                AigNode::Input { .. } | AigNode::Literal { .. } => 0,
                AigNode::And2 { a, b, .. } => {
                    1 + self.depths[a.node.id].max(self.depths[b.node.id])
                }
            },
            DepthFormula::Replacement {
                leaves,
                input_depths,
            } => {
                let mut depth = 0usize;
                for (i, leaf) in leaves.iter().enumerate() {
                    let cand_depth = self.depths[leaf.node.id] + input_depths[i] as usize;
                    depth = depth.max(cand_depth);
                }
                depth
            }
        }
    }

    fn record_depth_change(
        &mut self,
        node: AigRef,
        new_depth: usize,
        changes: &mut Vec<DepthChange>,
    ) {
        let old_depth = self.depths[node.id];
        if new_depth == old_depth {
            return;
        }
        let output_uses = self.output_use_counts[node.id];
        if output_uses != 0 {
            let old_count = self
                .output_depth_counts
                .get_mut(&old_depth)
                .expect("old output depth count should exist");
            *old_count -= output_uses;
            if *old_count == 0 {
                self.output_depth_counts.remove(&old_depth);
            }
            *self.output_depth_counts.entry(new_depth).or_insert(0) += output_uses;
        }
        self.depths[node.id] = new_depth;
        changes.push(DepthChange {
            node,
            old_depth,
            new_depth,
        });
    }

    fn propagate_from(&mut self, root: AigRef, new_depth: usize, changes: &mut Vec<DepthChange>) {
        let mut worklist = VecDeque::new();
        self.record_depth_change(root, new_depth, changes);
        worklist.extend(self.fanouts[root.id].iter().copied());

        while let Some(node) = worklist.pop_front() {
            let recomputed = self.recompute_depth(node);
            if recomputed == self.depths[node.id] {
                continue;
            }
            self.record_depth_change(node, recomputed, changes);
            worklist.extend(self.fanouts[node.id].iter().copied());
        }
    }

    fn apply_formula_change(
        &mut self,
        cand: &Replacement,
        new_root_depth: usize,
        changes: &mut Vec<DepthChange>,
    ) -> FormulaChange {
        let root = cand.root;
        let old_formula = self.formulas[root.id].clone();
        let old_deps = self.deps_for_formula(root, &old_formula);
        for dep in &old_deps {
            self.fanouts[dep.id].retain(|fanout| *fanout != root);
        }

        let new_formula = DepthFormula::Replacement {
            leaves: cand.leaf_ops.clone(),
            input_depths: cand.frag.input_depths(),
        };
        let new_deps = self.deps_for_formula(root, &new_formula);
        for dep in &new_deps {
            push_unique(&mut self.fanouts[dep.id], root);
        }
        self.formulas[root.id] = new_formula;

        self.propagate_from(root, new_root_depth, changes);
        FormulaChange {
            root,
            old_formula,
            old_deps,
            new_deps,
        }
    }

    fn rollback_formula_change(&mut self, formula_change: FormulaChange) {
        for dep in &formula_change.new_deps {
            self.fanouts[dep.id].retain(|fanout| *fanout != formula_change.root);
        }
        for dep in &formula_change.old_deps {
            push_unique(&mut self.fanouts[dep.id], formula_change.root);
        }
        self.formulas[formula_change.root.id] = formula_change.old_formula;
    }

    fn rollback_depth_changes(&mut self, changes: Vec<DepthChange>) {
        for change in changes.into_iter().rev() {
            let output_uses = self.output_use_counts[change.node.id];
            if output_uses != 0 {
                let new_count = self
                    .output_depth_counts
                    .get_mut(&change.new_depth)
                    .expect("new output depth count should exist");
                *new_count -= output_uses;
                if *new_count == 0 {
                    self.output_depth_counts.remove(&change.new_depth);
                }
                *self
                    .output_depth_counts
                    .entry(change.old_depth)
                    .or_insert(0) += output_uses;
            }
            self.depths[change.node.id] = change.old_depth;
        }
    }

    fn try_accept_replacement(&mut self, cand: &Replacement) -> bool {
        if !matches!(self.formulas[cand.root.id], DepthFormula::Original) {
            return false;
        }
        let old_output_node_depth = self.max_output_node_depth();
        let old_root_depth = self.depths[cand.root.id];
        let new_root_depth = self.replacement_depth(cand);
        if new_root_depth >= old_root_depth {
            return false;
        }

        let mut changes = Vec::new();
        let formula_change = self.apply_formula_change(cand, new_root_depth, &mut changes);
        if self.max_output_node_depth() < old_output_node_depth {
            true
        } else {
            self.rollback_depth_changes(changes);
            self.rollback_formula_change(formula_change);
            false
        }
    }
}

fn max_output_node_depth_from_depths(g: &GateFn, depths: &[usize]) -> usize {
    g.outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .map(|op| depths[op.node.id])
        .max()
        .unwrap_or(0)
}

/// Collects AND nodes on paths that reach the maximum output node depth.
///
/// `max_output_node_depth` is measured in AIG edges from an input/literal to an
/// output operand node. It is therefore one less than `deepest_path.len()` for
/// a non-empty path.
fn collect_critical_roots(
    g: &GateFn,
    depths: &[usize],
    max_output_node_depth: usize,
) -> Vec<AigRef> {
    if max_output_node_depth == 0 {
        return Vec::new();
    }

    let mut roots = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut worklist: Vec<AigRef> = g
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .filter_map(|op| {
            if depths[op.node.id] == max_output_node_depth {
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
        let node_depth = depths[node.id];
        for child in [a.node, b.node] {
            if depths[child.id] + 1 == node_depth {
                worklist.push(child);
            }
        }
    }

    let mut roots: Vec<AigRef> = roots.into_iter().collect();
    roots.sort_by(|a, b| {
        depths[b.id]
            .cmp(&depths[a.id])
            .then_with(|| a.id.cmp(&b.id))
    });
    roots
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

/// Rebuilds `g` while replacing each requested root, then runs DCE.
fn rebuild_with_replacements(g: &GateFn, replacements: &[Replacement]) -> GateFn {
    let mut gb = GateBuilder::new(g.name.clone(), GateBuilderOptions::opt());
    let replacements_by_root: BTreeMap<AigRef, &Replacement> = replacements
        .iter()
        .map(|replacement| (replacement.root, replacement))
        .collect();

    let mut orig_to_new: std::collections::HashMap<AigRef, AigOperand> =
        std::collections::HashMap::new();

    // Build primary inputs in the same order.
    for input in &g.inputs {
        let bv = gb.add_input(input.name.clone(), input.bit_vector.get_bit_count());
        for (i, op) in bv.iter_lsb_to_msb().enumerate() {
            // Map to the original input node by looking at the operand in the original
            // GateFn.
            let orig_op = input.bit_vector.get_lsb(i);
            gb.add_pir_node_ids(op.node, g.gates[orig_op.node.id].get_pir_node_ids());
            orig_to_new.insert(orig_op.node, *op);
        }
    }

    // Worklist-based rebuild from outputs.
    let mut worklist: Vec<AigRef> = g
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .map(|op| op.node)
        .collect();

    let mut processing: std::collections::BTreeSet<AigRef> = std::collections::BTreeSet::new();

    while let Some(r) = worklist.pop() {
        if orig_to_new.contains_key(&r) {
            continue;
        }
        if !processing.insert(r) {
            continue;
        }

        if let Some(repl) = replacements_by_root.get(&r).copied() {
            // Ensure leaf nodes are built first.
            let mut all_ready = true;
            let mut leaf_new_ops: Vec<AigOperand> = Vec::with_capacity(repl.leaf_ops.len());
            for leaf in &repl.leaf_ops {
                if let Some(op) = orig_to_new.get(&leaf.node).copied() {
                    let op = if leaf.negated { op.negate() } else { op };
                    leaf_new_ops.push(op);
                } else {
                    all_ready = false;
                    worklist.push(r);
                    worklist.push(leaf.node);
                }
            }
            if !all_ready {
                processing.remove(&r);
                continue;
            }

            let replacement_pir_node_ids = replacement_pir_node_ids(g, repl);
            let new_op = instantiate_fragment(
                &mut gb,
                &repl.frag,
                &leaf_new_ops,
                replacement_pir_node_ids.as_slice(),
            );
            orig_to_new.insert(r, new_op);
            processing.remove(&r);
            continue;
        }

        match &g.gates[r.id] {
            AigNode::Input { .. } => {
                // Inputs already mapped above.
                processing.remove(&r);
            }
            AigNode::Literal { value: v, .. } => {
                let op = if *v { gb.get_true() } else { gb.get_false() };
                gb.add_pir_node_ids(op.node, g.gates[r.id].get_pir_node_ids());
                orig_to_new.insert(r, op);
                processing.remove(&r);
            }
            AigNode::And2 { a, b, .. } => {
                if !orig_to_new.contains_key(&a.node) {
                    worklist.push(r);
                    worklist.push(a.node);
                    processing.remove(&r);
                    continue;
                }
                if !orig_to_new.contains_key(&b.node) {
                    worklist.push(r);
                    worklist.push(b.node);
                    processing.remove(&r);
                    continue;
                }
                let mut new_a = orig_to_new[&a.node];
                if a.negated {
                    new_a = new_a.negate();
                }
                let mut new_b = orig_to_new[&b.node];
                if b.negated {
                    new_b = new_b.negate();
                }
                let new_op = gb.add_and_binary(new_a, new_b);
                gb.add_pir_node_ids(new_op.node, g.gates[r.id].get_pir_node_ids());
                orig_to_new.insert(r, new_op);
                processing.remove(&r);
            }
        }
    }

    // Build outputs in the same order.
    for output in &g.outputs {
        let mut bits: Vec<AigOperand> = Vec::with_capacity(output.bit_vector.get_bit_count());
        for op in output.bit_vector.iter_lsb_to_msb() {
            let mut new_op = orig_to_new[&op.node];
            if op.negated {
                new_op = new_op.negate();
            }
            bits.push(new_op);
        }
        let bv = crate::aig::AigBitVector::from_lsb_is_index_0(&bits);
        gb.add_output(output.name.clone(), bv);
    }

    let mut out = gb.build();
    out = dce(&out);
    out
}

/// Performs iterative depth-first rewriting using the 4-input cut DB.
pub fn rewrite_gatefn_with_cut_db(g: &GateFn, db: &CutDb, opts: RewriteOptions) -> GateFn {
    let mut cur = g.clone();

    // Depth-only mode: in each global recompute round, cheaply evaluate many
    // critical-path candidates in a virtual depth model. Accepted replacements
    // are materialized together in one rebuild/DCE at the end of the round.
    let mut iter: usize = 0;
    loop {
        if opts.max_iterations != 0 && iter >= opts.max_iterations {
            log::info!(
                "cut-db rewrite: reached max_iterations={}; stopping",
                opts.max_iterations
            );
            break;
        }
        iter += 1;

        let t_iter0 = Instant::now();
        let t0 = Instant::now();
        let id_to_use_count = get_id_to_use_count(&cur);
        let t_use_count_ms = t0.elapsed().as_millis();
        let live_nodes: Vec<AigRef> = id_to_use_count.keys().cloned().collect();
        let t1 = Instant::now();
        let depth_stats = get_gate_depth(&cur, &live_nodes);
        let t_depth_ms = t1.elapsed().as_millis();
        let cur_path_len = depth_stats.deepest_path.len();
        let mut virtual_state = VirtualDepthState::new(&cur, &depth_stats.ref_to_depth);
        let cur_output_node_depth = virtual_state.max_output_node_depth();
        debug_assert_eq!(
            cur_path_len.saturating_sub(1),
            cur_output_node_depth,
            "summary deepest_path length and per-node depths should use adjacent units"
        );
        debug_assert_eq!(
            cur_output_node_depth,
            max_output_node_depth_from_depths(&cur, &virtual_state.depths),
            "virtual depth state should agree with dense output-depth scan"
        );

        log::info!(
            "cut-db rewrite round: live_nodes={} path_len={} output_node_depth={}",
            live_nodes.len(),
            cur_path_len,
            cur_output_node_depth
        );

        let t2 = Instant::now();
        let cuts_by_node = compute_cuts(&cur, opts.max_cuts_per_node);
        let t_cuts_ms = t2.elapsed().as_millis();
        let crit_roots = collect_critical_roots(
            &cur,
            &virtual_state.depths,
            virtual_state.max_output_node_depth(),
        );

        let mut accepted_replacements: Vec<Replacement> = Vec::new();
        let mut candidates_considered: usize = 0;
        let mut candidate_evals: usize = 0;

        let t_phase = Instant::now();
        'crit: for root in crit_roots.iter().copied() {
            if !matches!(cur.gates[root.id], AigNode::And2 { .. }) {
                continue;
            }
            let cands = choose_candidate_replacements_for_root(
                root,
                &cuts_by_node,
                &virtual_state.depths,
                db,
            );
            candidates_considered += cands.len();
            for cand in cands {
                if opts.max_candidate_evals_per_round != 0
                    && candidate_evals >= opts.max_candidate_evals_per_round
                {
                    break 'crit;
                }
                candidate_evals += 1;
                if virtual_state.try_accept_replacement(&cand) {
                    accepted_replacements.push(cand);
                    if opts.max_rewrites_per_round != 0
                        && accepted_replacements.len() >= opts.max_rewrites_per_round
                    {
                        break 'crit;
                    }
                }
            }
        }
        let t_phase_ms = t_phase.elapsed().as_millis();

        let mut rebuild_ms = 0;
        let mut after_path_len = cur_path_len;
        if !accepted_replacements.is_empty() {
            let t_rebuild = Instant::now();
            let new_g = rebuild_with_replacements(&cur, &accepted_replacements);
            rebuild_ms = t_rebuild.elapsed().as_millis();
            let new_id_to_use_count = get_id_to_use_count(&new_g);
            let new_live: Vec<AigRef> = new_id_to_use_count.keys().cloned().collect();
            let new_depth_stats = get_gate_depth(&new_g, &new_live);
            after_path_len = new_depth_stats.deepest_path.len();
            if after_path_len < cur_path_len {
                cur = new_g;
            } else {
                log::info!(
                    "cut-db rewrite: discarded non-improving replacement batch; before_path_len={} after_path_len={} replacements={}",
                    cur_path_len,
                    after_path_len,
                    accepted_replacements.len()
                );
                break;
            }
        }

        log::debug!(
            "cut-db rewrite round timings: use_count_ms={} depth_ms={} cuts_ms={} phase_ms={} rebuild_ms={} critical_roots={} cands_considered={} candidate_evals={} accepted={} before_path_len={} after_path_len={} round_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            t_cuts_ms,
            t_phase_ms,
            rebuild_ms,
            crit_roots.len(),
            candidates_considered,
            candidate_evals,
            accepted_replacements.len(),
            cur_path_len,
            after_path_len,
            t_iter0.elapsed().as_millis()
        );

        if accepted_replacements.is_empty() {
            if opts.max_candidate_evals_per_round != 0
                && candidate_evals >= opts.max_candidate_evals_per_round
            {
                log::info!(
                    "cut-db rewrite: hit per-round candidate eval cap ({}); stopping",
                    opts.max_candidate_evals_per_round
                );
            } else {
                log::info!("cut-db rewrite: no depth-improving candidate found; stopping");
            }
            break;
        }
    }

    cur
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

        let live_nodes: Vec<AigRef> = (0..g.gates.len()).map(|id| AigRef { id }).collect();
        let depth_stats = get_gate_depth(&g, &live_nodes);
        let depths: Vec<usize> = (0..g.gates.len())
            .map(|id| depth_stats.ref_to_depth[&AigRef { id }])
            .collect();
        let output_node_depth = max_output_node_depth_from_depths(&g, &depths);
        assert_eq!(depth_stats.deepest_path.len(), output_node_depth + 1);
        let critical_roots = collect_critical_roots(&g, &depths, output_node_depth);
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

        let rewritten = rewrite_gatefn_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
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
}
