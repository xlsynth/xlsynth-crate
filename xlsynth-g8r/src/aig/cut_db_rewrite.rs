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

use std::collections::BTreeSet;
use std::time::Instant;

use blake3::Hasher;

use crate::aig::dce::dce;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::topo::topo_sort_refs;
use crate::cut_db::fragment::{GateFnFragment, Lit};
use crate::cut_db::loader::CutDb;
use crate::cut_db::tt16::TruthTable16;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::use_count::get_id_to_use_count;

#[derive(Debug, Clone, Copy)]
pub struct RewriteOptions {
    pub max_cuts_per_node: usize,
    /// Maximum number of outer rewrite iterations to run.
    ///
    /// If set to `0`, the pass runs **to convergence** (until no improving
    /// rewrite exists under the acceptance criterion).
    pub max_iterations: usize,
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
            AigNode::Literal(v) => {
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
                gb.add_and_binary(a_op, b_op)
            }
        };
        ops.push(op);
    }

    op_from_lit(frag.output, &ops)
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
    depth_map: &std::collections::HashMap<AigRef, usize>,
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
                let leaf_depth = *depth_map.get(&leaf.node).unwrap_or(&0);
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

/// Rebuilds `g` while replacing `repl.root` with `repl.frag`, then runs DCE.
fn rebuild_with_replacement(g: &GateFn, repl: &Replacement) -> GateFn {
    let mut gb = GateBuilder::new(g.name.clone(), GateBuilderOptions::opt());

    let mut orig_to_new: std::collections::HashMap<AigRef, AigOperand> =
        std::collections::HashMap::new();

    // Build primary inputs in the same order.
    for input in &g.inputs {
        let bv = gb.add_input(input.name.clone(), input.bit_vector.get_bit_count());
        for (i, op) in bv.iter_lsb_to_msb().enumerate() {
            // Map to the original input node by looking at the operand in the original
            // GateFn.
            let orig_op = input.bit_vector.get_lsb(i);
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

        if r == repl.root {
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

            let new_op = instantiate_fragment(&mut gb, &repl.frag, &leaf_new_ops);
            orig_to_new.insert(r, new_op);
            processing.remove(&r);
            continue;
        }

        match &g.gates[r.id] {
            AigNode::Input { .. } => {
                // Inputs already mapped above.
                processing.remove(&r);
            }
            AigNode::Literal(v) => {
                let op = if *v { gb.get_true() } else { gb.get_false() };
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

    // Depth-only mode: iterate the critical path and accept only depth-improving
    // rewrites. We intentionally do not do any off-critical-path "area recovery"
    // in this pass (that can be handled by other cleanup passes).
    let mut iter: usize = 0;
    // Deterministic per-iteration effort cap for the depth phase. On large graphs,
    // exhaustively scanning the critical path can dominate runtime when no further
    // depth improvement exists (or is hard to find).
    let max_depth_phase_rebuilds_per_iter: usize = 128;
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
        let cur_global_depth = depth_stats.deepest_path.len();

        log::info!(
            "cut-db rewrite iter: live_nodes={} global_depth={}",
            live_nodes.len(),
            cur_global_depth
        );

        let t2 = Instant::now();
        let cuts_by_node = compute_cuts(&cur, opts.max_cuts_per_node);
        let t_cuts_ms = t2.elapsed().as_millis();
        let depth_map = depth_stats.ref_to_depth;
        let critical_path = depth_stats.deepest_path;

        let mut applied = false;
        let mut candidates_considered: usize = 0;
        let mut rebuilds_tried: usize = 0;

        let mut crit_roots: Vec<AigRef> = critical_path;
        crit_roots.sort_by(|a, b| {
            let da = *depth_map.get(a).unwrap_or(&0);
            let dbb = *depth_map.get(b).unwrap_or(&0);
            dbb.cmp(&da).then_with(|| a.id.cmp(&b.id))
        });
        let t_phase = Instant::now();
        'crit: for root in crit_roots {
            if !matches!(cur.gates[root.id], AigNode::And2 { .. }) {
                continue;
            }
            let cands = choose_candidate_replacements_for_root(root, &cuts_by_node, &depth_map, db);
            candidates_considered += cands.len();
            for cand in cands {
                if rebuilds_tried >= max_depth_phase_rebuilds_per_iter {
                    break 'crit;
                }
                rebuilds_tried += 1;
                let new_g = rebuild_with_replacement(&cur, &cand);
                let new_id_to_use_count = get_id_to_use_count(&new_g);
                let new_live: Vec<AigRef> = new_id_to_use_count.keys().cloned().collect();
                let new_depth_stats = get_gate_depth(&new_g, &new_live);
                let new_global_depth = new_depth_stats.deepest_path.len();
                if new_global_depth < cur_global_depth {
                    cur = new_g;
                    applied = true;
                    break 'crit;
                }
            }
        }
        let t_phase_ms = t_phase.elapsed().as_millis();
        log::debug!(
            "cut-db rewrite iter timings: use_count_ms={} depth_ms={} cuts_ms={} phase_ms={} cands_considered={} rebuilds_tried={} iter_elapsed_ms={}",
            t_use_count_ms,
            t_depth_ms,
            t_cuts_ms,
            t_phase_ms,
            candidates_considered,
            rebuilds_tried,
            t_iter0.elapsed().as_millis()
        );

        if !applied {
            if rebuilds_tried >= max_depth_phase_rebuilds_per_iter {
                log::info!(
                    "cut-db rewrite: hit per-iter rebuild cap ({}); stopping",
                    max_depth_phase_rebuilds_per_iter
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
    fn test_cut_db_rewrite_does_not_increase_global_depth() {
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

        let before = get_summary_stats(&g);
        let rewritten = rewrite_gatefn_with_cut_db(
            &g,
            &db,
            RewriteOptions {
                max_cuts_per_node: 32,
                max_iterations: 8,
            },
        );
        let after = get_summary_stats(&rewritten);

        assert!(
            after.deepest_path <= before.deepest_path,
            "global depth should not increase: before={} after={}",
            before.deepest_path,
            after.deepest_path
        );
    }
}
