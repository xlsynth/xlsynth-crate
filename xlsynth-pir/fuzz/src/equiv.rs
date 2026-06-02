// SPDX-License-Identifier: Apache-2.0

//! Equivalence utilities used by fuzz targets.
//!
//! Provides forward and reverse structural equivalence computations between two
//! XLS IR functions. Indices are returned as `usize` node indices into the
//! functions' node vectors. We refer to the two functions as `lhs` and `rhs`.

use std::collections::{HashMap, HashSet};

use xlsynth_pir::dce::get_dead_nodes;
use xlsynth_pir::ir::Fn;
use xlsynth_pir::ir::NodeRef;
use xlsynth_pir::ir_utils::{get_topological, operands};
use xlsynth_pir::node_hashing::{
    BwdHash, FwdHash, compute_node_backward_structural_hash, compute_node_structural_hash,
};

#[derive(Debug, Clone)]
pub struct Equivalences {
    pub lhs_to_rhs: HashMap<usize, Vec<usize>>,
    pub rhs_to_lhs: HashMap<usize, Vec<usize>>,
}

/// Computes the sets of nodes in `lhs` that are equivalent in the forward
/// direction (in a common-subexpression sense) to nodes in `rhs` and vice
/// versa.
pub fn compute_forward_equivalences(lhs: &Fn, rhs: &Fn) -> Equivalences {
    // Helper: compute forward structural hashes for all nodes via topo order.
    fn compute_forward_hashes(f: &Fn) -> Vec<FwdHash> {
        let order = get_topological(f);
        let mut hashes: Vec<Option<FwdHash>> = vec![None; f.nodes.len()];
        for nr in order {
            let child_hashes: Vec<FwdHash> = operands(&f.get_node(nr).payload)
                .into_iter()
                .map(|c| hashes[c.index].expect("child hash must be computed first"))
                .collect();
            let h = compute_node_structural_hash(f, nr, &child_hashes);
            hashes[nr.index] = Some(h);
        }
        hashes
            .into_iter()
            .map(|o| o.expect("hash must be set"))
            .collect()
    }

    let lhs_hashes = compute_forward_hashes(lhs);
    let rhs_hashes = compute_forward_hashes(rhs);

    // Build reverse indices by hash.
    let mut by_hash_lhs: HashMap<FwdHash, Vec<usize>> = HashMap::new();
    for (idx, h) in lhs_hashes.iter().enumerate() {
        by_hash_lhs.entry(*h).or_default().push(idx);
    }
    let mut by_hash_rhs: HashMap<FwdHash, Vec<usize>> = HashMap::new();
    for (idx, h) in rhs_hashes.iter().enumerate() {
        by_hash_rhs.entry(*h).or_default().push(idx);
    }

    // Produce dense maps including keys for nodes with no equivalents (empty vecs).
    let mut lhs_to_rhs: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, h) in lhs_hashes.iter().enumerate() {
        let mut v = by_hash_rhs.get(h).cloned().unwrap_or_default();
        v.sort_unstable();
        lhs_to_rhs.insert(idx, v);
    }

    let mut rhs_to_lhs: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, h) in rhs_hashes.iter().enumerate() {
        let mut v = by_hash_lhs.get(h).cloned().unwrap_or_default();
        v.sort_unstable();
        rhs_to_lhs.insert(idx, v);
    }

    Equivalences {
        lhs_to_rhs,
        rhs_to_lhs,
    }
}

/// Computes the sets of nodes in `lhs` that are reverse equivalent to nodes in
/// `rhs` and vice versa. A node is reverse equivalent to another node if it is
/// structurally identical with respect to the return node. Dead nodes are
/// ignored.
pub fn compute_reverse_equivalences_to_return(lhs: &Fn, rhs: &Fn) -> Equivalences {
    assert!(
        lhs.ret_node_ref.is_some(),
        "compute_reverse_equivalences_to_return: old function has no return node"
    );
    assert!(
        rhs.ret_node_ref.is_some(),
        "compute_reverse_equivalences_to_return: new function has no return node"
    );

    fn compute_backward_hashes_filter_dead(f: &Fn) -> (Vec<BwdHash>, HashSet<usize>) {
        let n = f.nodes.len();
        if n == 0 {
            return (vec![], HashSet::new());
        }
        // Build users with operand slots for the entire graph.
        let mut users: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (ui, node) in f.nodes.iter().enumerate() {
            for (slot, dep) in operands(&node.payload).into_iter().enumerate() {
                users[dep.index].push((ui, slot));
            }
        }
        // Dead set via helper.
        let dead_set: HashSet<usize> = get_dead_nodes(f).into_iter().map(|nr| nr.index).collect();
        // Compute backward hashes for all nodes using only live users.
        let order = get_topological(f);
        let mut out: Vec<Option<BwdHash>> = vec![None; n];
        // Sentinel to distinguish the return value as having an implicit user.
        let ret_index: Option<usize> = f.ret_node_ref.map(|nr| nr.index);
        let mut ret_sentinel_hasher = blake3::Hasher::new();
        ret_sentinel_hasher.update(b"__xlsynth_return_sink__");
        let ret_sentinel = BwdHash(ret_sentinel_hasher.finalize());
        for nr in order.into_iter().rev() {
            let i = nr.index;
            let mut pairs: Vec<(BwdHash, usize)> = Vec::new();
            for (u, slot) in users[i].iter() {
                if dead_set.contains(u) {
                    continue;
                }
                let h = out[*u].expect("live user backward hash must be computed before operands");
                pairs.push((h, *slot));
            }
            // Inject a sentinel user for the return node to prevent non-returns
            // that happen to have no (live) users from being reverse-equivalent
            // to the return value itself.
            if ret_index == Some(i) {
                pairs.push((ret_sentinel, usize::MAX));
            }
            out[i] = Some(compute_node_backward_structural_hash(
                f,
                NodeRef { index: i },
                &pairs,
            ));
        }
        let hashes: Vec<BwdHash> = out
            .into_iter()
            .map(|o| o.expect("backward hash must be computed for all nodes"))
            .collect();
        (hashes, dead_set)
    }

    let (lhs_bwd, lhs_dead) = compute_backward_hashes_filter_dead(lhs);
    let (rhs_bwd, rhs_dead) = compute_backward_hashes_filter_dead(rhs);

    // Build reverse indices by backward hash for live nodes only.
    let mut by_hash_lhs: HashMap<BwdHash, Vec<usize>> = HashMap::new();
    for (idx, bh) in lhs_bwd.iter().copied().enumerate() {
        if lhs_dead.contains(&idx) {
            continue;
        }
        by_hash_lhs.entry(bh).or_default().push(idx);
    }
    let mut by_hash_rhs: HashMap<BwdHash, Vec<usize>> = HashMap::new();
    for (idx, bh) in rhs_bwd.iter().copied().enumerate() {
        if rhs_dead.contains(&idx) {
            continue;
        }
        by_hash_rhs.entry(bh).or_default().push(idx);
    }

    // Produce dense maps including keys for nodes with no equivalents (empty vecs).
    let mut lhs_to_rhs: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, bh) in lhs_bwd.iter().copied().enumerate() {
        let mut v: Vec<usize> = if lhs_dead.contains(&idx) {
            Vec::new()
        } else {
            by_hash_rhs.get(&bh).cloned().unwrap_or_default()
        };
        v.sort_unstable();
        lhs_to_rhs.insert(idx, v);
    }

    let mut rhs_to_lhs: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, bh) in rhs_bwd.iter().copied().enumerate() {
        let mut v: Vec<usize> = if rhs_dead.contains(&idx) {
            Vec::new()
        } else {
            by_hash_lhs.get(&bh).cloned().unwrap_or_default()
        };
        v.sort_unstable();
        rhs_to_lhs.insert(idx, v);
    }

    Equivalences {
        lhs_to_rhs,
        rhs_to_lhs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir::PackageMember;
    use xlsynth_pir::ir_parser::Parser;

    fn parse_fn(ir: &str) -> Fn {
        let pkg_text = format!("package p\n\n{}\n", ir);
        let mut p = Parser::new(&pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        pkg.members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) => Some(f.clone()),
                _ => None,
            })
            .unwrap()
    }

    fn index_by_name(f: &Fn, name: &str) -> usize {
        f.nodes
            .iter()
            .position(|n| n.name.as_deref() == Some(name))
            .expect("node by name not found")
    }

    #[test]
    fn forward_equivalences_two_adds_equivalent() {
        // Two adds with different names are forward-equivalent.
        let f = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  a: bits[8] = param(name=a, id=1)
  b: bits[8] = param(name=b, id=2)
  sum1: bits[8] = add(a, b, id=10)
  sum2: bits[8] = add(a, b, id=11)
  ret r: bits[8] = identity(sum1, id=12)
}"#,
        );
        let eq = compute_forward_equivalences(&f, &f);
        let s1 = index_by_name(&f, "sum1");
        let s2 = index_by_name(&f, "sum2");
        let v1 = eq.lhs_to_rhs.get(&s1).unwrap();
        let v2 = eq.lhs_to_rhs.get(&s2).unwrap();
        assert!(v1.contains(&s1) && v1.contains(&s2));
        assert!(v2.contains(&s1) && v2.contains(&s2));
    }

    #[test]
    fn reverse_equivalences_return_is_unique_even_with_dead_clone() {
        // Both sides have same params; rhs has a dead node identical to return.
        // Ensure the dead node is not reverse-equivalent to the return value.
        let lhs = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  a: bits[8] = param(name=a, id=1)
  b: bits[8] = param(name=b, id=2)
  sum: bits[8] = add(a, b, id=10)
  ret r: bits[8] = identity(sum, id=11)
}"#,
        );
        let rhs = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  a: bits[8] = param(name=a, id=1)
  b: bits[8] = param(name=b, id=2)
  sum: bits[8] = add(a, b, id=20)
  ret r: bits[8] = identity(sum, id=21)
  dead_r: bits[8] = identity(dead_sum, id=31)
}"#,
        );

        let eq = compute_reverse_equivalences_to_return(&lhs, &rhs);
        let lhs_ret = lhs.ret_node_ref.unwrap().index;
        let rhs_ret = rhs.ret_node_ref.unwrap().index;
        let rhs_dead_r = index_by_name(&rhs, "dead_r");

        // Return must map to return on the other side.
        assert!(eq.lhs_to_rhs.get(&lhs_ret).unwrap().contains(&rhs_ret));
        // The dead clone must not map to the return.
        assert!(eq.rhs_to_lhs.get(&rhs_dead_r).unwrap().is_empty());
        assert!(!eq.rhs_to_lhs.get(&rhs_dead_r).unwrap().contains(&lhs_ret));
    }
}
