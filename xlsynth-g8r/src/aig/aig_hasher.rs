// SPDX-License-Identifier: Apache-2.0

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::aig::gate::{AigNode, AigOperand, AigRef};
use crate::aig::topo::postorder_for_aig_refs_node_only;

/// Holds the "latest/best" depth and AigRef for a given hash.
struct HashData {
    pub min_depth: Option<(usize, AigRef)>,
}

impl HashData {
    pub fn new() -> Self {
        Self { min_depth: None }
    }

    pub fn add(&mut self, aig_ref: AigRef, depth: usize) {
        if self.min_depth.is_none() || depth < self.min_depth.unwrap().0 {
            self.min_depth = Some((depth, aig_ref));
        }
    }
}

#[derive(Clone)]
struct DepthAndHash {
    pub depth: usize,
    pub hash: blake3::Hash,
}

pub struct AigHasher {
    /// Mapping from blake3 hash values to associated "current best node info"
    /// (depth and AigRef)
    hash_to_nodes: HashMap<blake3::Hash, HashData>,

    /// Mapping from AigRef to depth and hash, used for memoization of depth and
    /// hash computations.
    ref_to_depth_and_hash: HashMap<AigRef, DepthAndHash>,
}

impl AigHasher {
    pub fn new() -> Self {
        Self {
            hash_to_nodes: HashMap::new(),
            ref_to_depth_and_hash: HashMap::new(),
        }
    }

    fn compute_node_depth(node: &AigNode, get_depth: impl Fn(&AigOperand) -> usize) -> usize {
        match node {
            AigNode::Input { .. } => 0,
            AigNode::Literal(..) => 0,
            AigNode::And2 { a, b, .. } => {
                let a_depth = get_depth(a);
                let b_depth = get_depth(b);
                std::cmp::max(a_depth, b_depth) + 1
            }
        }
    }

    fn compute_node_hash(
        node: &AigNode,
        get_hash: impl Fn(&AigOperand) -> blake3::Hash,
    ) -> blake3::Hash {
        static FALSE_HASH: Lazy<blake3::Hash> = Lazy::new(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&[2]);
            hasher.update(&[0]);
            hasher.finalize()
        });
        static TRUE_HASH: Lazy<blake3::Hash> = Lazy::new(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&[2]);
            hasher.update(&[1]);
            hasher.finalize()
        });
        let mut hasher = blake3::Hasher::new();
        match node {
            AigNode::And2 { a, b, .. } => {
                hasher.update(&[0]);
                let mut hash_a_bytes = get_hash(a).as_bytes().to_vec();
                if a.negated {
                    hash_a_bytes[0] ^= 1;
                }
                let mut hash_b_bytes = get_hash(b).as_bytes().to_vec();
                if b.negated {
                    hash_b_bytes[0] ^= 1;
                }
                if hash_a_bytes <= hash_b_bytes {
                    hasher.update(&hash_a_bytes);
                    hasher.update(&hash_b_bytes);
                } else {
                    hasher.update(&hash_b_bytes);
                    hasher.update(&hash_a_bytes);
                }
            }
            AigNode::Input { name, lsb_index } => {
                hasher.update(&[1]);
                hasher.update(name.as_bytes());
                hasher.update(&lsb_index.to_le_bytes());
            }
            AigNode::Literal(val) => {
                return if *val { *TRUE_HASH } else { *FALSE_HASH };
            }
        }
        hasher.finalize()
    }

    /// Returns the depth and hash for a given AigRef, computing them if
    /// necessary.
    ///
    /// If the depth and hash are already computed, they are returned from the
    /// memoized cache. Otherwise, the depth and hash are computed from
    /// scratch and the result is cached.
    pub fn get_depth_and_hash(
        &mut self,
        aig_ref: &AigRef,
        nodes: &[AigNode],
    ) -> (usize, blake3::Hash) {
        if let Some(depth_and_hash) = self.ref_to_depth_and_hash.get(aig_ref) {
            return (depth_and_hash.depth, depth_and_hash.hash);
        }
        let postorder =
            postorder_for_aig_refs_node_only(&[*aig_ref], nodes, &self.ref_to_depth_and_hash);
        for current in postorder {
            let node = &nodes[current.id];
            let get_depth = |op: &AigOperand| self.ref_to_depth_and_hash[&op.node].depth;
            let get_hash = |op: &AigOperand| self.ref_to_depth_and_hash[&op.node].hash;
            let depth = Self::compute_node_depth(node, get_depth);
            let hash = Self::compute_node_hash(node, get_hash);
            self.ref_to_depth_and_hash
                .insert(current, DepthAndHash { depth, hash });
        }
        let result = self.ref_to_depth_and_hash[aig_ref].clone();
        (result.depth, result.hash)
    }

    pub fn get_depth(&mut self, aig_ref: &AigRef, nodes: &[AigNode]) -> usize {
        self.get_depth_and_hash(aig_ref, nodes).0
    }

    pub fn get_hash(&mut self, aig_ref: &AigRef, nodes: &[AigNode]) -> blake3::Hash {
        self.get_depth_and_hash(aig_ref, nodes).1
    }

    /// Feeds a reference to the AigHasher and returns the "latest/best"
    /// reference if there is one.
    ///
    /// If there is an equivalent node with a smaller depth, that node is
    /// returned instead.
    pub fn feed_ref(&mut self, aig_ref: &AigRef, nodes: &[AigNode]) -> Option<AigRef> {
        // Depth of this node is the max depth of all its operands plus one.
        let depth = self.get_depth(aig_ref, nodes);

        // We want to insert this ref into the formula data.
        // If there's an extra that has the same formula with <= depth, we return that
        // instead.
        let hash = self.get_hash(aig_ref, nodes);

        let hash_data = self.hash_to_nodes.entry(hash).or_insert_with(HashData::new);
        if let Some((min_depth, min_ref)) = hash_data.min_depth {
            if depth >= min_depth {
                // Note: if we found something existing/better there's no need to insert the
                // given node into the hash data.
                return Some(min_ref);
            }
        }
        hash_data.add(*aig_ref, depth);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::{AigNode, AigOperand, AigRef};
    use crate::test_utils::{setup_graph_with_redundancies, setup_simple_graph};

    #[test]
    fn stack_overflow_on_deep_and_tree() {
        // Construct a deep left-leaning AND tree
        let depth = 200_000; // Should be deep enough to overflow most stacks
        let mut nodes = Vec::with_capacity(depth + 1);
        // Start with a single input node
        nodes.push(AigNode::Input {
            name: "a".to_string(),
            lsb_index: 0,
        });
        for i in 1..=depth {
            let left = AigOperand {
                node: AigRef { id: i - 1 },
                negated: false,
            };
            let right = AigOperand {
                node: AigRef { id: 0 },
                negated: false,
            };
            nodes.push(AigNode::And2 {
                a: left,
                b: right,
                tags: None,
            });
        }
        let mut hasher = AigHasher::new();
        // This used to cause a stack overflow before the worklist-based implementation.
        let _ = hasher.get_depth_and_hash(&AigRef { id: depth }, &nodes);
    }

    #[test]
    fn test_simple_graph_depth_and_hash() {
        let tg = setup_simple_graph();
        let mut hasher = AigHasher::new();
        // Check depth and hash for output node 'o'
        let (depth, hash) = hasher.get_depth_and_hash(&tg.o.node, &tg.g.gates);
        assert_eq!(depth, 2, "Depth of output node 'o' should be 2");
        // Check that repeated calls are consistent
        let (depth2, hash2) = hasher.get_depth_and_hash(&tg.o.node, &tg.g.gates);
        assert_eq!(depth, depth2);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_graph_with_redundancies() {
        let tg = setup_graph_with_redundancies();
        let mut hasher = AigHasher::new();
        // Both outer0 and outer1 should have the same depth and hash
        let (depth0, hash0) = hasher.get_depth_and_hash(&tg.outer0.node, &tg.g.gates);
        let (depth1, hash1) = hasher.get_depth_and_hash(&tg.outer1.node, &tg.g.gates);
        assert_eq!(
            depth0, depth1,
            "Depths should be equal for redundant outputs"
        );
        assert_eq!(hash0, hash1, "Hashes should be equal for redundant outputs");
    }
}
