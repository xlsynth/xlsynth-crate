// SPDX-License-Identifier: Apache-2.0

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::gate::{AigNode, AigRef};

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

    fn get_depth_no_memo(&mut self, aig_ref: &AigRef, nodes: &[AigNode]) -> usize {
        if let Some(depth_and_hash) = self.ref_to_depth_and_hash.get(aig_ref) {
            return depth_and_hash.depth;
        }
        let node = &nodes[aig_ref.id];
        let result = match node {
            AigNode::Input { .. } => 0,
            AigNode::Literal(..) => 0,
            AigNode::And2 { a, b, .. } => {
                let a_depth = self.get_depth(&a.node, nodes);
                let b_depth = self.get_depth(&b.node, nodes);
                let result_depth = std::cmp::max(a_depth, b_depth) + 1;
                result_depth
            }
        };
        result
    }

    fn get_hash_no_memo(&mut self, aig_ref: &AigRef, nodes: &[AigNode]) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        let node = &nodes[aig_ref.id];

        static FALSE_HASH: Lazy<blake3::Hash> = Lazy::new(|| {
            let mut hasher = blake3::Hasher::new();
            // Use a type identifier for Literal False
            hasher.update(&[2]);
            hasher.update(&[0]);
            hasher.finalize()
        });
        static TRUE_HASH: Lazy<blake3::Hash> = Lazy::new(|| {
            let mut hasher = blake3::Hasher::new();
            // Use a type identifier for Literal True
            // Note: Changed from [3] to [2] to keep Literal type identifier consistent
            hasher.update(&[2]);
            hasher.update(&[1]);
            hasher.finalize()
        });

        // Hash based on node type and canonicalized inputs for commutativity
        match node {
            AigNode::And2 { a, b, .. } => {
                // Use a type identifier for And2
                hasher.update(&[0]);

                // Recursively get hashes for operands, including negation
                // Use the memoized get_hash to avoid recomputation and cycles
                let mut hash_a_bytes = self.get_hash(&a.node, nodes).as_bytes().to_vec();
                if a.negated {
                    hash_a_bytes[0] ^= 1;
                } // Simple way to incorporate negation

                let mut hash_b_bytes = self.get_hash(&b.node, nodes).as_bytes().to_vec();
                if b.negated {
                    hash_b_bytes[0] ^= 1;
                } // Simple way to incorporate negation

                // Sort operand hashes for canonical representation
                if hash_a_bytes <= hash_b_bytes {
                    hasher.update(&hash_a_bytes);
                    hasher.update(&hash_b_bytes);
                } else {
                    hasher.update(&hash_b_bytes);
                    hasher.update(&hash_a_bytes);
                }
            }
            AigNode::Input { name, lsb_index } => {
                // Use a type identifier for Input
                hasher.update(&[1]);
                hasher.update(name.as_bytes());
                // lsb_index is usize, not Option<usize>
                hasher.update(&lsb_index.to_le_bytes());
            }
            AigNode::Literal(val) => {
                // Return precomputed hashes for literals
                if *val {
                    return *TRUE_HASH;
                } else {
                    return *FALSE_HASH;
                }
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
        let depth = self.get_depth_no_memo(aig_ref, nodes);
        let hash = self.get_hash_no_memo(aig_ref, nodes);
        self.ref_to_depth_and_hash
            .insert(*aig_ref, DepthAndHash { depth, hash });
        (depth, hash)
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
