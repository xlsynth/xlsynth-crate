// SPDX-License-Identifier: Apache-2.0

//! Computes a graph edit distance between two XLS IR functions.

use std::collections::HashSet;

use crate::ir::{Fn, Node, NodePayload};
use crate::ir::{binop_to_operator, nary_op_to_operator, unop_to_operator};
use crate::node_hashing::FwdHash;
use crate::structural_similarity::collect_structural_entries;

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeSignature {
    WithOperands {
        op: String,
        ty: String,
        operands: Vec<String>,
    },
    Simple(String),
}

/// Options for ranking candidate functions against a query function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateRankingOptions {
    /// Maximum number of candidates to keep after the structural-hash prefilter
    /// and score with edit distance.
    pub prefilter_limit: usize,
}

impl Default for CandidateRankingOptions {
    fn default() -> Self {
        Self {
            prefilter_limit: usize::MAX,
        }
    }
}

/// Ranked candidate produced by structural prefiltering followed by edit
/// distance scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedFnCandidate {
    pub index: usize,
    pub shared_structural_hashes: usize,
    pub query_structural_hashes: usize,
    pub candidate_structural_hashes: usize,
    pub edit_distance: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StructuralPrefilterCandidate {
    index: usize,
    shared_structural_hashes: usize,
    query_structural_hashes: usize,
    candidate_structural_hashes: usize,
}

/// Ranks candidate functions by first applying a cheap structural-hash
/// prefilter, then computing edit distance for the survivors.
pub fn rank_fn_candidates_by_similarity(
    query: &Fn,
    candidates: &[&Fn],
    options: CandidateRankingOptions,
) -> Vec<RankedFnCandidate> {
    if candidates.is_empty() || options.prefilter_limit == 0 {
        return Vec::new();
    }

    let query_hashes: HashSet<FwdHash> = collect_structural_hashes(query);
    let query_hash_count = query_hashes.len();
    let prefilter_limit = std::cmp::min(options.prefilter_limit, candidates.len());
    let mut prefiltered: Vec<StructuralPrefilterCandidate> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let candidate_hashes = collect_structural_hashes(candidate);
            let shared_structural_hashes = candidate_hashes
                .iter()
                .filter(|hash| query_hashes.contains(hash))
                .count();
            StructuralPrefilterCandidate {
                index,
                shared_structural_hashes,
                query_structural_hashes: query_hash_count,
                candidate_structural_hashes: candidate_hashes.len(),
            }
        })
        .collect();

    prefiltered.sort_by(|lhs, rhs| {
        rhs.shared_structural_hashes
            .cmp(&lhs.shared_structural_hashes)
            .then_with(|| {
                lhs.candidate_structural_hashes
                    .cmp(&rhs.candidate_structural_hashes)
            })
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    prefiltered.truncate(prefilter_limit);

    let mut ranked: Vec<RankedFnCandidate> = prefiltered
        .into_iter()
        .map(|candidate| RankedFnCandidate {
            index: candidate.index,
            shared_structural_hashes: candidate.shared_structural_hashes,
            query_structural_hashes: candidate.query_structural_hashes,
            candidate_structural_hashes: candidate.candidate_structural_hashes,
            edit_distance: compute_edit_distance(query, candidates[candidate.index]),
        })
        .collect();
    ranked.sort_by(|lhs, rhs| {
        lhs.edit_distance
            .cmp(&rhs.edit_distance)
            .then_with(|| {
                rhs.shared_structural_hashes
                    .cmp(&lhs.shared_structural_hashes)
            })
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    ranked
}

/// Compute the edit distance between two functions based on their computational
/// nodes. Nodes whose payload operator is "get_param" or "nil" are filtered
/// out.
pub fn compute_edit_distance(lhs: &Fn, rhs: &Fn) -> u64 {
    let lhs_signatures: Vec<NodeSignature> = lhs
        .nodes
        .iter()
        .filter_map(|node| compute_node_signature(lhs, node))
        .collect();
    let rhs_signatures: Vec<NodeSignature> = rhs
        .nodes
        .iter()
        .filter_map(|node| compute_node_signature(rhs, node))
        .collect();

    let n = lhs_signatures.len();
    let m = rhs_signatures.len();

    // Initialize DP table.
    let mut dp = vec![vec![0; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    // Use standard Levenshtein edit distance over these signatures.
    for i in 1..=n {
        for j in 1..=m {
            let cost = if lhs_signatures[i - 1] == rhs_signatures[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = std::cmp::min(
                dp[i - 1][j] + 1, // deletion
                std::cmp::min(
                    dp[i][j - 1] + 1,        // insertion
                    dp[i - 1][j - 1] + cost, // substitution
                ),
            );
        }
    }
    dp[n][m] as u64
}

/// Returns a signature for a given node that is used for comparison in the edit
/// distance. Returns None for nodes that are not "visible" (e.g. get_param or
/// nil nodes).
fn compute_node_signature(f: &Fn, node: &Node) -> Option<NodeSignature> {
    let op = node.payload.get_operator();
    if op == "get_param" || op == "nil" {
        return None;
    }
    match &node.payload {
        NodePayload::Binop(op_val, left_ref, right_ref) => {
            let op_str = binop_to_operator(*op_val);
            let left_name = extract_operand_name(f.get_node(*left_ref));
            let right_name = extract_operand_name(f.get_node(*right_ref));
            Some(NodeSignature::WithOperands {
                op: op_str.to_string(),
                ty: format!("{}", node.ty),
                operands: vec![left_name, right_name],
            })
        }
        NodePayload::Unop(op_val, operand_ref) => {
            let op_str = unop_to_operator(*op_val);
            let operand_name = extract_operand_name(f.get_node(*operand_ref));
            Some(NodeSignature::WithOperands {
                op: op_str.to_string(),
                ty: format!("{}", node.ty),
                operands: vec![operand_name],
            })
        }
        NodePayload::Nary(op_val, operand_refs) => {
            let op_str = nary_op_to_operator(*op_val);
            let operand_names = operand_refs
                .iter()
                .map(|nr| extract_operand_name(f.get_node(*nr)))
                .collect();
            Some(NodeSignature::WithOperands {
                op: op_str.to_string(),
                ty: format!("{}", node.ty),
                operands: operand_names,
            })
        }
        _ => Some(NodeSignature::Simple(format!("{}:{}", op, node.ty))),
    }
}

/// Extract a canonical "name" for an operand. For a get_param node we use its
/// given name. For any other node we fall back to its operator.
fn extract_operand_name(node: &Node) -> String {
    if node.payload.get_operator() == "get_param" {
        node.name.clone().unwrap_or_else(|| "<anon>".to_string())
    } else {
        node.payload.get_operator().to_string()
    }
}

fn collect_structural_hashes(f: &Fn) -> HashSet<FwdHash> {
    let (entries, _) = collect_structural_entries(f);
    entries.into_iter().map(|entry| entry.hash).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Package;
    use crate::ir_parser::Parser;

    fn parse_ir_from_string(s: &str) -> Package {
        let mut parser = Parser::new(s);
        parser.parse_and_validate_package().unwrap()
    }

    #[test]
    fn test_edit_distance_inverter_vs_identity() {
        let inverter_pkg = parse_ir_from_string(
            r#"package inverter_pkg
            top fn inverter(x: bits[1]) -> bits[1] {
                ret not.2: bits[1] = not(x, id=2)
            }
            "#,
        );
        let inverter = inverter_pkg.get_top_fn().unwrap();
        let identity_pkg = parse_ir_from_string(
            r#"package identity_pkg
            top fn identity(x: bits[1]) -> bits[1] {
                ret identity.2: bits[1] = identity(x, id=2)
            }
            "#,
        );
        let identity = identity_pkg.get_top_fn().unwrap();
        let edit_distance = compute_edit_distance(&inverter, &identity);
        // The only difference is the operator in the node ("not" vs "identity").
        assert_eq!(edit_distance, 1);
    }

    #[test]
    fn test_edit_distance_and_with_converse_args() {
        let and_pkg = parse_ir_from_string(
            r#"package and_pkg
            top fn and(x: bits[1], y: bits[1]) -> bits[1] {
                ret and.3: bits[1] = and(x, y, id=3)
            }
            "#,
        );
        let and_fn = and_pkg.get_top_fn().unwrap();
        let and_converse_pkg = parse_ir_from_string(
            r#"package and_converse_pkg
            top fn and_converse(x: bits[1], y: bits[1]) -> bits[1] {
                ret and.3: bits[1] = and(y, x, id=3)
            }
            "#,
        );
        let and_converse_fn = and_converse_pkg.get_top_fn().unwrap();
        let edit_distance = compute_edit_distance(&and_fn, &and_converse_fn);
        // For the "and" node the operands differ in order,
        // causing a substitution cost of 1.
        assert_eq!(edit_distance, 1);
    }

    #[test]
    fn test_edit_distance_not_vs_neg() {
        let not_pkg = parse_ir_from_string(
            r#"package not_pkg
            top fn not(x: bits[1]) -> bits[1] {
                ret not.2: bits[1] = not(x, id=2)
            }
            "#,
        );
        let not_fn = not_pkg.get_top_fn().unwrap();
        let neg_pkg = parse_ir_from_string(
            r#"package neg_pkg
            top fn neg(x: bits[1]) -> bits[1] {
                ret neg.2: bits[1] = neg(x, id=2)
            }
            "#,
        );
        let neg_fn = neg_pkg.get_top_fn().unwrap();
        let edit_distance = compute_edit_distance(&not_fn, &neg_fn);
        // The only difference is the operator in the node ("not" vs "neg").
        assert_eq!(edit_distance, 1);
    }

    #[test]
    fn test_edit_distance_not_not_vs_neg_neg() {
        let not_not_pkg = parse_ir_from_string(
            r#"package not_not_pkg
            top fn not_not(x: bits[1]) -> bits[1] {
                not.2: bits[1] = not(x, id=2)
                ret not.3: bits[1] = not(not.2, id=3)
            }
            "#,
        );
        let not_not_fn = not_not_pkg.get_top_fn().unwrap();
        let neg_neg_pkg = parse_ir_from_string(
            r#"package neg_neg_pkg
            top fn neg_neg(x: bits[1]) -> bits[1] {
                neg.2: bits[1] = neg(x, id=2)
                ret neg.3: bits[1] = neg(neg.2, id=3)
            }
            "#,
        );
        let neg_neg_fn = neg_neg_pkg.get_top_fn().unwrap();
        let edit_distance = compute_edit_distance(&not_not_fn, &neg_neg_fn);
        assert_eq!(edit_distance, 2);
    }

    #[test]
    fn test_edit_distance_two_tuple_vs_singleton() {
        let two_tuple_pkg = parse_ir_from_string(
            r#"package two_tuple_pkg
            top fn two_tuple(x: bits[1]) -> (bits[1], bits[1]) {
                ret tuple.3: (bits[1], bits[1]) = tuple(x, x, id=3)
            }
            "#,
        );
        let two_tuple_fn = two_tuple_pkg.get_top_fn().unwrap();
        let singleton_pkg = parse_ir_from_string(
            r#"package singleton_pkg
            top fn singleton(x: bits[1]) -> (bits[1]) {
                ret tuple.2: (bits[1]) = tuple(x, id=2)
            }
            "#,
        );
        let singleton_fn = singleton_pkg.get_top_fn().unwrap();
        let edit_distance = compute_edit_distance(&two_tuple_fn, &singleton_fn);
        // The only difference is the operator in the node ("tuple" vs "identity").
        assert_eq!(edit_distance, 1);
    }

    #[test]
    fn test_rank_fn_candidates_by_similarity_prefilters_before_edit_distance() {
        let query_pkg = parse_ir_from_string(
            r#"package query_pkg
            top fn query(x: bits[8], y: bits[8]) -> bits[8] {
                add.3: bits[8] = add(x, y, id=3)
                ret sub.4: bits[8] = sub(add.3, y, id=4)
            }
            "#,
        );
        let exact_pkg = parse_ir_from_string(
            r#"package exact_pkg
            top fn exact(x: bits[8], y: bits[8]) -> bits[8] {
                add.3: bits[8] = add(x, y, id=3)
                ret sub.4: bits[8] = sub(add.3, y, id=4)
            }
            "#,
        );
        let close_pkg = parse_ir_from_string(
            r#"package close_pkg
            top fn close(x: bits[8], y: bits[8]) -> bits[8] {
                add.3: bits[8] = add(x, y, id=3)
                ret sub.4: bits[8] = sub(add.3, x, id=4)
            }
            "#,
        );
        let far_pkg = parse_ir_from_string(
            r#"package far_pkg
            top fn far(x: bits[8], y: bits[8]) -> bits[8] {
                umul.3: bits[8] = umul(x, y, id=3)
                ret not.4: bits[8] = not(umul.3, id=4)
            }
            "#,
        );

        let query = query_pkg.get_top_fn().unwrap();
        let exact = exact_pkg.get_top_fn().unwrap();
        let close = close_pkg.get_top_fn().unwrap();
        let far = far_pkg.get_top_fn().unwrap();
        let ranked = rank_fn_candidates_by_similarity(
            query,
            &[far, close, exact],
            CandidateRankingOptions { prefilter_limit: 2 },
        );

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].index, 2);
        assert_eq!(ranked[0].edit_distance, 0);
        assert_eq!(ranked[1].index, 1);
        assert_eq!(ranked[1].edit_distance, 1);
    }

    #[test]
    fn test_rank_fn_candidates_by_similarity_zero_limit() {
        let query_pkg = parse_ir_from_string(
            r#"package query_pkg
            top fn query(x: bits[1]) -> bits[1] {
                ret identity.2: bits[1] = identity(x, id=2)
            }
            "#,
        );
        let candidate_pkg = parse_ir_from_string(
            r#"package candidate_pkg
            top fn candidate(x: bits[1]) -> bits[1] {
                ret not.2: bits[1] = not(x, id=2)
            }
            "#,
        );
        let query = query_pkg.get_top_fn().unwrap();
        let candidate = candidate_pkg.get_top_fn().unwrap();

        let ranked = rank_fn_candidates_by_similarity(
            query,
            &[candidate],
            CandidateRankingOptions { prefilter_limit: 0 },
        );

        assert!(ranked.is_empty());
    }
}
