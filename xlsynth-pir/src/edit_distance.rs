// SPDX-License-Identifier: Apache-2.0

//! Computes a graph edit distance between two XLS IR functions.

use crate::ir::{Fn, Node, NodePayload};
use crate::ir::{binop_to_operator, nary_op_to_operator, unop_to_operator};

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeSignature {
    WithOperands {
        op: String,
        ty: String,
        operands: Vec<String>,
    },
    Simple(String),
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
}
