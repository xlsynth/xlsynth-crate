// SPDX-License-Identifier: Apache-2.0

//! Query parsing and matching helpers for simple PIR expression patterns.

use crate::ir;
use crate::ir_utils;
use std::collections::HashMap;

mod parser;

use self::parser::QueryParser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryExpr {
    Placeholder(String),
    Matcher(MatcherExpr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatcherExpr {
    pub kind: MatcherKind,
    pub user_count: Option<usize>,
    pub args: Vec<QueryExpr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatcherKind {
    AnyCmp,
    AnyMul,
}

/// Parses a query expression string into an AST.
pub fn parse_query(input: &str) -> Result<QueryExpr, String> {
    let mut parser = QueryParser::new(input);
    let expr = parser.parse_expr()?;
    parser.skip_ws();
    if !parser.is_done() {
        return Err(parser.error_at("unexpected trailing input"));
    }
    Ok(expr)
}

/// Finds all node references in `f` that satisfy the query expression.
pub fn find_matching_nodes(f: &ir::Fn, query: &QueryExpr) -> Vec<ir::NodeRef> {
    let users = ir_utils::compute_users(f);
    let mut matches = Vec::new();
    for (index, _node) in f.nodes.iter().enumerate() {
        let node_ref = ir::NodeRef { index };
        let mut bindings = HashMap::new();
        if matches_expr(query, f, &users, node_ref, &mut bindings) {
            matches.push(node_ref);
        }
    }
    matches
}

/// Matches `expr` against a concrete node, extending placeholder bindings as
/// needed.
fn matches_expr(
    expr: &QueryExpr,
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, std::collections::HashSet<ir::NodeRef>>,
    node_ref: ir::NodeRef,
    bindings: &mut HashMap<String, ir::NodeRef>,
) -> bool {
    match expr {
        QueryExpr::Placeholder(name) => match bindings.get(name) {
            Some(existing) => *existing == node_ref,
            None => {
                bindings.insert(name.clone(), node_ref);
                true
            }
        },
        QueryExpr::Matcher(matcher) => {
            let node = f.get_node(node_ref);
            if !matches_kind(&matcher.kind, &node.payload) {
                return false;
            }
            if let Some(expected_users) = matcher.user_count {
                let actual_users = users.get(&node_ref).map(|set| set.len()).unwrap_or(0);
                if actual_users != expected_users {
                    return false;
                }
            }
            let operands = ir_utils::operands(&node.payload);
            match_args(&matcher.args, &operands, f, users, bindings)
        }
    }
}

/// Matches the query arguments against a node's operands.
fn match_args(
    args: &[QueryExpr],
    operands: &[ir::NodeRef],
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, std::collections::HashSet<ir::NodeRef>>,
    bindings: &mut HashMap<String, ir::NodeRef>,
) -> bool {
    if args.is_empty() {
        return true;
    }
    if args.len() == 1 {
        for operand in operands {
            let mut local_bindings = bindings.clone();
            if matches_expr(&args[0], f, users, *operand, &mut local_bindings) {
                *bindings = local_bindings;
                return true;
            }
        }
        return false;
    }
    if args.len() != operands.len() {
        return false;
    }
    for (arg, operand) in args.iter().zip(operands.iter()) {
        if !matches_expr(arg, f, users, *operand, bindings) {
            return false;
        }
    }
    true
}

/// Checks whether the node payload satisfies a matcher kind.
fn matches_kind(kind: &MatcherKind, payload: &ir::NodePayload) -> bool {
    match kind {
        MatcherKind::AnyCmp => match payload {
            ir::NodePayload::Binop(op, _, _) => matches!(
                op,
                ir::Binop::Eq
                    | ir::Binop::Ne
                    | ir::Binop::Uge
                    | ir::Binop::Ugt
                    | ir::Binop::Ult
                    | ir::Binop::Ule
                    | ir::Binop::Sgt
                    | ir::Binop::Sge
                    | ir::Binop::Slt
                    | ir::Binop::Sle
            ),
            _ => false,
        },
        MatcherKind::AnyMul => match payload {
            ir::NodePayload::Binop(op, _, _) => matches!(
                op,
                ir::Binop::Umul | ir::Binop::Smul | ir::Binop::Umulp | ir::Binop::Smulp
            ),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;
    use crate::ir_parser::Parser;

    #[test]
    fn parse_basic_query() {
        let query = parse_query("$anycmp($anymul[1u](x, y))").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        assert_eq!(matcher.kind, MatcherKind::AnyCmp);
        assert_eq!(matcher.args.len(), 1);
    }

    #[test]
    fn find_matches_with_user_constraint() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  m: bits[8] = umul(x, y, id=3)
  ret cmp: bits[1] = eq(m, x, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");
        let query = parse_query("$anycmp($anymul[1u](x, y))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp");
    }
}
