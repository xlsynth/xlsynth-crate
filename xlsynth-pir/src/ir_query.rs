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
        let bindings = HashMap::new();
        if !match_solutions(query, f, &users, node_ref, &bindings).is_empty() {
            matches.push(node_ref);
        }
    }
    matches
}

type Bindings = HashMap<String, ir::NodeRef>;

/// Returns the set of binding environments that satisfy `expr` at `node_ref`,
/// starting from `bindings`.
fn match_solutions(
    expr: &QueryExpr,
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, std::collections::HashSet<ir::NodeRef>>,
    node_ref: ir::NodeRef,
    bindings: &Bindings,
) -> Vec<Bindings> {
    match expr {
        QueryExpr::Placeholder(name) => {
            if name == "_" {
                // Wildcard: matches any node without creating/consulting a binding.
                return vec![bindings.clone()];
            }
            match bindings.get(name) {
                Some(existing) => {
                    if *existing == node_ref {
                        vec![bindings.clone()]
                    } else {
                        vec![]
                    }
                }
                None => {
                    let mut out = bindings.clone();
                    out.insert(name.clone(), node_ref);
                    vec![out]
                }
            }
        }
        QueryExpr::Matcher(matcher) => {
            let node = f.get_node(node_ref);
            if !matches_kind(&matcher.kind, &node.payload) {
                return vec![];
            }
            if let Some(expected_users) = matcher.user_count {
                let actual_users = users.get(&node_ref).map(|set| set.len()).unwrap_or(0);
                if actual_users != expected_users {
                    return vec![];
                }
            }
            let operands = ir_utils::operands(&node.payload);
            match_args_solutions(&matcher.args, &operands, f, users, bindings)
        }
    }
}

/// Matches the query arguments against a node's operands.
fn match_args_solutions(
    args: &[QueryExpr],
    operands: &[ir::NodeRef],
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, std::collections::HashSet<ir::NodeRef>>,
    bindings: &Bindings,
) -> Vec<Bindings> {
    if args.is_empty() {
        return vec![bindings.clone()];
    }
    if args.len() == 1 {
        let mut out = Vec::new();
        for operand in operands {
            out.extend(match_solutions(&args[0], f, users, *operand, bindings));
        }
        return out;
    }
    if args.len() != operands.len() {
        return vec![];
    }
    let mut partials = vec![bindings.clone()];
    for (arg, operand) in args.iter().zip(operands.iter()) {
        let mut next = Vec::new();
        for b in &partials {
            next.extend(match_solutions(arg, f, users, *operand, b));
        }
        if next.is_empty() {
            return vec![];
        }
        partials = next;
    }
    partials
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
        let query = parse_query("$anycmp($anymul[1u](x, y), _)").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        assert_eq!(matcher.kind, MatcherKind::AnyCmp);
        assert_eq!(matcher.args.len(), 2);
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
        let query = parse_query("$anycmp($anymul[1u](x, y), _)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp");
    }

    #[test]
    fn single_arg_matcher_backtracks_shared_placeholder() {
        let pkg_text = r#"package test

fn main(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  c: bits[1] = eq(a, b, id=3)
  ret m: bits[1] = umul(c, b, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        // This should match `m` by binding `x` to `b` (not greedily to `a`).
        let query = parse_query("$anymul($anycmp(_, x), x)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "m");
    }

    #[test]
    fn wildcard_placeholder_underscore_matches_without_binding() {
        let pkg_text = r#"package test

fn main(a: bits[8] id=1, b: bits[8] id=2) -> bits[1] {
  ret c: bits[1] = eq(a, b, id=3)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("$anycmp(a, _)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "c");
    }

    #[test]
    fn parse_rejects_wrong_arity_for_binary_matchers() {
        let err = parse_query("$anycmp(x)").unwrap_err();
        assert!(
            err.contains("expects 2 arguments"),
            "unexpected error: {}",
            err
        );
        let err = parse_query("$anymul(x)").unwrap_err();
        assert!(
            err.contains("expects 2 arguments"),
            "unexpected error: {}",
            err
        );
    }
}
