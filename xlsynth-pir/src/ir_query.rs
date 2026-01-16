// SPDX-License-Identifier: Apache-2.0

//! Query parsing and matching helpers for simple PIR expression patterns.

use crate::ir;
use crate::ir_utils;
use std::collections::HashMap;
use std::collections::HashSet;
use xlsynth::IrBits;
use xlsynth::IrValue;

mod parser;

use self::parser::QueryParser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryExpr {
    Placeholder(String),
    Number(u64),
    Matcher(MatcherExpr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatcherExpr {
    pub kind: MatcherKind,
    pub user_count: Option<usize>,
    pub args: Vec<QueryExpr>,
    pub named_args: Vec<NamedArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatcherKind {
    AnyCmp,
    AnyMul,
    Users,
    Msb,
    OpName(String),
    Literal { predicate: Option<LiteralPredicate> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralPredicate {
    Pow2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedArg {
    pub name: String,
    pub value: NamedArgValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedArgValue {
    Bool(bool),
    Number(usize),
    Any,
    Expr(QueryExpr),
    ExprList(Vec<QueryExpr>),
}

impl MatcherKind {
    pub fn from_opname_and_predicate(
        opname: &str,
        predicate: Option<String>,
    ) -> Result<Self, String> {
        if opname == "literal" {
            let predicate = match predicate.as_deref() {
                None => None,
                Some("pow2") => Some(LiteralPredicate::Pow2),
                Some(other) => {
                    return Err(format!(
                        "unknown literal predicate [{}]; supported: [pow2]",
                        other
                    ));
                }
            };
            Ok(MatcherKind::Literal { predicate })
        } else if opname == "msb" {
            if let Some(pred) = predicate {
                Err(format!(
                    "unknown bracket clause [{}] for operator {}; only user-count constraints like [1u] are supported",
                    pred, opname
                ))
            } else {
                Ok(MatcherKind::Msb)
            }
        } else if let Some(pred) = predicate {
            Err(format!(
                "unknown bracket clause [{}] for operator {}; only user-count constraints like [1u] are supported",
                pred, opname
            ))
        } else {
            Ok(MatcherKind::OpName(opname.to_string()))
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum Binding {
    Node(ir::NodeRef),
    LiteralValue(IrValue),
}

type Bindings = HashMap<String, Binding>;

/// Returns the set of binding environments that satisfy `expr` at `node_ref`,
/// starting from `bindings`.
fn match_solutions(
    expr: &QueryExpr,
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
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
                Some(existing) => match existing {
                    Binding::Node(existing_node_ref) if *existing_node_ref == node_ref => {
                        vec![bindings.clone()]
                    }
                    _ => vec![],
                },
                None => {
                    let mut out = bindings.clone();
                    out.insert(name.clone(), Binding::Node(node_ref));
                    vec![out]
                }
            }
        }
        QueryExpr::Number(number) => match f.get_node(node_ref).payload {
            ir::NodePayload::Literal(ref value) => {
                if literal_matches_number(value, *number) {
                    vec![bindings.clone()]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        },
        QueryExpr::Matcher(matcher) => {
            if matches!(matcher.kind, MatcherKind::Users) {
                if matcher.args.len() != 1 {
                    return vec![];
                }
                let node = f.get_node(node_ref);
                let operands = ir_utils::operands(&node.payload);
                let mut out = Vec::new();
                for operand in operands {
                    out.extend(match_solutions(
                        &matcher.args[0],
                        f,
                        users,
                        operand,
                        bindings,
                    ));
                }
                return out;
            }
            let node = f.get_node(node_ref);
            if !matches_kind(&matcher.kind, &node.payload) {
                return vec![];
            }
            if matches!(matcher.kind, MatcherKind::Msb) && !matches_msb_slice(f, &node.payload) {
                return vec![];
            }
            if let Some(expected_users) = matcher.user_count {
                let actual_users = users.get(&node_ref).map(|set| set.len()).unwrap_or(0);
                if actual_users != expected_users {
                    return vec![];
                }
            }
            let named_arg_bindings =
                match_named_args_solutions(&matcher.named_args, &node.payload, f, users, bindings);
            if named_arg_bindings.is_empty() {
                return vec![];
            }
            if let MatcherKind::Literal { predicate } = matcher.kind {
                let mut out = Vec::new();
                for b in named_arg_bindings {
                    out.extend(match_literal_solutions(
                        predicate,
                        &matcher.args,
                        &node.payload,
                        &b,
                    ));
                }
                return out;
            }
            let operands = ir_utils::operands(&node.payload);
            let mut out = Vec::new();
            for b in named_arg_bindings {
                out.extend(match_args_solutions(&matcher.args, &operands, f, users, &b));
            }
            out
        }
    }
}

fn match_literal_solutions(
    predicate: Option<LiteralPredicate>,
    args: &[QueryExpr],
    payload: &ir::NodePayload,
    bindings: &Bindings,
) -> Vec<Bindings> {
    let ir::NodePayload::Literal(value) = payload else {
        return vec![];
    };

    if let Some(pred) = predicate {
        if !literal_satisfies_predicate(pred, value) {
            return vec![];
        }
    }

    if args.len() != 1 {
        return vec![];
    }

    match &args[0] {
        QueryExpr::Placeholder(name) => {
            if name == "_" {
                // Wildcard literal argument: match any literal value without binding.
                return vec![bindings.clone()];
            }

            match bindings.get(name) {
                Some(existing) => match existing {
                    Binding::LiteralValue(existing_value) if *existing_value == *value => {
                        vec![bindings.clone()]
                    }
                    _ => vec![],
                },
                None => {
                    let mut out = bindings.clone();
                    out.insert(name.clone(), Binding::LiteralValue(value.clone()));
                    vec![out]
                }
            }
        }
        QueryExpr::Number(number) => {
            if literal_matches_number(value, *number) {
                vec![bindings.clone()]
            } else {
                vec![]
            }
        }
        QueryExpr::Matcher(_) => vec![],
    }
}

fn literal_satisfies_predicate(pred: LiteralPredicate, value: &IrValue) -> bool {
    match pred {
        LiteralPredicate::Pow2 => {
            // Strict power-of-two: exactly one bit set; zero does not match.
            let Ok(bits) = value.to_bits() else {
                return false;
            };
            let mut set_bits: usize = 0;
            for i in 0..bits.get_bit_count() {
                let Ok(bit) = bits.get_bit(i) else {
                    return false;
                };
                if bit {
                    set_bits += 1;
                    if set_bits > 1 {
                        return false;
                    }
                }
            }
            set_bits == 1
        }
    }
}

fn literal_matches_number(value: &IrValue, number: u64) -> bool {
    let Ok(bits) = value.to_bits() else {
        return false;
    };
    let bit_count = bits.get_bit_count();

    if bit_count == 0 {
        return number == 0;
    }

    if number == 0 {
        for i in 0..bit_count {
            let Ok(bit) = bits.get_bit(i) else {
                return false;
            };
            if bit {
                return false;
            }
        }
        return true;
    }

    // Query numbers are widthless; treat them as an unsigned numeric value and
    // match literals whose upper bits are all zero beyond the value's u64
    // representation.
    if bit_count < 64 && (number >> bit_count) != 0 {
        return false;
    }

    let expected = IrBits::make_ubits(bit_count, number).unwrap();
    bits.equals(&expected)
}

/// Matches the query arguments against a node's operands.
fn match_args_solutions(
    args: &[QueryExpr],
    operands: &[ir::NodeRef],
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
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
        MatcherKind::Users => false,
        MatcherKind::Msb => matches!(payload, ir::NodePayload::BitSlice { .. }),
        MatcherKind::OpName(opname) => payload.get_operator() == opname,
        MatcherKind::Literal { .. } => matches!(payload, ir::NodePayload::Literal(_)),
    }
}

fn matches_msb_slice(f: &ir::Fn, payload: &ir::NodePayload) -> bool {
    match payload {
        ir::NodePayload::BitSlice { arg, start, width } => {
            if *width != 1 {
                return false;
            }
            let arg_bits = f.get_node(*arg).ty.bit_count();
            if arg_bits == 0 {
                return false;
            }
            *start == arg_bits - 1
        }
        _ => false,
    }
}

fn match_named_args_solutions(
    named_args: &[NamedArg],
    payload: &ir::NodePayload,
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
    bindings: &Bindings,
) -> Vec<Bindings> {
    if named_args.is_empty() {
        return vec![bindings.clone()];
    }
    let mut partials = vec![bindings.clone()];
    for arg in named_args {
        let mut next = Vec::new();
        for b in &partials {
            let mut matched = Vec::new();
            match payload {
                ir::NodePayload::BitSlice { width, .. } => {
                    if arg.name.as_str() != "width" {
                        continue;
                    }
                    match &arg.value {
                        NamedArgValue::Number(v) => {
                            if *v == *width {
                                matched.push(b.clone());
                            }
                        }
                        NamedArgValue::Any => matched.push(b.clone()),
                        _ => {}
                    }
                }
                ir::NodePayload::OneHot { lsb_prio, .. } => {
                    if arg.name.as_str() != "lsb_prio" {
                        continue;
                    }
                    match &arg.value {
                        NamedArgValue::Bool(v) => {
                            if *v == *lsb_prio {
                                matched.push(b.clone());
                            }
                        }
                        NamedArgValue::Any => matched.push(b.clone()),
                        _ => {}
                    }
                }
                ir::NodePayload::PrioritySel {
                    selector,
                    cases,
                    default,
                } => {
                    matched = match_select_named_arg(arg, *selector, cases, default, f, users, b);
                }
                ir::NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    matched = match_select_named_arg(arg, *selector, cases, default, f, users, b);
                }
                ir::NodePayload::OneHotSel { selector, cases } => {
                    matched = match_select_named_arg(arg, *selector, cases, &None, f, users, b);
                }
                _ => {}
            }
            next.extend(matched);
        }
        if next.is_empty() {
            return vec![];
        }
        partials = next;
    }
    partials
}

fn match_select_named_arg(
    arg: &NamedArg,
    selector: ir::NodeRef,
    cases: &[ir::NodeRef],
    default: &Option<ir::NodeRef>,
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
    bindings: &Bindings,
) -> Vec<Bindings> {
    match arg.name.as_str() {
        "selector" => match &arg.value {
            NamedArgValue::Any => vec![bindings.clone()],
            NamedArgValue::Expr(expr) => match_solutions(expr, f, users, selector, bindings),
            _ => vec![],
        },
        "cases" => match &arg.value {
            NamedArgValue::Any => vec![bindings.clone()],
            NamedArgValue::Expr(expr) => {
                match_args_solutions(&[expr.clone()], cases, f, users, bindings)
            }
            NamedArgValue::ExprList(exprs) => {
                if exprs.len() != cases.len() {
                    return vec![];
                }
                match_args_solutions(exprs, cases, f, users, bindings)
            }
            _ => vec![],
        },
        "default" => match &arg.value {
            NamedArgValue::Any => match default {
                Some(_) => vec![bindings.clone()],
                None => vec![],
            },
            NamedArgValue::Expr(expr) => match default {
                Some(node_ref) => match_solutions(expr, f, users, *node_ref, bindings),
                None => vec![],
            },
            _ => vec![],
        },
        _ => vec![],
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
        assert!(matcher.named_args.is_empty());
    }

    #[test]
    fn parse_users_query() {
        let query = parse_query("$users(encode(one_hot(x)))").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        assert_eq!(matcher.kind, MatcherKind::Users);
        assert_eq!(matcher.args.len(), 1);
    }

    /// Verifies the parser accepts concrete operator matchers like
    /// `sub(add(...), ...)` as well as literal matchers with value binders
    /// like `literal(L)`.
    #[test]
    fn parse_operator_and_literal_query() {
        let query = parse_query("sub(add(x, literal(L)), literal(L))").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        assert_eq!(matcher.kind, MatcherKind::OpName("sub".to_string()));
        assert_eq!(matcher.args.len(), 2);
        assert!(matcher.named_args.is_empty());
    }

    #[test]
    fn parse_priority_sel_with_cases_named_args() {
        let query = parse_query("priority_sel(selector=s, cases=[a, b], default=d)").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        assert_eq!(
            matcher.kind,
            MatcherKind::OpName("priority_sel".to_string())
        );
        assert_eq!(matcher.named_args.len(), 3);
        assert_eq!(matcher.named_args[0].name, "selector");
        assert_eq!(matcher.named_args[1].name, "cases");
        assert_eq!(matcher.named_args[2].name, "default");
        match &matcher.named_args[1].value {
            NamedArgValue::ExprList(exprs) => assert_eq!(exprs.len(), 2),
            other => panic!("expected cases list, got {:?}", other),
        }
    }

    #[test]
    fn parse_named_arg_true_is_expr_outside_lsb_prio() {
        let query = parse_query("priority_sel(selector=true, cases=[a], default=d)").unwrap();
        let QueryExpr::Matcher(matcher) = query else {
            panic!("expected matcher");
        };
        match &matcher.named_args[0].value {
            NamedArgValue::Expr(QueryExpr::Placeholder(name)) => assert_eq!(name, "true"),
            other => panic!("expected selector expr placeholder, got {:?}", other),
        }
    }

    #[test]
    fn parse_rejects_wrong_arity_for_literal_matcher() {
        let err = parse_query("literal()").unwrap_err();
        assert!(
            err.contains("literal expects 1 argument"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn find_matches_one_hot_with_wildcard_lsb_prio_named_arg() {
        let pkg_text = r#"package test

fn main(x: bits[4] id=1) -> bits[5] {
  oh_t: bits[5] = one_hot(x, lsb_prio=true, id=2)
  oh_f: bits[5] = one_hot(x, lsb_prio=false, id=3)
  ret out: bits[5] = xor(oh_t, oh_f, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("one_hot(x, lsb_prio=_)").unwrap();
        let matches = find_matching_nodes(f, &query);
        let mut ids: Vec<String> = matches
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["oh_f", "oh_t"]);
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

    /// Verifies `literal(L)` binds literal *values* (not node identity), so two
    /// distinct literal nodes with the same value can satisfy a shared binder.
    #[test]
    fn find_matches_with_literal_value_binding() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[8] {
  literal.2: bits[8] = literal(value=5, id=2)
  literal.3: bits[8] = literal(value=5, id=3)
  add.4: bits[8] = add(x, literal.2, id=4)
  ret out: bits[8] = sub(add.4, literal.3, id=5)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");
        let query = parse_query("sub(add(x, literal(L)), literal(L))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "out");
    }

    #[test]
    fn find_matches_one_hot_encode_zero_pattern() {
        let pkg_text = r#"package test

fn main(x: bits[4] id=1) -> bits[1] {
  rev: bits[4] = reverse(x, id=2)
  oh: bits[5] = one_hot(rev, lsb_prio=true, id=3)
  enc: bits[3] = encode(oh, id=4)
  zero: bits[3] = literal(value=0, id=5)
  ret out: bits[1] = eq(enc, zero, id=6)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");
        let query =
            parse_query("eq(encode(one_hot(reverse(x), lsb_prio=true)), literal(0))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "out");
    }

    /// Verifies `literal[pow2](L)` enforces a strict power-of-two constraint:
    /// exactly one bit set (so `0` does not match).
    #[test]
    fn find_matches_anycmp_with_literal_pow2_predicate() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[1] {
  pow2: bits[8] = literal(value=8, id=2)
  non: bits[8] = literal(value=6, id=3)
  cmp_pow2: bits[1] = eq(x, pow2, id=4)
  ret cmp_non: bits[1] = eq(x, non, id=5)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");
        let query = parse_query("$anycmp(x, literal[pow2](L))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp_pow2");
    }

    #[test]
    fn find_matches_anycmp_priority_sel() {
        let pkg_text = r#"package test

fn main(sel: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[1] {
  prio: bits[8] = priority_sel(sel, cases=[a, b], default=d, id=5)
  ret out: bits[1] = eq(prio, a, id=6)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("$anycmp(priority_sel(), _)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "out");
    }

    #[test]
    fn find_matches_priority_sel_with_numeric_named_args() {
        let pkg_text = r#"package test

fn main(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  sel_lit: bits[2] = literal(value=1, id=3)
  def_lit: bits[8] = literal(value=0, id=4)
  prio: bits[8] = priority_sel(sel_lit, cases=[a, b], default=def_lit, id=5)
  ret out: bits[8] = identity(prio, id=6)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("priority_sel(selector=1, cases=[_, _], default=0)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "prio");
    }

    #[test]
    fn find_matches_priority_sel_requires_case_count() {
        let pkg_text = r#"package test

fn main(sel: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  prio: bits[8] = priority_sel(sel, cases=[a, b], default=d, id=5)
  ret out: bits[8] = identity(prio, id=6)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("priority_sel(cases=[])").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert!(matches.is_empty());

        let query = parse_query("priority_sel(cases=[a])").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert!(matches.is_empty());
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
    fn find_matches_msb_shorthand() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[1] {
  neg.2: bits[8] = neg(x, id=2)
  ret msb: bits[1] = bit_slice(neg.2, start=7, width=1, id=3)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("msb(neg(x))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "msb");
    }

    #[test]
    fn find_matches_gate_operator() {
        let pkg_text = r#"package test

fn main(pred: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  gated: bits[8] = gate(pred, x, id=3)
  ret out: bits[8] = add(gated, x, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("gate(pred, x)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "gated");
    }

    #[test]
    fn find_matches_bit_slice_with_width_named_arg() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[3] {
  slice1: bits[1] = bit_slice(x, start=3, width=1, id=2)
  slice2: bits[2] = bit_slice(x, start=4, width=2, id=3)
  ret out: bits[3] = concat(slice2, slice1, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("bit_slice(x, width=1)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "slice1");
    }

    #[test]
    fn find_matches_bit_slice_with_width_wildcard() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[3] {
  slice1: bits[1] = bit_slice(x, start=3, width=1, id=2)
  slice2: bits[2] = bit_slice(x, start=4, width=2, id=3)
  ret out: bits[3] = concat(slice2, slice1, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("bit_slice(x, width=_)").unwrap();
        let matches = find_matching_nodes(f, &query);
        let mut ids: Vec<String> = matches
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["slice1", "slice2"]);
    }

    #[test]
    fn find_matches_users_of_encode_one_hot() {
        let pkg_text = r#"package test

fn main(x: bits[4] id=1) -> bits[1] {
  oh: bits[5] = one_hot(x, lsb_prio=true, id=2)
  enc: bits[3] = encode(oh, id=3)
  lit0: bits[3] = literal(value=0, id=4)
  lit1: bits[3] = literal(value=1, id=5)
  cmp0: bits[1] = eq(enc, lit0, id=6)
  cmp1: bits[1] = ne(enc, lit1, id=7)
  ret out: bits[1] = or(cmp0, cmp1, id=8)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("$users(encode(one_hot(x, lsb_prio=true)))").unwrap();
        let matches = find_matching_nodes(f, &query);
        let mut ids: Vec<String> = matches
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["cmp0", "cmp1"]);
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

    #[test]
    fn parse_rejects_unicode_without_panicking() {
        // NOTE: This is intentionally a Unicode whitespace (NBSP). The parser
        // operates on byte offsets and should return an error, not panic due to
        // slicing at invalid UTF-8 boundaries.
        let query = format!("$anycmp({}x, _)", '\u{00A0}');
        let err = parse_query(&query).unwrap_err();
        assert!(
            err.contains("expected placeholder"),
            "unexpected error: {}",
            err
        );
    }
}
