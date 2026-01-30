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
pub struct PlaceholderExpr {
    pub name: String,
    pub ty: Option<ir::Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryExpr {
    Placeholder(PlaceholderExpr),
    Number(u64),
    Numeric(NumericExpr),
    /// Variadic wildcard for matching n-ary operand lists in operator matchers.
    ///
    /// Used in query syntax as `...` inside an operator argument list, e.g.:
    /// - `nor(..., a, ...)` matches any-arity `nor` whose operands contain `a`.
    /// - `nor(a, ...)` matches any-arity `nor` whose first operand is `a`.
    /// - `nor(..., a)` matches any-arity `nor` whose last operand is `a`.
    Ellipsis,
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
    /// Numeric helper: yields the bit width of a bound placeholder node.
    ///
    /// This is not a node matcher; it is only valid in numeric named-arg
    /// contexts like `start=$width(x)` or `width=$width(x)`.
    Width,
    /// Helper matcher: matches a literal node that is all ones for its
    /// bit-width.
    AllOnes,
    /// Helper matcher: matches a literal node with low N bits set and upper
    /// bits clear.
    MaskLow,
    Msb,
    OpName(String),
    Literal {
        predicate: Option<LiteralPredicate>,
    },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericExpr {
    Number(u64),
    Width(PlaceholderExpr),
    Add(Box<NumericExpr>, Box<NumericExpr>),
    Sub(Box<NumericExpr>, Box<NumericExpr>),
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
    validate_ellipsis_placement(&expr)?;
    validate_width_matcher_placement(&expr)?;
    Ok(expr)
}

fn validate_ellipsis_placement(expr: &QueryExpr) -> Result<(), String> {
    fn walk(expr: &QueryExpr, ellipsis_allowed_here: bool) -> Result<(), String> {
        match expr {
            QueryExpr::Ellipsis => {
                if ellipsis_allowed_here {
                    Ok(())
                } else {
                    Err("ellipsis '...' is only valid inside an operator argument list".to_string())
                }
            }
            QueryExpr::Placeholder(_) | QueryExpr::Number(_) | QueryExpr::Numeric(_) => Ok(()),
            QueryExpr::Matcher(m) => {
                // Only explicit operator matchers (e.g. `nor(...)`) support ellipsis.
                let allow_in_args = matches!(m.kind, MatcherKind::OpName(_));
                for a in &m.args {
                    walk(a, allow_in_args)?;
                }

                // Named args usually never support ellipsis; they are not operand lists.
                //
                // Exception: select-like nodes expose `cases=[...]` which is explicitly an
                // operand list, and ellipsis provides useful "any arity" matching.
                for na in &m.named_args {
                    match &na.value {
                        NamedArgValue::Any | NamedArgValue::Bool(_) | NamedArgValue::Number(_) => {}
                        NamedArgValue::Expr(e) => walk(e, /* ellipsis_allowed_here= */ false)?,
                        NamedArgValue::ExprList(es) => {
                            let allow_in_list = na.name.as_str() == "cases";
                            for e in es {
                                walk(e, /* ellipsis_allowed_here= */ allow_in_list)?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }
    }

    walk(expr, /* ellipsis_allowed_here= */ false)
}

fn validate_width_matcher_placement(expr: &QueryExpr) -> Result<(), String> {
    fn validate_width_args(m: &MatcherExpr) -> Result<(), String> {
        // `$width(...)` is a numeric helper; its argument must be a placeholder
        // reference to an already-bound node.
        if m.args.len() != 1 {
            return Err("$width(...) expects exactly 1 argument".to_string());
        }
        match &m.args[0] {
            QueryExpr::Placeholder(p) => {
                if p.name == "_" {
                    return Err(
                        "$width(_) is not supported because '_' does not create a binding"
                            .to_string(),
                    );
                }
            }
            _ => {
                return Err(
                    "$width(...) expects a single placeholder identifier argument".to_string(),
                );
            }
        }
        Ok(())
    }

    fn validate_numeric_expr(expr: &NumericExpr) -> Result<(), String> {
        match expr {
            NumericExpr::Number(_) => Ok(()),
            NumericExpr::Width(placeholder) => {
                if placeholder.name == "_" {
                    return Err(
                        "$width(_) is not supported because '_' does not create a binding"
                            .to_string(),
                    );
                }
                Ok(())
            }
            NumericExpr::Add(lhs, rhs) | NumericExpr::Sub(lhs, rhs) => {
                validate_numeric_expr(lhs)?;
                validate_numeric_expr(rhs)?;
                Ok(())
            }
        }
    }

    fn walk(expr: &QueryExpr, width_allowed_here: bool) -> Result<(), String> {
        match expr {
            QueryExpr::Ellipsis | QueryExpr::Placeholder(_) | QueryExpr::Number(_) => Ok(()),
            QueryExpr::Numeric(expr) => {
                if width_allowed_here {
                    validate_numeric_expr(expr)?;
                }
                Ok(())
            }
            QueryExpr::Matcher(m) => {
                if matches!(m.kind, MatcherKind::Width) && !width_allowed_here {
                    return Err(
                        "$width(...) is only valid as a numeric expression for start=/width="
                            .to_string(),
                    );
                }

                if matches!(m.kind, MatcherKind::Width) {
                    validate_width_args(m)?;
                }

                if matches!(m.kind, MatcherKind::AllOnes) {
                    if !m.args.is_empty() {
                        return Err("$all_ones() expects 0 arguments".to_string());
                    }
                    return Ok(());
                }

                if matches!(m.kind, MatcherKind::MaskLow) {
                    if m.args.len() != 1 {
                        return Err("$mask_low(...) expects exactly 1 argument".to_string());
                    }
                    let QueryExpr::Numeric(expr) = &m.args[0] else {
                        return Err(
                            "$mask_low(...) expects a numeric expression argument".to_string()
                        );
                    };
                    validate_numeric_expr(expr)?;
                    return Ok(());
                }

                // Width is never allowed in positional operand lists; it doesn't match nodes.
                for a in &m.args {
                    walk(a, /* width_allowed_here= */ false)?;
                }

                // Allow width only in the numeric named args where we know how to interpret it.
                for na in &m.named_args {
                    match &na.value {
                        NamedArgValue::Any | NamedArgValue::Bool(_) | NamedArgValue::Number(_) => {}
                        NamedArgValue::Expr(e) => {
                            let allow = na.name.as_str() == "start" || na.name.as_str() == "width";
                            walk(e, /* width_allowed_here= */ allow)?;
                        }
                        NamedArgValue::ExprList(es) => {
                            for e in es {
                                walk(e, /* width_allowed_here= */ false)?;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }
    walk(expr, /* width_allowed_here= */ false)
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
        QueryExpr::Ellipsis => vec![],
        QueryExpr::Placeholder(placeholder) => {
            if let Some(ty) = &placeholder.ty {
                if f.get_node_ty(node_ref) != ty {
                    return vec![];
                }
            }

            if placeholder.name == "_" {
                // Wildcard: matches any node without creating/consulting a binding.
                return vec![bindings.clone()];
            }
            match bindings.get(&placeholder.name) {
                Some(existing) => match existing {
                    Binding::Node(existing_node_ref) if *existing_node_ref == node_ref => {
                        vec![bindings.clone()]
                    }
                    _ => vec![],
                },
                None => {
                    let mut out = bindings.clone();
                    out.insert(placeholder.name.clone(), Binding::Node(node_ref));
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
        QueryExpr::Numeric(_) => vec![],
        QueryExpr::Matcher(matcher) => {
            if matches!(matcher.kind, MatcherKind::AllOnes) {
                if !matcher.args.is_empty() {
                    return vec![];
                }
                let ir::NodePayload::Literal(value) = &f.get_node(node_ref).payload else {
                    return vec![];
                };
                let width = f.get_node(node_ref).ty.bit_count();
                if literal_is_all_ones(value, width) {
                    return vec![bindings.clone()];
                }
                return vec![];
            }
            if matches!(matcher.kind, MatcherKind::MaskLow) {
                if matcher.args.len() != 1 {
                    return vec![];
                }
                let QueryExpr::Numeric(expr) = &matcher.args[0] else {
                    return vec![];
                };
                let Some(low_bits) = eval_numeric_expr(expr, f, bindings) else {
                    return vec![];
                };
                let ir::NodePayload::Literal(value) = &f.get_node(node_ref).payload else {
                    return vec![];
                };
                if literal_is_mask_low(value, low_bits) {
                    return vec![bindings.clone()];
                }
                return vec![];
            }
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
            if let MatcherKind::Literal { predicate } = matcher.kind {
                let mut out = Vec::new();
                for b in match_named_args_solutions(
                    &matcher.named_args,
                    &node.payload,
                    f,
                    users,
                    bindings,
                ) {
                    out.extend(match_literal_solutions(
                        predicate,
                        &matcher.args,
                        &node.ty,
                        &node.payload,
                        &b,
                    ));
                }
                return out;
            }
            let operands = ir_utils::operands(&node.payload);
            // Important: match operands first (binding placeholders), then evaluate named
            // args. This allows named args like `start=$width(t)` to refer to placeholders
            // bound within the operand expressions.
            let mut out = Vec::new();
            let operand_bindings =
                match_args_solutions(&matcher.args, &operands, f, users, bindings);
            for b in operand_bindings {
                out.extend(match_named_args_solutions(
                    &matcher.named_args,
                    &node.payload,
                    f,
                    users,
                    &b,
                ));
            }
            out
        }
    }
}

fn match_literal_solutions(
    predicate: Option<LiteralPredicate>,
    args: &[QueryExpr],
    node_ty: &ir::Type,
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
        QueryExpr::Placeholder(placeholder) => {
            if let Some(ty) = &placeholder.ty {
                if node_ty != ty {
                    return vec![];
                }
            }

            if placeholder.name == "_" {
                // Wildcard literal argument: match any literal value without binding.
                return vec![bindings.clone()];
            }

            match bindings.get(&placeholder.name) {
                Some(existing) => match existing {
                    Binding::LiteralValue(existing_value) if *existing_value == *value => {
                        vec![bindings.clone()]
                    }
                    _ => vec![],
                },
                None => {
                    let mut out = bindings.clone();
                    out.insert(
                        placeholder.name.clone(),
                        Binding::LiteralValue(value.clone()),
                    );
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
        QueryExpr::Numeric(_) => vec![],
        QueryExpr::Ellipsis => vec![],
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

fn literal_is_all_ones(value: &IrValue, width: usize) -> bool {
    let Ok(bits) = value.to_bits() else {
        return false;
    };
    if bits.get_bit_count() != width {
        return false;
    }
    for i in 0..width {
        let Ok(bit) = bits.get_bit(i) else {
            return false;
        };
        if !bit {
            return false;
        }
    }
    true
}

fn literal_is_mask_low(value: &IrValue, low_bits: usize) -> bool {
    let Ok(bits) = value.to_bits() else {
        return false;
    };
    let bit_count = bits.get_bit_count();
    if low_bits > bit_count {
        return false;
    }
    for i in 0..low_bits {
        let Ok(bit) = bits.get_bit(i) else {
            return false;
        };
        if !bit {
            return false;
        }
    }
    for i in low_bits..bit_count {
        let Ok(bit) = bits.get_bit(i) else {
            return false;
        };
        if bit {
            return false;
        }
    }
    true
}

fn eval_numeric_expr(expr: &NumericExpr, f: &ir::Fn, bindings: &Bindings) -> Option<usize> {
    fn eval_inner(expr: &NumericExpr, f: &ir::Fn, bindings: &Bindings) -> Option<i64> {
        match expr {
            NumericExpr::Number(number) => i64::try_from(*number).ok(),
            NumericExpr::Width(placeholder) => {
                let node_ref = match bindings.get(&placeholder.name) {
                    Some(Binding::Node(nr)) => *nr,
                    _ => return None,
                };
                if let Some(ty) = &placeholder.ty {
                    if f.get_node_ty(node_ref) != ty {
                        return None;
                    }
                }
                i64::try_from(f.get_node_ty(node_ref).bit_count()).ok()
            }
            NumericExpr::Add(lhs, rhs) => {
                let lhs = eval_inner(lhs, f, bindings)?;
                let rhs = eval_inner(rhs, f, bindings)?;
                lhs.checked_add(rhs)
            }
            NumericExpr::Sub(lhs, rhs) => {
                let lhs = eval_inner(lhs, f, bindings)?;
                let rhs = eval_inner(rhs, f, bindings)?;
                lhs.checked_sub(rhs)
            }
        }
    }

    let value = eval_inner(expr, f, bindings)?;
    if value < 0 {
        return None;
    }
    usize::try_from(value).ok()
}

fn eval_query_numeric_expr(expr: &QueryExpr, f: &ir::Fn, bindings: &Bindings) -> Option<usize> {
    match expr {
        QueryExpr::Number(number) => usize::try_from(*number).ok(),
        QueryExpr::Numeric(expr) => eval_numeric_expr(expr, f, bindings),
        QueryExpr::Matcher(matcher) if matches!(matcher.kind, MatcherKind::Width) => {
            eval_width_expr(matcher, f, bindings)
        }
        _ => None,
    }
}

fn eval_width_expr(matcher: &MatcherExpr, f: &ir::Fn, bindings: &Bindings) -> Option<usize> {
    if matcher.args.len() != 1 {
        return None;
    }
    match &matcher.args[0] {
        QueryExpr::Placeholder(placeholder) => {
            if placeholder.name == "_" {
                return None;
            }
            match bindings.get(&placeholder.name) {
                Some(Binding::Node(nr)) => {
                    if let Some(ty) = &placeholder.ty {
                        if f.get_node_ty(*nr) != ty {
                            None
                        } else {
                            Some(f.get_node_ty(*nr).bit_count())
                        }
                    } else {
                        Some(f.get_node_ty(*nr).bit_count())
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
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
    if args.iter().any(|a| matches!(a, QueryExpr::Ellipsis)) {
        return match_args_with_ellipsis_solutions(args, operands, f, users, bindings);
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

fn match_args_with_ellipsis_solutions(
    pattern: &[QueryExpr],
    operands: &[ir::NodeRef],
    f: &ir::Fn,
    users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
    bindings: &Bindings,
) -> Vec<Bindings> {
    fn non_ellipsis_count(p: &[QueryExpr]) -> usize {
        p.iter()
            .filter(|e| !matches!(e, QueryExpr::Ellipsis))
            .count()
    }

    fn go(
        pattern: &[QueryExpr],
        operands: &[ir::NodeRef],
        f: &ir::Fn,
        users: &HashMap<ir::NodeRef, HashSet<ir::NodeRef>>,
        pi: usize,
        oi: usize,
        bindings: &Bindings,
    ) -> Vec<Bindings> {
        // Prune: must have enough operands left to match remaining non-ellipsis
        // pattern.
        let remaining_non_ellipsis = non_ellipsis_count(&pattern[pi..]);
        let remaining_operands = operands.len().saturating_sub(oi);
        if remaining_non_ellipsis > remaining_operands {
            return vec![];
        }

        if pi == pattern.len() {
            return if oi == operands.len() {
                vec![bindings.clone()]
            } else {
                vec![]
            };
        }

        match &pattern[pi] {
            QueryExpr::Ellipsis => {
                // Match any number of operands (including zero).
                let mut out = Vec::new();
                for k in oi..=operands.len() {
                    out.extend(go(pattern, operands, f, users, pi + 1, k, bindings));
                }
                out
            }
            other => {
                if oi >= operands.len() {
                    return vec![];
                }
                let mut out = Vec::new();
                let bs = match_solutions(other, f, users, operands[oi], bindings);
                for b in bs {
                    out.extend(go(pattern, operands, f, users, pi + 1, oi + 1, &b));
                }
                out
            }
        }
    }

    go(pattern, operands, f, users, 0, 0, bindings)
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
        MatcherKind::Width => false,
        MatcherKind::AllOnes => false,
        MatcherKind::MaskLow => false,
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
                ir::NodePayload::BitSlice { start, width, .. } => {
                    let actual: usize = match arg.name.as_str() {
                        "width" => *width,
                        "start" => *start,
                        _ => continue,
                    };

                    let expected: Option<usize> = match &arg.value {
                        NamedArgValue::Any => None,
                        NamedArgValue::Number(v) => Some(*v),
                        NamedArgValue::Expr(expr) => eval_query_numeric_expr(expr, f, b),
                        _ => None,
                    };

                    if let Some(v) = expected {
                        if v == actual {
                            matched.push(b.clone());
                        }
                    } else if matches!(arg.value, NamedArgValue::Any) {
                        matched.push(b.clone());
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
                if exprs.iter().any(|e| matches!(e, QueryExpr::Ellipsis)) {
                    match_args_solutions(exprs, cases, f, users, bindings)
                } else {
                    if exprs.len() != cases.len() {
                        return vec![];
                    }
                    match_args_solutions(exprs, cases, f, users, bindings)
                }
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

    #[test]
    fn parse_rejects_ellipsis_outside_operator_args() {
        // `...` cannot stand alone.
        assert!(parse_query("...").is_err());

        // `...` in non-operator matchers should be rejected (would otherwise
        // silently match nothing).
        assert!(parse_query("$users(...)").is_err());
        assert!(parse_query("literal(...)").is_err());

        // `...` in non-list named args should be rejected.
        assert!(parse_query("sel(selector=...)").is_err());
    }

    #[test]
    fn parse_allows_ellipsis_in_select_cases_list() {
        // `cases=[...]` is explicitly a variable-arity operand list.
        parse_query("sel(selector=s, cases=[...])").unwrap();
        parse_query("priority_sel(selector=s, cases=[..., a, ...], default=d)").unwrap();
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
            NamedArgValue::Expr(QueryExpr::Placeholder(p)) => assert_eq!(p.name, "true"),
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
    fn parse_rejects_wildcard_inside_width_matcher() {
        let err = parse_query("bit_slice(x, start=0, width=$width(_))").unwrap_err();
        assert_eq!(
            err,
            "$width(_) is not supported because '_' does not create a binding"
        );
    }

    #[test]
    fn parse_rejects_wrong_arity_for_all_ones_matcher() {
        let err = parse_query("$all_ones(x)").unwrap_err();
        assert!(
            err.contains("expects 0 arguments"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn parse_rejects_non_numeric_mask_low_arg() {
        let err = parse_query("$mask_low(x)").unwrap_err();
        assert!(
            err.contains("numeric expression"),
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
    fn find_matches_sel_with_cases_ellipsis_named_arg() {
        let pkg_text = r#"package test

fn main(s: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret out: bits[8] = sel(s, cases=[a, b], id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        // Should match regardless of the number of cases.
        let query = parse_query("sel(selector=s, cases=[...])").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "out");
    }

    #[test]
    fn find_matches_bit_slice_with_start_and_width_named_args() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[1] {
  slice0: bits[1] = bit_slice(x, start=0, width=1, id=2)
  ret slice3: bits[1] = bit_slice(x, start=3, width=1, id=3)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("bit_slice(x, start=3, width=1)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "slice3");
    }

    #[test]
    fn find_matches_bit_slice_with_start_width_matcher() {
        let pkg_text = r#"package test

fn main(t: bits[8] id=1) -> bits[1] {
  z: bits[1] = literal(value=0, id=2)
  widened: bits[9] = concat(z, t, id=3)
  ret slice: bits[1] = bit_slice(widened, start=8, width=1, id=4)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        // Match the carry bit by requiring start == $width(t), i.e. 8 for bits[8].
        let query = parse_query("bit_slice(concat(_, t), start=$width(t), width=1)").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "slice");
    }

    #[test]
    fn find_matches_all_ones_literal() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[1] {
  ones: bits[8] = literal(value=255, id=2)
  zeros: bits[8] = literal(value=0, id=3)
  cmp1: bits[1] = eq(x, ones, id=4)
  ret cmp0: bits[1] = eq(x, zeros, id=5)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("eq(x, $all_ones())").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp1");
    }

    #[test]
    fn find_matches_mask_low_literal() {
        let pkg_text = r#"package test

fn main(x: bits[8] id=1) -> bits[1] {
  mask4: bits[8] = literal(value=15, id=2)
  mask7: bits[8] = literal(value=127, id=3)
  cmp4: bits[1] = eq(x, mask4, id=4)
  ret cmp7: bits[1] = eq(x, mask7, id=5)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        let query = parse_query("eq(x, $mask_low(4))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp4");

        let query = parse_query("eq(x, $mask_low($width(x)-1))").unwrap();
        let matches = find_matching_nodes(f, &query);
        assert_eq!(matches.len(), 1);
        let node_id = ir::node_textual_id(f, matches[0]);
        assert_eq!(node_id, "cmp7");
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

    #[test]
    fn find_matches_variadic_ellipsis_in_nary_ops() {
        let pkg_text = r#"package test

top fn f(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[1] {
  nor.4: bits[1] = nor(b, a, c, id=4)
  and.5: bits[1] = and(a, nor.4, id=5)
  nor.6: bits[1] = nor(a, b, c, id=6)
  and.7: bits[1] = and(a, nor.6, id=7)
  nor.8: bits[1] = nor(b, c, a, id=8)
  ret and.9: bits[1] = and(a, nor.8, id=9)
}
"#;
        let mut parser = Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().expect("parse package");
        let f = pkg.get_top_fn().expect("top function");

        // Contains `a` anywhere in the NOR operands.
        let q_contains = parse_query("and(a, nor(..., a, ...))").unwrap();
        let matches = find_matching_nodes(f, &q_contains);
        let mut ids: Vec<String> = matches
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["and.5", "and.7", "and.9"]);

        // `nor(a, ...)` means `a` is the first NOR operand (and the same `a` as
        // the other AND operand).
        let q_first = parse_query("and(a, nor(a, ...))").unwrap();
        let ids: Vec<String> = find_matching_nodes(f, &q_first)
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        assert_eq!(ids, vec!["and.7"]);

        // `nor(..., a)` means `a` is the last NOR operand (and the same `a` as
        // the other AND operand).
        let q_last = parse_query("and(a, nor(..., a))").unwrap();
        let ids: Vec<String> = find_matching_nodes(f, &q_last)
            .into_iter()
            .map(|node_ref| ir::node_textual_id(f, node_ref))
            .collect();
        assert_eq!(ids, vec!["and.9"]);
    }
}
