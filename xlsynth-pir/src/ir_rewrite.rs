// SPDX-License-Identifier: Apache-2.0

//! Lightweight library support for matching and rewriting PIR functions.

use crate::dce;
use crate::ir::{self, NodePayload, NodeRef, Type};
use crate::ir_deduce;
use crate::ir_query;
use crate::ir_utils;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use xlsynth::IrValue;

#[derive(Debug, Clone)]
pub struct MatchPattern {
    text: String,
    query: ir_query::QueryExpr,
    node_binders: BTreeSet<String>,
    literal_binders: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct RewriteTemplate {
    text: String,
    expr: TemplateExpr,
    placeholder_refs: BTreeSet<String>,
    width_refs: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct MatchRewriteRule {
    match_pattern: MatchPattern,
    rewrite_template: RewriteTemplate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RewriteApplyMode {
    FirstMatch,
}

#[derive(Debug, Clone)]
pub struct MatchRewriteOutcome {
    rewritten_fn: ir::Fn,
    rewrote: bool,
    matched_root: Option<ir::NodeRef>,
}

impl MatchRewriteOutcome {
    pub fn rewritten_fn(&self) -> &ir::Fn {
        &self.rewritten_fn
    }

    pub fn into_rewritten_fn(self) -> ir::Fn {
        self.rewritten_fn
    }

    pub fn rewrote(&self) -> bool {
        self.rewrote
    }

    pub fn matched_root(&self) -> Option<ir::NodeRef> {
        self.matched_root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchPatternParseError {
    ParseError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteTemplateParseError {
    ParseError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchRewriteRuleBuildError {
    MatchPatternParse(MatchPatternParseError),
    RewriteTemplateParse(RewriteTemplateParseError),
    MixedBindingKind { name: String },
    UnboundRewritePlaceholder { name: String },
    WidthRequiresNodeBinding { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchRewriteRuleApplyError {
    AmbiguousMatch {
        root: ir::NodeRef,
        match_count: usize,
    },
    NumericEvaluation(String),
    UnsupportedRewriteOperator(String),
    InvalidRewriteTemplate(String),
    TypeDeduction(String),
    Validation(String),
    Toposort(String),
    LiteralConstruction(String),
}

impl fmt::Display for MatchPatternParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchPatternParseError::ParseError(msg) => write!(f, "{}", msg),
        }
    }
}

impl fmt::Display for RewriteTemplateParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewriteTemplateParseError::ParseError(msg) => write!(f, "{}", msg),
        }
    }
}

impl fmt::Display for MatchRewriteRuleBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchRewriteRuleBuildError::MatchPatternParse(err) => {
                write!(f, "match pattern parse failed: {}", err)
            }
            MatchRewriteRuleBuildError::RewriteTemplateParse(err) => {
                write!(f, "rewrite template parse failed: {}", err)
            }
            MatchRewriteRuleBuildError::MixedBindingKind { name } => {
                write!(
                    f,
                    "match pattern binder '{}' is used as both a node binder and a literal-value binder",
                    name
                )
            }
            MatchRewriteRuleBuildError::UnboundRewritePlaceholder { name } => {
                write!(
                    f,
                    "rewrite template placeholder '{}' is not bound by the match pattern",
                    name
                )
            }
            MatchRewriteRuleBuildError::WidthRequiresNodeBinding { name } => {
                write!(
                    f,
                    "$width({}) requires a node binder from the match pattern",
                    name
                )
            }
        }
    }
}

impl fmt::Display for MatchRewriteRuleApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchRewriteRuleApplyError::AmbiguousMatch { root, match_count } => write!(
                f,
                "root node {:?} matched {} distinct binding environments",
                root, match_count
            ),
            MatchRewriteRuleApplyError::NumericEvaluation(msg)
            | MatchRewriteRuleApplyError::UnsupportedRewriteOperator(msg)
            | MatchRewriteRuleApplyError::InvalidRewriteTemplate(msg)
            | MatchRewriteRuleApplyError::TypeDeduction(msg)
            | MatchRewriteRuleApplyError::Validation(msg)
            | MatchRewriteRuleApplyError::Toposort(msg)
            | MatchRewriteRuleApplyError::LiteralConstruction(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for MatchPatternParseError {}
impl std::error::Error for RewriteTemplateParseError {}
impl std::error::Error for MatchRewriteRuleBuildError {}
impl std::error::Error for MatchRewriteRuleApplyError {}

impl MatchPattern {
    pub fn parse(text: &str) -> Result<Self, MatchPatternParseError> {
        let query = ir_query::parse_query(text).map_err(MatchPatternParseError::ParseError)?;
        let mut node_binders = BTreeSet::new();
        let mut literal_binders = BTreeSet::new();
        collect_match_binders(&query, &mut node_binders, &mut literal_binders, false);
        Ok(Self {
            text: text.to_string(),
            query,
            node_binders,
            literal_binders,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl RewriteTemplate {
    pub fn parse(text: &str) -> Result<Self, RewriteTemplateParseError> {
        let mut parser = RewriteTemplateParser::new(text);
        let expr = parser
            .parse_expr()
            .map_err(RewriteTemplateParseError::ParseError)?;
        parser.skip_ws();
        if !parser.is_done() {
            return Err(RewriteTemplateParseError::ParseError(
                parser.error("unexpected trailing input"),
            ));
        }
        let mut placeholder_refs = BTreeSet::new();
        let mut width_refs = BTreeSet::new();
        collect_rewrite_refs(&expr, &mut placeholder_refs, &mut width_refs);
        Ok(Self {
            text: text.to_string(),
            expr,
            placeholder_refs,
            width_refs,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl MatchRewriteRule {
    pub fn new(
        match_pattern: MatchPattern,
        rewrite_template: RewriteTemplate,
    ) -> Result<Self, MatchRewriteRuleBuildError> {
        if let Some(name) = match_pattern
            .node_binders
            .intersection(&match_pattern.literal_binders)
            .next()
        {
            return Err(MatchRewriteRuleBuildError::MixedBindingKind { name: name.clone() });
        }

        for name in &rewrite_template.placeholder_refs {
            if !match_pattern.node_binders.contains(name)
                && !match_pattern.literal_binders.contains(name)
            {
                return Err(MatchRewriteRuleBuildError::UnboundRewritePlaceholder {
                    name: name.clone(),
                });
            }
        }

        for name in &rewrite_template.width_refs {
            if !match_pattern.node_binders.contains(name) {
                return Err(MatchRewriteRuleBuildError::WidthRequiresNodeBinding {
                    name: name.clone(),
                });
            }
        }

        Ok(Self {
            match_pattern,
            rewrite_template,
        })
    }

    pub fn from_strings(
        match_str: &str,
        rewrite_str: &str,
    ) -> Result<Self, MatchRewriteRuleBuildError> {
        let match_pattern = MatchPattern::parse(match_str)
            .map_err(MatchRewriteRuleBuildError::MatchPatternParse)?;
        let rewrite_template = RewriteTemplate::parse(rewrite_str)
            .map_err(MatchRewriteRuleBuildError::RewriteTemplateParse)?;
        Self::new(match_pattern, rewrite_template)
    }

    pub fn apply_to_fn(
        &self,
        f: &ir::Fn,
        mode: RewriteApplyMode,
    ) -> Result<MatchRewriteOutcome, MatchRewriteRuleApplyError> {
        match mode {
            RewriteApplyMode::FirstMatch => self.apply_first_match(f),
        }
    }

    fn apply_first_match(
        &self,
        f: &ir::Fn,
    ) -> Result<MatchRewriteOutcome, MatchRewriteRuleApplyError> {
        let mut rewritten = f.clone();
        let mut ret_reachable = vec![true; rewritten.nodes.len()];
        for dead_node in dce::get_dead_nodes(&rewritten) {
            ret_reachable[dead_node.index] = false;
        }
        let users = ir_utils::compute_users(&rewritten);
        for node_ref in rewritten.node_refs() {
            if !ret_reachable[node_ref.index] {
                continue;
            }
            if matches!(
                rewritten.get_node(node_ref).payload,
                NodePayload::Nil | NodePayload::GetParam(_)
            ) {
                continue;
            }

            let raw_bindings = ir_query::find_root_query_bindings(
                &self.match_pattern.query,
                &rewritten,
                &users,
                node_ref,
            );
            let unique_bindings = dedup_query_bindings(raw_bindings);
            if unique_bindings.is_empty() {
                continue;
            }
            if unique_bindings.len() > 1 {
                return Err(MatchRewriteRuleApplyError::AmbiguousMatch {
                    root: node_ref,
                    match_count: unique_bindings.len(),
                });
            }

            let bindings = unique_bindings
                .into_iter()
                .next()
                .expect("expected one binding");
            let rewritten_root = apply_rule_at_root(
                &mut rewritten,
                node_ref,
                &self.rewrite_template.expr,
                &bindings,
            )?;
            let old_to_new = ir_utils::compact_and_toposort_with_mapping_in_place(&mut rewritten)
                .map_err(MatchRewriteRuleApplyError::Toposort)?;
            let matched_root = old_to_new
                .get(rewritten_root.index)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    MatchRewriteRuleApplyError::Toposort(format!(
                        "rewritten root {} was removed during compaction",
                        rewritten_root.index
                    ))
                })?;
            return Ok(MatchRewriteOutcome {
                rewritten_fn: rewritten,
                rewrote: true,
                matched_root: Some(matched_root),
            });
        }

        Ok(MatchRewriteOutcome {
            rewritten_fn: rewritten,
            rewrote: false,
            matched_root: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplateExpr {
    Placeholder(String),
    Const(ConstTemplate),
    OperatorCall(TemplateCall),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstTemplate {
    value: TemplateNumericExpr,
    width: TemplateNumericExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TemplateCall {
    operator: String,
    args: Vec<TemplateExpr>,
    named_args: Vec<TemplateNamedArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TemplateNamedArg {
    name: String,
    value: TemplateNamedArgValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplateNamedArgValue {
    Bool(bool),
    String(String),
    Expr(TemplateExpr),
    ExprList(Vec<TemplateExpr>),
    Numeric(TemplateNumericExpr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplateNumericExpr {
    Number(u64),
    Width(String),
    Add(Box<TemplateNumericExpr>, Box<TemplateNumericExpr>),
    Sub(Box<TemplateNumericExpr>, Box<TemplateNumericExpr>),
}

fn collect_match_binders(
    expr: &ir_query::QueryExpr,
    node_binders: &mut BTreeSet<String>,
    literal_binders: &mut BTreeSet<String>,
    literal_context: bool,
) {
    match expr {
        ir_query::QueryExpr::Placeholder(placeholder) => {
            if placeholder.name == "_" {
                return;
            }
            if literal_context {
                literal_binders.insert(placeholder.name.clone());
            } else {
                node_binders.insert(placeholder.name.clone());
            }
        }
        ir_query::QueryExpr::Matcher(matcher) => {
            let literal_matcher = matches!(matcher.kind, ir_query::MatcherKind::Literal { .. });
            for arg in &matcher.args {
                collect_match_binders(arg, node_binders, literal_binders, literal_matcher);
            }
            for named_arg in &matcher.named_args {
                match &named_arg.value {
                    ir_query::NamedArgValue::Expr(expr) => {
                        collect_match_binders(expr, node_binders, literal_binders, false);
                    }
                    ir_query::NamedArgValue::ExprList(exprs) => {
                        for expr in exprs {
                            collect_match_binders(expr, node_binders, literal_binders, false);
                        }
                    }
                    ir_query::NamedArgValue::Bool(_)
                    | ir_query::NamedArgValue::Number(_)
                    | ir_query::NamedArgValue::Any
                    | ir_query::NamedArgValue::String(_) => {}
                }
            }
        }
        ir_query::QueryExpr::Number(_)
        | ir_query::QueryExpr::Numeric(_)
        | ir_query::QueryExpr::Ellipsis => {}
    }
}

fn collect_rewrite_refs(
    expr: &TemplateExpr,
    placeholder_refs: &mut BTreeSet<String>,
    width_refs: &mut BTreeSet<String>,
) {
    match expr {
        TemplateExpr::Placeholder(name) => {
            placeholder_refs.insert(name.clone());
        }
        TemplateExpr::Const(helper) => {
            collect_numeric_refs(&helper.value, width_refs);
            collect_numeric_refs(&helper.width, width_refs);
        }
        TemplateExpr::OperatorCall(call) => {
            for arg in &call.args {
                collect_rewrite_refs(arg, placeholder_refs, width_refs);
            }
            for named_arg in &call.named_args {
                match &named_arg.value {
                    TemplateNamedArgValue::Expr(expr) => {
                        collect_rewrite_refs(expr, placeholder_refs, width_refs);
                    }
                    TemplateNamedArgValue::ExprList(exprs) => {
                        for expr in exprs {
                            collect_rewrite_refs(expr, placeholder_refs, width_refs);
                        }
                    }
                    TemplateNamedArgValue::Numeric(expr) => {
                        collect_numeric_refs(expr, width_refs);
                    }
                    TemplateNamedArgValue::Bool(_) | TemplateNamedArgValue::String(_) => {}
                }
            }
        }
    }
}

fn collect_numeric_refs(expr: &TemplateNumericExpr, refs: &mut BTreeSet<String>) {
    match expr {
        TemplateNumericExpr::Number(_) => {}
        TemplateNumericExpr::Width(name) => {
            refs.insert(name.clone());
        }
        TemplateNumericExpr::Add(lhs, rhs) | TemplateNumericExpr::Sub(lhs, rhs) => {
            collect_numeric_refs(lhs, refs);
            collect_numeric_refs(rhs, refs);
        }
    }
}

fn dedup_query_bindings(bindings: Vec<ir_query::QueryBindings>) -> Vec<ir_query::QueryBindings> {
    let mut deduped: BTreeMap<String, ir_query::QueryBindings> = BTreeMap::new();
    for env in bindings {
        let key = canonical_bindings_key(&env);
        deduped.entry(key).or_insert(env);
    }
    deduped.into_values().collect()
}

fn canonical_bindings_key(bindings: &ir_query::QueryBindings) -> String {
    let mut entries: Vec<(String, String)> = bindings
        .iter()
        .map(|(name, binding)| (name.clone(), canonical_binding_value(binding)))
        .collect();
    entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    entries
        .into_iter()
        .map(|(name, value)| format!("{}={}", name, value))
        .collect::<Vec<String>>()
        .join(";")
}

fn canonical_binding_value(binding: &ir_query::Binding) -> String {
    match binding {
        ir_query::Binding::Node(node_ref) => format!("node:{}", node_ref.index),
        ir_query::Binding::LiteralValue { value, ty } => canonical_literal_binding_key(value, ty),
    }
}

fn canonical_literal_binding_key(value: &IrValue, ty: &Type) -> String {
    // Preserve both the matched node type and the value text so zero-length
    // arrays, which do not carry their element type in `IrValue` alone, still
    // dedup correctly.
    format!("lit:{}:{}", ty, value)
}

struct MaterializeState<'a> {
    f: &'a mut ir::Fn,
    bindings: &'a ir_query::QueryBindings,
    root_ref: NodeRef,
    original_root: ir::Node,
    root_clone_ref: Option<NodeRef>,
    literal_cache: HashMap<String, NodeRef>,
    const_cache: HashMap<String, NodeRef>,
}

impl<'a> MaterializeState<'a> {
    fn new(f: &'a mut ir::Fn, root_ref: NodeRef, bindings: &'a ir_query::QueryBindings) -> Self {
        Self {
            original_root: f.get_node(root_ref).clone(),
            f,
            bindings,
            root_ref,
            root_clone_ref: None,
            literal_cache: HashMap::new(),
            const_cache: HashMap::new(),
        }
    }

    fn ensure_root_clone(&mut self) -> Result<NodeRef, MatchRewriteRuleApplyError> {
        if let Some(root_clone_ref) = self.root_clone_ref {
            return Ok(root_clone_ref);
        }

        let clone_ref = push_node(
            self.f,
            self.original_root.ty.clone(),
            self.original_root.payload.clone(),
        )?;
        self.root_clone_ref = Some(clone_ref);
        Ok(clone_ref)
    }
}

fn apply_rule_at_root(
    f: &mut ir::Fn,
    root_ref: NodeRef,
    expr: &TemplateExpr,
    bindings: &ir_query::QueryBindings,
) -> Result<NodeRef, MatchRewriteRuleApplyError> {
    let mut state = MaterializeState::new(f, root_ref, bindings);
    let root_ty = state.original_root.ty.clone();
    match expr {
        TemplateExpr::Placeholder(name) => match bindings.get(name) {
            Some(ir_query::Binding::Node(node_ref)) => {
                ir_utils::replace_node_with_ref(state.f, root_ref, *node_ref)
                    .map_err(MatchRewriteRuleApplyError::Validation)?;
                Ok(*node_ref)
            }
            Some(ir_query::Binding::LiteralValue { value, ty }) => {
                ensure_rewrite_preserves_root_type(&root_ty, ty)?;
                ir_utils::replace_node_payload(
                    state.f,
                    root_ref,
                    NodePayload::Literal(value.clone()),
                    Some(ty.clone()),
                )
                .map_err(MatchRewriteRuleApplyError::Validation)?;
                Ok(root_ref)
            }
            None => Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "unknown placeholder '{}'",
                name
            ))),
        },
        TemplateExpr::Const(helper) => {
            let (payload, ty) = build_const_payload(&state, helper)?;
            ensure_rewrite_preserves_root_type(&root_ty, &ty)?;
            ir_utils::replace_node_payload(state.f, root_ref, payload, Some(ty))
                .map_err(MatchRewriteRuleApplyError::Validation)?;
            Ok(root_ref)
        }
        TemplateExpr::OperatorCall(call) => {
            let (payload, ty) = build_operator_payload(&mut state, call)?;
            ensure_rewrite_preserves_root_type(&root_ty, &ty)?;
            ir_utils::replace_node_payload(state.f, root_ref, payload, Some(ty))
                .map_err(MatchRewriteRuleApplyError::Validation)?;
            Ok(root_ref)
        }
    }
}

fn ensure_rewrite_preserves_root_type(
    root_ty: &Type,
    rewritten_ty: &Type,
) -> Result<(), MatchRewriteRuleApplyError> {
    if root_ty != rewritten_ty {
        return Err(MatchRewriteRuleApplyError::Validation(format!(
            "rewrite would change matched node type from {} to {}",
            root_ty, rewritten_ty
        )));
    }
    Ok(())
}

fn materialize_expr(
    state: &mut MaterializeState<'_>,
    expr: &TemplateExpr,
) -> Result<NodeRef, MatchRewriteRuleApplyError> {
    match expr {
        TemplateExpr::Placeholder(name) => match state.bindings.get(name) {
            Some(ir_query::Binding::Node(node_ref)) => {
                if *node_ref == state.root_ref {
                    state.ensure_root_clone()
                } else {
                    Ok(*node_ref)
                }
            }
            Some(ir_query::Binding::LiteralValue { value, ty }) => {
                if let Some(node_ref) = state.literal_cache.get(name) {
                    return Ok(*node_ref);
                }
                let node_ref = push_node(state.f, ty.clone(), NodePayload::Literal(value.clone()))?;
                state.literal_cache.insert(name.clone(), node_ref);
                Ok(node_ref)
            }
            None => Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "unknown placeholder '{}'",
                name
            ))),
        },
        TemplateExpr::Const(helper) => {
            let key = const_helper_key(state, helper)?;
            if let Some(node_ref) = state.const_cache.get(&key) {
                return Ok(*node_ref);
            }
            let (payload, ty) = build_const_payload(state, helper)?;
            let node_ref = push_node(state.f, ty, payload)?;
            state.const_cache.insert(key, node_ref);
            Ok(node_ref)
        }
        TemplateExpr::OperatorCall(call) => {
            let (payload, ty) = build_operator_payload(state, call)?;
            push_node(state.f, ty, payload)
        }
    }
}

fn build_const_payload(
    state: &MaterializeState<'_>,
    helper: &ConstTemplate,
) -> Result<(NodePayload, Type), MatchRewriteRuleApplyError> {
    let value = eval_numeric_expr(&helper.value, state)?;
    let width = eval_numeric_expr(&helper.width, state)?;
    let value_u64 = u64::try_from(value).map_err(|_| {
        MatchRewriteRuleApplyError::LiteralConstruction(format!(
            "$const(value={}, width={}) value does not fit in u64",
            value, width
        ))
    })?;
    let literal = IrValue::make_ubits(width, value_u64).map_err(|e| {
        MatchRewriteRuleApplyError::LiteralConstruction(format!(
            "$const(value={}, width={}) failed: {}",
            value, width, e
        ))
    })?;
    Ok((NodePayload::Literal(literal), Type::Bits(width)))
}

fn const_helper_key(
    state: &MaterializeState<'_>,
    helper: &ConstTemplate,
) -> Result<String, MatchRewriteRuleApplyError> {
    let value = eval_numeric_expr(&helper.value, state)?;
    let width = eval_numeric_expr(&helper.width, state)?;
    Ok(format!("value={} width={}", value, width))
}

fn build_operator_payload(
    state: &mut MaterializeState<'_>,
    call: &TemplateCall,
) -> Result<(NodePayload, Type), MatchRewriteRuleApplyError> {
    let positional_refs: Vec<NodeRef> = call
        .args
        .iter()
        .map(|expr| materialize_expr(state, expr))
        .collect::<Result<Vec<NodeRef>, _>>()?;
    let named_args = materialize_named_args(state, &call.named_args)?;

    let payload = if let Some(binop) = ir::operator_to_binop(&call.operator) {
        require_no_named_args(&call.operator, &named_args)?;
        if positional_refs.len() != 2 {
            return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "{} expects exactly 2 positional operands; got {}",
                call.operator,
                positional_refs.len()
            )));
        }
        NodePayload::Binop(binop, positional_refs[0], positional_refs[1])
    } else if let Some(unop) = ir::operator_to_unop(&call.operator) {
        require_no_named_args(&call.operator, &named_args)?;
        if positional_refs.len() != 1 {
            return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "{} expects exactly 1 positional operand; got {}",
                call.operator,
                positional_refs.len()
            )));
        }
        NodePayload::Unop(unop, positional_refs[0])
    } else if let Some(nary_op) = ir::operator_to_nary_op(&call.operator) {
        require_no_named_args(&call.operator, &named_args)?;
        NodePayload::Nary(nary_op, positional_refs)
    } else {
        build_special_payload(&call.operator, positional_refs, named_args)?
    };

    let operand_types: Vec<Type> = ir_utils::operands(&payload)
        .iter()
        .map(|node_ref| state.f.get_node_ty(*node_ref).clone())
        .collect();
    let ty = ir_deduce::deduce_result_type(&payload, &operand_types)
        .map_err(|e| MatchRewriteRuleApplyError::TypeDeduction(e.to_string()))?
        .ok_or_else(|| {
            MatchRewriteRuleApplyError::UnsupportedRewriteOperator(format!(
                "rewrite operator '{}' is not supported for type deduction",
                call.operator
            ))
        })?;
    payload
        .validate(state.f)
        .map_err(MatchRewriteRuleApplyError::Validation)?;
    Ok((payload, ty))
}

fn build_special_payload(
    operator: &str,
    positional_refs: Vec<NodeRef>,
    named_args: MaterializedNamedArgs,
) -> Result<NodePayload, MatchRewriteRuleApplyError> {
    match operator {
        "tuple" => {
            named_args.require_no_extra(operator, &[])?;
            Ok(NodePayload::Tuple(positional_refs))
        }
        "array" => {
            named_args.require_no_extra(operator, &[])?;
            Ok(NodePayload::Array(positional_refs))
        }
        "after_all" => {
            named_args.require_no_extra(operator, &[])?;
            Ok(NodePayload::AfterAll(positional_refs))
        }
        "bit_slice" => {
            named_args.require_no_extra(operator, &["start", "width"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "bit_slice expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::BitSlice {
                arg: positional_refs[0],
                start: named_args.require_numeric("start")?,
                width: named_args.require_numeric("width")?,
            })
        }
        "dynamic_bit_slice" => {
            named_args.require_no_extra(operator, &["width"])?;
            if positional_refs.len() != 2 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "dynamic_bit_slice expects exactly 2 positional operands".to_string(),
                ));
            }
            Ok(NodePayload::DynamicBitSlice {
                arg: positional_refs[0],
                start: positional_refs[1],
                width: named_args.require_numeric("width")?,
            })
        }
        "sign_ext" => {
            named_args.require_no_extra(operator, &["new_bit_count"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "sign_ext expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::SignExt {
                arg: positional_refs[0],
                new_bit_count: named_args.require_numeric("new_bit_count")?,
            })
        }
        "zero_ext" => {
            named_args.require_no_extra(operator, &["new_bit_count"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "zero_ext expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::ZeroExt {
                arg: positional_refs[0],
                new_bit_count: named_args.require_numeric("new_bit_count")?,
            })
        }
        "tuple_index" => {
            named_args.require_no_extra(operator, &["index"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "tuple_index expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::TupleIndex {
                tuple: positional_refs[0],
                index: named_args.require_numeric("index")?,
            })
        }
        "array_slice" => {
            named_args.require_no_extra(operator, &["width"])?;
            if positional_refs.len() != 2 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "array_slice expects exactly 2 positional operands".to_string(),
                ));
            }
            Ok(NodePayload::ArraySlice {
                array: positional_refs[0],
                start: positional_refs[1],
                width: named_args.require_numeric("width")?,
            })
        }
        "decode" => {
            named_args.require_no_extra(operator, &["width"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "decode expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::Decode {
                arg: positional_refs[0],
                width: named_args.require_numeric("width")?,
            })
        }
        "encode" => {
            named_args.require_no_extra(operator, &[])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "encode expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::Encode {
                arg: positional_refs[0],
            })
        }
        "one_hot" => {
            named_args.require_no_extra(operator, &["lsb_prio"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "one_hot expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::OneHot {
                arg: positional_refs[0],
                lsb_prio: named_args.require_bool("lsb_prio")?,
            })
        }
        "ext_prio_encode" => {
            named_args.require_no_extra(operator, &["lsb_prio"])?;
            if positional_refs.len() != 1 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "ext_prio_encode expects exactly 1 positional operand".to_string(),
                ));
            }
            Ok(NodePayload::ExtPrioEncode {
                arg: positional_refs[0],
                lsb_prio: named_args.require_bool("lsb_prio")?,
            })
        }
        "ext_carry_out" => {
            named_args.require_no_extra(operator, &[])?;
            if positional_refs.len() != 3 {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(
                    "ext_carry_out expects exactly 3 positional operands".to_string(),
                ));
            }
            Ok(NodePayload::ExtCarryOut {
                lhs: positional_refs[0],
                rhs: positional_refs[1],
                c_in: positional_refs[2],
            })
        }
        "sel" | "priority_sel" | "one_hot_sel" => {
            let selector = if let Some(selector) = named_args.exprs.get("selector") {
                if !positional_refs.is_empty() {
                    return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                        "{} does not allow positional selector args when selector=<expr> is provided",
                        operator
                    )));
                }
                *selector
            } else if positional_refs.len() == 1 {
                positional_refs[0]
            } else {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                    "{} requires selector=<expr> or one positional selector argument",
                    operator
                )));
            };

            let cases = named_args.require_expr_list("cases")?;
            if cases.is_empty() {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                    "{} requires a non-empty cases=[...] list",
                    operator
                )));
            }
            match operator {
                "sel" => {
                    named_args.require_no_extra(operator, &["selector", "cases", "default"])?;
                    Ok(NodePayload::Sel {
                        selector,
                        cases,
                        default: named_args.optional_expr("default"),
                    })
                }
                "priority_sel" => {
                    named_args.require_no_extra(operator, &["selector", "cases", "default"])?;
                    Ok(NodePayload::PrioritySel {
                        selector,
                        cases,
                        default: named_args.optional_expr("default"),
                    })
                }
                "one_hot_sel" => {
                    named_args.require_no_extra(operator, &["selector", "cases"])?;
                    Ok(NodePayload::OneHotSel { selector, cases })
                }
                _ => unreachable!(),
            }
        }
        _ => Err(MatchRewriteRuleApplyError::UnsupportedRewriteOperator(
            format!("rewrite operator '{}' is not supported", operator),
        )),
    }
}

fn require_no_named_args(
    operator: &str,
    named_args: &MaterializedNamedArgs,
) -> Result<(), MatchRewriteRuleApplyError> {
    named_args.require_no_extra(operator, &[])
}

#[derive(Debug, Default)]
struct MaterializedNamedArgs {
    exprs: BTreeMap<String, NodeRef>,
    expr_lists: BTreeMap<String, Vec<NodeRef>>,
    numerics: BTreeMap<String, usize>,
    bools: BTreeMap<String, bool>,
    strings: BTreeMap<String, String>,
}

impl MaterializedNamedArgs {
    fn require_no_extra(
        &self,
        operator: &str,
        allowed: &[&str],
    ) -> Result<(), MatchRewriteRuleApplyError> {
        let allowed: BTreeSet<&str> = allowed.iter().copied().collect();
        for name in self
            .exprs
            .keys()
            .chain(self.expr_lists.keys())
            .chain(self.numerics.keys())
            .chain(self.bools.keys())
            .chain(self.strings.keys())
        {
            if !allowed.contains(name.as_str()) {
                return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                    "{} does not support named argument '{}'",
                    operator, name
                )));
            }
        }
        Ok(())
    }

    fn require_numeric(&self, name: &str) -> Result<usize, MatchRewriteRuleApplyError> {
        self.numerics.get(name).copied().ok_or_else(|| {
            MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "missing numeric named argument '{}'",
                name
            ))
        })
    }

    fn require_bool(&self, name: &str) -> Result<bool, MatchRewriteRuleApplyError> {
        self.bools.get(name).copied().ok_or_else(|| {
            MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "missing boolean named argument '{}'",
                name
            ))
        })
    }

    fn require_expr_list(&self, name: &str) -> Result<Vec<NodeRef>, MatchRewriteRuleApplyError> {
        self.expr_lists.get(name).cloned().ok_or_else(|| {
            MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "missing expression-list named argument '{}'",
                name
            ))
        })
    }

    fn optional_expr(&self, name: &str) -> Option<NodeRef> {
        self.exprs.get(name).copied()
    }
}

fn materialize_named_args(
    state: &mut MaterializeState<'_>,
    named_args: &[TemplateNamedArg],
) -> Result<MaterializedNamedArgs, MatchRewriteRuleApplyError> {
    let mut out = MaterializedNamedArgs::default();
    let mut seen = BTreeSet::new();
    for arg in named_args {
        if !seen.insert(arg.name.clone()) {
            return Err(MatchRewriteRuleApplyError::InvalidRewriteTemplate(format!(
                "duplicate named argument '{}'",
                arg.name
            )));
        }
        match &arg.value {
            TemplateNamedArgValue::Bool(value) => {
                out.bools.insert(arg.name.clone(), *value);
            }
            TemplateNamedArgValue::String(value) => {
                out.strings.insert(arg.name.clone(), value.clone());
            }
            TemplateNamedArgValue::Expr(expr) => {
                let node_ref = materialize_expr(state, expr)?;
                out.exprs.insert(arg.name.clone(), node_ref);
            }
            TemplateNamedArgValue::ExprList(exprs) => {
                let node_refs = exprs
                    .iter()
                    .map(|expr| materialize_expr(state, expr))
                    .collect::<Result<Vec<NodeRef>, _>>()?;
                out.expr_lists.insert(arg.name.clone(), node_refs);
            }
            TemplateNamedArgValue::Numeric(expr) => {
                let value = eval_numeric_expr(expr, state)?;
                out.numerics.insert(arg.name.clone(), value);
            }
        }
    }
    Ok(out)
}

fn eval_numeric_expr(
    expr: &TemplateNumericExpr,
    state: &MaterializeState<'_>,
) -> Result<usize, MatchRewriteRuleApplyError> {
    fn eval_inner(
        expr: &TemplateNumericExpr,
        state: &MaterializeState<'_>,
    ) -> Result<i128, MatchRewriteRuleApplyError> {
        match expr {
            TemplateNumericExpr::Number(number) => Ok(i128::from(*number)),
            TemplateNumericExpr::Width(name) => match state.bindings.get(name) {
                Some(ir_query::Binding::Node(node_ref)) => {
                    Ok(i128::try_from(state.f.get_node_ty(*node_ref).bit_count()).unwrap())
                }
                Some(ir_query::Binding::LiteralValue { .. }) => {
                    Err(MatchRewriteRuleApplyError::NumericEvaluation(format!(
                        "$width({}) requires a node binding",
                        name
                    )))
                }
                None => Err(MatchRewriteRuleApplyError::NumericEvaluation(format!(
                    "unknown numeric placeholder '{}'",
                    name
                ))),
            },
            TemplateNumericExpr::Add(lhs, rhs) => {
                let lhs = eval_inner(lhs, state)?;
                let rhs = eval_inner(rhs, state)?;
                lhs.checked_add(rhs).ok_or_else(|| {
                    MatchRewriteRuleApplyError::NumericEvaluation(
                        "numeric expression overflow in addition".to_string(),
                    )
                })
            }
            TemplateNumericExpr::Sub(lhs, rhs) => {
                let lhs = eval_inner(lhs, state)?;
                let rhs = eval_inner(rhs, state)?;
                lhs.checked_sub(rhs).ok_or_else(|| {
                    MatchRewriteRuleApplyError::NumericEvaluation(
                        "numeric expression overflow in subtraction".to_string(),
                    )
                })
            }
        }
    }

    let value = eval_inner(expr, state)?;
    if value < 0 {
        return Err(MatchRewriteRuleApplyError::NumericEvaluation(
            "numeric expression evaluated to a negative value".to_string(),
        ));
    }
    usize::try_from(value).map_err(|_| {
        MatchRewriteRuleApplyError::NumericEvaluation(
            "numeric expression result does not fit in usize".to_string(),
        )
    })
}

#[cfg(test)]
fn literal_type(value: &IrValue) -> Result<Type, MatchRewriteRuleApplyError> {
    if let Ok(width) = value.bit_count() {
        return Ok(Type::Bits(width));
    }

    let rendered = value.to_string();
    if rendered == "token" {
        return Ok(Type::Token);
    }

    let element_count = value.get_element_count().map_err(|e| {
        MatchRewriteRuleApplyError::LiteralConstruction(format!(
            "failed to determine literal shape for {}: {}",
            rendered, e
        ))
    })?;

    let mut element_types = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let element = value.get_element(index).map_err(|e| {
            MatchRewriteRuleApplyError::LiteralConstruction(format!(
                "failed to get element {} for {}: {}",
                index, rendered, e
            ))
        })?;
        element_types.push(literal_type(&element)?);
    }

    if rendered.starts_with('(') {
        return Ok(Type::Tuple(
            element_types.into_iter().map(Box::new).collect(),
        ));
    }
    if rendered.starts_with('[') {
        let Some(element_ty) = element_types.first().cloned() else {
            return Err(MatchRewriteRuleApplyError::LiteralConstruction(
                "empty array literals are not supported".to_string(),
            ));
        };
        for candidate in element_types.iter().skip(1) {
            if candidate != &element_ty {
                return Err(MatchRewriteRuleApplyError::LiteralConstruction(format!(
                    "array literal {} has mixed element types {} and {}",
                    rendered, element_ty, candidate
                )));
            }
        }
        return Ok(Type::new_array(element_ty, element_count));
    }

    Err(MatchRewriteRuleApplyError::LiteralConstruction(format!(
        "failed to determine literal type for {}",
        rendered
    )))
}

fn push_node(
    f: &mut ir::Fn,
    ty: Type,
    payload: NodePayload,
) -> Result<NodeRef, MatchRewriteRuleApplyError> {
    payload
        .validate(f)
        .map_err(MatchRewriteRuleApplyError::Validation)?;
    let text_id = next_text_id(f);
    let index = f.nodes.len();
    f.nodes.push(ir::Node {
        text_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    Ok(NodeRef { index })
}

fn next_text_id(f: &ir::Fn) -> usize {
    f.nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

struct RewriteTemplateParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> RewriteTemplateParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn parse_expr(&mut self) -> Result<TemplateExpr, String> {
        self.skip_ws();
        if self
            .bytes
            .get(self.pos..)
            .is_some_and(|bytes| bytes.starts_with(b"..."))
        {
            return Err(self.error("ellipsis '...' is not supported in rewrite expressions"));
        }
        match self.peek() {
            Some(b'$') => {
                self.bump();
                let ident = self.parse_ident("rewrite helper")?;
                match ident.as_str() {
                    "const" => self.parse_const_helper(),
                    _ => Err(self.error(&format!("unknown rewrite helper ${}", ident))),
                }
            }
            Some(c) if c.is_ascii_digit() => Err(self
                .error("bare numeric literals are not valid rewrite expressions; use $const(...)")),
            Some(_) => {
                let ident = self.parse_ident("placeholder or operator")?;
                if ident == "_" {
                    return Err(self.error("wildcard '_' is not supported in rewrite expressions"));
                }
                self.skip_ws();
                if self.peek() == Some(b'[') {
                    return Err(
                        self.error("bracket clauses are not supported in rewrite expressions")
                    );
                }
                if self.peek() != Some(b'(') {
                    return Ok(TemplateExpr::Placeholder(ident));
                }
                self.expect('(')?;
                let args = self.parse_call_args()?;
                self.expect(')')?;
                Ok(TemplateExpr::OperatorCall(TemplateCall {
                    operator: ident,
                    args: args.args,
                    named_args: args.named_args,
                }))
            }
            None => Err(self.error("expected rewrite expression")),
        }
    }

    fn parse_const_helper(&mut self) -> Result<TemplateExpr, String> {
        self.expect('(')?;
        let args = self.parse_call_args()?;
        self.expect(')')?;
        if !args.args.is_empty() {
            return Err(self.error("$const(...) does not support positional arguments"));
        }

        let mut value: Option<TemplateNumericExpr> = None;
        let mut width: Option<TemplateNumericExpr> = None;
        for named_arg in args.named_args {
            match (named_arg.name.as_str(), named_arg.value) {
                ("value", TemplateNamedArgValue::Numeric(expr)) => value = Some(expr),
                ("width", TemplateNamedArgValue::Numeric(expr)) => width = Some(expr),
                ("value", _) => return Err(self.error("$const(value=...) expects a numeric value")),
                ("width", _) => return Err(self.error("$const(width=...) expects a numeric value")),
                (other, _) => {
                    return Err(self.error(&format!(
                        "$const(...) does not support named argument '{}'",
                        other
                    )));
                }
            }
        }

        let value = value.ok_or_else(|| self.error("$const(...) requires value=<numeric-expr>"))?;
        let width = width.ok_or_else(|| self.error("$const(...) requires width=<numeric-expr>"))?;
        Ok(TemplateExpr::Const(ConstTemplate { value, width }))
    }

    fn parse_call_args(&mut self) -> Result<ParsedTemplateArgs, String> {
        let mut args = Vec::new();
        let mut named_args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b')') {
            return Ok(ParsedTemplateArgs { args, named_args });
        }
        loop {
            if let Some(named_arg) = self.parse_named_arg()? {
                named_args.push(named_arg);
            } else {
                args.push(self.parse_expr()?);
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b')') => break,
                _ => return Err(self.error("expected ',' or ')'")),
            }
        }
        Ok(ParsedTemplateArgs { args, named_args })
    }

    fn parse_named_arg(&mut self) -> Result<Option<TemplateNamedArg>, String> {
        self.skip_ws();
        let start = self.pos;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {}
            _ => return Ok(None),
        }
        let ident = self.parse_ident("named argument")?;
        self.skip_ws();
        if self.peek() != Some(b'=') {
            self.pos = start;
            return Ok(None);
        }
        self.bump();
        self.skip_ws();
        let value = match self.peek() {
            Some(b'"') => TemplateNamedArgValue::String(self.parse_string_literal()?),
            Some(b'[') => TemplateNamedArgValue::ExprList(self.parse_expr_list()?),
            Some(c) if c.is_ascii_digit() => {
                TemplateNamedArgValue::Numeric(self.parse_numeric_expr()?)
            }
            Some(b'(') => TemplateNamedArgValue::Numeric(self.parse_numeric_expr()?),
            Some(b'$') => match self.peek_dollar_ident() {
                Some("width") => TemplateNamedArgValue::Numeric(self.parse_numeric_expr()?),
                Some("const") => TemplateNamedArgValue::Expr(self.parse_expr()?),
                Some(other) => {
                    return Err(self.error(&format!(
                        "unknown rewrite helper or numeric expression ${}",
                        other
                    )));
                }
                None => return Err(self.error("expected helper or numeric expression after '$'")),
            },
            _ => match self.parse_expr()? {
                TemplateExpr::Placeholder(name) if name == "true" => {
                    TemplateNamedArgValue::Bool(true)
                }
                TemplateExpr::Placeholder(name) if name == "false" => {
                    TemplateNamedArgValue::Bool(false)
                }
                expr => TemplateNamedArgValue::Expr(expr),
            },
        };
        Ok(Some(TemplateNamedArg { name: ident, value }))
    }

    fn parse_expr_list(&mut self) -> Result<Vec<TemplateExpr>, String> {
        self.expect('[')?;
        let mut exprs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(exprs);
        }
        loop {
            exprs.push(self.parse_expr()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
        Ok(exprs)
    }

    fn parse_numeric_expr(&mut self) -> Result<TemplateNumericExpr, String> {
        let mut expr = self.parse_numeric_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.bump();
                    let rhs = self.parse_numeric_factor()?;
                    expr = TemplateNumericExpr::Add(Box::new(expr), Box::new(rhs));
                }
                Some(b'-') => {
                    self.bump();
                    let rhs = self.parse_numeric_factor()?;
                    expr = TemplateNumericExpr::Sub(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_numeric_factor(&mut self) -> Result<TemplateNumericExpr, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.expect('(')?;
                let expr = self.parse_numeric_expr()?;
                self.expect(')')?;
                Ok(expr)
            }
            Some(b'$') => {
                self.bump();
                let ident = self.parse_ident("numeric expression")?;
                if ident != "width" {
                    return Err(self.error(&format!("unknown numeric helper ${}", ident)));
                }
                self.expect('(')?;
                let name = self.parse_ident("placeholder")?;
                self.expect(')')?;
                Ok(TemplateNumericExpr::Width(name))
            }
            Some(c) if c.is_ascii_digit() => {
                Ok(TemplateNumericExpr::Number(self.parse_u64("number")?))
            }
            _ => Err(self.error("expected numeric expression")),
        }
    }

    fn parse_string_literal(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        let mut chunk_start = self.pos;
        loop {
            match self.peek() {
                Some(b'"') => {
                    if chunk_start < self.pos {
                        let chunk = std::str::from_utf8(&self.bytes[chunk_start..self.pos])
                            .map_err(|_| self.error("invalid utf-8 in string literal"))?;
                        out.push_str(chunk);
                    }
                    self.bump();
                    break;
                }
                Some(b'\\') => {
                    if chunk_start < self.pos {
                        let chunk = std::str::from_utf8(&self.bytes[chunk_start..self.pos])
                            .map_err(|_| self.error("invalid utf-8 in string literal"))?;
                        out.push_str(chunk);
                    }
                    self.bump();
                    match self.peek() {
                        Some(b'"') => {
                            self.bump();
                            out.push('"');
                        }
                        Some(b'\\') => {
                            self.bump();
                            out.push('\\');
                        }
                        Some(b'n') => {
                            self.bump();
                            out.push('\n');
                        }
                        Some(b't') => {
                            self.bump();
                            out.push('\t');
                        }
                        Some(b'r') => {
                            self.bump();
                            out.push('\r');
                        }
                        _ => return Err(self.error("invalid escape in string literal")),
                    }
                    chunk_start = self.pos;
                }
                Some(_) => {
                    self.bump();
                }
                None => return Err(self.error("unterminated string literal")),
            }
        }
        Ok(out)
    }

    fn parse_ident(&mut self, ctx: &str) -> Result<String, String> {
        let start = self.pos;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                self.bump();
            }
            _ => return Err(self.error(&format!("expected {}", ctx))),
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_') {
            self.bump();
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .map(|s| s.to_string())
            .map_err(|_| self.error(&format!("invalid utf-8 in {}", ctx)))
    }

    fn parse_u64(&mut self, ctx: &str) -> Result<u64, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.pos == start {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error(&format!("invalid utf-8 in {}", ctx)))?;
        text.parse::<u64>()
            .map_err(|_| self.error(&format!("{} does not fit in u64", ctx)))
    }

    fn peek_dollar_ident(&self) -> Option<&'a str> {
        if self.peek() != Some(b'$') {
            return None;
        }
        let mut pos = self.pos + 1;
        match self.bytes.get(pos).copied() {
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                pos += 1;
            }
            _ => return None,
        }
        while matches!(self.bytes.get(pos).copied(), Some(c) if c.is_ascii_alphanumeric() || c == b'_')
        {
            pos += 1;
        }
        std::str::from_utf8(&self.bytes[self.pos + 1..pos]).ok()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
            self.bump();
        }
    }

    fn is_done(&self) -> bool {
        self.peek().is_none()
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        let byte = ch as u8;
        self.skip_ws();
        if self.peek() == Some(byte) {
            self.bump();
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", ch)))
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn error(&self, msg: &str) -> String {
        format!("{} at byte {}", msg, self.pos)
    }
}

struct ParsedTemplateArgs {
    args: Vec<TemplateExpr>,
    named_args: Vec<TemplateNamedArg>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;
    use std::collections::HashMap;

    fn parse_fn(text: &str) -> ir::Fn {
        let mut parser = ir_parser::Parser::new(text);
        parser.parse_fn().expect("function should parse")
    }

    fn normalized_fn_text(text: &str) -> String {
        let f = parse_fn(text);
        f.to_string()
    }

    #[test]
    fn match_pattern_parse_accepts_existing_ir_query() {
        let pattern = MatchPattern::parse("sub(add(x, literal(L)), literal(L))").unwrap();
        assert_eq!(pattern.text, "sub(add(x, literal(L)), literal(L))");
        assert!(pattern.node_binders.contains("x"));
        assert!(pattern.literal_binders.contains("L"));
    }

    #[test]
    fn rewrite_template_parse_accepts_operator_and_const_syntax() {
        let template = RewriteTemplate::parse(
            "and(bit_slice(x, start=0, width=1), eq(s, $const(value=0, width=$width(s))))",
        )
        .unwrap();
        assert_eq!(
            template.text,
            "and(bit_slice(x, start=0, width=1), eq(s, $const(value=0, width=$width(s))))"
        );
        assert!(template.placeholder_refs.contains("x"));
        assert!(template.placeholder_refs.contains("s"));
        assert!(template.width_refs.contains("s"));
    }

    #[test]
    fn rewrite_template_parse_preserves_utf8_string_literals() {
        let template =
            RewriteTemplate::parse(r#"assert(tok, pred, message="µ\n", label="µ")"#).unwrap();
        let TemplateExpr::OperatorCall(call) = &template.expr else {
            panic!("expected operator call");
        };
        assert_eq!(call.operator, "assert");
        assert_eq!(
            call.named_args,
            vec![
                TemplateNamedArg {
                    name: "message".to_string(),
                    value: TemplateNamedArgValue::String("µ\n".to_string()),
                },
                TemplateNamedArg {
                    name: "label".to_string(),
                    value: TemplateNamedArgValue::String("µ".to_string()),
                },
            ]
        );
    }

    #[test]
    fn rewrite_template_parse_rejects_wildcard() {
        let err = RewriteTemplate::parse("and(_, x)").unwrap_err();
        assert!(err.to_string().contains("wildcard '_'"));
    }

    #[test]
    fn rewrite_template_parse_rejects_ellipsis() {
        let err = RewriteTemplate::parse("and(..., x)").unwrap_err();
        assert!(err.to_string().contains("ellipsis '...'"));
    }

    #[test]
    fn rewrite_template_parse_rejects_query_only_matcher_syntax() {
        let err = RewriteTemplate::parse("$users(x)").unwrap_err();
        assert!(err.to_string().contains("unknown rewrite helper"));
    }

    #[test]
    fn match_rewrite_rule_new_rejects_unbound_placeholder() {
        let pattern = MatchPattern::parse("and(x, y)").unwrap();
        let template = RewriteTemplate::parse("or(x, z)").unwrap();
        let err = MatchRewriteRule::new(pattern, template).unwrap_err();
        assert_eq!(
            err,
            MatchRewriteRuleBuildError::UnboundRewritePlaceholder {
                name: "z".to_string()
            }
        );
    }

    #[test]
    fn match_rewrite_rule_new_rejects_mixed_binding_kinds() {
        let pattern = MatchPattern::parse("and(x, literal(x))").unwrap();
        let template = RewriteTemplate::parse("x").unwrap();
        let err = MatchRewriteRule::new(pattern, template).unwrap_err();
        assert_eq!(
            err,
            MatchRewriteRuleBuildError::MixedBindingKind {
                name: "x".to_string()
            }
        );
    }

    #[test]
    fn match_rewrite_rule_from_strings_matches_parse_and_new_validation() {
        let err = MatchRewriteRule::from_strings("and(x, y)", "or(x, z)").unwrap_err();
        assert_eq!(
            err,
            MatchRewriteRuleBuildError::UnboundRewritePlaceholder {
                name: "z".to_string()
            }
        );
    }

    #[test]
    fn apply_rewrites_root_to_existing_bound_node() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[x, x], id=10)
}"#;
        let rule = MatchRewriteRule::from_strings("sel(selector=p, cases=[x, x])", "x").unwrap();
        let outcome = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap();
        assert!(outcome.rewrote());
        assert_eq!(outcome.matched_root(), Some(NodeRef { index: 2 }));
        assert_eq!(outcome.rewritten_fn().ret_node_ref, outcome.matched_root());
        assert_eq!(
            outcome.rewritten_fn().to_string(),
            normalized_fn_text(
                r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  ret x: bits[8] = param(name=x, id=2)
}"#
            )
        );
    }

    #[test]
    fn apply_rewrites_root_with_fresh_helper_nodes() {
        let ir_text = r#"fn t(x: bits[8] id=1, s: bits[8] id=2) -> bits[1] {
  shll.10: bits[8] = shll(x, s, id=10)
  ret bit_slice.11: bits[1] = bit_slice(shll.10, start=0, width=1, id=11)
}"#;
        let rule = MatchRewriteRule::from_strings(
            "bit_slice(shll(x, s), start=0, width=1)",
            "and(bit_slice(x, start=0, width=1), eq(s, $const(value=0, width=$width(s))))",
        )
        .unwrap();
        let outcome = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap();
        assert!(outcome.rewrote());
        let out_text = outcome.rewritten_fn().to_string();
        assert!(
            out_text.contains("and("),
            "expected and in output:\n{}",
            out_text
        );
        assert!(
            out_text.contains("eq("),
            "expected eq in output:\n{}",
            out_text
        );
        assert!(
            out_text.contains("literal(value=0"),
            "expected zero literal in output:\n{}",
            out_text
        );
    }

    #[test]
    fn apply_rewrite_supports_const_width_from_node_binding() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret eq.10: bits[1] = eq(x, y, id=10)
}"#;
        let rule =
            MatchRewriteRule::from_strings("eq(x, y)", "eq(x, $const(value=1, width=$width(y)))")
                .unwrap();
        let outcome = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap();
        let out_text = outcome.rewritten_fn().to_string();
        assert!(
            out_text.contains("literal(value=1"),
            "expected one literal in output:\n{}",
            out_text
        );
    }

    #[test]
    fn apply_rewrite_supports_bound_literal_value_placeholders() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[1] {
  literal.10: bits[8] = literal(value=1, id=10)
  ret eq.11: bits[1] = eq(x, literal.10, id=11)
}"#;
        let rule = MatchRewriteRule::from_strings("eq(x, literal(L))", "ne(x, L)").unwrap();
        let outcome = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap();
        let out_text = outcome.rewritten_fn().to_string();
        assert!(
            out_text.contains("ne("),
            "expected ne in output:\n{}",
            out_text
        );
        assert!(
            out_text.contains("literal(value=1"),
            "expected literal binding to materialize in output:\n{}",
            out_text
        );
    }

    #[test]
    fn apply_rewrite_supports_token_literal_placeholders() {
        let ir_text = r#"fn t() -> token {
  ret literal.10: token = literal(value=token, id=10)
}"#;
        let input = parse_fn(ir_text);
        let rule = MatchRewriteRule::from_strings("literal(L)", "L").unwrap();
        let outcome = rule
            .apply_to_fn(&input, RewriteApplyMode::FirstMatch)
            .unwrap();
        assert!(outcome.rewrote());
        assert_eq!(outcome.rewritten_fn().to_string(), input.to_string());
    }

    #[test]
    fn apply_rule_at_root_uses_bound_literal_node_type() {
        let mut f = parse_fn(
            r#"fn t() -> bits[1] {
  ret literal.10: bits[1] = literal(value=0, id=10)
}"#,
        );
        let empty_array_ty = Type::new_array(Type::Bits(8), 0);
        let ret_node_ref = f.ret_node_ref.expect("return node");
        f.ret_ty = empty_array_ty.clone();
        f.get_node_mut(ret_node_ref).ty = empty_array_ty.clone();

        // Current `IrValue` helpers do not expose empty-array construction, so
        // use a synthetic binding to verify the rewrite path consumes the
        // stored literal node type instead of re-deriving it from the value.
        let bindings = HashMap::from([(
            "L".to_string(),
            ir_query::Binding::LiteralValue {
                value: IrValue::make_token(),
                ty: empty_array_ty.clone(),
            },
        )]);
        let rewritten_root = apply_rule_at_root(
            &mut f,
            ret_node_ref,
            &TemplateExpr::Placeholder("L".to_string()),
            &bindings,
        )
        .unwrap();
        assert_eq!(rewritten_root, ret_node_ref);
        assert_eq!(f.get_node(ret_node_ref).ty, empty_array_ty);
        assert!(matches!(
            f.get_node(ret_node_ref).payload,
            NodePayload::Literal(_)
        ));
    }

    #[test]
    fn apply_no_match_returns_unchanged_function() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret add.10: bits[8] = add(x, y, id=10)
}"#;
        let input = parse_fn(ir_text);
        let rule = MatchRewriteRule::from_strings("sub(x, y)", "add(y, x)").unwrap();
        let outcome = rule
            .apply_to_fn(&input, RewriteApplyMode::FirstMatch)
            .unwrap();
        assert!(!outcome.rewrote());
        assert_eq!(outcome.matched_root(), None);
        assert_eq!(outcome.rewritten_fn().to_string(), input.to_string());
    }

    #[test]
    fn apply_first_match_skips_ret_unreachable_nodes() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  dead_add: bits[8] = add(x, x, id=10)
  live_add: bits[8] = add(y, y, id=11)
  ret identity.12: bits[8] = identity(live_add, id=12)
}"#;
        let rule = MatchRewriteRule::from_strings("add(a, a)", "a").unwrap();
        let outcome = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap();
        assert!(outcome.rewrote());
        assert_eq!(outcome.matched_root(), Some(NodeRef { index: 2 }));
        assert_eq!(
            outcome.rewritten_fn().to_string(),
            normalized_fn_text(
                r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  dead_add: bits[8] = add(x, x, id=10)
  ret identity.12: bits[8] = identity(y, id=12)
}"#
            )
        );
    }

    #[test]
    fn apply_rejects_empty_sel_cases_list() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[x, x], id=10)
}"#;
        let rule = MatchRewriteRule::from_strings(
            "sel(selector=p, cases=[x, x])",
            "sel(selector=p, cases=[], default=x)",
        )
        .unwrap();
        let err = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap_err();
        assert!(matches!(
            err,
            MatchRewriteRuleApplyError::InvalidRewriteTemplate(ref msg)
                if msg == "sel requires a non-empty cases=[...] list"
        ));
    }

    #[test]
    fn apply_rejects_positional_selector_when_selector_named_arg_is_present() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[x, x], id=10)
}"#;
        let rule = MatchRewriteRule::from_strings(
            "sel(selector=p, cases=[x, x])",
            "sel(x, selector=p, cases=[x, x])",
        )
        .unwrap();
        let err = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap_err();
        assert!(matches!(
            err,
            MatchRewriteRuleApplyError::InvalidRewriteTemplate(ref msg)
                if msg
                    == "sel does not allow positional selector args when selector=<expr> is provided"
        ));
    }

    #[test]
    fn apply_errors_on_ambiguous_match_bindings() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret and.10: bits[8] = and(a, b, id=10)
}"#;
        let rule = MatchRewriteRule::from_strings("and(x)", "x").unwrap();
        let err = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap_err();
        assert_eq!(
            err,
            MatchRewriteRuleApplyError::AmbiguousMatch {
                root: NodeRef { index: 3 },
                match_count: 2,
            }
        );
    }

    #[test]
    fn apply_rejects_type_changing_rewrite() {
        let ir_text = r#"fn t(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  add.10: bits[8] = add(x, y, id=10)
  ret not.11: bits[8] = not(add.10, id=11)
}"#;
        let rule = MatchRewriteRule::from_strings("add(x, y)", "eq(x, y)").unwrap();
        let err = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap_err();
        assert!(matches!(
            err,
            MatchRewriteRuleApplyError::Validation(ref msg)
                if msg == "rewrite would change matched node type from bits[8] to bits[1]"
        ));
    }

    #[test]
    fn canonical_binding_key_distinguishes_literal_widths() {
        let narrow = HashMap::from([(
            "L".to_string(),
            ir_query::Binding::LiteralValue {
                value: IrValue::parse_typed("bits[8]:1").unwrap(),
                ty: Type::Bits(8),
            },
        )]);
        let wide = HashMap::from([(
            "L".to_string(),
            ir_query::Binding::LiteralValue {
                value: IrValue::parse_typed("bits[16]:1").unwrap(),
                ty: Type::Bits(16),
            },
        )]);
        assert_ne!(
            canonical_bindings_key(&narrow),
            canonical_bindings_key(&wide)
        );
    }

    #[test]
    fn literal_type_supports_non_bits_literals() {
        let token = IrValue::make_token();
        assert_eq!(literal_type(&token).unwrap(), Type::Token);

        let array = IrValue::make_array(&[
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            literal_type(&array).unwrap(),
            Type::new_array(Type::Bits(8), 2)
        );

        let tuple = IrValue::make_tuple(&[IrValue::make_ubits(1, 1).unwrap(), array]);
        assert_eq!(
            literal_type(&tuple).unwrap(),
            Type::Tuple(vec![
                Box::new(Type::Bits(1)),
                Box::new(Type::new_array(Type::Bits(8), 2))
            ])
        );
    }

    #[test]
    fn apply_errors_on_ambiguous_literal_bindings_with_different_widths() {
        let ir_text = r#"fn t() -> bits[24] {
  literal.10: bits[8] = literal(value=1, id=10)
  literal.11: bits[16] = literal(value=1, id=11)
  ret concat.12: bits[24] = concat(literal.10, literal.11, id=12)
}"#;
        let rule = MatchRewriteRule::from_strings("concat(literal(L))", "L").unwrap();
        let err = rule
            .apply_to_fn(&parse_fn(ir_text), RewriteApplyMode::FirstMatch)
            .unwrap_err();
        assert_eq!(
            err,
            MatchRewriteRuleApplyError::AmbiguousMatch {
                root: NodeRef { index: 3 },
                match_count: 2,
            }
        );
    }

    #[test]
    fn repeated_runs_are_deterministic() {
        let ir_text = r#"fn t(x: bits[8] id=1, s: bits[8] id=2) -> bits[1] {
  shll.10: bits[8] = shll(x, s, id=10)
  ret bit_slice.11: bits[1] = bit_slice(shll.10, start=0, width=1, id=11)
}"#;
        let rule = MatchRewriteRule::from_strings(
            "bit_slice(shll(x, s), start=0, width=1)",
            "and(bit_slice(x, start=0, width=1), eq(s, $const(value=0, width=$width(s))))",
        )
        .unwrap();
        let f = parse_fn(ir_text);
        let lhs = rule
            .apply_to_fn(&f, RewriteApplyMode::FirstMatch)
            .unwrap()
            .into_rewritten_fn()
            .to_string();
        let rhs = rule
            .apply_to_fn(&f, RewriteApplyMode::FirstMatch)
            .unwrap()
            .into_rewritten_fn()
            .to_string();
        assert_eq!(lhs, rhs);
    }
}
