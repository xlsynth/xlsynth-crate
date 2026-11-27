// SPDX-License-Identifier: Apache-2.0

use super::{MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE};
use crate::ast::SourceNode::{Annotation, Expression, Statement, TestStatement};
use crate::ast::{
    CallbackParam, Expr, MatchArm, MatchPattern, Module, Param, ParametricBinding, ScopeStatement,
    SourceNode, Stmt, Type,
};
use std::collections::HashSet;

/// Bound match patterns throughout a supplied AST before any recursive clones.
pub(super) fn validate_source_match_patterns(module: &Module) -> Result<(), String> {
    let mut pending = module_nodes(module, false);
    while let Some(node) = pending.pop() {
        if let Expression(Expr::Match { arms, .. }) = node {
            validate_match_pattern_size(arms)?;
            validate_tuple_pattern_bindings(arms)?;
        }
        node.push_children(&mut pending);
    }
    Ok(())
}

/// Check rest markers, lexical references and duplicate tuple spellings before
/// preparing the arm-local bindings.
pub(super) fn validate_tuple_pattern_bindings(arms: &[MatchArm]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for arm in arms {
        for pattern in &arm.patterns {
            let mut bindings = HashSet::new();
            let mut contains_literal = false;
            for (depth, node) in pattern.nodes() {
                match node {
                    MatchPattern::Binding(name) => {
                        if arm.patterns.len() != 1 {
                            return Err(format!(
                                "cannot bind '{name}' in a match arm with multiple patterns"
                            ));
                        }
                        if !crate::parser::is_match_binding_name(name) {
                            return Err(format!("invalid match binding '{name}'"));
                        }
                        if !bindings.insert(name) {
                            return Err(format!("duplicate match binding '{name}'"));
                        }
                    }
                    MatchPattern::Tuple(fields) => {
                        if fields
                            .iter()
                            .filter(|f| matches!(f, MatchPattern::Rest))
                            .nth(1)
                            .is_some()
                        {
                            return Err("duplicate rest marker in tuple pattern".into());
                        }
                    }
                    MatchPattern::Rest if depth == 1 => {
                        return Err("rest marker requires a tuple pattern".into());
                    }
                    MatchPattern::Literal(_) | MatchPattern::Range { .. } => {
                        contains_literal = true
                    }
                    MatchPattern::Wildcard | MatchPattern::Rest => { /* No name at this node. */ }
                }
            }
            // Source AST literals omit spelling distinctions such as u8 vs
            // uN[8]. The parser checks exact spellings for tuples that contain
            // literals.
            if !contains_literal && !seen.insert(pattern) {
                return Err(if matches!(pattern, MatchPattern::Wildcard) {
                    "exact-duplicate wildcard match pattern".into()
                } else {
                    "exact-duplicate match pattern".into()
                });
            }
        }
    }
    Ok(())
}

/// Local declarations, assertions, and range bounds can evaluate helpers, so
/// retain the lexical operands used to determine constexpr values.
pub(super) fn needs_constexpr_operands(module: &Module, include_tests: bool) -> bool {
    if !module.const_asserts.is_empty() {
        return true;
    }
    let mut pending = module_nodes(module, include_tests);
    while let Some(node) = pending.pop() {
        if matches!(
            node,
            Statement(Stmt::ConstAssert(_)) | Expression(Expr::Range { .. })
        ) {
            return true;
        }
        if let Expression(Expr::Scope { statements, .. }) = node
            && statements.iter().any(|s| {
                matches!(
                    s,
                    ScopeStatement::ConstAssert(_)
                        | ScopeStatement::Const(_)
                        | ScopeStatement::TypeAlias(_)
                )
            })
        {
            return true;
        }
        node.push_children(&mut pending);
    }
    false
}

/// Source roots for checking an AST, including the procedural harnesses when
/// enabled.
fn module_nodes(module: &Module, include_tests: bool) -> Vec<SourceNode<'_>> {
    let mut pending = Vec::new();
    pending.extend(
        module
            .type_aliases
            .iter()
            .map(|alias| Annotation(&alias.ty)),
    );
    for def in &module.structs {
        pending.extend(def.fields.iter().map(|p| Annotation(&p.ty)));
        pending.extend(parametrics(&def.value_parametrics));
    }
    for def in &module.enums {
        pending.extend(def.underlying.iter().map(Annotation));
        pending.extend(def.members.iter().map(|m| Expression(&m.value)));
    }
    for def in &module.constants {
        pending.extend(def.ty.iter().map(Annotation));
        pending.push(Expression(&def.value));
    }
    pending.extend(
        module
            .const_asserts
            .values()
            .map(|a| Expression(&a.predicate)),
    );
    for f in &module.externs {
        pending.extend(signature(&f.params, &f.result));
    }
    for f in &module.functions {
        pending.extend(signature(&f.params, &f.result));
        pending.extend(callbacks(&f.callback_params));
        pending.extend(parametrics(&f.value_parametrics));
        pending.push(Expression(&f.body));
    }
    for b in &module.extern_blocks {
        pending.extend(signature(&b.params, &b.result));
    }
    if include_tests {
        for test in &module.tests {
            pending.extend(test.statements.iter().map(TestStatement));
            if let Some(q) = &test.quickcheck {
                pending.extend(signature(&q.params, &q.result));
            }
        }
    }
    for b in module.blocks.iter().chain(
        module
            .invariants
            .iter()
            .filter(|_| include_tests)
            .map(|i| &i.block),
    ) {
        pending.extend(signature(&b.params, &b.result));
        pending.extend(callbacks(&b.callback_params));
        pending.extend(parametrics(&b.value_parametrics));
        pending.extend(b.statements.iter().map(Statement));
        pending.push(Expression(&b.output));
    }
    pending
}

fn signature<'a>(params: &'a [Param], result: &'a Type) -> impl Iterator<Item = SourceNode<'a>> {
    params
        .iter()
        .map(|p| Annotation(&p.ty))
        .chain([Annotation(result)])
}

fn callbacks(formals: &[CallbackParam]) -> impl Iterator<Item = SourceNode<'_>> {
    formals
        .iter()
        .flat_map(|c| c.params.iter().chain([&c.result]).map(Annotation))
}

fn parametrics(formals: &[ParametricBinding]) -> impl Iterator<Item = SourceNode<'_>> {
    formals.iter().flat_map(|p| {
        [Annotation(&p.ty)]
            .into_iter()
            .chain(p.default.iter().map(Expression))
    })
}

/// Bound the depth and total node count across a match's alternatives.
pub(super) fn validate_match_pattern_size(arms: &[MatchArm]) -> Result<(), String> {
    for (index, (depth, _)) in arms
        .iter()
        .flat_map(|arm| &arm.patterns)
        .flat_map(MatchPattern::nodes)
        .enumerate()
    {
        if depth > MAX_ELABORATION_DEPTH || index >= MAX_TYPE_SIZE {
            return Err("match pattern exceeds the elaboration limit".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_tuple_pattern_bindings;
    use crate::ast::{Expr, MatchArm, MatchPattern};
    use crate::{check_module, parse_module};

    #[test]
    fn rejects_malformed_programmatic_rest_and_match_bindings() {
        for (pattern, diagnostic) in [
            (MatchPattern::Rest, "rest marker requires a tuple pattern"),
            (
                MatchPattern::Tuple(vec![MatchPattern::Rest, MatchPattern::Rest]),
                "duplicate rest marker in tuple pattern",
            ),
            (
                MatchPattern::Tuple(vec![
                    MatchPattern::Binding("v".into()),
                    MatchPattern::Tuple(vec![MatchPattern::Binding("v".into())]),
                ]),
                "duplicate match binding 'v'",
            ),
        ] {
            let arm = MatchArm {
                patterns: vec![pattern],
                value: Expr::Bool(false),
            };
            assert_eq!(
                validate_tuple_pattern_bindings(&[arm]).unwrap_err(),
                diagnostic
            );
        }
    }

    /// Find the pattern inside a tuple that is cloned during scope capture.
    fn scoped_tuple_pattern(body: &mut Expr) -> Option<&mut MatchPattern> {
        let Expr::Call { args, .. } = body else {
            return None;
        };
        let Expr::Scope { result, .. } = args.first_mut()? else {
            return None;
        };
        let Expr::Tuple(fields) = result.as_mut() else {
            return None;
        };
        let Expr::Match { arms, .. } = fields.first_mut()? else {
            return None;
        };
        arms.first_mut()?.patterns.first_mut()
    }

    #[test]
    fn rejects_deep_programmatic_match_pattern_before_cloning_its_scope() {
        let source = r#"
            fn project(x: (bool,)) -> bool { x.0 }
            fn f(x: bool) -> bool {
                project({
                    let y = x;
                    (match (y,) { (_,) => y },)
                })
            }
        "#;
        let mut module = parse_module(source).unwrap();
        check_module(&module).unwrap();
        let oversized = (0..16_384).fold(MatchPattern::Wildcard, |field, _| {
            MatchPattern::Tuple(vec![field])
        });
        *scoped_tuple_pattern(&mut module.functions[1].body).unwrap() = oversized;
        let result = check_module(&module);
        // Dismantle the caller-owned malformed pattern without recursive drop.
        let mut pattern = std::mem::replace(
            scoped_tuple_pattern(&mut module.functions[1].body).unwrap(),
            MatchPattern::Wildcard,
        );
        while let MatchPattern::Tuple(mut fields) = pattern {
            pattern = fields.pop().unwrap();
        }
        assert_eq!(
            result.unwrap_err(),
            "match pattern exceeds the elaboration limit"
        );
    }
}
