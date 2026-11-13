// SPDX-License-Identifier: Apache-2.0

use super::{MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE};
use crate::ast::SourceNode::{Annotation, Expression, Statement, TestStatement};
use crate::ast::{
    CallbackParam, Expr, MatchArm, MatchPattern, Module, Param, ParametricBinding, ScopeStatement,
    SourceNode, Stmt, Type,
};

/// Bound match patterns throughout a supplied AST before any recursive clones.
pub(super) fn validate_source_match_patterns(module: &Module) -> Result<(), String> {
    let mut pending = module_nodes(module, false);
    while let Some(node) = pending.pop() {
        if let Expression(Expr::Match { arms, .. }) = node {
            validate_match_pattern_size(arms)?;
        }
        node.push_children(&mut pending);
    }
    Ok(())
}

/// A helper used transitively by an assertion also needs its constexpr inputs.
pub(super) fn has_const_assert(module: &Module, include_tests: bool) -> bool {
    if !module.const_asserts.is_empty() {
        return true;
    }
    let mut pending = module_nodes(module, include_tests);
    while let Some(node) = pending.pop() {
        if matches!(node, Statement(Stmt::ConstAssert(_))) {
            return true;
        }
        if let Expression(Expr::Scope { statements, .. }) = node
            && statements
                .iter()
                .any(|s| matches!(s, ScopeStatement::ConstAssert(_)))
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
    use crate::ast::{Expr, MatchPattern};
    use crate::{check_module, parse_module};

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
