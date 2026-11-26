// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use super::{
    BindingPattern, Expr, ForIterable, LetBinding, ScopeStatement, Stmt, StructArgument, TestStmt,
    TestStmtKind, Type, WidthSliceType,
};
use SourceNode::{Annotation, Expression, Iterable, Statement, TestStatement};

/// A source node whose children can contain value expressions.
#[derive(Clone, Copy)]
pub(crate) enum SourceNode<'a> {
    Statement(&'a Stmt),
    TestStatement(&'a TestStmt),
    Expression(&'a Expr),
    Iterable(&'a ForIterable),
    Annotation(&'a Type),
}

impl<'a> SourceNode<'a> {
    fn struct_argument(arg: &'a StructArgument) -> Self {
        match arg {
            StructArgument::Type(ty) => Annotation(ty),
            StructArgument::Value(value) => Expression(value),
        }
    }

    /// Queue a node's children for an iterative walk over source expressions.
    pub(crate) fn push_children(self, pending: &mut Vec<Self>) {
        self.queue_children(pending);
    }

    fn queue_children(self, pending: &mut impl SourceQueue<'a>) {
        match self {
            TestStatement(stmt) => match &stmt.kind {
                TestStmtKind::Reg { ty, init, .. } => {
                    pending.push(Annotation(ty));
                    pending.extend(init.iter().map(Expression));
                }
                TestStmtKind::Let(binding) => {
                    pending.push(Expression(&binding.value));
                    pending.extend(binding.ty.iter().map(Annotation));
                }
                TestStmtKind::Property(expr)
                | TestStmtKind::Assign { value: expr, .. }
                | TestStmtKind::Assert {
                    predicate: expr, ..
                } => pending.push(Expression(expr)),
                TestStmtKind::Tick { reset } => pending.extend(reset.iter().map(Expression)),
                TestStmtKind::AssertEq { actual, expected } => {
                    pending.extend([Expression(actual), Expression(expected)]);
                }
                TestStmtKind::For { iterable, body, .. } => {
                    pending.push(Iterable(iterable));
                    pending.extend(body.iter().map(TestStatement));
                }
                TestStmtKind::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    pending.push(Expression(condition));
                    pending.extend(then_body.iter().chain(else_body).map(TestStatement));
                }
            },
            Statement(stmt) => match stmt {
                Stmt::ConstAssert(assertion) => pending.push(Expression(&assertion.predicate)),
                Stmt::Assert { predicate, .. } => pending.push(Expression(predicate)),
                Stmt::Reg { name, ty, reset } => {
                    pending.extend(reset.iter().map(Expression));
                    pending.bind_name(name);
                    pending.push(Annotation(ty));
                }
                Stmt::Let { name, value, ty } => {
                    pending.bind_name(name);
                    pending.push(Expression(value));
                    pending.extend(ty.iter().map(Annotation));
                }
                Stmt::LetPattern(binding) => pending.push_let(binding),
                Stmt::Wire { name, ty } => {
                    pending.bind_name(name);
                    pending.push(Annotation(ty));
                }
                Stmt::Write { value, .. } | Stmt::DriveWire { value, .. } => {
                    pending.push(Expression(value));
                }
                Stmt::If { condition, body } => {
                    pending.leave_scope();
                    pending.extend(body.iter().rev().map(Statement));
                    pending.push(Expression(condition));
                }
            },
            Iterable(ForIterable::Array(expr)) => pending.push(Expression(expr)),
            Iterable(ForIterable::Range { start, limit, .. }) => {
                pending.extend([limit, start].map(Expression));
            }
            Annotation(ty) => match ty {
                Type::ParametricBits { signed, width } => {
                    pending.extend([signed.as_ref(), width.as_ref()].map(Expression));
                }
                Type::ParametricArray { element, size } => {
                    pending.push(Annotation(element));
                    pending.push(Expression(size));
                }
                Type::ParametricStruct { args, .. } => {
                    pending.extend(args.iter().map(Expression));
                }
                Type::TypeApplied { args, .. } => pending.extend(args.iter().map(Annotation)),
                Type::MixedApplied { args, .. } => {
                    pending.extend(args.iter().map(Self::struct_argument));
                }
                Type::Array { element, .. } => pending.push(Annotation(element)),
                Type::Tuple(fields) => pending.extend(fields.iter().map(Annotation)),
                Type::Bits { .. }
                | Type::Bool
                | Type::Named(_)
                | Type::Enum { .. }
                | Type::Applied { .. } => { /* These types contain no value expressions. */ }
            },
            Expression(expr) => match expr {
                Expr::Array(args) | Expr::Tuple(args) | Expr::Call { args, .. } => {
                    pending.extend(args.iter().map(Expression));
                }
                Expr::ArrayLiteral {
                    annotation,
                    members,
                    ..
                } => {
                    pending.extend(annotation.iter().map(Annotation));
                    pending.extend(members.iter().map(Expression));
                }
                Expr::MapBeforeUse {
                    args, builtin_args, ..
                } => {
                    pending.extend(
                        args.iter()
                            .chain(builtin_args.iter().flatten())
                            .map(Expression),
                    );
                }
                Expr::ParametricCall {
                    args, parametrics, ..
                } => {
                    pending.extend(args.iter().chain(parametrics).map(Expression));
                }
                Expr::SelectedOperand(value) => pending.push(Expression(value)),
                Expr::UnrollFor(fold) => {
                    pending.leave_scope();
                    pending.push(Expression(&fold.result));
                    pending.extend(fold.body.iter().rev().map(Statement));
                    pending.push_loop_inputs(
                        &fold.element,
                        &fold.accumulator,
                        &fold.annotation,
                        &fold.iterable,
                        &fold.seed,
                    );
                }
                Expr::UnrollMap(map) => {
                    pending.leave_scope();
                    pending.push(Expression(&map.result));
                    pending.extend(map.body.iter().rev().map(Statement));
                    pending.bind_pattern(&map.element);
                    pending.push(Iterable(&map.iterable));
                }
                Expr::For(fold) => {
                    pending.leave_scope();
                    pending.push(Expression(&fold.body));
                    pending.push_loop_inputs(
                        &fold.element,
                        &fold.accumulator,
                        &fold.annotation,
                        &fold.iterable,
                        &fold.seed,
                    );
                }
                Expr::Map(map) => {
                    pending.leave_scope();
                    pending.push(Expression(&map.body));
                    pending.extend(map.result_type.iter().map(Annotation));
                    pending.bind_name(&map.parameter);
                    pending.extend(map.parameter_type.iter().map(Annotation));
                    pending.push(Iterable(&map.iterable));
                }
                Expr::NamedMap {
                    iterable,
                    parametrics,
                    ..
                } => {
                    pending.push(Iterable(iterable));
                    pending.extend(parametrics.iter().map(Expression));
                }
                Expr::Scope { statements, result } => {
                    pending.leave_scope();
                    pending.push(Expression(result));
                    for statement in statements.iter().rev() {
                        match statement {
                            ScopeStatement::Let(binding) => pending.push_let(binding),
                            ScopeStatement::Const(def) => {
                                pending.bind_name(&def.name);
                                pending.push(Expression(&def.value));
                                pending.extend(def.ty.iter().map(Annotation));
                            }
                            ScopeStatement::TypeAlias(alias) => {
                                pending.bind_name(&alias.name);
                                pending.push(Annotation(&alias.ty));
                            }
                            ScopeStatement::ConstAssert(assertion) => {
                                pending.push(Expression(&assertion.predicate));
                            }
                            ScopeStatement::Expression(expr) => pending.push(Expression(expr)),
                        }
                    }
                }
                Expr::If {
                    condition,
                    then_value,
                    else_value,
                }
                | Expr::ConstIf {
                    condition,
                    then_value,
                    else_value,
                } => {
                    pending.extend([condition.as_ref(), then_value, else_value].map(Expression));
                }
                Expr::Match { subject, arms } => {
                    for arm in arms.iter().rev() {
                        pending.leave_scope();
                        pending.push(Expression(&arm.value));
                        for name in arm.binding_names() {
                            pending.bind_name(name);
                        }
                        pending.extend(
                            arm.patterns
                                .iter()
                                .flat_map(|p| p.literals())
                                .map(Expression),
                        );
                    }
                    pending.push(Expression(subject));
                }
                Expr::Struct { fields, base, .. }
                | Expr::AppliedStruct { fields, base, .. }
                | Expr::ParametricStruct { fields, base, .. }
                | Expr::TypeAppliedStruct { fields, base, .. }
                | Expr::MixedAppliedStruct { fields, base, .. } => {
                    match expr {
                        Expr::ParametricStruct { args, .. } => {
                            pending.extend(args.iter().map(Expression));
                        }
                        Expr::TypeAppliedStruct { args, .. } => {
                            pending.extend(args.iter().map(Annotation));
                        }
                        Expr::MixedAppliedStruct { args, .. } => {
                            pending.extend(args.iter().map(Self::struct_argument));
                        }
                        _ => { /* This constructor has no source arguments to visit. */ }
                    }
                    pending.extend(fields.iter().map(|(_, value)| Expression(value)));
                    pending.extend(base.as_deref().map(|splat| Expression(&splat.value)));
                }
                Expr::Field { value, .. } | Expr::Unary { value, .. } => {
                    pending.push(Expression(value));
                }
                Expr::Cast { value, ty } => {
                    pending.push(Expression(value));
                    pending.push(Annotation(ty));
                }
                Expr::TypeBuiltin { types, args, .. } => {
                    pending.extend(types.iter().map(Annotation));
                    pending.extend(args.iter().map(Expression));
                }
                Expr::Index { value, index }
                | Expr::Binary {
                    left: value,
                    right: index,
                    ..
                } => {
                    pending.extend([value.as_ref(), index.as_ref()].map(Expression));
                }
                Expr::WidthSlice {
                    value,
                    start,
                    width,
                } => {
                    pending.extend([value.as_ref(), start.as_ref()].map(Expression));
                    match width {
                        WidthSliceType::Bits { signed, bit_count } => {
                            pending.extend([signed.as_ref(), bit_count.as_ref()].map(Expression));
                        }
                        WidthSliceType::Concrete(ty) => pending.push(Annotation(ty)),
                    }
                }
                Expr::Zero(ty) | Expr::Literal { ty: Some(ty), .. } => {
                    pending.push(Annotation(ty));
                }
                Expr::Name(_)
                | Expr::Bool(_)
                | Expr::ByteString(_)
                | Expr::Literal { ty: None, .. } => { /* Leaf values have no child expressions. */ }
            },
        }
    }
}

/// Traverse a type annotation or expression with its lexical binders in scope.
#[derive(Default)]
pub(crate) struct ScopedSourceNodes<'a> {
    pending: Vec<ScopeStep<'a>>,
    locals: BTreeSet<&'a str>,
    introduced: Vec<&'a str>,
}

/// Actions between source nodes introduce bindings after their initializers
/// and remove them on leaving a nested lexical scope.
enum ScopeStep<'a> {
    Visit(SourceNode<'a>),
    Bind(&'a str),
    BindPattern(&'a BindingPattern),
    Leave(usize),
}

impl<'a> ScopedSourceNodes<'a> {
    pub(crate) fn push(&mut self, node: SourceNode<'a>) {
        SourceQueue::push(self, node);
    }

    pub(crate) fn is_local(&self, name: &str) -> bool {
        self.locals
            .contains(name.split_once("::").map_or(name, |(head, _)| head))
    }

    fn bind(&mut self, name: &'a str) {
        if name != "_" && self.locals.insert(name) {
            self.introduced.push(name);
        }
    }

    /// Return the next node with only its preceding lexical binders visible.
    pub(crate) fn next(&mut self) -> Option<SourceNode<'a>> {
        while let Some(step) = self.pending.pop() {
            match step {
                ScopeStep::Visit(node) => {
                    node.queue_children(self);
                    return Some(node);
                }
                ScopeStep::Bind(name) => self.bind(name),
                ScopeStep::BindPattern(pattern) => {
                    pattern.visit_names(&mut |name| self.bind(name));
                }
                ScopeStep::Leave(start) => {
                    for name in self.introduced.drain(start..) {
                        self.locals.remove(name);
                    }
                }
            }
        }
        None
    }
}

/// Schedule children in stack order. Scope-aware consumers also receive the
/// binding after its initializer and the scope exit after its final child.
trait SourceQueue<'a>: Extend<SourceNode<'a>> {
    fn push(&mut self, node: SourceNode<'a>);

    fn push_let(&mut self, binding: &'a LetBinding) {
        self.bind_pattern(&binding.pattern);
        self.push(Expression(&binding.value));
        self.extend(binding.ty.iter().map(Annotation));
    }

    /// Visit the iterable and seed in the enclosing scope before binding the
    /// element and accumulator for the loop body.
    fn push_loop_inputs(
        &mut self,
        element: &'a BindingPattern,
        accumulator: &'a BindingPattern,
        annotation: &'a Option<(Type, Type)>,
        iterable: &'a ForIterable,
        seed: &'a Expr,
    ) {
        self.bind_pattern(accumulator);
        self.bind_pattern(element);
        self.push(Expression(seed));
        self.push(Iterable(iterable));
        if let Some((element, accumulator)) = annotation {
            self.extend([accumulator, element].map(Annotation));
        }
    }

    fn bind_name(&mut self, name: &'a str);
    fn bind_pattern(&mut self, pattern: &'a BindingPattern);
    fn leave_scope(&mut self);
}

impl<'a> SourceQueue<'a> for Vec<SourceNode<'a>> {
    fn push(&mut self, node: SourceNode<'a>) {
        Vec::push(self, node);
    }

    // Syntax-only visitors collect children regardless of lexical bindings.
    fn bind_name(&mut self, _: &'a str) {}
    fn bind_pattern(&mut self, _: &'a BindingPattern) {}
    fn leave_scope(&mut self) {}
}

impl<'a> Extend<SourceNode<'a>> for ScopedSourceNodes<'a> {
    fn extend<T: IntoIterator<Item = SourceNode<'a>>>(&mut self, nodes: T) {
        self.pending.extend(nodes.into_iter().map(ScopeStep::Visit));
    }
}

impl<'a> SourceQueue<'a> for ScopedSourceNodes<'a> {
    fn push(&mut self, node: SourceNode<'a>) {
        self.pending.push(ScopeStep::Visit(node));
    }

    fn bind_name(&mut self, name: &'a str) {
        self.pending.push(ScopeStep::Bind(name));
    }
    fn bind_pattern(&mut self, pattern: &'a BindingPattern) {
        self.pending.push(ScopeStep::BindPattern(pattern));
    }
    fn leave_scope(&mut self) {
        self.pending.push(ScopeStep::Leave(self.introduced.len()));
    }
}
