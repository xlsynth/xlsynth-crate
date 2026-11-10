// SPDX-License-Identifier: Apache-2.0

use super::{Expr, ForIterable, LetBinding, Stmt, Type, WidthSliceType};
use SourceNode::{Annotation, Expression, Iterable, Statement};

/// A source node whose children can contain value expressions.
pub(crate) enum SourceNode<'a> {
    Statement(&'a Stmt),
    Expression(&'a Expr),
    Iterable(&'a ForIterable),
    Annotation(&'a Type),
}

impl<'a> SourceNode<'a> {
    /// Queue a node's children for an iterative walk over source expressions.
    pub(crate) fn push_children(self, pending: &mut Vec<Self>) {
        match self {
            Statement(stmt) => match stmt {
                Stmt::Reg { ty, reset, .. } => {
                    pending.push(Annotation(ty));
                    pending.extend(reset.iter().map(Expression));
                }
                Stmt::Let { value, ty, .. } | Stmt::LetPattern(LetBinding { value, ty, .. }) => {
                    pending.push(Expression(value));
                    pending.extend(ty.iter().map(Annotation));
                }
                Stmt::Write { value, .. } | Stmt::Connect { value, .. } => {
                    pending.push(Expression(value));
                }
                Stmt::If { condition, body } => {
                    pending.push(Expression(condition));
                    pending.extend(body.iter().map(Statement));
                }
                Stmt::Inst { .. } => { /* Explicit instances have no value expressions. */ }
            },
            Iterable(ForIterable::Array(expr)) => pending.push(Expression(expr)),
            Iterable(ForIterable::Range { start, limit, .. }) => {
                pending.extend([start, limit].map(Expression));
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
                Expr::UnrollFor(fold) => {
                    pending.extend(fold.body.iter().map(Statement));
                    if let Some((element, accumulator)) = &fold.annotation {
                        pending.extend([element, accumulator].map(Annotation));
                    }
                    pending.extend(
                        [&fold.start, &fold.limit, &fold.seed, &fold.result].map(Expression),
                    );
                }
                Expr::For(fold) => {
                    pending.push(Iterable(&fold.iterable));
                    if let Some((element, accumulator)) = &fold.annotation {
                        pending.extend([element, accumulator].map(Annotation));
                    }
                    pending.extend([&fold.seed, &fold.body].map(Expression));
                }
                Expr::Map(map) => {
                    pending.push(Iterable(&map.iterable));
                    pending.push(Expression(&map.body));
                    pending.extend(
                        map.parameter_type
                            .iter()
                            .chain(&map.result_type)
                            .map(Annotation),
                    );
                }
                Expr::NamedMap {
                    iterable,
                    parametrics,
                    ..
                } => {
                    pending.push(Iterable(iterable));
                    pending.extend(parametrics.iter().map(Expression));
                }
                Expr::Scope { bindings, result } => {
                    pending.push(Expression(result));
                    for binding in bindings {
                        pending.push(Expression(&binding.value));
                        pending.extend(binding.ty.iter().map(Annotation));
                    }
                }
                Expr::If {
                    condition,
                    then_value,
                    else_value,
                } => {
                    pending.extend([condition.as_ref(), then_value, else_value].map(Expression));
                }
                Expr::Match { subject, arms } => {
                    pending.push(Expression(subject));
                    for arm in arms {
                        pending.push(Expression(&arm.value));
                        pending.extend(
                            arm.patterns
                                .iter()
                                .flat_map(|p| p.literals())
                                .map(Expression),
                        );
                    }
                }
                Expr::Struct { fields, .. }
                | Expr::AppliedStruct { fields, .. }
                | Expr::ParametricStruct { fields, .. }
                | Expr::TypeAppliedStruct { fields, .. } => {
                    match expr {
                        Expr::ParametricStruct { args, .. } => {
                            pending.extend(args.iter().map(Expression));
                        }
                        Expr::TypeAppliedStruct { args, .. } => {
                            pending.extend(args.iter().map(Annotation));
                        }
                        _ => { /* This constructor has no source arguments to visit. */ }
                    }
                    pending.extend(fields.iter().map(|(_, value)| Expression(value)));
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
