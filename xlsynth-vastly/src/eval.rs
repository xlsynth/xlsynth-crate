// SPDX-License-Identifier: Apache-2.0

use crate::Env;
use crate::Error;
use crate::EvalResult;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::BinaryOp;
use crate::ast::Expr;
use crate::ast::UnaryOp;
use crate::parser::parse_expr;
use crate::sv_ast::Span;
use crate::value::LogicBit;
use crate::value::RelOp;

pub fn eval_expr(expr: &str, env: &Env) -> Result<EvalResult> {
    let ast = parse_expr(expr)?;
    let value = eval_ast_with_calls(&ast, env, None, None)?;
    Ok(EvalResult { value })
}

pub trait CallResolver {
    fn call(&self, name: &str, args: &[Value4], expected_width: Option<u32>) -> Result<Value4>;
}

pub trait EvalObserver {
    fn on_ternary(&mut self, context_span: Option<Span>, expr: &Expr, cond: &Value4);
}

fn merged_signedness(lhs: &Value4, rhs: &Value4) -> Signedness {
    if lhs.signedness == Signedness::Signed && rhs.signedness == Signedness::Signed {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    }
}

fn operand_with_own_sign_ctx(
    v: Value4,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
) -> Value4 {
    let v = if let Some(signedness) = expected_signedness {
        v.with_signedness(signedness)
    } else {
        v
    };
    let Some(width) = expected_width else {
        return v;
    };
    if width <= v.width { v } else { v.resize(width) }
}

fn operand_with_merged_sign_ctx(
    lhs: Value4,
    rhs: Value4,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
) -> (Value4, Value4) {
    let width = expected_width.unwrap_or(0).max(lhs.width.max(rhs.width));
    let signedness = expected_signedness.unwrap_or_else(|| merged_signedness(&lhs, &rhs));
    let lhs = lhs.with_signedness(signedness).resize(width);
    let rhs = rhs.with_signedness(signedness).resize(width);
    (lhs, rhs)
}

fn ternary_branch_needs_width_recontext(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::UnbasedUnsized(_)
            | Expr::Call { .. }
            | Expr::Unary { .. }
            | Expr::Binary { .. }
            | Expr::Ternary { .. }
    )
}

fn unary_operand_expected_width(op: UnaryOp, expected_width: Option<u32>) -> Option<u32> {
    match op {
        UnaryOp::BitNot | UnaryOp::UnaryPlus | UnaryOp::UnaryMinus => expected_width,
        UnaryOp::LogicalNot
        | UnaryOp::ReduceAnd
        | UnaryOp::ReduceNand
        | UnaryOp::ReduceOr
        | UnaryOp::ReduceNor
        | UnaryOp::ReduceXor
        | UnaryOp::ReduceXnor => None,
    }
}

fn binary_operand_expected_widths(
    op: BinaryOp,
    expected_width: Option<u32>,
) -> (Option<u32>, Option<u32>) {
    match op {
        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Mod
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor => (expected_width, expected_width),
        BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sshr => (expected_width, None),
        BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge => (None, None),
        BinaryOp::Eq | BinaryOp::Neq | BinaryOp::CaseEq | BinaryOp::CaseNeq => (None, None),
    }
}

fn binary_operand_expected_signednesses(
    op: BinaryOp,
    expected_signedness: Option<Signedness>,
) -> (Option<Signedness>, Option<Signedness>) {
    match op {
        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Mod
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor => (expected_signedness, expected_signedness),
        BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sshr => (expected_signedness, None),
        BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::Eq
        | BinaryOp::Neq
        | BinaryOp::CaseEq
        | BinaryOp::CaseNeq => (None, None),
    }
}

pub(crate) fn eval_unary_op(
    op: UnaryOp,
    v: Value4,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
) -> Value4 {
    match op {
        UnaryOp::LogicalNot => v.logical_not(),
        UnaryOp::BitNot => {
            operand_with_own_sign_ctx(v, expected_width, expected_signedness).bitwise_not()
        }
        UnaryOp::UnaryPlus => operand_with_own_sign_ctx(v, expected_width, expected_signedness),
        UnaryOp::UnaryMinus => {
            operand_with_own_sign_ctx(v, expected_width, expected_signedness).unary_minus()
        }
        UnaryOp::ReduceAnd => v.reduce_and(),
        UnaryOp::ReduceNand => v.reduce_and().logical_not(),
        UnaryOp::ReduceOr => v.reduce_or(),
        UnaryOp::ReduceNor => v.reduce_or().logical_not(),
        UnaryOp::ReduceXor => v.reduce_xor(),
        UnaryOp::ReduceXnor => v.reduce_xor().logical_not(),
    }
}

pub(crate) fn eval_binary_op(
    op: BinaryOp,
    lhs: Value4,
    rhs: Value4,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
) -> Value4 {
    match op {
        BinaryOp::Add => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.add(&rhs)
        }
        BinaryOp::Sub => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.sub(&rhs)
        }
        BinaryOp::Mul => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.mul(&rhs)
        }
        BinaryOp::Div => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.div(&rhs)
        }
        BinaryOp::Mod => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.modu(&rhs)
        }
        BinaryOp::Shl => {
            operand_with_own_sign_ctx(lhs, expected_width, expected_signedness).shl(&rhs)
        }
        BinaryOp::Shr => {
            operand_with_own_sign_ctx(lhs, expected_width, expected_signedness).shr(&rhs)
        }
        BinaryOp::Sshr => {
            operand_with_own_sign_ctx(lhs, expected_width, expected_signedness).sshr(&rhs)
        }
        BinaryOp::BitAnd => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.bitwise_and(&rhs)
        }
        BinaryOp::BitOr => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.bitwise_or(&rhs)
        }
        BinaryOp::BitXor => {
            let (lhs, rhs) =
                operand_with_merged_sign_ctx(lhs, rhs, expected_width, expected_signedness);
            lhs.bitwise_xor(&rhs)
        }
        BinaryOp::LogicalAnd => lhs.logical_and(&rhs),
        BinaryOp::LogicalOr => lhs.logical_or(&rhs),
        BinaryOp::Lt => lhs.cmp_rel(&rhs, RelOp::Lt),
        BinaryOp::Le => lhs.cmp_rel(&rhs, RelOp::Le),
        BinaryOp::Gt => lhs.cmp_rel(&rhs, RelOp::Gt),
        BinaryOp::Ge => lhs.cmp_rel(&rhs, RelOp::Ge),
        BinaryOp::Eq => lhs.eq_logical(&rhs),
        BinaryOp::Neq => lhs.neq_logical(&rhs),
        BinaryOp::CaseEq => lhs.eq_case(&rhs),
        BinaryOp::CaseNeq => lhs.neq_case(&rhs),
    }
}

pub(crate) fn eval_ast_with_calls(
    expr: &Expr,
    env: &Env,
    calls: Option<&dyn CallResolver>,
    expected_width: Option<u32>,
) -> Result<Value4> {
    eval_ast_with_calls_inner(expr, env, calls, expected_width, None, None, None)
}

fn eval_ast_with_calls_inner(
    expr: &Expr,
    env: &Env,
    calls: Option<&dyn CallResolver>,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
    mut observer: Option<&mut dyn EvalObserver>,
    context_span: Option<Span>,
) -> Result<Value4> {
    match expr {
        Expr::Ident(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownIdentifier(name.clone())),
        Expr::Literal(v) => Ok(v.clone()),
        Expr::UnbasedUnsized(bit) => {
            let w = expected_width.unwrap_or(1);
            Ok(Value4::new(
                w,
                crate::Signedness::Unsigned,
                vec![*bit; w as usize],
            ))
        }
        Expr::Call { name, args } => {
            if name == "$signed" || name == "$unsigned" {
                if args.len() != 1 {
                    return Err(Error::Parse(format!(
                        "builtin cast `{name}` expects 1 argument, got {}",
                        args.len()
                    )));
                }
                let v = if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        &args[0],
                        env,
                        calls,
                        expected_width,
                        expected_signedness,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(
                        &args[0],
                        env,
                        calls,
                        expected_width,
                        expected_signedness,
                        None,
                        context_span,
                    )?
                };
                return Ok(if name == "$signed" {
                    v.with_signedness(crate::Signedness::Signed)
                } else {
                    v.with_signedness(crate::Signedness::Unsigned)
                });
            }
            let Some(calls) = calls else {
                return Err(Error::Parse(format!(
                    "function call `{name}` not supported in this context"
                )));
            };
            let mut avs: Vec<Value4> = Vec::with_capacity(args.len());
            for a in args {
                let v = if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        a,
                        env,
                        Some(calls),
                        None,
                        None,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(a, env, Some(calls), None, None, None, context_span)?
                };
                avs.push(v);
            }
            calls.call(name, &avs, expected_width)
        }
        Expr::Concat(parts) => {
            let mut vs: Vec<Value4> = Vec::with_capacity(parts.len());
            for p in parts {
                let v = if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(p, env, calls, None, None, Some(obs), context_span)?
                } else {
                    eval_ast_with_calls_inner(p, env, calls, None, None, None, context_span)?
                };
                vs.push(v);
            }
            Ok(Value4::concat(&vs))
        }
        Expr::Replicate { count, expr } => {
            let c = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(count, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(count, env, calls, None, None, None, context_span)?
            };
            let count_u = c.to_u32_saturating_if_known().unwrap_or(0);
            let v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(expr, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(expr, env, calls, None, None, None, context_span)?
            };
            Ok(Value4::replicate(count_u, &v))
        }
        Expr::Index { expr, index } => {
            let v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(expr, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(expr, env, calls, None, None, None, context_span)?
            };
            let idx_v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(index, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(index, env, calls, None, None, None, context_span)?
            };
            match idx_v.to_u32_saturating_if_known() {
                Some(idx_u) => Ok(v.index(idx_u)),
                None => Ok(Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])),
            }
        }
        Expr::Slice { expr, msb, lsb } => {
            let v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(expr, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(expr, env, calls, None, None, None, context_span)?
            };
            let msb_v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(msb, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(msb, env, calls, None, None, None, context_span)?
            };
            let lsb_v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(lsb, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(lsb, env, calls, None, None, None, context_span)?
            };
            match (
                msb_v.to_u32_saturating_if_known(),
                lsb_v.to_u32_saturating_if_known(),
            ) {
                (Some(msb_u), Some(lsb_u)) => Ok(v.slice(msb_u, lsb_u)),
                _ => {
                    let width = expected_width.unwrap_or(1);
                    Ok(Value4::new(
                        width,
                        Signedness::Unsigned,
                        vec![LogicBit::X; width as usize],
                    ))
                }
            }
        }
        Expr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => {
            let v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(expr, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(expr, env, calls, None, None, None, context_span)?
            };
            let base_v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(base, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(base, env, calls, None, None, None, context_span)?
            };
            let width_v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(width, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(width, env, calls, None, None, None, context_span)?
            };
            let width_u = width_v.to_u32_saturating_if_known().ok_or_else(|| {
                Error::Parse("indexed slice width is unknown (contains x/z)".to_string())
            })?;
            match base_v.to_u32_saturating_if_known() {
                Some(base_u) => Ok(v.indexed_slice(base_u, width_u, *upward)),
                None => Ok(Value4::new(
                    width_u,
                    Signedness::Unsigned,
                    vec![LogicBit::X; width_u as usize],
                )),
            }
        }
        Expr::Unary { op, expr } => {
            let child_expected_width = unary_operand_expected_width(*op, expected_width);
            let v = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(
                    expr,
                    env,
                    calls,
                    child_expected_width,
                    expected_signedness,
                    Some(obs),
                    context_span,
                )?
            } else {
                eval_ast_with_calls_inner(
                    expr,
                    env,
                    calls,
                    child_expected_width,
                    expected_signedness,
                    None,
                    context_span,
                )?
            };
            Ok(eval_unary_op(*op, v, expected_width, expected_signedness))
        }
        Expr::Binary { op, lhs, rhs } => {
            let (lhs_expected_width, rhs_expected_width) =
                binary_operand_expected_widths(*op, expected_width);
            let (lhs_expected_signedness, rhs_expected_signedness) =
                binary_operand_expected_signednesses(*op, expected_signedness);
            let a0 = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(
                    lhs,
                    env,
                    calls,
                    lhs_expected_width,
                    lhs_expected_signedness,
                    Some(obs),
                    context_span,
                )?
            } else {
                eval_ast_with_calls_inner(
                    lhs,
                    env,
                    calls,
                    lhs_expected_width,
                    lhs_expected_signedness,
                    None,
                    context_span,
                )?
            };
            let b0 = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(
                    rhs,
                    env,
                    calls,
                    rhs_expected_width,
                    rhs_expected_signedness,
                    Some(obs),
                    context_span,
                )?
            } else {
                eval_ast_with_calls_inner(
                    rhs,
                    env,
                    calls,
                    rhs_expected_width,
                    rhs_expected_signedness,
                    None,
                    context_span,
                )?
            };
            let op_expected_width = match op {
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor => Some(expected_width.unwrap_or(0).max(a0.width.max(b0.width))),
                BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::CaseEq
                | BinaryOp::CaseNeq => Some(a0.width.max(b0.width)),
                BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sshr => expected_width,
                BinaryOp::LogicalAnd | BinaryOp::LogicalOr => None,
            };
            let op_lhs_expected_width_rhs_expected_width = match op {
                BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::CaseEq
                | BinaryOp::CaseNeq => (op_expected_width, op_expected_width),
                _ => binary_operand_expected_widths(*op, op_expected_width),
            };
            let op_expected_signedness = match op {
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor => {
                    Some(expected_signedness.unwrap_or_else(|| merged_signedness(&a0, &b0)))
                }
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    Some(merged_signedness(&a0, &b0))
                }
                BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sshr => expected_signedness,
                BinaryOp::LogicalAnd
                | BinaryOp::LogicalOr
                | BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::CaseEq
                | BinaryOp::CaseNeq => None,
            };
            let op_lhs_expected_signedness_rhs_expected_signedness = match op {
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    (op_expected_signedness, op_expected_signedness)
                }
                _ => binary_operand_expected_signednesses(*op, op_expected_signedness),
            };
            let needs_recontext = op_expected_signedness != expected_signedness
                || op_expected_width != expected_width
                || op_lhs_expected_width_rhs_expected_width
                    != (lhs_expected_width, rhs_expected_width)
                || op_lhs_expected_signedness_rhs_expected_signedness
                    != (lhs_expected_signedness, rhs_expected_signedness);
            let (a, b) = if needs_recontext {
                let a = if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        lhs,
                        env,
                        calls,
                        op_lhs_expected_width_rhs_expected_width.0,
                        op_lhs_expected_signedness_rhs_expected_signedness.0,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(
                        lhs,
                        env,
                        calls,
                        op_lhs_expected_width_rhs_expected_width.0,
                        op_lhs_expected_signedness_rhs_expected_signedness.0,
                        None,
                        context_span,
                    )?
                };
                let b = if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        rhs,
                        env,
                        calls,
                        op_lhs_expected_width_rhs_expected_width.1,
                        op_lhs_expected_signedness_rhs_expected_signedness.1,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(
                        rhs,
                        env,
                        calls,
                        op_lhs_expected_width_rhs_expected_width.1,
                        op_lhs_expected_signedness_rhs_expected_signedness.1,
                        None,
                        context_span,
                    )?
                };
                (a, b)
            } else {
                (a0, b0)
            };
            Ok(eval_binary_op(
                *op,
                a,
                b,
                op_expected_width,
                op_expected_signedness,
            ))
        }
        Expr::Ternary { cond, t, f } => {
            let c = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(cond, env, calls, None, None, Some(obs), context_span)?
            } else {
                eval_ast_with_calls_inner(cond, env, calls, None, None, None, context_span)?
            };
            if let Some(obs) = observer.as_deref_mut() {
                obs.on_ternary(context_span, expr, &c);
            }
            let tv0 = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(
                    t,
                    env,
                    calls,
                    expected_width,
                    expected_signedness,
                    Some(obs),
                    context_span,
                )?
            } else {
                eval_ast_with_calls_inner(
                    t,
                    env,
                    calls,
                    expected_width,
                    expected_signedness,
                    None,
                    context_span,
                )?
            };
            let fv0 = if let Some(obs) = observer.as_deref_mut() {
                eval_ast_with_calls_inner(
                    f,
                    env,
                    calls,
                    expected_width,
                    expected_signedness,
                    Some(obs),
                    context_span,
                )?
            } else {
                eval_ast_with_calls_inner(
                    f,
                    env,
                    calls,
                    expected_width,
                    expected_signedness,
                    None,
                    context_span,
                )?
            };
            let branch_expected_width =
                Some(expected_width.unwrap_or(0).max(tv0.width.max(fv0.width)));
            let recontext_t =
                branch_expected_width != expected_width && ternary_branch_needs_width_recontext(t);
            let recontext_f =
                branch_expected_width != expected_width && ternary_branch_needs_width_recontext(f);
            let tv = if recontext_t {
                if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        t,
                        env,
                        calls,
                        branch_expected_width,
                        expected_signedness,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(
                        t,
                        env,
                        calls,
                        branch_expected_width,
                        expected_signedness,
                        None,
                        context_span,
                    )?
                }
            } else {
                tv0
            };
            let fv = if recontext_f {
                if let Some(obs) = observer.as_deref_mut() {
                    eval_ast_with_calls_inner(
                        f,
                        env,
                        calls,
                        branch_expected_width,
                        expected_signedness,
                        Some(obs),
                        context_span,
                    )?
                } else {
                    eval_ast_with_calls_inner(
                        f,
                        env,
                        calls,
                        branch_expected_width,
                        expected_signedness,
                        None,
                        context_span,
                    )?
                }
            } else {
                fv0
            };
            let tv = operand_with_own_sign_ctx(tv, expected_width, expected_signedness);
            let fv = operand_with_own_sign_ctx(fv, expected_width, expected_signedness);
            Ok(operand_with_own_sign_ctx(
                Value4::ternary(&c, &tv, &fv),
                expected_width,
                expected_signedness,
            ))
        }
    }
}
