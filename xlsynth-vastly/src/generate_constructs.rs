// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::ast::Expr;
use crate::eval::eval_ast_with_calls;
use crate::sv_ast::AlwaysFf;
use crate::sv_ast::GenerateBranch;
use crate::sv_ast::Lhs;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::Stmt;
use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;

pub fn elaborate_combo_items(
    src: &str,
    params: &BTreeMap<String, Value4>,
    items: &[ModuleItem],
) -> Result<Vec<ModuleItem>> {
    let mut env = crate::Env::new();
    for (name, value) in params {
        env.insert(name.clone(), value.clone());
    }
    let mut substs = BTreeMap::new();
    let mut out = Vec::new();
    elaborate_items_impl(src, items, &mut env, &mut substs, false, &mut out)?;
    Ok(out)
}

pub fn elaborate_pipeline_items(
    src: &str,
    params: &BTreeMap<String, Value4>,
    items: &[ModuleItem],
) -> Result<Vec<ModuleItem>> {
    let mut env = crate::Env::new();
    for (name, value) in params {
        env.insert(name.clone(), value.clone());
    }
    let mut substs = BTreeMap::new();
    let mut out = Vec::new();
    elaborate_items_impl(src, items, &mut env, &mut substs, false, &mut out)?;
    Ok(out)
}

fn elaborate_items_impl(
    src: &str,
    items: &[ModuleItem],
    env: &mut crate::Env,
    substs: &mut BTreeMap<String, Value4>,
    in_generate: bool,
    out: &mut Vec<ModuleItem>,
) -> Result<()> {
    for item in items {
        elaborate_item(src, item, env, substs, in_generate, out)?;
    }
    Ok(())
}

fn elaborate_item(
    src: &str,
    item: &ModuleItem,
    env: &mut crate::Env,
    substs: &mut BTreeMap<String, Value4>,
    in_generate: bool,
    out: &mut Vec<ModuleItem>,
) -> Result<()> {
    match item {
        ModuleItem::Decl { .. } => {
            if in_generate {
                return Err(Error::Parse(
                    "declarations inside generate blocks are not supported".to_string(),
                ));
            }
            out.push(item.clone());
        }
        ModuleItem::Assign {
            lhs,
            rhs,
            rhs_span,
            span,
        } => {
            out.push(ModuleItem::Assign {
                lhs: substitute_lhs(lhs, substs),
                rhs: substitute_expr(rhs, substs),
                rhs_span: *rhs_span,
                span: *span,
            });
        }
        ModuleItem::Function { .. } => {
            if substs.is_empty() {
                out.push(item.clone());
            } else {
                return Err(Error::Parse(
                    "functions inside generate blocks are not supported".to_string(),
                ));
            }
        }
        ModuleItem::AlwaysFf { always_ff, span } => {
            out.push(ModuleItem::AlwaysFf {
                always_ff: AlwaysFf {
                    clk_name: always_ff.clk_name.clone(),
                    body: substitute_stmt(&always_ff.body, substs),
                },
                span: *span,
            });
        }
        ModuleItem::GenerateFor {
            genvar,
            start,
            limit,
            body,
            span: _,
        } => {
            let start_u = eval_const_u32(start, env)?;
            let limit_u = eval_const_u32(limit, env)?;
            for idx in start_u..limit_u {
                let genvar_value = u32_value(idx);
                let prev_env = env.insert(genvar.clone(), genvar_value.clone());
                let prev_subst = substs.insert(genvar.clone(), genvar_value);
                let elaborate_result = elaborate_items_impl(src, body, env, substs, true, out);
                restore_env_binding(env, genvar, prev_env);
                restore_subst_binding(substs, genvar, prev_subst);
                elaborate_result?;
            }
        }
        ModuleItem::GenerateIf { branches, span: _ } => {
            if let Some(selected) = select_branch(branches, env)? {
                elaborate_items_impl(src, &selected.body, env, substs, true, out)?;
            }
        }
    }
    Ok(())
}

fn restore_env_binding(env: &mut crate::Env, key: &str, previous: Option<Value4>) {
    match previous {
        Some(value) => {
            env.insert(key.to_string(), value);
        }
        None => {
            env.remove(key);
        }
    }
}

fn restore_subst_binding(map: &mut BTreeMap<String, Value4>, key: &str, previous: Option<Value4>) {
    match previous {
        Some(value) => {
            map.insert(key.to_string(), value);
        }
        None => {
            map.remove(key);
        }
    }
}

fn select_branch<'a>(
    branches: &'a [GenerateBranch],
    env: &crate::Env,
) -> Result<Option<&'a GenerateBranch>> {
    for branch in branches {
        match &branch.cond {
            Some(cond) => {
                if eval_const_bool(cond, env)? {
                    return Ok(Some(branch));
                }
            }
            None => return Ok(Some(branch)),
        }
    }
    Ok(None)
}

fn eval_const_u32(expr: &Expr, env: &crate::Env) -> Result<u32> {
    let value = eval_ast_with_calls(expr, env, None, None)?;
    value
        .to_u32_if_known()
        .ok_or_else(|| Error::Parse("generate expression must be a known u32".to_string()))
}

fn eval_const_bool(expr: &Expr, env: &crate::Env) -> Result<bool> {
    let value = eval_ast_with_calls(expr, env, None, None)?;
    match value.to_bool4() {
        LogicBit::One => Ok(true),
        LogicBit::Zero => Ok(false),
        LogicBit::X | LogicBit::Z => Err(Error::Parse(
            "generate condition must evaluate to a known boolean".to_string(),
        )),
    }
}

fn u32_value(value: u32) -> Value4 {
    Value4::parse_numeric_token(32, Signedness::Unsigned, &value.to_string()).unwrap()
}

fn substitute_expr(expr: &Expr, substs: &BTreeMap<String, Value4>) -> Expr {
    match expr {
        Expr::Ident(name) => substs
            .get(name)
            .cloned()
            .map(Expr::Literal)
            .unwrap_or_else(|| Expr::Ident(name.clone())),
        Expr::Literal(v) => Expr::Literal(v.clone()),
        Expr::UnsizedNumber(v) => Expr::UnsizedNumber(v.clone()),
        Expr::UnbasedUnsized(bit) => Expr::UnbasedUnsized(*bit),
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| substitute_expr(arg, substs))
                .collect(),
        },
        Expr::Concat(parts) => Expr::Concat(
            parts
                .iter()
                .map(|part| substitute_expr(part, substs))
                .collect(),
        ),
        Expr::Replicate { count, expr } => Expr::Replicate {
            count: Box::new(substitute_expr(count, substs)),
            expr: Box::new(substitute_expr(expr, substs)),
        },
        Expr::Cast { width, expr } => Expr::Cast {
            width: Box::new(substitute_expr(width, substs)),
            expr: Box::new(substitute_expr(expr, substs)),
        },
        Expr::Index { expr, index } => Expr::Index {
            expr: Box::new(substitute_expr(expr, substs)),
            index: Box::new(substitute_expr(index, substs)),
        },
        Expr::Slice { expr, msb, lsb } => Expr::Slice {
            expr: Box::new(substitute_expr(expr, substs)),
            msb: Box::new(substitute_expr(msb, substs)),
            lsb: Box::new(substitute_expr(lsb, substs)),
        },
        Expr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => Expr::IndexedSlice {
            expr: Box::new(substitute_expr(expr, substs)),
            base: Box::new(substitute_expr(base, substs)),
            width: Box::new(substitute_expr(width, substs)),
            upward: *upward,
        },
        Expr::Unary { op, expr } => Expr::Unary {
            op: *op,
            expr: Box::new(substitute_expr(expr, substs)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: *op,
            lhs: Box::new(substitute_expr(lhs, substs)),
            rhs: Box::new(substitute_expr(rhs, substs)),
        },
        Expr::Ternary { cond, t, f } => Expr::Ternary {
            cond: Box::new(substitute_expr(cond, substs)),
            t: Box::new(substitute_expr(t, substs)),
            f: Box::new(substitute_expr(f, substs)),
        },
    }
}

fn substitute_lhs(lhs: &Lhs, substs: &BTreeMap<String, Value4>) -> Lhs {
    match lhs {
        Lhs::Ident(name) => Lhs::Ident(name.clone()),
        Lhs::Index { base, index } => Lhs::Index {
            base: base.clone(),
            index: substitute_expr(index, substs),
        },
        Lhs::PackedIndex { base, indices } => Lhs::PackedIndex {
            base: base.clone(),
            indices: indices
                .iter()
                .map(|index| substitute_expr(index, substs))
                .collect(),
        },
        Lhs::Slice { base, msb, lsb } => Lhs::Slice {
            base: base.clone(),
            msb: substitute_expr(msb, substs),
            lsb: substitute_expr(lsb, substs),
        },
    }
}

fn substitute_stmt(stmt: &Stmt, substs: &BTreeMap<String, Value4>) -> Stmt {
    match stmt {
        Stmt::Begin(stmts) => Stmt::Begin(
            stmts
                .iter()
                .map(|stmt| substitute_stmt(stmt, substs))
                .collect(),
        ),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => Stmt::If {
            cond: substitute_expr(cond, substs),
            then_branch: Box::new(substitute_stmt(then_branch, substs)),
            else_branch: else_branch
                .as_ref()
                .map(|stmt| Box::new(substitute_stmt(stmt, substs))),
        },
        Stmt::NbaAssign { lhs, rhs } => Stmt::NbaAssign {
            lhs: substitute_lhs(lhs, substs),
            rhs: substitute_expr(rhs, substs),
        },
        Stmt::Display { fmt, args } => Stmt::Display {
            fmt: fmt.clone(),
            args: args
                .iter()
                .map(|arg| substitute_expr(arg, substs))
                .collect(),
        },
        Stmt::Empty => Stmt::Empty,
    }
}
