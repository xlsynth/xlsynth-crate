// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::Expr;
use crate::eval::eval_ast_with_calls;
use crate::sv_ast::AlwaysFf;
use crate::sv_ast::ComboGenerateBranch;
use crate::sv_ast::ComboItem;
use crate::sv_ast::GenerateBranch;
use crate::sv_ast::Lhs;
use crate::sv_ast::PipelineItem;
use crate::sv_ast::Stmt;

pub fn elaborate_combo_items(
    src: &str,
    params: &BTreeMap<String, Value4>,
    items: &[ComboItem],
) -> Result<Vec<ComboItem>> {
    let mut env = crate::Env::new();
    for (name, value) in params {
        env.insert(name.clone(), value.clone());
    }
    elaborate_combo_items_impl(src, items, &env, &BTreeMap::new())
}

pub fn elaborate_pipeline_items(
    src: &str,
    params: &BTreeMap<String, Value4>,
    items: &[PipelineItem],
) -> Result<Vec<PipelineItem>> {
    let mut env = crate::Env::new();
    for (name, value) in params {
        env.insert(name.clone(), value.clone());
    }
    elaborate_pipeline_items_impl(src, items, &env, &BTreeMap::new())
}

fn elaborate_combo_items_impl(
    src: &str,
    items: &[ComboItem],
    env: &crate::Env,
    substs: &BTreeMap<String, Value4>,
) -> Result<Vec<ComboItem>> {
    let mut out = Vec::new();
    for item in items {
        elaborate_combo_item(src, item, env, substs, &mut out)?;
    }
    Ok(out)
}

fn elaborate_combo_item(
    src: &str,
    item: &ComboItem,
    env: &crate::Env,
    substs: &BTreeMap<String, Value4>,
    out: &mut Vec<ComboItem>,
) -> Result<()> {
    match item {
        ComboItem::WireDecl(_) => out.push(item.clone()),
        ComboItem::Assign { lhs, rhs, rhs_text } => {
            let base_text = rhs_text
                .as_deref()
                .unwrap_or_else(|| src[rhs.start..rhs.end].trim());
            let substituted_text = if substs.is_empty() {
                None
            } else {
                Some(substitute_text(base_text, substs))
            };
            out.push(ComboItem::Assign {
                lhs: substitute_lhs(lhs, substs),
                rhs: *rhs,
                rhs_text: substituted_text,
            });
        }
        ComboItem::Function(_) => {
            if substs.is_empty() {
                out.push(item.clone());
            } else {
                return Err(Error::Parse(
                    "functions inside generate blocks are not supported".to_string(),
                ));
            }
        }
        ComboItem::GenerateFor {
            genvar,
            start,
            limit,
            body,
        } => {
            let start_u = eval_const_u32(start, env)?;
            let limit_u = eval_const_u32(limit, env)?;
            for idx in start_u..limit_u {
                let genvar_value = u32_value(idx);
                let mut next_env = env.clone();
                next_env.insert(genvar.clone(), genvar_value.clone());
                let mut next_substs = substs.clone();
                next_substs.insert(genvar.clone(), genvar_value);
                out.extend(elaborate_combo_items_impl(
                    src,
                    body,
                    &next_env,
                    &next_substs,
                )?);
            }
        }
        ComboItem::GenerateIf { branches } => {
            if let Some(selected) = select_combo_branch(branches, env)? {
                out.extend(elaborate_combo_items_impl(
                    src,
                    &selected.body,
                    env,
                    substs,
                )?);
            }
        }
    }
    Ok(())
}

fn elaborate_pipeline_items_impl(
    src: &str,
    items: &[PipelineItem],
    env: &crate::Env,
    substs: &BTreeMap<String, Value4>,
) -> Result<Vec<PipelineItem>> {
    let mut out = Vec::new();
    for item in items {
        elaborate_pipeline_item(src, item, env, substs, &mut out)?;
    }
    Ok(out)
}

fn elaborate_pipeline_item(
    src: &str,
    item: &PipelineItem,
    env: &crate::Env,
    substs: &BTreeMap<String, Value4>,
    out: &mut Vec<PipelineItem>,
) -> Result<()> {
    match item {
        PipelineItem::Decl { .. } => out.push(item.clone()),
        PipelineItem::Assign {
            lhs,
            rhs,
            rhs_text,
            span,
        } => {
            let base_text = rhs_text
                .as_deref()
                .unwrap_or_else(|| src[rhs.start..rhs.end].trim());
            let substituted_text = if substs.is_empty() {
                None
            } else {
                Some(substitute_text(base_text, substs))
            };
            out.push(PipelineItem::Assign {
                lhs: substitute_lhs(lhs, substs),
                rhs: *rhs,
                rhs_text: substituted_text,
                span: *span,
            });
        }
        PipelineItem::Function { .. } => {
            if substs.is_empty() {
                out.push(item.clone());
            } else {
                return Err(Error::Parse(
                    "functions inside generate blocks are not supported".to_string(),
                ));
            }
        }
        PipelineItem::AlwaysFf { always_ff, span } => {
            out.push(PipelineItem::AlwaysFf {
                always_ff: AlwaysFf {
                    clk_name: always_ff.clk_name.clone(),
                    body: substitute_stmt(&always_ff.body, substs),
                },
                span: *span,
            });
        }
        PipelineItem::GenerateFor {
            genvar,
            start,
            limit,
            body,
            ..
        } => {
            let start_u = eval_const_u32(start, env)?;
            let limit_u = eval_const_u32(limit, env)?;
            for idx in start_u..limit_u {
                let genvar_value = u32_value(idx);
                let mut next_env = env.clone();
                next_env.insert(genvar.clone(), genvar_value.clone());
                let mut next_substs = substs.clone();
                next_substs.insert(genvar.clone(), genvar_value);
                out.extend(elaborate_pipeline_items_impl(
                    src,
                    body,
                    &next_env,
                    &next_substs,
                )?);
            }
        }
        PipelineItem::GenerateIf { branches, .. } => {
            if let Some(selected) = select_pipeline_branch(branches, env)? {
                out.extend(elaborate_pipeline_items_impl(
                    src,
                    &selected.body,
                    env,
                    substs,
                )?);
            }
        }
    }
    Ok(())
}

fn select_combo_branch<'a>(
    branches: &'a [ComboGenerateBranch],
    env: &crate::Env,
) -> Result<Option<&'a ComboGenerateBranch>> {
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

fn select_pipeline_branch<'a>(
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

fn substitute_text(text: &str, substs: &BTreeMap<String, Value4>) -> String {
    let mut out = String::from(text);
    for (name, value) in substs {
        let replacement = value
            .to_u32_if_known()
            .map(|v| v.to_string())
            .unwrap_or_else(|| value.to_bit_string_msb_first());
        out = replace_ident_token(&out, name, &replacement);
    }
    out
}

fn replace_ident_token(s: &str, target: &str, replacement: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = &s[start..i];
            if ident == target {
                out.push_str(replacement);
            } else {
                out.push_str(ident);
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}
