// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::Error;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::BinaryOp;
use crate::ast::Expr;
use crate::ast_spanned::SpannedExpr;
use crate::ast_spanned::SpannedExprKind;
use crate::module_compile::DeclInfo;
use crate::sv_ast::Lhs;
use crate::sv_ast::Span;
use crate::sv_ast::Stmt;

fn dims_width(dims: &[u32]) -> Result<u32> {
    dims.iter().try_fold(1u32, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::Parse("packed decl width overflow".to_string()))
    })
}

fn stride_after(dims: &[u32], dim_index: usize) -> Result<u32> {
    if dim_index + 1 >= dims.len() {
        Ok(1)
    } else {
        dims_width(&dims[dim_index + 1..])
    }
}

fn remaining_width_after(dims: &[u32], consumed_indices: usize) -> Result<u32> {
    if consumed_indices > dims.len() {
        return Err(Error::Parse(
            "too many packed indices for declaration".to_string(),
        ));
    }
    if consumed_indices == dims.len() {
        Ok(1)
    } else {
        dims_width(&dims[consumed_indices..])
    }
}

fn build_linear_index<T: Clone, FMul, FAdd>(
    indices: &[T],
    packed_dims: &[u32],
    mut mul: FMul,
    mut add: FAdd,
) -> Result<T>
where
    FMul: FnMut(T, u32) -> T,
    FAdd: FnMut(T, T) -> T,
{
    let mut acc: Option<T> = None;
    for (dim_index, index) in indices.iter().cloned().enumerate() {
        let stride = stride_after(packed_dims, dim_index)?;
        let term = mul(index, stride);
        acc = Some(match acc {
            Some(prev) => add(prev, term),
            None => term,
        });
    }
    acc.ok_or_else(|| Error::Parse("internal: empty packed index chain".to_string()))
}

fn checked_packed_offset(dims: &[u32], indices: &[u32]) -> Result<u32> {
    let mut offset = 0u32;
    for (dim_index, index) in indices.iter().copied().enumerate() {
        let dim_width = dims
            .get(dim_index)
            .copied()
            .ok_or_else(|| Error::Parse("too many packed indices for declaration".to_string()))?;
        if index >= dim_width {
            return Err(Error::Parse(format!(
                "packed index {index} out of bounds for dimension {dim_index} (size {dim_width})"
            )));
        }
        let stride = stride_after(dims, dim_index)?;
        let term = index
            .checked_mul(stride)
            .ok_or_else(|| Error::Parse("packed index offset overflow".to_string()))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| Error::Parse("packed index offset overflow".to_string()))?;
    }
    Ok(offset)
}

fn checked_packed_offset_if_in_bounds(dims: &[u32], indices: &[u32]) -> Result<Option<u32>> {
    let mut offset = 0u32;
    for (dim_index, index) in indices.iter().copied().enumerate() {
        let dim_width = dims
            .get(dim_index)
            .copied()
            .ok_or_else(|| Error::Parse("too many packed indices for declaration".to_string()))?;
        if index >= dim_width {
            return Ok(None);
        }
        let stride = stride_after(dims, dim_index)?;
        let term = index
            .checked_mul(stride)
            .ok_or_else(|| Error::Parse("packed index offset overflow".to_string()))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| Error::Parse("packed index offset overflow".to_string()))?;
    }
    Ok(Some(offset))
}

fn literal_u32(value: u32) -> Expr {
    Expr::Literal(
        Value4::parse_numeric_token(32, Signedness::Unsigned, &value.to_string()).unwrap(),
    )
}

fn literal_u32_spanned(value: u32, span: Span) -> SpannedExpr {
    SpannedExpr {
        span,
        kind: SpannedExprKind::Literal(
            Value4::parse_numeric_token(32, Signedness::Unsigned, &value.to_string()).unwrap(),
        ),
    }
}

fn mul_expr(lhs: Expr, factor: u32) -> Expr {
    if factor == 1 {
        lhs
    } else {
        Expr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(literal_u32(factor)),
        }
    }
}

fn mul_expr_spanned(lhs: SpannedExpr, factor: u32, span: Span) -> SpannedExpr {
    if factor == 1 {
        lhs
    } else {
        SpannedExpr {
            span,
            kind: SpannedExprKind::Binary {
                op: BinaryOp::Mul,
                lhs: Box::new(lhs),
                rhs: Box::new(literal_u32_spanned(factor, span)),
            },
        }
    }
}

fn add_expr(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op: BinaryOp::Add,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn add_expr_spanned(lhs: SpannedExpr, rhs: SpannedExpr, span: Span) -> SpannedExpr {
    SpannedExpr {
        span,
        kind: SpannedExprKind::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    }
}

fn collect_index_chain(expr: &Expr) -> Option<(String, Vec<Expr>)> {
    match expr {
        Expr::Ident(name) => Some((name.clone(), Vec::new())),
        Expr::Index { expr, index } => {
            let (name, mut indices) = collect_index_chain(expr)?;
            indices.push((**index).clone());
            Some((name, indices))
        }
        _ => None,
    }
}

fn collect_spanned_index_chain(expr: &SpannedExpr) -> Option<(SpannedExpr, Vec<SpannedExpr>)> {
    match &expr.kind {
        SpannedExprKind::Ident(_) => Some((expr.clone(), Vec::new())),
        SpannedExprKind::Index { expr: inner, index } => {
            let (base, mut indices) = collect_spanned_index_chain(inner)?;
            indices.push((**index).clone());
            Some((base, indices))
        }
        _ => None,
    }
}

fn lower_expr_index_chain(
    base_name: &str,
    indices: Vec<Expr>,
    decls: &BTreeMap<String, DeclInfo>,
) -> Result<Option<Expr>> {
    let Some(info) = decls.get(base_name) else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Ok(None);
    }
    if indices.len() > info.packed_dims.len() {
        return Err(Error::Parse(format!(
            "too many packed indices for `{base_name}`"
        )));
    }
    let linear_index = build_linear_index(&indices, &info.packed_dims, mul_expr, add_expr)?;
    let remaining_width = remaining_width_after(&info.packed_dims, indices.len())?;
    let base_expr = Expr::Ident(base_name.to_string());
    if remaining_width == 1 {
        Ok(Some(Expr::Index {
            expr: Box::new(base_expr),
            index: Box::new(linear_index),
        }))
    } else {
        Ok(Some(Expr::IndexedSlice {
            expr: Box::new(base_expr),
            base: Box::new(linear_index),
            width: Box::new(literal_u32(remaining_width)),
            upward: true,
        }))
    }
}

fn lower_spanned_index_chain(
    expr_span: Span,
    base: SpannedExpr,
    indices: Vec<SpannedExpr>,
    decls: &BTreeMap<String, DeclInfo>,
) -> Result<Option<SpannedExpr>> {
    let SpannedExprKind::Ident(base_name) = &base.kind else {
        return Ok(None);
    };
    let Some(info) = decls.get(base_name) else {
        return Ok(None);
    };
    if indices.is_empty() {
        return Ok(None);
    }
    if indices.len() > info.packed_dims.len() {
        return Err(Error::Parse(format!(
            "too many packed indices for `{base_name}`"
        )));
    }
    let linear_index = build_linear_index(
        &indices,
        &info.packed_dims,
        |index, stride| mul_expr_spanned(index, stride, expr_span),
        |lhs, rhs| add_expr_spanned(lhs, rhs, expr_span),
    )?;
    let remaining_width = remaining_width_after(&info.packed_dims, indices.len())?;
    if remaining_width == 1 {
        Ok(Some(SpannedExpr {
            span: expr_span,
            kind: SpannedExprKind::Index {
                expr: Box::new(base),
                index: Box::new(linear_index),
            },
        }))
    } else {
        Ok(Some(SpannedExpr {
            span: expr_span,
            kind: SpannedExprKind::IndexedSlice {
                expr: Box::new(base),
                base: Box::new(linear_index),
                width: Box::new(literal_u32_spanned(remaining_width, expr_span)),
                upward: true,
            },
        }))
    }
}

pub fn rewrite_packed_expr(expr: Expr, decls: &BTreeMap<String, DeclInfo>) -> Result<Expr> {
    if let Some((base_name, indices)) = collect_index_chain(&expr) {
        if !indices.is_empty() {
            let mut rewritten_indices = Vec::with_capacity(indices.len());
            for index in indices {
                rewritten_indices.push(rewrite_packed_expr(index, decls)?);
            }
            if let Some(lowered) = lower_expr_index_chain(&base_name, rewritten_indices, decls)? {
                return Ok(lowered);
            }
        }
    }
    match expr {
        Expr::Ident(_) | Expr::Literal(_) | Expr::UnsizedNumber(_) | Expr::UnbasedUnsized(_) => {
            Ok(expr)
        }
        Expr::Call { name, args } => {
            let mut out = Vec::with_capacity(args.len());
            for arg in args {
                out.push(rewrite_packed_expr(arg, decls)?);
            }
            Ok(Expr::Call { name, args: out })
        }
        Expr::Concat(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                out.push(rewrite_packed_expr(part, decls)?);
            }
            Ok(Expr::Concat(out))
        }
        Expr::Replicate { count, expr } => Ok(Expr::Replicate {
            count: Box::new(rewrite_packed_expr(*count, decls)?),
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
        }),
        Expr::Cast { width, expr } => Ok(Expr::Cast {
            width: Box::new(rewrite_packed_expr(*width, decls)?),
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
        }),
        Expr::Index { expr, index } => Ok(Expr::Index {
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
            index: Box::new(rewrite_packed_expr(*index, decls)?),
        }),
        Expr::Slice { expr, msb, lsb } => Ok(Expr::Slice {
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
            msb: Box::new(rewrite_packed_expr(*msb, decls)?),
            lsb: Box::new(rewrite_packed_expr(*lsb, decls)?),
        }),
        Expr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => Ok(Expr::IndexedSlice {
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
            base: Box::new(rewrite_packed_expr(*base, decls)?),
            width: Box::new(rewrite_packed_expr(*width, decls)?),
            upward,
        }),
        Expr::Unary { op, expr } => Ok(Expr::Unary {
            op,
            expr: Box::new(rewrite_packed_expr(*expr, decls)?),
        }),
        Expr::Binary { op, lhs, rhs } => Ok(Expr::Binary {
            op,
            lhs: Box::new(rewrite_packed_expr(*lhs, decls)?),
            rhs: Box::new(rewrite_packed_expr(*rhs, decls)?),
        }),
        Expr::Ternary { cond, t, f } => Ok(Expr::Ternary {
            cond: Box::new(rewrite_packed_expr(*cond, decls)?),
            t: Box::new(rewrite_packed_expr(*t, decls)?),
            f: Box::new(rewrite_packed_expr(*f, decls)?),
        }),
    }
}

pub fn rewrite_packed_spanned_expr(
    expr: SpannedExpr,
    decls: &BTreeMap<String, DeclInfo>,
) -> Result<SpannedExpr> {
    if let Some((base, indices)) = collect_spanned_index_chain(&expr) {
        if !indices.is_empty() {
            let mut rewritten_indices = Vec::with_capacity(indices.len());
            for index in indices {
                rewritten_indices.push(rewrite_packed_spanned_expr(index, decls)?);
            }
            if let Some(lowered) =
                lower_spanned_index_chain(expr.span, base, rewritten_indices, decls)?
            {
                return Ok(lowered);
            }
        }
    }
    match expr.kind {
        SpannedExprKind::Ident(_)
        | SpannedExprKind::Literal(_)
        | SpannedExprKind::UnsizedNumber(_)
        | SpannedExprKind::UnbasedUnsized(_) => Ok(expr),
        SpannedExprKind::Call { name, args } => {
            let mut out = Vec::with_capacity(args.len());
            for arg in args {
                out.push(rewrite_packed_spanned_expr(arg, decls)?);
            }
            Ok(SpannedExpr {
                span: expr.span,
                kind: SpannedExprKind::Call { name, args: out },
            })
        }
        SpannedExprKind::Concat(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                out.push(rewrite_packed_spanned_expr(part, decls)?);
            }
            Ok(SpannedExpr {
                span: expr.span,
                kind: SpannedExprKind::Concat(out),
            })
        }
        SpannedExprKind::Replicate { count, expr: inner } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Replicate {
                count: Box::new(rewrite_packed_spanned_expr(*count, decls)?),
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
            },
        }),
        SpannedExprKind::Cast { width, expr: inner } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Cast {
                width: Box::new(rewrite_packed_spanned_expr(*width, decls)?),
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
            },
        }),
        SpannedExprKind::Index { expr: inner, index } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Index {
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
                index: Box::new(rewrite_packed_spanned_expr(*index, decls)?),
            },
        }),
        SpannedExprKind::Slice {
            expr: inner,
            msb,
            lsb,
        } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Slice {
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
                msb: Box::new(rewrite_packed_spanned_expr(*msb, decls)?),
                lsb: Box::new(rewrite_packed_spanned_expr(*lsb, decls)?),
            },
        }),
        SpannedExprKind::IndexedSlice {
            expr: inner,
            base,
            width,
            upward,
        } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::IndexedSlice {
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
                base: Box::new(rewrite_packed_spanned_expr(*base, decls)?),
                width: Box::new(rewrite_packed_spanned_expr(*width, decls)?),
                upward,
            },
        }),
        SpannedExprKind::Unary { op, expr: inner } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Unary {
                op,
                expr: Box::new(rewrite_packed_spanned_expr(*inner, decls)?),
            },
        }),
        SpannedExprKind::Binary { op, lhs, rhs } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Binary {
                op,
                lhs: Box::new(rewrite_packed_spanned_expr(*lhs, decls)?),
                rhs: Box::new(rewrite_packed_spanned_expr(*rhs, decls)?),
            },
        }),
        SpannedExprKind::Ternary { cond, t, f } => Ok(SpannedExpr {
            span: expr.span,
            kind: SpannedExprKind::Ternary {
                cond: Box::new(rewrite_packed_spanned_expr(*cond, decls)?),
                t: Box::new(rewrite_packed_spanned_expr(*t, decls)?),
                f: Box::new(rewrite_packed_spanned_expr(*f, decls)?),
            },
        }),
    }
}

pub fn rewrite_packed_lhs(lhs: Lhs, decls: &BTreeMap<String, DeclInfo>) -> Result<Lhs> {
    match lhs {
        Lhs::Ident(_) => Ok(lhs),
        Lhs::Index { base, index } => Ok(Lhs::Index {
            base,
            index: rewrite_packed_expr(index, decls)?,
        }),
        Lhs::PackedIndex { base, indices } => {
            let mut out = Vec::with_capacity(indices.len());
            for index in indices {
                out.push(rewrite_packed_expr(index, decls)?);
            }
            Ok(Lhs::PackedIndex { base, indices: out })
        }
        Lhs::Slice { base, msb, lsb } => Ok(Lhs::Slice {
            base,
            msb: rewrite_packed_expr(msb, decls)?,
            lsb: rewrite_packed_expr(lsb, decls)?,
        }),
    }
}

pub fn rewrite_packed_stmt(stmt: Stmt, decls: &BTreeMap<String, DeclInfo>) -> Result<Stmt> {
    match stmt {
        Stmt::Begin(stmts) => {
            let mut out = Vec::with_capacity(stmts.len());
            for stmt in stmts {
                out.push(rewrite_packed_stmt(stmt, decls)?);
            }
            Ok(Stmt::Begin(out))
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => Ok(Stmt::If {
            cond: rewrite_packed_expr(cond, decls)?,
            then_branch: Box::new(rewrite_packed_stmt(*then_branch, decls)?),
            else_branch: match else_branch {
                Some(stmt) => Some(Box::new(rewrite_packed_stmt(*stmt, decls)?)),
                None => None,
            },
        }),
        Stmt::NbaAssign { lhs, rhs } => Ok(Stmt::NbaAssign {
            lhs: rewrite_packed_lhs(lhs, decls)?,
            rhs: rewrite_packed_expr(rhs, decls)?,
        }),
        Stmt::Display { fmt, args } => {
            let mut out = Vec::with_capacity(args.len());
            for arg in args {
                out.push(rewrite_packed_expr(arg, decls)?);
            }
            Ok(Stmt::Display { fmt, args: out })
        }
        Stmt::Empty => Ok(Stmt::Empty),
    }
}

pub fn packed_index_selection(info: &DeclInfo, indices: &[u32]) -> Result<(u32, u32)> {
    if indices.is_empty() {
        return Ok((0, info.width));
    }
    if indices.len() > info.packed_dims.len() {
        return Err(Error::Parse(
            "too many packed indices for declaration".to_string(),
        ));
    }
    let offset = checked_packed_offset(&info.packed_dims, indices)?;
    let width = remaining_width_after(&info.packed_dims, indices.len())?;
    Ok((offset, width))
}

pub fn packed_index_selection_if_in_bounds(
    info: &DeclInfo,
    indices: &[u32],
) -> Result<Option<(u32, u32)>> {
    if indices.is_empty() {
        return Ok(Some((0, info.width)));
    }
    if indices.len() > info.packed_dims.len() {
        return Err(Error::Parse(
            "too many packed indices for declaration".to_string(),
        ));
    }
    let Some(offset) = checked_packed_offset_if_in_bounds(&info.packed_dims, indices)? else {
        return Ok(None);
    };
    let width = remaining_width_after(&info.packed_dims, indices.len())?;
    Ok(Some((offset, width)))
}
