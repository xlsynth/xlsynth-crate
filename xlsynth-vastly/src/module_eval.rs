// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::Expr as VExpr;
use crate::compiled_module::CompiledSeqBlock;
use crate::compiled_module::State;
use crate::packed::packed_index_selection_if_in_bounds;
use crate::sv_ast::Lhs;
use crate::sv_ast::Stmt;

pub fn step_module(m: &CompiledSeqBlock, inputs: &crate::Env, state: &State) -> Result<State> {
    // Build evaluation environment: inputs shadow state (as per plan default).
    let mut env = crate::Env::new();
    for (k, v) in m.consts.iter() {
        env.insert(k.clone(), v.clone());
    }
    for (k, v) in state.iter() {
        env.insert(k.clone(), v.clone());
    }
    for (k, v) in inputs.iter() {
        env.insert(k.clone(), v.clone());
    }

    step_module_with_env(m, &env, state)
}

pub fn step_module_with_env(
    m: &CompiledSeqBlock,
    env: &crate::Env,
    state: &State,
) -> Result<State> {
    let mut exec_env = crate::Env::new();
    for (k, v) in m.consts.iter() {
        exec_env.insert(k.clone(), v.clone());
    }
    for (k, v) in env.iter() {
        exec_env.insert(k.clone(), v.clone());
    }
    let mut pending: BTreeMap<String, PendingBits> = BTreeMap::new();
    exec_stmt(&m.body, &exec_env, m, &mut pending)?;

    let mut next = state.clone();
    for (reg, pb) in pending {
        let info = m
            .decls
            .get(&reg)
            .ok_or_else(|| Error::Parse(format!("no decl info for reg {reg}")))?;
        let old = next.get(&reg).cloned().unwrap_or_else(|| {
            Value4::new(
                info.width,
                info.signedness,
                vec![LogicBit::X; info.width as usize],
            )
        });
        let mut bits = old.bits_lsb_first().to_vec();
        for (i, bit) in pb.bits.into_iter().enumerate() {
            if let Some(b) = bit {
                if i < bits.len() {
                    bits[i] = b;
                }
            }
        }
        next.insert(reg, Value4::new(info.width, info.signedness, bits));
    }
    Ok(next)
}

#[derive(Debug, Clone)]
struct PendingBits {
    bits: Vec<Option<LogicBit>>,
}

fn exec_stmt(
    stmt: &Stmt,
    env: &crate::Env,
    m: &CompiledSeqBlock,
    pending: &mut BTreeMap<String, PendingBits>,
) -> Result<()> {
    match stmt {
        Stmt::Begin(stmts) => {
            for s in stmts {
                exec_stmt(s, env, m, pending)?;
            }
            Ok(())
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = eval_expr(cond, env)?;
            match c.to_bool4() {
                LogicBit::One => exec_stmt(then_branch, env, m, pending),
                LogicBit::Zero => {
                    if let Some(e) = else_branch {
                        exec_stmt(e, env, m, pending)
                    } else {
                        Ok(())
                    }
                }
                // v1 choice: treat X/Z as false for control flow.
                LogicBit::X | LogicBit::Z => {
                    if let Some(e) = else_branch {
                        exec_stmt(e, env, m, pending)
                    } else {
                        Ok(())
                    }
                }
            }
        }
        Stmt::NbaAssign { lhs, rhs } => {
            let rhs_v = eval_expr(rhs, env)?;
            apply_nba(lhs, &rhs_v, env, m, pending)
        }
        Stmt::Display { .. } => Ok(()),
        Stmt::Empty => Ok(()),
    }
}

fn eval_expr(e: &VExpr, env: &crate::Env) -> Result<Value4> {
    // Reuse the already-parsed expression AST by evaluating it directly via
    // eval_ast. Unfortunately eval_ast is private; so we round-trip via
    // rendering isn't ideal. For v1 simplicity: evaluate by temporarily
    // exposing a string rendering if needed.
    //
    // Instead, we can re-run eval_expr by rendering the AST; keep it consistent
    // with fuzz renderer.
    let s = render_vexpr(e);
    crate::eval_expr(&s, env).map(|r| r.value)
}

fn ensure_pending<'a>(
    pending: &'a mut BTreeMap<String, PendingBits>,
    reg: &str,
    width: u32,
) -> &'a mut PendingBits {
    pending
        .entry(reg.to_string())
        .or_insert_with(|| PendingBits {
            bits: vec![None; width as usize],
        })
}

fn apply_nba(
    lhs: &Lhs,
    rhs: &Value4,
    env: &crate::Env,
    m: &CompiledSeqBlock,
    pending: &mut BTreeMap<String, PendingBits>,
) -> Result<()> {
    match lhs {
        Lhs::Ident(reg) => {
            let info = m
                .decls
                .get(reg)
                .ok_or_else(|| Error::Parse(format!("no decl for {reg}")))?;
            let rhs2 = rhs.resize(info.width);
            let pb = ensure_pending(pending, reg, info.width);
            for i in 0..(info.width as usize) {
                pb.bits[i] = Some(rhs2.bits_lsb_first()[i]);
            }
            Ok(())
        }
        Lhs::Index { base, index } => {
            let info = m
                .decls
                .get(base)
                .ok_or_else(|| Error::Parse(format!("no decl for {base}")))?;
            let idx_v = eval_expr(index, env)?;
            let Some(idx_u) = idx_v.to_u32_saturating_if_known() else {
                // Unknown index => conservative: clobber whole reg to X.
                let pb = ensure_pending(pending, base, info.width);
                for i in 0..(info.width as usize) {
                    pb.bits[i] = Some(LogicBit::X);
                }
                return Ok(());
            };
            let Some((offset, elem_width)) = packed_index_selection_if_in_bounds(info, &[idx_u])?
            else {
                // Out-of-bounds indexed write is a no-op.
                return Ok(());
            };
            let pb = ensure_pending(pending, base, info.width);
            if elem_width == 1 {
                if offset < info.width {
                    pb.bits[offset as usize] = Some(rhs.resize(1).bits_lsb_first()[0]);
                }
            } else {
                let rhs2 = rhs.resize(elem_width);
                for i in 0..elem_width {
                    let dst = offset + i;
                    if dst < info.width {
                        pb.bits[dst as usize] = Some(rhs2.bits_lsb_first()[i as usize]);
                    }
                }
            }
            Ok(())
        }
        Lhs::PackedIndex { base, indices } => {
            let info = m
                .decls
                .get(base)
                .ok_or_else(|| Error::Parse(format!("no decl for {base}")))?;
            let mut idx_vals: Vec<u32> = Vec::with_capacity(indices.len());
            for index in indices {
                let idx_v = eval_expr(index, env)?;
                let Some(idx_u) = idx_v.to_u32_saturating_if_known() else {
                    let pb = ensure_pending(pending, base, info.width);
                    for i in 0..(info.width as usize) {
                        pb.bits[i] = Some(LogicBit::X);
                    }
                    return Ok(());
                };
                idx_vals.push(idx_u);
            }
            let Some((offset, elem_width)) = packed_index_selection_if_in_bounds(info, &idx_vals)?
            else {
                // Out-of-bounds indexed write is a no-op.
                return Ok(());
            };
            let rhs2 = rhs.resize(elem_width);
            let pb = ensure_pending(pending, base, info.width);
            for i in 0..elem_width {
                let dst = offset + i;
                if dst < info.width {
                    pb.bits[dst as usize] = Some(rhs2.bits_lsb_first()[i as usize]);
                }
            }
            Ok(())
        }
        Lhs::Slice { base, msb, lsb } => {
            let info = m
                .decls
                .get(base)
                .ok_or_else(|| Error::Parse(format!("no decl for {base}")))?;
            let msb_v = eval_expr(msb, env)?;
            let lsb_v = eval_expr(lsb, env)?;
            let (Some(msb_u), Some(lsb_u)) = (
                msb_v.to_u32_saturating_if_known(),
                lsb_v.to_u32_saturating_if_known(),
            ) else {
                let pb = ensure_pending(pending, base, info.width);
                for i in 0..(info.width as usize) {
                    pb.bits[i] = Some(LogicBit::X);
                }
                return Ok(());
            };
            if msb_u < lsb_u {
                return Ok(());
            }
            let w = msb_u - lsb_u + 1;
            let rhs2 = rhs.resize(w);
            let pb = ensure_pending(pending, base, info.width);
            for i in 0..w {
                let dst = lsb_u + i;
                if dst < info.width {
                    pb.bits[dst as usize] = Some(rhs2.bits_lsb_first()[i as usize]);
                }
            }
            Ok(())
        }
    }
}

fn render_vexpr(e: &VExpr) -> String {
    match e {
        VExpr::Ident(name) => name.clone(),
        VExpr::Literal(v) | VExpr::UnsizedNumber(v) => {
            let bits = v.to_bit_string_msb_first();
            match v.signedness {
                Signedness::Signed => format!("{}'sb{}", v.width, bits),
                Signedness::Unsigned => format!("{}'b{}", v.width, bits),
            }
        }
        VExpr::UnbasedUnsized(bit) => format!("'{}", bit.as_char()),
        VExpr::Call { name, args } => {
            let mut out = String::new();
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&render_vexpr(a));
            }
            out.push(')');
            out
        }
        VExpr::Concat(parts) => {
            let mut out = String::new();
            out.push('{');
            for (i, p) in parts.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&render_vexpr(p));
            }
            out.push('}');
            out
        }
        VExpr::Replicate { count, expr } => {
            let c = render_vexpr(count);
            let ee = render_vexpr(expr);
            format!("{{{c}{{{ee}}}}}")
        }
        VExpr::Cast { width, expr } => {
            let ww = render_vexpr(width);
            let ee = render_vexpr(expr);
            format!("({ww})'({ee})")
        }
        VExpr::Index { expr, index } => {
            let ee = render_vexpr(expr);
            let ii = render_vexpr(index);
            format!("({ee})[{ii}]")
        }
        VExpr::Slice { expr, msb, lsb } => {
            let ee = render_vexpr(expr);
            let mm = render_vexpr(msb);
            let ll = render_vexpr(lsb);
            format!("({ee})[{mm}:{ll}]")
        }
        VExpr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => {
            let ee = render_vexpr(expr);
            let bb = render_vexpr(base);
            let ww = render_vexpr(width);
            let dir = if *upward { "+:" } else { "-:" };
            format!("({ee})[{bb} {dir} {ww}]")
        }
        VExpr::Unary { op, expr } => {
            let inner = render_vexpr(expr);
            let op_str = match op {
                crate::ast::UnaryOp::LogicalNot => "!",
                crate::ast::UnaryOp::BitNot => "~",
                crate::ast::UnaryOp::UnaryPlus => "+",
                crate::ast::UnaryOp::UnaryMinus => "-",
                crate::ast::UnaryOp::ReduceAnd => "&",
                crate::ast::UnaryOp::ReduceNand => "~&",
                crate::ast::UnaryOp::ReduceOr => "|",
                crate::ast::UnaryOp::ReduceNor => "~|",
                crate::ast::UnaryOp::ReduceXor => "^",
                crate::ast::UnaryOp::ReduceXnor => "^~",
            };
            format!("({op_str}{inner})")
        }
        VExpr::Binary { op, lhs, rhs } => {
            let a = render_vexpr(lhs);
            let b = render_vexpr(rhs);
            let op_str = match op {
                crate::ast::BinaryOp::Add => "+",
                crate::ast::BinaryOp::Sub => "-",
                crate::ast::BinaryOp::Mul => "*",
                crate::ast::BinaryOp::Div => "/",
                crate::ast::BinaryOp::Mod => "%",
                crate::ast::BinaryOp::Shl => "<<",
                crate::ast::BinaryOp::Shr => ">>",
                crate::ast::BinaryOp::Sshr => ">>>",
                crate::ast::BinaryOp::BitAnd => "&",
                crate::ast::BinaryOp::BitOr => "|",
                crate::ast::BinaryOp::BitXor => "^",
                crate::ast::BinaryOp::LogicalAnd => "&&",
                crate::ast::BinaryOp::LogicalOr => "||",
                crate::ast::BinaryOp::Lt => "<",
                crate::ast::BinaryOp::Le => "<=",
                crate::ast::BinaryOp::Gt => ">",
                crate::ast::BinaryOp::Ge => ">=",
                crate::ast::BinaryOp::Eq => "==",
                crate::ast::BinaryOp::Neq => "!=",
                crate::ast::BinaryOp::CaseEq => "===",
                crate::ast::BinaryOp::CaseNeq => "!==",
            };
            format!("({a}{op_str}{b})")
        }
        VExpr::Ternary { cond, t, f } => {
            let c = render_vexpr(cond);
            let tt = render_vexpr(t);
            let ff = render_vexpr(f);
            format!("(({c})?({tt}):({ff}))")
        }
    }
}
