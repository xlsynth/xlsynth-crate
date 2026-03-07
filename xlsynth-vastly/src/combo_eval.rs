// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::CoverageCounters;
use crate::Env;
use crate::Error;
use crate::Result;
use crate::Signedness;
use crate::SourceText;
use crate::SpanKey;
use crate::Value4;
use crate::ast::Expr;
use crate::ast_spanned::SpannedExpr;
use crate::ast_spanned::SpannedExprKind;
use crate::combo_compile::CasezArm;
use crate::combo_compile::CasezPattern;
use crate::combo_compile::CompiledComboModule;
use crate::combo_compile::CompiledFunction;
use crate::combo_compile::CompiledFunctionBody;
use crate::eval::CallResolver;
use crate::eval::binary_operand_expected_signednesses;
use crate::eval::binary_operand_expected_widths;
use crate::eval::eval_ast_with_calls;
use crate::eval::eval_ast_with_calls_and_ctx;
use crate::eval::eval_binary_op;
use crate::eval::eval_unary_op;
use crate::eval::merged_signedness;
use crate::eval::operand_with_own_sign_ctx;
use crate::eval::replication_count_to_u32;
use crate::eval::unary_operand_expected_width;
use crate::packed::packed_index_selection;
use crate::packed::packed_index_selection_if_in_bounds;
use crate::sv_ast::Lhs;
use crate::value::LogicBit;

pub struct ComboEvalPlan {
    /// Assign indices in evaluation order.
    pub assign_order: Vec<usize>,
}

pub fn plan_combo_eval(m: &CompiledComboModule) -> Result<ComboEvalPlan> {
    let mut lhs_to_idxs: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, a) in m.assigns.iter().enumerate() {
        lhs_to_idxs
            .entry(a.lhs_base().to_string())
            .or_default()
            .push(i);
    }
    let mut const_env = Env::new();
    for (name, value) in &m.consts {
        const_env.insert(name.clone(), value.clone());
    }

    for (lhs, writer_indices) in &lhs_to_idxs {
        if writer_indices.len() <= 1 {
            continue;
        }
        let info = m
            .decls
            .get(lhs)
            .ok_or_else(|| Error::Parse(format!("no decl for assign lhs `{lhs}`")))?;
        let mut seen_writers_by_bit: Vec<Option<usize>> = vec![None; info.width as usize];
        for &writer in writer_indices {
            let assign = &m.assigns[writer];
            let mut deps: BTreeSet<String> = BTreeSet::new();
            collect_idents(&assign.rhs, &mut deps);
            if deps.contains(lhs) {
                return Err(Error::Parse(format!(
                    "multiple assigns to `{lhs}` with RHS dependency on `{lhs}` are not supported"
                )));
            }
            let static_bits =
                static_written_bits(&assign.lhs, info, &const_env).map_err(|e| match e {
                    Error::Parse(msg) => Error::Parse(format!(
                        "multiple assigns to `{lhs}` require static disjoint LHS selections: {msg}"
                    )),
                    other => other,
                })?;
            for bit in static_bits {
                let slot = bit as usize;
                if let Some(prev_writer) = seen_writers_by_bit[slot] {
                    return Err(Error::Parse(format!(
                        "overlapping assigns to `{lhs}` are not supported (assign {prev_writer} overlaps assign {writer})"
                    )));
                }
                seen_writers_by_bit[slot] = Some(writer);
            }
        }
    }

    let mut indeg: Vec<u32> = vec![0; m.assigns.len()];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m.assigns.len()];

    for (i, a) in m.assigns.iter().enumerate() {
        let mut deps: BTreeSet<String> = BTreeSet::new();
        collect_idents(&a.rhs, &mut deps);
        for d in deps {
            if let Some(writers) = lhs_to_idxs.get(&d) {
                for &j in writers {
                    // i depends on j => edge j -> i
                    succ[j].push(i);
                    indeg[i] += 1;
                }
            }
        }
    }

    let mut q: VecDeque<usize> = VecDeque::new();
    for i in 0..indeg.len() {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }
    let mut order: Vec<usize> = Vec::with_capacity(m.assigns.len());
    while let Some(n) = q.pop_front() {
        order.push(n);
        for &s in &succ[n] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                q.push_back(s);
            }
        }
    }
    if order.len() != m.assigns.len() {
        return Err(Error::Parse(
            "combinational cycle detected in assigns".to_string(),
        ));
    }
    Ok(ComboEvalPlan {
        assign_order: order,
    })
}

pub fn eval_combo(
    m: &CompiledComboModule,
    plan: &ComboEvalPlan,
    inputs: &BTreeMap<String, Value4>,
) -> Result<BTreeMap<String, Value4>> {
    let mut env = Env::new();
    // Initialize everything to X.
    for (name, info) in &m.decls {
        env.insert(name.clone(), x_value(info.width, info.signedness));
    }
    for (name, value) in &m.consts {
        env.insert(name.clone(), value.clone());
    }
    // Drive inputs.
    for p in &m.input_ports {
        let v = inputs
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("missing input `{}`", p.name)))?;
        let info = m
            .decls
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("no decl for input `{}`", p.name)))?;
        env.insert(p.name.clone(), coerce_to_declinfo(v, info));
    }

    eval_combo_assigns(m, plan, &mut env)?;

    // Snapshot.
    let mut out: BTreeMap<String, Value4> = BTreeMap::new();
    for name in m.decls.keys() {
        if let Some(v) = env.get(name) {
            out.insert(name.clone(), v.clone());
        }
    }
    Ok(out)
}

pub fn eval_combo_seeded(
    m: &CompiledComboModule,
    plan: &ComboEvalPlan,
    seed: &Env,
) -> Result<BTreeMap<String, Value4>> {
    let mut env = Env::new();
    // Initialize everything to X.
    for (name, info) in &m.decls {
        env.insert(name.clone(), x_value(info.width, info.signedness));
    }
    for (name, value) in &m.consts {
        env.insert(name.clone(), value.clone());
    }
    // Overlay seed values.
    for (k, v) in seed.iter() {
        let info = m
            .decls
            .get(k)
            .ok_or_else(|| Error::Parse(format!("no decl for seeded identifier `{}`", k)))?;
        env.insert(k.clone(), coerce_to_declinfo(v, info));
    }

    eval_combo_assigns(m, plan, &mut env)?;

    // Snapshot.
    let mut out: BTreeMap<String, Value4> = BTreeMap::new();
    for name in m.decls.keys() {
        if let Some(v) = env.get(name) {
            out.insert(name.clone(), v.clone());
        }
    }
    Ok(out)
}

fn eval_combo_assigns(m: &CompiledComboModule, plan: &ComboEvalPlan, env: &mut Env) -> Result<()> {
    for &ai in &plan.assign_order {
        let a = &m.assigns[ai];
        let lhs_base = a.lhs_base();
        let info = m
            .decls
            .get(lhs_base)
            .ok_or_else(|| Error::Parse(format!("no decl for assign lhs `{lhs_base}`")))?;
        let expected_width = lhs_expected_write_width(&a.lhs, info)?;
        let resolver = ComboResolver {
            funcs: &m.functions,
            globals: env,
        };
        let rhs_v = eval_ast_with_calls(&a.rhs, env, Some(&resolver), expected_width)?;
        apply_lhs_to_env(&a.lhs, &rhs_v, env, info)?;
    }
    Ok(())
}

pub fn eval_combo_seeded_with_coverage(
    m: &CompiledComboModule,
    plan: &ComboEvalPlan,
    seed: &Env,
    src: &SourceText,
    cov: &mut CoverageCounters,
    fn_meta: &BTreeMap<String, crate::pipeline_compile::FunctionMeta>,
) -> Result<BTreeMap<String, Value4>> {
    let mut env = Env::new();
    for (name, info) in &m.decls {
        env.insert(name.clone(), x_value(info.width, info.signedness));
    }
    for (name, value) in &m.consts {
        env.insert(name.clone(), value.clone());
    }
    for (k, v) in seed.iter() {
        let info = m
            .decls
            .get(k)
            .ok_or_else(|| Error::Parse(format!("no decl for seeded identifier `{}`", k)))?;
        env.insert(k.clone(), coerce_to_declinfo(v, info));
    }
    for &ai in &plan.assign_order {
        let a = &m.assigns[ai];
        let lhs_base = a.lhs_base();
        cov.hit_span(src, a.rhs_span);
        let info = m
            .decls
            .get(lhs_base)
            .ok_or_else(|| Error::Parse(format!("no decl for assign lhs `{lhs_base}`")))?;
        let expected_width = lhs_expected_write_width(&a.lhs, info)?;
        let rhs_v = eval_spanned_expr_with_funcs(
            &a.rhs_spanned,
            &env,
            &m.functions,
            expected_width,
            None,
            cov,
            src,
            fn_meta,
        )?;
        apply_lhs_to_env(&a.lhs, &rhs_v, &mut env, info)?;
    }
    let mut out: BTreeMap<String, Value4> = BTreeMap::new();
    for name in m.decls.keys() {
        if let Some(v) = env.get(name) {
            out.insert(name.clone(), v.clone());
        }
    }
    Ok(out)
}

fn eval_spanned_expr_with_funcs(
    expr: &SpannedExpr,
    env: &Env,
    funcs: &BTreeMap<String, CompiledFunction>,
    expected_width: Option<u32>,
    expected_signedness: Option<Signedness>,
    cov: &mut CoverageCounters,
    src: &SourceText,
    fn_meta: &BTreeMap<String, crate::pipeline_compile::FunctionMeta>,
) -> Result<Value4> {
    match &expr.kind {
        SpannedExprKind::Ident(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownIdentifier(name.clone())),
        SpannedExprKind::Literal(v) => Ok(v.clone()),
        SpannedExprKind::UnsizedNumber(v) => Ok(v.clone()),
        SpannedExprKind::UnbasedUnsized(bit) => {
            let w = expected_width.unwrap_or(1);
            Ok(Value4::new(w, Signedness::Unsigned, vec![*bit; w as usize]))
        }
        SpannedExprKind::Call { name, args } => {
            if name == "$signed" || name == "$unsigned" {
                if args.len() != 1 {
                    return Err(Error::Parse(format!(
                        "builtin cast `{name}` expects 1 argument, got {}",
                        args.len()
                    )));
                }
                let v = eval_spanned_expr_with_funcs(
                    &args[0], env, funcs, None, None, cov, src, fn_meta,
                )?;
                return Ok(if name == "$signed" {
                    v.with_signedness(Signedness::Signed)
                } else {
                    v.with_signedness(Signedness::Unsigned)
                });
            }
            *cov.function_calls.entry(name.clone()).or_insert(0) += 1;
            let f = funcs
                .get(name)
                .ok_or_else(|| Error::Parse(format!("unknown function `{name}`")))?;
            if args.len() != f.args.len() {
                return Err(Error::Parse(format!(
                    "function `{name}` arg count mismatch: got {} want {}",
                    args.len(),
                    f.args.len()
                )));
            }
            let mut avs: Vec<Value4> = Vec::with_capacity(args.len());
            for a in args {
                avs.push(eval_spanned_expr_with_funcs(
                    a, env, funcs, None, None, cov, src, fn_meta,
                )?);
            }
            eval_compiled_function(
                f,
                &avs,
                funcs,
                env,
                expected_width,
                expected_signedness,
                cov,
                src,
                fn_meta,
            )
        }
        SpannedExprKind::Concat(parts) => {
            let mut vs: Vec<Value4> = Vec::with_capacity(parts.len());
            for p in parts {
                vs.push(eval_spanned_expr_with_funcs(
                    p, env, funcs, None, None, cov, src, fn_meta,
                )?);
            }
            Ok(Value4::concat(&vs))
        }
        SpannedExprKind::Replicate { count, expr } => {
            let c = eval_spanned_expr_with_funcs(count, env, funcs, None, None, cov, src, fn_meta)?;
            let count_u = replication_count_to_u32(&c)?;
            let v = eval_spanned_expr_with_funcs(expr, env, funcs, None, None, cov, src, fn_meta)?;
            Ok(Value4::replicate(count_u, &v))
        }
        SpannedExprKind::Cast { width, expr } => {
            let width_v =
                eval_spanned_expr_with_funcs(width, env, funcs, None, None, cov, src, fn_meta)?;
            let width_u = width_v
                .to_u32_if_known()
                .ok_or_else(|| Error::Parse("cast width must be known".to_string()))?;
            let v = eval_spanned_expr_with_funcs(
                expr,
                env,
                funcs,
                Some(width_u),
                None,
                cov,
                src,
                fn_meta,
            )?;
            Ok(v.resize(width_u))
        }
        SpannedExprKind::Index { expr, index } => {
            let v = eval_spanned_expr_with_funcs(expr, env, funcs, None, None, cov, src, fn_meta)?;
            let idx_v =
                eval_spanned_expr_with_funcs(index, env, funcs, None, None, cov, src, fn_meta)?;
            match idx_v.to_u32_saturating_if_known() {
                Some(idx_u) => Ok(v.index(idx_u)),
                None => Ok(Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])),
            }
        }
        SpannedExprKind::Slice { expr, msb, lsb } => {
            let v = eval_spanned_expr_with_funcs(expr, env, funcs, None, None, cov, src, fn_meta)?;
            let msb_v =
                eval_spanned_expr_with_funcs(msb, env, funcs, None, None, cov, src, fn_meta)?;
            let lsb_v =
                eval_spanned_expr_with_funcs(lsb, env, funcs, None, None, cov, src, fn_meta)?;
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
        SpannedExprKind::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => {
            let v = eval_spanned_expr_with_funcs(expr, env, funcs, None, None, cov, src, fn_meta)?;
            let base_v =
                eval_spanned_expr_with_funcs(base, env, funcs, None, None, cov, src, fn_meta)?;
            let width_v =
                eval_spanned_expr_with_funcs(width, env, funcs, None, None, cov, src, fn_meta)?;
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
        SpannedExprKind::Unary { op, expr } => {
            let child_expected_width = unary_operand_expected_width(*op, expected_width);
            let v = eval_spanned_expr_with_funcs(
                expr,
                env,
                funcs,
                child_expected_width,
                expected_signedness,
                cov,
                src,
                fn_meta,
            )?;
            Ok(eval_unary_op(*op, v, expected_width, expected_signedness))
        }
        SpannedExprKind::Binary { op, lhs, rhs } => {
            let (lhs_expected_width, rhs_expected_width) =
                binary_operand_expected_widths(*op, expected_width);
            let (lhs_expected_signedness, rhs_expected_signedness) =
                binary_operand_expected_signednesses(*op, expected_signedness);
            let a0 = eval_spanned_expr_with_funcs(
                lhs,
                env,
                funcs,
                lhs_expected_width,
                lhs_expected_signedness,
                cov,
                src,
                fn_meta,
            )?;
            let b0 = eval_spanned_expr_with_funcs(
                rhs,
                env,
                funcs,
                rhs_expected_width,
                rhs_expected_signedness,
                cov,
                src,
                fn_meta,
            )?;
            let op_expected_width = match op {
                crate::ast::BinaryOp::Add
                | crate::ast::BinaryOp::Sub
                | crate::ast::BinaryOp::Mul
                | crate::ast::BinaryOp::Div
                | crate::ast::BinaryOp::Mod
                | crate::ast::BinaryOp::BitAnd
                | crate::ast::BinaryOp::BitOr
                | crate::ast::BinaryOp::BitXor => {
                    Some(expected_width.unwrap_or(0).max(a0.width.max(b0.width)))
                }
                crate::ast::BinaryOp::Lt
                | crate::ast::BinaryOp::Le
                | crate::ast::BinaryOp::Gt
                | crate::ast::BinaryOp::Ge
                | crate::ast::BinaryOp::Eq
                | crate::ast::BinaryOp::Neq
                | crate::ast::BinaryOp::CaseEq
                | crate::ast::BinaryOp::CaseNeq => Some(a0.width.max(b0.width)),
                crate::ast::BinaryOp::Shl
                | crate::ast::BinaryOp::Shr
                | crate::ast::BinaryOp::Sshr => expected_width,
                crate::ast::BinaryOp::LogicalAnd | crate::ast::BinaryOp::LogicalOr => None,
            };
            let op_lhs_expected_width_rhs_expected_width = match op {
                crate::ast::BinaryOp::Lt
                | crate::ast::BinaryOp::Le
                | crate::ast::BinaryOp::Gt
                | crate::ast::BinaryOp::Ge
                | crate::ast::BinaryOp::Eq
                | crate::ast::BinaryOp::Neq
                | crate::ast::BinaryOp::CaseEq
                | crate::ast::BinaryOp::CaseNeq => (op_expected_width, op_expected_width),
                _ => binary_operand_expected_widths(*op, op_expected_width),
            };
            let op_expected_signedness = match op {
                crate::ast::BinaryOp::Add
                | crate::ast::BinaryOp::Sub
                | crate::ast::BinaryOp::Mul
                | crate::ast::BinaryOp::Div
                | crate::ast::BinaryOp::Mod
                | crate::ast::BinaryOp::BitAnd
                | crate::ast::BinaryOp::BitOr
                | crate::ast::BinaryOp::BitXor => {
                    Some(expected_signedness.unwrap_or_else(|| merged_signedness(&a0, &b0)))
                }
                crate::ast::BinaryOp::Lt
                | crate::ast::BinaryOp::Le
                | crate::ast::BinaryOp::Gt
                | crate::ast::BinaryOp::Ge
                | crate::ast::BinaryOp::Eq
                | crate::ast::BinaryOp::Neq
                | crate::ast::BinaryOp::CaseEq
                | crate::ast::BinaryOp::CaseNeq => Some(merged_signedness(&a0, &b0)),
                crate::ast::BinaryOp::Shl
                | crate::ast::BinaryOp::Shr
                | crate::ast::BinaryOp::Sshr => expected_signedness,
                crate::ast::BinaryOp::LogicalAnd | crate::ast::BinaryOp::LogicalOr => None,
            };
            let op_lhs_expected_signedness_rhs_expected_signedness = match op {
                crate::ast::BinaryOp::Lt
                | crate::ast::BinaryOp::Le
                | crate::ast::BinaryOp::Gt
                | crate::ast::BinaryOp::Ge
                | crate::ast::BinaryOp::Eq
                | crate::ast::BinaryOp::Neq
                | crate::ast::BinaryOp::CaseEq
                | crate::ast::BinaryOp::CaseNeq => (op_expected_signedness, op_expected_signedness),
                _ => binary_operand_expected_signednesses(*op, op_expected_signedness),
            };
            let needs_recontext = op_expected_signedness != expected_signedness
                || op_expected_width != expected_width
                || op_lhs_expected_width_rhs_expected_width
                    != (lhs_expected_width, rhs_expected_width)
                || op_lhs_expected_signedness_rhs_expected_signedness
                    != (lhs_expected_signedness, rhs_expected_signedness);
            let (a, b) = if needs_recontext {
                let a = eval_spanned_expr_with_funcs(
                    lhs,
                    env,
                    funcs,
                    op_lhs_expected_width_rhs_expected_width.0,
                    op_lhs_expected_signedness_rhs_expected_signedness.0,
                    cov,
                    src,
                    fn_meta,
                )?;
                let b = eval_spanned_expr_with_funcs(
                    rhs,
                    env,
                    funcs,
                    op_lhs_expected_width_rhs_expected_width.1,
                    op_lhs_expected_signedness_rhs_expected_signedness.1,
                    cov,
                    src,
                    fn_meta,
                )?;
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
        SpannedExprKind::Ternary { cond, t, f } => {
            let c = eval_spanned_expr_with_funcs(cond, env, funcs, None, None, cov, src, fn_meta)?;
            cov.record_ternary_decision_with_spans(
                SpanKey::from(expr.span),
                SpanKey::from(t.span),
                SpanKey::from(f.span),
                c.to_bool4(),
            );
            let tv0 = eval_spanned_expr_with_funcs(
                t,
                env,
                funcs,
                expected_width,
                expected_signedness,
                cov,
                src,
                fn_meta,
            )?;
            let fv0 = eval_spanned_expr_with_funcs(
                f,
                env,
                funcs,
                expected_width,
                expected_signedness,
                cov,
                src,
                fn_meta,
            )?;
            let branch_expected_width =
                Some(expected_width.unwrap_or(0).max(tv0.width.max(fv0.width)));
            let recontext_t = branch_expected_width != expected_width
                && ternary_branch_needs_width_recontext_spanned(t);
            let recontext_f = branch_expected_width != expected_width
                && ternary_branch_needs_width_recontext_spanned(f);
            let tv = if recontext_t {
                eval_spanned_expr_with_funcs(
                    t,
                    env,
                    funcs,
                    branch_expected_width,
                    expected_signedness,
                    cov,
                    src,
                    fn_meta,
                )?
            } else {
                tv0
            };
            let fv = if recontext_f {
                eval_spanned_expr_with_funcs(
                    f,
                    env,
                    funcs,
                    branch_expected_width,
                    expected_signedness,
                    cov,
                    src,
                    fn_meta,
                )?
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

fn ternary_branch_needs_width_recontext_spanned(expr: &SpannedExpr) -> bool {
    matches!(
        expr.kind,
        SpannedExprKind::UnbasedUnsized(_)
            | SpannedExprKind::Call { .. }
            | SpannedExprKind::Unary { .. }
            | SpannedExprKind::Binary { .. }
            | SpannedExprKind::Ternary { .. }
    )
}

fn eval_compiled_function(
    f: &CompiledFunction,
    args: &[Value4],
    funcs: &BTreeMap<String, CompiledFunction>,
    globals: &Env,
    expected_width: Option<u32>,
    _expected_signedness: Option<Signedness>,
    cov: &mut CoverageCounters,
    src: &SourceText,
    fn_meta: &BTreeMap<String, crate::pipeline_compile::FunctionMeta>,
) -> Result<Value4> {
    let mut env = init_function_env(f, args, globals);
    if let Some(meta) = fn_meta.get(&f.name) {
        // Attribute function execution only to scaffold spans, not whole body,
        // so unselected arms remain line-missed.
        for s in &meta.scaffold_spans {
            cov.bump_span(*s);
            cov.hit_span(src, *s);
        }
    }
    match &f.body {
        CompiledFunctionBody::Expr { expr, expr_spanned } => {
            if let Some(meta) = fn_meta.get(&f.name) {
                if let Some(s) = meta.assign_expr_span {
                    cov.bump_span(s);
                    cov.hit_span(src, s);
                }
            }
            let v = if let Some(spanned) = expr_spanned {
                let ret = function_return_decl(f);
                eval_spanned_expr_with_funcs(
                    spanned,
                    &env,
                    funcs,
                    Some(ret.width),
                    Some(ret.signedness),
                    cov,
                    src,
                    fn_meta,
                )?
            } else {
                let ret = function_return_decl(f);
                let resolver = ComboResolver { funcs, globals };
                eval_ast_with_calls_and_ctx(
                    expr,
                    &env,
                    Some(&resolver),
                    Some(ret.width),
                    Some(ret.signedness),
                )?
            };
            Ok(coerce_to_declinfo(&v, &function_return_decl(f)))
        }
        CompiledFunctionBody::Casez { selector, arms } => {
            let resolver = ComboResolver { funcs, globals };
            let sel_v = eval_ast_with_calls(selector, &env, Some(&resolver), None)?;
            let sel_bits = sel_v.to_bit_string_msb_first();
            let mut default_arm: Option<&CasezArm> = None;
            let mut default_idx: Option<usize> = None;
            for (i, arm) in arms.iter().enumerate() {
                if arm.pat.is_none() {
                    default_arm = Some(arm);
                    default_idx = Some(i);
                    continue;
                }
                let pat = arm.pat.as_ref().unwrap();
                if casez_matches(&sel_bits, pat) {
                    if let Some(meta) = fn_meta.get(&f.name) {
                        if let Some(am) = meta.arms.get(i) {
                            cov.bump_selected_arm(am.arm_span);
                            cov.bump_span(am.value_span);
                            cov.hit_span(src, am.value_span);
                            cov.hit_span(src, am.arm_span);
                        }
                    }
                    let mut v =
                        eval_ast_with_calls(&arm.value, &env, Some(&resolver), Some(f.ret_width))?;
                    v = coerce_to_declinfo(&v, &function_return_decl(f));
                    return Ok(v);
                }
            }
            if let Some(d) = default_arm {
                if let Some(meta) = fn_meta.get(&f.name) {
                    if let Some(di) = default_idx {
                        if let Some(am) = meta.arms.get(di) {
                            cov.bump_selected_arm(am.arm_span);
                            cov.bump_span(am.value_span);
                            cov.hit_span(src, am.value_span);
                            cov.hit_span(src, am.arm_span);
                        }
                    } else if let Some(am) = meta.arms.last() {
                        cov.bump_selected_arm(am.arm_span);
                        cov.bump_span(am.value_span);
                        cov.hit_span(src, am.value_span);
                        cov.hit_span(src, am.arm_span);
                    }
                }
                let mut v =
                    eval_ast_with_calls(&d.value, &env, Some(&resolver), Some(f.ret_width))?;
                v = coerce_to_declinfo(&v, &function_return_decl(f));
                return Ok(v);
            }
            let w = expected_width.unwrap_or(f.ret_width);
            Ok(x_value(w, Signedness::Unsigned))
        }
        CompiledFunctionBody::Procedure { assigns } => {
            let resolver = ComboResolver { funcs, globals };
            exec_function_procedure(f, &mut env, assigns, &resolver)
        }
    }
}

fn x_value(width: u32, signedness: Signedness) -> Value4 {
    Value4::new(width, signedness, vec![LogicBit::X; width as usize])
}

fn function_return_decl(f: &CompiledFunction) -> crate::module_compile::DeclInfo {
    crate::module_compile::DeclInfo {
        width: f.ret_width,
        signedness: f.ret_signedness,
        packed_dims: vec![f.ret_width],
        unpacked_dims: vec![],
    }
}

fn coerce_to_declinfo(v: &Value4, info: &crate::module_compile::DeclInfo) -> Value4 {
    let resized = v.resize(info.width);
    Value4::new(
        info.width,
        info.signedness,
        resized.bits_lsb_first().to_vec(),
    )
}

fn lhs_expected_write_width(
    lhs: &Lhs,
    info: &crate::module_compile::DeclInfo,
) -> Result<Option<u32>> {
    match lhs {
        Lhs::Ident(_) => Ok(Some(info.width)),
        Lhs::Index { .. } => {
            let (_offset, width) = packed_index_selection(info, &[0])?;
            Ok(Some(width))
        }
        Lhs::PackedIndex { indices, .. } => {
            let zeros = vec![0; indices.len()];
            let (_offset, width) = packed_index_selection(info, &zeros)?;
            Ok(Some(width))
        }
        Lhs::Slice { .. } => Ok(None),
    }
}

fn apply_lhs_to_env(
    lhs: &Lhs,
    rhs: &Value4,
    env: &mut Env,
    info: &crate::module_compile::DeclInfo,
) -> Result<()> {
    match lhs {
        Lhs::Ident(base) => {
            env.insert(base.clone(), coerce_to_declinfo(rhs, info));
            Ok(())
        }
        Lhs::Index { base, index } => {
            let index_v = eval_ast_with_calls(index, env, None, None)?;
            let Some(index_u) = index_v.to_u32_saturating_if_known() else {
                clobber_lhs_base_to_x(base, info, env);
                return Ok(());
            };
            let Some((offset, width)) = packed_index_selection_if_in_bounds(info, &[index_u])?
            else {
                // Out-of-bounds indexed write is a no-op.
                return Ok(());
            };
            write_partial_lhs(base, info, rhs, offset, width, env);
            Ok(())
        }
        Lhs::PackedIndex { base, indices } => {
            let mut index_values: Vec<u32> = Vec::with_capacity(indices.len());
            for index in indices {
                let index_v = eval_ast_with_calls(index, env, None, None)?;
                let Some(index_u) = index_v.to_u32_saturating_if_known() else {
                    clobber_lhs_base_to_x(base, info, env);
                    return Ok(());
                };
                index_values.push(index_u);
            }
            let Some((offset, width)) = packed_index_selection_if_in_bounds(info, &index_values)?
            else {
                // Out-of-bounds indexed write is a no-op.
                return Ok(());
            };
            write_partial_lhs(base, info, rhs, offset, width, env);
            Ok(())
        }
        Lhs::Slice { base, msb, lsb } => {
            let msb_v = eval_ast_with_calls(msb, env, None, None)?;
            let lsb_v = eval_ast_with_calls(lsb, env, None, None)?;
            let (Some(msb_u), Some(lsb_u)) = (
                msb_v.to_u32_saturating_if_known(),
                lsb_v.to_u32_saturating_if_known(),
            ) else {
                clobber_lhs_base_to_x(base, info, env);
                return Ok(());
            };
            if msb_u < lsb_u {
                return Ok(());
            }
            let width = msb_u - lsb_u + 1;
            write_partial_lhs(base, info, rhs, lsb_u, width, env);
            Ok(())
        }
    }
}

fn write_partial_lhs(
    base: &str,
    info: &crate::module_compile::DeclInfo,
    rhs: &Value4,
    offset: u32,
    width: u32,
    env: &mut Env,
) {
    let current = env
        .get(base)
        .cloned()
        .unwrap_or_else(|| x_value(info.width, info.signedness));
    let mut bits = current.resize(info.width).bits_lsb_first().to_vec();
    let rhs_bits = rhs.resize(width);
    for i in 0..width {
        let dst = offset + i;
        if dst < info.width {
            bits[dst as usize] = rhs_bits.bits_lsb_first()[i as usize];
        }
    }
    env.insert(
        base.to_string(),
        Value4::new(info.width, info.signedness, bits),
    );
}

fn clobber_lhs_base_to_x(base: &str, info: &crate::module_compile::DeclInfo, env: &mut Env) {
    env.insert(base.to_string(), x_value(info.width, info.signedness));
}

fn static_written_bits(
    lhs: &Lhs,
    info: &crate::module_compile::DeclInfo,
    consts: &Env,
) -> Result<Vec<u32>> {
    match lhs {
        Lhs::Ident(_) => Ok((0..info.width).collect()),
        Lhs::Index { index, .. } => {
            let Some(index_u) = eval_static_u32(index, consts)? else {
                return Err(Error::Parse(
                    "index expression is not statically known".to_string(),
                ));
            };
            let Some((offset, width)) = packed_index_selection_if_in_bounds(info, &[index_u])?
            else {
                return Ok(Vec::new());
            };
            Ok((0..width)
                .map(|i| offset + i)
                .filter(|bit| *bit < info.width)
                .collect())
        }
        Lhs::PackedIndex { indices, .. } => {
            let mut values: Vec<u32> = Vec::with_capacity(indices.len());
            for index in indices {
                let Some(index_u) = eval_static_u32(index, consts)? else {
                    return Err(Error::Parse(
                        "packed index expression is not statically known".to_string(),
                    ));
                };
                values.push(index_u);
            }
            let Some((offset, width)) = packed_index_selection_if_in_bounds(info, &values)? else {
                return Ok(Vec::new());
            };
            Ok((0..width)
                .map(|i| offset + i)
                .filter(|bit| *bit < info.width)
                .collect())
        }
        Lhs::Slice { msb, lsb, .. } => {
            let Some(msb_u) = eval_static_u32(msb, consts)? else {
                return Err(Error::Parse(
                    "slice msb expression is not statically known".to_string(),
                ));
            };
            let Some(lsb_u) = eval_static_u32(lsb, consts)? else {
                return Err(Error::Parse(
                    "slice lsb expression is not statically known".to_string(),
                ));
            };
            if msb_u < lsb_u {
                return Ok(Vec::new());
            }
            Ok((lsb_u..=msb_u).filter(|bit| *bit < info.width).collect())
        }
    }
}

fn eval_static_u32(expr: &Expr, consts: &Env) -> Result<Option<u32>> {
    match eval_ast_with_calls(expr, consts, None, None) {
        Ok(value) => Ok(value.to_u32_saturating_if_known()),
        Err(_) => Ok(None),
    }
}

fn init_function_env(f: &CompiledFunction, args: &[Value4], globals: &Env) -> Env {
    let mut env = globals.clone();
    for (arg, av) in f.args.iter().zip(args.iter()) {
        let info = crate::module_compile::DeclInfo {
            width: arg.width,
            signedness: arg.signedness,
            packed_dims: vec![arg.width],
            unpacked_dims: vec![],
        };
        env.insert(arg.name.clone(), coerce_to_declinfo(av, &info));
    }
    for (name, info) in &f.locals {
        env.insert(name.clone(), x_value(info.width, info.signedness));
    }
    let ret = function_return_decl(f);
    env.insert(f.name.clone(), x_value(ret.width, ret.signedness));
    env
}

fn function_target_decl(
    f: &CompiledFunction,
    lhs: &str,
) -> Option<crate::module_compile::DeclInfo> {
    if lhs == f.name {
        return Some(function_return_decl(f));
    }
    if let Some(info) = f.locals.get(lhs) {
        return Some(info.clone());
    }
    f.args
        .iter()
        .find(|a| a.name == lhs)
        .map(|a| crate::module_compile::DeclInfo {
            width: a.width,
            signedness: a.signedness,
            packed_dims: vec![a.width],
            unpacked_dims: vec![],
        })
}

fn exec_function_procedure(
    f: &CompiledFunction,
    env: &mut Env,
    assigns: &[crate::combo_compile::FunctionAssign],
    resolver: &dyn CallResolver,
) -> Result<Value4> {
    for a in assigns {
        let info = function_target_decl(f, &a.lhs).ok_or_else(|| {
            Error::Parse(format!(
                "unknown function-local assignment target `{}` in `{}`",
                a.lhs, f.name
            ))
        })?;
        let rhs_v = eval_ast_with_calls(&a.expr, env, Some(resolver), Some(info.width))?;
        env.insert(a.lhs.clone(), coerce_to_declinfo(&rhs_v, &info));
    }
    let ret = env
        .get(&f.name)
        .cloned()
        .unwrap_or_else(|| x_value(f.ret_width, f.ret_signedness));
    Ok(coerce_to_declinfo(&ret, &function_return_decl(f)))
}

fn collect_idents(e: &Expr, out: &mut BTreeSet<String>) {
    match e {
        Expr::Ident(s) => {
            out.insert(s.clone());
        }
        Expr::Literal(_) => {}
        Expr::UnsizedNumber(_) => {}
        Expr::UnbasedUnsized(_) => {}
        Expr::Call { args, .. } => {
            for a in args {
                collect_idents(a, out);
            }
        }
        Expr::Concat(ps) => {
            for p in ps {
                collect_idents(p, out);
            }
        }
        Expr::Replicate { count, expr } => {
            collect_idents(count, out);
            collect_idents(expr, out);
        }
        Expr::Cast { width, expr } => {
            collect_idents(width, out);
            collect_idents(expr, out);
        }
        Expr::Index { expr, index } => {
            collect_idents(expr, out);
            collect_idents(index, out);
        }
        Expr::Slice { expr, msb, lsb } => {
            collect_idents(expr, out);
            collect_idents(msb, out);
            collect_idents(lsb, out);
        }
        Expr::IndexedSlice {
            expr, base, width, ..
        } => {
            collect_idents(expr, out);
            collect_idents(base, out);
            collect_idents(width, out);
        }
        Expr::Unary { expr, .. } => {
            collect_idents(expr, out);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_idents(lhs, out);
            collect_idents(rhs, out);
        }
        Expr::Ternary { cond, t, f } => {
            collect_idents(cond, out);
            collect_idents(t, out);
            collect_idents(f, out);
        }
    }
}

struct ComboResolver<'a> {
    funcs: &'a BTreeMap<String, CompiledFunction>,
    globals: &'a Env,
}

impl<'a> CallResolver for ComboResolver<'a> {
    fn call(&self, name: &str, args: &[Value4], expected_width: Option<u32>) -> Result<Value4> {
        let f = self
            .funcs
            .get(name)
            .ok_or_else(|| Error::Parse(format!("unknown function `{name}`")))?;
        if args.len() != f.args.len() {
            return Err(Error::Parse(format!(
                "function `{name}` arg count mismatch: got {} want {}",
                args.len(),
                f.args.len()
            )));
        }
        let mut env = init_function_env(f, args, self.globals);
        match &f.body {
            CompiledFunctionBody::Expr { expr, .. } => {
                let mut v = eval_ast_with_calls(expr, &env, Some(self), Some(f.ret_width))?;
                v = coerce_to_declinfo(&v, &function_return_decl(f));
                Ok(v)
            }
            CompiledFunctionBody::Casez { selector, arms } => {
                let sel_v = eval_ast_with_calls(selector, &env, Some(self), None)?;
                let sel_bits = sel_v.to_bit_string_msb_first();

                let mut default_arm: Option<&CasezArm> = None;
                for arm in arms {
                    if arm.pat.is_none() {
                        default_arm = Some(arm);
                        continue;
                    }
                    let pat = arm.pat.as_ref().unwrap();
                    if casez_matches(&sel_bits, pat) {
                        let mut v =
                            eval_ast_with_calls(&arm.value, &env, Some(self), Some(f.ret_width))?;
                        v = coerce_to_declinfo(&v, &function_return_decl(f));
                        return Ok(v);
                    }
                }
                if let Some(d) = default_arm {
                    let mut v = eval_ast_with_calls(&d.value, &env, Some(self), Some(f.ret_width))?;
                    v = coerce_to_declinfo(&v, &function_return_decl(f));
                    return Ok(v);
                }

                let w = expected_width.unwrap_or(f.ret_width);
                Ok(x_value(w, Signedness::Unsigned))
            }
            CompiledFunctionBody::Procedure { assigns } => {
                exec_function_procedure(f, &mut env, assigns, self)
            }
        }
    }
}

fn casez_matches(sel_bits_msb: &str, pat: &CasezPattern) -> bool {
    // Align widths by truncating/left-padding sel with Xs (conservative).
    let mut sel: String = sel_bits_msb.to_string();
    if sel.len() > pat.width as usize {
        sel = sel[sel.len() - pat.width as usize..].to_string();
    } else if sel.len() < pat.width as usize {
        let mut pad = "x".repeat(pat.width as usize - sel.len());
        pad.push_str(&sel);
        sel = pad;
    }
    let bits = pat.bits_msb.as_str();
    if bits.len() != pat.width as usize {
        // Accept shorter patterns by left-padding with zeros (generator shouldn't do
        // this).
        if bits.len() > pat.width as usize {
            return false;
        }
        let mut pad = "0".repeat(pat.width as usize - bits.len());
        pad.push_str(bits);
        return casez_matches(
            &sel,
            &CasezPattern {
                width: pat.width,
                bits_msb: pad,
            },
        );
    }

    for (sb, pb) in sel.chars().zip(bits.chars()) {
        // `casez` treats z/? bits in either selector or pattern as wildcards.
        if matches!(sb, '?' | 'z' | 'Z') || matches!(pb, '?' | 'z' | 'Z') {
            continue;
        }
        match pb {
            '0' => {
                if sb != '0' {
                    return false;
                }
            }
            '1' => {
                if sb != '1' {
                    return false;
                }
            }
            'x' | 'X' => {
                if !matches!(sb, 'x' | 'X') {
                    return false;
                }
            }
            other => {
                // Unknown pattern char => no match.
                let _ = other;
                return false;
            }
        }
    }
    true
}
