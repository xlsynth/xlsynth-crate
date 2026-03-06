// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::combo_compile::CasezArm;
use crate::combo_compile::CasezPattern;
use crate::combo_compile::ComboAssign;
use crate::combo_compile::ComboFunction;
use crate::combo_compile::ComboFunctionImpl;
use crate::combo_compile::CompiledComboModule;
use crate::combo_compile::FunctionAssign;
use crate::combo_compile::FunctionVar;
use crate::combo_compile::Port;
use crate::combo_compile::PortDir;
use crate::module_compile::CompiledModule;
use crate::module_compile::DeclInfo;
use crate::packed::rewrite_packed_expr;
use crate::packed::rewrite_packed_spanned_expr;
use crate::packed::rewrite_packed_stmt;
use crate::sim_observer::SimObserver;
use crate::sv_ast::Lhs;
use crate::sv_ast::PipelineItem;
use crate::sv_ast::PipelineModule;
use crate::sv_ast::PortDir as SvPortDir;
use crate::sv_ast::Span;
use crate::sv_ast::Stmt;

#[derive(Debug, Clone)]
pub struct CompiledPipelineModule {
    pub module_name: String,
    pub clk_name: String,
    pub combo: CompiledComboModule,
    pub seqs: Vec<CompiledModule>,
    pub seq_spans: Vec<Span>,
    pub observers: Vec<SimObserver>,
    pub observer_spans: Vec<Span>,
    pub fn_meta: BTreeMap<String, FunctionMeta>,
}

#[derive(Debug, Clone)]
pub struct FunctionMeta {
    pub def_span: Span,
    pub body_span: Span,
    pub scaffold_spans: Vec<Span>,
    pub arms: Vec<FunctionArmMeta>,
    pub assign_expr_span: Option<Span>,
}

#[derive(Debug, Clone)]
pub struct FunctionArmMeta {
    pub arm_span: Span,
    pub value_span: Span,
}

impl CompiledPipelineModule {
    pub fn initial_state_x(&self) -> crate::module_compile::State {
        let mut out = BTreeMap::new();
        for seq in &self.seqs {
            for name in &seq.state_regs {
                let info = seq.decls.get(name).expect("decl checked");
                out.insert(
                    name.clone(),
                    Value4::new(
                        info.width,
                        info.signedness,
                        vec![LogicBit::X; info.width as usize],
                    ),
                );
            }
        }
        out
    }
}

pub fn compile_pipeline_module(src: &str) -> Result<CompiledPipelineModule> {
    compile_pipeline_module_with_defines(src, &BTreeSet::new())
}

pub fn compile_pipeline_module_with_defines(
    src: &str,
    defines: &BTreeSet<String>,
) -> Result<CompiledPipelineModule> {
    let parsed: PipelineModule =
        crate::sv_parser::parse_pipeline_module_with_defines(src, defines)?;

    let module_name = parsed.name;

    let mut input_ports_all: Vec<Port> = Vec::new();
    let mut output_ports: Vec<Port> = Vec::new();
    let mut decls: BTreeMap<String, DeclInfo> = BTreeMap::new();

    for p in &parsed.ports {
        let dir = match p.dir {
            SvPortDir::Input => PortDir::Input,
            SvPortDir::Output => PortDir::Output,
        };
        let port = Port {
            dir: dir.clone(),
            name: p.name.clone(),
            width: p.width,
        };
        match dir {
            PortDir::Input => input_ports_all.push(port),
            PortDir::Output => output_ports.push(port),
        }
        decls.insert(p.name.clone(), decl_info_from_port_decl(p));
    }

    let mut assigns: Vec<ComboAssign> = Vec::new();
    let mut functions: BTreeMap<String, ComboFunction> = BTreeMap::new();
    let mut always_ffs: Vec<(crate::sv_ast::AlwaysFf, Span)> = Vec::new();
    let mut fn_meta: BTreeMap<String, FunctionMeta> = BTreeMap::new();
    let mut observers: Vec<SimObserver> = Vec::new();
    let mut observer_spans: Vec<Span> = Vec::new();

    for it in &parsed.items {
        match it {
            PipelineItem::Decl { decl: d, .. } => {
                if decls.contains_key(&d.name) {
                    return Err(Error::Parse(format!("duplicate decl `{}`", d.name)));
                }
                decls.insert(d.name.clone(), decl_info_from_decl(d));
            }
            PipelineItem::Assign { .. }
            | PipelineItem::Function { .. }
            | PipelineItem::AlwaysFf { .. } => {}
        }
    }

    for it in &parsed.items {
        match it {
            PipelineItem::Decl { .. } => {}
            PipelineItem::Assign { lhs_ident, rhs, .. } => {
                let rhs_src = src[rhs.start..rhs.end].trim();
                let rhs_expr = rewrite_packed_expr(crate::parser::parse_expr(rhs_src)?, &decls)?;
                let mut rhs_spanned = crate::parser_spanned::parse_expr_spanned(rhs_src)?;
                rhs_spanned.shift_spans(rhs.start);
                rhs_spanned = rewrite_packed_spanned_expr(rhs_spanned, &decls)?;
                assigns.push(ComboAssign {
                    lhs: Lhs::Ident(lhs_ident.clone()),
                    rhs: rhs_expr,
                    rhs_span: *rhs,
                    rhs_spanned,
                });
            }
            PipelineItem::Function {
                func: f,
                span,
                body_span,
                begin_span,
                end_span,
            } => {
                if functions.contains_key(&f.name) {
                    return Err(Error::Parse(format!("duplicate function `{}`", f.name)));
                }
                if fn_meta.contains_key(&f.name) {
                    return Err(Error::Parse(format!(
                        "duplicate function meta `{}`",
                        f.name
                    )));
                }
                let mut fn_decls = decls.clone();
                for arg in &f.args {
                    fn_decls.insert(arg.name.clone(), decl_info_from_decl(arg));
                }
                for local in &f.locals {
                    fn_decls.insert(local.name.clone(), decl_info_from_decl(local));
                }
                let args: Vec<FunctionVar> = f
                    .args
                    .iter()
                    .map(|a| FunctionVar {
                        name: a.name.clone(),
                        width: a.width,
                        signedness: if a.signed {
                            Signedness::Signed
                        } else {
                            Signedness::Unsigned
                        },
                    })
                    .collect();
                let locals: BTreeMap<String, DeclInfo> = f
                    .locals
                    .iter()
                    .map(|d| (d.name.clone(), decl_info_from_decl(d)))
                    .collect();

                let body = match &f.body {
                    crate::sv_ast::ComboFunctionBody::UniqueCasez { selector, arms, .. } => {
                        let selector_src = src[selector.start..selector.end].trim();
                        let selector_expr = rewrite_packed_expr(
                            crate::parser::parse_expr(selector_src)?,
                            &fn_decls,
                        )?;

                        let mut out_arms: Vec<CasezArm> = Vec::new();
                        for a in arms {
                            let value_src = src[a.value.start..a.value.end].trim();
                            let value_expr = rewrite_packed_expr(
                                crate::parser::parse_expr(value_src)?,
                                &fn_decls,
                            )?;
                            let pat = a.pat.as_ref().map(|p| CasezPattern {
                                width: p.width,
                                bits_msb: p.bits_msb.clone(),
                            });
                            out_arms.push(CasezArm {
                                pat,
                                value: value_expr,
                            });
                        }
                        ComboFunctionImpl::Casez {
                            selector: selector_expr,
                            arms: out_arms,
                        }
                    }
                    crate::sv_ast::ComboFunctionBody::Assign { value } => {
                        let value_src = src[value.start..value.end].trim();
                        let expr =
                            rewrite_packed_expr(crate::parser::parse_expr(value_src)?, &fn_decls)?;
                        let mut expr_spanned =
                            crate::parser_spanned::parse_expr_spanned(value_src)?;
                        expr_spanned.shift_spans(value.start);
                        ComboFunctionImpl::Expr {
                            expr,
                            expr_spanned: Some(expr_spanned),
                        }
                    }
                    crate::sv_ast::ComboFunctionBody::Procedure { assigns } => {
                        let mut out_assigns: Vec<FunctionAssign> =
                            Vec::with_capacity(assigns.len());
                        for a in assigns {
                            let value_src = src[a.value.start..a.value.end].trim();
                            let expr = rewrite_packed_expr(
                                crate::parser::parse_expr(value_src)?,
                                &fn_decls,
                            )?;
                            out_assigns.push(FunctionAssign {
                                lhs: a.lhs.clone(),
                                expr,
                            });
                        }
                        ComboFunctionImpl::Procedure {
                            assigns: out_assigns,
                        }
                    }
                };
                functions.insert(
                    f.name.clone(),
                    ComboFunction {
                        name: f.name.clone(),
                        ret_width: f.ret_width,
                        ret_signedness: if f.ret_signed {
                            Signedness::Signed
                        } else {
                            Signedness::Unsigned
                        },
                        args,
                        locals,
                        body,
                    },
                );

                let mut arms_meta: Vec<FunctionArmMeta> = Vec::new();
                let mut assign_expr_span: Option<Span> = None;
                let mut scaffold_spans: Vec<Span> = Vec::new();
                match &f.body {
                    crate::sv_ast::ComboFunctionBody::UniqueCasez {
                        casez_span,
                        endcase_span,
                        arms,
                        ..
                    } => {
                        scaffold_spans.push(*begin_span);
                        scaffold_spans.push(*casez_span);
                        scaffold_spans.push(*endcase_span);
                        scaffold_spans.push(*end_span);
                        for a in arms {
                            arms_meta.push(FunctionArmMeta {
                                arm_span: a.arm_span,
                                value_span: a.value,
                            });
                        }
                    }
                    crate::sv_ast::ComboFunctionBody::Assign { value } => {
                        scaffold_spans.push(*begin_span);
                        scaffold_spans.push(*end_span);
                        assign_expr_span = Some(*value);
                    }
                    crate::sv_ast::ComboFunctionBody::Procedure { .. } => {
                        scaffold_spans.push(*begin_span);
                        scaffold_spans.push(*end_span);
                    }
                }
                fn_meta.insert(
                    f.name.clone(),
                    FunctionMeta {
                        def_span: *span,
                        body_span: *body_span,
                        scaffold_spans,
                        arms: arms_meta,
                        assign_expr_span,
                    },
                );
            }
            PipelineItem::AlwaysFf {
                always_ff: af,
                span,
            } => {
                always_ffs.push((
                    crate::sv_ast::AlwaysFf {
                        clk_name: af.clk_name.clone(),
                        body: rewrite_packed_stmt(af.body.clone(), &decls)?,
                    },
                    *span,
                ));
            }
        }
    }

    let clk_name = always_ffs
        .first()
        .map(|(af, _)| af.clk_name.clone())
        .or_else(|| {
            let has_clk = input_ports_all.iter().any(|p| p.name == "clk");
            if has_clk {
                Some("clk".to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Error::Parse(
                "no always_ff clock found and no `clk` input port; pass a module with an input named `clk`".to_string(),
            )
        })?;

    // Partition always_ff blocks into zero or more stateful seq blocks and observer
    // blocks.
    let mut stateful: Vec<(crate::sv_ast::AlwaysFf, Span)> = Vec::new();
    for (af, af_span) in always_ffs {
        if af.clk_name != clk_name {
            return Err(Error::Parse(format!(
                "always_ff clock mismatch: saw `{}` but expected `{}`",
                af.clk_name, clk_name
            )));
        }
        let has_nba = stmt_contains_nba(&af.body);
        let has_disp = stmt_contains_display(&af.body);
        if has_nba {
            if has_disp {
                return Err(Error::Parse(
                    "mixed nba assigns and $display in one always_ff is not supported".to_string(),
                ));
            }
            stateful.push((af, af_span));
        } else if has_disp {
            observers.extend(crate::sim_observer::extract_observers(&clk_name, &af.body)?);
            observer_spans.push(af_span);
        } else {
            // Empty / unsupported: ignore (v1).
        }
    }

    let input_ports: Vec<Port> = input_ports_all
        .into_iter()
        .filter(|p| p.name != clk_name)
        .collect();

    let combo = CompiledComboModule {
        module_name: module_name.clone(),
        consts: parsed.params.clone(),
        input_ports,
        output_ports,
        decls: decls.clone(),
        assigns,
        functions,
    };

    let mut seen_state_regs: BTreeSet<String> = BTreeSet::new();
    let mut seqs: Vec<CompiledModule> = Vec::with_capacity(stateful.len());
    let mut seq_spans: Vec<Span> = Vec::new();
    for (af, af_span) in stateful {
        let mut state_regs: BTreeSet<String> = BTreeSet::new();
        collect_state_regs(&af.body, &mut state_regs);
        for r in &state_regs {
            if !decls.contains_key(r) {
                return Err(Error::Parse(format!(
                    "state reg `{}` must have a `logic` declaration in v1",
                    r
                )));
            }
            if !seen_state_regs.insert(r.clone()) {
                return Err(Error::Parse(format!(
                    "state reg `{}` assigned from multiple always_ff blocks is not supported",
                    r
                )));
            }
        }
        seqs.push(CompiledModule {
            module_name: module_name.clone(),
            clk_name: af.clk_name,
            consts: parsed.params.clone(),
            decls: decls.clone(),
            state_regs,
            body: af.body,
        });
        seq_spans.push(af_span);
    }

    Ok(CompiledPipelineModule {
        module_name,
        clk_name,
        combo,
        seqs,
        seq_spans,
        observers,
        observer_spans,
        fn_meta,
    })
}

fn stmt_contains_nba(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Begin(stmts) => stmts.iter().any(stmt_contains_nba),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_contains_nba(then_branch)
                || else_branch
                    .as_ref()
                    .map(|e| stmt_contains_nba(e))
                    .unwrap_or(false)
        }
        Stmt::NbaAssign { .. } => true,
        Stmt::Display { .. } => false,
        Stmt::Empty => false,
    }
}

fn stmt_contains_display(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Begin(stmts) => stmts.iter().any(stmt_contains_display),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_contains_display(then_branch)
                || else_branch
                    .as_ref()
                    .map(|e| stmt_contains_display(e))
                    .unwrap_or(false)
        }
        Stmt::NbaAssign { .. } => false,
        Stmt::Display { .. } => true,
        Stmt::Empty => false,
    }
}

fn decl_info_from_decl(d: &crate::sv_ast::Decl) -> DeclInfo {
    DeclInfo {
        width: d.width,
        signedness: if d.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
        packed_dims: d.packed_dims.clone(),
    }
}

fn decl_info_from_port_decl(p: &crate::sv_ast::PortDecl) -> DeclInfo {
    DeclInfo {
        width: p.width,
        signedness: if p.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
        packed_dims: p.packed_dims.clone(),
    }
}

fn collect_state_regs(stmt: &Stmt, out: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Begin(stmts) => {
            for s in stmts {
                collect_state_regs(s, out);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_state_regs(then_branch, out);
            if let Some(e) = else_branch {
                collect_state_regs(e, out);
            }
        }
        Stmt::NbaAssign { lhs, .. } => match lhs {
            Lhs::Ident(b) => {
                out.insert(b.clone());
            }
            Lhs::Index { base, .. } => {
                out.insert(base.clone());
            }
            Lhs::PackedIndex { base, .. } => {
                out.insert(base.clone());
            }
            Lhs::Slice { base, .. } => {
                out.insert(base.clone());
            }
        },
        Stmt::Display { .. } => {}
        Stmt::Empty => {}
    }
}
