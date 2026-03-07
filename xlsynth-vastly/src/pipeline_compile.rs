// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Value4;
use crate::compiled_module::CompiledFunction;
use crate::compiled_module::ModuleAssign;
use crate::compiled_module::Port;
use crate::packed::rewrite_packed_stmt;
use crate::sim_observer::SimObserver;
use crate::sv_ast::Lhs;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::ParsedModule;
use crate::sv_ast::Span;
use crate::sv_ast::Stmt;

#[derive(Debug, Clone)]
pub struct CompiledPipelineModule {
    pub module_name: String,
    pub clk_name: String,
    pub combo: crate::compiled_module::CompiledModule,
    pub seqs: Vec<crate::compiled_module::CompiledSeqBlock>,
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
    pub fn initial_state_x(&self) -> crate::compiled_module::State {
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
    let parse_src = src;
    let parsed: ParsedModule =
        crate::sv_parser::parse_pipeline_module_with_defines(parse_src, defines)?;
    let items = crate::generate_constructs::elaborate_pipeline_items(
        parse_src,
        &parsed.params,
        &parsed.items,
    )?;

    let module_name = parsed.name.clone();

    let (input_ports_all, output_ports, mut decls) =
        crate::compiled_module::lower_ports(&parsed.ports);

    let mut assigns: Vec<ModuleAssign> = Vec::new();
    let mut functions: BTreeMap<String, CompiledFunction> = BTreeMap::new();
    let mut always_ffs: Vec<(crate::sv_ast::AlwaysFf, Span)> = Vec::new();
    let mut fn_meta: BTreeMap<String, FunctionMeta> = BTreeMap::new();
    let mut observers: Vec<SimObserver> = Vec::new();
    let mut observer_spans: Vec<Span> = Vec::new();

    crate::compiled_module::extend_decls_from_items(
        &items, &mut decls, /* reject_duplicates= */ true,
    )?;

    for it in &items {
        match it {
            ModuleItem::Decl { .. } => {}
            ModuleItem::Assign {
                lhs, rhs, rhs_text, ..
            } => {
                assigns.push(crate::compiled_module::lower_assign(
                    parse_src,
                    lhs,
                    *rhs,
                    rhs_text.as_deref(),
                    &decls,
                )?);
            }
            ModuleItem::Function {
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
                functions.insert(
                    f.name.clone(),
                    crate::compiled_module::lower_function(parse_src, f, &decls)?,
                );

                let mut arms_meta: Vec<FunctionArmMeta> = Vec::new();
                let mut assign_expr_span: Option<Span> = None;
                let mut scaffold_spans: Vec<Span> = Vec::new();
                match &f.body {
                    crate::sv_ast::FunctionBody::UniqueCasez {
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
                    crate::sv_ast::FunctionBody::Assign { value } => {
                        scaffold_spans.push(*begin_span);
                        scaffold_spans.push(*end_span);
                        assign_expr_span = Some(*value);
                    }
                    crate::sv_ast::FunctionBody::Procedure { .. } => {
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
            ModuleItem::AlwaysFf {
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
            ModuleItem::GenerateFor { .. } | ModuleItem::GenerateIf { .. } => {
                unreachable!("pipeline items should be elaborated")
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

    let combo = crate::compiled_module::CompiledModule {
        module_name: module_name.clone(),
        consts: parsed.params.clone(),
        input_ports,
        output_ports,
        decls: decls.clone(),
        assigns,
        functions,
    };

    let seq_spans: Vec<Span> = stateful.iter().map(|(_, span)| *span).collect();
    let mut groups: Vec<(BTreeSet<String>, Vec<Stmt>)> = Vec::new();
    for (af, _af_span) in stateful {
        let mut regs = BTreeSet::new();
        collect_state_regs(&af.body, &mut regs);
        let mut hits = Vec::new();
        for (idx, (group_regs, _)) in groups.iter().enumerate() {
            if !group_regs.is_disjoint(&regs) {
                hits.push(idx);
            }
        }

        if hits.is_empty() {
            groups.push((regs, vec![af.body]));
            continue;
        }

        let first = hits[0];
        groups[first].0.extend(regs);
        groups[first].1.push(af.body);
        for idx in hits.into_iter().skip(1).rev() {
            let (other_regs, other_bodies) = groups.remove(idx);
            groups[first].0.extend(other_regs);
            groups[first].1.extend(other_bodies);
        }
    }

    let mut seqs: Vec<crate::compiled_module::CompiledSeqBlock> = Vec::with_capacity(groups.len());
    for (state_regs, bodies) in groups {
        for r in &state_regs {
            if !decls.contains_key(r) {
                return Err(Error::Parse(format!(
                    "state reg `{}` must have a `logic` declaration in v1",
                    r
                )));
            }
        }
        seqs.push(crate::compiled_module::CompiledSeqBlock {
            module_name: module_name.clone(),
            clk_name: clk_name.clone(),
            consts: parsed.params.clone(),
            decls: decls.clone(),
            state_regs,
            body: Stmt::Begin(bodies),
        });
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
