// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::Expr;
use crate::ast_spanned::SpannedExpr;
use crate::module_compile::DeclInfo;
use crate::packed::rewrite_packed_expr;
use crate::packed::rewrite_packed_lhs;
use crate::packed::rewrite_packed_spanned_expr;
use crate::parser::parse_expr;
use crate::parser_spanned::parse_expr_spanned;
use crate::sv_ast::Decl;
use crate::sv_ast::FunctionBody;
use crate::sv_ast::Lhs;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::ParsedModule;
use crate::sv_ast::PortDir as SvPortDir;
use crate::sv_ast::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortDir {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub dir: PortDir,
    pub name: String,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct ModuleAssign {
    pub lhs: Lhs,
    pub rhs: Expr,
    pub rhs_span: Span,
    pub rhs_spanned: SpannedExpr,
}

impl ModuleAssign {
    pub fn lhs_base(&self) -> &str {
        match &self.lhs {
            Lhs::Ident(base) => base,
            Lhs::Index { base, .. } => base,
            Lhs::PackedIndex { base, .. } => base,
            Lhs::Slice { base, .. } => base,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CasezPattern {
    pub width: u32,
    /// MSB-first chars, with '?' and 'z' as don't-care.
    pub bits_msb: String,
}

#[derive(Debug, Clone)]
pub struct CasezArm {
    pub pat: Option<CasezPattern>, // None => default
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct FunctionVar {
    pub name: String,
    pub width: u32,
    pub signedness: Signedness,
}

#[derive(Debug, Clone)]
pub struct FunctionAssign {
    pub lhs: String,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub name: String,
    pub ret_width: u32,
    pub ret_signedness: Signedness,
    pub args: Vec<FunctionVar>,
    pub locals: BTreeMap<String, DeclInfo>,
    pub body: CompiledFunctionBody,
}

#[derive(Debug, Clone)]
pub enum CompiledFunctionBody {
    Casez {
        selector: Expr,
        arms: Vec<CasezArm>,
    },
    Expr {
        expr: Expr,
        expr_spanned: Option<SpannedExpr>,
    },
    Procedure {
        assigns: Vec<FunctionAssign>,
    },
}

#[derive(Debug, Clone)]
pub struct CompiledComboModule {
    pub module_name: String,
    pub consts: BTreeMap<String, Value4>,
    pub input_ports: Vec<Port>,
    pub output_ports: Vec<Port>,
    pub decls: BTreeMap<String, DeclInfo>,
    pub assigns: Vec<ModuleAssign>,
    pub functions: BTreeMap<String, CompiledFunction>,
}

pub fn compile_combo_module(src: &str) -> Result<CompiledComboModule> {
    let parse_src = src;
    let parsed: ParsedModule = crate::sv_parser::parse_combo_module(parse_src)?;
    let items = crate::generate_constructs::elaborate_combo_items(
        parse_src,
        &parsed.params,
        &parsed.items,
    )?;

    let module_name = parsed.name;

    let mut input_ports: Vec<Port> = Vec::new();
    let mut output_ports: Vec<Port> = Vec::new();
    let mut decls: BTreeMap<String, DeclInfo> = BTreeMap::new();

    for p in &parsed.ports {
        let dir = match p.dir {
            SvPortDir::Input => PortDir::Input,
            SvPortDir::Output => PortDir::Output,
        };
        let dir_for_vecs = dir.clone();
        let port = Port {
            dir,
            name: p.name.clone(),
            width: p.width,
        };
        match dir_for_vecs {
            PortDir::Input => input_ports.push(port),
            PortDir::Output => output_ports.push(port),
        }
        decls.insert(p.name.clone(), decl_info_from_port_decl(p));
    }

    let mut functions: BTreeMap<String, CompiledFunction> = BTreeMap::new();
    let mut assigns: Vec<ModuleAssign> = Vec::new();

    for it in &items {
        match it {
            ModuleItem::Decl { decl: d, .. } => {
                decls.insert(d.name.clone(), decl_info_from_decl(d));
            }
            ModuleItem::Assign { .. } | ModuleItem::Function { .. } => {}
            ModuleItem::AlwaysFf { .. } => {
                return Err(crate::Error::Parse(
                    "always_ff is not supported in combo modules".to_string(),
                ));
            }
            ModuleItem::GenerateFor { .. } | ModuleItem::GenerateIf { .. } => {
                unreachable!("combo items should be elaborated")
            }
        }
    }

    for it in &items {
        match it {
            ModuleItem::Decl { .. } => {}
            ModuleItem::Assign {
                lhs, rhs, rhs_text, ..
            } => {
                let rhs_src = rhs_text
                    .as_deref()
                    .unwrap_or_else(|| parse_src[rhs.start..rhs.end].trim());
                let lhs = rewrite_packed_lhs(lhs.clone(), &decls)?;
                let rhs_expr = parse_expr(rhs_src)?;
                let rhs_expr = rewrite_packed_expr(rhs_expr, &decls)?;
                let mut rhs_spanned = parse_expr_spanned(rhs_src)?;
                rhs_spanned.shift_spans(rhs.start);
                rhs_spanned = rewrite_packed_spanned_expr(rhs_spanned, &decls)?;
                assigns.push(ModuleAssign {
                    lhs,
                    rhs: rhs_expr,
                    rhs_span: *rhs,
                    rhs_spanned,
                });
            }
            ModuleItem::Function { func: f, .. } => {
                let mut fn_decls = decls.clone();
                for arg in &f.args {
                    fn_decls.insert(arg.name.clone(), decl_info_from_decl(arg));
                }
                for local in &f.locals {
                    fn_decls.insert(local.name.clone(), decl_info_from_decl(local));
                }
                let args: Vec<FunctionVar> =
                    f.args.iter().map(lower_decl_to_function_var).collect();
                let locals: BTreeMap<String, DeclInfo> = f
                    .locals
                    .iter()
                    .map(|d| (d.name.clone(), decl_info_from_decl(d)))
                    .collect();

                let body = match &f.body {
                    FunctionBody::UniqueCasez { selector, arms, .. } => {
                        let selector_src = parse_src[selector.start..selector.end].trim();
                        let selector_expr =
                            rewrite_packed_expr(parse_expr(selector_src)?, &fn_decls);
                        let selector_expr = selector_expr?;

                        let mut out_arms: Vec<CasezArm> = Vec::new();
                        for a in arms {
                            let value_src = parse_src[a.value.start..a.value.end].trim();
                            let value_expr = rewrite_packed_expr(parse_expr(value_src)?, &fn_decls);
                            let value_expr = value_expr?;
                            let pat = a.pat.as_ref().map(|p| CasezPattern {
                                width: p.width,
                                bits_msb: p.bits_msb.clone(),
                            });
                            out_arms.push(CasezArm {
                                pat,
                                value: value_expr,
                            });
                        }
                        CompiledFunctionBody::Casez {
                            selector: selector_expr,
                            arms: out_arms,
                        }
                    }
                    FunctionBody::Assign { value } => {
                        let value_src = parse_src[value.start..value.end].trim();
                        let expr = rewrite_packed_expr(parse_expr(value_src)?, &fn_decls);
                        let expr = expr?;
                        let mut expr_spanned = parse_expr_spanned(value_src)?;
                        expr_spanned.shift_spans(value.start);
                        expr_spanned = rewrite_packed_spanned_expr(expr_spanned, &fn_decls)?;
                        CompiledFunctionBody::Expr {
                            expr,
                            expr_spanned: Some(expr_spanned),
                        }
                    }
                    FunctionBody::Procedure { assigns } => {
                        let mut out_assigns = Vec::with_capacity(assigns.len());
                        for a in assigns {
                            let value_src = parse_src[a.value.start..a.value.end].trim();
                            let expr = rewrite_packed_expr(parse_expr(value_src)?, &fn_decls);
                            let expr = expr?;
                            out_assigns.push(FunctionAssign {
                                lhs: a.lhs.clone(),
                                expr,
                            });
                        }
                        CompiledFunctionBody::Procedure {
                            assigns: out_assigns,
                        }
                    }
                };

                functions.insert(
                    f.name.clone(),
                    CompiledFunction {
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
            }
            ModuleItem::AlwaysFf { .. } => {
                return Err(crate::Error::Parse(
                    "always_ff is not supported in combo modules".to_string(),
                ));
            }
            ModuleItem::GenerateFor { .. } | ModuleItem::GenerateIf { .. } => {
                unreachable!("combo items should be elaborated")
            }
        }
    }

    Ok(CompiledComboModule {
        module_name,
        consts: parsed.params.clone(),
        input_ports,
        output_ports,
        decls,
        assigns,
        functions,
    })
}

fn decl_info_from_decl(d: &Decl) -> DeclInfo {
    DeclInfo {
        width: d.width,
        signedness: if d.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
        packed_dims: d.packed_dims.clone(),
        unpacked_dims: d.unpacked_dims.clone(),
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
        unpacked_dims: p.unpacked_dims.clone(),
    }
}

fn lower_decl_to_function_var(d: &Decl) -> FunctionVar {
    FunctionVar {
        name: d.name.clone(),
        width: d.width,
        signedness: if d.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
    }
}
