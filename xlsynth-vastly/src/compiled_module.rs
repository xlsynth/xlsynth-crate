// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::ast::Expr;
use crate::ast_spanned::SpannedExpr;
use crate::packed::rewrite_packed_expr;
use crate::packed::rewrite_packed_lhs;
use crate::packed::rewrite_packed_spanned_expr;
use crate::parser_spanned::parse_expr_spanned;
use crate::sv_ast::Decl;
use crate::sv_ast::FunctionBody;
use crate::sv_ast::FunctionDecl;
use crate::sv_ast::Lhs;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::PortDecl;
use crate::sv_ast::PortDir as SvPortDir;
use crate::sv_ast::Span;
use crate::sv_ast::Stmt;
use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;

pub type State = BTreeMap<String, Value4>;

#[derive(Debug, Clone)]
pub struct DeclInfo {
    pub width: u32,
    pub signedness: Signedness,
    pub packed_dims: Vec<u32>,
    pub unpacked_dims: Vec<u32>,
}

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
    pub rhs_spanned: Option<SpannedExpr>,
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

/// Compiled data-path representation shared by combo and pipeline flows.
#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub module_name: String,
    pub consts: BTreeMap<String, Value4>,
    pub input_ports: Vec<Port>,
    pub output_ports: Vec<Port>,
    pub decls: BTreeMap<String, DeclInfo>,
    pub assigns: Vec<ModuleAssign>,
    pub functions: BTreeMap<String, CompiledFunction>,
}

/// Compiled sequential region (typically one merged `always_ff` block group).
#[derive(Debug, Clone)]
pub struct CompiledSeqBlock {
    pub module_name: String,
    pub clk_name: String,
    pub consts: BTreeMap<String, Value4>,
    pub decls: BTreeMap<String, DeclInfo>,
    pub state_regs: BTreeSet<String>,
    pub body: Stmt,
}

impl CompiledSeqBlock {
    pub fn initial_state_x(&self) -> State {
        let mut s: State = BTreeMap::new();
        for name in &self.state_regs {
            let info = self.decls.get(name).expect("decl checked");
            s.insert(
                name.clone(),
                Value4::new(
                    info.width,
                    info.signedness,
                    vec![LogicBit::X; info.width as usize],
                ),
            );
        }
        s
    }

    pub fn step(&self, inputs: &crate::Env, state: &State) -> Result<State> {
        crate::module_eval::step_module(self, inputs, state)
    }
}

/// Lowers module header ports into runtime port vectors and declaration info.
pub(crate) fn lower_ports(
    ports: &[PortDecl],
) -> (Vec<Port>, Vec<Port>, BTreeMap<String, DeclInfo>) {
    let mut input_ports: Vec<Port> = Vec::new();
    let mut output_ports: Vec<Port> = Vec::new();
    let mut decls: BTreeMap<String, DeclInfo> = BTreeMap::new();

    for p in ports {
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
            PortDir::Input => input_ports.push(port),
            PortDir::Output => output_ports.push(port),
        }
        decls.insert(p.name.clone(), decl_info_from_port_decl(p));
    }

    (input_ports, output_ports, decls)
}

/// Adds `logic` declarations from module items into `decls`.
pub(crate) fn extend_decls_from_items(
    items: &[ModuleItem],
    decls: &mut BTreeMap<String, DeclInfo>,
    reject_duplicates: bool,
) -> Result<()> {
    for it in items {
        if let ModuleItem::Decl { decl: d, .. } = it {
            if reject_duplicates && decls.contains_key(&d.name) {
                return Err(Error::Parse(format!("duplicate decl `{}`", d.name)));
            }
            decls.insert(d.name.clone(), decl_info_from_decl(d));
        }
    }
    Ok(())
}

/// Lowers an assign item using packed rewrites and span-preserving parsed AST.
pub(crate) fn lower_assign(
    lhs: &Lhs,
    rhs: &Expr,
    rhs_spanned: &SpannedExpr,
    rhs_span: Span,
    decls: &BTreeMap<String, DeclInfo>,
    preserve_spans: bool,
) -> Result<ModuleAssign> {
    let lhs = rewrite_packed_lhs(lhs.clone(), decls)?;
    let rhs = rewrite_packed_expr(rhs.clone(), decls)?;
    let rhs_spanned = if preserve_spans {
        Some(rewrite_packed_spanned_expr(rhs_spanned.clone(), decls)?)
    } else {
        None
    };
    Ok(ModuleAssign {
        lhs,
        rhs,
        rhs_span,
        rhs_spanned,
    })
}

/// Lowers a function declaration into compiled function form.
pub(crate) fn lower_function(
    parse_src: &str,
    f: &FunctionDecl,
    decls: &BTreeMap<String, DeclInfo>,
    preserve_spans: bool,
) -> Result<CompiledFunction> {
    let mut fn_decls = decls.clone();
    for arg in &f.args {
        fn_decls.insert(arg.name.clone(), decl_info_from_decl(arg));
    }
    for local in &f.locals {
        fn_decls.insert(local.name.clone(), decl_info_from_decl(local));
    }

    let args: Vec<FunctionVar> = f.args.iter().map(lower_decl_to_function_var).collect();
    let locals: BTreeMap<String, DeclInfo> = f
        .locals
        .iter()
        .map(|d| (d.name.clone(), decl_info_from_decl(d)))
        .collect();

    let body = match &f.body {
        FunctionBody::UniqueCasez { selector, arms, .. } => {
            let selector = rewrite_packed_expr(selector.clone(), &fn_decls)?;
            let mut out_arms: Vec<CasezArm> = Vec::new();
            for a in arms {
                let value = rewrite_packed_expr(a.value.clone(), &fn_decls)?;
                let pat = a.pat.as_ref().map(|p| CasezPattern {
                    width: p.width,
                    bits_msb: p.bits_msb.clone(),
                });
                out_arms.push(CasezArm { pat, value });
            }
            CompiledFunctionBody::Casez {
                selector,
                arms: out_arms,
            }
        }
        FunctionBody::Assign { value, value_span } => {
            let expr = rewrite_packed_expr(value.clone(), &fn_decls)?;
            let expr_spanned = if preserve_spans {
                Some(rewrite_packed_spanned_expr(
                    parse_expr_spanned_in_span(parse_src, *value_span)?,
                    &fn_decls,
                )?)
            } else {
                None
            };
            CompiledFunctionBody::Expr { expr, expr_spanned }
        }
        FunctionBody::Procedure { assigns } => {
            let mut out_assigns: Vec<FunctionAssign> = Vec::with_capacity(assigns.len());
            for a in assigns {
                let expr = rewrite_packed_expr(a.value.clone(), &fn_decls)?;
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

    Ok(CompiledFunction {
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
    })
}

fn parse_expr_spanned_in_span(parse_src: &str, span: Span) -> Result<SpannedExpr> {
    let expr_src = parse_src[span.start..span.end].trim();
    let mut spanned = parse_expr_spanned(expr_src)?;
    spanned.shift_spans(span.start);
    Ok(spanned)
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

fn decl_info_from_port_decl(p: &PortDecl) -> DeclInfo {
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
