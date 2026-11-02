// SPDX-License-Identifier: Apache-2.0

//! Elaborates combinational instances into ordinary, typed continuous
//! assignments.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::Expr;
use crate::ast_spanned::{SpannedExpr, SpannedExprKind};
use crate::generate_constructs::elaborate_pipeline_items;
use crate::sv_ast::{
    Decl, FunctionBody, FunctionDecl, Lhs, ModuleItem, ParsedModule, PortConnection, PortDir,
};
use crate::{Error, Result};

/// Expands the selected top's direct instances before packed lowering and
/// scheduling.
pub(crate) fn elaborate_top(
    src: &str,
    mut top: ParsedModule,
    modules: &BTreeMap<String, ParsedModule>,
) -> Result<ParsedModule> {
    let items = elaborate_pipeline_items(src, &top.params, &top.items)?;
    let bindings = parent_bindings(&top, &items);
    let mut instances = BTreeSet::new();
    top.items.clear();
    for item in items {
        let ModuleItem::Instance {
            module,
            name,
            mut connections,
            ..
        } = item
        else {
            top.items.push(item);
            continue;
        };
        if bindings.contains_key(&name) || !instances.insert(name.clone()) {
            return Err(Error::Parse(format!("duplicate instance `{name}`")));
        }
        let child = modules.get(&module).ok_or_else(|| {
            Error::Parse(format!(
                "unknown child module `{module}` in instance `{name}`"
            ))
        })?;
        let scope = InstanceScope {
            instance: &name,
            locals: BTreeSet::new(),
        };
        for port in &child.ports {
            let formal = scope.qualified(&port.name);
            let connection = connections
                .remove(&port.name)
                .ok_or_else(|| Error::Parse(format!("missing port connection `{formal}`")))?;
            let span = connection.spanned.span;
            top.items.push(ModuleItem::Decl {
                decl: Decl {
                    name: formal.clone(),
                    width: port.width,
                    signed: port.signed,
                    packed_dims: port.packed_dims.clone(),
                    unpacked_dims: port.unpacked_dims.clone(),
                },
                span,
            });
            top.items
                .push(connect_port(port.dir, formal, connection, &bindings)?);
        }
        if let Some(port) = connections.keys().next() {
            return Err(Error::Parse(format!(
                "unknown port `{port}` on instance `{name}` of `{module}`"
            )));
        }
        for (param, value) in &child.params {
            top.params.insert(scope.qualified(param), value.clone());
        }
        for mut item in elaborate_pipeline_items(src, &child.params, &child.items)? {
            match &mut item {
                ModuleItem::Decl { decl, .. } => decl.name = scope.qualified(&decl.name),
                ModuleItem::Assign {
                    lhs,
                    rhs,
                    rhs_spanned,
                    ..
                } => {
                    scope.qualify_lhs(lhs);
                    scope.qualify_expr(rhs);
                    scope.qualify_spanned(rhs_spanned);
                }
                ModuleItem::Function { func, .. } => scope.qualify_function(func),
                ModuleItem::AlwaysFf { .. } | ModuleItem::Instance { .. } => {
                    return Err(Error::Parse(format!(
                        "instance `{name}` of `{module}` must be a combinational leaf module"
                    )));
                }
                ModuleItem::GenerateIf { .. } | ModuleItem::GenerateFor { .. } => {
                    unreachable!("generate items were elaborated");
                }
            }
            top.items.push(item);
        }
    }
    Ok(top)
}

/// Names visible to a port connection in the enclosing module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Binding {
    ReadOnly,
    Writable,
    Function,
}

/// Indexes declarations before binding instances, regardless of source order.
fn parent_bindings(top: &ParsedModule, items: &[ModuleItem]) -> BTreeMap<String, Binding> {
    let mut bindings: BTreeMap<_, _> = top
        .params
        .keys()
        .map(|p| (p.clone(), Binding::ReadOnly))
        .collect();
    for port in &top.ports {
        let binding = match port.dir {
            PortDir::Input => Binding::ReadOnly,
            PortDir::Output => Binding::Writable,
        };
        bindings.insert(port.name.clone(), binding);
    }
    for item in items {
        match item {
            ModuleItem::Decl { decl, .. } => bindings.insert(decl.name.clone(), Binding::Writable),
            ModuleItem::Function { func, .. } => {
                bindings.insert(func.name.clone(), Binding::Function)
            }
            _ => continue, // Only declarations introduce names for connection expressions.
        };
    }
    bindings
}

/// Models the port boundary as an assignment with the destination's declared
/// type.
fn connect_port(
    dir: PortDir,
    formal: String,
    connection: PortConnection,
    bindings: &BTreeMap<String, Binding>,
) -> Result<ModuleItem> {
    let PortConnection { expr, spanned } = connection;
    let span = spanned.span;
    let (lhs, rhs, rhs_spanned) = match dir {
        PortDir::Input => {
            let mut reads = BTreeSet::new();
            let mut calls = BTreeSet::new();
            expr.collect_refs(&mut reads, &mut calls);
            for name in reads {
                if !matches!(
                    bindings.get(&name),
                    Some(Binding::ReadOnly | Binding::Writable)
                ) {
                    return Err(Error::Parse(format!(
                        "input `{formal}` references unknown signal or parameter `{name}`"
                    )));
                }
            }
            for name in calls {
                if bindings.get(&name) != Some(&Binding::Function) {
                    return Err(Error::Parse(format!(
                        "input `{formal}` calls unknown function `{name}`"
                    )));
                }
            }
            (Lhs::Ident(formal), expr, spanned)
        }
        PortDir::Output => {
            let Expr::Ident(actual) = expr else {
                return Err(Error::Parse(format!(
                    "output `{formal}` must connect to a signal name"
                )));
            };
            if bindings.get(&actual) != Some(&Binding::Writable) {
                return Err(Error::Parse(format!(
                    "output `{formal}` requires a declared writable signal, got `{actual}`"
                )));
            }
            (
                Lhs::Ident(actual),
                Expr::Ident(formal.clone()),
                SpannedExpr {
                    span,
                    kind: SpannedExprKind::Ident(formal),
                },
            )
        }
    };
    Ok(ModuleItem::Assign {
        lhs,
        rhs,
        rhs_spanned,
        rhs_span: span,
        span,
    })
}

/// Qualifies module names by instance path while preserving a helper's lexical
/// bindings.
struct InstanceScope<'a> {
    instance: &'a str,
    locals: BTreeSet<String>,
}

impl InstanceScope<'_> {
    fn qualified(&self, name: &str) -> String {
        format!("{}.{name}", self.instance)
    }

    fn qualify_ident(&self, name: &mut String) {
        if !self.locals.contains(name) {
            *name = self.qualified(name);
        }
    }

    fn qualify_call(&self, name: &mut String) {
        if !name.starts_with('$') {
            *name = self.qualified(name);
        }
    }

    fn qualify_expr(&self, expr: &mut Expr) {
        expr.visit_mut(&mut |node| match node {
            Expr::Ident(name) => self.qualify_ident(name),
            Expr::Call { name, .. } => self.qualify_call(name),
            _ => { /* Only names change during elaboration. */ }
        });
    }

    fn qualify_spanned(&self, expr: &mut SpannedExpr) {
        expr.visit_mut(&mut |node| match &mut node.kind {
            SpannedExprKind::Ident(name) => self.qualify_ident(name),
            SpannedExprKind::Call { name, .. } => self.qualify_call(name),
            _ => { /* Preserve the original expression and source span. */ }
        });
    }

    fn qualify_lhs(&self, lhs: &mut Lhs) {
        let base = match lhs {
            Lhs::Ident(base) => base,
            Lhs::Index { base, index } => {
                self.qualify_expr(index);
                base
            }
            Lhs::PackedIndex { base, indices } => {
                for index in indices {
                    self.qualify_expr(index);
                }
                base
            }
            Lhs::Slice { base, msb, lsb } => {
                self.qualify_expr(msb);
                self.qualify_expr(lsb);
                base
            }
        };
        *base = self.qualified(base);
    }

    /// Binds helper calls and free variables to this instance, including the
    /// return variable.
    fn qualify_function(&self, func: &mut FunctionDecl) {
        let body_scope = InstanceScope {
            instance: self.instance,
            locals: func
                .args
                .iter()
                .chain(&func.locals)
                .map(|d| d.name.clone())
                .collect(),
        };
        func.name = self.qualified(&func.name);
        match &mut func.body {
            FunctionBody::Assign {
                value,
                value_spanned,
                ..
            } => {
                body_scope.qualify_expr(value);
                body_scope.qualify_spanned(value_spanned);
            }
            FunctionBody::Procedure { assigns } => {
                for assign in assigns {
                    body_scope.qualify_ident(&mut assign.lhs);
                    body_scope.qualify_expr(&mut assign.value);
                }
            }
            FunctionBody::UniqueCasez { selector, arms, .. } => {
                body_scope.qualify_expr(selector);
                for arm in arms {
                    body_scope.qualify_expr(&mut arm.value);
                }
            }
        }
    }
}
