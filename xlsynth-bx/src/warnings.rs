// SPDX-License-Identifier: Apache-2.0

//! Diagnostics for valid BX source that may merit a change.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::ast::{Block, Expr, ForIterable, Module, SourcePosition, Stmt, Type, WidthSliceType};
use crate::bx_import::{LinkedSources, UnlinkedBlockScope, library_item};

/// A BX warning category, also used in source-level `#[allow(...)]` attributes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum BxWarningKind {
    /// A concrete block contains only computations that belong in a function.
    CombinationalBlock,
}

impl BxWarningKind {
    /// Stable spelling used in diagnostics and source attributes.
    pub fn name(self) -> &'static str {
        match self {
            Self::CombinationalBlock => "combinational_block",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "combinational_block" => Some(Self::CombinationalBlock),
            _ => None,
        }
    }
}

/// A warning on a source declaration. Imported block names include their
/// module path. A root parsed from text has no filename until its caller
/// supplies one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BxWarning {
    pub kind: BxWarningKind,
    /// The declaration's source name, qualified for imported BX modules.
    pub declaration: String,
    pub file: Option<PathBuf>,
    pub position: Option<SourcePosition>,
}

impl BxWarning {
    /// Render a diagnostic using the caller's filename for the root module.
    pub fn display_with_root<'a>(&'a self, root: &'a Path) -> impl fmt::Display + 'a {
        WarningDisplay {
            warning: self,
            file: self.file.as_deref().unwrap_or(root),
        }
    }
}

impl fmt::Display for BxWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display_with_root(Path::new("<input>")).fmt(f)
    }
}

/// A diagnostic paired with its source path for rendering.
struct WarningDisplay<'a> {
    warning: &'a BxWarning,
    file: &'a Path,
}

impl fmt::Display for WarningDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let warning = self.warning;
        write!(f, "{}", self.file.display())?;
        if let Some(pos) = warning.position {
            write!(f, ":{}:{}", pos.line, pos.column)?;
        }
        write!(f, ": warning[{}]: ", warning.kind.name())?;
        match warning.kind {
            BxWarningKind::CombinationalBlock => write!(
                f,
                "block '{}' has no registers or child instances. Consider defining it as a function.",
                warning.declaration
            ),
        }
    }
}

/// Accumulate facts from checked bodies by declaration, including all uses of
/// a callback template. Report warnings in source order when checking finishes.
#[derive(Default)]
pub(crate) struct WarningCollector<'m> {
    hardware: BTreeMap<&'m str, bool>,
}

impl<'m> WarningCollector<'m> {
    pub(crate) fn record_block(&mut self, name: &'m str, requires_hardware: bool) {
        *self.hardware.entry(name).or_default() |= requires_hardware;
    }

    /// Include warnings from unused imports and uninstantiated callable
    /// templates, which have a declaration but no checked body.
    pub(crate) fn finish(
        self,
        module: &Module,
        linked: Option<&LinkedSources>,
        is_block: impl Fn(&str) -> bool,
    ) -> Vec<BxWarning> {
        let mut warnings = Vec::new();
        let mut local_scope = None;
        for block in &module.blocks {
            let kind = BxWarningKind::CombinationalBlock;
            if !block.type_params.is_empty()
                || block.allowed_warnings.contains(&kind)
                || self
                    .hardware
                    .get(block.name.as_str())
                    .copied()
                    .unwrap_or_else(|| match template_hardware(block, &is_block) {
                        TemplateHardware::Known(hardware) => hardware,
                        TemplateHardware::ShadowableCall if linked.is_some() => true,
                        TemplateHardware::ShadowableCall => {
                            // Reuse lexical resolution only if a possible child
                            // has a same-named local. An unchecked, unresolved
                            // body is not grounds for suggesting conversion to
                            // a function.
                            local_scope
                                .get_or_insert_with(|| UnlinkedBlockScope::new(module))
                                .as_ref()
                                .ok()
                                .and_then(|scope| scope.qualify(block).ok())
                                .is_none_or(|b| {
                                    template_hardware(&b, &is_block)
                                        != TemplateHardware::Known(false)
                                })
                        }
                    })
            {
                continue;
            }
            let (name, file) = match library_item(&block.name) {
                Some((module, name)) => (
                    format!("{}::{name}", module.replace('.', "::")),
                    linked.and_then(|l| l.module_paths.get(module)).cloned(),
                ),
                None => (block.name.clone(), None),
            };
            warnings.push(BxWarning {
                kind,
                declaration: name,
                file,
                position: block.name_position,
            });
        }
        warnings
    }
}

/// A source node that may contain a structural declaration or child call.
enum SourceNode<'a> {
    Statement(&'a Stmt),
    Expression(&'a Expr),
    Iterable(&'a ForIterable),
    Annotation(&'a Type),
}

/// A child call in an unchecked body can share its spelling with a local
/// value. Other bodies can be classified directly from their source.
#[derive(PartialEq)]
enum TemplateHardware {
    Known(bool),
    ShadowableCall,
}

/// Inspect the source rather than just emitted instances. A child in a
/// zero-trip unroll, even within an otherwise pure expression, prevents
/// writing the same body as a function. A callable formal names a function.
fn template_hardware(block: &Block, is_block: &impl Fn(&str) -> bool) -> TemplateHardware {
    let formals: BTreeSet<_> = block
        .callback_params
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    let mut binders: BTreeSet<_> = block.params.iter().map(|p| p.name.clone()).collect();
    binders.insert(block.clock.clone());
    binders.extend(block.reset.iter().cloned());
    let mut callees = BTreeSet::new();
    let mut pending = vec![SourceNode::Expression(&block.output)];
    pending.extend(block.statements.iter().map(SourceNode::Statement));
    while let Some(node) = pending.pop() {
        match node {
            SourceNode::Statement(stmt) => match stmt {
                Stmt::Reg { .. }
                | Stmt::Inst { .. }
                | Stmt::Write { .. }
                | Stmt::Connect { .. } => {
                    return TemplateHardware::Known(true);
                }
                Stmt::Let { name, value, ty } => {
                    binders.insert(name.clone());
                    pending.push(SourceNode::Expression(value));
                    pending.extend(ty.iter().map(SourceNode::Annotation));
                }
                Stmt::LetPattern(binding) => {
                    binding.pattern.visit_names(&mut |name| {
                        binders.insert(name.to_owned());
                    });
                    pending.push(SourceNode::Expression(&binding.value));
                    pending.extend(binding.ty.iter().map(SourceNode::Annotation));
                }
                Stmt::If { condition, body } => {
                    pending.push(SourceNode::Expression(condition));
                    pending.extend(body.iter().map(SourceNode::Statement));
                }
            },
            SourceNode::Iterable(ForIterable::Array(expr)) => {
                pending.push(SourceNode::Expression(expr))
            }
            SourceNode::Iterable(ForIterable::Range { start, limit, .. }) => {
                pending.extend([start, limit].map(SourceNode::Expression));
            }
            SourceNode::Annotation(ty) => match ty {
                Type::ParametricBits { signed, width } => {
                    pending.extend([signed.as_ref(), width.as_ref()].map(SourceNode::Expression));
                }
                Type::ParametricArray { element, size } => {
                    pending.push(SourceNode::Annotation(element));
                    pending.push(SourceNode::Expression(size));
                }
                Type::Array { element, .. } => pending.push(SourceNode::Annotation(element)),
                Type::Tuple(fields) => pending.extend(fields.iter().map(SourceNode::Annotation)),
                _ => { /* Concrete scalar and nominal annotations introduce no expressions. */ }
            },
            SourceNode::Expression(expr) => match expr {
                Expr::Call { name, args }
                | Expr::MapBeforeUse { name, args, .. }
                | Expr::ParametricCall { name, args, .. } => {
                    if !formals.contains(name.as_str()) && is_block(name) {
                        callees.insert(name.as_str());
                    }
                    pending.extend(args.iter().map(SourceNode::Expression));
                    if let Expr::ParametricCall { parametrics, .. } = expr {
                        pending.extend(parametrics.iter().map(SourceNode::Expression));
                    }
                }
                Expr::UnrollFor(fold) => {
                    binders.insert(fold.index.clone());
                    binders.insert(fold.accumulator.clone());
                    pending.extend(fold.body.iter().map(SourceNode::Statement));
                    if let Some((element, accumulator)) = &fold.annotation {
                        pending.extend([element, accumulator].map(SourceNode::Annotation));
                    }
                    pending.extend(
                        [&fold.start, &fold.limit, &fold.seed, &fold.result]
                            .map(SourceNode::Expression),
                    );
                }
                Expr::For(fold) => {
                    for pattern in [&fold.element, &fold.accumulator] {
                        pattern.visit_names(&mut |name| {
                            binders.insert(name.to_owned());
                        });
                    }
                    pending.push(SourceNode::Iterable(&fold.iterable));
                    if let Some((element, accumulator)) = &fold.annotation {
                        pending.extend([element, accumulator].map(SourceNode::Annotation));
                    }
                    pending.extend([&fold.seed, &fold.body].map(SourceNode::Expression));
                }
                Expr::Map(map) => {
                    binders.insert(map.parameter.clone());
                    pending.push(SourceNode::Iterable(&map.iterable));
                    pending.push(SourceNode::Expression(&map.body));
                    pending.extend(
                        map.parameter_type
                            .iter()
                            .chain(&map.result_type)
                            .map(SourceNode::Annotation),
                    );
                }
                Expr::NamedMap {
                    iterable,
                    parametrics,
                    ..
                } => {
                    pending.push(SourceNode::Iterable(iterable));
                    pending.extend(parametrics.iter().map(SourceNode::Expression));
                }
                Expr::Scope { bindings, result } => {
                    for binding in bindings {
                        binding.pattern.visit_names(&mut |name| {
                            binders.insert(name.to_owned());
                        });
                    }
                    pending.push(SourceNode::Expression(result));
                    pending.extend(bindings.iter().map(|b| SourceNode::Expression(&b.value)));
                    pending.extend(
                        bindings
                            .iter()
                            .filter_map(|b| b.ty.as_ref())
                            .map(SourceNode::Annotation),
                    );
                }
                Expr::If {
                    condition,
                    then_value,
                    else_value,
                } => {
                    pending.extend(
                        [condition.as_ref(), then_value, else_value].map(SourceNode::Expression),
                    );
                }
                Expr::Match { subject, arms } => {
                    pending.push(SourceNode::Expression(subject));
                    for arm in arms {
                        pending.push(SourceNode::Expression(&arm.value));
                        pending.extend(
                            arm.patterns
                                .iter()
                                .flat_map(|p| p.literals())
                                .map(SourceNode::Expression),
                        );
                    }
                }
                Expr::Array(items) | Expr::Tuple(items) => {
                    pending.extend(items.iter().map(SourceNode::Expression));
                }
                Expr::Struct { fields, .. } | Expr::AppliedStruct { fields, .. } => {
                    pending.extend(
                        fields
                            .iter()
                            .map(|(_, value)| SourceNode::Expression(value)),
                    );
                }
                Expr::Field { value, .. } | Expr::Unary { value, .. } => {
                    pending.push(SourceNode::Expression(value));
                }
                Expr::Cast { value, ty } => {
                    pending.push(SourceNode::Expression(value));
                    pending.push(SourceNode::Annotation(ty));
                }
                Expr::Index { value, index }
                | Expr::Binary {
                    left: value,
                    right: index,
                    ..
                } => {
                    pending.extend([value.as_ref(), index.as_ref()].map(SourceNode::Expression));
                }
                Expr::WidthSlice {
                    value,
                    start,
                    width,
                } => {
                    pending.extend([value.as_ref(), start.as_ref()].map(SourceNode::Expression));
                    match width {
                        WidthSliceType::Bits { signed, bit_count } => pending.extend(
                            [signed.as_ref(), bit_count.as_ref()].map(SourceNode::Expression),
                        ),
                        WidthSliceType::Concrete(ty) => pending.push(SourceNode::Annotation(ty)),
                    }
                }
                Expr::Zero(ty) | Expr::Literal { ty: Some(ty), .. } => {
                    pending.push(SourceNode::Annotation(ty));
                }
                Expr::Name(_) | Expr::Bool(_) | Expr::Literal { ty: None, .. } => {
                    // Leaf values introduce no hardware.
                }
            },
        }
    }
    if callees.is_empty() {
        TemplateHardware::Known(false)
    } else if callees.iter().any(|name| !binders.contains(*name)) {
        TemplateHardware::Known(true)
    } else {
        TemplateHardware::ShadowableCall
    }
}

#[cfg(test)]
mod tests {
    use crate::{ImportOptions, check_module, check_source_with_options, parse_module};
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn structural_children_exempt_their_parent_even_in_a_zero_trip_unroll() -> Result<(), String> {
        let source = r#"
extern block device(clk, x: u8) -> u8;
block pure(clk, x: u8) -> u8 { x + u8:1 }
block composed(clk, x: u8) -> u8 {
    let y = pure(x);
    y
}
block explicit(clk, x: u8) -> u8 {
    inst d: device;
    d.x := x;
    d.out
}
block skipped(clk, x: u8) -> u8 {
    unroll_for! (_, a) in u8:0..u8:0 {
        let (_, y) = (u8:0, a);
        let z = device(y);
        z
    }(x)
}
block nested(clk, x: u8) -> u8 {
    x + unroll_for! (_, a) in u8:0..u8:0 {
        let y = device(a);
        y
    }(x)
}
"#;
        let checked = check_module(&parse_module(source)?)?;
        let names: Vec<_> = checked
            .warnings()
            .iter()
            .map(|w| w.declaration.as_str())
            .collect();
        assert_eq!(names, ["pure"]);
        Ok(())
    }

    #[test]
    fn skipped_local_callees_remain_values_with_and_without_imports_or_specializations()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("empty.bx"), "")?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let declarations = r#"
fn id(x: u8) -> u8 { x }
block child(clk, x: u8) -> u8 { reg q: u8; q := x; q }
block local(clk, x: u8) -> u8 {
    unroll_for! (_, acc) in u8:0..u8:0 {
        let child = acc;
        let y = child(x);
        y
    }(x)
}
block template<F: fn(u8) -> u8>(clk, x: u8) -> u8 {
    unroll_for! (_, child) in u8:0..u8:0 {
        let y = child(F(x));
        y
    }(x)
}
block wrapper<F: fn(u8) -> u8>(clk, x: u8) -> u8 {
    let first = child(F(x));
    unroll_for! (_, child) in u8:0..u8:0 { let y = child(x); y }(first)
}
block port<F: fn(u8) -> u8>(clk, child: u8) -> u8 {
    unroll_for! (_, a) in u8:0..u8:0 { let y = child(F(a)); y }(child)
}
"#;
        let instance =
            "block parent(clk, x: u8) -> u8 { let y = template<id>(x); let z = port<id>(y); z }";
        for import in ["", "use empty;"] {
            for call in ["", instance] {
                let source = format!("{import}\n{declarations}\n{call}");
                let checked = check_source_with_options(&source, &options)?;
                let names: Vec<_> = checked
                    .warnings()
                    .iter()
                    .map(|w| w.declaration.as_str())
                    .collect();
                assert_eq!(names, ["local", "template"], "{source}");
            }
        }
        Ok(())
    }

    #[test]
    fn check_callable_templates_once_in_their_declaring_scope() -> Result<(), String> {
        let source = r#"
fn bump(x: u8) -> u8 { x + u8:1 }
fn square(x: u8) -> u8 { x * x }
block op(clk, x: u8) -> u8 { reg q: u8; q := x; q }
block transform<op: fn(u8) -> u8>(clk, x: u8) -> u8 {
    let y = op(x);
    y
}
block unused<F: fn(u8) -> u8>(clk, x: u8) -> u8 { F(x) }
block compose<F: fn(u8) -> u8>(clk, x: u8) -> u8 {
    let y = op(F(x));
    y
}
block parent(clk, x: u8) -> u8 {
    let a = transform<bump>(x);
    let b = transform<square>(a);
    b
}
"#;
        let checked = check_module(&parse_module(source)?)?;
        let names: Vec<_> = checked
            .warnings()
            .iter()
            .map(|w| w.declaration.as_str())
            .collect();
        assert_eq!(names, ["transform", "unused"]);
        Ok(())
    }
}
