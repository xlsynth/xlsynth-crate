// SPDX-License-Identifier: Apache-2.0

//! Diagnostics for valid BX source that may merit a change.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::ast::{Block, Expr, Module, SourceNode, SourcePosition, Stmt};
use crate::bx_import::{LibraryModuleInfo, LinkedSources, UnlinkedBlockScope, library_item};

/// A BX warning category, also used in source-level `#[allow(...)]` attributes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum BxWarningKind {
    /// A concrete block contains only computations that belong in a function.
    CombinationalBlock,
    /// A block in a module marked `#![require_block_timing]` has no output
    /// annotation.
    MissingBlockTiming,
    /// A struct update supplies every member explicitly.
    UselessStructSplat,
}

impl BxWarningKind {
    /// Stable spelling used in diagnostics and source attributes.
    pub fn name(self) -> &'static str {
        match self {
            Self::CombinationalBlock => "combinational_block",
            Self::MissingBlockTiming => "missing_block_timing",
            Self::UselessStructSplat => "useless_struct_splat",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "combinational_block" => Some(Self::CombinationalBlock),
            "missing_block_timing" => Some(Self::MissingBlockTiming),
            "useless_struct_splat" => Some(Self::UselessStructSplat),
            _ => None,
        }
    }
}

/// A warning at a source declaration or expression. Imported declaration
/// names include their module path. A root parsed from text has no filename
/// until its caller supplies one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BxWarning {
    pub kind: BxWarningKind,
    /// The declaration's source name, qualified for imported BX modules.
    pub declaration: String,
    pub file: Option<PathBuf>,
    pub position: Option<SourcePosition>,
}

impl BxWarning {
    /// Locate a warning in its root or imported source file.
    fn at(
        kind: BxWarningKind,
        owner: &str,
        position: Option<SourcePosition>,
        library: Option<&LibraryModuleInfo>,
    ) -> Self {
        let declaration = match library_item(owner) {
            Some((module, name)) => format!("{}::{name}", module.replace('.', "::")),
            None => owner.to_owned(),
        };
        Self {
            kind,
            declaration,
            file: library.map(|source| source.path.clone()),
            position,
        }
    }

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
            BxWarningKind::MissingBlockTiming => write!(
                f,
                "block '{}' requires an output timing annotation",
                warning.declaration
            ),
            BxWarningKind::UselessStructSplat => f.write_str(
                "all struct fields are supplied explicitly. Remove the trailing struct splat.",
            ),
        }
    }
}

/// Accumulate facts from checked bodies by declaration, including all uses of
/// a callback template. Report warnings in source order when checking finishes.
#[derive(Default)]
pub(crate) struct WarningCollector<'m> {
    hardware: BTreeMap<&'m str, bool>,
    /// A source occurrence can be checked by multiple specializations or
    /// iterations.
    splats: BTreeMap<&'m str, BTreeSet<Option<SourcePosition>>>,
}

impl<'m> WarningCollector<'m> {
    pub(crate) fn record_useless_splat(
        &mut self,
        owner: &'m str,
        position: Option<SourcePosition>,
    ) {
        self.splats.entry(owner).or_default().insert(position);
    }

    pub(crate) fn record_block(&mut self, name: &'m str, requires_hardware: bool) {
        *self.hardware.entry(name).or_default() |= requires_hardware;
    }

    /// Include warnings from unused imports and uninstantiated block
    /// templates, which have a declaration but no checked body.
    pub(crate) fn finish(
        self,
        root: &Module,
        linked: Option<&LinkedSources>,
        positions: &BTreeMap<&str, usize>,
        is_block: impl Fn(&str) -> bool,
    ) -> Vec<BxWarning> {
        let mut warnings = Vec::new();
        let mut local_scope = None;
        let module = linked.map_or(root, |sources| &sources.module);
        let library_for = |owner: &str| {
            library_item(owner).and_then(|(name, _)| linked?.library_modules.get(name))
        };
        for block in &module.blocks {
            let owner = block.name.as_str();
            let library = library_for(owner);
            let require_timing =
                library.map_or(root.require_block_timing, |m| m.require_block_timing);
            let kind = BxWarningKind::MissingBlockTiming;
            if require_timing
                && block.output_cycles.is_none()
                && !block.allowed_warnings.contains(&kind)
            {
                warnings.push((
                    owner,
                    BxWarning::at(kind, owner, block.name_position, library),
                ));
            }

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
            warnings.push((
                owner,
                BxWarning::at(kind, owner, block.name_position, library),
            ));
        }
        if !self.splats.is_empty() {
            let kind = BxWarningKind::UselessStructSplat;
            let allowed: BTreeSet<_> = module
                .blocks
                .iter()
                .map(|b| (&b.name, &b.allowed_warnings))
                .chain(&module.declaration_allowances)
                .filter_map(|(name, allowances)| {
                    allowances.contains(&kind).then_some(name.as_str())
                })
                .collect();
            for (owner, sites) in self.splats {
                if !allowed.contains(owner) {
                    let library = library_for(owner);
                    warnings.extend(
                        sites
                            .into_iter()
                            .map(|site| (owner, BxWarning::at(kind, owner, site, library))),
                    );
                }
            }
            warnings.sort_by_key(|(owner, warning)| {
                (
                    positions.get(owner).copied().unwrap_or(usize::MAX),
                    warning.position,
                )
            });
        }
        warnings.into_iter().map(|(_, warning)| warning).collect()
    }
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
/// writing the same body as a function. A callable or value formal shadows
/// any block with the same name.
fn template_hardware(block: &Block, is_block: &impl Fn(&str) -> bool) -> TemplateHardware {
    let formals: BTreeSet<_> = block
        .callback_params
        .iter()
        .map(|p| p.name.as_str())
        .chain(block.value_parametrics.iter().map(|p| p.name.as_str()))
        .collect();
    let mut binders: BTreeSet<_> = block.params.iter().map(|p| p.name.clone()).collect();
    binders.insert(block.clock.clone());
    binders.extend(block.reset.iter().cloned());
    let mut callees = BTreeSet::new();
    let mut pending = vec![SourceNode::Expression(&block.output)];
    pending.extend(block.statements.iter().map(SourceNode::Statement));
    while let Some(node) = pending.pop() {
        match node {
            SourceNode::Statement(
                Stmt::Reg { .. } | Stmt::Inst { .. } | Stmt::Write { .. } | Stmt::Connect { .. },
            ) => return TemplateHardware::Known(true),
            SourceNode::Statement(Stmt::Let { name, .. } | Stmt::Wire { name, .. }) => {
                binders.insert(name.clone());
            }
            SourceNode::Statement(Stmt::LetPattern(binding)) => {
                binding.pattern.visit_names(&mut |name| {
                    binders.insert(name.to_owned());
                });
            }
            SourceNode::Expression(
                Expr::Call { name, .. }
                | Expr::MapBeforeUse { name, .. }
                | Expr::ParametricCall { name, .. },
            ) => {
                if !formals.contains(name.as_str()) && is_block(name) {
                    callees.insert(name.as_str());
                }
                if let SourceNode::Expression(Expr::MapBeforeUse { args, .. }) = node {
                    // Only the ordinary arguments can belong to a child call.
                    pending.extend(args.iter().map(SourceNode::Expression));
                    continue;
                }
            }
            SourceNode::Expression(Expr::UnrollFor(fold)) => {
                binders.insert(fold.index.clone());
                binders.insert(fold.accumulator.clone());
            }
            SourceNode::Expression(Expr::For(fold)) => {
                for pattern in [&fold.element, &fold.accumulator] {
                    pattern.visit_names(&mut |name| {
                        binders.insert(name.to_owned());
                    });
                }
            }
            SourceNode::Expression(Expr::Map(map)) => {
                binders.insert(map.parameter.clone());
            }
            SourceNode::Expression(Expr::Scope { bindings, .. }) => {
                for binding in bindings {
                    binding.pattern.visit_names(&mut |name| {
                        binders.insert(name.to_owned());
                    });
                }
            }
            _ => { /* These nodes do not declare binders or name a potential child. */ }
        }
        node.push_children(&mut pending);
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
