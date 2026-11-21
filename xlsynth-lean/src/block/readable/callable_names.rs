// SPDX-License-Identifier: Apache-2.0

//! Choose Lean names for the selected block and its readable helpers.
//!
//! A DSLX module `a.f` can contain a function `eq_1` even when module `a`
//! also defines a function `f`. Defining `XLS.a.f` in Lean would reserve
//! `XLS.a.f.eq_1`. In that case the implementation lives in a separate
//! namespace and an `export` exposes it as `XLS.a.f`. Lean resolves that
//! alias by name but cannot use it in a `_root_` reference. Generated calls
//! and proofs therefore use the rooted implementation path.

use std::collections::BTreeMap;

use super::*;

/// Stores the Lean paths used to define and reference helpers and the block,
/// along with the public aliases that expose their source names when needed.
pub(super) struct CallableNames<'a> {
    /// Rooted declaration paths keyed by the helper symbols used in the IR.
    pub(super) helpers: BTreeMap<&'a str, String>,
    /// Rooted path for the selected block's abbreviation of the evaluator.
    pub(super) block: String,
    /// Lean commands exposing source names after the backing declarations
    /// exist.
    pub(super) aliases: String,
}

/// How a declaration under a callable's source name could collide in Lean.
#[derive(Clone, Copy)]
enum ChildName {
    /// Even an abbreviation at the parent name would reserve this child.
    Reserved,
    /// Elaborating a proof in the parent's body could create this child.
    AuxiliaryProof,
}

/// Recognize child names used by Lean, including numbered names such as `eq_1`.
fn child_name(name: &str) -> Option<ChildName> {
    let numbered = |prefix| {
        name.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    // Lean 4.34 reserves equation, function case and congruence theorem
    // names even for a simple nonrecursive definition.
    if matches!(
        name,
        "eq_def" | "eq_unfold" | "fun_cases" | "fun_cases_unfolding" | "congr_simp"
    ) || numbered("eq_")
        || numbered("hcongr_")
    {
        Some(ChildName::Reserved)
    } else if numbered("_proof_") {
        Some(ChildName::AuxiliaryProof)
    } else {
        None
    }
}

impl<'a> CallableNames<'a> {
    /// Choose paths after collecting helper, type, and block names.
    ///
    /// A child declaration can appear after its parent in emission order. A
    /// helper with no source declaration keeps its generated helper name.
    pub(super) fn new(
        block: &'a BlockContext<'_>,
        declarations: &[TypeDecl],
        interface: &StatefulInterface,
        bx_namespace: &str,
    ) -> Result<Self> {
        let names = block.helpers.iter().filter_map(|f| {
            let (module, local) = interface.helpers.get(&f.name)?.source_name.as_ref()?;
            Some((module.as_str(), local.as_str()))
        });
        let (block_module, block_name) = declaring_name(bx_namespace, &interface.block_name);
        let mut children = BTreeMap::new();
        for (module, local) in names
            .chain(
                declarations
                    .iter()
                    .map(|d| declaring_name(bx_namespace, d.name())),
            )
            .chain([(block_module, block_name)])
        {
            if let Some(parent) = module.rsplit_once('.')
                && let Some(child) = child_name(local)
            {
                // An export satisfies both cases. A real abbrev would still
                // reserve its own equation names.
                if matches!(child, ChildName::Reserved) {
                    children.insert(parent, child);
                } else {
                    children.entry(parent).or_insert(child);
                }
            }
        }
        let mut aliases = String::new();
        let mut path = |module: &str, local: &str, has_body: bool| -> Result<String> {
            let child = children.get(&(module, local));
            let role = if has_body { "function" } else { "block" };
            let local = source_names::declaration_identifier(module, local, role)?;
            let module = module_identifier(module)?;
            let backing = format!("{module}.«xls:implementation».{local}");
            match child {
                Some(ChildName::Reserved) => writeln!(
                    aliases,
                    "\nnamespace {module}\nexport «xls:implementation» ({local})\nend {module}"
                )?,
                // Nontrivial proofs inside a body (e.g. a vector literal's
                // length proof) can create _proof_N. Keep those in the backing
                // definition, leaving a real transparent constant at the source name.
                Some(ChildName::AuxiliaryProof) if has_body => {
                    writeln!(aliases, "\nabbrev {module}.{local} := _root_.{backing}")?
                }
                _ => return Ok(format!("_root_.{module}.{local}")),
            }
            Ok(format!("_root_.{backing}"))
        };
        let mut helpers = BTreeMap::new();
        for f in &block.helpers {
            if let Some(signature) = interface.helpers.get(&f.name) {
                let name = match &signature.source_name {
                    Some((module, local)) => path(module, local, true)?,
                    None => format!(
                        "_root_.XlsReadable.Helpers.{}",
                        lean::helper_identifier(&f.name)?
                    ),
                };
                helpers.insert(f.name.as_str(), name);
            }
        }
        let block = path(block_module, block_name, false)?;
        Ok(Self {
            helpers,
            block,
            aliases,
        })
    }
}
