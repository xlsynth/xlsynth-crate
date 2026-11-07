// SPDX-License-Identifier: Apache-2.0

//! Choose declaration and reference paths together, before emitting helper
//! bodies.

use std::collections::BTreeMap;

use super::*;

pub(super) struct CallableNames<'a> {
    pub(super) helpers: BTreeMap<&'a str, String>,
    pub(super) block: String,
    pub(super) aliases: String,
}

#[derive(Clone, Copy)]
enum ChildName {
    Reserved,
    AuxiliaryProof,
}

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
        let mut children = BTreeMap::new();
        for (module, local) in names
            .chain(
                declarations
                    .iter()
                    .map(|d| declaring_name(bx_namespace, d.name())),
            )
            .chain([(bx_namespace, block.graph.name.as_str())])
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
        let block = path(bx_namespace, &block.graph.name, false)?;
        Ok(Self {
            helpers,
            block,
            aliases,
        })
    }
}
