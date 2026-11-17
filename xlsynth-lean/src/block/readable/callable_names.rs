// SPDX-License-Identifier: Apache-2.0

//! Choose the Lean name for the selected block. A declaration under that
//! name can occupy a child Lean normally reserves for an abbreviation.
//! Expose the public name with an export in that case.

use super::*;

pub(super) struct CallableNames {
    pub(super) block: String,
    pub(super) aliases: String,
}

/// Recognize child names that Lean reserves even for a simple abbreviation.
fn reserved_child(name: &str) -> bool {
    let numbered = |prefix| {
        name.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    matches!(
        name,
        "eq_def" | "eq_unfold" | "fun_cases" | "fun_cases_unfolding" | "congr_simp"
    ) || numbered("eq_")
        || numbered("hcongr_")
}

impl CallableNames {
    pub(super) fn new(
        block: &BlockContext<'_>,
        declarations: &[TypeDecl],
        bx_namespace: &str,
    ) -> Result<Self> {
        let local = block.graph.name.as_str();
        let reserved = declarations.iter().any(|decl| {
            let (owner, child) = declaring_name(bx_namespace, decl.name());
            owner.rsplit_once('.') == Some((bx_namespace, local)) && reserved_child(child)
        });
        let local = source_names::declaration_identifier(bx_namespace, local, "block")?;
        let module = module_identifier(bx_namespace)?;
        if reserved {
            Ok(Self {
                block: format!("_root_.{module}.«xls:implementation».{local}"),
                aliases: format!(
                    "\nnamespace {module}\nexport «xls:implementation» ({local})\nend {module}\n"
                ),
            })
        } else {
            Ok(Self {
                block: format!("_root_.{module}.{local}"),
                aliases: String::new(),
            })
        }
    }
}
