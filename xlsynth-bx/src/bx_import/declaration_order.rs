// SPDX-License-Identifier: Apache-2.0

//! Verify source declaration order after resolving references in the ordinary
//! BX scope.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use super::{BxUse, Names};
use crate::ast::Module;
use crate::dslx_import::ImportResolver;

/// Declarations in one source file and its candidate forward references in
/// source order.
pub(crate) struct DeclarationOrder {
    references: RefCell<BTreeMap<usize, DeclarationReferences>>,
    library: Option<(String, PathBuf)>,
}

impl DeclarationOrder {
    /// Check that source positions account for every declaration and import
    /// binding.
    pub(super) fn new(
        module: &Module,
        names: &Names,
        library: Option<&BxUse>,
    ) -> Result<Self, String> {
        if module.declaration_order.len()
            != names.definitions.len() + names.uses.len() + module.tests.len()
            || names
                .definitions
                .iter()
                .chain(names.uses.keys())
                .chain(module.tests.iter().map(|test| &test.name))
                .any(|name| !names.positions.contains_key(name))
        {
            return Err("require_define_before_use needs a declaration_order containing each declaration and use binding once".into());
        }
        Ok(Self {
            references: RefCell::default(),
            library: library.map(|source| (source.subject.clone(), source.path.clone())),
        })
    }

    /// Retain the resolved source spelling and any native binding that may
    /// leave a builtin visible.
    pub(super) fn record(
        &self,
        position: usize,
        owner: &str,
        spelling: &str,
        native_builtin: Option<&str>,
    ) {
        self.references
            .borrow_mut()
            .entry(position)
            .or_insert_with(|| DeclarationReferences {
                owner: owner.to_owned(),
                references: Vec::new(),
            })
            .references
            .push(ForwardReference {
                spelling: spelling.to_owned(),
                native_builtin: native_builtin.map(str::to_owned),
            });
    }

    /// Report the first violation in source declaration order, with the file
    /// context for an imported module.
    pub(crate) fn verify(self, resolver: &ImportResolver) -> Result<(), String> {
        let context = |error| match &self.library {
            Some((subject, path)) => format!("BX import '{subject}' ({}): {error}", path.display()),
            None => error,
        };
        for declaration in self.references.into_inner().into_values() {
            for reference in declaration.references {
                reference
                    .verify(&declaration.owner, resolver)
                    .map_err(context)?;
            }
        }
        Ok(())
    }
}

/// The source declaration and its references to later module bindings.
struct DeclarationReferences {
    owner: String,
    references: Vec<ForwardReference>,
}

/// A reference to a later source binding, pending native builtin resolution.
struct ForwardReference {
    spelling: String,
    native_builtin: Option<String>,
}

impl ForwardReference {
    /// Report the forward reference when the source binding supplies the
    /// selected value.
    fn verify(&self, owner: &str, resolver: &ImportResolver) -> Result<(), String> {
        if let Some(alias) = &self.native_builtin
            && let Some((subject, member)) = resolver.lookup_enum_prefix(alias)?
        {
            // A later enum or alias leaves map and zero! in scope. Other
            // ordinary builtins also stay visible before an imported struct.
            let builtin = if matches!(self.spelling.as_str(), "map" | "zero!") {
                resolver.is_enum_or_alias(&subject, &member)
            } else {
                resolver.has_public_type(&subject, &member)
            };
            if builtin {
                return Ok(());
            }
        }
        Err(format!(
            "in '{owner}': '{}' is declared later in the module",
            self.spelling
        ))
    }
}
