// SPDX-License-Identifier: Apache-2.0

//! Classify BX use leaves before handing DSLX imports to the native frontend.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::ImportOptions;
use crate::ast::UsePath;
use crate::import_structs::ModuleSearch;
use crate::parser::{MAX_SYNTAX_DEPTH, is_use_identifier};

/// A BX file selected for a module or a member import.
pub(super) struct BxUse {
    pub subject: String,
    pub path: PathBuf,
    pub member: Option<String>,
}

/// A discovered module, before deciding whether a use refers to a member.
enum ModuleFile {
    Bx(PathBuf),
    Dslx,
}

/// File lookup decisions shared by all BX modules in one compilation.
pub(super) struct UseLookup<'a> {
    search: ModuleSearch<'a>,
    files: BTreeMap<String, Option<ModuleFile>>,
}

/// Check a programmatically constructed use path before treating its names as
/// filesystem components or interpolating them into a DSLX import.
pub(crate) fn validate_use_path(path: &UsePath) -> Result<(), String> {
    if path.components.is_empty() {
        return Err("empty DSLX use path".into());
    }
    if path.components.len() > MAX_SYNTAX_DEPTH {
        return Err(format!(
            "DSLX use path exceeds {MAX_SYNTAX_DEPTH} components"
        ));
    }
    if !path.components.iter().all(|name| is_use_identifier(name)) {
        return Err(format!("invalid DSLX use path {:?}", path.components));
    }
    Ok(())
}

impl<'a> UseLookup<'a> {
    pub(super) fn new(options: &'a ImportOptions<'a>) -> Self {
        let stdlib = options
            .dslx_stdlib_path
            .unwrap_or(xlsynth::default_dslx_stdlib_path());
        Self {
            search: ModuleSearch::new(stdlib, &options.additional_search_paths),
            files: BTreeMap::new(),
        }
    }

    /// Prefer a full module to a member of its parent, matching DSLX use
    /// semantics. For each subject, an existing DSLX source takes precedence.
    pub(super) fn resolve(&mut self, path: &UsePath) -> Result<Option<BxUse>, String> {
        validate_use_path(path)?;
        let (leaf, parent) = path.components.split_last().expect("validated use path");
        for (components, member) in [(path.components.as_slice(), None), (parent, Some(leaf))] {
            if components.is_empty() {
                continue;
            }
            let subject = components.join(".");
            let file = self.files.entry(subject.clone()).or_insert_with(|| {
                if self.search.file(&subject, "x").is_some() {
                    Some(ModuleFile::Dslx)
                } else {
                    self.search
                        .file(&subject, "bx")
                        .map(|file| ModuleFile::Bx(file.physical))
                }
            });
            match file {
                Some(ModuleFile::Dslx) => return Ok(None),
                Some(ModuleFile::Bx(path)) => {
                    return Ok(Some(BxUse {
                        subject,
                        path: path.clone(),
                        member: member.cloned(),
                    }));
                }
                None => { /* A missing full module can still name a parent member. */ }
            }
        }
        // The native import pass diagnoses a missing source after the checker
        // has validated conflicting declarations in the importing file.
        Ok(None)
    }
}
