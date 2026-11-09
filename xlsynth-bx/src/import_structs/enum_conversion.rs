// SPDX-License-Identifier: Apache-2.0

//! Preserve nominal enum identity in the v0.61 converter's specialization
//! names.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth::dslx::{self, TypecheckedModule};

use super::{ImportedStructs, NativeSourceFiles};
use crate::OutlineKind;
use crate::parser::dslx_tokens;
use crate::parser::outline::visit_numeric_enums;
use crate::parser::source_bytes::{
    SourceEdit, apply_source_edits, decode_dslx_source, decode_dslx_source_with_columns,
};

/// Native conversion context and its matching map to the original source.
pub(crate) struct ConversionProbe<'a> {
    pub(crate) checked: TypecheckedModule,
    pub(crate) files: &'a NativeSourceFiles,
}

impl ImportedStructs {
    /// Convert checked calls with globally distinct spellings for enum type
    /// arguments.
    pub(crate) fn conversion_probe(
        &self,
        checked: &TypecheckedModule,
        source: &str,
        module: &str,
        rewrite_enums: bool,
    ) -> Result<ConversionProbe<'_>, String> {
        let snapshot = if rewrite_enums {
            self.enum_conversion_snapshot
                .get_or_init(|| self.make_enum_conversion_snapshot())
                .as_ref()
                .map_err(Clone::clone)?
                .as_ref()
        } else {
            None
        };
        let (files, checked) = if let Some(files) = snapshot {
            let checked = dslx::parse_and_typecheck(
                &files.native_source(source),
                &format!("{module}.x"),
                module,
                &mut files.import_data(),
            )
            .map_err(|e| files.remap_diagnostic(e.to_string()))?;
            (files, checked)
        } else {
            let files = self.files.as_ref().expect("imported batch has a snapshot");
            (files, checked.clone())
        };
        Ok(ConversionProbe { checked, files })
    }

    /// v0.61 mangles a type-valued binding using the enum's unqualified name.
    /// Install a unique native declaration and retain its source name as an
    /// alias. This also covers specializations reached through maps and
    /// methods.
    fn make_enum_conversion_snapshot(&self) -> Result<Option<NativeSourceFiles>, String> {
        let ambiguous = self.ambiguous_enum_names()?;
        if ambiguous.is_empty() {
            return Ok(None);
        }

        // Avoid source declarations and explicit siblings of generated IR
        // symbols (e.g. classify__BXEnum0). The mangler separates identifier
        // components with underscores.
        const PREFIX: &str = "BXEnum";
        let occupied: BTreeSet<_> = self
            .graph
            .sources()
            .flat_map(|s| s.source.split(|b| !b.is_ascii_alphanumeric()))
            .filter_map(|word| {
                std::str::from_utf8(word.strip_prefix(PREFIX.as_bytes())?)
                    .ok()?
                    .parse::<usize>()
                    .ok()
            })
            .collect();
        let mut names = (0..).filter(|id| !occupied.contains(id));
        let mut overrides = BTreeMap::new();
        for module in self.graph.sources().filter(|s| {
            s.outline.declarations.iter().any(|(name, decl)| {
                ambiguous.contains(name.as_str()) && matches!(decl.kind, OutlineKind::Enum { .. })
            })
        }) {
            let (decoded, columns) = decode_dslx_source_with_columns(&module.source)?;
            let mut edits = Vec::new();
            visit_numeric_enums(&decoded, |name, end, public| {
                if !ambiguous.contains(name.text) {
                    return;
                }
                let id = names
                    .next()
                    .expect("finite source reserves finitely many names");
                let native_name = format!("{PREFIX}{id}");
                edits.push(SourceEdit {
                    range: name.start..name.start + name.text.len(),
                    text: native_name.clone(),
                });
                let visibility = if public { "pub " } else { "" };
                edits.push(SourceEdit {
                    range: end..end,
                    text: format!(" {visibility}type {} = {native_name};", name.text),
                });
            })?;
            let mut rewritten = apply_source_edits(decoded.as_bytes(), edits);
            rewritten.columns = rewritten.columns.with_original(columns);
            overrides.insert(module.subject.clone(), rewritten);
        }
        NativeSourceFiles::capture_with_overrides(&self.graph, overrides).map(Some)
    }

    /// Identify spellings that can collide in a type argument or with the
    /// explicit name of a function or impl method. For example,
    /// `classify__Mode` can occupy the symbol generated for
    /// `classify<Mode>`.
    fn ambiguous_enum_names(&self) -> Result<BTreeSet<&str>, String> {
        let mut declarations = BTreeSet::new();
        let mut enums = BTreeSet::new();
        let mut ambiguous = BTreeSet::new();
        for module in self.graph.sources() {
            for (name, decl) in &module.outline.declarations {
                let name = name.as_str();
                if !declarations.insert(name) {
                    ambiguous.insert(name);
                }
                if matches!(decl.kind, OutlineKind::Enum { .. }) {
                    enums.insert(name);
                    // Native type mangling uses underscores as delimiters and
                    // prints concrete bits such as u84 using their width.
                    if !name.bytes().all(|b| b.is_ascii_alphanumeric())
                        || name.strip_prefix(['u', 's']).is_some_and(|width| {
                            !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit())
                        })
                    {
                        ambiguous.insert(name);
                    }
                }
            }
        }
        if enums.is_empty() {
            return Ok(enums);
        }
        for module in self.graph.sources() {
            let source = decode_dslx_source(&module.source)?;
            for pair in dslx_tokens(&source)?.windows(2) {
                if pair[0].text == "fn" {
                    for part in pair[1].text.split('_') {
                        if let Some(&name) = enums.get(part) {
                            ambiguous.insert(name);
                        }
                    }
                }
            }
        }
        ambiguous.retain(|name| enums.contains(name));
        Ok(ambiguous)
    }
}
