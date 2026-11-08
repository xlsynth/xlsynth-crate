// SPDX-License-Identifier: Apache-2.0

//! Relocate frozen DSLX modules for native imports and restore source-level
//! identifiers and positions when they cross back into BX.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;
use xlsynth::dslx::ImportData;

use super::OutlineGraph;
use super::XlsRunfiles;
use crate::parser::source_bytes::{SourceColumns, relocate_dslx_imports};

/// Source identity and positions before injecting private import components.
struct SourceMap {
    logical: PathBuf,
    columns: SourceColumns,
    mangled_physical: String,
    mangled_logical: String,
}

/// Private native import roots and the mapping back to the Rust snapshot.
pub(crate) struct NativeSourceFiles {
    directory: TempDir,
    roots: Vec<PathBuf>,
    namespace: String,
    mangled_prefix: String,
    mangled_directory: String,
    sources: BTreeMap<PathBuf, SourceMap>,
    subjects: BTreeSet<String>,
    short_components: BTreeMap<String, String>,
    renamed_modules: Vec<(String, String)>,
}

impl NativeSourceFiles {
    /// Namespacing twice keeps both native full and depot-stripped searches
    /// inside private module names, including when a source lives in CWD.
    pub(super) fn capture(graph: &OutlineGraph) -> Result<Self, String> {
        let runfiles = XlsRunfiles::from_env();
        let (directory, namespace) = loop {
            let directory = tempfile::Builder::new()
                .prefix("bx_snapshot_")
                .rand_bytes(16)
                .tempdir()
                .map_err(|e| format!("creating DSLX source snapshot: {e}"))?;
            let name = directory
                .path()
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("DSLX snapshot directory must have a UTF-8 name")?;
            let namespace = format!("bxpkg_{name}");
            if !Path::new(&namespace).exists()
                && !runfiles.contains_prefix(&namespace)
                && graph.sources().all(|s| {
                    !s.subject.contains(&namespace)
                        && !s.source_path.to_string_lossy().contains(&namespace)
                        && !s
                            .source
                            .windows(namespace.len())
                            .any(|w| w == namespace.as_bytes())
                })
            {
                break (directory, namespace);
            }
        };
        let base = std::path::absolute(directory.path())
            .map_err(|e| format!("locating DSLX source snapshot: {e}"))?;
        // The C API takes filesystem bytes but exposes diagnostic and IR text
        // as lossy UTF-8. Use the same spelling for reverse source mappings.
        let base_text = base.to_string_lossy();
        let mangled_directory = mangle_scope_path(&base_text);
        let subjects: BTreeSet<_> = graph.sources().map(|s| s.subject.clone()).collect();
        let short_components = compact_components(
            &subjects,
            &graph.member_use_names,
            base_text.len() + 2 * namespace.len() + 22,
        )?;
        let mut renamed_modules = Vec::new();
        let mut roots = Vec::new();
        let mut sources = BTreeMap::new();
        // XLS uses sanitized filenames in generated identifiers. Put paths
        // such as a/b.x and a_b.x in different roots so their sanitized
        // snapshot names also distinguish the two originals.
        let mut collisions = BTreeMap::<String, usize>::new();
        for source in graph.sources() {
            let encoded = encode_subject(&source.subject, &short_components);
            let relative = format!("{}.x", encoded.replace('.', "/"));
            if encoded != source.subject {
                renamed_modules.push((encoded, source.subject.clone()));
            }
            let index = collisions.entry(mangle_scope_path(&relative)).or_default();
            if *index == roots.len() {
                roots.push(base.join(format!("root{index}")));
            }
            let physical = roots[*index]
                .join(&namespace)
                .join(&namespace)
                .join(relative);
            *index += 1;
            let relocated =
                relocate_dslx_imports(&source.source, &namespace, &short_components, &subjects);
            fs::create_dir_all(physical.parent().expect("snapshot source has a parent"))
                .and_then(|()| fs::write(&physical, &relocated.bytes))
                .map_err(|e| format!("writing frozen DSLX module '{}': {e}", source.subject))?;
            let logical = source.source_path.clone();
            let rendered_path = physical.to_string_lossy();
            sources.insert(
                PathBuf::from(rendered_path.as_ref()),
                SourceMap {
                    mangled_physical: mangle_scope_path(&rendered_path),
                    mangled_logical: mangle_scope_path(&logical.to_string_lossy()),
                    logical,
                    columns: relocated.columns,
                },
            );
        }
        let mangled_prefix = format!("{namespace}_{namespace}_");
        renamed_modules.sort_by_key(|(encoded, _)| Reverse(encoded.len()));
        Ok(Self {
            directory,
            roots,
            namespace,
            mangled_prefix,
            mangled_directory,
            sources,
            subjects,
            short_components,
            renamed_modules,
        })
    }

    pub(crate) fn import_data(&self) -> ImportData {
        let paths: Vec<_> = self.roots.iter().map(PathBuf::as_path).collect();
        // Standard-library subjects are relocated like every other module.
        ImportData::new(Some(self.directory.path()), &paths)
    }

    pub(crate) fn native_source(&self, source: &str) -> String {
        String::from_utf8(
            relocate_dslx_imports(
                source.as_bytes(),
                &self.namespace,
                &self.short_components,
                &self.subjects,
            )
            .bytes,
        )
        .expect("inserting ASCII module paths preserves UTF-8")
    }

    /// The exact native DSLX module name assigned to a captured subject.
    pub(crate) fn native_module_name(&self, subject: &str) -> String {
        format!(
            "{0}.{0}.{1}",
            self.namespace,
            encode_subject(subject, &self.short_components)
        )
    }

    pub(crate) fn logical_path(&self, physical: &Path) -> Option<&Path> {
        self.sources.get(physical).map(|s| s.logical.as_path())
    }

    #[cfg(test)]
    pub(super) fn physical_path(&self, logical: &Path) -> Option<&Path> {
        self.sources
            .iter()
            .find_map(|(path, source)| (source.logical == logical).then_some(path.as_path()))
    }

    pub(crate) fn original_column(&self, physical: &Path, line: usize, column: usize) -> usize {
        self.sources
            .get(physical)
            .map_or(column, |s| s.columns.original_column(line, column))
    }

    /// Restore native identifiers before IR parsing, including lambda capture
    /// scopes and loop-body references. Slash-delimited file_number paths
    /// stay available for source and column mapping after parsing.
    pub(crate) fn restore_ir_names(&self, text: &str) -> String {
        if !text.contains(&self.namespace) {
            return text.to_owned();
        }
        let mut text = text.to_owned();
        if text.contains(&self.mangled_directory) {
            let mut sources: Vec<_> = self.sources.values().collect();
            sources.sort_by_key(|s| Reverse(s.mangled_physical.len()));
            for source in sources {
                if text.contains(&source.mangled_physical) {
                    text = replace_location(
                        &text,
                        &source.mangled_physical,
                        &source.mangled_logical,
                        &source.columns,
                        true,
                    );
                }
            }
        }
        for (encoded, original) in &self.renamed_modules {
            text = text.replace(
                &format!("{}{}__", self.mangled_prefix, encoded.replace('.', "_")),
                &format!("{}__", original.replace('.', "_")),
            );
        }
        text.replace(&self.mangled_prefix, "")
    }

    /// Restore filenames, one-based source positions, and module names in
    /// native errors, including filenames containing spaces.
    pub(crate) fn remap_diagnostic(&self, mut message: String) -> String {
        let mut paths: Vec<_> = self.sources.iter().collect();
        paths.sort_by_key(|(physical, _)| Reverse(physical.as_os_str().len()));
        for (physical, source) in paths {
            let filename = physical.to_str().expect("UTF-8 snapshot path");
            if message.contains(filename) {
                message = replace_location(
                    &message,
                    filename,
                    &source.logical.to_string_lossy(),
                    &source.columns,
                    false,
                );
            }
        }
        for (encoded, original) in &self.renamed_modules {
            message = message
                .replace(&format!("{0}.{0}.{encoded}", self.namespace), original)
                .replace(
                    &format!("{0}::{0}::{1}", self.namespace, encoded.replace('.', "::")),
                    &original.replace('.', "::"),
                );
        }
        self.restore_ir_names(&message)
            .replace(&format!("{0}.{0}.", self.namespace), "")
            .replace(&format!("{0}::{0}::", self.namespace), "")
    }
}

fn encode_subject(subject: &str, short: &BTreeMap<String, String>) -> String {
    subject
        .split('.')
        .map(|c| short.get(c).map_or(c, String::as_str))
        .collect::<Vec<_>>()
        .join(".")
}

/// An import reached through a stripped depot root or a runfiles manifest
/// can contain a logical component longer than a native filename. Assign
/// disjoint short module identifiers only where the snapshot needs them.
fn compact_components(
    subjects: &BTreeSet<String>,
    member_use_names: &BTreeSet<String>,
    path_overhead: usize,
) -> Result<BTreeMap<String, String>, String> {
    // Include space for the terminating NUL and the ".x" extension.
    let path_limit = if cfg!(target_os = "macos") {
        1023
    } else {
        4095
    };
    let mut short = BTreeMap::new();
    let mangled: Vec<_> = subjects.iter().map(|s| s.replace('.', "_")).collect();
    let mut serial = 0;
    for subject in subjects {
        let mut components: Vec<_> = subject.split('.').collect();
        components.sort_by_key(|part| Reverse(part.len()));
        let mut length = encode_subject(subject, &short).len() + path_overhead;
        for part in components {
            if short.contains_key(part) {
                continue;
            }
            if part.len() <= 240 && length <= path_limit {
                break;
            }
            let alias = loop {
                let next = format!("Q{serial}");
                serial += 1;
                if !member_use_names.contains(&next)
                    && mangled.iter().all(|name| !name.contains(&next))
                {
                    break next;
                }
            };
            if alias.len() >= part.len() {
                break;
            }
            short.insert(part.to_owned(), alias);
            length = encode_subject(subject, &short).len() + path_overhead;
        }
        if length > path_limit {
            return Err(
                "DSLX snapshot import path is too long (check TMPDIR and module depth)".into(),
            );
        }
    }
    Ok(short)
}

/// Mirror the scope substitutions in native MangleDslxName.
fn mangle_scope_path(path: &str) -> String {
    let mut mangled = String::with_capacity(path.len());
    for ch in path.chars() {
        match ch {
            '>' | '<' | '/' | ':' | '-' | '.' => mangled.push('_'),
            ',' => mangled.push_str("__"),
            ' ' => { /* Spaces are removed from native scope names. */ }
            _ => mangled.push(ch),
        }
    }
    mangled
}

/// Read one one-based line and column after a filename or the span separator.
fn position(text: &str, start: char, sep: char) -> Option<(usize, usize, &str)> {
    let text = text.strip_prefix(start)?;
    let (line, rest) = text.split_once(sep)?;
    if line.is_empty() || !line.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let end = rest.bytes().take_while(u8::is_ascii_digit).count();
    let column: usize = rest[..end].parse().ok()?;
    let line: usize = line.parse().ok()?;
    Some((line.checked_sub(1)?, column.checked_sub(1)?, &rest[end..]))
}

/// Replace a filename, remapping a following position or span if present.
fn replace_location(
    text: &str,
    from: &str,
    to: &str,
    columns: &SourceColumns,
    mangled: bool,
) -> String {
    let sep = if mangled { '_' } else { ':' };
    let range = if mangled { '_' } else { '-' };
    let mut result = String::new();
    let mut rest = text;
    while let Some((before, tail)) = rest.split_once(from) {
        result.push_str(before);
        result.push_str(to);
        rest = tail;
        if let Some((line, column, tail)) = position(rest, sep, sep) {
            result.push_str(&format!(
                "{sep}{}{sep}{}",
                line + 1,
                columns.original_column(line, column) + 1
            ));
            rest = tail;
            if let Some((line, column, tail)) = position(rest, range, sep) {
                result.push_str(&format!(
                    "{range}{}{sep}{}",
                    line + 1,
                    columns.original_column(line, column) + 1
                ));
                rest = tail;
            }
        }
    }
    result.push_str(rest);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_module_components_cannot_capture_a_member_use() -> Result<(), String> {
        let long = "m".repeat(260);
        let subject = format!("parent.{long}");
        let subjects = BTreeSet::from(["parent".into(), subject.clone()]);
        let member_names = BTreeSet::from(["Q0".into()]);
        let short = compact_components(&subjects, &member_names, 120)?;
        assert_eq!(encode_subject(&subject, &short), "parent.Q1");
        Ok(())
    }
}
