// SPDX-License-Identifier: Apache-2.0

//! Temporary source-spelling parser used only while the DSLX AST bindings
//! cannot expose nested type annotations.
//!
//! TODO(xlsynth-ffi): Delete this entire module after the FFI exposes the
//! parsed DSLX information currently recovered here from source text:
//!
//! * Module imports: expose each parsed import's subject and optional alias so
//!   `RustBridgeBuilder::start_module_with_text` does not split module text on
//!   semicolons or parse `import ... as ...`.
//! * Nested type annotations: expose typed downcasts/accessors for array type
//!   annotations, including the array element `TypeAnnotation`, so generated
//!   Rust type names can be derived recursively from the AST.
//! * Struct member annotations: make `StructMember::get_type()` sufficient for
//!   all source-qualified nested references by exposing the annotation variants
//!   below it; then `StructDef::to_text()` should not be consulted for member
//!   type spelling.
//! * Type alias annotations: make `TypeAlias::get_type_annotation()`
//!   recursively inspectable for arrays and imported type refs; then
//!   `TypeAlias::to_text()` should not be parsed for the alias RHS.
//! * Type references: preserve the existing typed `ColonRef`/resolved
//!   import-subject path for `foo::Bar`, but make it reachable through arrays
//!   and aliases without using source text.
//!
//! When those FFI pieces exist, `RustBridgeBuilder` should pass typed
//! `dslx::TypeAnnotation` values down the conversion recursion and this module
//! should have no remaining callers.

use std::collections::BTreeMap;

/// Extracts import aliases and canonical Rust module paths from DSLX module
/// text.
///
/// The returned map is intentionally narrow: it exists only to recover the
/// module path behind source-qualified type annotations while the typed DSLX
/// bindings do not expose import declarations directly.
pub(super) fn import_aliases_from_module_text(
    module_text: &str,
    sanitize_module_segment: impl Fn(&str) -> String,
) -> BTreeMap<String, String> {
    module_text
        .split(';')
        .filter_map(|statement| {
            let import_text = statement.trim().strip_prefix("import ")?;
            let (subject, alias) = if let Some((subject, alias)) = import_text.split_once(" as ") {
                (subject.trim(), alias.trim())
            } else {
                let subject = import_text.trim();
                (subject, subject.rsplit('.').next().unwrap_or(subject))
            };
            let module_name = subject
                .split('.')
                .filter(|segment| !segment.is_empty())
                .map(&sanitize_module_segment)
                .collect::<Vec<_>>()
                .join(".");
            Some((alias.to_string(), module_name))
        })
        .collect()
}

/// Splits on a delimiter while ignoring delimiters inside nested DSLX syntax.
fn top_level_split(input: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut square_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut angle_depth = 0usize;
    for (index, ch) in input.char_indices() {
        match ch {
            '[' => square_depth = square_depth.saturating_add(1),
            ']' => square_depth = square_depth.saturating_sub(1),
            '(' => paren_depth = paren_depth.saturating_add(1),
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '{' => brace_depth = brace_depth.saturating_add(1),
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '<' => angle_depth = angle_depth.saturating_add(1),
            '>' => angle_depth = angle_depth.saturating_sub(1),
            _ => {}
        }
        if ch == delimiter
            && square_depth == 0
            && paren_depth == 0
            && brace_depth == 0
            && angle_depth == 0
        {
            parts.push(&input[start..index]);
            start = index + ch.len_utf8();
        }
    }
    parts.push(&input[start..]);
    parts
}

/// Splits a DSLX `name: annotation` field without splitting `foo::Bar`.
fn split_name_annotation(input: &str) -> Option<(String, String)> {
    let bytes = input.as_bytes();
    for (index, ch) in input.char_indices() {
        if ch == ':'
            && bytes.get(index.wrapping_sub(1)) != Some(&b':')
            && bytes.get(index + 1) != Some(&b':')
        {
            let name = input[..index].trim();
            let annotation = input[index + 1..].trim();
            if !name.is_empty() && !annotation.is_empty() {
                return Some((name.to_string(), annotation.to_string()));
            }
        }
    }
    None
}

/// Extracts source-spelled field annotations from one DSLX struct definition.
///
/// The typed `StructMember` data remains authoritative for member names and
/// concrete types; this source text is only a fallback for annotations such as
/// arrays of imported DSLX types whose nested type references are not yet
/// exposed by the FFI.
pub(super) fn struct_member_annotations_from_text(struct_text: &str) -> BTreeMap<String, String> {
    let Some(open) = struct_text.find('{') else {
        return BTreeMap::new();
    };
    let Some(close) = struct_text.rfind('}') else {
        return BTreeMap::new();
    };
    top_level_split(&struct_text[open + 1..close], ',')
        .into_iter()
        .filter_map(split_name_annotation)
        .collect()
}

/// Returns the right-hand-side annotation text for one DSLX type alias.
///
/// Empty or malformed alias text returns `None` so callers keep using the
/// concrete structural type.
pub(super) fn type_alias_annotation_from_text(alias_text: &str) -> Option<&str> {
    let equals = alias_text.find('=')?;
    let rest = &alias_text[equals + 1..];
    let semicolon = rest.rfind(';').unwrap_or(rest.len());
    Some(rest[..semicolon].trim()).filter(|annotation| !annotation.is_empty())
}

/// Returns the element annotation from a source-spelled DSLX array annotation.
///
/// The parser walks from the final bracket so nested array elements keep their
/// original annotation text for the next recursive conversion step.
pub(super) fn array_element_annotation(annotation_text: &str) -> Option<&str> {
    let trimmed = annotation_text.trim();
    if !trimmed.ends_with(']') {
        return None;
    }
    let mut depth = 0usize;
    for (index, ch) in trimmed.char_indices().rev() {
        if ch == ']' {
            depth = depth.saturating_add(1);
        } else if ch == '[' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let element = trimmed[..index].trim();
                return (!element.is_empty()).then_some(element);
            }
        }
    }
    None
}

/// Returns whether an annotation names a built-in DSLX bits-like type.
fn is_builtin_bits_annotation(annotation_text: &str) -> bool {
    if annotation_text == "bool"
        || annotation_text.starts_with("uN[")
        || annotation_text.starts_with("sN[")
        || annotation_text.starts_with("bits[")
    {
        return true;
    }
    let digits = annotation_text
        .strip_prefix('u')
        .or_else(|| annotation_text.strip_prefix('s'));
    digits.is_some_and(|digits| !digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit()))
}

/// Converts a source-spelled DSLX type reference into a Rust bridge path.
///
/// Built-in bits-like annotations, empty text, and array annotations return
/// `None` so callers can fall back to the typed concrete conversion path.
pub(super) fn rust_type_ref_from_annotation_text(
    current_module_path: &[String],
    annotation_text: &str,
    import_aliases: &BTreeMap<String, String>,
) -> Option<String> {
    let trimmed = annotation_text.trim();
    if trimmed.is_empty() || trimmed.contains('[') || is_builtin_bits_annotation(trimmed) {
        return None;
    }
    if let Some((module_ref, attr)) = trimmed.rsplit_once("::") {
        let module_key = module_ref.rsplit('.').next().unwrap_or(module_ref);
        let module_name = import_aliases
            .get(module_ref)
            .or_else(|| import_aliases.get(module_key))
            .cloned()
            .unwrap_or_else(|| {
                module_ref
                    .split('.')
                    .filter(|segment| !segment.is_empty())
                    .map(super::sanitize_module_segment)
                    .collect::<Vec<_>>()
                    .join(".")
            });
        let target_module_path = super::rust_module_path_from_dslx_module_name(&module_name);
        Some(super::rust_type_path_between_module_paths(
            current_module_path,
            &target_module_path,
            attr,
        ))
    } else {
        Some(trimmed.to_string())
    }
}
