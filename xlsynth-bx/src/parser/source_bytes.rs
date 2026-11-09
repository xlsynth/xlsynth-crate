// SPDX-License-Identifier: Apache-2.0

//! Adapt legacy DSLX file bytes to the Rust parser's UTF-8 source view.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::ops::Range;

use super::{dslx_identifier_tail, identifier_start};

/// Keep a native-printed non-UTF-8 byte character as one token for lexical
/// inspection. Neither the original source nor the evaluated value is changed.
pub(crate) fn native_display_tokens_source(text: &str) -> Cow<'_, str> {
    if text.contains("'\u{fffd}'") {
        Cow::Owned(text.replace("'\u{fffd}'", "'?'"))
    } else {
        Cow::Borrowed(text)
    }
}

/// Lexical contexts where the native byte scanner treats a non-UTF-8 byte
/// differently. In particular a prime on an identifier never opens a literal.
#[derive(Clone, Copy)]
enum Region {
    Code,
    Identifier,
    LineComment,
    Quoted { delimiter: u8, escaped: bool },
    Character { close: usize },
}

/// A native source copy with private module paths and the offsets they added.
pub(crate) struct RelocatedSource {
    pub(crate) bytes: Vec<u8>,
    pub(crate) columns: SourceColumns,
}

/// Changed token ranges on each line, in increasing source order.
#[derive(Default)]
pub(crate) struct SourceColumns {
    edits: Vec<ColumnEdit>,
    original: Option<Box<SourceColumns>>,
}

/// One path edit, with zero-based byte columns in both source views.
struct ColumnEdit {
    line: usize,
    original: Range<usize>,
    relocated: Range<usize>,
}

impl SourceColumns {
    /// Compose two line-preserving rewrites, newest to oldest.
    pub(crate) fn with_original(mut self, original: Self) -> Self {
        if self.edits.is_empty() {
            return original;
        }
        if !original.edits.is_empty() {
            let original = match self.original.take() {
                Some(prior) => prior.with_original(original),
                None => original,
            };
            self.original = Some(Box::new(original));
        }
        self
    }

    /// Translate a native zero-based byte column back to the input file.
    pub(crate) fn original_column(&self, line: usize, column: usize) -> usize {
        let prior = self
            .edits
            .partition_point(|i| (i.line, i.relocated.start) <= (line, column));
        let column = match self.edits[..prior].last() {
            Some(i) if i.line == line && column >= i.relocated.end => {
                i.original.end + (column - i.relocated.end)
            }
            Some(i) if i.line == line => {
                i.original.start + (column - i.relocated.start).min(i.original.len())
            }
            _ => column,
        };
        self.original
            .as_ref()
            .map_or(column, |original| original.original_column(line, column))
    }
}

/// Replace a source byte range with single-line text, retaining its newlines.
pub(crate) struct SourceEdit {
    pub(crate) range: Range<usize>,
    pub(crate) text: String,
}

/// Splice only the selected tokens, preserving newlines and tracking columns
/// across multiple expansions or contractions on the same line.
pub(crate) fn apply_source_edits(bytes: &[u8], edits: Vec<SourceEdit>) -> RelocatedSource {
    let mut output = Vec::with_capacity(bytes.len());
    let mut columns = SourceColumns::default();
    let (mut line, mut original_col, mut relocated_col) = (0, 0, 0);
    let mut copied = 0;
    for edit in edits {
        let unchanged = &bytes[copied..edit.range.start];
        output.extend_from_slice(unchanged);
        if let Some(newline) = unchanged.iter().rposition(|&b| b == b'\n') {
            line += unchanged.iter().filter(|&&b| b == b'\n').count();
            original_col = unchanged.len() - newline - 1;
            relocated_col = original_col;
        } else {
            original_col += unchanged.len();
            relocated_col += unchanged.len();
        }
        let mut replacement = edit.text.as_bytes();
        for (i, part) in bytes[edit.range.clone()].split(|&b| b == b'\n').enumerate() {
            if i != 0 {
                output.push(b'\n');
                line += 1;
                original_col = 0;
                relocated_col = 0;
            }
            columns.edits.push(ColumnEdit {
                line,
                original: original_col..original_col + part.len(),
                relocated: relocated_col..relocated_col + replacement.len(),
            });
            original_col += part.len();
            relocated_col += replacement.len();
            output.extend_from_slice(replacement);
            replacement = &[];
        }
        copied = edit.range.end;
    }
    output.extend_from_slice(&bytes[copied..]);
    RelocatedSource {
        bytes: output,
        columns,
    }
}

/// Look past trivia between an import path component and its delimiter.
fn next_path_byte(mut tail: &[u8]) -> Option<u8> {
    loop {
        let first = *tail.first()?;
        if first.is_ascii_whitespace() || first == 0xa0 {
            tail = &tail[1..];
        } else if tail.starts_with(b"//") {
            tail = &tail[tail.iter().position(|&b| b == b'\n')?..];
        } else {
            return Some(first);
        }
    }
}

/// Prefix subjects in native import/use declarations. Only path components
/// too long for the snapshot filesystem are shortened. Preserve explicit and
/// implicit local bindings and leave type/function use leaves untouched.
pub(crate) fn relocate_dslx_imports(
    bytes: &[u8],
    namespace: &str,
    short_components: &BTreeMap<String, String>,
    modules: &BTreeSet<String>,
) -> RelocatedSource {
    let import_prefix = format!("{namespace}.{namespace}.");
    let use_prefix = format!("{namespace}::{namespace}::");
    let mut edits = Vec::new();
    let mut region = Region::Code;
    let mut depth = 0usize;
    let mut declaration = None;
    let mut need_prefix = false;
    let mut use_path = Vec::<&str>::new();
    let mut groups = Vec::new();
    for (offset, &byte) in bytes.iter().enumerate() {
        let in_code = matches!(region, Region::Code)
            || (matches!(region, Region::Identifier) && !dslx_identifier_tail(byte));
        if in_code {
            match byte {
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    if declaration == Some("use") && byte == b'{' {
                        groups.push(use_path.len());
                    }
                }
                b')' | b']' | b'}' => {
                    depth = depth.saturating_sub(1);
                    if declaration == Some("use") && byte == b'}' {
                        groups.pop();
                    }
                }
                b',' if declaration == Some("use") => {
                    if let Some(&parent) = groups.last() {
                        use_path.truncate(parent);
                    }
                }
                b';' => declaration = None,
                _ if identifier_start(byte) && (declaration.is_some() || depth == 0) => {
                    let tail = &bytes[offset..];
                    let len = tail
                        .iter()
                        .position(|&b| !dslx_identifier_tail(b))
                        .unwrap_or(tail.len());
                    let word = std::str::from_utf8(&tail[..len]).expect("ASCII DSLX identifier");
                    if let Some(kind) = declaration {
                        if need_prefix {
                            let prefix = if kind == "use" {
                                &use_prefix
                            } else {
                                &import_prefix
                            };
                            edits.push(SourceEdit {
                                range: offset..offset,
                                text: prefix.clone(),
                            });
                            need_prefix = false;
                        }
                        if short_components.is_empty() || word == "as" {
                            declaration = None;
                        } else {
                            if kind == "use" {
                                use_path.push(word);
                            }
                            if let Some(short) = short_components.get(word) {
                                let next = next_path_byte(&tail[len..]);
                                let module = kind == "import"
                                    || next == Some(b':')
                                    || modules.contains(&use_path.join("."));
                                if module {
                                    let edit = if kind == "import" && next == Some(b';') {
                                        // The original leaf becomes an explicit
                                        // alias. Leave
                                        // its identifier and source position
                                        // intact.
                                        SourceEdit {
                                            range: offset..offset,
                                            text: format!("{short} as "),
                                        }
                                    } else {
                                        SourceEdit {
                                            range: offset..offset + len,
                                            text: short.clone(),
                                        }
                                    };
                                    edits.push(edit);
                                }
                            }
                        }
                    } else if matches!(word, "import" | "use") {
                        declaration = Some(word);
                        need_prefix = true;
                        use_path.clear();
                        groups.clear();
                    }
                }
                _ => { /* Punctuation and trivia do not bind a module name. */ }
            }
        }
        region = advance(region, byte, offset, bytes);
    }
    apply_source_edits(bytes, edits)
}

/// Present a byte-oriented DSLX module as UTF-8 for outlining. The native
/// importer separately consumes the original bytes, retaining its exact source
/// positions. Invalid UTF-8 in literals keeps its value through DSLX's hex
/// escape. An invalid byte in a comment or a Latin-1 space occupies one space.
pub(crate) fn decode_dslx_source(bytes: &[u8]) -> Result<Cow<'_, str>, String> {
    decode_dslx_source_with_columns(bytes).map(|(source, _)| source)
}

/// Keep the original byte columns of legacy bytes expanded into DSLX escapes.
pub(crate) fn decode_dslx_source_with_columns(
    bytes: &[u8],
) -> Result<(Cow<'_, str>, SourceColumns), String> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return Ok((Cow::Borrowed(text), SourceColumns::default()));
    }
    let mut columns = SourceColumns::default();
    let (mut line, mut raw_start, mut decoded_start) = (0, 0, 0);
    let mut output = String::with_capacity(bytes.len());
    let mut region = Region::Code;
    let mut offset = 0;
    for chunk in bytes.utf8_chunks() {
        if let Some(last) = chunk.valid().rfind('\n') {
            line += chunk.valid().bytes().filter(|&b| b == b'\n').count();
            raw_start = offset + last + 1;
            decoded_start = output.len() + last + 1;
        }
        output.push_str(chunk.valid());
        for &byte in chunk.valid().as_bytes() {
            region = advance(region, byte, offset, bytes);
            offset += 1;
        }
        for &byte in chunk.invalid() {
            match region {
                Region::LineComment => output.push(' '),
                Region::Code | Region::Identifier if byte == 0xa0 => output.push(' '),
                Region::Character { .. } | Region::Quoted { escaped: false, .. } => {
                    let begin = output.len() - decoded_start;
                    write!(output, "\\x{byte:02x}").expect("writing to a String");
                    columns.edits.push(ColumnEdit {
                        line,
                        original: offset - raw_start..offset - raw_start + 1,
                        relocated: begin..output.len() - decoded_start,
                    });
                }
                _ => {
                    return Err(format!(
                        "unexpected DSLX byte 0x{byte:02x} at offset {offset}"
                    ));
                }
            }
            region = advance(region, byte, offset, bytes);
            offset += 1;
        }
    }
    Ok((Cow::Owned(output), columns))
}

/// Follow DSLX quoting and line comments without decoding valid source text.
fn advance(region: Region, byte: u8, offset: usize, bytes: &[u8]) -> Region {
    match region {
        Region::Identifier if dslx_identifier_tail(byte) => Region::Identifier,
        Region::Code | Region::Identifier => match byte {
            b'/' if bytes.get(offset + 1) == Some(&b'/') => Region::LineComment,
            b'"' | b'`' => Region::Quoted {
                delimiter: byte,
                escaped: false,
            },
            b'\'' if bytes.get(offset + 1) == Some(&b'\\') => Region::Character {
                close: offset
                    + if bytes.get(offset + 2) == Some(&b'x') {
                        5
                    } else {
                        3
                    },
            },
            b'\'' if bytes.get(offset + 2) == Some(&b'\'') => {
                Region::Character { close: offset + 2 }
            }
            _ if identifier_start(byte) => Region::Identifier,
            _ => Region::Code,
        },
        Region::LineComment if byte == b'\n' => Region::Code,
        Region::Quoted {
            delimiter,
            escaped: false,
        } if byte == delimiter => Region::Code,
        Region::Quoted { delimiter, escaped } => Region::Quoted {
            delimiter,
            escaped: !escaped && byte == b'\\',
        },
        Region::Character { close } if offset == close => Region::Code,
        current => current,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};
    use xlsynth_pir::{IrValue, ir_eval::eval_fn_in_package_value, ir_parser};

    use super::*;
    use crate::parse_dslx_outline;

    #[test]
    fn relocate_only_module_subjects_in_byte_source() {
        let input = b"// import hidden; \xff\n#[attr(s=`use hidden`)]\n\
use // import ignored\n pkg::{wire, b::Packet}; import\xa0helper as h; fn import'(){\n\
 let use' = \"import fake;\\\" use fake;\0\"; let c = '}'; { import token; use token; } }";
        let expected = b"// import hidden; \xff\n#[attr(s=`use hidden`)]\n\
use // import ignored\n root::root::pkg::{wire, b::Packet}; import\xa0root.root.helper as h; fn import'(){\n\
 let use' = \"import fake;\\\" use fake;\0\"; let c = '}'; { import token; use token; } }";
        let relocated = relocate_dslx_imports(input, "root", &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(relocated.bytes, expected);
        // Every original byte on the line, including the first byte of each
        // subject, has its original native column after removing both inserts.
        let original = input.split(|&b| b == b'\n').nth(3).unwrap();
        let output = expected.split(|&b| b == b'\n').nth(3).unwrap();
        let first = original.windows(3).position(|s| s == b"pkg").unwrap();
        let second = original.windows(6).position(|s| s == b"helper").unwrap();
        for (col, byte) in original.iter().enumerate() {
            let added = usize::from(col >= first) * b"root::root::".len()
                + usize::from(col >= second) * b"root.root.".len();
            assert_eq!(output[col + added], *byte);
            assert_eq!(relocated.columns.original_column(3, col + added), col);
        }
        assert_eq!(relocated.columns.original_column(4, 19), 19);
    }

    #[test]
    fn shortened_components_preserve_member_leaves_and_local_bindings() {
        let short = BTreeMap::from([("depot".into(), "d".into()), ("Huge".into(), "q".into())]);
        let modules = BTreeSet::from(["depot.pkg".into(), "depot.Huge".into(), "Huge".into()]);
        let input = b"use depot::{pkg::Huge, Huge::Member, Huge}; import depot.Huge; import Huge as Alias; pub fn f() { Huge::read(); \"depot.Huge\" }";
        let expected = b"use ns::ns::d::{pkg::Huge, q::Member, q}; import ns.ns.d.q as Huge; import ns.ns.q as Alias; pub fn f() { Huge::read(); \"depot.Huge\" }";
        let relocated = relocate_dslx_imports(input, "ns", &short, &modules);
        assert_eq!(relocated.bytes, expected);
        for token in [
            b"Huge,".as_slice(),
            b"Huge;",
            b"Alias",
            b"pub fn",
            b"Huge::read",
        ] {
            let source_column = input.windows(token.len()).position(|w| w == token).unwrap();
            let native_column = expected
                .windows(token.len())
                .position(|w| w == token)
                .unwrap();
            assert_eq!(
                relocated.columns.original_column(0, native_column),
                source_column
            );
        }
    }

    #[test]
    fn relocated_native_calls_keep_values_and_source_columns()
    -> Result<(), Box<dyn std::error::Error>> {
        for (declarations, value) in [
            (b"import pkg.wire as wire; import pkg.aux; ".as_slice(), b"wire::plus(x) + aux::bias()".as_slice()),
            (b"#![feature(type_inference_v1)]\n#![feature(use_syntax)]\nuse pkg::{wire::H, aux::T}; pub struct Frame { h:H, t:T } ",
             b"x + u8:6 + zero!<Frame>().h.n + zero!<Frame>().t.n"),
        ] {
            let source = [
                declarations,
                b"pub fn calc(x: u8) -> u8 { let s' = \"\xff\"; ", value, b" + s'[u32:0] }",
            ].concat();
            let relocated = relocate_dslx_imports(&source, "ns", &BTreeMap::new(), &BTreeSet::new());
            let mut positions = Vec::new();
            for (bytes, root, columns) in [
                (source.as_slice(), "pkg", SourceColumns::default()),
                (relocated.bytes.as_slice(), "ns/ns/pkg", relocated.columns),
            ] {
                let dir = tempdir()?;
                let deps = dir.path().join(root);
                fs::create_dir_all(&deps)?;
                fs::write(deps.join("wire.x"), "pub struct Header { n: u8 } pub type H = Header; pub fn plus(x:u8) -> u8 { x + u8:2 }")?;
                fs::write(deps.join("aux.x"), "pub struct Token { n: u8 } pub type T = Token; pub fn bias() -> u8 { u8:4 }")?;
                fs::write(dir.path().join("design.x"), bytes)?;
                let mut data =
                    ImportData::new(Some(xlsynth::default_dslx_stdlib_path()), &[dir.path()]);
                let checked = parse_and_typecheck(
                    "import design; fn main(x:u8) -> u8 { design::calc(x) }",
                    "caller.x",
                    "caller",
                    &mut data,
                )?;
                let ir = checked.to_ir_text()?;
                let package = ir_parser::Parser::new(&ir)
                    .parse_package()
                    .map_err(|e| e.to_string())?;
                let calc = package.get_fn("__design__calc").ok_or("missing calc")?;
                assert_eq!(
                    eval_fn_in_package_value(&package, calc, &[IrValue::make_ubits(8, 37)?]),
                    IrValue::make_ubits(8, 42)?
                );
                let coords: Vec<_> = calc
                    .nodes
                    .iter()
                    .filter_map(|n| n.pos.as_ref())
                    .flatten()
                    .map(|p| (p.lineno, columns.original_column(p.lineno, p.colno)))
                    .collect();
                assert!(!coords.is_empty());
                positions.push(coords);
            }
            assert_eq!(positions[0], positions[1]);
        }
        Ok(())
    }

    #[test]
    fn legacy_literal_bytes_match_native_file_imports() -> Result<(), Box<dyn std::error::Error>> {
        let original = b"// discarded \xff\0\n\
pub\xa0fn bytes() -> (u8, u8, u8, u8) {\n\
    let s' = \"\xff\0\xa0\"; (s'[u32:0], s'[u32:1], s'[u32:2], '\xff')\n\
}";
        let adapted = decode_dslx_source(original)?;
        assert_eq!(
            parse_dslx_outline(&adapted)?.functions["bytes"].syntax,
            "(u8, u8, u8, u8)"
        );
        let dir = tempdir()?;
        let expected = IrValue::parse_typed("(bits[8]:255, bits[8]:0, bits[8]:160, bits[8]:255)")?;
        for (module, source) in [
            ("raw", original.as_slice()),
            ("adapted", adapted.as_bytes()),
        ] {
            fs::write(dir.path().join(format!("{module}.x")), source)?;
            let mut data =
                ImportData::new(Some(xlsynth::default_dslx_stdlib_path()), &[dir.path()]);
            let checked = parse_and_typecheck(
                &format!("import {module}; fn main() -> (u8, u8, u8, u8) {{ {module}::bytes() }}"),
                "caller.x",
                "caller",
                &mut data,
            )?;
            let ir = checked.to_ir_text()?;
            let package = ir_parser::Parser::new(&ir)
                .parse_package()
                .map_err(|e| e.to_string())?;
            let main = package.get_fn("__caller__main").ok_or("missing caller")?;
            assert_eq!(eval_fn_in_package_value(&package, main, &[]), expected);
        }
        Ok(())
    }
}
