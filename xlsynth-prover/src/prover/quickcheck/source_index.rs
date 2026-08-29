// SPDX-License-Identifier: Apache-2.0

//! Best-effort identifier locations for diagnostics, not a DSLX parser. XLS
//! discovery is authoritative; callers only look up names discovered by XLS.

use std::collections::HashMap;
use std::ops::Range;

/// Finds unique top-level function names, ignoring comments and literals.
/// Unbalanced delimiters or unterminated literals invalidate the index rather
/// than risking an incorrect highlight. All ranges are UTF-8 byte offsets.
pub(super) fn function_name_spans(source: &str) -> HashMap<String, Range<usize>> {
    let mut names = HashMap::new();
    let mut nesting = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut after_fn = false;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(b"//") {
            i += bytes[i..]
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(bytes.len() - i);
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'!' | b'\''))
            {
                i += 1;
            }
            let name = &source[start..i];
            if after_fn {
                // Duplicate declarations are ambiguous, even if one would
                // later be rejected by the real parser.
                names
                    .entry(name.to_owned())
                    .and_modify(|span| *span = None)
                    .or_insert(Some(start..i));
            }
            after_fn = nesting.is_empty() && name == "fn";
            continue;
        }
        after_fn = false;
        let char_literal = c == b'\''
            && (bytes.get(i + 1).is_none_or(|&b| b == b'\\')
                || bytes.get(i + 2).is_none_or(|&b| b == b'\''));
        if matches!(c, b'"' | b'`') || char_literal {
            let quote = c;
            i += 1;
            loop {
                match bytes.get(i) {
                    Some(&b) if b == quote => {
                        i += 1;
                        break;
                    }
                    Some(b'\\') => i += 2,
                    Some(_) => i += 1,
                    None => return HashMap::new(),
                }
            }
            continue;
        }
        match c {
            b'(' => nesting.push(b')'),
            b'[' => nesting.push(b']'),
            b'{' => nesting.push(b'}'),
            b')' | b']' | b'}' => {
                if nesting.pop() != Some(c) {
                    return HashMap::new();
                }
            }
            _ => {
                // Other punctuation (including a non-literal apostrophe) and
                // non-ASCII bytes cannot introduce an ASCII DSLX fn keyword.
            }
        }
        i += 1;
    }
    if !nesting.is_empty() {
        return HashMap::new();
    }
    names
        .into_iter()
        .filter_map(|(name, span)| span.map(|span| (name, span)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_literals_comments_and_nested_names() {
        let source = r#"
// Unicode λ and fn fake_comment() {}
const S = "escaped \" fn fake_string() { }";
const T = `fn fake_backtick() { \` }`;
const C = '{';
const E = '\'';
fn helper(x: u8) -> u8 { let x' = x; x' }
#[quickcheck]
pub fn // fn fake_between() {}
  actual(
    x: u8,
  ) -> bool { trace_fmt!("fn fake_body() {}", x); true }
fn other() {} fn last'() {}
impl Record { fn nested() {} }
"#;
        let spans = function_name_spans(source);
        let mut names: Vec<_> = spans.keys().map(String::as_str).collect();
        names.sort();
        assert_eq!(names, ["actual", "helper", "last'", "other"]);
        for (name, span) in spans {
            assert_eq!(&source[span], name);
        }
    }

    #[test]
    fn offsets_are_bytes_and_preserve_crlf() {
        let source = "// λ🌍\r\n#[quickcheck]\r\nfn check(x: u8) -> bool { true }\r\n";
        let span = function_name_spans(source).remove("check").unwrap();
        assert_eq!(
            span,
            source.find("check(x").unwrap()..source.find("check(x").unwrap() + 5
        );
        assert_eq!(&source[span], "check");
    }

    #[test]
    fn uncertain_locations_are_not_reported() {
        for source in [
            "fn duplicate() {} fn duplicate() {}",
            "fn unbalanced() {",
            "fn mismatched(] {}",
            "fn before_string() {} const S = \"unterminated",
            "fn before_backtick() {} const S = `unterminated",
        ] {
            assert!(function_name_spans(source).is_empty(), "{source}");
        }
    }
}
