// SPDX-License-Identifier: Apache-2.0

use flate2::read::MultiGzDecoder;
use regex::Regex;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::liberty_parser::{Block, BlockMember, Value};
use super::{CharReader, LibertyParser};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Child,
    Descendant,
}

#[derive(Debug, Clone)]
pub struct QueryStep {
    pub axis: Axis,
    pub block_type: Option<String>, // None means wildcard '*'
    pub qual0_eq: Option<String>,
    pub qual0_matches: Option<Regex>,
}

#[derive(Debug, Clone)]
pub struct QueryMatch<'a> {
    pub block: &'a Block,
    pub path: String,
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Identifier(s) => s.clone(),
        Value::Number(n) => format!("{n}"),
        Value::Tuple(xs) => {
            let parts: Vec<String> = xs.iter().map(|x| value_to_string(x.as_ref())).collect();
            format!("({})", parts.join(","))
        }
    }
}

fn block_label(b: &Block) -> String {
    if let Some(q0) = b.qualifiers.first() {
        format!("{}({})", b.block_type, value_to_string(q0))
    } else {
        b.block_type.clone()
    }
}

fn child_subblocks(block: &Block) -> impl Iterator<Item = &Block> {
    block.members.iter().filter_map(|m| match m {
        BlockMember::SubBlock(sb) => Some(sb.as_ref()),
        BlockMember::BlockAttr(_) => None,
    })
}

fn block_matches(block: &Block, step: &QueryStep) -> bool {
    if let Some(bt) = &step.block_type {
        if &block.block_type != bt {
            return false;
        }
    }
    if let Some(want) = &step.qual0_eq {
        let got = block.qualifiers.first().map(value_to_string);
        return got.as_ref() == Some(want);
    }
    if let Some(regex) = &step.qual0_matches {
        if let Some(got) = block.qualifiers.first().map(value_to_string) {
            return regex.is_match(&got);
        }
        return false;
    }
    true
}

fn collect_descendants<'a>(
    root: &'a Block,
    include_self: bool,
    out: &mut Vec<(&'a Block, String)>,
    path: &str,
) {
    if include_self {
        out.push((root, path.to_string()));
    }
    for child in child_subblocks(root) {
        let child_path = format!("{}/{}", path, block_label(child));
        collect_descendants(child, true, out, &child_path);
    }
}

fn slice_query(input: &str, start: usize, end: usize, context: &str) -> Result<String, String> {
    input
        .get(start..end)
        .map(str::to_string)
        .ok_or_else(|| format!("Invalid UTF-8 boundaries while parsing {context}"))
}

fn parse_quoted_or_bare(input: &str, i: &mut usize) -> Result<String, String> {
    let bytes = input.as_bytes();
    if *i >= bytes.len() {
        return Err("Unexpected end of query while parsing predicate value".to_string());
    }
    let c = bytes[*i];
    if c == b'"' || c == b'\'' {
        let quote = c;
        *i += 1;
        let start = *i;
        while *i < bytes.len() && bytes[*i] != quote {
            *i += 1;
        }
        if *i >= bytes.len() {
            return Err("Unterminated quoted predicate value".to_string());
        }
        let out = slice_query(input, start, *i, "quoted predicate value")?;
        *i += 1; // closing quote
        Ok(out)
    } else {
        let start = *i;
        while *i < bytes.len() {
            if bytes[*i] == b']' || bytes[*i].is_ascii_whitespace() {
                break;
            }
            *i += 1;
        }
        if *i == start {
            return Err("Empty predicate value".to_string());
        }
        slice_query(input, start, *i, "bare predicate value")
    }
}

fn skip_ws(input: &str, i: &mut usize) {
    let bytes = input.as_bytes();
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

pub fn parse_query(query: &str) -> Result<Vec<QueryStep>, String> {
    let bytes = query.as_bytes();
    let mut i = 0usize;
    let mut steps = Vec::new();

    while i < bytes.len() {
        let axis = if bytes[i] == b'/' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                i += 2;
                Axis::Descendant
            } else {
                i += 1;
                Axis::Child
            }
        } else if steps.is_empty() {
            // Allow omitted leading slash as convenience: treat as descendant.
            Axis::Descendant
        } else {
            return Err(format!(
                "Expected '/' or '//' before next query step at byte offset {i}"
            ));
        };

        skip_ws(query, &mut i);
        if i >= bytes.len() {
            break;
        }

        // Parse block token: '*' or identifier-ish.
        let block_type = if bytes[i] == b'*' {
            i += 1;
            None
        } else {
            let start = i;
            while i < bytes.len() {
                if bytes[i] == b'/' || bytes[i] == b'[' || bytes[i].is_ascii_whitespace() {
                    break;
                }
                i += 1;
            }
            if i == start {
                return Err(format!("Expected block token at byte offset {start}"));
            }
            Some(slice_query(query, start, i, "block token")?)
        };

        skip_ws(query, &mut i);
        let mut qual0_eq = None;
        let mut qual0_matches = None;
        if i < bytes.len() && bytes[i] == b'[' {
            i += 1;
            skip_ws(query, &mut i);

            // Supported predicate syntax:
            //   [qual0='foo']
            //   [qual0=foo]
            //   [matches(qual0, 'regex')]
            if bytes[i..].starts_with(b"qual0") {
                i += "qual0".len();
                skip_ws(query, &mut i);
                if i >= bytes.len() || bytes[i] != b'=' {
                    return Err("Expected '=' in predicate".to_string());
                }
                i += 1;
                skip_ws(query, &mut i);
                qual0_eq = Some(parse_quoted_or_bare(query, &mut i)?);
                skip_ws(query, &mut i);
            } else if bytes[i..].starts_with(b"matches") {
                i += "matches".len();
                skip_ws(query, &mut i);
                if i >= bytes.len() || bytes[i] != b'(' {
                    return Err("Expected '(' after matches".to_string());
                }
                i += 1;
                skip_ws(query, &mut i);
                if !bytes[i..].starts_with(b"qual0") {
                    return Err("Only matches(qual0, <regex>) is currently supported".to_string());
                }
                i += "qual0".len();
                skip_ws(query, &mut i);
                if i >= bytes.len() || bytes[i] != b',' {
                    return Err("Expected ',' in matches(...) predicate".to_string());
                }
                i += 1;
                skip_ws(query, &mut i);
                let pattern = parse_quoted_or_bare(query, &mut i)?;
                let compiled = Regex::new(&pattern)
                    .map_err(|e| format!("Invalid regex in matches(...): {e}"))?;
                skip_ws(query, &mut i);
                if i >= bytes.len() || bytes[i] != b')' {
                    return Err("Expected ')' to close matches(...) predicate".to_string());
                }
                i += 1;
                skip_ws(query, &mut i);
                qual0_matches = Some(compiled);
            } else {
                return Err(
                    "Only predicates [qual0=...] and [matches(qual0, ...)] are currently supported"
                        .to_string(),
                );
            }
            if i >= bytes.len() || bytes[i] != b']' {
                return Err("Expected closing ']' for predicate".to_string());
            }
            i += 1;
        }

        steps.push(QueryStep {
            axis,
            block_type,
            qual0_eq,
            qual0_matches,
        });
    }

    if steps.is_empty() {
        return Err("Empty query".to_string());
    }
    Ok(steps)
}

pub fn run_query<'a>(root: &'a Block, steps: &[QueryStep]) -> Vec<QueryMatch<'a>> {
    let root_path = format!("/{}", block_label(root));
    let mut current: Vec<QueryMatch<'a>> = vec![QueryMatch {
        block: root,
        path: root_path,
    }];

    for (step_index, step) in steps.iter().enumerate() {
        let mut next = Vec::new();
        for item in &current {
            match step.axis {
                Axis::Child => {
                    // First child-step can match the root itself for convenience:
                    // /library/cell...
                    if step_index == 0 && block_matches(item.block, step) {
                        next.push(QueryMatch {
                            block: item.block,
                            path: item.path.clone(),
                        });
                    }
                    for child in child_subblocks(item.block) {
                        if block_matches(child, step) {
                            next.push(QueryMatch {
                                block: child,
                                path: format!("{}/{}", item.path, block_label(child)),
                            });
                        }
                    }
                }
                Axis::Descendant => {
                    let mut all = Vec::new();
                    let include_self = step_index == 0;
                    collect_descendants(item.block, include_self, &mut all, &item.path);
                    for (cand, path) in all {
                        if block_matches(cand, step) {
                            next.push(QueryMatch { block: cand, path });
                        }
                    }
                }
            }
        }
        current = next;
        if current.is_empty() {
            break;
        }
    }

    current
}

pub fn parse_liberty_file_to_ast(path: &Path) -> Result<Block, String> {
    let file = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let is_gz = path.extension().map(|x| x == "gz").unwrap_or(false);
    let reader: Box<dyn Read> = if is_gz {
        Box::new(MultiGzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(file)
    };
    let char_reader = CharReader::new(reader);
    let mut parser = LibertyParser::new_from_iter(char_reader);
    parser.parse().map_err(|e| format!("parse error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_block(block_type: &str, qual0: Option<&str>, children: Vec<Block>) -> Block {
        let mut qualifiers = Vec::new();
        if let Some(q) = qual0 {
            qualifiers.push(Value::Identifier(q.to_string()));
        }
        let members = children
            .into_iter()
            .map(|c| BlockMember::SubBlock(Box::new(c)))
            .collect();
        Block {
            block_type: block_type.to_string(),
            qualifiers,
            members,
        }
    }

    #[test]
    fn test_parse_query_basic() {
        let q = "//cell[qual0='FOO']//timing";
        let steps = parse_query(q).expect("query should parse");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].axis, Axis::Descendant);
        assert_eq!(steps[0].block_type.as_deref(), Some("cell"));
        assert_eq!(steps[0].qual0_eq.as_deref(), Some("FOO"));
        assert!(steps[0].qual0_matches.is_none());
        assert_eq!(steps[1].axis, Axis::Descendant);
        assert_eq!(steps[1].block_type.as_deref(), Some("timing"));
    }

    #[test]
    fn test_parse_query_matches_qual0_regex() {
        let q = "//cell[matches(qual0, '^A.*$')]//timing";
        let steps = parse_query(q).expect("query should parse");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].axis, Axis::Descendant);
        assert_eq!(steps[0].block_type.as_deref(), Some("cell"));
        assert!(steps[0].qual0_eq.is_none());
        assert!(
            steps[0]
                .qual0_matches
                .as_ref()
                .expect("regex should parse")
                .is_match("ABC")
        );
        assert!(
            !steps[0]
                .qual0_matches
                .as_ref()
                .expect("regex should parse")
                .is_match("XYZ")
        );
    }

    #[test]
    fn test_parse_query_non_ascii_whitespace_does_not_panic() {
        let q = "//cell\u{00A0}//timing";
        let parse_attempt = std::panic::catch_unwind(|| parse_query(q));
        assert!(parse_attempt.is_ok(), "parser should not panic");
        let steps = parse_attempt
            .expect("parser should return a parse result")
            .expect("query should parse");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].axis, Axis::Descendant);
        assert_eq!(steps[0].block_type.as_deref(), Some("cell\u{00A0}"));
        assert_eq!(steps[1].axis, Axis::Descendant);
        assert_eq!(steps[1].block_type.as_deref(), Some("timing"));
    }

    #[test]
    fn test_run_query_matches_descendants() {
        let ast = mk_block(
            "library",
            Some("L"),
            vec![
                mk_block(
                    "cell",
                    Some("A"),
                    vec![mk_block(
                        "pin",
                        Some("Y"),
                        vec![mk_block("timing", None, vec![])],
                    )],
                ),
                mk_block("cell", Some("B"), vec![]),
            ],
        );
        let steps = parse_query("//cell[qual0='A']//timing").unwrap();
        let out = run_query(&ast, &steps);
        assert_eq!(out.len(), 1);
        assert!(out[0].path.contains("/library(") || out[0].path.starts_with("/library"));
        assert_eq!(out[0].block.block_type, "timing");
    }

    #[test]
    fn test_run_query_matches_regex_predicate() {
        let ast = mk_block(
            "library",
            Some("L"),
            vec![
                mk_block(
                    "cell",
                    Some("INV"),
                    vec![mk_block(
                        "pin",
                        Some("Y"),
                        vec![mk_block("timing", None, vec![])],
                    )],
                ),
                mk_block(
                    "cell",
                    Some("NAND2"),
                    vec![mk_block(
                        "pin",
                        Some("Y"),
                        vec![mk_block("timing", None, vec![])],
                    )],
                ),
            ],
        );
        let steps = parse_query("//cell[matches(qual0, '^INV$')]//timing").unwrap();
        let out = run_query(&ast, &steps);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].block.block_type, "timing");
        assert!(out[0].path.contains("INV"));
    }
}
