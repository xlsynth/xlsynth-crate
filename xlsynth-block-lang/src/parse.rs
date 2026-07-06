// SPDX-License-Identifier: Apache-2.0

//! Parser for the small amount of syntax that is structural at block level.

use std::collections::BTreeSet;
use std::path::Path;

use crate::BlockDiagnostic;

#[derive(Debug, Clone)]
pub(crate) struct Module {
    /// Ordinary DSLX declarations with blocks and procs removed. This is used
    /// for expression-helper conversion because PIR does not parse proc IR.
    pub prelude: String,
    /// Ordinary DSLX declarations and procs with only blocks removed. This is
    /// used when a proc target must be converted by official XLS codegen.
    pub proc_prelude: String,
    pub proc_names: BTreeSet<String>,
    pub blocks: Vec<BlockDecl>,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockDecl {
    pub name: String,
    pub is_public: bool,
    pub params: Vec<ParamDecl>,
    pub ports: Vec<PortDecl>,
    pub items: Vec<BlockItem>,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ParamDecl {
    pub name: String,
    pub ty: String,
    pub default: Option<String>,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub(crate) struct PortDecl {
    pub direction: PortDirection,
    pub name: String,
    pub ty: String,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum BlockItem {
    Let {
        names: Vec<String>,
        statement: String,
        expression: String,
        expression_offset: usize,
        offset: usize,
    },
    Const {
        name: String,
        statement: String,
        expression: String,
        expression_offset: usize,
        offset: usize,
    },
    ForwardRegister {
        name: String,
        ty: String,
        offset: usize,
    },
    Register(RegisterDecl),
    Assign {
        target: String,
        expression: String,
        expression_offset: usize,
        offset: usize,
    },
    Instance(InstanceDecl),
    Assert {
        predicate: String,
        predicate_offset: usize,
        label: String,
    },
    Cover {
        predicate: String,
        predicate_offset: usize,
        label: String,
    },
    ConstAssert {
        predicate: String,
        predicate_offset: usize,
    },
    Conditional {
        condition: String,
        condition_offset: usize,
        then_items: Vec<BlockItem>,
        else_items: Vec<BlockItem>,
        offset: usize,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct RegisterDecl {
    pub name: String,
    pub ty: Option<String>,
    pub ty_offset: Option<usize>,
    pub init_value: Option<String>,
    pub init_value_offset: Option<usize>,
    pub enable: Option<String>,
    pub enable_offset: Option<usize>,
    pub next: String,
    pub next_offset: usize,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct InstanceDecl {
    pub name: String,
    pub target: String,
    pub parametrics: Option<String>,
    pub bindings: Vec<InstanceBinding>,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct InstanceBinding {
    pub port: String,
    pub expression: String,
    pub expression_offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Token {
    pub(crate) text: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Parses top-level block declarations while retaining all ordinary DSLX text.
pub(crate) fn parse_module(source: &str, path: &Path) -> Result<Module, BlockDiagnostic> {
    let tokens = lex(source, path)?;
    let mut blocks = Vec::new();
    let mut block_ranges = Vec::new();
    let mut proc_ranges = Vec::new();
    let mut proc_names = BTreeSet::new();
    let mut i = 0;
    let mut brace_depth = 0usize;
    while i < tokens.len() {
        let is_public_block = brace_depth == 0
            && tokens[i].text == "pub"
            && tokens.get(i + 1).is_some_and(|token| token.text == "block");
        let is_private_block = brace_depth == 0 && tokens[i].text == "block";
        if is_public_block || is_private_block {
            let start_i = i;
            let range_start_i = outer_attribute_start(&tokens, start_i);
            let (block, next_i) = parse_block(source, &tokens, i, is_public_block, path)?;
            let end = tokens[next_i - 1].end;
            block_ranges.push((tokens[range_start_i].start, end));
            blocks.push(block);
            i = next_i;
            continue;
        }
        let is_public_proc = brace_depth == 0
            && tokens[i].text == "pub"
            && tokens.get(i + 1).is_some_and(|token| token.text == "proc");
        let is_private_proc = brace_depth == 0 && tokens[i].text == "proc";
        if is_public_proc || is_private_proc {
            let start_i = i;
            let range_start_i = outer_attribute_start(&tokens, start_i);
            let proc_i = if is_public_proc { i + 1 } else { i };
            let name = identifier(&tokens, proc_i + 1, "proc name", path)?;
            let open = tokens
                .iter()
                .enumerate()
                .skip(proc_i + 2)
                .find(|(_, token)| token.text == "{")
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    BlockDiagnostic::new(
                        path,
                        Some(tokens[proc_i].start),
                        format!("proc '{name}' has no body"),
                    )
                })?;
            let close = matching_delimiter(&tokens, open, "{", "}", path)?;
            proc_ranges.push((tokens[range_start_i].start, tokens[close].end));
            proc_names.insert(name);
            i = close + 1;
            continue;
        }
        match tokens[i].text.as_str() {
            "{" => brace_depth += 1,
            "}" => brace_depth = brace_depth.saturating_sub(1),
            _ => {}
        }
        i += 1;
    }

    if blocks.is_empty() {
        return Err(BlockDiagnostic::new(
            path,
            None,
            "source contains no block declaration",
        ));
    }

    let proc_prelude = remove_ranges(source, &block_ranges);
    let mut all_removed = block_ranges;
    all_removed.extend(proc_ranges);
    all_removed.sort_by_key(|range| range.0);
    let prelude = remove_ranges(source, &all_removed);

    Ok(Module {
        prelude,
        proc_prelude,
        proc_names,
        blocks,
    })
}

/// Returns the first token in the contiguous `#[...]` attribute sequence that
/// decorates the item beginning at `item_i`.
fn outer_attribute_start(tokens: &[Token], item_i: usize) -> usize {
    let mut start = item_i;
    loop {
        if start == 0 || tokens[start - 1].text != "]" {
            break;
        }
        let mut depth = 1usize;
        let mut cursor = start - 1;
        let mut open = None;
        while cursor != 0 {
            cursor -= 1;
            match tokens[cursor].text.as_str() {
                "]" => depth += 1,
                "[" => {
                    depth -= 1;
                    if depth == 0 {
                        open = Some(cursor);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(open) = open else {
            break;
        };
        if open == 0 || tokens[open - 1].text != "#" {
            break;
        }
        start = open - 1;
    }
    start
}

fn remove_ranges(source: &str, ranges: &[(usize, usize)]) -> String {
    let mut prelude = String::new();
    let mut previous = 0usize;
    for (start, end) in ranges {
        prelude.push_str(&source[previous..*start]);
        prelude.push('\n');
        previous = *end;
    }
    prelude.push_str(&source[previous..]);
    prelude
}

fn parse_block(
    source: &str,
    tokens: &[Token],
    mut i: usize,
    is_public: bool,
    path: &Path,
) -> Result<(BlockDecl, usize), BlockDiagnostic> {
    let offset = tokens[i].start;
    if is_public {
        i += 1;
    }
    expect(tokens, i, "block", path)?;
    i += 1;
    let name = identifier(tokens, i, "block name", path)?;
    i += 1;

    let mut params = Vec::new();
    if tokens.get(i).is_some_and(|token| token.text == "<") {
        let close = matching_angle_delimiter(tokens, i, "(", path)?;
        for range in split_top_level_with_angles(tokens, i + 1, close, ",") {
            if range.0 == range.1 {
                continue;
            }
            params.push(parse_param(source, tokens, range, path)?);
        }
        i = close + 1;
    }

    expect(tokens, i, "(", path)?;
    let ports_close = matching_delimiter(tokens, i, "(", ")", path)?;
    let mut ports = Vec::new();
    for range in split_top_level_with_angles(tokens, i + 1, ports_close, ",") {
        if range.0 == range.1 {
            continue;
        }
        ports.push(parse_port(source, tokens, range, path)?);
    }
    i = ports_close + 1;

    expect(tokens, i, "{", path)?;
    let body_close = matching_delimiter(tokens, i, "{", "}", path)?;
    let items = parse_body(source, tokens, i + 1, body_close, path)?;
    Ok((
        BlockDecl {
            name,
            is_public,
            params,
            ports,
            items,
            offset,
        },
        body_close + 1,
    ))
}

fn parse_param(
    source: &str,
    tokens: &[Token],
    range: (usize, usize),
    path: &Path,
) -> Result<ParamDecl, BlockDiagnostic> {
    let colon = find_top_level(tokens, range.0, range.1, ":").ok_or_else(|| {
        BlockDiagnostic::new(
            path,
            Some(tokens[range.0].start),
            "parametric binding needs ':'",
        )
    })?;
    let name = identifier(tokens, range.0, "parametric binding name", path)?;
    let equals = find_top_level(tokens, colon + 1, range.1, "=");
    let ty_end = equals.unwrap_or(range.1);
    let ty = token_slice(source, tokens, colon + 1, ty_end)
        .trim()
        .to_string();
    let default = equals.map(|eq| {
        strip_outer_braces(token_slice(source, tokens, eq + 1, range.1).trim()).to_string()
    });
    Ok(ParamDecl {
        name,
        ty,
        default,
        offset: tokens[range.0].start,
    })
}

fn parse_port(
    source: &str,
    tokens: &[Token],
    range: (usize, usize),
    path: &Path,
) -> Result<PortDecl, BlockDiagnostic> {
    let direction = match tokens[range.0].text.as_str() {
        "input" => PortDirection::Input,
        "output" => PortDirection::Output,
        other => {
            return Err(BlockDiagnostic::new(
                path,
                Some(tokens[range.0].start),
                format!("expected input or output port, got '{other}'"),
            ));
        }
    };
    let name = identifier(tokens, range.0 + 1, "port name", path)?;
    expect(tokens, range.0 + 2, ":", path)?;
    let ty = token_slice(source, tokens, range.0 + 3, range.1)
        .trim()
        .to_string();
    if ty.is_empty() {
        return Err(BlockDiagnostic::new(
            path,
            Some(tokens[range.0].start),
            format!("port '{name}' has no type"),
        ));
    }
    Ok(PortDecl {
        direction,
        name,
        ty,
        offset: tokens[range.0].start,
    })
}

fn parse_body(
    source: &str,
    tokens: &[Token],
    mut i: usize,
    end: usize,
    path: &Path,
) -> Result<Vec<BlockItem>, BlockDiagnostic> {
    let mut items = Vec::new();
    while i < end {
        let offset = tokens[i].start;
        match tokens[i].text.as_str() {
            "let" | "const" => {
                let is_const = tokens[i].text == "const";
                let semi = find_statement_end(tokens, i, end, path)?;
                let equals = find_top_level(tokens, i, semi, "=").ok_or_else(|| {
                    BlockDiagnostic::new(path, Some(offset), "binding needs an equals sign")
                })?;
                let statement = source[tokens[i].start..tokens[semi].end].to_string();
                let expression = token_slice(source, tokens, equals + 1, semi)
                    .trim()
                    .to_string();
                let expression_offset = tokens
                    .get(equals + 1)
                    .map_or(tokens[semi].start, |token| token.start);
                if is_const {
                    let name = identifier(tokens, i + 1, "binding name", path)?;
                    items.push(BlockItem::Const {
                        name,
                        statement,
                        expression,
                        expression_offset,
                        offset,
                    });
                } else {
                    let names = let_pattern_names(tokens, i + 1, semi, path)?;
                    items.push(BlockItem::Let {
                        names,
                        statement,
                        expression,
                        expression_offset,
                        offset,
                    });
                }
                i = semi + 1;
            }
            "declreg" => {
                let semi = find_statement_end(tokens, i, end, path)?;
                let name = identifier(tokens, i + 1, "register name", path)?;
                expect(tokens, i + 2, ":", path)?;
                let ty = token_slice(source, tokens, i + 3, semi).trim().to_string();
                items.push(BlockItem::ForwardRegister { name, ty, offset });
                i = semi + 1;
            }
            "reg" => {
                let (register, next_i) = parse_register(source, tokens, i, end, path)?;
                items.push(BlockItem::Register(register));
                i = next_i;
            }
            "assign" => {
                let semi = find_statement_end(tokens, i, end, path)?;
                let target = identifier(tokens, i + 1, "output assignment target", path)?;
                expect(tokens, i + 2, "=", path)?;
                let expression = token_slice(source, tokens, i + 3, semi).trim().to_string();
                items.push(BlockItem::Assign {
                    target,
                    expression,
                    expression_offset: tokens[i + 3].start,
                    offset,
                });
                i = semi + 1;
            }
            "inst" => {
                let (instance, next_i) = parse_instance(source, tokens, i, end, path)?;
                items.push(BlockItem::Instance(instance));
                i = next_i;
            }
            "assert" | "cover" => {
                let is_assert = tokens[i].text == "assert";
                expect(tokens, i + 1, "!", path)?;
                expect(tokens, i + 2, "(", path)?;
                let close = matching_delimiter(tokens, i + 2, "(", ")", path)?;
                let args = split_top_level(tokens, i + 3, close, ",");
                if args.len() != 2 {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(offset),
                        "assert!/cover! requires a predicate and string label",
                    ));
                }
                let predicate = token_slice(source, tokens, args[0].0, args[0].1)
                    .trim()
                    .to_string();
                let predicate_offset = tokens[args[0].0].start;
                if args[1].1.saturating_sub(args[1].0) != 1 {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(tokens[args[1].0].start),
                        "assert!/cover! label must be exactly one string literal token",
                    ));
                }
                let label_token = &tokens[args[1].0];
                let label = parse_string_literal(&label_token.text).map_err(|message| {
                    BlockDiagnostic::new(path, Some(label_token.start), message)
                })?;
                let mut next_i = close + 1;
                if tokens.get(next_i).is_some_and(|token| token.text == ";") {
                    next_i += 1;
                }
                if is_assert {
                    items.push(BlockItem::Assert {
                        predicate,
                        predicate_offset,
                        label,
                    });
                } else {
                    items.push(BlockItem::Cover {
                        predicate,
                        predicate_offset,
                        label,
                    });
                }
                i = next_i;
            }
            "const_assert" => {
                expect(tokens, i + 1, "!", path)?;
                expect(tokens, i + 2, "(", path)?;
                let close = matching_delimiter(tokens, i + 2, "(", ")", path)?;
                let predicate = token_slice(source, tokens, i + 3, close).trim().to_string();
                let predicate_offset = tokens[i + 3].start;
                let mut next_i = close + 1;
                if tokens.get(next_i).is_some_and(|token| token.text == ";") {
                    next_i += 1;
                }
                items.push(BlockItem::ConstAssert {
                    predicate,
                    predicate_offset,
                });
                i = next_i;
            }
            "if" => {
                let open = find_top_level(tokens, i + 1, end, "{").ok_or_else(|| {
                    BlockDiagnostic::new(path, Some(offset), "structural if has no body")
                })?;
                let close = matching_delimiter(tokens, open, "{", "}", path)?;
                let condition = token_slice(source, tokens, i + 1, open).trim().to_string();
                let then_items = parse_body(source, tokens, open + 1, close, path)?;
                let mut next_i = close + 1;
                let else_items = if tokens.get(next_i).is_some_and(|token| token.text == "else") {
                    expect(tokens, next_i + 1, "{", path)?;
                    let else_close = matching_delimiter(tokens, next_i + 1, "{", "}", path)?;
                    let result = parse_body(source, tokens, next_i + 2, else_close, path)?;
                    next_i = else_close + 1;
                    result
                } else {
                    Vec::new()
                };
                items.push(BlockItem::Conditional {
                    condition,
                    condition_offset: tokens[i + 1].start,
                    then_items,
                    else_items,
                    offset,
                });
                i = next_i;
            }
            other => {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(offset),
                    format!("unsupported block-level item starting with '{other}'"),
                ));
            }
        }
    }
    Ok(items)
}

fn let_pattern_names(
    tokens: &[Token],
    start: usize,
    end: usize,
    path: &Path,
) -> Result<Vec<String>, BlockDiagnostic> {
    let equals = find_top_level(tokens, start, end, "=").ok_or_else(|| {
        BlockDiagnostic::new(
            path,
            Some(tokens[start].start),
            "let binding is missing '='",
        )
    })?;
    let pattern_end = find_top_level(tokens, start, equals, ":").unwrap_or(equals);
    let names = tokens[start..pattern_end]
        .iter()
        .filter(|token| {
            token.text != "_"
                && token
                    .text
                    .as_bytes()
                    .first()
                    .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        })
        .map(|token| token.text.clone())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(BlockDiagnostic::new(
            path,
            Some(tokens[start].start),
            "let pattern declares no symbols",
        ));
    }
    Ok(names)
}

fn parse_register(
    source: &str,
    tokens: &[Token],
    i: usize,
    end: usize,
    path: &Path,
) -> Result<(RegisterDecl, usize), BlockDiagnostic> {
    let offset = tokens[i].start;
    let name = identifier(tokens, i + 1, "register name", path)?;
    let mut cursor = i + 2;
    let (ty, ty_offset) = if tokens.get(cursor).is_some_and(|token| token.text == ":") {
        cursor += 1;
        let open = find_top_level(tokens, cursor, end, "{").ok_or_else(|| {
            BlockDiagnostic::new(
                path,
                Some(offset),
                "register declaration needs a contract body",
            )
        })?;
        let ty_offset = tokens.get(cursor).map(|token| token.start);
        let ty = token_slice(source, tokens, cursor, open).trim().to_string();
        cursor = open;
        (Some(ty), ty_offset)
    } else {
        (None, None)
    };
    expect(tokens, cursor, "{", path)?;
    let close = matching_delimiter(tokens, cursor, "{", "}", path)?;
    let mut init_value = None;
    let mut init_value_seen = false;
    let mut init_value_offset = None;
    let mut enable = None;
    let mut enable_offset = None;
    let mut next = None;
    let mut next_offset = None;
    for field in split_top_level(tokens, cursor + 1, close, ",") {
        if field.0 == field.1 {
            continue;
        }
        let field_name = identifier(tokens, field.0, "register field", path)?;
        expect(tokens, field.0 + 1, ":", path)?;
        let value = token_slice(source, tokens, field.0 + 2, field.1)
            .trim()
            .to_string();
        match field_name.as_str() {
            "init_value" => {
                if init_value_seen {
                    return Err(duplicate_field(path, tokens[field.0].start, "init_value"));
                }
                init_value_seen = true;
                init_value = (value != "none").then_some(value);
                init_value_offset = Some(tokens[field.0 + 2].start);
            }
            "en" => {
                if enable.replace(value).is_some() {
                    return Err(duplicate_field(path, tokens[field.0].start, "en"));
                }
                enable_offset = Some(tokens[field.0 + 2].start);
            }
            "next" => {
                if next.replace(value).is_some() {
                    return Err(duplicate_field(path, tokens[field.0].start, "next"));
                }
                next_offset = Some(tokens[field.0 + 2].start);
            }
            other => {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(tokens[field.0].start),
                    format!("unknown register field '{other}'"),
                ));
            }
        }
    }
    let next = next.ok_or_else(|| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("register '{name}' is missing required next field"),
        )
    })?;
    let next_offset = next_offset.expect("a parsed next expression has an offset");
    let next_i = close + 1;
    if tokens.get(next_i).is_some_and(|token| token.text == ";") {
        return Err(BlockDiagnostic::new(
            path,
            Some(tokens[next_i].start),
            "reg blocks do not end with a semicolon",
        ));
    }
    Ok((
        RegisterDecl {
            name,
            ty,
            ty_offset,
            init_value,
            init_value_offset,
            enable,
            enable_offset,
            next,
            next_offset,
            offset,
        },
        next_i,
    ))
}

fn parse_instance(
    source: &str,
    tokens: &[Token],
    i: usize,
    end: usize,
    path: &Path,
) -> Result<(InstanceDecl, usize), BlockDiagnostic> {
    let offset = tokens[i].start;
    let name = identifier(tokens, i + 1, "instance name", path)?;
    expect(tokens, i + 2, ":", path)?;
    let target = identifier(tokens, i + 3, "instance target", path)?;
    let mut cursor = i + 4;
    let parametrics = if tokens.get(cursor).is_some_and(|token| token.text == "<") {
        let close = matching_angle_delimiter(tokens, cursor, "{", path)?;
        let text = token_slice(source, tokens, cursor + 1, close)
            .trim()
            .to_string();
        cursor = close + 1;
        Some(text)
    } else {
        None
    };
    if cursor >= end {
        return Err(BlockDiagnostic::new(
            path,
            Some(offset),
            "instance declaration needs a binding body",
        ));
    }
    expect(tokens, cursor, "{", path)?;
    let close = matching_delimiter(tokens, cursor, "{", "}", path)?;
    let mut bindings = Vec::new();
    for field in split_top_level(tokens, cursor + 1, close, ",") {
        if field.0 == field.1 {
            continue;
        }
        let port = identifier(tokens, field.0, "instance input port", path)?;
        expect(tokens, field.0 + 1, ":", path)?;
        let value = token_slice(source, tokens, field.0 + 2, field.1)
            .trim()
            .to_string();
        if bindings
            .iter()
            .any(|binding: &InstanceBinding| binding.port == port)
        {
            return Err(BlockDiagnostic::new(
                path,
                Some(tokens[field.0].start),
                format!("instance input '{port}' is bound more than once"),
            ));
        }
        bindings.push(InstanceBinding {
            port,
            expression: value,
            expression_offset: tokens[field.0 + 2].start,
        });
    }
    let mut next_i = close + 1;
    if tokens.get(next_i).is_some_and(|token| token.text == ";") {
        next_i += 1;
    }
    Ok((
        InstanceDecl {
            name,
            target,
            parametrics,
            bindings,
            offset,
        },
        next_i,
    ))
}

fn duplicate_field(path: &Path, offset: usize, field: &str) -> BlockDiagnostic {
    BlockDiagnostic::new(
        path,
        Some(offset),
        format!("register field '{field}' is specified more than once"),
    )
}

pub(crate) fn lex(source: &str, path: &Path) -> Result<Vec<Token>, BlockDiagnostic> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if byte == b'/' && bytes.get(i + 1) == Some(&b'/') {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if byte == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let start = i;
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth != 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth != 0 {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(start),
                    "unterminated block comment",
                ));
            }
            continue;
        }
        let start = i;
        if byte == b'"' {
            i += 1;
            let mut escaped = false;
            while i < bytes.len() {
                if escaped {
                    escaped = false;
                } else if bytes[i] == b'\\' {
                    escaped = true;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            if bytes.get(i.saturating_sub(1)) != Some(&b'"') {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(start),
                    "unterminated string literal",
                ));
            }
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
        } else if byte.is_ascii_digit() {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
        } else {
            i += 1;
            if matches!(
                (byte, bytes.get(i).copied()),
                (b':', Some(b':'))
                    | (b'.', Some(b'.'))
                    | (b'=', Some(b'='))
                    | (b'!', Some(b'='))
                    | (b'<', Some(b'='))
                    | (b'>', Some(b'='))
                    | (b'&', Some(b'&'))
                    | (b'|', Some(b'|'))
                    | (b'<', Some(b'<'))
                    | (b'>', Some(b'>'))
                    | (b'+', Some(b':'))
                    | (b'-', Some(b'>'))
            ) {
                i += 1;
            }
        }
        tokens.push(Token {
            text: source[start..i].to_string(),
            start,
            end: i,
        });
    }
    Ok(tokens)
}

fn expect(tokens: &[Token], i: usize, expected: &str, path: &Path) -> Result<(), BlockDiagnostic> {
    if tokens.get(i).is_some_and(|token| token.text == expected) {
        Ok(())
    } else {
        let offset = tokens.get(i).map(|token| token.start);
        let found = tokens
            .get(i)
            .map(|token| token.text.as_str())
            .unwrap_or("end of file");
        Err(BlockDiagnostic::new(
            path,
            offset,
            format!("expected '{expected}', got '{found}'"),
        ))
    }
}

fn identifier(
    tokens: &[Token],
    i: usize,
    context: &str,
    path: &Path,
) -> Result<String, BlockDiagnostic> {
    let token = tokens.get(i).ok_or_else(|| {
        BlockDiagnostic::new(path, None, format!("expected {context}, got end of file"))
    })?;
    let valid = token
        .text
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && token
            .text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if valid {
        Ok(token.text.clone())
    } else {
        Err(BlockDiagnostic::new(
            path,
            Some(token.start),
            format!("expected {context}, got '{}'", token.text),
        ))
    }
}

fn matching_delimiter(
    tokens: &[Token],
    open_i: usize,
    open: &str,
    close: &str,
    path: &Path,
) -> Result<usize, BlockDiagnostic> {
    expect(tokens, open_i, open, path)?;
    let mut depth = 0usize;
    for (i, token) in tokens.iter().enumerate().skip(open_i) {
        if token.text == open {
            depth += 1;
        } else if token.text == close {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
    }
    Err(BlockDiagnostic::new(
        path,
        Some(tokens[open_i].start),
        format!("unterminated '{open}' group"),
    ))
}

/// Matches a syntactic generic/parametric angle group without treating an
/// ordinary comparison expression such as `u32:1 < u32:2` as nested angles.
fn matching_angle_delimiter(
    tokens: &[Token],
    open_i: usize,
    expected_after: &str,
    path: &Path,
) -> Result<usize, BlockDiagnostic> {
    expect(tokens, open_i, "<", path)?;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    for i in open_i + 1..tokens.len() {
        match tokens[i].text.as_str() {
            "(" => paren_depth += 1,
            ")" => paren_depth = paren_depth.saturating_sub(1),
            "{" => brace_depth += 1,
            "}" => brace_depth = brace_depth.saturating_sub(1),
            "[" => bracket_depth += 1,
            "]" => bracket_depth = bracket_depth.saturating_sub(1),
            _ => {}
        }
        if paren_depth != 0 || brace_depth != 0 || bracket_depth != 0 {
            continue;
        }
        // The outer close is the top-level `>` immediately followed by the
        // construct that owns this angle group. The lexer retains `>>` as a
        // shift token, so it cannot be mistaken for the outer delimiter.
        if tokens[i].text == ">"
            && tokens
                .get(i + 1)
                .is_some_and(|token| token.text == expected_after)
        {
            return Ok(i);
        }
    }
    Err(BlockDiagnostic::new(
        path,
        Some(tokens[open_i].start),
        "unterminated '<' group",
    ))
}

/// Returns true when `<` has the lexical shape of a DSLX generic application.
/// DSLX formatting places spaces around comparison operators, while generic
/// arguments are attached to their target (`f<T>` / `Type<N>`).
pub(crate) fn is_generic_open(tokens: &[Token], i: usize) -> bool {
    if tokens.get(i).is_none_or(|token| token.text != "<") || i == 0 {
        return false;
    }
    let previous = &tokens[i - 1];
    let current = &tokens[i];
    let valid_target = previous
        .text
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || matches!(previous.text.as_str(), "]" | ">");
    valid_target && (previous.end == current.start || has_spaced_generic_close_shape(tokens, i))
}

fn has_spaced_generic_close_shape(tokens: &[Token], open_i: usize) -> bool {
    let mut depth = 1usize;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    for i in open_i + 1..tokens.len() {
        match tokens[i].text.as_str() {
            "(" => paren_depth += 1,
            ")" => paren_depth = paren_depth.saturating_sub(1),
            "{" => brace_depth += 1,
            "}" => brace_depth = brace_depth.saturating_sub(1),
            "[" => bracket_depth += 1,
            "]" => bracket_depth = bracket_depth.saturating_sub(1),
            _ => {}
        }
        if paren_depth != 0 || brace_depth != 0 || bracket_depth != 0 {
            continue;
        }
        match tokens[i].text.as_str() {
            "<" => depth += 1,
            ">" => depth -= 1,
            ";" | "}" if depth == 1 => return false,
            _ => {}
        }
        if depth == 0 {
            return tokens.get(i + 1).is_none_or(|next| {
                matches!(
                    next.text.as_str(),
                    "(" | "{" | "[" | "]" | ")" | "," | ";" | ":" | "=" | ">"
                )
            });
        }
    }
    false
}

fn split_top_level(
    tokens: &[Token],
    start: usize,
    end: usize,
    separator: &str,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut item_start = start;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    let mut angle = 0usize;
    for i in start..end {
        match tokens[i].text.as_str() {
            "(" => paren += 1,
            ")" => paren = paren.saturating_sub(1),
            "{" => brace += 1,
            "}" => brace = brace.saturating_sub(1),
            "[" => bracket += 1,
            "]" => bracket = bracket.saturating_sub(1),
            "<" if is_generic_open(tokens, i) => angle += 1,
            ">" if angle != 0 => angle -= 1,
            text if text == separator && paren == 0 && brace == 0 && bracket == 0 && angle == 0 => {
                result.push((item_start, i));
                item_start = i + 1;
            }
            _ => {}
        }
    }
    result.push((item_start, end));
    result
}

fn split_top_level_with_angles(
    tokens: &[Token],
    start: usize,
    end: usize,
    separator: &str,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut item_start = start;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    let mut angle = 0usize;
    for i in start..end {
        match tokens[i].text.as_str() {
            "(" => paren += 1,
            ")" => paren = paren.saturating_sub(1),
            "{" => brace += 1,
            "}" => brace = brace.saturating_sub(1),
            "[" => bracket += 1,
            "]" => bracket = bracket.saturating_sub(1),
            "<" if is_generic_open(tokens, i) => angle += 1,
            ">" if angle != 0 => angle -= 1,
            text if text == separator && paren == 0 && brace == 0 && bracket == 0 && angle == 0 => {
                result.push((item_start, i));
                item_start = i + 1;
            }
            _ => {}
        }
    }
    result.push((item_start, end));
    result
}

fn find_top_level(tokens: &[Token], start: usize, end: usize, needle: &str) -> Option<usize> {
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
        if token.text == needle && paren == 0 && brace == 0 && bracket == 0 {
            return Some(i);
        }
        match token.text.as_str() {
            "(" => paren += 1,
            ")" => paren = paren.saturating_sub(1),
            "{" => brace += 1,
            "}" => brace = brace.saturating_sub(1),
            "[" => bracket += 1,
            "]" => bracket = bracket.saturating_sub(1),
            _ => {}
        }
    }
    None
}

fn find_statement_end(
    tokens: &[Token],
    start: usize,
    end: usize,
    path: &Path,
) -> Result<usize, BlockDiagnostic> {
    find_top_level(tokens, start, end, ";").ok_or_else(|| {
        BlockDiagnostic::new(
            path,
            Some(tokens[start].start),
            "statement is missing terminating semicolon",
        )
    })
}

fn token_slice<'a>(source: &'a str, tokens: &[Token], start: usize, end: usize) -> &'a str {
    if start == end {
        ""
    } else {
        &source[tokens[start].start..tokens[end - 1].end]
    }
}

fn strip_outer_braces(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        trimmed[1..trimmed.len() - 1].trim()
    } else {
        trimmed
    }
}

/// Decodes one DSLX string token with the escape rules used by XLS 0.53.
fn parse_string_literal(text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.len() < 2 || !text.starts_with('"') || !text.ends_with('"') {
        return Err("assert!/cover! label must be a string literal".to_string());
    }
    let mut out = String::new();
    let mut chars = text[1..text.len() - 1].chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let escaped = chars
                .next()
                .ok_or_else(|| "unterminated escape in assert!/cover! label".to_string())?;
            match escaped {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '0' => out.push('\0'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                'x' => {
                    let high = chars.next().ok_or_else(|| {
                        "DSLX hexadecimal string escape requires two digits".to_string()
                    })?;
                    let low = chars.next().ok_or_else(|| {
                        "DSLX hexadecimal string escape requires two digits".to_string()
                    })?;
                    let value = high
                        .to_digit(16)
                        .zip(low.to_digit(16))
                        .map(|(high, low)| high * 16 + low)
                        .ok_or_else(|| {
                            "DSLX hexadecimal string escape requires two digits".to_string()
                        })?;
                    if value >= 0x80 {
                        return Err(
                            "assert!/cover! labels do not support non-UTF-8 byte escapes (\\x80..\\xff)"
                                .to_string(),
                        );
                    }
                    out.push(char::from(value as u8));
                }
                'u' => {
                    if chars.next() != Some('{') {
                        return Err(
                            "DSLX Unicode string escape requires braces, for example \\u{41}"
                                .to_string(),
                        );
                    }
                    let mut digits = String::new();
                    let mut closed = false;
                    for next in chars.by_ref() {
                        if next == '}' {
                            closed = true;
                            break;
                        }
                        digits.push(next);
                    }
                    if !closed
                        || digits.is_empty()
                        || digits.len() > 6
                        || !digits.chars().all(|digit| digit.is_ascii_hexdigit())
                    {
                        return Err(
                            "DSLX Unicode string escape requires one to six hexadecimal digits"
                                .to_string(),
                        );
                    }
                    let value = u32::from_str_radix(&digits, 16)
                        .expect("validated Unicode escape has only hexadecimal digits");
                    let decoded = char::from_u32(value).ok_or_else(|| {
                        "DSLX Unicode string escape is not a Unicode scalar value".to_string()
                    })?;
                    out.push(decoded);
                }
                '\n' | '\r' => {
                    return Err(
                        "DSLX string literals do not support backslash-newline escapes".to_string(),
                    );
                }
                other => {
                    return Err(format!(
                        "unsupported DSLX string escape '\\{other}' in assert!/cover! label"
                    ));
                }
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn parses_ports_registers_and_assignments() {
        let source = r#"
fn invert(x: bool) -> bool { !x }

pub block sample<N: u32 = {u32:8}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output ready: bool,
  input valid: bool,
  output data: uN[N],
) {
  let ready_c = invert(valid);
  reg q: uN[N] {
    init_value: uN[N]:0,
    en: valid,
    next: q + uN[N]:1,
  }
  assign ready = ready_c;
  assign data = q;
}
"#;
        let module = parse_module(source, Path::new("sample.x")).unwrap();
        assert!(module.prelude.contains("fn invert"));
        assert_eq!(module.blocks.len(), 1);
        let block = &module.blocks[0];
        assert_eq!(block.name, "sample");
        assert_eq!(block.params[0].default.as_deref(), Some("u32:8"));
        assert_eq!(block.ports.len(), 5);
        assert_eq!(block.items.len(), 4);
    }

    #[test]
    fn rejects_legacy_init_register_field() {
        let source = r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output data: bool,
) {
  reg q: bool { init: false, next: true, }
  assign data = q;
}
"#;
        let error = parse_module(source, Path::new("sample.x")).unwrap_err();
        assert_eq!(error.offset, source.find("init"));
        assert!(error.message.contains("unknown register field 'init'"));
    }

    #[test]
    fn rejects_duplicate_init_value_after_explicit_none() {
        let source = r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output data: bool,
) {
  reg q: bool { init_value: none, init_value: false, next: true, }
  assign data = q;
}
"#;
        let error = parse_module(source, Path::new("sample.x")).unwrap_err();
        let second_field = source.rfind("init_value").unwrap();
        assert_eq!(error.offset, Some(second_field));
        assert!(
            error
                .message
                .contains("register field 'init_value' is specified more than once")
        );
    }

    #[test]
    fn rejects_reg_trailing_semicolon() {
        let source = r#"
block sample(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  reg q: bool { next: true, };
  assign y = q;
}
"#;
        let error = parse_module(source, Path::new("sample.x")).unwrap_err();
        assert!(error.message.contains("do not end with a semicolon"));
    }

    #[test]
    fn keeps_generic_commas_and_comparison_defaults_inside_parametrics() {
        let source = r#"
pub block sample<
  SELECT: bool = {u32:1 < u32:2},
  WIDTH: u32 = {choose < u32:8, u32:9 > ()},
>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = SELECT;
}
"#;
        let module = parse_module(source, Path::new("sample.x")).unwrap();
        assert_eq!(module.blocks[0].params.len(), 2);
        assert_eq!(
            module.blocks[0].params[0].default.as_deref(),
            Some("u32:1 < u32:2")
        );
        assert_eq!(
            module.blocks[0].params[1].default.as_deref(),
            Some("choose < u32:8, u32:9 > ()")
        );
    }

    #[test]
    fn removes_outer_proc_attributes_with_the_proc_from_expression_prelude() {
        let source = r#"
#[test_proc]
proc Worker {
  config() { () }
  init { () }
  next(state: ()) { () }
}

pub block wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#;
        let module = parse_module(source, Path::new("sample.x")).unwrap();
        assert!(!module.prelude.contains("test_proc"));
        assert!(!module.prelude.contains("proc Worker"));
        assert!(module.proc_prelude.contains("#[test_proc]"));
        assert!(module.proc_prelude.contains("proc Worker"));
    }

    #[test]
    fn records_all_tuple_pattern_bindings() {
        let source = r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let (first, (second, _)) = (x, (x, x));
  assign y = first + second;
}
"#;
        let module = parse_module(source, Path::new("sample.x")).unwrap();
        let BlockItem::Let { names, .. } = &module.blocks[0].items[0] else {
            panic!("expected let binding");
        };
        assert_eq!(names, &["first", "second"]);
    }

    #[test]
    fn decodes_utf8_dslx_property_label_escapes() {
        let source = r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assert!(true, "line\nhex\x21 bytes\x00\x7f unicode\u{1f642} quote\"");
  assign y = true;
}
"#;
        let module = parse_module(source, Path::new("sample.x")).unwrap();
        let BlockItem::Assert { label, .. } = &module.blocks[0].items[0] else {
            panic!("expected assertion");
        };
        assert_eq!(label, "line\nhex! bytes\0\u{7f} unicode🙂 quote\"");
    }

    #[test]
    fn requires_property_label_to_be_exactly_one_string_token() {
        for label in [
            r#""first" "second""#,
            r#"("wrapped")"#,
            r#""left" + "right""#,
        ] {
            let source = format!(
                r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {{
  cover!(true, {label});
  assign y = true;
}}
"#
            );
            let error = parse_module(&source, Path::new("sample.x")).unwrap_err();
            assert!(
                error
                    .message
                    .contains("must be exactly one string literal token"),
                "unexpected diagnostic for {label}: {}",
                error.message
            );
        }
    }

    #[test]
    fn rejects_unknown_and_malformed_property_label_escapes() {
        for (label, expected) in [
            (r#""bad\q""#, "unsupported DSLX string escape"),
            (r#""bad\x1""#, "requires two digits"),
            (r#""bad\x80""#, "do not support non-UTF-8 byte escapes"),
            (
                r#""bad\
continuation""#,
                "do not support backslash-newline escapes",
            ),
            (r#""bad\u{1_f}""#, "one to six hexadecimal digits"),
            (r#""bad\u{110000}""#, "not a Unicode scalar value"),
        ] {
            let source = format!(
                r#"
pub block sample(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {{
  assert!(true, {label});
  assign y = true;
}}
"#
            );
            let error = parse_module(&source, Path::new("sample.x")).unwrap_err();
            assert!(
                error.message.contains(expected),
                "unexpected diagnostic for {label}: {}",
                error.message
            );
        }
    }
}
