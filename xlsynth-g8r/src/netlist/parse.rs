// SPDX-License-Identifier: Apache-2.0

//! Token scanner and parser for gate-level netlists.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth::IrBits;

pub type PortId = SymbolU32;
pub type NetId = SymbolU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetIndex(pub usize);

/// Index into `NetlistModule.instances`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstIndex(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Net {
    pub name: NetId,
    /// Optional (msb, lsb) width for this net.
    pub width: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetlistPort {
    pub direction: PortDirection,
    pub width: Option<(u32, u32)>, // (msb, lsb)
    pub name: PortId,
}

/// Parsed gate-level module.
///
/// Invariants enforced by the parser:
/// - `instances` is the list of instance declarations in the module body.
/// - `instance_name` values are **unique within a module**; if the input
///   netlist contains multiple instances with the same name, parsing fails with
///   a `ScanError` instead of constructing a `NetlistModule`.
/// - Gate-level parsing honors Verilog implicit-wire semantics: if an
///   identifier is used in a net context but has not been explicitly declared,
///   the parser (by default) synthesizes an implicit 1-bit `wire` for that
///   name, scoped to the current module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetlistModule {
    pub name: PortId,
    pub ports: Vec<NetlistPort>,
    pub wires: Vec<NetIndex>,
    pub instances: Vec<NetlistInstance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetRef {
    Simple(NetIndex),
    BitSelect(NetIndex, u32),       // b[5]
    PartSelect(NetIndex, u32, u32), // b[msb:lsb]
    Literal(IrBits),
    Unconnected,
    Concat(Vec<NetRef>), // { a, b[5], c[7:0], 1'b0 }
}

impl NetRef {
    /// Collect all `NetIndex` values reachable from this `NetRef` into `out`.
    pub fn collect_net_indices(&self, out: &mut Vec<NetIndex>) {
        match self {
            NetRef::Simple(idx) | NetRef::BitSelect(idx, _) | NetRef::PartSelect(idx, _, _) => {
                out.push(*idx);
            }
            NetRef::Concat(elems) => {
                for e in elems {
                    e.collect_net_indices(out);
                }
            }
            NetRef::Literal(_) | NetRef::Unconnected => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetlistInstance {
    pub type_name: PortId,
    pub instance_name: PortId,
    pub connections: Vec<(PortId, NetRef)>, // (port, net ref)
    pub inst_lineno: u32,
    pub inst_colno: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Keyword {
    Wire,
    Module,
    Endmodule,
    Input,
    Output,
    Inout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
    Inout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotationValue {
    I64(i64),
    String(String),
    VerilogInt { width: Option<usize>, value: IrBits },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenPayload {
    Identifier(String),
    Keyword(Keyword),
    OParen,
    CParen,
    OBrack,
    CBrack,
    OBrace,
    CBrace,
    Colon,
    Semi,
    Comma,
    Dot,
    Equals,
    Comment(String),
    Annotation { key: String, value: AnnotationValue },
    VerilogInt { width: Option<usize>, value: IrBits },
}

fn is_simple_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl fmt::Display for AnnotationValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnnotationValue::I64(i) => write!(f, "{}", i),
            AnnotationValue::String(s) => {
                // Annotation strings cannot contain embedded '"' per scanner rules.
                let s = s.replace('"', "");
                write!(f, "\"{}\"", s)
            }
            AnnotationValue::VerilogInt { width, value } => {
                let v = xlsynth::IrValue::from_bits(value).to_u32().unwrap();
                match width {
                    Some(w) => write!(f, "{}'d{}", w, v),
                    None => write!(f, "{}", v),
                }
            }
        }
    }
}

impl fmt::Display for TokenPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenPayload::Identifier(s) => {
                if is_simple_identifier(s) {
                    write!(f, "{}", s)
                } else {
                    // Escaped identifier: leading backslash, terminated by whitespace
                    write!(f, "\\{} ", s)
                }
            }
            TokenPayload::Keyword(Keyword::Module) => write!(f, "module"),
            TokenPayload::Keyword(Keyword::Wire) => write!(f, "wire"),
            TokenPayload::Keyword(Keyword::Endmodule) => write!(f, "endmodule"),
            TokenPayload::Keyword(Keyword::Input) => write!(f, "input"),
            TokenPayload::Keyword(Keyword::Output) => write!(f, "output"),
            TokenPayload::Keyword(Keyword::Inout) => write!(f, "inout"),
            TokenPayload::OParen => write!(f, "("),
            TokenPayload::CParen => write!(f, ")"),
            TokenPayload::OBrack => write!(f, "["),
            TokenPayload::CBrack => write!(f, "]"),
            TokenPayload::OBrace => write!(f, "{{"),
            TokenPayload::CBrace => write!(f, "}}"),
            TokenPayload::Colon => write!(f, ":"),
            TokenPayload::Semi => write!(f, ";"),
            TokenPayload::Comma => write!(f, ","),
            TokenPayload::Dot => write!(f, "."),
            TokenPayload::Equals => write!(f, "="),
            TokenPayload::Comment(s) => write!(f, "//{}\n", s.replace('\n', " ")),
            TokenPayload::Annotation { key, value } => {
                write!(f, "(* {} = {} *)", key, value)
            }
            TokenPayload::VerilogInt { width, value } => {
                let v = xlsynth::IrValue::from_bits(value).to_u32().unwrap();
                match width {
                    Some(w) => write!(f, "{}'d{}", w, v),
                    None => write!(f, "{}", v),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    pub lineno: u32,
    pub colno: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: Pos,
    pub limit: Pos,
}

impl Span {
    pub fn to_human_string(&self) -> String {
        format!(
            "{}:{}..{}:{}",
            self.start.lineno, self.start.colno, self.limit.lineno, self.limit.colno
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub payload: TokenPayload,
    pub span: Span,
}

#[derive(Debug)]
pub struct ScanError {
    pub message: String,
    pub span: Span,
}

pub struct TokenScanner<R: Read + 'static> {
    reader: BufReader<R>,
    pub pos: Pos,
    lookahead: Option<Token>,
    done: bool,
    line_lookup: Box<dyn Fn(u32) -> Option<String>>, // line number -> line text
}

impl<R: Read + 'static> TokenScanner<R> {
    /// Construct a TokenScanner with a custom line lookup callback.
    pub fn with_line_lookup(reader: R, line_lookup: Box<dyn Fn(u32) -> Option<String>>) -> Self {
        Self {
            reader: BufReader::new(reader),
            pos: Pos {
                lineno: 1,
                colno: 1,
            },
            lookahead: None,
            done: false,
            line_lookup,
        }
    }

    /// Construct a TokenScanner for a file, using the file path for line
    /// lookup.
    pub fn from_file_with_path(reader: R, path: std::path::PathBuf) -> Self {
        let path_clone = path.clone();
        Self::with_line_lookup(
            reader,
            Box::new(move |lineno| {
                use std::io::{BufRead, BufReader};
                let file = std::fs::File::open(&path_clone).ok()?;
                let reader = BufReader::new(file);
                reader
                    .lines()
                    .nth((lineno - 1) as usize)
                    .and_then(Result::ok)
            }),
        )
    }
}

impl<'a> TokenScanner<std::io::Cursor<&'a [u8]>> {
    /// Construct a TokenScanner for a string (for tests), using a Vec<String>
    /// for line lookup.
    pub fn from_str(input: &'a str) -> Self {
        let lines: Vec<String> = input.lines().map(|s| s.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        Self::with_line_lookup(std::io::Cursor::new(input.as_bytes()), Box::new(lookup))
    }
}

impl<R: Read + 'static> TokenScanner<R> {
    #[inline]
    fn peekb(&mut self) -> Option<u8> {
        if self.done {
            return None;
        }
        match self.reader.fill_buf() {
            Ok(buf) => buf.first().copied(),
            Err(_) => {
                self.done = true;
                None
            }
        }
    }

    #[inline]
    fn popb(&mut self) -> Option<u8> {
        if self.done {
            return None;
        }
        match self.reader.fill_buf() {
            Ok(buf) => {
                if let Some(&b) = buf.first() {
                    if b == b'\n' {
                        self.pos.lineno += 1;
                        self.pos.colno = 1;
                    } else {
                        self.pos.colno += 1;
                    }
                    // Consume exactly one byte.
                    self.reader.consume(1);
                    Some(b)
                } else {
                    self.done = true;
                    None
                }
            }
            Err(_) => {
                self.done = true;
                None
            }
        }
    }

    pub fn peekc(&mut self) -> Option<char> {
        self.peekb().map(|b| b as char)
    }

    pub fn popc(&mut self) -> Option<char> {
        self.popb().map(|b| b as char)
    }

    pub fn peekt(&mut self) -> Result<Option<&Token>, ScanError> {
        if self.lookahead.is_none() && !self.done {
            self.lookahead = self.next_token()?;
        }
        Ok(self.lookahead.as_ref())
    }

    pub fn popt(&mut self) -> Result<Option<Token>, ScanError> {
        if self.lookahead.is_none() && !self.done {
            self.lookahead = self.next_token()?;
        }
        Ok(self.lookahead.take())
    }

    pub fn pop_identifier_or_error(&mut self) -> Result<Token, ScanError> {
        match self.popt()? {
            Some(tok) => match &tok.payload {
                TokenPayload::Identifier(_) => Ok(tok),
                _ => Err(ScanError {
                    message: "Expected identifier".to_string(),
                    span: tok.span,
                }),
            },
            None => Err(ScanError {
                message: "Unexpected EOF, expected identifier".to_string(),
                span: Span {
                    start: self.pos,
                    limit: self.pos,
                },
            }),
        }
    }

    pub fn pop_i64_or_error(&mut self) -> Result<i64, ScanError> {
        match self.popt()? {
            Some(tok) => match &tok.payload {
                TokenPayload::Annotation {
                    value: AnnotationValue::I64(i),
                    ..
                } => Ok(*i),
                _ => Err(ScanError {
                    message: "Expected integer value".to_string(),
                    span: tok.span,
                }),
            },
            None => Err(ScanError {
                message: "Unexpected EOF, expected integer value".to_string(),
                span: Span {
                    start: self.pos,
                    limit: self.pos,
                },
            }),
        }
    }

    fn error_with_context(&self, msg: &str, span: Span) -> ScanError {
        let line = (self.line_lookup)(span.start.lineno)
            .unwrap_or_else(|| "<line unavailable>".to_string());
        let col = (span.start.colno as usize).saturating_sub(1);
        log::error!("ScanError: {} @ {}", msg, span.to_human_string());
        log::error!("{}", line);
        log::error!("{}^", " ".repeat(col));
        ScanError {
            message: format!("{}", msg),
            span,
        }
    }

    fn pop_annotation(&mut self, start: Pos) -> Result<Token, ScanError> {
        // We have already seen '(', next should be '*'
        assert_eq!(self.popc(), Some('('));
        assert_eq!(self.popc(), Some('*'));
        // Skip whitespace
        while let Some(c) = self.peekc() {
            if c.is_whitespace() {
                self.popc();
            } else {
                break;
            }
        }
        // Parse key
        let mut key = String::new();
        while let Some(c) = self.peekc() {
            if c.is_alphanumeric() || c == '_' {
                key.push(self.popc().unwrap());
            } else {
                break;
            }
        }
        // Skip whitespace
        while let Some(c) = self.peekc() {
            if c.is_whitespace() {
                self.popc();
            } else {
                break;
            }
        }
        // Expect '='
        assert_eq!(self.popc(), Some('='));
        // Skip whitespace
        while let Some(c) = self.peekc() {
            if c.is_whitespace() {
                self.popc();
            } else {
                break;
            }
        }
        // Parse value (string, integer, or Verilog integer)
        let value: AnnotationValue = match self.peekc() {
            Some('"') => {
                self.popc(); // consume opening quote
                let mut s = String::new();
                while let Some(c) = self.popc() {
                    if c == '"' {
                        break;
                    }
                    s.push(c);
                }
                AnnotationValue::String(s)
            }
            Some(c) if c.is_ascii_digit() => {
                // Check for Verilog integer: $width'$base$value
                let mut num = String::new();
                while let Some(c) = self.peekc() {
                    if c.is_ascii_digit() {
                        num.push(self.popc().unwrap());
                    } else {
                        break;
                    }
                }
                let width = if !num.is_empty() {
                    Some(num.parse::<usize>().unwrap())
                } else {
                    None
                };
                if self.peekc() == Some('\'') {
                    self.popc(); // consume '
                    // Now parse base and value as a string until whitespace or '*' or ')'
                    let mut base_and_value = String::new();
                    while let Some(c) = self.peekc() {
                        if c.is_whitespace() || c == '*' || c == ')' {
                            break;
                        }
                        base_and_value.push(self.popc().unwrap());
                    }
                    // Convert Verilog base to Rust-style
                    let base_and_value = if let Some((_base, _rest)) = base_and_value
                        .split_once(|c: char| c == 'b' || c == 'h' || c == 'o' || c == 'd')
                    {
                        let (base_char, _value_str) = base_and_value.split_at(1);
                        let prefix = match base_char {
                            "b" => "0b",
                            "h" => "0x",
                            "o" => "0o",
                            "d" => "",
                            _ => "",
                        };
                        format!("{}{}", prefix, &base_and_value[1..])
                    } else {
                        base_and_value.clone()
                    };
                    let irbits_str = if let Some(width) = width {
                        format!("bits[{}]:{}", width, base_and_value)
                    } else {
                        base_and_value.clone()
                    };
                    let value = xlsynth::IrValue::parse_typed(&irbits_str)
                        .unwrap()
                        .to_bits()
                        .unwrap();
                    // Skip whitespace
                    while let Some(c) = self.peekc() {
                        if c.is_whitespace() {
                            self.popc();
                        } else {
                            break;
                        }
                    }
                    // Expect '*)'
                    if self.popc() != Some('*') {
                        return Err(self.error_with_context(
                            "Expected '*' to close annotation",
                            Span {
                                start,
                                limit: self.pos,
                            },
                        ));
                    }
                    if self.popc() != Some(')') {
                        return Err(self.error_with_context(
                            "Expected ')' to close annotation",
                            Span {
                                start,
                                limit: self.pos,
                            },
                        ));
                    }
                    let limit = self.pos;
                    return Ok(Token {
                        payload: TokenPayload::Annotation {
                            key,
                            value: AnnotationValue::VerilogInt { width, value },
                        },
                        span: Span { start, limit },
                    });
                } else {
                    // Not a Verilog int, treat as plain integer
                    AnnotationValue::I64(num.parse().unwrap())
                }
            }
            Some(c) if c == '-' => {
                let mut num = String::new();
                num.push(self.popc().unwrap());
                while let Some(c) = self.peekc() {
                    if c.is_ascii_digit() {
                        num.push(self.popc().unwrap());
                    } else {
                        break;
                    }
                }
                AnnotationValue::I64(num.parse().unwrap())
            }
            _ => {
                return Err(self.error_with_context(
                    "Expected annotation value",
                    Span {
                        start,
                        limit: self.pos,
                    },
                ));
            }
        };
        // Skip whitespace
        while let Some(c) = self.peekc() {
            if c.is_whitespace() {
                self.popc();
            } else {
                break;
            }
        }
        // Expect '*)'
        if self.popc() != Some('*') {
            return Err(self.error_with_context(
                "Expected '*' to close annotation",
                Span {
                    start,
                    limit: self.pos,
                },
            ));
        }
        if self.popc() != Some(')') {
            return Err(self.error_with_context(
                "Expected ')' to close annotation",
                Span {
                    start,
                    limit: self.pos,
                },
            ));
        }
        let limit = self.pos;
        Ok(Token {
            payload: TokenPayload::Annotation { key, value },
            span: Span { start, limit },
        })
    }

    fn pop_identifier(&mut self, start: Pos) -> Token {
        let mut ident = String::new();
        // Check if it's an escaped identifier
        if self.peekc() == Some('\\') {
            self.popc(); // Consume the backslash character.
            while let Some(c) = self.peekc() {
                if c.is_whitespace() {
                    // End of escaped identifier, consume the whitespace
                    self.popc();
                    break;
                }
                ident.push(self.popc().unwrap());
            }
        } else {
            // Regular identifier
            while let Some(c) = self.peekc() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(self.popc().unwrap());
                } else {
                    break;
                }
            }
        }
        let limit = self.pos;
        if let Some(kw) = Keyword::from_str(&ident) {
            Token {
                payload: TokenPayload::Keyword(kw),
                span: Span { start, limit },
            }
        } else {
            Token {
                payload: TokenPayload::Identifier(ident),
                span: Span { start, limit },
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Option<Token>, ScanError> {
        // Always skip whitespace (except newlines) before any token logic
        loop {
            match self.peekb() {
                Some(b) if b.is_ascii_whitespace() && b != b'\n' => {
                    self.popb();
                }
                _ => break,
            }
        }
        let start = self.pos;
        let b = match self.peekb() {
            Some(b) => b,
            None => return Ok(None),
        };
        // Handle Verilog preprocessor-style directive lines we want to ignore.
        // Currently, we skip lines that begin with `` `timescale`` anywhere in
        // the file. This consumes through the end of the line and resumes
        // scanning as if the directive were not present.
        if b == b'`' {
            let directive_start = start;
            // consume backtick
            self.popb();
            // read directive word (letters/underscores)
            let mut word = String::new();
            while let Some(ch) = self.peekc() {
                if ch.is_ascii_alphabetic() || ch == '_' {
                    word.push(self.popc().unwrap());
                } else {
                    break;
                }
            }
            if word == "timescale" {
                // consume until end-of-line (including the newline if present)
                while let Some(ch) = self.popc() {
                    if ch == '\n' {
                        break;
                    }
                }
                log::trace!("TokenScanner: skipped `timescale directive line");
                // continue scanning next token
                return match self.next_token()? {
                    Some(tok) => Ok(Some(tok)),
                    None => Ok(None),
                };
            } else {
                // Not a supported directive: report unexpected backtick at its position
                return Err(self.error_with_context(
                    "Unexpected character '`'",
                    Span {
                        start: directive_start,
                        limit: directive_start,
                    },
                ));
            }
        }
        // Handle annotation: use non-consuming two-byte lookahead for "(*"
        if b == b'(' {
            if let Ok(buf) = self.reader.fill_buf() {
                if buf.len() >= 2 && buf[0] == b'(' && buf[1] == b'*' {
                    return self.pop_annotation(start).map(Some);
                }
            }
        }
        // Handle identifier/keyword
        //
        // Note that "escaped identifiers" can begin with the backslash character.
        if b.is_ascii_alphabetic() || b == b'_' || b == b'\\' {
            return Ok(Some(self.pop_identifier(start)));
        }
        // Handle number (plain integer literal or Verilog-style literal)
        if b.is_ascii_digit() {
            let mut num = String::new();
            while let Some(c) = self.peekc() {
                if c.is_ascii_digit() {
                    num.push(self.popc().unwrap());
                } else {
                    break;
                }
            }
            let width = if !num.is_empty() {
                Some(num.parse::<usize>().unwrap())
            } else {
                None
            };
            if self.peekc() == Some('\'') {
                self.popc(); // consume '
                // Now parse base and value as a string, but only consume characters
                // that are valid inside a Verilog number literal (base char plus
                // digits/hex digits, x/z/?, and underscores). This ensures we do
                // not accidentally swallow structural punctuation like '}' that
                // should be tokenized separately.
                let mut base_and_value = String::new();
                while let Some(c) = self.peekc() {
                    if c.is_ascii_alphanumeric()
                        || c == '_'
                        || c == 'x'
                        || c == 'X'
                        || c == 'z'
                        || c == 'Z'
                        || c == '?'
                    {
                        base_and_value.push(self.popc().unwrap());
                    } else {
                        break;
                    }
                }
                // Convert Verilog base to Rust-style
                let base_and_value = if let Some((_base, _rest)) = base_and_value
                    .split_once(|c: char| c == 'b' || c == 'h' || c == 'o' || c == 'd')
                {
                    let (base_char, _value_str) = base_and_value.split_at(1);
                    let prefix = match base_char {
                        "b" => "0b",
                        "h" => "0x",
                        "o" => "0o",
                        "d" => "",
                        _ => "",
                    };
                    format!("{}{}", prefix, &base_and_value[1..])
                } else {
                    base_and_value.clone()
                };
                let irbits_str = if let Some(width) = width {
                    format!("bits[{}]:{}", width, base_and_value)
                } else {
                    base_and_value.clone()
                };
                let value = xlsynth::IrValue::parse_typed(&irbits_str)
                    .unwrap()
                    .to_bits()
                    .unwrap();
                let limit = self.pos;
                return Ok(Some(Token {
                    payload: TokenPayload::VerilogInt { width, value },
                    span: Span { start, limit },
                }));
            } else {
                // Parse as IrBits (decimal, width 32 per Verilog standard)
                let irbits_str = format!("bits[32]:{}", num);
                let value = match xlsynth::IrValue::parse_typed(&irbits_str) {
                    Ok(v) => v.to_bits().unwrap(),
                    Err(e) => {
                        let line = (self.line_lookup)(start.lineno)
                            .unwrap_or_else(|| "<line unavailable>".to_string());
                        let col = (start.colno as usize).saturating_sub(1);
                        panic!(
                            "Failed to parse integer literal '{}': {}\nLine {}: {}\n{}^",
                            irbits_str,
                            e,
                            start.lineno,
                            line,
                            " ".repeat(col)
                        );
                    }
                };
                let limit = self.pos;
                return Ok(Some(Token {
                    payload: TokenPayload::VerilogInt { width: None, value },
                    span: Span { start, limit },
                }));
            }
        }
        // Handle line comments
        if b == b'/' {
            self.popb();
            match self.peekc() {
                Some('/') => {
                    self.popc();
                    let mut comment = String::new();
                    while let Some(ch) = self.popc() {
                        if ch == '\n' {
                            break;
                        }
                        comment.push(ch);
                    }
                    let limit = self.pos;
                    return Ok(Some(Token {
                        payload: TokenPayload::Comment(comment),
                        span: Span { start, limit },
                    }));
                }
                Some('*') => {
                    // Skip block comment
                    self.popc(); // consume '*'
                    let mut prev = None;
                    while let Some(ch) = self.popc() {
                        if prev == Some('*') && ch == '/' {
                            break;
                        }
                        prev = Some(ch);
                    }
                    // After skipping, try to get the next token
                    return match self.next_token()? {
                        Some(tok) => Ok(Some(tok)),
                        None => Ok(None),
                    };
                }
                _ => {
                    // Not a comment, error
                    let limit = self.pos;
                    return Err(self.error_with_context("Unexpected '/'", Span { start, limit }));
                }
            }
        }
        // Handle punctuation
        let payload = match b {
            b'(' => {
                self.popb();
                TokenPayload::OParen
            }
            b')' => {
                self.popb();
                TokenPayload::CParen
            }
            b'[' => {
                self.popb();
                TokenPayload::OBrack
            }
            b']' => {
                self.popb();
                TokenPayload::CBrack
            }
            b'{' => {
                self.popb();
                TokenPayload::OBrace
            }
            b'}' => {
                self.popb();
                TokenPayload::CBrace
            }
            b':' => {
                self.popb();
                TokenPayload::Colon
            }
            b';' => {
                self.popb();
                TokenPayload::Semi
            }
            b',' => {
                self.popb();
                TokenPayload::Comma
            }
            b'.' => {
                self.popb();
                TokenPayload::Dot
            }
            b'=' => {
                self.popb();
                TokenPayload::Equals
            }
            b'\n' => {
                self.popb();
                return match self.next_token()? {
                    Some(tok) => Ok(Some(tok)),
                    None => Ok(None),
                };
            } // skip newlines
            _ => {
                // Error for unknown token
                let limit = self.pos;
                return Err(self.error_with_context(
                    &format!("Unexpected character '{}'", b as char),
                    Span { start, limit },
                ));
            }
        };
        let limit = self.pos;
        Ok(Some(Token {
            payload,
            span: Span { start, limit },
        }))
    }
}

impl Keyword {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "module" => Some(Keyword::Module),
            "wire" => Some(Keyword::Wire),
            "endmodule" => Some(Keyword::Endmodule),
            "input" => Some(Keyword::Input),
            "output" => Some(Keyword::Output),
            "inout" => Some(Keyword::Inout),
            _ => None,
        }
    }
}

// --- Recursive descent parser ---
pub struct Parser<R: Read + 'static> {
    scanner: TokenScanner<R>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
    pub nets: Vec<Net>,
    net_index_by_name: HashMap<NetId, NetIndex>,
    /// Index into `nets` where the current module's nets begin.
    /// Used to scope name lookups so that widths are checked per-module
    /// instead of across the entire file.
    current_module_net_start: usize,
    /// Span where each net's current width was first determined, keyed by
    /// net name. Used only for diagnostics when reporting conflicting
    /// widths during parsing; not carried into result artifacts.
    net_width_span_by_name: HashMap<NetId, Span>,
    /// Whether the parser should synthesize implicit nets when an identifier
    /// is used in a net context but has not been explicitly declared.
    allow_implicit_nets: bool,
    /// Names of nets that were first introduced implicitly (via use in a
    /// net context) rather than via an explicit declaration. Tracked
    /// per-module for diagnostics; not carried into result artifacts.
    implicit_net_by_name: HashSet<NetId>,
}

impl<R: Read + 'static> Parser<R> {
    /// Construct a parser with explicit control over implicit-net handling.
    pub fn new_with_options(scanner: TokenScanner<R>, allow_implicit_nets: bool) -> Self {
        Self {
            scanner,
            interner: StringInterner::new(),
            nets: Vec::new(),
            net_index_by_name: HashMap::new(),
            current_module_net_start: 0,
            net_width_span_by_name: HashMap::new(),
            allow_implicit_nets,
            implicit_net_by_name: HashSet::new(),
        }
    }

    /// Construct a parser with implicit nets enabled (default gate-level
    /// behavior matching Verilog's implicit wire semantics).
    pub fn new(scanner: TokenScanner<R>) -> Self {
        Self::new_with_options(scanner, /* allow_implicit_nets= */ true)
    }

    fn parse_netref_expr(&mut self) -> Result<NetRef, ScanError> {
        // Optionally skip comments/annotations at start
        loop {
            let peek = self.scanner.peekt()?;
            match peek {
                Some(tok) => match &tok.payload {
                    TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                        self.scanner.popt()?;
                    }
                    _ => break,
                },
                None => break,
            }
        }
        if let Some(tok) = self.scanner.peekt()? {
            if matches!(tok.payload, TokenPayload::OBrace) {
                // Parse concatenation: { expr (, expr)* }
                self.scanner.popt()?; // consume '{'
                let mut elems: Vec<NetRef> = Vec::new();
                loop {
                    // Skip comments/annotations between elements
                    loop {
                        let peek = self.scanner.peekt()?;
                        match peek {
                            Some(tok) => match &tok.payload {
                                TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                                    self.scanner.popt()?;
                                }
                                _ => break,
                            },
                            None => break,
                        }
                    }
                    // If next is '}', end of concatenation
                    if let Some(tok2) = self.scanner.peekt()? {
                        if matches!(tok2.payload, TokenPayload::CBrace) {
                            self.scanner.popt()?; // consume '}'
                            break;
                        }
                    }
                    // Parse one element (identifier with optional select, literal, or nested
                    // concat)
                    let elem = self.parse_netref_expr()?;
                    elems.push(elem);
                    // Optional comma
                    if let Some(next) = self.scanner.peekt()? {
                        if matches!(next.payload, TokenPayload::Comma) {
                            self.scanner.popt()?; // consume ','
                            continue;
                        }
                    }
                    // Expect closing '}'
                    let cbrace = self.scanner.popt()?.ok_or_else(|| ScanError {
                        message: "expected '}' to close concatenation".to_string(),
                        span: Span {
                            start: self.scanner.pos,
                            limit: self.scanner.pos,
                        },
                    })?;
                    if !matches!(cbrace.payload, TokenPayload::CBrace) {
                        return Err(ScanError {
                            message: "expected '}' to close concatenation".to_string(),
                            span: cbrace.span,
                        });
                    }
                    break;
                }
                return Ok(NetRef::Concat(elems));
            }
        }
        // Parse identifier or literal
        let net_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected net name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        match net_tok.payload {
            TokenPayload::Identifier(s) => {
                let net_sym = self.interner.get_or_intern(s);
                // Lookup declared net (scoped to the current module).
                let net_idx = if let Some(&idx) = self.net_index_by_name.get(&net_sym) {
                    idx
                } else if let Some(pos) = self.nets[self.current_module_net_start..]
                    .iter()
                    .position(|n| n.name == net_sym)
                {
                    let idx = NetIndex(self.current_module_net_start + pos);
                    self.net_index_by_name.insert(net_sym, idx);
                    idx
                } else {
                    if self.allow_implicit_nets {
                        // Create an implicit 1-bit wire for this net, using a
                        // width of (0, 0) to reflect a single bit.
                        let idx = NetIndex(self.nets.len());
                        self.nets.push(Net {
                            name: net_sym,
                            width: Some((0, 0)),
                        });
                        self.net_index_by_name.insert(net_sym, idx);
                        self.net_width_span_by_name.insert(net_sym, net_tok.span);
                        self.implicit_net_by_name.insert(net_sym);
                        idx
                    } else {
                        return Err(ScanError {
                            message: format!(
                                "net '{}' not declared as wire",
                                self.interner.resolve(net_sym).unwrap()
                            ),
                            span: net_tok.span,
                        });
                    }
                };
                // Optional bit/part select
                if let Some(next) = self.scanner.peekt()? {
                    match &next.payload {
                        TokenPayload::OBrack => {
                            self.scanner.popt()?; // consume '['
                            let msb_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                                message: "expected msb or index in net reference".to_string(),
                                span: Span {
                                    start: self.scanner.pos,
                                    limit: self.scanner.pos,
                                },
                            })?;
                            let msb = match msb_tok.payload {
                                TokenPayload::VerilogInt { value, .. } => {
                                    xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                                }
                                _ => {
                                    return Err(ScanError {
                                        message: "expected integer for msb/index in net reference"
                                            .to_string(),
                                        span: msb_tok.span,
                                    });
                                }
                            };
                            let next2 = self.scanner.popt()?.ok_or_else(|| ScanError {
                                message: "expected ':' or ']' in net reference".to_string(),
                                span: Span {
                                    start: self.scanner.pos,
                                    limit: self.scanner.pos,
                                },
                            })?;
                            return match next2.payload {
                                TokenPayload::Colon => {
                                    let lsb_tok =
                                        self.scanner.popt()?.ok_or_else(|| ScanError {
                                            message: "expected lsb in net reference".to_string(),
                                            span: Span {
                                                start: self.scanner.pos,
                                                limit: self.scanner.pos,
                                            },
                                        })?;
                                    let lsb = match lsb_tok.payload {
                                        TokenPayload::VerilogInt { value, .. } => {
                                            xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                                        }
                                        _ => {
                                            return Err(ScanError {
                                                message:
                                                    "expected integer for lsb in net reference"
                                                        .to_string(),
                                                span: lsb_tok.span,
                                            });
                                        }
                                    };
                                    let cbrack_tok =
                                        self.scanner.popt()?.ok_or_else(|| ScanError {
                                            message: "expected ']' after part-select".to_string(),
                                            span: Span {
                                                start: self.scanner.pos,
                                                limit: self.scanner.pos,
                                            },
                                        })?;
                                    if !matches!(cbrack_tok.payload, TokenPayload::CBrack) {
                                        return Err(ScanError {
                                            message: "expected ']' after part-select".to_string(),
                                            span: cbrack_tok.span,
                                        });
                                    }
                                    Ok(NetRef::PartSelect(net_idx, msb, lsb))
                                }
                                TokenPayload::CBrack => Ok(NetRef::BitSelect(net_idx, msb)),
                                _ => Err(ScanError {
                                    message: "expected ':' or ']' in net reference".to_string(),
                                    span: next2.span,
                                }),
                            };
                        }
                        _ => {}
                    }
                }
                Ok(NetRef::Simple(net_idx))
            }
            TokenPayload::VerilogInt { value, width: _ } => Ok(NetRef::Literal(value)),
            other => Err(ScanError {
                message: format!("expected identifier for net name, got {:?}", other),
                span: net_tok.span,
            }),
        }
    }

    /// Ensures there is a `Net` with the given `name` present in `self.nets`.
    /// If it already exists, reconciles optional `width` information:
    /// - (None, None) -> keep None
    /// - (Some(w), None) or (None, Some(w)) -> set to Some(w)
    /// - (Some(a), Some(b)) where a != b -> error
    /// Returns the `NetIndex` for the ensured net.
    fn ensure_net(
        &mut self,
        name: NetId,
        width: Option<(u32, u32)>,
        err_span: Span,
    ) -> Result<NetIndex, ScanError> {
        if let Some(&idx) = self.net_index_by_name.get(&name) {
            let existing = &mut self.nets[idx.0];
            match (existing.width, width) {
                (None, None) => {}
                (None, Some(w)) => {
                    existing.width = Some(w);
                    self.net_width_span_by_name.insert(name, err_span);
                }
                (Some(_), None) => {}
                (Some(a), Some(b)) => {
                    if a != b {
                        debug_assert!(
                            self.net_width_span_by_name.contains_key(&name),
                            "Net with known width should carry a width span for diagnostics"
                        );
                        let prev_span = self
                            .net_width_span_by_name
                            .get(&name)
                            .copied()
                            .unwrap_or(err_span);
                        return Err(ScanError {
                            message: format!(
                                "conflicting widths for net '{}': {:?} vs {:?}; previously determined width was {:?} @ {}",
                                self.interner.resolve(name).unwrap_or("<unknown>"),
                                a,
                                b,
                                a,
                                prev_span.to_human_string()
                            ),
                            span: err_span,
                        });
                    }
                }
            }
            Ok(idx)
        } else {
            let idx = NetIndex(self.nets.len());
            self.nets.push(Net { name, width });
            if width.is_some() {
                self.net_width_span_by_name.insert(name, err_span);
            }
            self.net_index_by_name.insert(name, idx);
            Ok(idx)
        }
    }

    /// Parses optional "[idx]" or "[msb:lsb]" bit/part-select that may follow
    /// an identifier in an assign statement, for either the LHS or RHS.
    fn parse_optional_assign_bit_or_part_select(
        &mut self,
        side_label: &str,
    ) -> Result<(), ScanError> {
        if let Some(next) = self.scanner.peekt()? {
            if matches!(next.payload, TokenPayload::OBrack) {
                // consume '['
                self.scanner.popt()?;
                // parse msb or idx
                let t0 = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: format!(
                        "expected index or msb in assign {} bit/part-select",
                        side_label
                    ),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                match t0.payload {
                    TokenPayload::VerilogInt { .. } => {}
                    _ => {
                        return Err(ScanError {
                            message: format!(
                                "expected integer in assign {} bit/part-select",
                                side_label
                            ),
                            span: t0.span,
                        });
                    }
                }
                // Optional : lsb
                if let Some(peek) = self.scanner.peekt()? {
                    if matches!(peek.payload, TokenPayload::Colon) {
                        self.scanner.popt()?; // consume ':'
                        let t1 = self.scanner.popt()?.ok_or_else(|| ScanError {
                            message: format!("expected lsb in {} part-select", side_label),
                            span: Span {
                                start: self.scanner.pos,
                                limit: self.scanner.pos,
                            },
                        })?;
                        match t1.payload {
                            TokenPayload::VerilogInt { .. } => {}
                            _ => {
                                return Err(ScanError {
                                    message: format!(
                                        "expected integer for lsb in {} part-select",
                                        side_label
                                    ),
                                    span: t1.span,
                                });
                            }
                        }
                    }
                }
                // expect ']'
                let t_cb = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: format!("expected ']' after {} bit/part-select", side_label),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(t_cb.payload, TokenPayload::CBrack) {
                    return Err(ScanError {
                        message: format!("expected ']' after {} bit/part-select", side_label),
                        span: t_cb.span,
                    });
                }
            }
        }
        Ok(())
    }

    /// Parses: assign <ident>([msb:lsb]|[idx]) = <literal_or_ident>;
    /// Only accepts RHS as either a Verilog integer literal or a simple
    /// identifier; errors otherwise. The assign statement is currently ignored
    /// semantically and only parsed for basic structural validation.
    fn parse_assign_literal(&mut self) -> Result<(), ScanError> {
        // consume 'assign' identifier
        let t_assign = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected 'assign'".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        match t_assign.payload {
            TokenPayload::Identifier(ref s) if s == "assign" => {}
            _ => {
                return Err(ScanError {
                    message: "expected 'assign'".to_string(),
                    span: t_assign.span,
                });
            }
        }

        // LHS base identifier
        let t_name = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected identifier on left-hand side of assign".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(t_name.payload, TokenPayload::Identifier(_)) {
            return Err(ScanError {
                message: "expected identifier on left-hand side of assign".to_string(),
                span: t_name.span,
            });
        }

        // Optional bit- or part-select on the LHS: [idx] or [msb:lsb]
        self.parse_optional_assign_bit_or_part_select("LHS")?;

        // expect '='
        let t_eq = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected '=' in assign".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(t_eq.payload, TokenPayload::Equals) {
            return Err(ScanError {
                message: "expected '=' in assign".to_string(),
                span: t_eq.span,
            });
        }

        // RHS literal or identifier (for simple feed-throughs like "assign out = in;"
        // or "assign out[0] = in[0];"). We allow an optional bit- or part-select
        // after an identifier, mirroring the LHS handling.
        let t_rhs = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected literal or identifier on right-hand side of assign".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        match t_rhs.payload {
            TokenPayload::VerilogInt { .. } => {}
            TokenPayload::Identifier(_) => {
                // Optional bit- or part-select on the RHS identifier: [idx] or [msb:lsb]
                self.parse_optional_assign_bit_or_part_select("RHS")?;
            }
            _ => {
                return Err(ScanError {
                    message: "only literal or identifier RHS supported in assign".to_string(),
                    span: t_rhs.span,
                });
            }
        }

        // expect ';'
        let t_semi = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected ';' after assign".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(t_semi.payload, TokenPayload::Semi) {
            return Err(ScanError {
                message: "expected ';' after assign".to_string(),
                span: t_semi.span,
            });
        }
        Ok(())
    }
    pub fn parse_file(&mut self) -> Result<Vec<NetlistModule>, ScanError> {
        log::trace!("parse_file: start");
        let mut modules = Vec::new();
        loop {
            match self.scanner.peekt()? {
                Some(tok) => {
                    log::trace!(
                        "parse_file: top-level token: {:?} at {:?}",
                        tok.payload,
                        tok.span
                    );
                    match &tok.payload {
                        TokenPayload::Keyword(Keyword::Module) => {
                            log::trace!("parse_file: found 'module', parsing module body");
                            modules.push(self.parse_module()?);
                        }
                        // Skip comments and annotations at file scope
                        TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                            log::trace!("parse_file: skipping comment/annotation at top level");
                            self.scanner.popt()?;
                        }
                        // Unexpected token at file scope: stop.
                        _ => {
                            log::trace!(
                                "parse_file: stopping on unexpected top-level token: {:?}",
                                tok.payload
                            );
                            break;
                        }
                    }
                }
                None => break,
            }
        }
        log::trace!("parse_file: done; modules parsed: {}", modules.len());
        Ok(modules)
    }

    pub fn parse_module(&mut self) -> Result<NetlistModule, ScanError> {
        // Each module has its own namespace for nets. Reset the per-module
        // name index and record where this module's nets begin in the global
        // `nets` vector so that lookups (and width checks) are scoped
        // per-module instead of across the entire file.
        self.current_module_net_start = self.nets.len();
        self.net_index_by_name.clear();
        self.net_width_span_by_name.clear();
        self.implicit_net_by_name.clear();

        // Expect 'module' keyword
        let tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected 'module' keyword".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(tok.payload, TokenPayload::Keyword(Keyword::Module)) {
            return Err(ScanError {
                message: "expected 'module' keyword".to_string(),
                span: tok.span,
            });
        }
        // Expect module name (identifier)
        let name_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected module name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        let name = match name_tok.payload {
            TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
            _ => {
                return Err(ScanError {
                    message: "expected identifier for module name".to_string(),
                    span: name_tok.span,
                });
            }
        };
        // Expect '('
        let oparen_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected '(' after module name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(oparen_tok.payload, TokenPayload::OParen) {
            return Err(ScanError {
                message: "expected '(' after module name".to_string(),
                span: oparen_tok.span,
            });
        }
        // Parse port list: identifier[, identifier ...]
        let mut port_names = Vec::new();
        loop {
            let t = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected token in port list".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            match t.payload {
                TokenPayload::Identifier(s) => {
                    let sym = self.interner.get_or_intern(s);
                    port_names.push(sym);
                    // If next is ',', continue
                    if let Some(next) = self.scanner.peekt()? {
                        if matches!(next.payload, TokenPayload::Comma) {
                            self.scanner.popt()?; // consume ','
                            continue;
                        }
                    }
                }
                TokenPayload::CParen => break,
                _ => {
                    return Err(ScanError {
                        message: format!("unexpected token in port list: {:?}", t.payload),
                        span: t.span,
                    });
                }
            }
        }
        // Expect ';'
        let semi_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected ';' after port list".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(semi_tok.payload, TokenPayload::Semi) {
            return Err(ScanError {
                message: "expected ';' after port list".to_string(),
                span: semi_tok.span,
            });
        }
        // Parse body: ports, wires, and instances until 'endmodule'
        let mut ports = Vec::new();
        let mut wires = Vec::new();
        let mut instances = Vec::new();
        // Enforce uniqueness of instance names within a module: we track the
        // set of already-seen names and reject duplicates.
        let mut instance_names: HashSet<PortId> = HashSet::new();
        loop {
            match self.scanner.peekt()? {
                Some(tok) => match &tok.payload {
                    TokenPayload::Keyword(Keyword::Input) => {
                        let mut decls = self.parse_port_decl(PortDirection::Input)?;
                        ports.append(&mut decls);
                    }
                    TokenPayload::Keyword(Keyword::Output) => {
                        let mut decls = self.parse_port_decl(PortDirection::Output)?;
                        ports.append(&mut decls);
                    }
                    TokenPayload::Keyword(Keyword::Inout) => {
                        let mut decls = self.parse_port_decl(PortDirection::Inout)?;
                        ports.append(&mut decls);
                    }
                    TokenPayload::Keyword(Keyword::Wire) => {
                        let wire_indices = self.parse_wire_decl()?;
                        wires.extend(wire_indices);
                    }
                    TokenPayload::Keyword(Keyword::Endmodule) => {
                        self.scanner.popt()?; // consume 'endmodule'
                        break;
                    }
                    TokenPayload::Identifier(s) if s == "assign" => {
                        self.parse_assign_literal()?;
                    }
                    TokenPayload::Identifier(_) => {
                        let instance = self.parse_instance()?;
                        if !instance_names.insert(instance.instance_name) {
                            let instance_start_pos = Pos {
                                lineno: instance.inst_lineno,
                                colno: instance.inst_colno,
                            };
                            return Err(ScanError {
                                message: format!(
                                    "duplicate instance name '{}' in module",
                                    self.interner.resolve(instance.instance_name).unwrap()
                                ),
                                span: Span {
                                    start: instance_start_pos,
                                    limit: instance_start_pos,
                                },
                            });
                        }
                        instances.push(instance);
                    }
                    // Skip comments and annotations
                    TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                        self.scanner.popt()?;
                    }
                    _ => {
                        // Unexpected token
                        break;
                    }
                },
                None => break,
            }
        }
        Ok(NetlistModule {
            name,
            ports,
            wires,
            instances,
        })
    }

    /// Parses: input/output/inout [msb:lsb] foo, bar, ...;
    pub fn parse_port_decl(
        &mut self,
        direction: PortDirection,
    ) -> Result<Vec<NetlistPort>, ScanError> {
        // Consume the keyword
        let kw_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected port direction keyword".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(
            kw_tok.payload,
            TokenPayload::Keyword(Keyword::Input)
                | TokenPayload::Keyword(Keyword::Output)
                | TokenPayload::Keyword(Keyword::Inout)
        ) {
            return Err(ScanError {
                message: "expected port direction keyword".to_string(),
                span: kw_tok.span,
            });
        }
        // Optional width: [msb:lsb]
        let mut width = None;
        if let Some(next) = self.scanner.peekt()? {
            if matches!(next.payload, TokenPayload::OBrack) {
                self.scanner.popt()?; // consume '['
                let msb_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected msb in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                let msb = match msb_tok.payload {
                    TokenPayload::VerilogInt { value, .. } => {
                        xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                    }
                    _ => {
                        return Err(ScanError {
                            message: "expected integer for msb".to_string(),
                            span: msb_tok.span,
                        });
                    }
                };
                let colon_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected ':' in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(colon_tok.payload, TokenPayload::Colon) {
                    return Err(ScanError {
                        message: "expected ':' in width".to_string(),
                        span: colon_tok.span,
                    });
                }
                let lsb_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected lsb in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                let lsb = match lsb_tok.payload {
                    TokenPayload::VerilogInt { value, .. } => {
                        xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                    }
                    _ => {
                        return Err(ScanError {
                            message: "expected integer for lsb".to_string(),
                            span: lsb_tok.span,
                        });
                    }
                };
                let cbrack_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected ']' after width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(cbrack_tok.payload, TokenPayload::CBrack) {
                    return Err(ScanError {
                        message: "expected ']' after width".to_string(),
                        span: cbrack_tok.span,
                    });
                }
                width = Some((msb, lsb));
            }
        }
        // Parse one or more identifiers (comma-separated)
        let mut ports = Vec::new();
        loop {
            let t = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected identifier in port decl".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            let name = match t.payload {
                TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
                _ => {
                    return Err(ScanError {
                        message: "expected identifier in port decl".to_string(),
                        span: t.span,
                    });
                }
            };
            // Ensure a corresponding net exists for this port (non-ANSI body decls imply
            // nets). If a width is present here and not previously known,
            // record it; if conflicting, error.
            let _net_index = self.ensure_net(name, width, t.span)?;
            ports.push(NetlistPort {
                direction: direction.clone(),
                width,
                name,
            });
            // If next is ',', continue
            if let Some(next) = self.scanner.peekt()? {
                if matches!(next.payload, TokenPayload::Comma) {
                    self.scanner.popt()?; // consume ','
                    continue;
                }
            }
            // Otherwise, expect ';'
            let semi = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected ';' after port decl".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(semi.payload, TokenPayload::Semi) {
                return Err(ScanError {
                    message: "expected ';' after port decl".to_string(),
                    span: semi.span,
                });
            }
            break;
        }
        Ok(ports)
    }

    /// Parses: wire foo, bar, baz;
    pub fn parse_wire_decl(&mut self) -> Result<Vec<NetIndex>, ScanError> {
        // Expect 'wire' keyword
        let tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected 'wire' keyword".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(tok.payload, TokenPayload::Keyword(Keyword::Wire)) {
            return Err(ScanError {
                message: "expected 'wire' keyword".to_string(),
                span: tok.span,
            });
        }
        // Optional width: [msb:lsb]
        let mut width = None;
        if let Some(next) = self.scanner.peekt()? {
            if matches!(next.payload, TokenPayload::OBrack) {
                self.scanner.popt()?; // consume '['
                let msb_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected msb in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                let msb = match msb_tok.payload {
                    TokenPayload::VerilogInt { value, .. } => {
                        xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                    }
                    _ => {
                        return Err(ScanError {
                            message: "expected integer for msb".to_string(),
                            span: msb_tok.span,
                        });
                    }
                };
                let colon_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected ':' in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(colon_tok.payload, TokenPayload::Colon) {
                    return Err(ScanError {
                        message: "expected ':' in width".to_string(),
                        span: colon_tok.span,
                    });
                }
                let lsb_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected lsb in width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                let lsb = match lsb_tok.payload {
                    TokenPayload::VerilogInt { value, .. } => {
                        xlsynth::IrValue::from_bits(&value).to_u32().unwrap()
                    }
                    _ => {
                        return Err(ScanError {
                            message: "expected integer for lsb".to_string(),
                            span: lsb_tok.span,
                        });
                    }
                };
                let cbrack_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected ']' after width".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(cbrack_tok.payload, TokenPayload::CBrack) {
                    return Err(ScanError {
                        message: "expected ']' after width".to_string(),
                        span: cbrack_tok.span,
                    });
                }
                width = Some((msb, lsb));
            }
        }
        let mut net_indices = Vec::new();
        loop {
            let t = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected identifier in wire decl".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            let name = match t.payload {
                TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
                ref other => {
                    let line = (self.scanner.line_lookup)(t.span.start.lineno)
                        .unwrap_or_else(|| "<line unavailable>".to_string());
                    log::error!(
                        "expected identifier in wire decl, got {:?} at {:?}\n{}\n{}^",
                        other,
                        t.span,
                        line,
                        " ".repeat((t.span.start.colno as usize).saturating_sub(1))
                    );
                    return Err(ScanError {
                        message: "expected identifier in wire decl".to_string(),
                        span: t.span,
                    });
                }
            };
            // Ensure/dedupe nets for wires too; reconcile widths per rules above.
            let net_idx = self.ensure_net(name, width, t.span)?;
            net_indices.push(net_idx);
            // If next is ',', continue
            if let Some(next) = self.scanner.peekt()? {
                if matches!(next.payload, TokenPayload::Comma) {
                    self.scanner.popt()?; // consume ','
                    continue;
                }
            }
            // Otherwise, expect ';'
            let semi = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected ';' after wire decl".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(semi.payload, TokenPayload::Semi) {
                return Err(ScanError {
                    message: "expected ';' after wire decl".to_string(),
                    span: semi.span,
                });
            }
            break;
        }
        Ok(net_indices)
    }

    /// Parses: TypeName InstanceName ( .Port(Net), ... );
    pub fn parse_instance(&mut self) -> Result<NetlistInstance, ScanError> {
        // Type name
        let type_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected type name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        let type_name = match type_tok.payload {
            TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
            _ => {
                return Err(ScanError {
                    message: "expected identifier for type name".to_string(),
                    span: type_tok.span,
                });
            }
        };
        // Skip comments/annotations before instance name
        loop {
            let peek = self.scanner.peekt()?;
            match peek {
                Some(tok) => match &tok.payload {
                    TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                        self.scanner.popt()?;
                    }
                    _ => break,
                },
                None => break,
            }
        }
        // Instance name
        let inst_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected instance name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        let instance_name = match inst_tok.payload {
            TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
            ref other => {
                return Err(ScanError {
                    message: format!("expected identifier for instance name; got {:?}", other),
                    span: inst_tok.span,
                });
            }
        };
        // Expect '('
        let oparen = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected '(' after instance name".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(oparen.payload, TokenPayload::OParen) {
            return Err(ScanError {
                message: "expected '(' after instance name".to_string(),
                span: oparen.span,
            });
        }
        // Allow empty port list: TypeName InstanceName ( );
        // Skip comments/annotations immediately inside the parentheses
        loop {
            let peek = self.scanner.peekt()?;
            match peek {
                Some(tok) => match &tok.payload {
                    TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                        self.scanner.popt()?;
                    }
                    _ => break,
                },
                None => break,
            }
        }
        if let Some(tok) = self.scanner.peekt()? {
            if matches!(tok.payload, TokenPayload::CParen) {
                self.scanner.popt()?; // consume ')'
                let semi = self.scanner.popt()?.ok_or_else(|| ScanError {
                    message: "expected ';' after instance".to_string(),
                    span: Span {
                        start: self.scanner.pos,
                        limit: self.scanner.pos,
                    },
                })?;
                if !matches!(semi.payload, TokenPayload::Semi) {
                    return Err(ScanError {
                        message: "expected ';' after instance".to_string(),
                        span: semi.span,
                    });
                }
                return Ok(NetlistInstance {
                    type_name,
                    instance_name,
                    connections: Vec::new(),
                    inst_lineno: inst_tok.span.start.lineno,
                    inst_colno: inst_tok.span.start.colno,
                });
            }
        }
        let mut connections = Vec::new();
        loop {
            // Expect .Port
            let dot = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected '.' before port name".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(dot.payload, TokenPayload::Dot) {
                return Err(ScanError {
                    message: "expected '.' before port name".to_string(),
                    span: dot.span,
                });
            }
            let port_tok = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected port name".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            let port = match port_tok.payload {
                TokenPayload::Identifier(s) => self.interner.get_or_intern(s),
                _ => {
                    return Err(ScanError {
                        message: "expected identifier for port name".to_string(),
                        span: port_tok.span,
                    });
                }
            };
            // Expect '('
            let oparen2 = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected '(' before net name".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(oparen2.payload, TokenPayload::OParen) {
                return Err(ScanError {
                    message: "expected '(' before net name".to_string(),
                    span: oparen2.span,
                });
            }
            // Net name or net reference expression (or unconnected)
            // Skip any inline comments/annotations inside the parentheses
            loop {
                let peek = self.scanner.peekt()?;
                match peek {
                    Some(tok) => match &tok.payload {
                        TokenPayload::Comment(_) | TokenPayload::Annotation { .. } => {
                            self.scanner.popt()?;
                        }
                        _ => break,
                    },
                    None => break,
                }
            }
            // Allow immediate ')' to denote an unconnected port
            let is_unconnected = matches!(self.scanner.peekt()?, Some(tok) if matches!(tok.payload, TokenPayload::CParen));
            let net_ref = if is_unconnected {
                NetRef::Unconnected
            } else {
                self.parse_netref_expr()?
            };
            // Expect ')'
            let cparen2 = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected ')' after net name".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(cparen2.payload, TokenPayload::CParen) {
                return Err(ScanError {
                    message: "expected ')' after net name".to_string(),
                    span: cparen2.span,
                });
            }
            connections.push((port, net_ref));
            // If next is ',', continue
            if let Some(next) = self.scanner.peekt()? {
                if matches!(next.payload, TokenPayload::Comma) {
                    self.scanner.popt()?; // consume ','
                    continue;
                }
            }
            // Otherwise, expect ')'
            let cparen = self.scanner.popt()?.ok_or_else(|| ScanError {
                message: "expected ')' after instance connections".to_string(),
                span: Span {
                    start: self.scanner.pos,
                    limit: self.scanner.pos,
                },
            })?;
            if !matches!(cparen.payload, TokenPayload::CParen) {
                return Err(ScanError {
                    message: "expected ')' after instance connections".to_string(),
                    span: cparen.span,
                });
            }
            break;
        }
        // Expect ';'
        let semi = self.scanner.popt()?.ok_or_else(|| ScanError {
            message: "expected ';' after instance".to_string(),
            span: Span {
                start: self.scanner.pos,
                limit: self.scanner.pos,
            },
        })?;
        if !matches!(semi.payload, TokenPayload::Semi) {
            return Err(ScanError {
                message: "expected ';' after instance".to_string(),
                span: semi.span,
            });
        }
        Ok(NetlistInstance {
            type_name,
            instance_name,
            connections,
            inst_lineno: inst_tok.span.start.lineno,
            inst_colno: inst_tok.span.start.colno,
        })
    }

    /// Get the source line for a given line number (1-based), or None if
    /// unavailable.
    pub fn get_line(&self, lineno: u32) -> Option<String> {
        (self.scanner.line_lookup)(lineno)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use std::io::Cursor;

    #[test]
    fn test_token_scanner_oparen() {
        let input = "(";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::OParen));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_cparen() {
        let input = ")";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::CParen));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_semi() {
        let input = ";";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Semi));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_line_comment() {
        let input = "// hello world\n";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Comment(ref s) if s == " hello world"));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_parse_module_with_assign_literal_bit() {
        let src = r#"
module m(a, out);
  input a;
  output [3:0] out;
  assign out[1] = 1'b0;
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_parse_module_with_assign_literal_partselect() {
        let src = r#"
module m(a, out);
  input a;
  output [7:0] out;
  assign out[7:4] = 4'h0;
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_instance_source_position_is_captured() {
        let src = r#"
module m(a, b);
  input a;
  output b;
  wire a, b;
  INVX1 u1 (.A(a), .Y(b));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let m = &modules[0];
        assert_eq!(m.instances.len(), 1);
        let inst = &m.instances[0];
        // 'u1' starts on line 6 (leading newline in the raw string), after two spaces +
        // 'INVX1' (5 chars) + one space => column 9.
        assert_eq!(inst.inst_lineno, 6);
        assert_eq!(inst.inst_colno, 9);
    }

    #[test]
    fn test_parse_instance_with_empty_port_list() {
        let src = r#"
module top();
  DUMMY_CELL u0 ();
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let m = &modules[0];
        assert_eq!(m.instances.len(), 1);
        assert!(m.instances[0].connections.is_empty());
    }

    #[test]
    fn test_parse_instance_with_concat_expr() {
        let src = r#"
module top(a, b, c);
  input a;
  input b;
  input c;
  wire a, b, c;
  TYPE u1 (.MY_PORT({a, b, c}));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let insts = &modules[0].instances;
        assert_eq!(insts.len(), 1);
        let (_p, netref) = &insts[0].connections[0];
        match netref {
            NetRef::Concat(v) => {
                assert_eq!(v.len(), 3);
                for r in v {
                    match r {
                        NetRef::Simple(_) => {}
                        other => panic!("expected Simple, got {:?}", other),
                    }
                }
            }
            other => panic!("expected Concat, got {:?}", other),
        }
    }

    #[test]
    fn test_token_scanner_block_comment_only() {
        let input = "/* block comment */";
        let mut scanner = TokenScanner::from_str(input);
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_block_comment_between_tokens() {
        let input = "( /* block comment */ )";
        let mut scanner = TokenScanner::from_str(input);
        let t1 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t1.payload, TokenPayload::OParen));
        let t2 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t2.payload, TokenPayload::CParen));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    #[should_panic(expected = "ScanError")] // Should panic on unknown token
    fn test_token_scanner_error_on_unknown() {
        let input = "@";
        let mut scanner = TokenScanner::from_str(input);
        scanner.popt().unwrap();
    }

    #[test]
    fn test_token_scanner_annotation() {
        let input = r#"(* src = "foo.sv" *)"#;
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        match t.payload {
            TokenPayload::Annotation { ref key, ref value } => {
                assert_eq!(key, "src");
                match value {
                    AnnotationValue::String(s) => assert_eq!(s, "foo.sv"),
                    _ => panic!("Expected string annotation value"),
                }
            }
            _ => panic!("Expected annotation token"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_keyword() {
        let input = "module";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Keyword(Keyword::Module)));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_identifier() {
        let input = "foo_bar123";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Identifier(ref s) if s == "foo_bar123"));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_comma() {
        let input = ",";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Comma));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_annotation_binary() {
        let input = r#"(* force_downto = 32'b00000000000000000000000000000001 *)"#;
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        match t.payload {
            TokenPayload::Annotation { ref key, ref value } => {
                assert_eq!(key, "force_downto");
                match value {
                    AnnotationValue::VerilogInt { width, value } => {
                        assert_eq!(*width, Some(32));
                        assert_eq!(value.get_bit_count(), 32);
                        assert_eq!(value.to_string(), "bits[32]:1");
                    }
                    _ => panic!("Expected VerilogInt annotation value"),
                }
            }
            _ => panic!("Expected annotation token"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_verilog_int_annotation() {
        let input = r#"(* force_downto = 32'b00000000000000000000000000000001 *)"#;
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        match t.payload {
            TokenPayload::Annotation { ref key, ref value } => {
                assert_eq!(key, "force_downto");
                match value {
                    AnnotationValue::VerilogInt { width, value } => {
                        assert_eq!(*width, Some(32));
                        assert_eq!(value.get_bit_count(), 32);
                        assert_eq!(value.to_string(), "bits[32]:1");
                    }
                    _ => panic!("Expected VerilogInt annotation value"),
                }
            }
            _ => panic!("Expected annotation token"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_stub() {
        let input = "( ) ; // comment\n)";
        let mut scanner = TokenScanner::from_str(input);
        let t1 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t1.payload, TokenPayload::OParen));
        let t2 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t2.payload, TokenPayload::CParen));
        let t3 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t3.payload, TokenPayload::Semi));
        let t4 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t4.payload, TokenPayload::Comment(ref s) if s == " comment"));
        let t5 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t5.payload, TokenPayload::CParen));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_input_declaration() {
        let input = "input [255:0] a;";
        let mut scanner = TokenScanner::from_str(input);
        // 'input' should be a keyword (if supported), otherwise identifier
        let t1 = scanner.popt().expect("no scan error").unwrap();
        match t1.payload {
            TokenPayload::Identifier(ref s) => assert_eq!(s, "input"),
            TokenPayload::Keyword(_) => assert_eq!(format!("{:?}", t1.payload), "Keyword(Input)"),
            _ => panic!("Expected identifier or keyword for 'input'"),
        }
        let t2 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t2.payload, TokenPayload::OBrack));
        let t3 = scanner.popt().expect("no scan error").unwrap();
        match t3.payload {
            TokenPayload::VerilogInt { width, ref value } => {
                assert_eq!(width, None);
                assert_eq!(value.get_bit_count(), 32); // Default width for IrBits is now 32
                assert_eq!(value.to_string(), "bits[32]:255");
            }
            _ => panic!("Expected VerilogInt for 255"),
        }
        // Colon not yet supported
        let t4 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t4.payload, TokenPayload::Colon));
        let t5 = scanner.popt().expect("no scan error").unwrap();
        match t5.payload {
            TokenPayload::VerilogInt { width, ref value } => {
                assert_eq!(width, None);
                assert_eq!(value.get_bit_count(), 32); // Default width for IrBits is now 32
                assert_eq!(value.to_string(), "bits[32]:0");
            }
            _ => panic!("Expected VerilogInt for 0"),
        }
        let t6 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t6.payload, TokenPayload::CBrack));
        let t7 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t7.payload, TokenPayload::Identifier(ref s) if s == "a"));
        let t8 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t8.payload, TokenPayload::Semi));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_annotation_tokenization() {
        let input = r#"(* foo = "bar" *)"#;
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        match t.payload {
            TokenPayload::Annotation { ref key, ref value } => {
                assert_eq!(key, "foo");
                match value {
                    AnnotationValue::String(s) => assert_eq!(s, "bar"),
                    _ => panic!("Expected string annotation value"),
                }
            }
            _ => panic!("Expected annotation token"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_annotation_with_leading_whitespace() {
        let input = r#"  (* foo = "bar" *)"#;
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        match t.payload {
            TokenPayload::Annotation { ref key, ref value } => {
                assert_eq!(key, "foo");
                match value {
                    AnnotationValue::String(s) => assert_eq!(s, "bar"),
                    _ => panic!("Expected string annotation value"),
                }
            }
            _ => panic!("Expected annotation token"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_skip_timescale_at_top() {
        let src = "`timescale 10ps/10ps\nmodule m(); endmodule\n";
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_skip_timescale_with_leading_spaces() {
        let src = "   `timescale 1ns/1ps\nmodule m(); endmodule\n";
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_skip_timescale_after_comment() {
        let src = "// comment\n`timescale 1ns/1ps\nmodule m(); endmodule\n";
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_skip_timescale_inside_module() {
        let src = "module m();\n`timescale 1ns/1ps\nendmodule\n";
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_token_scanner_endmodule_keyword() {
        let input = "endmodule";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(
            t.payload,
            TokenPayload::Keyword(Keyword::Endmodule)
        ));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_parse_wire_decl_with_width() {
        let input = "wire [7:0] foo, bar;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 2);
        let foo = &parser.nets[net_indices[0].0];
        let bar = &parser.nets[net_indices[1].0];
        assert_eq!(foo.width, Some((7, 0)));
        assert_eq!(bar.width, Some((7, 0)));
    }

    #[test]
    fn test_parse_wire_decl_without_width() {
        let input = "wire foo, bar;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 2);
        let foo = &parser.nets[net_indices[0].0];
        let bar = &parser.nets[net_indices[1].0];
        assert_eq!(foo.width, None);
        assert_eq!(bar.width, None);
    }

    #[test]
    fn test_parse_wire_decl_real_world_examples() {
        // Example: wire [255:0] a;
        let input = "wire [255:0] a;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 1);
        let a = &parser.nets[net_indices[0].0];
        assert_eq!(a.width, Some((255, 0)));

        // Example: wire clk;
        let input = "wire clk;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 1);
        let clk = &parser.nets[net_indices[0].0];
        assert_eq!(clk.width, None);

        // Example: wire [26:0] out;
        let input = "wire [26:0] out;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 1);
        let out = &parser.nets[net_indices[0].0];
        assert_eq!(out.width, Some((26, 0)));

        // Example: wire \p0_umul_8[0] ;
        let input = "wire \\p0_umul_8[0] ;";
        let mut parser = Parser::new(TokenScanner::from_str(input));
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 1);
        let p0_net = &parser.nets[net_indices[0].0];
        assert_eq!(
            parser.interner.resolve(p0_net.name).unwrap(),
            "p0_umul_8[0]"
        );
        assert_eq!(p0_net.width, None);
    }

    #[test]
    fn test_parse_wire_decl_with_annotation_between() {
        // Example with annotation before wire decl
        let input = r#"(* src = "foo.sv:3.22-3.23" *)
wire [255:0] a;"#;
        let mut parser = Parser::new(TokenScanner::from_str(input));
        // Skip annotation tokens before wire decl
        loop {
            let peek = parser.scanner.peekt().expect("no scan error");
            match peek {
                Some(tok) => match &tok.payload {
                    TokenPayload::Annotation { .. } | TokenPayload::Comment(_) => {
                        parser.scanner.popt().expect("no scan error");
                    }
                    TokenPayload::Keyword(Keyword::Wire) => break,
                    _ => panic!("unexpected token before wire decl: {:?}", tok.payload),
                },
                None => panic!("unexpected EOF before wire decl"),
            }
        }
        // Now parse the wire decl
        let net_indices = parser.parse_wire_decl().expect("no error");
        assert_eq!(net_indices.len(), 1);
        let a = &parser.nets[net_indices[0].0];
        assert_eq!(a.width, Some((255, 0)));
    }

    #[test]
    fn test_parse_instance_with_bit_and_part_select() {
        // Setup: declare wire [7:0] b;
        let mut parser = Parser::new(TokenScanner::from_str("wire [7:0] b;"));
        parser.parse_wire_decl().expect("no error");
        // Instance with .A(b)
        let input = "TypeName inst (.A(b));";
        let mut parser2 = Parser::new(TokenScanner::from_str(input));
        parser2.nets = parser.nets.clone();
        parser2.interner = parser.interner.clone();
        let inst = parser2.parse_instance().expect("no error");
        assert_eq!(inst.connections.len(), 1);
        match &inst.connections[0].1 {
            NetRef::Simple(_) => {}
            _ => panic!("Expected Simple net ref"),
        }
        // Instance with .A(b[5])
        let input = "TypeName inst (.A(b[5]));";
        let mut parser3 = Parser::new(TokenScanner::from_str(input));
        parser3.nets = parser.nets.clone();
        parser3.interner = parser.interner.clone();
        let inst = parser3.parse_instance().expect("no error");
        assert_eq!(inst.connections.len(), 1);
        match &inst.connections[0].1 {
            NetRef::BitSelect(_, idx) => assert_eq!(*idx, 5),
            _ => panic!("Expected BitSelect net ref"),
        }
        // Instance with .A(b[7:0])
        let input = "TypeName inst (.A(b[7:0]));";
        let mut parser4 = Parser::new(TokenScanner::from_str(input));
        parser4.nets = parser.nets.clone();
        parser4.interner = parser.interner.clone();
        let inst = parser4.parse_instance().expect("no error");
        assert_eq!(inst.connections.len(), 1);
        match &inst.connections[0].1 {
            NetRef::PartSelect(_, msb, lsb) => {
                assert_eq!((*msb, *lsb), (7, 0));
            }
            _ => panic!("Expected PartSelect net ref"),
        }
    }

    #[test]
    fn test_token_scanner_escaped_identifier() {
        let input = "\\escaped-ident ;";
        let mut scanner = TokenScanner::from_str(input);
        let t1 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t1.payload, TokenPayload::Identifier(ref s) if s == "escaped-ident"));
        let t2 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t2.payload, TokenPayload::Semi));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_escaped_identifier_with_special_chars() {
        let input = "\\!@#$%^&*()-+= ;";
        let mut scanner = TokenScanner::from_str(input);
        let t1 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t1.payload, TokenPayload::Identifier(ref s) if s == "!@#$%^&*()-+="));
        let t2 = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t2.payload, TokenPayload::Semi));
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_token_scanner_escaped_identifier_leading_backslash_from_error() {
        // This test is based on the error message provided:
        // wire \p0_umul_8[0] ;
        let input = "\\p0_umul_8[0] ;";
        let mut scanner = TokenScanner::from_str(input);
        let t = scanner.popt().expect("no scan error").unwrap();
        assert!(matches!(t.payload, TokenPayload::Identifier(ref s) if s == "p0_umul_8[0]"));
        assert!(scanner.popt().expect("no scan error").unwrap().payload == TokenPayload::Semi);
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_parse_instance_with_literal_tieoff() {
        // Setup: declare wire b;
        let mut parser = Parser::new(TokenScanner::from_str("wire b;"));
        parser.parse_wire_decl().expect("no error");
        // Instance with .A(1'b0)
        let input = "TypeName inst (.A(1'b0));";
        let mut parser2 = Parser::new(TokenScanner::from_str(input));
        parser2.nets = parser.nets.clone();
        parser2.interner = parser.interner.clone();
        let inst = parser2
            .parse_instance()
            .expect("should parse instance with literal tie-off");
        assert_eq!(inst.connections.len(), 1);
        let (port, netref) = &inst.connections[0];
        let port_name = parser2.interner.resolve(*port).unwrap();
        assert_eq!(port_name, "A");
        match netref {
            NetRef::Literal(bits) => {
                // Should be 1'b0, i.e. width=1, value=0
                assert_eq!(bits.get_bit_count(), 1);
                assert_eq!(bits.to_string(), "bits[1]:0");
            }
            _ => panic!("Expected Literal net ref, got {:?}", netref),
        }
    }

    #[test]
    fn test_token_scanner_verilog_style_literal() {
        let input = "1'b0 8'hFF 16'd42";
        let mut scanner = TokenScanner::from_str(input);
        let t1 = scanner.popt().expect("no scan error").unwrap();
        match t1.payload {
            TokenPayload::VerilogInt { width, ref value } => {
                assert_eq!(width, Some(1));
                assert_eq!(value.get_bit_count(), 1);
                assert_eq!(value.to_string(), "bits[1]:0");
            }
            _ => panic!("Expected VerilogInt for 1'b0"),
        }
        let t2 = scanner.popt().expect("no scan error").unwrap();
        match t2.payload {
            TokenPayload::VerilogInt { width, ref value } => {
                assert_eq!(width, Some(8));
                assert_eq!(value.get_bit_count(), 8);
                assert_eq!(value.to_string(), "bits[8]:255");
            }
            _ => panic!("Expected VerilogInt for 8'hFF"),
        }

        let t3 = scanner.popt().expect("no scan error").unwrap();
        match t3.payload {
            TokenPayload::VerilogInt { width, ref value } => {
                assert_eq!(width, Some(16));
                assert_eq!(value.get_bit_count(), 16);
                assert_eq!(value.to_string(), "bits[16]:42");
            }
            _ => panic!("Expected VerilogInt for 16'd42"),
        }
        assert!(scanner.popt().expect("no scan error").is_none());
    }

    #[test]
    fn test_implicit_net_created_on_use_in_instance_connection() {
        // Module where a net is only used in instance connections (no explicit
        // wire declaration). With implicit nets enabled, parsing should succeed
        // and the net should appear in the global nets array.
        let src = r#"
module m(a, y);
  input a;
  output y;
  DummyCell u0 (.in_valid(a), .in(a), .out_valid(data_valid_d));
  DummyCell u1 (.in_valid(data_valid_d), .in(a), .out_valid(y));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        // There should be a net named data_valid_d created implicitly.
        let dv_nets: Vec<&Net> = parser
            .nets
            .iter()
            .filter(|n| parser.interner.resolve(n.name).unwrap() == "data_valid_d")
            .collect();
        assert_eq!(dv_nets.len(), 1);
        assert_eq!(dv_nets[0].width, Some((0, 0)));
    }

    #[test]
    fn test_concat_with_trailing_verilog_literal_and_close_brace() {
        // Regression test: previously, the '}' in "3'b0})" could be swallowed
        // into the VerilogInt token, causing the concat parser to fail with
        // "expected '}' to close concatenation".
        let src = r#"
module m(a, y);
  input [1:0] a;
  output [3:0] y;
  MYCELL u1 (.Y({a, 3'b0}));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let insts = &modules[0].instances;
        assert_eq!(insts.len(), 1);
        let (port_sym, netref) = &insts[0].connections[0];
        let port_name = parser.interner.resolve(*port_sym).unwrap();
        assert_eq!(port_name, "Y");
        match netref {
            NetRef::Concat(elems) => {
                assert_eq!(elems.len(), 2);
            }
            other => panic!("expected Concat for .Y connection, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_net_errors_when_implicit_nets_disabled() {
        // When implicit nets are disabled, undeclared nets should still
        // produce a clear error rather than being synthesized implicitly.
        let src = r#"
module m(a, y);
  input a;
  output y;
  DummyCell u0 (.in_valid(a), .in(a), .out_valid(data_valid_d));
endmodule
"#;
        let mut parser = Parser::new_with_options(
            TokenScanner::from_str(src),
            /* allow_implicit_nets= */ false,
        );
        let err = parser
            .parse_file()
            .expect_err("should error on unknown net");
        assert!(
            err.message.contains("not declared as wire"),
            "unexpected error message: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_instance_unconnected_output() {
        let src = r#"
module m(a, y);
  input a;
  output y;
  wire a, y;
  INVX1 u1 (.A(a), .Y());
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let insts = &modules[0].instances;
        assert_eq!(insts.len(), 1);
        let conn = &insts[0].connections[1].1;
        match conn {
            NetRef::Unconnected => {}
            _ => panic!("expected Unconnected for .Y()"),
        }
    }

    #[test]
    fn test_parse_instance_unconnected_input_and_comments() {
        let src = r#"
module m(y);
  output y;
  wire y;
  AND2 u1 (.A(/*c*/ ), .B( // c
 ), .Y(y));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let insts = &modules[0].instances;
        assert_eq!(insts.len(), 1);
        // .A() should be Unconnected
        match &insts[0].connections[0].1 {
            NetRef::Unconnected => {}
            other => panic!("expected Unconnected for .A(), got {:?}", other),
        }
        // .B() should be Unconnected
        match &insts[0].connections[1].1 {
            NetRef::Unconnected => {}
            other => panic!("expected Unconnected for .B(), got {:?}", other),
        }
    }

    #[test]
    fn test_non_ansi_body_ports_create_nets() {
        let src = r#"
module m(a, y);
  input a;
  output y;
  INVX1 u1 (.A(a), .Y(y));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        // Ensure nets for a and y exist with no widths
        let a_net = parser
            .nets
            .iter()
            .find(|n| parser.interner.resolve(n.name).unwrap() == "a")
            .unwrap();
        let y_net = parser
            .nets
            .iter()
            .find(|n| parser.interner.resolve(n.name).unwrap() == "y")
            .unwrap();
        assert_eq!(a_net.width, None);
        assert_eq!(y_net.width, None);
    }

    #[test]
    fn test_non_ansi_vector_output_and_bitselect() {
        let src = r#"
module m(a, out);
  input a;
  output [7:0] out;
  Type inst (.A(out[3]));
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let out_net = parser
            .nets
            .iter()
            .find(|n| parser.interner.resolve(n.name).unwrap() == "out")
            .unwrap();
        assert_eq!(out_net.width, Some((7, 0)));
        // And the instance should have one connection with a BitSelect
        let inst = &modules[0].instances[0];
        match inst.connections[0].1 {
            NetRef::BitSelect(_, idx) => assert_eq!(idx, 3),
            ref other => panic!("expected BitSelect, got {:?}", other),
        }
    }

    #[test]
    fn test_duplicate_wire_after_port_decl_same_width_ok() {
        let src = r#"
module m(a);
  input [3:0] a;
  wire [3:0] a;
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        let a_nets: Vec<&Net> = parser
            .nets
            .iter()
            .filter(|n| parser.interner.resolve(n.name).unwrap() == "a")
            .collect();
        assert_eq!(a_nets.len(), 1);
        assert_eq!(a_nets[0].width, Some((3, 0)));
    }

    #[test]
    fn test_conflicting_widths_error() {
        let src = r#"
module m(out);
  output [7:0] out;
  wire [3:0] out;
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let err = parser
            .parse_file()
            .expect_err("should error on width conflict");
        assert!(err.message.contains("conflicting widths for net 'out'"));
        assert!(
            err.message.contains("previously determined width was"),
            "error message should mention previous width location for diagnostics: {}",
            err.message
        );
    }

    #[test]
    fn test_conflicting_widths_across_modules_are_ok() {
        // Two separate modules may legitimately use the same net name with
        // different widths; width consistency is a per-module property, not
        // global to the file.
        let src = r#"
module m1(gen_in);
  input [7:0] gen_in;
endmodule

module m2(gen_in);
  input [3:0] gen_in;
endmodule
"#;
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse should succeed");
        assert_eq!(modules.len(), 2);
        // Sanity check: each module sees exactly one net named gen_in, with
        // its own width.
        let gen_in_nets: Vec<&Net> = parser
            .nets
            .iter()
            .filter(|n| parser.interner.resolve(n.name).unwrap() == "gen_in")
            .collect();
        assert_eq!(gen_in_nets.len(), 2);
        assert!(gen_in_nets.iter().any(|n| n.width == Some((7, 0))));
        assert!(gen_in_nets.iter().any(|n| n.width == Some((3, 0))));
    }
}
