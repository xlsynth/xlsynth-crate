// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokKind {
    Ident(String),
    Number(String),
    /// `casez` pattern literal like `2'b?1` (MSB-first bits, may include '?').
    CasezPattern {
        width: u32,
        bits_msb: String,
    },

    KwModule,
    KwEndmodule,
    KwInput,
    KwOutput,
    KwParameter,
    KwWire,
    KwLogic,
    KwSigned,
    KwReg,
    KwAssign,
    KwFunction,
    KwEndfunction,
    KwAutomatic,
    KwUnique,
    KwCasez,
    KwEndcase,
    KwAlwaysFf,
    KwPosedge,
    KwBegin,
    KwEnd,
    KwIf,
    KwElse,

    At,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Colon,
    Semi,
    Comma,
    Eq,  // =
    Leq, // <=

    // Expression-ish single-char tokens we pass through into expression slices.
    Other(char),

    End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tok {
    pub kind: TokKind,
    pub start: usize,
    pub end: usize,
}

pub struct Lexer<'a> {
    s: &'a str,
    idx: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Self {
        Self { s, idx: 0 }
    }

    pub fn next(&mut self) -> Result<Tok> {
        self.skip_ws_and_comments();
        let start = self.idx;
        if self.idx >= self.s.len() {
            return Ok(Tok {
                kind: TokKind::End,
                start,
                end: start,
            });
        }

        let b = self.s.as_bytes()[self.idx];
        let kind = match b {
            b'@' => {
                self.idx += 1;
                TokKind::At
            }
            b'(' => {
                self.idx += 1;
                TokKind::LParen
            }
            b')' => {
                self.idx += 1;
                TokKind::RParen
            }
            b'[' => {
                self.idx += 1;
                TokKind::LBracket
            }
            b']' => {
                self.idx += 1;
                TokKind::RBracket
            }
            b':' => {
                self.idx += 1;
                TokKind::Colon
            }
            b';' => {
                self.idx += 1;
                TokKind::Semi
            }
            b',' => {
                self.idx += 1;
                TokKind::Comma
            }
            b'=' => {
                self.idx += 1;
                TokKind::Eq
            }
            b'<' => {
                if self.s[self.idx..].starts_with("<=") {
                    self.idx += 2;
                    TokKind::Leq
                } else {
                    self.idx += 1;
                    TokKind::Other('<')
                }
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                let ident = self.lex_ident();
                match ident.as_str() {
                    "module" => TokKind::KwModule,
                    "endmodule" => TokKind::KwEndmodule,
                    "input" => TokKind::KwInput,
                    "output" => TokKind::KwOutput,
                    "parameter" => TokKind::KwParameter,
                    "wire" => TokKind::KwWire,
                    "logic" => TokKind::KwLogic,
                    "signed" => TokKind::KwSigned,
                    "reg" => TokKind::KwReg,
                    "assign" => TokKind::KwAssign,
                    "function" => TokKind::KwFunction,
                    "endfunction" => TokKind::KwEndfunction,
                    "automatic" => TokKind::KwAutomatic,
                    "unique" => TokKind::KwUnique,
                    "casez" => TokKind::KwCasez,
                    "endcase" => TokKind::KwEndcase,
                    "always_ff" => TokKind::KwAlwaysFf,
                    "posedge" => TokKind::KwPosedge,
                    "begin" => TokKind::KwBegin,
                    "end" => TokKind::KwEnd,
                    "if" => TokKind::KwIf,
                    "else" => TokKind::KwElse,
                    _ => TokKind::Ident(ident),
                }
            }
            b'0'..=b'9' | b'\'' => {
                let n = self.lex_numberish();
                if let Some((width, bits_msb)) = parse_casez_pattern_literal(&n) {
                    TokKind::CasezPattern { width, bits_msb }
                } else {
                    TokKind::Number(n)
                }
            }
            _ => {
                let c = self.s[self.idx..].chars().next().unwrap();
                self.idx += c.len_utf8();
                TokKind::Other(c)
            }
        };
        let end = self.idx;
        Ok(Tok { kind, start, end })
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.s[self.idx..].chars().next() {
                if c.is_whitespace() {
                    self.idx += c.len_utf8();
                } else {
                    break;
                }
                if self.idx >= self.s.len() {
                    return;
                }
            }

            if self.s[self.idx..].starts_with("//") {
                while self.idx < self.s.len() {
                    let c = self.s[self.idx..].chars().next().unwrap();
                    self.idx += c.len_utf8();
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }
            if self.s[self.idx..].starts_with("/*") {
                self.idx += 2;
                while self.idx < self.s.len() && !self.s[self.idx..].starts_with("*/") {
                    let c = self.s[self.idx..].chars().next().unwrap();
                    self.idx += c.len_utf8();
                }
                if self.s[self.idx..].starts_with("*/") {
                    self.idx += 2;
                }
                continue;
            }
            break;
        }
    }

    fn lex_ident(&mut self) -> String {
        let start = self.idx;
        while self.idx < self.s.len() {
            let c = self.s[self.idx..].chars().next().unwrap();
            if c == '_' || c.is_ascii_alphanumeric() {
                self.idx += c.len_utf8();
            } else {
                break;
            }
        }
        self.s[start..self.idx].to_string()
    }

    fn lex_numberish(&mut self) -> String {
        let start = self.idx;
        while self.idx < self.s.len() {
            let c = self.s[self.idx..].chars().next().unwrap();
            if c.is_whitespace() {
                break;
            }
            // NOTE: Do NOT break on `?` so `2'b?1` stays a single token for `casez`
            // patterns.
            if matches!(c, ';' | ',' | ')' | '(' | '[' | ']' | '{' | '}' | ':') {
                break;
            }
            self.idx += c.len_utf8();
        }
        self.s[start..self.idx].to_string()
    }
}

pub fn lex_all(s: &str) -> Result<Vec<Tok>> {
    let mut l = Lexer::new(s);
    let mut toks = Vec::new();
    loop {
        let t = l.next()?;
        let end = matches!(t.kind, TokKind::End);
        toks.push(t);
        if end {
            return Ok(toks);
        }
        if toks.len() > 200_000 {
            return Err(Error::Parse("SV lexer runaway".to_string()));
        }
    }
}

/// Lexer helper: interpret a number-ish lexeme as a `casez` pattern
/// (`<w>'b<bits>`), returning `(width, bits_msb)`.
fn parse_casez_pattern_literal(s: &str) -> Option<(u32, String)> {
    // Strict subset: parse only `<width>'b<bits>` with optional underscores in
    // bits. Accept `?` in bits (casez patterns), and accept 0/1/x/z as-is.
    let compact: String = s.chars().filter(|c| *c != '_').collect();
    let s = compact.as_str();
    let tick = s.find('\'')?;
    let w_str = s[..tick].trim();
    if w_str.is_empty() {
        return None;
    }
    let width: u32 = w_str.parse().ok()?;
    let rest = s[tick + 1..].trim();
    let bits = rest.strip_prefix('b').or_else(|| rest.strip_prefix('B'))?;
    let bits = bits.trim();
    if bits.is_empty() {
        return None;
    }
    // Only treat as casez pattern token if it contains '?' or is used in a casez
    // context. We can still safely promote all binary-sized literals here; keep
    // it strict and simple.
    Some((width, bits.to_string()))
}
