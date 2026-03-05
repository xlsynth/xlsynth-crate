// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::Result;
use crate::ast_spanned::SpannedTok;
use crate::lexer::Token;
use crate::sv_ast::Span;

pub struct LexerSpanned<'a> {
    s: &'a str,
    idx: usize,
}

impl<'a> LexerSpanned<'a> {
    pub fn new(s: &'a str) -> Self {
        Self { s, idx: 0 }
    }

    pub fn next_token(&mut self) -> Result<SpannedTok> {
        self.skip_ws();
        let start = self.idx;
        if self.idx >= self.s.len() {
            return Ok(SpannedTok {
                tok: Token::End,
                span: Span { start, end: start },
            });
        }

        let b = self.peek_byte().unwrap();
        let tok = match b {
            b'$' => Token::Ident(self.lex_dollar_ident()?),
            b'\'' => Token::Number(self.lex_numberish()),
            b'(' => {
                self.idx += 1;
                Token::LParen
            }
            b')' => {
                self.idx += 1;
                Token::RParen
            }
            b'{' => {
                self.idx += 1;
                Token::LBrace
            }
            b'}' => {
                self.idx += 1;
                Token::RBrace
            }
            b'[' => {
                self.idx += 1;
                Token::LBracket
            }
            b']' => {
                self.idx += 1;
                Token::RBracket
            }
            b',' => {
                self.idx += 1;
                Token::Comma
            }
            b'?' => {
                self.idx += 1;
                Token::Question
            }
            b':' => {
                self.idx += 1;
                Token::Colon
            }
            b'+' => {
                if self.consume_str("+:") {
                    Token::PlusColon
                } else {
                    self.idx += 1;
                    Token::Plus
                }
            }
            b'-' => {
                if self.consume_str("-:") {
                    Token::MinusColon
                } else {
                    self.idx += 1;
                    Token::Minus
                }
            }
            b'*' => {
                self.idx += 1;
                Token::Star
            }
            b'/' => {
                self.idx += 1;
                Token::Slash
            }
            b'%' => {
                self.idx += 1;
                Token::Percent
            }
            b'!' => {
                if self.consume_str("!==") {
                    Token::BangEqEq
                } else if self.consume_str("!=") {
                    Token::BangEq
                } else {
                    self.idx += 1;
                    Token::Bang
                }
            }
            b'~' => {
                self.idx += 1;
                Token::Tilde
            }
            b'&' => {
                if self.consume_str("&&") {
                    Token::AndAnd
                } else {
                    self.idx += 1;
                    Token::And
                }
            }
            b'|' => {
                if self.consume_str("||") {
                    Token::OrOr
                } else {
                    self.idx += 1;
                    Token::Or
                }
            }
            b'^' => {
                self.idx += 1;
                Token::Caret
            }
            b'<' => {
                if self.consume_str("<<<") {
                    return Err(Error::Lex("unexpected '<<<' (not supported)".to_string()));
                }
                if self.consume_str("<<") {
                    Token::Shl
                } else if self.consume_str("<=") {
                    Token::Le
                } else {
                    self.idx += 1;
                    Token::Lt
                }
            }
            b'>' => {
                if self.consume_str(">>>") {
                    Token::Sshr
                } else if self.consume_str(">>") {
                    Token::Shr
                } else if self.consume_str(">=") {
                    Token::Ge
                } else {
                    self.idx += 1;
                    Token::Gt
                }
            }
            b'=' => {
                if self.consume_str("===") {
                    Token::EqEqEq
                } else if self.consume_str("==") {
                    Token::EqEq
                } else {
                    return Err(Error::Lex("unexpected '='".to_string()));
                }
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => Token::Ident(self.lex_ident()),
            b'0'..=b'9' => Token::Number(self.lex_numberish()),
            _ => {
                return Err(Error::Lex(format!(
                    "unexpected character {:?} at byte {}",
                    self.peek_char(),
                    self.idx
                )));
            }
        };
        let end = self.idx;
        Ok(SpannedTok {
            tok,
            span: Span { start, end },
        })
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.idx += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn lex_ident(&mut self) -> String {
        let start = self.idx;
        while let Some(c) = self.peek_char() {
            if c == '_' || c.is_ascii_alphanumeric() {
                self.idx += c.len_utf8();
            } else {
                break;
            }
        }
        self.s[start..self.idx].to_string()
    }

    fn lex_dollar_ident(&mut self) -> Result<String> {
        let start = self.idx;
        self.idx += 1; // '$'
        let mut saw_ident = false;
        while let Some(c) = self.peek_char() {
            if c == '_' || c.is_ascii_alphanumeric() {
                self.idx += c.len_utf8();
                saw_ident = true;
            } else {
                break;
            }
        }
        if !saw_ident {
            return Err(Error::Lex(format!(
                "unexpected standalone '$' at byte {}",
                start
            )));
        }
        Ok(self.s[start..self.idx].to_string())
    }

    fn lex_numberish(&mut self) -> String {
        let start = self.idx;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '\'' {
                self.idx += c.len_utf8();
            } else {
                break;
            }
        }
        self.s[start..self.idx].to_string()
    }

    fn consume_str(&mut self, lit: &str) -> bool {
        if self.s[self.idx..].starts_with(lit) {
            self.idx += lit.len();
            true
        } else {
            false
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.s[self.idx..].chars().next()
    }

    fn peek_byte(&self) -> Option<u8> {
        self.s.as_bytes().get(self.idx).copied()
    }
}
