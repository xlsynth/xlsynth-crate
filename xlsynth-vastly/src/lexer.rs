// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Ident(String),
    Number(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Question,
    Colon,
    PlusColon,
    MinusColon,
    Bang,
    Tilde,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    AndAnd,
    OrOr,
    And,
    Or,
    Caret,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Sshr,
    EqEq,
    BangEq,
    EqEqEq,
    BangEqEq,
    End,
}

pub struct Lexer<'a> {
    s: &'a str,
    idx: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Self {
        Self { s, idx: 0 }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_ws();
        if self.idx >= self.s.len() {
            return Ok(Token::End);
        }

        let b = self.peek_byte().unwrap();
        match b {
            b'$' => Ok(Token::Ident(self.lex_dollar_ident()?)),
            b'\'' => Ok(Token::Number(self.lex_numberish())),
            b'(' => {
                self.idx += 1;
                Ok(Token::LParen)
            }
            b')' => {
                self.idx += 1;
                Ok(Token::RParen)
            }
            b'{' => {
                self.idx += 1;
                Ok(Token::LBrace)
            }
            b'}' => {
                self.idx += 1;
                Ok(Token::RBrace)
            }
            b'[' => {
                self.idx += 1;
                Ok(Token::LBracket)
            }
            b']' => {
                self.idx += 1;
                Ok(Token::RBracket)
            }
            b',' => {
                self.idx += 1;
                Ok(Token::Comma)
            }
            b'?' => {
                self.idx += 1;
                Ok(Token::Question)
            }
            b':' => {
                self.idx += 1;
                Ok(Token::Colon)
            }
            b'+' => {
                if self.consume_str("+:") {
                    Ok(Token::PlusColon)
                } else {
                    self.idx += 1;
                    Ok(Token::Plus)
                }
            }
            b'-' => {
                if self.consume_str("-:") {
                    Ok(Token::MinusColon)
                } else {
                    self.idx += 1;
                    Ok(Token::Minus)
                }
            }
            b'*' => {
                self.idx += 1;
                Ok(Token::Star)
            }
            b'/' => {
                self.idx += 1;
                Ok(Token::Slash)
            }
            b'%' => {
                self.idx += 1;
                Ok(Token::Percent)
            }
            b'!' => {
                if self.consume_str("!==") {
                    return Ok(Token::BangEqEq);
                }
                if self.consume_str("!=") {
                    return Ok(Token::BangEq);
                }
                self.idx += 1;
                Ok(Token::Bang)
            }
            b'~' => {
                self.idx += 1;
                Ok(Token::Tilde)
            }
            b'&' => {
                if self.consume_str("&&") {
                    Ok(Token::AndAnd)
                } else {
                    self.idx += 1;
                    Ok(Token::And)
                }
            }
            b'|' => {
                if self.consume_str("||") {
                    Ok(Token::OrOr)
                } else {
                    self.idx += 1;
                    Ok(Token::Or)
                }
            }
            b'^' => {
                self.idx += 1;
                Ok(Token::Caret)
            }
            b'<' => {
                if self.consume_str("<<<") {
                    return Err(Error::Lex("unexpected '<<<' (not supported)".to_string()));
                }
                if self.consume_str("<<") {
                    return Ok(Token::Shl);
                }
                if self.consume_str("<=") {
                    return Ok(Token::Le);
                }
                self.idx += 1;
                Ok(Token::Lt)
            }
            b'>' => {
                if self.consume_str(">>>") {
                    return Ok(Token::Sshr);
                }
                if self.consume_str(">>") {
                    return Ok(Token::Shr);
                }
                if self.consume_str(">=") {
                    return Ok(Token::Ge);
                }
                self.idx += 1;
                Ok(Token::Gt)
            }
            b'=' => {
                if self.consume_str("===") {
                    return Ok(Token::EqEqEq);
                }
                if self.consume_str("==") {
                    return Ok(Token::EqEq);
                }
                Err(Error::Lex("unexpected '='".to_string()))
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => Ok(Token::Ident(self.lex_ident())),
            b'0'..=b'9' => Ok(Token::Number(self.lex_numberish())),
            _ => Err(Error::Lex(format!(
                "unexpected character {:?} at byte {}",
                self.peek_char(),
                self.idx
            ))),
        }
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

    /// Lexes a Verilog-ish numeric literal chunk, conservatively.
    ///
    /// Examples: `123`, `4'b10xz`, `8'shFF`, `'hff`, `16'd_12`.
    fn lex_numberish(&mut self) -> String {
        let start = self.idx;
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                break;
            }
            // Stop on clear expression delimiters.
            if matches!(c, '(' | ')' | '{' | '}' | '[' | ']' | ',' | '?' | ':') {
                break;
            }
            // Stop on operator prefix chars.
            if matches!(
                c,
                '!' | '~' | '&' | '|' | '=' | '+' | '-' | '*' | '/' | '%' | '<' | '>' | '^'
            ) {
                break;
            }

            // Otherwise keep consuming; parser will validate.
            self.idx += c.len_utf8();
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

    fn peek_byte(&self) -> Option<u8> {
        self.s.as_bytes().get(self.idx).copied()
    }

    fn peek_char(&self) -> Option<char> {
        self.s[self.idx..].chars().next()
    }
}
