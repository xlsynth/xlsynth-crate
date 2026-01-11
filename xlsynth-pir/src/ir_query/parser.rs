// SPDX-License-Identifier: Apache-2.0

//! Parser for the lightweight IR query language.

use super::{MatcherExpr, MatcherKind, QueryExpr};

pub struct QueryParser<'a> {
    input: &'a str,
    chars: Vec<char>,
    pos: usize,
}

impl<'a> QueryParser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn parse_expr(&mut self) -> Result<QueryExpr, String> {
        self.skip_ws();
        match self.peek() {
            Some('$') => {
                self.bump();
                let ident = self.parse_ident("matcher name")?;
                let kind = match ident.as_str() {
                    "anycmp" => MatcherKind::AnyCmp,
                    "anymul" => MatcherKind::AnyMul,
                    _ => return Err(self.error(&format!("unknown matcher ${}", ident))),
                };
                let user_count = if self.peek() == Some('[') {
                    Some(self.parse_user_constraint()?)
                } else {
                    None
                };
                self.skip_ws();
                self.expect('(')?;
                let args = self.parse_args()?;
                self.expect(')')?;
                Ok(QueryExpr::Matcher(MatcherExpr {
                    kind,
                    user_count,
                    args,
                }))
            }
            Some(_) => {
                let ident = self.parse_ident("placeholder")?;
                Ok(QueryExpr::Placeholder(ident))
            }
            None => Err(self.error("expected query expression")),
        }
    }

    pub fn is_done(&self) -> bool {
        self.peek().is_none()
    }

    pub fn error_at(&self, msg: &str) -> String {
        self.error(msg)
    }

    pub fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }

    fn parse_args(&mut self) -> Result<Vec<QueryExpr>, String> {
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(')') {
            return Ok(args);
        }
        loop {
            let expr = self.parse_expr()?;
            args.push(expr);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(')') => break,
                _ => return Err(self.error("expected ',' or ')'")),
            }
        }
        Ok(args)
    }

    fn parse_user_constraint(&mut self) -> Result<usize, String> {
        self.expect('[')?;
        self.skip_ws();
        let number = self.parse_number("user count")?;
        self.skip_ws();
        if self.peek() != Some('u') {
            return Err(self.error("expected user count suffix 'u'"));
        }
        self.bump();
        self.skip_ws();
        self.expect(']')?;
        Ok(number)
    }

    fn parse_number(&mut self, ctx: &str) -> Result<usize, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        self.input[start..self.pos]
            .parse::<usize>()
            .map_err(|e| self.error(&format!("invalid {}: {}", ctx, e)))
    }

    fn parse_ident(&mut self, ctx: &str) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        self.skip_ws();
        if self.peek() == Some(ch) {
            self.bump();
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", ch)))
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn error(&self, msg: &str) -> String {
        format!("{} at byte {}", msg, self.pos)
    }
}
