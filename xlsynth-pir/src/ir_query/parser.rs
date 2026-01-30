// SPDX-License-Identifier: Apache-2.0

//! Parser for the lightweight IR query language.

use crate::ir;

use super::{
    MatcherExpr, MatcherKind, NamedArg, NamedArgValue, NumericExpr, PlaceholderExpr, QueryExpr,
};

pub struct QueryParser<'a> {
    bytes: &'a [u8],
    pos: usize, // byte offset
}

impl<'a> QueryParser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    pub fn parse_expr(&mut self) -> Result<QueryExpr, String> {
        self.skip_ws();

        // Variadic wildcard used inside operator arg lists, e.g. `nor(..., a, ...)`.
        if self
            .bytes
            .get(self.pos..)
            .is_some_and(|s| s.starts_with(b"..."))
        {
            self.pos += 3;
            return Ok(QueryExpr::Ellipsis);
        }

        match self.peek() {
            Some(b'$') => {
                self.bump();
                let ident = self.parse_ident("matcher name")?;
                let kind = match ident.as_str() {
                    "anycmp" => MatcherKind::AnyCmp,
                    "anymul" => MatcherKind::AnyMul,
                    "users" => MatcherKind::Users,
                    "width" => MatcherKind::Width,
                    "all_ones" => MatcherKind::AllOnes,
                    "mask_low" => MatcherKind::MaskLow,
                    _ => return Err(self.error(&format!("unknown matcher ${}", ident))),
                };
                let user_count = if matches!(kind, MatcherKind::AnyCmp | MatcherKind::AnyMul) {
                    if self.peek() == Some(b'[') {
                        Some(self.parse_user_constraint()?)
                    } else {
                        None
                    }
                } else if self.peek() == Some(b'[') {
                    return Err(
                        self.error("matchers only support user-count constraints like [1u]")
                    );
                } else {
                    None
                };
                self.skip_ws();
                self.expect('(')?;
                if matches!(kind, MatcherKind::MaskLow) {
                    let expr = self.parse_numeric_expr()?;
                    self.expect(')')?;
                    return Ok(QueryExpr::Matcher(MatcherExpr {
                        kind,
                        user_count,
                        args: vec![QueryExpr::Numeric(expr)],
                        named_args: vec![],
                    }));
                }
                let parsed_args = self.parse_args()?;
                self.expect(')')?;
                if !parsed_args.named_args.is_empty() {
                    return Err(self.error("matchers do not support named arguments"));
                }
                let expected_arity = expected_arity(&kind);
                if parsed_args.args.len() != expected_arity {
                    return Err(self.error(&format!(
                        "matcher ${} expects {} arguments; got {}. Use '_' as a wildcard argument if needed",
                        ident,
                        expected_arity,
                        parsed_args.args.len()
                    )));
                }
                Ok(QueryExpr::Matcher(MatcherExpr {
                    kind,
                    user_count,
                    args: parsed_args.args,
                    named_args: parsed_args.named_args,
                }))
            }
            Some(c) if c.is_ascii_digit() => {
                let number = self.parse_u64("number")?;
                Ok(QueryExpr::Number(number))
            }
            Some(_) => {
                let ident = self.parse_ident("placeholder or operator")?;
                self.skip_ws();

                // Placeholder type constraint: `name: bits[1]`, `x: token`, etc.
                if self.peek() == Some(b':') {
                    self.bump();
                    let ty = self.parse_type("type constraint")?;
                    return Ok(QueryExpr::Placeholder(PlaceholderExpr {
                        name: ident,
                        ty: Some(ty),
                    }));
                }

                // If an identifier is followed by a bracket clause and/or an argument
                // list, interpret it as an operator matcher (e.g. `add(x, y)`).
                //
                // Otherwise it is a node placeholder binding (e.g. `x`, `y`, `_`).
                if self.peek() != Some(b'[') && self.peek() != Some(b'(') {
                    return Ok(QueryExpr::Placeholder(PlaceholderExpr {
                        name: ident,
                        ty: None,
                    }));
                }

                let mut user_count: Option<usize> = None;
                let mut predicate: Option<String> = None;
                if self.peek() == Some(b'[') {
                    match self.parse_bracket_clause()? {
                        BracketClause::UserCount(n) => user_count = Some(n),
                        BracketClause::Ident(s) => predicate = Some(s),
                    }
                }

                self.skip_ws();
                self.expect('(')?;
                let parsed_args = self.parse_args()?;
                self.expect(')')?;

                let kind = MatcherKind::from_opname_and_predicate(&ident, predicate)
                    .map_err(|e| self.error(&e))?;

                if matches!(kind, MatcherKind::Literal { .. }) {
                    if !parsed_args.named_args.is_empty() {
                        return Err(self.error("literal does not support named arguments"));
                    }
                    if parsed_args.args.len() != 1 {
                        return Err(self.error(&format!(
                            "literal expects 1 argument; got {}. Use '_' as a wildcard argument if needed",
                            parsed_args.args.len()
                        )));
                    }
                }

                if matches!(kind, MatcherKind::Msb) {
                    if !parsed_args.named_args.is_empty() {
                        return Err(self.error("msb does not support named arguments"));
                    }
                    if parsed_args.args.len() != 1 {
                        return Err(self.error(&format!(
                            "msb expects 1 argument; got {}. Use '_' as a wildcard argument if needed",
                            parsed_args.args.len()
                        )));
                    }
                }

                Ok(QueryExpr::Matcher(MatcherExpr {
                    kind,
                    user_count,
                    args: parsed_args.args,
                    named_args: parsed_args.named_args,
                }))
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
        while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
            self.bump();
        }
    }

    fn parse_args(&mut self) -> Result<ParsedArgs, String> {
        let mut args = Vec::new();
        let mut named_args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b')') {
            return Ok(ParsedArgs { args, named_args });
        }
        loop {
            if let Some(named_arg) = self.parse_named_arg()? {
                named_args.push(named_arg);
            } else {
                let expr = self.parse_expr()?;
                args.push(expr);
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b')') => break,
                _ => return Err(self.error("expected ',' or ')'")),
            }
        }
        Ok(ParsedArgs { args, named_args })
    }

    fn parse_user_constraint(&mut self) -> Result<usize, String> {
        self.expect('[')?;
        self.skip_ws();
        let number = self.parse_number("user count")?;
        self.skip_ws();
        if self.peek() != Some(b'u') {
            return Err(self.error("expected user count suffix 'u'"));
        }
        self.bump();
        self.skip_ws();
        self.expect(']')?;
        Ok(number)
    }

    fn parse_bracket_clause(&mut self) -> Result<BracketClause, String> {
        self.expect('[')?;
        self.skip_ws();
        match self.peek() {
            Some(c) if c.is_ascii_digit() => {
                let number = self.parse_number("user count")?;
                self.skip_ws();
                if self.peek() != Some(b'u') {
                    return Err(self.error("expected user count suffix 'u'"));
                }
                self.bump();
                self.skip_ws();
                self.expect(']')?;
                Ok(BracketClause::UserCount(number))
            }
            Some(_) => {
                let ident = self.parse_ident("bracket clause")?;
                self.skip_ws();
                self.expect(']')?;
                Ok(BracketClause::Ident(ident))
            }
            None => Err(self.error("expected bracket clause")),
        }
    }

    fn parse_number(&mut self, ctx: &str) -> Result<usize, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("numeric slice must be valid UTF-8");
        s.parse::<usize>()
            .map_err(|e| self.error(&format!("invalid {}: {}", ctx, e)))
    }

    fn parse_u64(&mut self, ctx: &str) -> Result<u64, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("numeric slice must be valid UTF-8");
        s.parse::<u64>()
            .map_err(|e| self.error(&format!("invalid {}: {}", ctx, e)))
    }

    fn parse_ident(&mut self, ctx: &str) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_') {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error(&format!("expected {}", ctx)));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("identifier slice must be valid UTF-8");
        Ok(s.to_string())
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        assert!(ch.is_ascii(), "query parser only expects ASCII delimiters");
        let b = ch as u8;
        self.skip_ws();
        if self.peek() == Some(b) {
            self.bump();
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", ch)))
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn error(&self, msg: &str) -> String {
        format!("{} at byte {}", msg, self.pos)
    }

    fn parse_named_arg(&mut self) -> Result<Option<NamedArg>, String> {
        self.skip_ws();
        let start = self.pos;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {}
            _ => return Ok(None),
        }
        let ident = self.parse_ident("named argument")?;
        self.skip_ws();
        if self.peek() != Some(b'=') {
            self.pos = start;
            return Ok(None);
        }
        self.bump();
        self.skip_ws();
        let value = match self.peek() {
            Some(b'[') => NamedArgValue::ExprList(self.parse_expr_list()?),
            _ => {
                if ident == "lsb_prio" {
                    let expr = self.parse_expr()?;
                    return self.parse_bool_named_arg(ident, expr);
                }

                if ident == "width" || ident == "start" {
                    // Width/start named args accept numeric expressions like:
                    // - `0`
                    // - `$width(x)`
                    // - `$width(x)-1`
                    //
                    // Keep `_` as a wildcard (NamedArgValue::Any) by routing it
                    // through the normal expression parser.
                    match self.peek() {
                        Some(c) if c.is_ascii_digit() => {
                            let expr = self.parse_numeric_expr()?;
                            NamedArgValue::Expr(QueryExpr::Numeric(expr))
                        }
                        Some(b'$') | Some(b'(') => {
                            let expr = self.parse_numeric_expr()?;
                            NamedArgValue::Expr(QueryExpr::Numeric(expr))
                        }
                        _ => {
                            let expr = self.parse_expr()?;
                            match expr {
                                QueryExpr::Placeholder(ref p)
                                    if p.name == "_" && p.ty.is_none() =>
                                {
                                    NamedArgValue::Any
                                }
                                QueryExpr::Number(number) => {
                                    let number = usize::try_from(number).map_err(|_| {
                                        self.error("named argument number does not fit in usize")
                                    })?;
                                    NamedArgValue::Number(number)
                                }
                                _ => NamedArgValue::Expr(expr),
                            }
                        }
                    }
                } else {
                    let expr = self.parse_expr()?;
                    match expr {
                        QueryExpr::Placeholder(ref p) if p.name == "_" && p.ty.is_none() => {
                            NamedArgValue::Any
                        }
                        QueryExpr::Number(number) if ident == "width" || ident == "start" => {
                            let number = usize::try_from(number).map_err(|_| {
                                self.error("named argument number does not fit in usize")
                            })?;
                            NamedArgValue::Number(number)
                        }
                        _ => NamedArgValue::Expr(expr),
                    }
                }
            }
        };
        Ok(Some(NamedArg { name: ident, value }))
    }

    fn parse_expr_list(&mut self) -> Result<Vec<QueryExpr>, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(items);
        }
        loop {
            let expr = self.parse_expr()?;
            items.push(expr);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
        Ok(items)
    }

    fn parse_numeric_expr(&mut self) -> Result<NumericExpr, String> {
        let mut expr = self.parse_numeric_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.bump();
                    let rhs = self.parse_numeric_factor()?;
                    expr = NumericExpr::Add(Box::new(expr), Box::new(rhs));
                }
                Some(b'-') => {
                    self.bump();
                    let rhs = self.parse_numeric_factor()?;
                    expr = NumericExpr::Sub(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_numeric_factor(&mut self) -> Result<NumericExpr, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.expect('(')?;
                let expr = self.parse_numeric_expr()?;
                self.expect(')')?;
                Ok(expr)
            }
            Some(b'$') => {
                self.bump();
                let ident = self.parse_ident("numeric matcher name")?;
                if ident != "width" {
                    return Err(self.error(&format!("unknown numeric matcher ${}", ident)));
                }
                self.expect('(')?;
                let name = self.parse_ident("placeholder")?;
                let mut ty: Option<ir::Type> = None;
                self.skip_ws();
                if self.peek() == Some(b':') {
                    self.bump();
                    ty = Some(self.parse_type("type constraint")?);
                }
                self.expect(')')?;
                Ok(NumericExpr::Width(PlaceholderExpr { name, ty }))
            }
            Some(c) if c.is_ascii_digit() => {
                let number = self.parse_u64("number")?;
                Ok(NumericExpr::Number(number))
            }
            _ => Err(self.error("expected numeric expression")),
        }
    }

    fn parse_bool_named_arg(
        &mut self,
        name: String,
        expr: QueryExpr,
    ) -> Result<Option<NamedArg>, String> {
        match expr {
            QueryExpr::Placeholder(ref p) if p.name == "_" && p.ty.is_none() => {
                Ok(Some(NamedArg {
                    name,
                    value: NamedArgValue::Any,
                }))
            }
            QueryExpr::Placeholder(ref p) if p.name == "true" && p.ty.is_none() => {
                Ok(Some(NamedArg {
                    name,
                    value: NamedArgValue::Bool(true),
                }))
            }
            QueryExpr::Placeholder(ref p) if p.name == "false" && p.ty.is_none() => {
                Ok(Some(NamedArg {
                    name,
                    value: NamedArgValue::Bool(false),
                }))
            }
            _ => Err(self.error("lsb_prio expects boolean literal or '_'")),
        }
    }

    fn parse_type(&mut self, ctx: &str) -> Result<ir::Type, String> {
        self.skip_ws();

        let mut ty: ir::Type = if self.peek() == Some(b'(') {
            self.expect('(')?;
            let mut members: Vec<Box<ir::Type>> = Vec::new();
            self.skip_ws();
            if self.peek() != Some(b')') {
                loop {
                    let member = self.parse_type("tuple member type")?;
                    members.push(Box::new(member));
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.bump();
                            self.skip_ws();
                        }
                        Some(b')') => break,
                        _ => return Err(self.error("expected ',' or ')' in tuple type")),
                    }
                }
            }
            self.expect(')')?;
            ir::Type::Tuple(members)
        } else {
            let kw = self.parse_ident(ctx)?;
            match kw.as_str() {
                "bits" => {
                    self.expect('[')?;
                    let count = self.parse_number("bit count")?;
                    self.expect(']')?;
                    let mut bits_ty = ir::Type::Bits(count);
                    // `bits` can have array dimensions directly attached: `bits[8][4]`.
                    while self.peek() == Some(b'[') {
                        self.expect('[')?;
                        let n = self.parse_number("array type size")?;
                        self.expect(']')?;
                        bits_ty = ir::Type::new_array(bits_ty, n);
                    }
                    bits_ty
                }
                "token" => ir::Type::Token,
                _ => return Err(self.error(&format!("expected type, got {}", kw))),
            }
        };

        // Additional array dimensions after any base type: `(bits[1], bits[2])[3]`,
        // etc.
        while self.peek() == Some(b'[') {
            self.expect('[')?;
            let n = self.parse_number("array type size")?;
            self.expect(']')?;
            ty = ir::Type::new_array(ty, n);
        }
        Ok(ty)
    }
}

enum BracketClause {
    UserCount(usize),
    Ident(String),
}

struct ParsedArgs {
    args: Vec<QueryExpr>,
    named_args: Vec<NamedArg>,
}

fn expected_arity(kind: &MatcherKind) -> usize {
    match kind {
        MatcherKind::AnyCmp | MatcherKind::AnyMul => 2,
        MatcherKind::Users => 1,
        MatcherKind::Width => 1,
        MatcherKind::AllOnes => 0,
        MatcherKind::MaskLow => 1,
        _ => 0,
    }
}
