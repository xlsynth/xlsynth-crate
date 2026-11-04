// SPDX-License-Identifier: Apache-2.0

//! Parser for BX's structural declarations and DSLX-style value expressions.

use crate::ast::{
    BinaryOp, Block, Expr, ExternBlock, ExternFn, Module, Param, Stmt, StructDef, Type, UnaryOp,
};

/// A borrowed spelling and byte offset in the original UTF-8 input.
#[derive(Clone, Copy)]
struct Token<'s> {
    text: &'s str,
    start: usize,
}

fn diagnostic(source: &str, offset: usize, message: &str) -> String {
    let prefix = &source[..offset];
    let line = 1 + prefix.bytes().filter(|&b| b == b'\n').count();
    let column = 1 + prefix.rsplit('\n').next().unwrap_or("").chars().count();
    format!("{line}:{column}: {message}")
}

fn identifier_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Tokenize source without interpreting integer widths or values.
fn lex(source: &str) -> Result<Vec<Token<'_>>, String> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let rest = &source[i..];
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
            continue;
        }
        if let Some(comment) = rest.strip_prefix("/*") {
            let end = comment
                .find("*/")
                .ok_or_else(|| diagnostic(source, start, "unterminated block comment"))?;
            i += end + 4;
            continue;
        }
        if identifier_start(b) || b.is_ascii_digit() {
            i += 1;
            while i < bytes.len() && (identifier_start(bytes[i]) || bytes[i].is_ascii_digit()) {
                i += 1;
            }
        } else if b == b'"' {
            let end = rest[1..]
                .find(['"', '\n', '\r'])
                .ok_or_else(|| diagnostic(source, start, "unterminated symbol string"))?;
            i += end + 1;
            if bytes[i] != b'"' {
                return Err(diagnostic(source, start, "unterminated symbol string"));
            }
            i += 1;
        } else if rest.starts_with(":=")
            || rest.starts_with("::")
            || rest.starts_with("->")
            || rest.get(..2).and_then(BinaryOp::from_spelling).is_some()
        {
            i += 2;
        } else if b"(){}[]:,.=;!&|^+-*/%<>".contains(&b) {
            i += 1;
        } else {
            return Err(diagnostic(source, start, "unexpected character"));
        }
        tokens.push(Token {
            text: &source[start..i],
            start,
        });
    }
    tokens.push(Token {
        text: "",
        start: source.len(),
    });
    Ok(tokens)
}

/// Parse a source module. Syntax diagnostics carry one-based line and column.
pub fn parse_module(source: &str) -> Result<Module, String> {
    Parser::new(source, false)?.module()
}

/// Parse a concrete value type printed by the DSLX type checker. In this
/// representation parentheses always delimit tuples, including singletons.
pub(crate) fn parse_value_type(source: &str) -> Result<Type, String> {
    let mut parser = Parser::new(source, true)?;
    let ty = parser.ty()?;
    if !parser.peek(0).is_empty() {
        return Err(parser.error("expected end of DSLX value type"));
    }
    crate::check::validate_structural_type(&ty)?;
    Ok(ty)
}

/// Cursor over the token stream, whose final element always represents EOF.
struct Parser<'s> {
    source: &'s str,
    tokens: Vec<Token<'s>>,
    pos: usize,
    depth: usize,
    dslx_type_display: bool,
}

/// The depth of a parsed type includes both tuple and array constructors.
struct ParsedType {
    value: Type,
    depth: usize,
}

/// Contents of braces before determining whether they form a guard or a value.
struct Body {
    statements: Vec<Stmt>,
    output: Option<ParsedExpr>,
}

/// Retains the built tree's height until its parent is constructed. Syntactic
/// nesting alone cannot bound a left-associative expression tree.
struct ParsedExpr {
    value: Expr,
    height: usize,
}

impl ParsedExpr {
    fn leaf(value: Expr) -> Self {
        Self { value, height: 1 }
    }
}

impl<'s> Parser<'s> {
    fn new(source: &'s str, dslx_type_display: bool) -> Result<Self, String> {
        Ok(Self {
            source,
            tokens: lex(source)?,
            pos: 0,
            depth: 0,
            dslx_type_display,
        })
    }

    /// Report excessive nesting before recursion consumes the host thread
    /// stack.
    fn nested<T>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.depth >= 64 {
            return Err(self.error("syntax nesting exceeds 64 levels"));
        }
        self.depth += 1;
        let result = parse(self);
        self.depth -= 1;
        result
    }

    fn peek(&self, n: usize) -> &'s str {
        self.tokens.get(self.pos + n).map_or("", |t| t.text)
    }

    fn eat(&mut self, text: &str) -> bool {
        if self.peek(0) == text {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn error(&self, message: &str) -> String {
        let token = self.tokens[self.pos];
        let found = if token.text.is_empty() {
            "end of input"
        } else {
            token.text
        };
        diagnostic(
            self.source,
            token.start,
            &format!("{message}, found '{found}'"),
        )
    }

    fn expect(&mut self, text: &str) -> Result<(), String> {
        if self.eat(text) {
            Ok(())
        } else {
            Err(self.error(&format!("expected '{text}'")))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        let text = self.peek(0);
        let keyword = matches!(
            text,
            "pub"
                | "block"
                | "struct"
                | "extern"
                | "fn"
                | "reg"
                | "let"
                | "inst"
                | "if"
                | "else"
                | "as"
                | "true"
                | "false"
                | "bool"
                | "bits"
                | "uN"
                | "sN"
        );
        if text
            .as_bytes()
            .first()
            .copied()
            .is_some_and(identifier_start)
            && !keyword
            && fixed_width(text).is_none()
        {
            self.pos += 1;
            Ok(text.to_owned())
        } else {
            Err(self.error("expected identifier"))
        }
    }

    fn module(&mut self) -> Result<Module, String> {
        let mut module = Module::default();
        while !self.peek(0).is_empty() {
            let public = self.eat("pub");
            if self.eat("block") {
                module.blocks.push(self.block(public)?);
            } else if self.eat("struct") {
                let name = self.ident()?;
                self.expect("{")?;
                let fields = self.list("}", Self::param)?;
                module.structs.push(StructDef {
                    name,
                    public,
                    fields,
                });
            } else if public {
                return Err(self.error("expected block or struct after pub"));
            } else if self.eat("import") {
                if self.peek(0) != "std" {
                    return Err(self.error("BX only supports 'import std;'"));
                }
                if module.std_import {
                    return Err(self.error("duplicate 'import std;'"));
                }
                self.pos += 1;
                self.expect(";")?;
                module.std_import = true;
            } else if self.eat("extern") {
                if self.eat("fn") {
                    module.externs.push(self.extern_fn()?);
                } else if self.eat("block") {
                    let signature = self.block_signature()?;
                    self.expect(";")?;
                    module.extern_blocks.push(signature);
                } else {
                    return Err(self.error("expected fn or block after extern"));
                }
            } else {
                return Err(self.error("expected a block, struct, extern, or import declaration"));
            }
        }
        Ok(module)
    }

    /// A module qualifier is permitted only on a function declaration or call.
    fn name_or_function_path(&mut self) -> Result<String, String> {
        let head = self.ident()?;
        if !self.eat("::") {
            return Ok(head);
        }
        let function = self.ident()?;
        if self.peek(0) != "(" {
            return Err(self.error("expected '(' after module-qualified function name"));
        }
        Ok(format!("{head}::{function}"))
    }

    /// Parse comma-separated items, allowing an empty list and a trailing
    /// comma.
    fn list<T>(
        &mut self,
        end: &str,
        mut item: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let mut items = Vec::new();
        while !self.eat(end) {
            items.push(item(self)?);
            if self.eat(end) {
                break;
            }
            self.expect(",")?;
        }
        Ok(items)
    }

    fn param(&mut self) -> Result<Param, String> {
        let name = self.ident()?;
        self.expect(":")?;
        Ok(Param {
            name,
            ty: self.ty()?,
        })
    }

    fn extern_fn(&mut self) -> Result<ExternFn, String> {
        let name = self.name_or_function_path()?;
        self.expect("(")?;
        let params = self.list(")", Self::param)?;
        self.expect("->")?;
        let result = self.ty()?;
        let symbol = if self.eat("=") {
            let s = self
                .peek(0)
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .filter(|s| !s.is_empty())
                .ok_or_else(|| self.error("expected a quoted IR symbol"))?;
            self.pos += 1;
            Some(s.to_owned())
        } else {
            None
        };
        self.expect(";")?;
        Ok(ExternFn {
            name,
            params,
            result,
            symbol,
        })
    }

    /// Parse the common interface of a defined or external clocked block.
    fn block_signature(&mut self) -> Result<ExternBlock, String> {
        let name = self.ident()?;
        self.expect("(")?;
        self.expect("clk")?;
        let clock = "clk".to_owned();
        let mut reset = None;
        let mut params = Vec::new();
        if !self.eat(")") {
            self.expect(",")?;
            if self.eat("rst") {
                reset = Some("rst".to_owned());
                if self.peek(0) != ")" {
                    self.expect(",")?;
                }
            }
            params = self.list(")", |parser| {
                if matches!(parser.peek(0), "clk" | "rst") {
                    return Err(parser.error("clock and reset roles cannot be data ports"));
                }
                parser.param()
            })?;
        }
        self.expect("->")?;
        let result = self.ty()?;
        Ok(ExternBlock {
            name,
            clock,
            reset,
            params,
            result,
        })
    }

    fn block(&mut self, public: bool) -> Result<Block, String> {
        let signature = self.block_signature()?;
        self.expect("{")?;
        let body = self.body()?;
        let output = body
            .output
            .ok_or_else(|| self.error("block requires an output expression"))?;
        Ok(Block {
            name: signature.name,
            public,
            clock: signature.clock,
            reset: signature.reset,
            params: signature.params,
            result: signature.result,
            statements: body.statements,
            output: output.value,
        })
    }

    fn ty(&mut self) -> Result<Type, String> {
        Ok(self.parsed_ty()?.value)
    }

    fn parsed_ty(&mut self) -> Result<ParsedType, String> {
        let mut ty = self.type_atom()?;
        while self.eat("[") {
            ty.depth += 1;
            if ty.depth > 64 {
                return Err(self.error("type nesting exceeds 64 levels"));
            }
            let size = self.dimension()?;
            self.expect("]")?;
            ty.value = Type::Array {
                element: Box::new(ty.value),
                size,
            };
        }
        Ok(ty)
    }

    /// Read a concrete bit width or array dimension without truncation.
    fn dimension(&mut self) -> Result<usize, String> {
        let text = self.peek(0);
        if !text.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            return Err(self.error("expected a concrete decimal size"));
        }
        let size = text
            .replace('_', "")
            .parse::<usize>()
            .map_err(|_| self.error("expected a concrete decimal size that fits usize"))?;
        self.pos += 1;
        Ok(size)
    }

    fn type_atom(&mut self) -> Result<ParsedType, String> {
        if self.eat("(") {
            return self.nested(|parser| {
                if parser.dslx_type_display {
                    let fields = parser.list(")", Self::parsed_ty)?;
                    parser.tuple_type(fields)
                } else {
                    parser.parenthesized(Self::parsed_ty, Self::tuple_type)
                }
            });
        }
        let spelling = self.peek(0);
        let value = if self.eat("bool") {
            Type::Bool
        } else if matches!(spelling, "bits" | "uN" | "sN") {
            self.pos += 1;
            self.expect("[")?;
            let width = self.dimension()?;
            self.expect("]")?;
            Type::Bits {
                signed: spelling == "sN",
                width,
            }
        } else if let Some(digits) = fixed_width(spelling) {
            let width = digits
                .parse::<usize>()
                .map_err(|_| self.error("bit width is too large"))?;
            self.pos += 1;
            Type::Bits {
                signed: spelling.starts_with('s'),
                width,
            }
        } else if self.dslx_type_display {
            return Err(self.error("unsupported DSLX result type (expected bits, tuple, or array)"));
        } else {
            Type::Named(self.ident()?)
        };
        Ok(ParsedType { value, depth: 0 })
    }

    fn tuple_type(&self, fields: Vec<ParsedType>) -> Result<ParsedType, String> {
        let depth = 1 + fields.iter().map(|field| field.depth).max().unwrap_or(0);
        if depth > 64 {
            return Err(self.error("type nesting exceeds 64 levels"));
        }
        Ok(ParsedType {
            value: Type::Tuple(fields.into_iter().map(|field| field.value).collect()),
            depth,
        })
    }

    /// Distinguish grouping (x) from a singleton tuple (x,).
    fn parenthesized<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T, String>,
        tuple: impl FnOnce(&Self, Vec<T>) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.eat(")") {
            return tuple(self, Vec::new());
        }
        let first = item(self)?;
        if self.eat(")") {
            return Ok(first);
        }
        self.expect(",")?;
        let mut fields = vec![first];
        fields.extend(self.list(")", item)?);
        tuple(self, fields)
    }

    /// Read declarations and updates followed by an optional trailing value.
    fn body(&mut self) -> Result<Body, String> {
        let mut statements = Vec::new();
        while !self.eat("}") {
            if self.eat("reg") {
                let param = self.param()?;
                let reset = if self.eat("=") {
                    Some(self.expr()?.value)
                } else {
                    None
                };
                statements.push(Stmt::Reg {
                    name: param.name,
                    ty: param.ty,
                    reset,
                });
            } else if self.eat("let") {
                let name = self.ident()?;
                let ty = if self.eat(":") {
                    Some(self.ty()?)
                } else {
                    None
                };
                self.expect("=")?;
                statements.push(Stmt::Let {
                    name,
                    ty,
                    value: self.expr()?.value,
                });
            } else if self.eat("inst") {
                let name = self.ident()?;
                self.expect(":")?;
                statements.push(Stmt::Inst {
                    name,
                    block: self.ident()?,
                });
            } else if self.peek(1) == ":=" {
                let name = self.ident()?;
                self.expect(":=")?;
                statements.push(Stmt::Write {
                    name,
                    value: self.expr()?.value,
                });
            } else if self.peek(1) == "." && self.peek(3) == ":=" {
                let instance = self.ident()?;
                self.expect(".")?;
                let port = self.ident()?;
                self.expect(":=")?;
                statements.push(Stmt::Connect {
                    instance,
                    port,
                    value: self.expr()?.value,
                });
            } else if self.eat("if") {
                let condition = self.condition()?;
                let then_body = self.nested(Self::body)?;
                if self.eat("else") {
                    let then_value = then_body.value_only().ok_or_else(|| {
                        self.error("value-producing if requires an expression in each arm")
                    })?;
                    let expr = self.if_tail(condition, then_value)?;
                    let output = self.expr_tail(expr, 0, true)?;
                    self.expect("}")?;
                    return Ok(Body {
                        statements,
                        output: Some(output),
                    });
                }
                if then_body.output.is_some() {
                    return Err(self.error("value-producing if requires an else arm"));
                }
                statements.push(Stmt::If {
                    condition: condition.value,
                    body: then_body.statements,
                });
                continue;
            } else {
                let output = self.expr()?;
                self.expect("}")?;
                return Ok(Body {
                    statements,
                    output: Some(output),
                });
            }
            self.expect(";")?;
        }
        Ok(Body {
            statements,
            output: None,
        })
    }

    fn expr(&mut self) -> Result<ParsedExpr, String> {
        self.expr_at(0, true)
    }

    /// Parse the condition and the opening brace. An unparenthesized name here
    /// refers to the condition signal even when a brace follows it.
    fn condition(&mut self) -> Result<ParsedExpr, String> {
        let condition = self.expr_at(0, false)?;
        self.expect("{")?;
        Ok(condition)
    }

    fn if_expr(&mut self) -> Result<ParsedExpr, String> {
        let condition = self.condition()?;
        let then_value = self.expr()?;
        self.expect("}")?;
        self.expect("else")?;
        self.if_tail(condition, then_value)
    }

    fn if_tail(
        &mut self,
        condition: ParsedExpr,
        then_value: ParsedExpr,
    ) -> Result<ParsedExpr, String> {
        let else_value = if self.eat("if") {
            self.nested(Self::if_expr)?
        } else {
            self.expect("{")?;
            let value = self.expr()?;
            self.expect("}")?;
            value
        };
        self.compose(
            condition
                .height
                .max(then_value.height)
                .max(else_value.height),
            Expr::If {
                condition: Box::new(condition.value),
                then_value: Box::new(then_value.value),
                else_value: Box::new(else_value.value),
            },
        )
    }

    /// Check each new node before it can become part of a deeper tree. Thus
    /// both checking and destruction recurse over a bounded path.
    fn compose(&self, child_height: usize, value: Expr) -> Result<ParsedExpr, String> {
        if child_height >= 64 {
            return Err(self.error("expression tree exceeds 64 levels"));
        }
        Ok(ParsedExpr {
            value,
            height: child_height + 1,
        })
    }

    /// Compute a variadic node's height once from the direct children's
    /// heights.
    fn aggregate(
        &self,
        items: Vec<ParsedExpr>,
        make: impl FnOnce(Vec<Expr>) -> Expr,
    ) -> Result<ParsedExpr, String> {
        let height = items.iter().map(|item| item.height).max().unwrap_or(0);
        self.compose(
            height,
            make(items.into_iter().map(|item| item.value).collect()),
        )
    }

    /// Parse operators according to DSLX precedence, highest for field access.
    fn expr_at(&mut self, min: u8, allow_struct: bool) -> Result<ParsedExpr, String> {
        self.nested(|parser| {
            let left = parser.atom(allow_struct)?;
            parser.expr_tail(left, min, allow_struct)
        })
    }

    fn expr_tail(
        &mut self,
        mut left: ParsedExpr,
        min: u8,
        allow_struct: bool,
    ) -> Result<ParsedExpr, String> {
        loop {
            if self.eat(".") {
                let name = if decimal(self.peek(0)) {
                    let index = self.peek(0).to_owned();
                    self.pos += 1;
                    index
                } else {
                    self.ident()?
                };
                left = self.compose(
                    left.height,
                    Expr::Field {
                        value: Box::new(left.value),
                        name,
                    },
                )?;
            } else if self.eat("[") {
                let index = self.expr()?;
                self.expect("]")?;
                left = self.compose(
                    left.height.max(index.height),
                    Expr::Index {
                        value: Box::new(left.value),
                        index: Box::new(index.value),
                    },
                )?;
            } else if min <= 10 && self.eat("as") {
                let ty = self.ty()?;
                left = self.compose(
                    left.height,
                    Expr::Cast {
                        value: Box::new(left.value),
                        ty,
                    },
                )?;
            } else {
                let Some(op) =
                    BinaryOp::from_spelling(self.peek(0)).filter(|op| op.precedence() >= min)
                else {
                    return Ok(left);
                };
                self.pos += 1;
                let right = self.expr_at(op.precedence() + 1, allow_struct)?;
                left = self.compose(
                    left.height.max(right.height),
                    Expr::Binary {
                        op,
                        left: Box::new(left.value),
                        right: Box::new(right.value),
                    },
                )?;
            }
        }
    }

    fn atom(&mut self, allow_struct: bool) -> Result<ParsedExpr, String> {
        let text = self.peek(0);
        if self.eat("[") {
            let items = self.list("]", Self::expr)?;
            return self.aggregate(items, Expr::Array);
        }
        if self.eat("(") {
            return self.parenthesized(Self::expr, |parser, fields| {
                parser.aggregate(fields, Expr::Tuple)
            });
        }
        if self.eat("if") {
            return self.if_expr();
        }
        if self.eat("true") || self.eat("false") {
            return Ok(ParsedExpr::leaf(Expr::Bool(text == "true")));
        }
        if let Some(op) = UnaryOp::from_spelling(text) {
            self.pos += 1;
            let value = self.expr_at(11, allow_struct)?;
            return self.compose(
                value.height,
                Expr::Unary {
                    op,
                    value: Box::new(value.value),
                },
            );
        }
        if matches!(text, "bits" | "uN" | "sN" | "bool") || fixed_width(text).is_some() {
            let ty = self.ty()?;
            self.expect(":")?;
            let negative = self.eat("-");
            return self.literal(Some(ty), negative);
        }
        if text.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            return self.literal(None, false);
        }
        let name = self.name_or_function_path()?;
        if self.eat("(") {
            let args = self.list(")", Self::expr)?;
            return self.aggregate(args, |args| Expr::Call { name, args });
        }
        if allow_struct && self.eat("{") {
            let mut height = 0;
            let fields = self.list("}", |parser| {
                let field = parser.ident()?;
                let value = if parser.eat(":") {
                    parser.expr()?
                } else {
                    ParsedExpr::leaf(Expr::Name(field.clone()))
                };
                height = height.max(value.height);
                Ok((field, value.value))
            })?;
            return self.compose(height, Expr::Struct { name, fields });
        }
        Ok(ParsedExpr::leaf(Expr::Name(name)))
    }

    /// Preserve arbitrary-width numbers as text for checking against their
    /// type.
    fn literal(&mut self, ty: Option<Type>, negative: bool) -> Result<ParsedExpr, String> {
        if !self
            .peek(0)
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_digit)
        {
            return Err(self.error("expected an integer literal"));
        }
        let mut text = self.peek(0).replace('_', "");
        let (digits, radix) = if let Some(s) = text.strip_prefix("0x") {
            (s, 16)
        } else if let Some(s) = text.strip_prefix("0b") {
            (s, 2)
        } else {
            (text.as_str(), 10)
        };
        if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
            return Err(self.error("expected an integer literal"));
        }
        self.pos += 1;
        if negative {
            text.insert(0, '-');
        }
        Ok(ParsedExpr::leaf(Expr::Literal { ty, text }))
    }
}

impl Body {
    fn value_only(self) -> Option<ParsedExpr> {
        if self.statements.is_empty() {
            self.output
        } else {
            None
        }
    }
}

fn decimal(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

fn fixed_width(spelling: &str) -> Option<&str> {
    spelling.strip_prefix(['u', 's']).filter(|s| decimal(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_import_is_contextual_and_may_appear_after_a_block() {
        let src = "block import(clk, import: u8) -> u8 { import }";
        assert!(!parse_module(src).unwrap().std_import);
        assert!(
            parse_module(&format!("{src} import std;"))
                .unwrap()
                .std_import
        );
        for (source, error) in [
            ("import std; import std;", "duplicate 'import std;'"),
            ("import arithmetic;", "only supports 'import std;'"),
            ("import std", "expected ';'"),
            ("pub import std;", "expected block or struct after pub"),
        ] {
            assert!(
                parse_module(source).unwrap_err().contains(error),
                "{source}"
            );
        }
    }

    #[test]
    fn concrete_dslx_result_types_preserve_singletons_and_check_layout_limits() {
        let singleton = Type::Tuple(vec![Type::Bits {
            signed: true,
            width: 8,
        }]);
        assert_eq!(
            parse_value_type("(sN[8])[3]").unwrap(),
            Type::Array {
                element: Box::new(singleton),
                size: 3
            },
        );
        assert_eq!(parse_value_type("()").unwrap(), Type::Tuple(vec![]));
        let bx_group = parse_module("block b(clk) -> (s8) { s8:0 }").unwrap();
        assert_eq!(
            bx_group.blocks[0].result,
            Type::Bits {
                signed: true,
                width: 8
            }
        );
        for invalid in [
            "(uN[8], Ordering)[3]",
            "uN[8] extra",
            "uN[10000000]",
            "uN[8][10000000]",
        ] {
            assert!(parse_value_type(invalid).is_err(), "{invalid}");
        }
        let grouped = format!("(uN[8]{})[1]", "[1]".repeat(63));
        assert!(
            parse_value_type(&grouped)
                .unwrap_err()
                .contains("type nesting")
        );
    }

    #[test]
    fn distinguishes_a_guarded_update_from_a_final_if_expression() {
        let module = parse_module(
            r#"
            pub block accumulator(clk, rst, en: bool, x: uN[80]) -> uN[80] {
                reg sum: uN[80] = uN[80]:0x1000_0000_0000_0000_0000;
                if en { sum := sum + x; }
                if en { sum + x } else { sum }
            }
        "#,
        )
        .unwrap();
        let block = &module.blocks[0];
        assert_eq!(block.reset.as_deref(), Some("rst"));
        assert!(
            matches!(&block.statements[..], [Stmt::Reg { reset: Some(_), .. }, Stmt::If { body, .. }] if matches!(&body[..], [Stmt::Write { name, .. }] if name == "sum"))
        );
        assert!(matches!(&block.output, Expr::If { .. }));
    }

    #[test]
    fn keeps_aggregate_types_and_named_instance_connections() {
        let module = parse_module(
            r#"
            struct Result { value: s8, spare: (), }
            extern fn invert(value: s8) -> s8 = "__logic__invert";
            block wrapper(clk, value: s8) -> (Result,) {
                inst lane: child;
                lane.input := invert(value);
                (Result { value: lane.value, spare: () },)
            }
        "#,
        )
        .unwrap();
        assert_eq!(module.externs[0].symbol.as_deref(), Some("__logic__invert"));
        let block = &module.blocks[0];
        assert_eq!(
            block.result,
            Type::Tuple(vec![Type::Named("Result".to_owned())])
        );
        assert!(
            matches!(&block.statements[..], [Stmt::Inst { .. }, Stmt::Connect { instance, port, .. }] if instance == "lane" && port == "input")
        );
    }

    #[test]
    fn qualified_extern_declarations_and_calls_share_one_canonical_name() {
        let source = r#"
extern fn arithmetic :: mul_wide(x: u8, y: u8) -> u16;
extern fn local(x: u16) -> u16 = "local_impl";
block product(clk, x: u8, y: u8) -> u16 { local(arithmetic::mul_wide(x, y)) }
"#;
        let module = parse_module(source).unwrap();
        assert_eq!(module.externs[0].name, "arithmetic::mul_wide");
        assert_eq!(module.externs[1].name, "local");
        let Expr::Call { name, args } = &module.blocks[0].output else {
            panic!("expected local call");
        };
        assert_eq!(name, "local");
        assert!(
            matches!(&args[..], [Expr::Call { name, args }] if name == "arithmetic::mul_wide" && args.len() == 2)
        );
    }

    #[test]
    fn extern_blocks_share_the_clocked_interface_grammar_with_defined_blocks() {
        let source = r#"
extern block lane(clk, rst, x: u8) -> (u8, bool);
extern block pulse(clk) -> bool;
block wrap(clk, rst, x: u8) -> (u8, bool) {
    let stage = lane(x);
    stage
}
"#;
        let module = parse_module(source).unwrap();
        let lane = &module.extern_blocks[0];
        let wrap = &module.blocks[0];
        assert_eq!(lane.name, "lane");
        assert_eq!(lane.clock, wrap.clock);
        assert_eq!(lane.reset, wrap.reset);
        assert_eq!(lane.params, wrap.params);
        assert_eq!(lane.result, wrap.result);
        assert_eq!(module.extern_blocks[1].reset, None);
        assert!(
            matches!(&wrap.statements[..], [Stmt::Let { value: Expr::Call { name, .. }, .. }] if name == "lane")
        );
        for bad in [
            "extern block module::lane(clk) -> bool;",
            "extern block lane(clk) -> bool = \"impl\";",
            "extern block lane(clk) -> bool { true }",
            "extern block lane(x: u8) -> bool;",
            "extern block lane(clk, rst: bool) -> bool;",
            "extern block lane(clk) -> bool",
        ] {
            assert!(parse_module(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_malformed_or_non_function_qualifications() {
        for name in [
            "::mul_wide",
            "arithmetic::",
            "arithmetic::sub::mul_wide",
            "arithmetic::1",
            "arithmetic:::mul_wide",
        ] {
            let declaration = format!("extern fn {name}() -> u8;");
            let call = format!("block b(clk) -> u8 {{ {name}() }}");
            assert!(parse_module(&declaration).is_err(), "{declaration}");
            assert!(parse_module(&call).is_err(), "{call}");
        }
        for source in [
            "block b(clk) -> u8 { arithmetic::value }",
            "block b(clk) -> u8 { arithmetic::Payload { x: u8:0 } }",
            "struct arithmetic::Payload { x: u8 }",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
    }

    #[test]
    fn keeps_array_dimensions_in_dslx_order_and_indexes_nested_arrays() {
        let module = parse_module(
            r#"
            pub struct Grid { rows: bits[80][2][3] }
            extern fn make(x: u8[2]) -> (bool, u8)[3];
            block read(clk, g: Grid, i: u2) -> uN[80][1] { [g.rows[i][u1:0]] }
        "#,
        )
        .unwrap();
        assert!(module.structs[0].public);
        assert_eq!(
            module.structs[0].fields[0].ty,
            Type::Array {
                size: 3,
                element: Box::new(Type::Array {
                    size: 2,
                    element: Box::new(Type::Bits {
                        signed: false,
                        width: 80
                    }),
                })
            }
        );
        let Expr::Array(values) = &module.blocks[0].output else {
            panic!("expected array literal")
        };
        assert!(
            matches!(&values[..], [Expr::Index { value, .. }] if matches!(value.as_ref(), Expr::Index { .. }))
        );
    }

    #[test]
    fn groups_casts_and_operators_like_dslx() {
        let module =
            parse_module("block b(clk, x: s8, y: u16, z: u16) -> u16 { -x as u16 + y * z }")
                .unwrap();
        let Expr::Binary { op, left, right } = &module.blocks[0].output else {
            panic!("expected addition");
        };
        assert_eq!(*op, BinaryOp::Add);
        assert!(
            matches!(left.as_ref(), Expr::Cast { value, .. } if matches!(value.as_ref(), Expr::Unary { op: UnaryOp::Neg, .. }))
        );
        assert!(matches!(
            right.as_ref(),
            Expr::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn malformed_source_has_a_location() {
        assert_eq!(
            parse_module("block b(clk) -> u8 {\n  u8:0x\n}").unwrap_err(),
            "2:6: expected an integer literal, found '0x'"
        );
        for source in [
            "block b(clock, data: u8) -> u8 { data }",
            "block b(clk, reset, data: u8) -> u8 { data }",
            "block b(clk, rst: bool) -> bool { rst }",
            "block b(clk, data: u8, rst: bool) -> u8 { data }",
            "block b(clk) -> () { if true { () } }",
            "block b(clk) -> u8 { /* missing close",
            "extern fn f() -> u8 = \"missing close",
            "pub extern fn f() -> u8;",
            "block b(clk, x: u8[2]) -> u8 { x[] }",
            "block b(clk, x: u8[_8]) -> u8 { x[u1:0] }",
            "block b(clk) -> u8 { u8:_42 }",
            "block b(clk) -> u8 {",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
    }

    #[test]
    fn rejects_deep_nesting_before_exhausting_the_thread_stack() {
        let source = format!("block b(clk) -> bool {{ {}true }}", "!".repeat(10_000));
        assert!(
            parse_module(&source)
                .unwrap_err()
                .contains("syntax nesting exceeds")
        );
    }
}
