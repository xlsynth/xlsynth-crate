// SPDX-License-Identifier: Apache-2.0

//! Parser for BX's structural declarations and DSLX-style value expressions.

use std::collections::BTreeSet;

use crate::ast::{
    BinaryOp, Block, Expr, ExternBlock, ExternFn, Module, Param, Stmt, StructDef, Type, UnaryOp,
    UsePath, WidthSliceType,
};

/// Bound recursive syntax, including DSLX use trees generated from a BX AST.
pub(crate) const MAX_SYNTAX_DEPTH: usize = 64;

/// A borrowed spelling and byte offset in the original UTF-8 input.
#[derive(Clone, Copy)]
struct Token<'s> {
    text: &'s str,
    start: usize,
    character: Option<u8>,
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

fn dslx_identifier_tail(b: u8) -> bool {
    identifier_start(b) || b.is_ascii_digit() || matches!(b, b'!' | b'\'')
}

/// DSLX struct and field names can contain primes and exclamation marks.
pub(crate) fn is_dslx_member_name(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes.next().is_some_and(identifier_start) && bytes.all(dslx_identifier_tail)
}

/// Identifier words used for local BX names and generated function symbols.
pub(crate) fn is_identifier_word(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes.next().is_some_and(identifier_start)
        && bytes.all(|b| identifier_start(b) || b.is_ascii_digit())
}

/// Path components are identifier tokens in the DSLX `use` grammar.
pub(crate) fn is_use_identifier(text: &str) -> bool {
    is_dslx_member_name(text)
        && fixed_width(text).is_none()
        && !matches!(
            text,
            "as" | "const"
                | "else"
                | "enum"
                | "false"
                | "fn"
                | "for"
                | "if"
                | "impl"
                | "import"
                | "in"
                | "out"
                | "let"
                | "match"
                | "pub"
                | "proc"
                | "self"
                | "struct"
                | "trait"
                | "true"
                | "type"
                | "unroll_for!"
                | "use"
                | "mut"
                | "bits"
                | "token"
                | "uN"
                | "sN"
                | "xN"
                | "bool"
                | "chan"
                | "Self"
        )
}

/// A local BX binding may use any identifier word other than reserved syntax.
pub(crate) fn is_identifier(text: &str) -> bool {
    let keyword = matches!(
        text,
        "use"
            | "pub"
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
            | "xN"
    );
    is_identifier_word(text)
        && !text.contains(['!', '\''])
        && !keyword
        && fixed_width(text).is_none()
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
        let negative_number = b == b'-' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
        let mut character = None;
        if identifier_start(b) {
            i += 1;
            while i < bytes.len() && dslx_identifier_tail(bytes[i]) {
                // Keep `a!=b` as inequality, and `a.field!==b` as equality
                // on the DSLX field named `field!`.
                if bytes[i..].starts_with(b"!=") && !bytes[i..].starts_with(b"!==") {
                    break;
                }
                i += 1;
            }
        } else if b.is_ascii_digit() || negative_number {
            let magnitude = rest.strip_prefix('-').unwrap_or(rest);
            let (digits, radix) = radix_digits(magnitude);
            let digit_count = digits
                .chars()
                .take_while(|c| c.is_digit(radix) || (radix != 10 && *c == '_'))
                .count();
            i += rest.len() - digits.len() + digit_count;
        } else if b == b'\'' {
            let (value, len) = scan_character(rest)
                .ok_or_else(|| diagnostic(source, start, "invalid byte character literal"))?;
            character = Some(value);
            i += len;
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
            || rest.starts_with("+:")
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
            character,
        });
    }
    tokens.push(Token {
        text: "",
        start: source.len(),
        character: None,
    });
    Ok(tokens)
}

/// Decode one DSLX byte character and return its value and token length.
fn scan_character(source: &str) -> Option<(u8, usize)> {
    match source.as_bytes() {
        [b'\'', b'\\', b'x', _, _, b'\'', ..] => {
            let digits = source.get(3..5)?;
            if !digits.bytes().all(|digit| digit.is_ascii_hexdigit()) {
                return None;
            }
            Some((u8::from_str_radix(digits, 16).ok()?, 6))
        }
        [b'\'', b'\\', escaped, b'\'', ..] => {
            let value = match escaped {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'0' => 0,
                b'\\' | b'\'' | b'"' => *escaped,
                _ => return None,
            };
            Some((value, 4))
        }
        [b'\'', value, b'\'', ..] if value.is_ascii() && !matches!(value, b'\n' | b'\\') => {
            Some((*value, 3))
        }
        _ => None,
    }
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
    spelling: ExprSpelling,
}

/// Source distinctions needed for DSLX postfix and comparison precedence.
#[derive(Clone, Copy, PartialEq)]
enum ExprSpelling {
    Ordinary,
    Parenthesized,
    AnnotatedLiteral,
}

impl ParsedExpr {
    fn leaf(value: Expr) -> Self {
        Self {
            value,
            height: 1,
            spelling: ExprSpelling::Ordinary,
        }
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
        if self.depth >= MAX_SYNTAX_DEPTH {
            return Err(self.error(&format!("syntax nesting exceeds {MAX_SYNTAX_DEPTH} levels")));
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
        self.take_identifier(is_identifier)
    }

    fn take_identifier(&mut self, valid: fn(&str) -> bool) -> Result<String, String> {
        let text = self.peek(0);
        if valid(text) {
            self.pos += 1;
            Ok(text.to_owned())
        } else {
            Err(self.error("expected identifier"))
        }
    }

    fn module(&mut self) -> Result<Module, String> {
        let mut module = Module::default();
        let mut use_bindings = BTreeSet::new();
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
            } else if self.eat("use") {
                self.use_tree(&mut Vec::new(), &mut module.uses, &mut use_bindings)?;
                self.expect(";")?;
            } else if self.peek(0) == "import" {
                return Err(self
                    .error("replace `import` with `use` and separate path components with `::`"));
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
                return Err(self.error("expected a block, struct, extern, or use declaration"));
            }
        }
        Ok(module)
    }

    /// Flatten one DSLX use tree, retaining each full path and its leaf
    /// binding.
    fn use_tree(
        &mut self,
        prefix: &mut Vec<String>,
        paths: &mut Vec<UsePath>,
        bindings: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        self.nested(|parser| {
            let offset = parser.tokens[parser.pos].start;
            let name = parser.take_identifier(is_use_identifier)?;
            let has_children = parser.eat("::");
            if !has_children && !bindings.insert(name.clone()) {
                return Err(diagnostic(
                    parser.source,
                    offset,
                    &format!("duplicate use binding '{name}'"),
                ));
            }
            prefix.push(name);
            if !has_children {
                paths.push(UsePath {
                    components: prefix.clone(),
                });
            } else if parser.eat("{") {
                loop {
                    parser.use_tree(prefix, paths, bindings)?;
                    if parser.eat("}") {
                        break;
                    }
                    parser.expect(",")?;
                }
            } else {
                parser.use_tree(prefix, paths, bindings)?;
            }
            prefix.pop();
            Ok(())
        })
    }

    /// Parse a local name or a member of an imported DSLX module.
    fn name_or_qualified_member(&mut self) -> Result<String, String> {
        let head = self.take_identifier(|word| is_identifier(word) || is_use_identifier(word))?;
        if !self.eat("::") {
            return Ok(head);
        }
        let member = self.take_identifier(is_dslx_member_name)?;
        Ok(format!("{head}::{member}"))
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
        if !is_identifier(self.peek(0)) {
            return Err(self.error("expected a BX identifier for the extern function"));
        }
        let name = self.name_or_qualified_member()?;
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

    /// Parse a slice result annotation, retaining any expressions that set its
    /// signedness or width. Return their maximum height for the enclosing
    /// slice.
    fn width_slice_type(&mut self) -> Result<(WidthSliceType, usize), String> {
        let spelling = self.peek(0);
        if !matches!(spelling, "bits" | "uN" | "sN" | "xN") {
            // DSLX treats parentheses in a type annotation as tuple syntax,
            // including a single parenthesized member.
            let ty = if self.eat("(") {
                Type::Tuple(self.list(")", Self::ty)?)
            } else {
                self.ty()?
            };
            return Ok((WidthSliceType::Concrete(ty), 0));
        }
        self.pos += 1;
        let signed = if spelling == "xN" {
            self.expect("[")?;
            let expr = self.expr()?;
            self.expect("]")?;
            expr
        } else {
            ParsedExpr::leaf(Expr::Bool(spelling == "sN"))
        };
        self.expect("[")?;
        let bit_count = self.expr()?;
        self.expect("]")?;
        Ok((
            WidthSliceType::Bits {
                signed: Box::new(signed.value),
                bit_count: Box::new(bit_count.value),
            },
            signed.height.max(bit_count.height),
        ))
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
            return Err(self.error("expected a concrete nonnegative size"));
        }
        let text = normalized_integer(text)
            .ok_or_else(|| self.error("expected a concrete nonnegative size"))?;
        let (digits, radix) = radix_digits(&text);
        let size = usize::from_str_radix(digits, radix)
            .map_err(|_| self.error("expected a concrete nonnegative size that fits usize"))?;
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
        } else if matches!(spelling, "bits" | "uN" | "sN" | "xN") {
            self.pos += 1;
            let signed = if spelling == "xN" {
                self.expect("[")?;
                let signed = match self.peek(0) {
                    "true" => true,
                    "false" => false,
                    _ => return Err(self.error("expected a concrete boolean signedness")),
                };
                self.pos += 1;
                self.expect("]")?;
                signed
            } else {
                spelling == "sN"
            };
            self.expect("[")?;
            let width = self.dimension()?;
            self.expect("]")?;
            Type::Bits { signed, width }
        } else if !self.dslx_type_display
            && let Some(width) = fixed_width(spelling)
        {
            // Native DSLX bits display as uN[W] or sN[W]. A displayed name
            // such as u84 can instead denote an enum declared by DSLX.
            self.pos += 1;
            Type::Bits {
                signed: spelling.starts_with('s'),
                width,
            }
        } else if self.dslx_type_display {
            return Err(self.error("unsupported DSLX result type (expected bits, tuple, or array)"));
        } else {
            Type::Named(self.name_or_qualified_member()?)
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
            // A DSLX use leaf can be named reg or inst. A following name
            // distinguishes a hardware declaration from a call or constructor.
            if self.peek(0) == "reg" && is_identifier(self.peek(1)) {
                self.pos += 1;
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
            } else if self.peek(0) == "inst" && is_identifier(self.peek(1)) {
                self.pos += 1;
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
            spelling: ExprSpelling::Ordinary,
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
                let name = if number_token(self.peek(0)) {
                    let offset = self.tokens[self.pos].start;
                    let index = u32::try_from(self.dimension()?).map_err(|_| {
                        diagnostic(self.source, offset, "tuple index must fit in u32")
                    })?;
                    index.to_string()
                } else {
                    self.take_identifier(is_dslx_member_name)?
                };
                left = self.compose(
                    left.height,
                    Expr::Field {
                        value: Box::new(left.value),
                        name,
                    },
                )?;
            } else if self.peek(0) == "[" {
                left = self.subscript(left)?;
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
                if op.is_comparison()
                    && left.spelling != ExprSpelling::Parenthesized
                    && matches!(left.value, Expr::Binary { op: prev, .. } if prev.is_comparison())
                {
                    return Err(self.error("comparison operators cannot be chained"));
                }
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

    /// Distinguish array indexing from a packed width slice after parsing the
    /// offset expression shared by both.
    fn subscript(&mut self, left: ParsedExpr) -> Result<ParsedExpr, String> {
        let needs_parens = left.spelling == ExprSpelling::AnnotatedLiteral
            || (left.spelling != ExprSpelling::Parenthesized
                && matches!(left.value, Expr::If { .. }));
        self.expect("[")?;
        let subscript = self.expr()?;
        let mut height = left.height.max(subscript.height);
        let value = Box::new(left.value);
        let expr = if self.eat("+:") {
            if needs_parens {
                return Err(
                    self.error("parenthesize a typed literal or conditional before slicing")
                );
            }
            let (width, type_height) = self.width_slice_type()?;
            height = height.max(type_height);
            Expr::WidthSlice {
                value,
                start: Box::new(subscript.value),
                width,
            }
        } else {
            Expr::Index {
                value,
                index: Box::new(subscript.value),
            }
        };
        self.expect("]")?;
        self.compose(height, expr)
    }

    fn atom(&mut self, allow_struct: bool) -> Result<ParsedExpr, String> {
        let text = self.peek(0);
        if self.eat("[") {
            let items = self.list("]", Self::expr)?;
            return self.aggregate(items, Expr::Array);
        }
        if self.eat("(") {
            let mut expr = self.parenthesized(Self::expr, |parser, fields| {
                parser.aggregate(fields, Expr::Tuple)
            })?;
            expr.spelling = ExprSpelling::Parenthesized;
            return Ok(expr);
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
        if self.starts_bit_literal()? {
            let ty = self.ty()?;
            self.expect(":")?;
            return self.annotated_literal(ty);
        }
        if number_token(text) || self.tokens[self.pos].character.is_some() {
            return self.literal(None);
        }
        let name = self.name_or_qualified_member()?;
        if self.eat("(") {
            let args = self.list(")", Self::expr)?;
            return self.aggregate(args, |args| Expr::Call { name, args });
        }
        if allow_struct && self.eat("{") {
            let mut height = 0;
            let fields = self.list("}", |parser| {
                let field = parser.take_identifier(is_dslx_member_name)?;
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
        if name.contains("::") {
            return Err(
                self.error("expected a call or struct literal after a module-qualified name")
            );
        }
        Ok(ParsedExpr::leaf(Expr::Name(name)))
    }

    /// Recognize a bit literal's annotation and suggest the DSLX spelling
    /// when a formerly accepted shorthand is followed by a colon.
    fn starts_bit_literal(&self) -> Result<bool, String> {
        let name = self.peek(0);
        if matches!(name, "bits" | "uN" | "sN" | "xN" | "bool") || fixed_width(name).is_some() {
            return Ok(true);
        }
        if self.peek(1) == ":"
            && let Some(hint) = explicit_width_hint(name)
        {
            return Err(self.error(&hint));
        }
        Ok(false)
    }

    /// Parse a literal term after a type annotation. Parentheses group the
    /// literal, and the outermost annotation supplies its final type.
    fn annotated_literal(&mut self, ty: Type) -> Result<ParsedExpr, String> {
        if self.eat("(") {
            let literal = self.nested(|parser| parser.annotated_literal(ty))?;
            self.expect(")")?;
            return Ok(literal);
        }
        if self.starts_bit_literal()? {
            self.ty()?;
            self.expect(":")?;
            return self.nested(|parser| parser.annotated_literal(ty));
        }
        self.literal(Some(ty))
    }

    /// Preserve arbitrary-width numbers as text for checking against their
    /// type.
    fn literal(&mut self, ty: Option<Type>) -> Result<ParsedExpr, String> {
        let annotated = ty.is_some();
        let (ty, text) = if let Some(byte) = self.tokens[self.pos].character {
            let ty = ty.unwrap_or(Type::Bits {
                signed: false,
                width: 8,
            });
            if !matches!(ty, Type::Bits { width: 8.., .. }) {
                return Err(self.error("a character literal requires at least eight bits"));
            }
            (Some(ty), format!("0x{byte:02x}"))
        } else if matches!(self.peek(0), "true" | "false") {
            if !matches!(ty, Some(Type::Bool | Type::Bits { width: 1.., .. })) {
                return Err(self.error("a boolean literal requires at least one bit"));
            }
            let bit = if self.peek(0) == "true" { "0b1" } else { "0b0" };
            (ty, bit.to_owned())
        } else {
            let text = normalized_integer(self.peek(0))
                .ok_or_else(|| self.error("expected an integer literal"))?;
            (ty, text)
        };
        self.pos += 1;
        let mut expr = ParsedExpr::leaf(Expr::Literal { ty, text });
        if annotated {
            expr.spelling = ExprSpelling::AnnotatedLiteral;
        }
        Ok(expr)
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

/// DSLX decimal digits have no leading zero unless the entire spelling is `0`.
fn is_canonical_decimal(text: &str) -> bool {
    !text.is_empty()
        && (text.len() == 1 || !text.starts_with('0'))
        && text.bytes().all(|b| b.is_ascii_digit())
}

/// Check the DSLX radix and separator rules before removing separators.
fn normalized_integer(spelling: &str) -> Option<String> {
    let magnitude = spelling.strip_prefix('-').unwrap_or(spelling);
    let (digits, radix) = radix_digits(magnitude);
    if radix == 10 {
        return is_canonical_decimal(digits).then(|| spelling.to_owned());
    }
    if !digits.chars().any(|c| c.is_digit(radix))
        || !digits.chars().all(|c| c == '_' || c.is_digit(radix))
    {
        return None;
    }
    Some(spelling.replace('_', ""))
}

/// Split an unsigned literal magnitude into digits and its DSLX radix.
pub(crate) fn radix_digits(magnitude: &str) -> (&str, u32) {
    if let Some(digits) = magnitude.strip_prefix("0x") {
        (digits, 16)
    } else if let Some(digits) = magnitude.strip_prefix("0b") {
        (digits, 2)
    } else {
        (magnitude, 10)
    }
}

fn number_token(text: &str) -> bool {
    text.strip_prefix('-')
        .unwrap_or(text)
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_digit)
}

/// DSLX reserves only u1 through u64 and s1 through s64 as bit type keywords.
fn fixed_width(spelling: &str) -> Option<usize> {
    let digits = spelling.strip_prefix(['u', 's'])?;
    if digits.len() > 2 || !is_canonical_decimal(digits) {
        return None;
    }
    digits.parse().ok().filter(|width| (1..=64).contains(width))
}

/// A migration hint for an unresolved zero-width or wide bit shorthand.
pub(crate) fn explicit_width_hint(spelling: &str) -> Option<String> {
    let digits = spelling.strip_prefix(['u', 's'])?;
    (is_canonical_decimal(digits) && fixed_width(spelling).is_none())
        .then(|| format!("use '{}N[{digits}]' for this bit width", &spelling[..1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn use_trees_bind_each_leaf_and_preserve_its_full_path() {
        let module =
            parse_module("use std; use device::{math::{mac, Header'}, constants::BIAS, timing};")
                .unwrap();
        let paths: Vec<_> = module
            .uses
            .iter()
            .map(|path| (path.components.join("::"), path.binding().unwrap()))
            .collect();
        assert_eq!(
            paths,
            [
                ("std".into(), "std"),
                ("device::math::mac".into(), "mac"),
                ("device::math::Header'".into(), "Header'"),
                ("device::constants::BIAS".into(), "BIAS"),
                ("device::timing".into(), "timing"),
            ]
        );
    }

    #[test]
    fn use_declarations_reject_ambiguous_bindings_and_invalid_trees() {
        for (source, error) in [
            ("use std; use std;", "duplicate use binding 'std'"),
            ("use a::math; use b::math;", "duplicate use binding 'math'"),
            ("use a::{b::item, c::item};", "duplicate use binding 'item'"),
            ("use {foo, bar};", "expected identifier"),
            ("use a::{};", "expected identifier"),
            ("use a::{b,};", "expected identifier"),
            ("use a::{b c};", "expected ','"),
            ("use a::{b", "expected ','"),
            ("use a::;", "expected identifier"),
            ("use ::a;", "expected identifier"),
            ("use a::*;", "expected identifier"),
            ("use foo.bar;", "expected ';'"),
            ("use foo as f;", "expected ';'"),
            ("use lib::fn;", "expected identifier"),
            ("use lib::u64;", "expected identifier"),
            ("use std", "expected ';'"),
            ("pub use std;", "expected block or struct after pub"),
            ("import a.b;", "replace `import` with `use`"),
        ] {
            assert!(
                parse_module(source).unwrap_err().contains(error),
                "{source}"
            );
        }
        let deep = format!("use {}leaf;", "a::".repeat(64));
        assert!(parse_module(&deep).unwrap_err().contains("syntax nesting"));
    }

    #[test]
    fn uses_can_bind_dslx_identifiers_that_are_special_to_bx() {
        let module = parse_module(
            r#"
use ip::reg;
use packet::{Header', inst};
block b(clk, x: u8) -> u8 { reg::block(x) }
block h(clk, x: u8) -> Header' { Header' { field: x } }
block i(clk, x: u8) -> u8 { inst(x) }
"#,
        )
        .unwrap();
        assert!(
            matches!(&module.blocks[0].output, Expr::Call { name, .. } if name == "reg::block")
        );
        assert!(matches!(&module.blocks[1].output, Expr::Struct { name, .. } if name == "Header'"));
        assert!(matches!(&module.blocks[2].output, Expr::Call { name, .. } if name == "inst"));
        // The removed declaration keyword is still available as a BX name.
        assert!(parse_module("block import(clk, import: u8) -> u8 { import }").is_ok());
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
            "(u84, uN[8])",
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
    fn rejects_malformed_paths_qualified_values_and_qualified_declarations() {
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
    fn packed_slices_follow_array_indexing_and_take_an_offset_expression() {
        let module = parse_module(
            "block b(clk, words: u16[2], i: u1, offset: u4) -> u8 { words[i][offset + u4:1+:uN[8]] }",
        )
        .unwrap();
        let Expr::WidthSlice {
            value,
            start,
            width,
        } = &module.blocks[0].output
        else {
            panic!("expected width slice");
        };
        assert!(matches!(value.as_ref(), Expr::Index { .. }));
        assert!(matches!(
            start.as_ref(),
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert_eq!(
            *width,
            WidthSliceType::Bits {
                signed: Box::new(Expr::Bool(false)),
                bit_count: Box::new(Expr::Literal {
                    ty: None,
                    text: "8".into()
                })
            }
        );
    }

    #[test]
    fn negative_number_tokens_are_distinct_from_unary_negation() {
        let module = parse_module(
            "block b(clk, x: u8) -> (u1, u1, u1) { (x[-0+:u1], x[- 0+:u1], x[-(0)+:u1]) }",
        )
        .unwrap();
        let Expr::Tuple(items) = &module.blocks[0].output else {
            panic!("expected tuple of slices");
        };
        let starts: Vec<_> = items
            .iter()
            .map(|expr| {
                let Expr::WidthSlice { start, .. } = expr else {
                    panic!("expected slice");
                };
                start.as_ref()
            })
            .collect();
        assert!(matches!(starts[0], Expr::Literal { ty: None, text } if text == "-0"));
        for expr in &starts[1..] {
            assert!(matches!(
                expr,
                Expr::Unary {
                    op: UnaryOp::Neg,
                    ..
                }
            ));
        }
    }

    #[test]
    fn parses_parametric_slice_types_and_parenthesized_sources() {
        let module = parse_module(
            r#"block b(clk, en: bool, x: xN[false][0x10]) -> (u4, s4) {
                ((if en { x } else { u16:0 })[0+:bits[0b100]],
                 (u8:0xff)[1+:xN[!false][2 + 2]])
            }"#,
        )
        .unwrap();
        let Expr::Tuple(items) = &module.blocks[0].output else {
            panic!("expected tuple");
        };
        assert!(
            matches!(&items[0], Expr::WidthSlice { value, .. } if matches!(value.as_ref(), Expr::If { .. }))
        );
        let Expr::WidthSlice {
            width: WidthSliceType::Bits { signed, bit_count },
            ..
        } = &items[1]
        else {
            panic!("expected parametric slice type");
        };
        assert!(matches!(
            signed.as_ref(),
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
        assert!(matches!(
            bit_count.as_ref(),
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn annotations_take_parenthesized_literal_terms() {
        for (offset, text) in [
            ("u8:((2))", "2"),
            ("u8:('a')", "0x61"),
            ("u8:(false)", "0b0"),
            ("u8:(u3:65)", "65"),
        ] {
            let source = format!("block b(clk, x: u8) -> u1 {{ x[{offset}+:u1] }}");
            let module = parse_module(&source).unwrap();
            let Expr::WidthSlice { start, .. } = &module.blocks[0].output else {
                panic!("expected width slice");
            };
            assert_eq!(
                start.as_ref(),
                &Expr::Literal {
                    ty: Some(Type::Bits {
                        signed: false,
                        width: 8,
                    }),
                    text: text.into(),
                }
            );
        }
        for expr in [
            "x[u8:(x)+:u1]",
            "x[u8:(1 + 1)+:u1]",
            "x[u8:(- 1)+:u1]",
            "x[u8:(2,)+:u1]",
            "u8:(2)[0+:u1]",
        ] {
            let source = format!("block b(clk, x: u8) -> u1 {{ {expr} }}");
            assert!(parse_module(&source).is_err(), "{source}");
        }
        assert!(parse_module("block b(clk) -> u1 { (u8:(2))[0+:u1] }").is_ok());
    }

    #[test]
    fn tuple_projections_use_unsigned_u32_literal_indices() {
        for spelling in ["1", "0x_01_", "0b_0__1_"] {
            let source = format!("block b(clk, pair: (u8, u8)) -> u8 {{ pair.{spelling} }}");
            let module = parse_module(&source).unwrap();
            assert!(
                matches!(&module.blocks[0].output, Expr::Field { name, .. } if name == "1"),
                "{spelling}"
            );
        }
        for spelling in ["-0", "-1", "01", "0b2", "0x__", "0x1_0000_0000", "'a'"] {
            let source = format!("block b(clk, pair: (u8, u8)) -> u8 {{ pair.{spelling} }}");
            assert!(parse_module(&source).is_err(), "{source}");
        }
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
    fn slice_numbers_follow_dslx_radix_and_separator_rules() {
        for literal in ["0", "19", "0x_f__f_", "0b_10__1_"] {
            let source = format!("block b(clk, x: uN[{literal}]) -> u1 {{ x[{literal}+:u1] }}");
            assert!(parse_module(&source).is_ok(), "{source}");
        }
        // The keyword after a decimal or binary literal starts a new token.
        for offset in ["1as u8", "0b1as u8"] {
            let source = format!("block b(clk, x: u8) -> u1 {{ x[{offset}+:u1] }}");
            assert!(parse_module(&source).is_ok(), "{source}");
        }
        for literal in [
            "00", "08", "1_0", "0_1", "0x__", "0b_", "0o10", "0B1", "0X8",
        ] {
            for source in [
                format!("block b(clk, x: u8) -> u1 {{ x[{literal}+:u1] }}"),
                format!("block b(clk, x: uN[{literal}]) -> u1 {{ x[0+:u1] }}"),
            ] {
                assert!(parse_module(&source).is_err(), "{source}");
            }
        }
        assert!(fixed_width("u08").is_none());
    }

    #[test]
    fn byte_characters_preserve_dslx_escapes_and_intrinsic_width() {
        for (spelling, byte) in [
            ("'A'", 65),
            ("'''", 39),
            (r"'\0'", 0),
            (r"'\t'", 9),
            (r"'\n'", 10),
            (r"'\r'", 13),
            (r"'\\'", 92),
            (r"'\''", 39),
            (r#"'\"'"#, 34),
            (r"'\xff'", 255),
        ] {
            let source = format!("block b(clk, x: u8) -> u1 {{ x[{spelling}+:u1] }}");
            let module = parse_module(&source).unwrap();
            let Expr::WidthSlice { start, .. } = &module.blocks[0].output else {
                panic!("expected slice");
            };
            assert_eq!(
                start.as_ref(),
                &Expr::Literal {
                    ty: Some(Type::Bits {
                        signed: false,
                        width: 8
                    }),
                    text: format!("0x{byte:02x}"),
                }
            );
        }
        assert!(parse_module("block b(clk) -> u1 { 'A'[0+:u1] }").is_ok());
        assert!(parse_module("block b(clk, x: u8) -> u4 { x[0+:uN[u32:'\\x04']] }").is_ok());
        for invalid in [
            "''",
            "'ab'",
            "'é'",
            "'\n'",
            r"'\a'",
            r"'\u{41}'",
            r"'\xF'",
            r"'\x123'",
            r"u7:'\0'",
            "u8:'A'[0+:u1]",
        ] {
            let source = format!("block b(clk) -> u8 {{ {invalid} }}");
            assert!(parse_module(&source).is_err(), "{source}");
        }
    }

    #[test]
    fn comparison_chains_in_offsets_require_parentheses() {
        for (offset, accepted) in [
            ("1 == 1 == 1", false),
            ("1 == (1) == 1", false),
            ("(1 == 1) == 1", true),
            ("1 == (1 == 1)", true),
        ] {
            let source = format!("block b(clk, x: u8) -> u1 {{ x[{offset}+:u1] }}");
            assert_eq!(parse_module(&source).is_ok(), accepted, "{source}");
        }
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
            "block b(clk, x: u8) -> u4 { x[+:u4] }",
            "block b(clk, x: u8) -> u4 { x[0+:] }",
            "block b(clk, x: u8) -> u4 { x[0+:4] }",
            "block b(clk, x: u8, offset: u8) -> u4 { x[offset-1+:u4] }",
            "block b(clk) -> u4 { u8:0xd9[6+:u4] }",
            "block b(clk, x: u8) -> u4 { if true { x } else { x }[0+:u4] }",
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
