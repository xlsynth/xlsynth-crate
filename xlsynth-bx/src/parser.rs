// SPDX-License-Identifier: Apache-2.0

//! Parser for BX's structural declarations and DSLX-style value expressions.

pub mod outline;
pub(crate) mod source_bytes;

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::{IrBits, IrValue};

use crate::BxWarningKind;
use crate::ast::{
    BinaryOp, BindingPattern, Block, CallbackParam, ConstDef, EnumDef, EnumMember, Expr,
    ExternBlock, ExternFn, ForIterable, ForLoop, Function, LetBinding, MapExpr, MatchArm,
    MatchPattern, Module, OutputCycles, Param, ParametricBinding, SourcePosition, Stmt, StructDef,
    Type, UnaryOp, UnrollFor, UsePath, WidthSliceType,
};

/// Bound recursive syntax, including DSLX use trees generated from a BX AST.
pub(crate) const MAX_SYNTAX_DEPTH: usize = 64;

/// A borrowed spelling and byte offset in the original UTF-8 input.
#[derive(Clone, Copy)]
pub(crate) struct Token<'s> {
    pub(crate) text: &'s str,
    pub(crate) start: usize,
    character: Option<u8>,
}

const SOURCE_START: SourcePosition = SourcePosition { line: 1, column: 1 };

/// Advance a diagnostic position across UTF-8 source text.
fn advance_position(start: SourcePosition, text: &str) -> SourcePosition {
    let line = start.line + text.bytes().filter(|&b| b == b'\n').count();
    let column = match text.rsplit_once('\n') {
        Some((_, suffix)) => 1 + suffix.chars().count(),
        None => start.column + text.chars().count(),
    };
    SourcePosition { line, column }
}

pub(crate) fn diagnostic(source: &str, offset: usize, message: &str) -> String {
    let SourcePosition { line, column } = advance_position(SOURCE_START, &source[..offset]);
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

/// DSLX accepts the special name `self` as a name definition or reference.
fn is_function_identifier(text: &str) -> bool {
    is_use_identifier(text) || text == "self"
}

/// DSLX permits `self` only at the head of an enum path and `_` only after
/// a qualifier. Both spellings are valid in declarations.
pub(crate) fn validate_enum_member_path(name: &str) -> Result<(), String> {
    let mut components = name.split("::");
    if components.next() == Some("_") {
        return Err("wildcard pattern '_' cannot be used as a reference".into());
    }
    if components.any(|member| member == "self") {
        return Err("expected an identifier after '::', got 'self'".into());
    }
    Ok(())
}

/// Pure BX expressions can refer to BX names and DSLX member spellings.
fn is_value_identifier(text: &str) -> bool {
    is_identifier(text) || is_function_identifier(text)
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

/// Tokenize checked DSLX source, preserving identifier and call punctuation.
pub(crate) fn dslx_tokens(source: &str) -> Result<Vec<Token<'_>>, String> {
    lex(source, Dialect::Dslx, None)
}

/// Tokenize a module, or a callback span using the lexical rules of a local
/// function. Offsets always refer to the full source for diagnostics.
fn lex(
    source: &str,
    dialect: Dialect,
    function_body: Option<std::ops::Range<usize>>,
) -> Result<Vec<Token<'_>>, String> {
    let mut tokens = Vec::new();
    let callback = function_body.is_some();
    let span = function_body.unwrap_or(0..source.len());
    let bytes = &source.as_bytes()[..span.end];
    let mut i = span.start;
    let mut brace_depth = 0usize;
    // A function or enum ends at its closing brace. A constant ends at its
    // semicolon, even when its initializer contains braced expressions.
    let mut dslx_item = None;
    let mut signature_depth = 0usize;
    let mut in_function_body = false;
    while i < bytes.len() {
        let start = i;
        let rest = &source[i..span.end];
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
            if dialect == Dialect::Bx && (dslx_item.is_some() || callback) {
                let message = match dslx_item {
                    Some("const") => "use a // comment in a constant",
                    Some("enum") => "use a // comment in an enum",
                    _ => "use a // comment in a function",
                };
                return Err(diagnostic(source, start, message));
            }
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
                if dialect == Dialect::Bx
                    && dslx_item.is_none()
                    && !callback
                    && bytes[i..].starts_with(b"!=")
                    && !bytes[i..].starts_with(b"!==")
                {
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
        } else if dialect == Dialect::Dslx && b == b'\'' && scan_character(rest).is_none() {
            // DSLX also uses a standalone apostrophe token.
            i += 1;
        } else if b == b'\'' {
            let (value, len) = scan_character(rest)
                .ok_or_else(|| diagnostic(source, start, "invalid byte character literal"))?;
            character = Some(value);
            i += len;
        } else if dialect == Dialect::Dslx && matches!(b, b'"' | b'`') {
            // An escaped quote or a delimiter in a string is part of the
            // literal. Preserve the original spelling for the outline.
            let mut escaped = false;
            let end = rest.as_bytes()[1..]
                .iter()
                .position(|&c| {
                    let closes = c == b && !escaped;
                    escaped = c == b'\\' && !escaped;
                    closes
                })
                .ok_or_else(|| diagnostic(source, start, "unterminated DSLX string"))?;
            i += end + 2;
        } else if b == b'"' {
            let end = rest[1..]
                .find(['"', '\n', '\r'])
                .ok_or_else(|| diagnostic(source, start, "unterminated symbol string"))?;
            i += end + 1;
            if bytes[i] != b'"' {
                return Err(diagnostic(source, start, "unterminated symbol string"));
            }
            i += 1;
        } else if dialect == Dialect::Dslx && rest.starts_with(">>") {
            // Separate closing angles in nested parametric type arguments.
            // Expressions in an outline retain their source spelling.
            i += 1;
        } else if dialect == Dialect::Bx && rest.starts_with("..=") {
            i += 3;
        } else if rest.starts_with(":=")
            || rest.starts_with("+:")
            || rest.starts_with("::")
            || rest.starts_with("->")
            || rest.starts_with("=>")
            || (dialect == Dialect::Bx && rest.starts_with(".."))
            || rest.get(..2).and_then(BinaryOp::from_spelling).is_some()
        {
            i += 2;
        } else if b"#(){}[]:,.=;!&|^+-*/%<>".contains(&b) || (dialect == Dialect::Bx && b == b'@') {
            i += 1;
        } else {
            return Err(diagnostic(source, start, "unexpected character"));
        }
        let text = &source[start..i];
        // DSLX consumes a trailing bang in a name even when '=' immediately
        // follows it. Keep BX's compact inequality spelling in block bodies.
        let item_start = brace_depth == 0
            && tokens
                .last()
                .is_none_or(|t: &Token| matches!(t.text, "pub" | ";" | "}"));
        match text {
            "fn" | "const" | "enum" if item_start && dslx_item.is_none() => dslx_item = Some(text),
            "(" | "[" if dslx_item == Some("fn") && !in_function_body => signature_depth += 1,
            ")" | "]" if dslx_item == Some("fn") && !in_function_body => {
                signature_depth = signature_depth.saturating_sub(1);
            }
            "{" => {
                // Defaults (= {expr}) and type dimensions can contain braces
                // before the function body. Keep DSLX lexing through both.
                if dslx_item == Some("fn")
                    && brace_depth == 0
                    && signature_depth == 0
                    && tokens.last().is_some_and(|t| t.text != "=")
                {
                    in_function_body = true;
                }
                brace_depth += 1;
            }
            "}" => brace_depth = brace_depth.saturating_sub(1),
            _ => { /* Other tokens do not enter or leave a definition. */ }
        }
        if brace_depth == 0
            && (matches!(
                (dslx_item, text),
                (Some("enum"), "}") | (Some("const"), ";")
            ) || (in_function_body && text == "}"))
        {
            dslx_item = None;
            in_function_body = false;
        }
        tokens.push(Token {
            text,
            start,
            character,
        });
    }
    tokens.push(Token {
        text: "",
        start: span.end,
        character: None,
    });
    Ok(tokens)
}

/// The DSLX outline scanner additionally recognizes backtick strings and
/// standalone apostrophes.
#[derive(Clone, Copy, PartialEq)]
enum Dialect {
    Bx,
    Dslx,
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

/// Parse a source module without loading its imports. Imported spellings of
/// `f<a,b>(x)` are parsed as calls. Use [`crate::parse_module_with_options`] to
/// distinguish these calls from tuples of comparisons on imported constants,
/// or an imported `zero!<callback>()` from the intrinsic `zero!<Type>()`.
/// Syntax diagnostics carry one-based line and column.
pub fn parse_module(source: &str) -> Result<Module, String> {
    Parser::new(source, false)?.module_with_imports(None)
}

/// Parse with the callable templates exported by the module's imports. Their
/// source spellings distinguish parametric arguments from pairs of comparisons.
pub(crate) fn parse_with_templates(
    source: &str,
    templates: &BTreeSet<String>,
) -> Result<Module, String> {
    Parser::new(source, false)?.module_with_imports(Some(templates))
}

/// Collect use leaves before parsing expressions that depend on import kinds.
pub(crate) fn module_uses(source: &str) -> Result<Vec<UsePath>, String> {
    let mut paths = Vec::new();
    Parser::new(source, false)?.scan_module_bindings(|path, _| paths.push(path))?;
    Ok(paths)
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
    dialect: Dialect,
    tokens: Vec<Token<'s>>,
    pos: usize,
    depth: usize,
    dslx_type_display: bool,
    value_context: ValueContext,
    /// Pure operands such as counted-loop iterables and map inputs permit
    /// lexical value scopes while keeping the enclosing identifier dialect.
    expression_grammar: ExpressionGrammar,
    /// Tracks module and lexical bindings that hide the builtin map in a block.
    block_map_shadowed: bool,
    /// Until its use is reached, an imported map may be a callable or an enum.
    forward_map_import: bool,
    /// Type parameters that can annotate integer literals in this block.
    block_type_params: BTreeSet<&'s str>,
    /// Function value parametrics permit expressions in bit and array
    /// dimensions.
    dependent_types: bool,
    /// Local template declarations disambiguate angle brackets from
    /// comparisons in a comma-separated expression list. Block callers can
    /// use function and block templates declared later in the source.
    callable_templates: BTreeMap<String, TemplateBinding>,
    /// Entries hidden by currently active data or callable bindings.
    hidden_templates: Vec<(String, TemplateBinding)>,
}

/// A binding can be a callable template itself, or a module exporting
/// templates. The syntax-only parser has no member list for a use binding.
struct TemplateBinding {
    direct: Option<usize>,
    members: Option<BTreeSet<String>>,
}

impl TemplateBinding {
    fn local(start: usize) -> Self {
        Self {
            direct: Some(start),
            members: Some(BTreeSet::new()),
        }
    }
}

/// Lambdas use DSLX function grammar. Inside a BX block they can also name
/// the block's instance output ports, including in capture diagnostics.
#[derive(Clone, Copy, PartialEq)]
enum ValueContext {
    Block,
    /// Functions, module constants, and enum initializers use DSLX value
    /// grammar.
    DslxItem,
    BlockLambda,
}

/// Pure block scopes allow lexical lets while retaining BX fold precedence.
#[derive(Clone, Copy, PartialEq)]
enum ExpressionGrammar {
    Block,
    BlockScope,
    DslxOperand,
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

/// The inferred types and explicitly supplied callbacks of a block, each in
/// its own declaration order.
#[derive(Default)]
struct BlockParametrics {
    type_params: Vec<String>,
    callback_params: Vec<CallbackParam>,
}

/// Retains the built tree's height until its parent is constructed. Syntactic
/// nesting alone cannot bound a left-associative expression tree.
struct ParsedExpr {
    value: Expr,
    height: usize,
    spelling: ExprSpelling,
}

/// An iterable being parsed, retaining expression heights through parentheses.
enum ParsedIterable {
    Array(ParsedExpr),
    Range {
        start: ParsedExpr,
        limit: ParsedExpr,
        inclusive: bool,
    },
}

impl ParsedIterable {
    fn into_ast(self) -> (ForIterable, usize) {
        match self {
            Self::Array(value) => (ForIterable::Array(value.value), value.height),
            Self::Range {
                start,
                limit,
                inclusive,
            } => (
                ForIterable::Range {
                    start: start.value,
                    limit: limit.value,
                    inclusive,
                },
                start.height.max(limit.height),
            ),
        }
    }
}

/// Source distinctions needed for DSLX postfix and comparison precedence.
#[derive(Clone, Copy, PartialEq)]
enum ExprSpelling {
    Ordinary,
    Parenthesized,
    AnnotatedLiteral,
}

/// Distinguishes a struct constructor from a brace terminating the subject
/// of a conditional, range, or match.
#[derive(Clone, Copy, PartialEq)]
enum Braces {
    Struct,
    Body,
    MatchArms,
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
        let tokens = lex(source, Dialect::Bx, None)?;
        let mut callable_templates = BTreeMap::new();
        let mut block_header = None;
        for ts in tokens.windows(3) {
            match (ts[0].text, ts[1].text, ts[2].text) {
                ("fn", name, "<") => {
                    callable_templates.insert(name.to_owned(), TemplateBinding::local(ts[1].start));
                }
                ("block", _, "<") => block_header = Some(ts[1]),
                (":", "fn", "(") => {
                    if let Some(name) = block_header.take() {
                        callable_templates
                            .insert(name.text.to_owned(), TemplateBinding::local(name.start));
                    }
                }
                (">", _, _) => block_header = None,
                _ => { /* Other tokens do not identify a callable template. */ }
            }
        }
        Ok(Self {
            source,
            dialect: Dialect::Bx,
            tokens,
            pos: 0,
            depth: 0,
            dslx_type_display,
            value_context: ValueContext::Block,
            expression_grammar: ExpressionGrammar::Block,
            block_map_shadowed: false,
            forward_map_import: false,
            block_type_params: BTreeSet::new(),
            dependent_types: false,
            callable_templates,
            hidden_templates: Vec::new(),
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
        } else if text == ">" && matches!(self.peek(0), ">>" | ">=") {
            // In a type argument list, consume just the closing angle. This
            // leaves the outer '>' of zero!<Packet<2>>() or the '=' of a let.
            let token = &mut self.tokens[self.pos];
            token.text = &token.text[1..];
            token.start += 1;
            true
        } else {
            false
        }
    }

    fn error(&self, message: &str) -> String {
        let token = self.tokens[self.pos];
        if self.dialect == Dialect::Bx
            && token.text == "#"
            && self.peek(1) == "["
            && self.peek(2) == "allow"
        {
            return diagnostic(
                self.source,
                token.start,
                "place a BX warning attribute before the block definition, including its optional 'pub'",
            );
        }
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

    fn dslx_value_grammar(&self) -> bool {
        self.value_context != ValueContext::Block
    }

    /// Restore callable names used to disambiguate syntax when leaving this
    /// scope.
    fn with_name_scope<T>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let first = self.hidden_templates.len();
        let map_shadowed = self.block_map_shadowed;
        let result = parse(self);
        self.block_map_shadowed = map_shadowed;
        self.callable_templates
            .extend(self.hidden_templates.drain(first..));
        result
    }

    fn shadow_callable(&mut self, name: &str) {
        self.block_map_shadowed |= name == "map";
        if let Some(entry) = self.callable_templates.remove_entry(name) {
            self.hidden_templates.push(entry);
        }
    }

    /// A wildcard in a loop or lambda discards its value without binding a
    /// name.
    fn hide_pattern_name(&mut self, name: &str) {
        if name != "_" {
            self.shadow_callable(name);
        }
    }

    fn identifier_predicate(&self) -> fn(&str) -> bool {
        if self.dslx_value_grammar() {
            is_function_identifier
        } else {
            is_identifier
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        self.take_identifier(self.identifier_predicate())
    }

    fn take_identifier(&mut self, valid: fn(&str) -> bool) -> Result<String, String> {
        let text = self.peek(0);
        if valid(text) && (!self.dslx_value_grammar() || is_function_identifier(text)) {
            self.pos += 1;
            Ok(text.to_owned())
        } else {
            Err(self.error("expected identifier"))
        }
    }

    /// Find top-level imports and bare callable declarations that shadow map.
    fn scan_module_bindings(
        &mut self,
        mut visit: impl FnMut(UsePath, usize),
    ) -> Result<(), String> {
        let mut depth = 0usize;
        let mut bindings = BTreeSet::new();
        let mut paths = Vec::new();
        while !self.peek(0).is_empty() {
            let token = self.tokens[self.pos];
            self.pos += 1;
            match token.text {
                "{" => depth += 1,
                "}" => depth = depth.saturating_sub(1),
                "fn" | "block" | "struct"
                    if depth == 0
                        && self.peek(0) == "map"
                        && matches!(self.peek(1), "(" | "<" | "{") =>
                {
                    self.block_map_shadowed = true;
                }
                "use"
                    if depth == 0
                        && (self.pos == 1
                            || matches!(self.tokens[self.pos - 2].text, ";" | "}")) =>
                {
                    self.use_tree(&mut paths, Some(&mut bindings))?;
                    self.expect(";")?;
                    for path in paths.drain(..) {
                        self.forward_map_import |= path.binding() == Some("map");
                        visit(path, token.start);
                    }
                }
                _ => { /* The module parser checks all other syntax. */ }
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// Install visible import names before parsing, including uses declared
    /// after a block. Local value bindings shadow a module and all its members.
    fn module_with_imports(
        &mut self,
        templates: Option<&BTreeSet<String>>,
    ) -> Result<Module, String> {
        if self
            .tokens
            .iter()
            .any(|token| matches!(token.text, "<" | "map"))
        {
            let mut imports = BTreeMap::new();
            self.scan_module_bindings(|mut path, start| {
                let name = path.components.pop().expect("parsed use leaf");
                let binding = TemplateBinding {
                    direct: templates
                        .is_none_or(|known| known.contains(&name))
                        .then_some(start),
                    members: templates.map(|_| BTreeSet::new()),
                };
                imports.insert(name, binding);
            })?;
            for name in templates.into_iter().flatten() {
                if let Some((head, member)) = name.split_once("::")
                    && let Some(members) = imports.get_mut(head).and_then(|b| b.members.as_mut())
                {
                    members.insert(member.to_owned());
                }
            }
            self.callable_templates.extend(imports);
        }
        self.module()
    }

    fn module(&mut self) -> Result<Module, String> {
        let mut module = Module::default();
        let mut use_bindings = BTreeSet::new();
        // Advance from the previous name so recording positions is linear
        // across all declarations, including multiple blocks on one line.
        let mut position = SOURCE_START;
        let mut offset = 0;
        while !self.peek(0).is_empty() {
            let allowed_warnings = self.block_warning_attributes()?;
            let public = self.eat("pub");
            if self.eat("block") {
                let name_offset = self.tokens[self.pos].start;
                position = advance_position(position, &self.source[offset..name_offset]);
                offset = name_offset;
                let block =
                    self.with_name_scope(|p| p.block(public, position, allowed_warnings))?;
                module.declaration_order.push(block.name.clone());
                module.blocks.push(block);
            } else if !allowed_warnings.is_empty() {
                return Err(self.error("expected a block definition after #[allow(...)]"));
            } else if self.eat("fn") {
                self.value_context = ValueContext::DslxItem;
                let function = self.with_name_scope(|p| p.function(public))?;
                self.value_context = ValueContext::Block;
                module.declaration_order.push(function.name.clone());
                module.functions.push(function);
            } else if self.eat("const") {
                self.value_context = ValueContext::DslxItem;
                let constant = self.with_name_scope(|p| p.const_def(public))?;
                self.value_context = ValueContext::Block;
                module.declaration_order.push(constant.name.clone());
                module.constants.push(constant);
            } else if self.eat("struct") {
                let name = self.ident()?;
                module.declaration_order.push(name.clone());
                self.expect("{")?;
                let fields = self.list("}", Self::param)?;
                module.structs.push(StructDef {
                    name,
                    public,
                    fields,
                });
            } else if self.eat("enum") {
                self.value_context = ValueContext::DslxItem;
                let def = self.with_name_scope(|p| p.enum_def(public))?;
                self.value_context = ValueContext::Block;
                self.block_map_shadowed |= def.name == "map";
                module.declaration_order.push(def.name.clone());
                module.enums.push(def);
            } else if public {
                return Err(self.error("expected block, const, enum, fn, or struct after pub"));
            } else if self.eat("use") {
                let start = module.uses.len();
                self.use_tree(&mut module.uses, Some(&mut use_bindings))?;
                self.block_map_shadowed |= module.uses[start..]
                    .iter()
                    .any(|path| path.binding() == Some("map"));
                module.declaration_order.extend(
                    module.uses[start..]
                        .iter()
                        .filter_map(UsePath::binding)
                        .map(str::to_owned),
                );
                self.expect(";")?;
            } else if self.peek(0) == "import" {
                return Err(self
                    .error("replace `import` with `use` and separate path components with `::`"));
            } else if self.eat("extern") {
                if self.eat("fn") {
                    let function = self.extern_fn()?;
                    module.declaration_order.push(function.name.clone());
                    module.externs.push(function);
                } else if self.eat("block") {
                    let name = self.ident()?;
                    // TODO: 2026-11-06 Parse extern input and output cycles
                    // for checking timing across block instances.
                    let signature = self.block_signature(name, Self::param)?;
                    module.declaration_order.push(signature.name.clone());
                    self.expect(";")?;
                    module.extern_blocks.push(signature);
                } else {
                    return Err(self.error("expected fn or block after extern"));
                }
            } else {
                return Err(self.error(
                    "expected a block, const, enum, fn, struct, extern, or use declaration",
                ));
            }
        }
        Ok(module)
    }

    /// Read per-block warning suppressions before an optional visibility
    /// modifier.
    fn block_warning_attributes(&mut self) -> Result<BTreeSet<BxWarningKind>, String> {
        let mut allowed = BTreeSet::new();
        while self.eat("#") {
            self.expect("[")?;
            self.expect("allow")?;
            self.expect("(")?;
            if self.peek(0) == ")" {
                return Err(self.error("expected a BX warning name in #[allow(...)]"));
            }
            loop {
                let name = self.peek(0);
                let warning = BxWarningKind::from_name(name)
                    .ok_or_else(|| self.error("unknown BX warning in #[allow(...)]"))?;
                self.pos += 1;
                allowed.insert(warning);
                if !self.eat(",") || self.peek(0) == ")" {
                    break;
                }
            }
            self.expect(")")?;
            self.expect("]")?;
        }
        Ok(allowed)
    }

    /// Flatten one DSLX use tree in source order. BX additionally bounds the
    /// path depth and rejects duplicate leaf bindings in its own namespace.
    fn use_tree(
        &mut self,
        paths: &mut Vec<UsePath>,
        mut bx_bindings: Option<&mut BTreeSet<String>>,
    ) -> Result<(), String> {
        let mut prefix = Vec::new();
        let mut groups = Vec::new();
        loop {
            if bx_bindings.is_some() && prefix.len() >= MAX_SYNTAX_DEPTH {
                return Err(
                    self.error(&format!("syntax nesting exceeds {MAX_SYNTAX_DEPTH} levels"))
                );
            }
            let offset = self.tokens[self.pos].start;
            let name = self.take_identifier(is_use_identifier)?;
            let has_children = self.eat("::");
            if !has_children
                && let Some(bindings) = &mut bx_bindings
                && !bindings.insert(name.clone())
            {
                return Err(diagnostic(
                    self.source,
                    offset,
                    &format!("duplicate use binding '{name}'"),
                ));
            }
            prefix.push(name);
            if has_children {
                if self.eat("{") {
                    groups.push(prefix.len());
                }
                continue;
            }
            paths.push(UsePath {
                components: prefix.clone(),
            });
            loop {
                let Some(&parent) = groups.last() else {
                    return Ok(());
                };
                if self.eat("}") {
                    groups.pop();
                } else {
                    self.expect(",")?;
                    prefix.truncate(parent);
                    break;
                }
            }
        }
    }

    /// Parse a local name or a member of an imported DSLX module.
    fn name_or_qualified_member(&mut self) -> Result<String, String> {
        let head = self.take_identifier(is_value_identifier)?;
        if !self.eat("::") {
            return Ok(head);
        }
        let member = self.take_identifier(is_dslx_member_name)?;
        Ok(format!("{head}::{member}"))
    }

    /// An imported enum value can have a module, type, and member component.
    fn named_value(&mut self, mut name: String) -> Result<Expr, String> {
        if self.eat("::") {
            let member = self.take_identifier(is_function_identifier)?;
            name.push_str("::");
            name.push_str(&member);
        }
        Ok(Expr::Name(name))
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
        let name = if self.dslx_value_grammar() {
            // A self parameter is reserved for DSLX impl and trait methods.
            self.take_identifier(is_use_identifier)?
        } else {
            self.ident()?
        };
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

    /// Parse a module constant with an optional DSLX type annotation.
    fn const_def(&mut self, public: bool) -> Result<ConstDef, String> {
        let name = self.ident()?;
        let ty = if self.eat(":") {
            Some(self.ty()?)
        } else {
            None
        };
        self.expect("=")?;
        let value = self.expr()?.value;
        self.expect(";")?;
        Ok(ConstDef {
            name,
            public,
            ty,
            value,
        })
    }

    /// Parse DSLX's numeric enum declaration and its compile-time values.
    fn enum_def(&mut self, public: bool) -> Result<EnumDef, String> {
        let name = self.ident()?;
        let underlying = if self.eat(":") {
            Some(self.ty()?)
        } else {
            None
        };
        self.expect("{")?;
        let members = self.list("}", |p| {
            let name = p.ident()?;
            p.expect("=")?;
            Ok(EnumMember {
                name,
                value: p.expr()?.value,
            })
        })?;
        if underlying.is_none() && members.is_empty() {
            return Err(self.error("an empty numeric enum requires an explicit underlying type"));
        }
        Ok(EnumDef {
            name,
            public,
            underlying,
            members,
        })
    }

    /// Parse pure local lets and a trailing value. A body ending in a
    /// semicolon or an empty body returns unit, as in DSLX.
    fn function(&mut self, public: bool) -> Result<Function, String> {
        let name = self.ident()?;
        let mut value_parametrics = Vec::new();
        let mut callback_params = Vec::new();
        if self.eat("<") {
            // An early callback annotation may use a module constant even
            // when the first value parametric is declared later in this list.
            self.dependent_types = true;
            if self.peek(0) == ">" {
                return Err(self.error("expected a function parameter"));
            }
            self.list(">", |p| {
                if p.peek(1) == ":" && p.peek(2) == "fn" {
                    let param = p.callback_param()?;
                    p.shadow_callable(&param.name);
                    callback_params.push(param);
                } else {
                    // DSLX permits `self` as a value parametric even though a
                    // data parameter named `self` requires an impl method.
                    let name = p.ident()?;
                    p.expect(":")?;
                    let ty = p.ty()?;
                    let default = if p.eat("=") {
                        p.expect("{")?;
                        let value = p.expr()?.value;
                        p.expect("}")?;
                        Some(value)
                    } else {
                        None
                    };
                    p.shadow_callable(&name);
                    value_parametrics.push(ParametricBinding {
                        name,
                        ty,
                        default,
                        index: value_parametrics.len() + callback_params.len(),
                    });
                }
                Ok(())
            })?;
        }
        self.expect("(")?;
        let params = self.list(")", Self::param)?;
        let result = if self.eat("->") {
            self.ty()?
        } else {
            Type::Tuple(Vec::new())
        };
        self.expect("{")?;
        for param in &params {
            self.shadow_callable(&param.name);
        }
        let body = self.value_scope()?.value;
        self.dependent_types = false;
        Ok(Function {
            name,
            public,
            callback_params,
            value_parametrics,
            params,
            result,
            body,
        })
    }

    /// Parse a named callable in a function or block's compile-time parameter
    /// list.
    fn callback_param(&mut self) -> Result<CallbackParam, String> {
        let name = self.take_identifier(is_use_identifier)?;
        self.expect(":")?;
        self.expect("fn")?;
        self.expect("(")?;
        let params = self.list(")", Self::ty)?;
        let result = if self.eat("->") {
            self.ty()?
        } else {
            Type::Tuple(Vec::new())
        };
        Ok(CallbackParam {
            name,
            params,
            result,
        })
    }

    /// Read a DSLX value block after its opening brace. A trailing semicolon
    /// discards the preceding expression and supplies unit as the result.
    fn value_scope(&mut self) -> Result<ParsedExpr, String> {
        self.with_name_scope(|p| p.value_scope_contents(false))
    }

    fn value_scope_contents(&mut self, dslx_tuple_patterns: bool) -> Result<ParsedExpr, String> {
        let mut bindings = Vec::new();
        let mut height = 0;
        let output = loop {
            if self.eat("}") {
                break ParsedExpr::leaf(Expr::Tuple(Vec::new()));
            }
            if self.eat("let") {
                let (binding, initializer_height) = self.let_binding(dslx_tuple_patterns)?;
                height = height.max(initializer_height);
                bindings.push(binding);
                self.expect(";")?;
            } else {
                let value = self.expr()?;
                if self.eat("}") {
                    break value;
                }
                self.expect(";")?;
                height = height.max(value.height);
                bindings.push(LetBinding {
                    pattern: BindingPattern::Name("_".into()),
                    ty: None,
                    value: value.value,
                });
            }
        };
        if bindings.is_empty() {
            return Ok(output);
        }
        self.compose(
            height.max(output.height),
            Expr::Scope {
                bindings,
                result: Box::new(output.value),
            },
        )
    }

    /// Parse nested tuple binders, with at most one rest marker per tuple.
    fn binding_pattern(
        &mut self,
        names: &mut BTreeSet<String>,
        valid_name: fn(&str) -> bool,
    ) -> Result<BindingPattern, String> {
        if self.eat("(") {
            return self.nested(|parser| {
                let mut rest = false;
                let fields = parser.list(")", |p| {
                    if p.eat("..") {
                        if rest {
                            return Err(p.error("duplicate rest marker in tuple pattern"));
                        }
                        rest = true;
                        Ok(BindingPattern::Rest)
                    } else {
                        p.binding_pattern(names, valid_name)
                    }
                })?;
                Ok(BindingPattern::Tuple(fields))
            });
        }
        let name = self.take_identifier(valid_name)?;
        if name != "_" && !names.insert(name.clone()) {
            return Err(self.error(&format!("duplicate name '{name}' in binding pattern")));
        }
        Ok(BindingPattern::Name(name))
    }

    /// Parse the common port list and result of a defined or external block.
    /// Externs pass the ordinary typed-port parser. A defined block's
    /// `data_param` also records an optional stage, as in `r: u8 @1`.
    fn block_signature(
        &mut self,
        name: String,
        mut data_param: impl FnMut(&mut Self) -> Result<Param, String>,
    ) -> Result<ExternBlock, String> {
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
                data_param(parser)
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

    /// Parse a decimal cycle number in an input or output annotation.
    fn cycle_number(&mut self, port_kind: &str) -> Result<u64, String> {
        let cycle = self.peek(0).parse::<u64>().map_err(|_| {
            self.error(&format!(
                "expected an unsigned 64-bit decimal {port_kind} cycle"
            ))
        })?;
        self.pos += 1;
        Ok(cycle)
    }

    /// Parse `@N` or `@(port: N, ...)` after the result type.
    fn output_cycles(&mut self) -> Result<Option<OutputCycles>, String> {
        if !self.eat("@") {
            return Ok(None);
        }
        if !self.eat("(") {
            return Ok(Some(OutputCycles::Uniform(self.cycle_number("output")?)));
        }
        let entries = self.list(")", |parser| {
            let name = parser.ident()?;
            parser.expect(":")?;
            Ok((name, parser.cycle_number("output")?))
        })?;
        let mut cycles = BTreeMap::new();
        for (name, cycle) in entries {
            if cycles.contains_key(&name) {
                return Err(self.error(&format!("duplicate output cycle for '{name}'")));
            }
            cycles.insert(name, cycle);
        }
        Ok(Some(OutputCycles::ByPort(cycles)))
    }

    /// Read block type parameters and callable signatures in one namespace.
    fn block_parametrics(&mut self) -> Result<BlockParametrics, String> {
        let mut params = BlockParametrics::default();
        if !self.eat("<") {
            return Ok(params);
        }
        if self.peek(0) == ">" {
            return Err(self.error("expected a block type parameter"));
        }
        let mut names = BTreeSet::new();
        self.list(">", |p| {
            let token = p.tokens[p.pos];
            let callable = p.peek(1) == ":" && p.peek(2) == "fn";
            if !names.insert(token.text) {
                let kind = if callable { "callable" } else { "type" };
                return Err(diagnostic(
                    p.source,
                    token.start,
                    &format!("duplicate block {kind} parameter '{}'", token.text),
                ));
            }
            if callable {
                params.callback_params.push(p.callback_param()?);
            } else {
                let name = p.ident()?;
                if p.eat(":") {
                    p.expect("type")?;
                }
                p.block_type_params.insert(token.text);
                params.type_params.push(name);
            }
            Ok(())
        })?;
        Ok(params)
    }

    fn block(
        &mut self,
        public: bool,
        name_position: SourcePosition,
        allowed_warnings: BTreeSet<BxWarningKind>,
    ) -> Result<Block, String> {
        let name = self.ident()?;
        let params = self.block_parametrics()?;
        let mut input_cycles = BTreeMap::new();
        let signature = self.block_signature(name, |parser| {
            let param = parser.param()?;
            if parser.eat("@") {
                input_cycles.insert(param.name.clone(), parser.cycle_number("input")?);
            }
            Ok(param)
        })?;
        let output_cycles = self.output_cycles()?;
        self.expect("{")?;
        for name in &params.type_params {
            self.shadow_callable(name);
        }
        for formal in &params.callback_params {
            self.shadow_callable(&formal.name);
        }
        self.shadow_callable(&signature.clock);
        if let Some(reset) = &signature.reset {
            self.shadow_callable(reset);
        }
        for port in &signature.params {
            self.shadow_callable(&port.name);
        }
        let body = self.body()?;
        self.block_type_params.clear();
        let output = body
            .output
            .ok_or_else(|| self.error("block requires an output expression"))?;
        Ok(Block {
            name: signature.name,
            name_position: Some(name_position),
            allowed_warnings,
            public,
            type_params: params.type_params,
            callback_params: params.callback_params,
            clock: signature.clock,
            reset: signature.reset,
            params: signature.params,
            input_cycles,
            result: signature.result,
            output_cycles,
            statements: body.statements,
            output: output.value,
        })
    }

    fn ty(&mut self) -> Result<Type, String> {
        let singleton_tuples = self.dslx_type_display || self.dslx_value_grammar();
        Ok(self.parsed_ty(singleton_tuples)?.value)
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
        let (signed, bit_count) = self.bit_type_expressions()?;
        Ok((
            WidthSliceType::Bits {
                signed: Box::new(signed.value),
                bit_count: Box::new(bit_count.value),
            },
            signed.height.max(bit_count.height),
        ))
    }

    /// Read the expressions controlling signedness and width in a bits
    /// annotation.
    fn bit_type_expressions(&mut self) -> Result<(ParsedExpr, ParsedExpr), String> {
        let spelling = self.peek(0);
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
        Ok((signed, bit_count))
    }

    fn parsed_ty(&mut self, singleton_tuples: bool) -> Result<ParsedType, String> {
        let mut ty = self.type_atom(singleton_tuples)?;
        while self.eat("[") {
            let element = Box::new(ty.value);
            ty.value = if self.dependent_types && !self.plain_dimension(0) {
                let size = self.expr()?;
                ty.depth = ty.depth.max(size.height);
                Type::ParametricArray {
                    element,
                    size: Box::new(size.value),
                }
            } else {
                Type::Array {
                    element,
                    size: self.dimension()?,
                }
            };
            ty.depth += 1;
            if ty.depth > MAX_SYNTAX_DEPTH {
                return Err(self.error("type nesting exceeds 64 levels"));
            }
            self.expect("]")?;
        }
        Ok(ty)
    }

    /// The original concrete dimension syntax is one numeric token and a
    /// bracket.
    fn plain_dimension(&self, offset: usize) -> bool {
        number_token(self.peek(offset)) && self.peek(offset + 1) == "]"
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

    /// Read the concrete u32 arguments of a DSLX struct application.
    fn struct_arguments(&mut self) -> Result<Vec<u32>, String> {
        self.expect("<")?;
        self.list(">", |p| {
            if p.eat("u32") {
                p.expect(":")?;
            }
            let value = normalized_integer(p.peek(0))
                .and_then(|text| {
                    let (digits, radix) = radix_digits(&text);
                    u32::from_str_radix(digits, radix).ok()
                })
                .ok_or_else(|| {
                    p.error("expected a concrete nonnegative struct argument that fits u32")
                })?;
            p.pos += 1;
            Ok(value)
        })
    }

    fn type_atom(&mut self, singleton_tuples: bool) -> Result<ParsedType, String> {
        if self.eat("(") {
            return self.nested(|parser| {
                let member = |p: &mut Self| p.parsed_ty(singleton_tuples);
                if singleton_tuples {
                    let fields = parser.list(")", member)?;
                    parser.tuple_type(fields)
                } else {
                    parser.parenthesized(member, Self::tuple_type)
                }
            });
        }
        let spelling = self.peek(0);
        let value = if self.eat("bool") {
            Type::Bool
        } else if matches!(spelling, "bits" | "uN" | "sN" | "xN") {
            let concrete = if spelling == "xN" {
                matches!(self.peek(2), "true" | "false")
                    && self.peek(3) == "]"
                    && self.plain_dimension(5)
            } else {
                self.plain_dimension(2)
            };
            if self.dependent_types && !concrete {
                let (signed, width) = self.bit_type_expressions()?;
                return Ok(ParsedType {
                    value: Type::ParametricBits {
                        signed: Box::new(signed.value),
                        width: Box::new(width.value),
                    },
                    depth: signed.height.max(width.height),
                });
            }
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
            // A cast can be followed immediately by a less-than comparison.
            // Block type parameters can name scalar cast targets, including
            // in a tuple of comparisons such as (x as T < 1, 2 > (y)).
            let after_as = self.pos > 0 && self.tokens[self.pos - 1].text == "as";
            let name = self.name_or_qualified_member()?;
            if self.peek(0) == "<" && (!after_as || self.struct_arguments_end(self.pos).is_some()) {
                Type::Applied {
                    name,
                    args: self.struct_arguments()?,
                }
            } else {
                Type::Named(name)
            }
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
        self.with_name_scope(Self::block_body_contents)
    }

    fn block_body_contents(&mut self) -> Result<Body, String> {
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
                self.shadow_callable(&param.name);
                statements.push(Stmt::Reg {
                    name: param.name,
                    ty: param.ty,
                    reset,
                });
            } else if self.eat("let") {
                statements.push(self.let_stmt()?.0);
            } else if self.peek(0) == "inst" && is_identifier(self.peek(1)) {
                self.pos += 1;
                let name = self.ident()?;
                self.expect(":")?;
                self.shadow_callable(&name);
                statements.push(Stmt::Inst {
                    name,
                    block: if self.peek(1) == "::" {
                        self.name_or_qualified_member()?
                    } else {
                        self.ident()?
                    },
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
                if self.arm_has_else() {
                    let then_value = self.nested(Self::expression_scope)?;
                    self.expect("else")?;
                    let expr = self.if_tail(condition, then_value)?;
                    let output = self.expr_tail(expr, 0, Braces::Struct)?;
                    self.expect("}")?;
                    return Ok(Body {
                        statements,
                        output: Some(output),
                    });
                }
                let then_body = self.nested(Self::body)?;
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

    /// A block-level `if` with an `else` has two pure value scopes. A guard
    /// without an `else` can contain register writes and other statements.
    /// The opening brace of the first arm has already been consumed.
    fn arm_has_else(&self) -> bool {
        let mut depth = 0;
        for (i, token) in self.tokens[self.pos..].iter().enumerate() {
            match token.text {
                "{" => depth += 1,
                "}" if depth == 0 => return self.peek(i + 1) == "else",
                "}" => depth -= 1,
                _ => { /* Only braces delimit the arm. */ }
            }
        }
        false
    }

    /// Keep the named block statement representation for a simple binding.
    fn let_stmt(&mut self) -> Result<(Stmt, usize), String> {
        let (binding, height) = self.let_binding(true)?;
        let statement = match binding {
            LetBinding {
                pattern: BindingPattern::Name(name),
                ty,
                value,
            } if name != "_" => Stmt::Let { name, ty, value },
            binding => Stmt::LetPattern(binding),
        };
        Ok((statement, height))
    }

    /// Parse block and function let patterns alike. Each name enters scope
    /// after the entire initializer, including its annotation.
    fn let_binding(&mut self, dslx_tuple_patterns: bool) -> Result<(LetBinding, usize), String> {
        let mut names = BTreeSet::new();
        // Simple block lets can name child instances. Tuple members are
        // lexical values and can use primes and bangs in either dialect.
        let tuple_pattern = self.peek(0) == "(";
        let valid_name = if !tuple_pattern {
            self.identifier_predicate()
        } else if dslx_tuple_patterns {
            is_function_identifier
        } else {
            is_value_identifier
        };
        let pattern = self.binding_pattern(&mut names, valid_name)?;
        let ty = if self.eat(":") {
            // DSLX treats `(T)` as a singleton tuple at every nesting level.
            // Ordinary BX annotations retain their parenthesized type grammar.
            Some(if tuple_pattern && dslx_tuple_patterns {
                self.parsed_ty(true)?.value
            } else {
                self.ty()?
            })
        } else {
            None
        };
        self.expect("=")?;
        let value = self.expr()?;
        for name in names {
            self.shadow_callable(&name);
        }
        Ok((
            LetBinding {
                pattern,
                ty,
                value: value.value,
            },
            value.height,
        ))
    }

    fn expr(&mut self) -> Result<ParsedExpr, String> {
        self.full_expr(Braces::Struct)
    }

    /// DSLX ends a full expression at the closing brace of a standalone scope
    /// or the closing parenthesis of a fold that begins that expression.
    fn full_expr(&mut self, braces: Braces) -> Result<ParsedExpr, String> {
        if self.peek(0) == "{" && (self.in_value_body() || self.peek(1) == "let") {
            self.pos += 1;
            return self.nested(Self::expression_scope);
        }
        let dslx_precedence =
            self.dslx_value_grammar() || self.expression_grammar == ExpressionGrammar::DslxOperand;
        if dslx_precedence && self.eat("unroll_for!") {
            return self.nested(Self::unroll_for);
        }
        if dslx_precedence && self.starts_counted_for() {
            self.pos += 1;
            return self.nested(Self::counted_for);
        }
        self.expr_at(0, braces)
    }

    /// Parse the condition and the opening brace. An unparenthesized name here
    /// refers to the condition signal even when a brace follows it.
    fn condition(&mut self) -> Result<ParsedExpr, String> {
        let condition = self.full_expr(Braces::Body)?;
        self.expect("{")?;
        Ok(condition)
    }

    fn if_expr(&mut self) -> Result<ParsedExpr, String> {
        let condition = self.condition()?;
        let then_value = self.expression_scope()?;
        if self.in_value_body() && self.peek(0) != "else" {
            return self.compose(
                condition.height.max(then_value.height),
                Expr::If {
                    condition: Box::new(condition.value),
                    then_value: Box::new(then_value.value),
                    else_value: Box::new(Expr::Tuple(Vec::new())),
                },
            );
        }
        self.expect("else")?;
        self.if_tail(condition, then_value)
    }

    /// Enable DSLX tuple lets in a block's pure expression scope while
    /// preserving its fold precedence and the grammar of existing operands.
    fn expression_scope(&mut self) -> Result<ParsedExpr, String> {
        if self.dslx_value_grammar() || self.expression_grammar == ExpressionGrammar::DslxOperand {
            return self.value_scope();
        }
        let outer = std::mem::replace(&mut self.expression_grammar, ExpressionGrammar::BlockScope);
        let result = self.with_name_scope(|p| p.value_scope_contents(true));
        self.expression_grammar = outer;
        result
    }

    /// Braced match subjects and arm results introduce a pure value scope
    /// in both functions and blocks.
    fn match_value(&mut self, braces: Braces) -> Result<ParsedExpr, String> {
        if self.eat("{") {
            self.nested(Self::value_scope)
        } else {
            self.full_expr(braces)
        }
    }

    /// Retain arms in source order without constructing a nested condition
    /// tree. The result height depends on the deepest arm, not their count.
    fn match_expr(&mut self) -> Result<ParsedExpr, String> {
        let subject = self.match_value(Braces::MatchArms)?;
        self.expect("{")?;
        let mut height = subject.height;
        let mut range_spellings = BTreeSet::new();
        let arms = self.list("}", |parser| {
            let mut patterns = vec![parser.match_pattern(&mut range_spellings)?];
            while parser.eat("|") {
                patterns.push(parser.match_pattern(&mut range_spellings)?);
            }
            parser.expect("=>")?;
            let value = parser.match_value(Braces::Struct)?;
            height = height.max(value.height);
            Ok(MatchArm {
                patterns,
                value: value.value,
            })
        })?;
        self.compose(
            height,
            Expr::Match {
                subject: Box::new(subject.value),
                arms,
            },
        )
    }

    fn match_pattern(
        &mut self,
        range_spellings: &mut BTreeSet<String>,
    ) -> Result<MatchPattern, String> {
        if self.eat("_") {
            return Ok(MatchPattern::Wildcard);
        }
        let mut end = 1;
        while end < 5 && self.peek(end) == "::" && is_function_identifier(self.peek(end + 1)) {
            end += 2;
        }
        if is_function_identifier(self.peek(0)) && matches!(self.peek(end), "=>" | "|") {
            // The checker resolves a scalar constant, value parametric, or enum
            // member. DSLX requires numeric literals on both sides of a range.
            let name = self.name_or_qualified_member()?;
            return Ok(MatchPattern::Literal(self.named_value(name)?));
        }
        let first = self.pos;
        let start = self.pattern_literal()?;
        let separator = self.pos;
        let inclusive = self.eat("..=");
        if inclusive || self.eat("..") {
            let end = self.pattern_literal()?;
            // An empty interval contributes no coverage, but DSLX still
            // rejects repeated source patterns, including in one alternative.
            let spelling = format!(
                "{}{}{}",
                literal_pattern_spelling(&self.tokens[first..separator]),
                self.tokens[separator].text,
                literal_pattern_spelling(&self.tokens[separator + 1..self.pos]),
            );
            if !range_spellings.insert(spelling) {
                return Err(diagnostic(
                    self.source,
                    self.tokens[first].start,
                    "duplicate range pattern",
                ));
            }
            Ok(MatchPattern::Range {
                start,
                end,
                inclusive,
            })
        } else {
            Ok(MatchPattern::Literal(start))
        }
    }

    /// DSLX range endpoints use the same number grammar as literal patterns.
    fn pattern_literal(&mut self) -> Result<Expr, String> {
        if self.eat("true") {
            return Ok(Expr::Bool(true));
        }
        if self.eat("false") {
            return Ok(Expr::Bool(false));
        }
        let literal = if self.starts_bit_literal()? {
            let ty = self.ty()?;
            if !matches!(
                ty,
                Type::Bool | Type::Bits { .. } | Type::ParametricBits { .. }
            ) && !matches!(&ty, Type::Named(name) if self.block_type_params.contains(name.as_str()))
            {
                return Err(self.error("match patterns require a scalar literal type"));
            }
            self.expect(":")?;
            self.annotated_literal(ty)?
        } else if number_token(self.peek(0)) || self.tokens[self.pos].character.is_some() {
            self.literal(None)?
        } else {
            return Err(self.error("expected a scalar match pattern (literal, range, or _)"));
        };
        Ok(literal.value)
    }

    /// In a match subject, a named field starts a struct constructor while a
    /// scalar literal or wildcard starts the pattern list.
    fn struct_braces(&self, braces: Braces) -> bool {
        match braces {
            Braces::Struct => true,
            Braces::Body => false,
            Braces::MatchArms => {
                let first = self.peek(1);
                let next = self.peek(2);
                if first == "}" {
                    return matches!(next, "." | "[" | "as" | "{")
                        || BinaryOp::from_spelling(next).is_some();
                }
                let field = is_function_identifier(first)
                    || !self.dslx_value_grammar() && is_identifier(first);
                if next == ":" && self.block_type_params.contains(first) {
                    // Both a field initializer and an annotated integer pattern
                    // can start with `T:0`. Only the latter continues to an arm
                    // arrow before a comma or brace.
                    return self.tokens[self.pos + 3..]
                        .iter()
                        .find(|t| matches!(t.text, "=>" | "," | "{" | "}"))
                        .is_none_or(|t| t.text != "=>");
                }
                field && matches!(next, ":" | "," | "}")
            }
        }
    }

    /// BX block names predate the match keyword. An arrow in the following
    /// braces distinguishes `match (x) { ... }` from a call named `match`,
    /// and `match - x { ... }` from a subtraction in an existing guard.
    fn block_match_ahead(&self) -> bool {
        let next = self.peek(1);
        if !matches!(next, "(" | "[" | "{" | "-") {
            return next != "as"
                && (is_dslx_member_name(next)
                    || number_token(next)
                    || self.tokens[self.pos + 1].character.is_some()
                    || next == "!");
        }
        let mut depth = 0usize;
        let mut brace_offset = None;
        for (offset, token) in self.expression_lookahead(self.pos + 1) {
            match token.text {
                "(" | "[" | "{" => {
                    if depth == 0 {
                        brace_offset = (token.text == "{").then_some(offset);
                    }
                    depth += 1;
                }
                "=>" if depth == 1 && brace_offset.is_some_and(|start| start > 0) => return true,
                "," | ";" | "=>" if depth == 0 => return false,
                ")" | "]" | "}" => {
                    if depth == 0 {
                        return false;
                    }
                    depth -= 1;
                    if depth == 0 && brace_offset.is_some() {
                        let after = self.peek(offset + 2);
                        if !matches!(after, "{" | "." | "[" | "as")
                            && BinaryOp::from_spelling(after).is_none()
                        {
                            return false;
                        }
                    }
                }
                _ => { /* Other tokens can occur inside the subject or arm. */ }
            }
        }
        false
    }

    /// Both loop spellings allow an optional element and accumulator type pair.
    fn loop_annotation(&mut self) -> Result<Option<(Type, Type)>, String> {
        if !self.eat(":") {
            return Ok(None);
        }
        self.expect("(")?;
        let element = self.ty()?;
        self.expect(",")?;
        let accumulator = self.ty()?;
        self.eat(",");
        self.expect(")")?;
        Ok(Some((element, accumulator)))
    }

    /// Parse a pure counted fold in either a block or a local function.
    fn counted_for(&mut self) -> Result<ParsedExpr, String> {
        self.with_value_grammar(Self::counted_for_contents)
    }

    fn counted_for_contents(&mut self) -> Result<ParsedExpr, String> {
        self.expect("(")?;
        let mut names = BTreeSet::new();
        let element = self.binding_pattern(&mut names, self.identifier_predicate())?;
        self.expect(",")?;
        let accumulator = self.binding_pattern(&mut names, self.identifier_predicate())?;
        self.eat(",");
        self.expect(")")?;
        let annotation = self.loop_annotation()?;
        self.expect("in")?;
        let (iterable, height) = self.for_iterable(Braces::Body)?.into_ast();
        self.expect("{")?;
        let body = self.with_name_scope(|p| {
            for name in names {
                p.shadow_callable(&name);
            }
            p.value_scope()
        })?;
        self.expect("(")?;
        let seed = self.expr()?;
        self.expect(")")?;
        self.compose(
            height.max(body.height).max(seed.height),
            Expr::For(Box::new(ForLoop {
                element,
                accumulator,
                annotation,
                iterable,
                body: body.value,
                seed: seed.value,
            })),
        )
    }

    fn in_value_body(&self) -> bool {
        self.dslx_value_grammar() || self.expression_grammar != ExpressionGrammar::Block
    }

    /// Enable value blocks with pure lets without hiding BX port names.
    fn with_value_grammar<T>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let outer = std::mem::replace(&mut self.expression_grammar, ExpressionGrammar::DslxOperand);
        let result = parse(self);
        self.expression_grammar = outer;
        result
    }

    /// Parse the range operator below all binary operators. Parentheses can
    /// enclose an entire range, one endpoint, or an array-producing expression.
    fn for_iterable(&mut self, braces: Braces) -> Result<ParsedIterable, String> {
        let start = if self.eat("(") {
            let grouped = self.nested(|p| {
                p.parenthesized(
                    |p| p.for_iterable(Braces::Struct),
                    |p, fields| {
                        let items = fields
                            .into_iter()
                            .map(|field| match field {
                                ParsedIterable::Array(expr) => Ok(expr),
                                ParsedIterable::Range { .. } => {
                                    Err(p.error("expected a tuple member expression"))
                                }
                            })
                            .collect::<Result<_, _>>()?;
                        p.aggregate(items, Expr::Tuple).map(ParsedIterable::Array)
                    },
                )
            })?;
            let ParsedIterable::Array(mut value) = grouped else {
                return Ok(grouped);
            };
            value.spelling = ExprSpelling::Parenthesized;
            self.expr_tail(value, 0, braces)?
        } else {
            self.full_expr(braces)?
        };
        if !matches!(self.peek(0), ".." | "..=") {
            return Ok(ParsedIterable::Array(start));
        }
        let inclusive = self.peek(0) == "..=";
        self.pos += 1;
        let limit = self.expr_at(0, braces)?;
        Ok(ParsedIterable::Range {
            start,
            limit,
            inclusive,
        })
    }

    /// In blocks, `for` is also a legal legacy name. A parenthesized binder
    /// followed by `in` or a type annotation identifies the counted loop.
    fn starts_counted_for(&self) -> bool {
        if self.peek(0) != "for" || self.peek(1) != "(" {
            return false;
        }
        if self.dslx_value_grammar() {
            return true;
        }
        let mut depth = 1usize;
        for (i, token) in self.tokens[self.pos + 2..].iter().enumerate() {
            match token.text {
                "(" => depth += 1,
                ")" => {
                    depth -= 1;
                    if depth == 0 {
                        return matches!(self.peek(i + 3), "in" | ":");
                    }
                }
                _ => { /* Only parentheses affect the binder's closing position. */ }
            }
        }
        false
    }

    /// Parse the binders, range, iteration body, and initial accumulator of
    /// a DSLX-style unrolled fold.
    fn unroll_for(&mut self) -> Result<ParsedExpr, String> {
        self.expect("(")?;
        let index = self.ident()?;
        self.expect(",")?;
        let accumulator_offset = self.tokens[self.pos].start;
        let accumulator = self.ident()?;
        if index == accumulator && index != "_" {
            return Err(diagnostic(
                self.source,
                accumulator_offset,
                &format!("duplicate loop binding '{accumulator}'"),
            ));
        }
        self.eat(",");
        self.expect(")")?;
        let annotation = self.loop_annotation()?;
        self.expect("in")?;
        // Range endpoints are operands of '..', even at the start of the
        // iterable.
        let start = self.expr_at(0, Braces::Struct)?;
        self.expect("..")?;
        let limit = self.expr_at(0, Braces::Body)?;
        self.expect("{")?;
        let mut body = Vec::new();
        let mut height = start.height.max(limit.height);
        let result = self.with_name_scope(|p| {
            p.hide_pattern_name(&index);
            p.hide_pattern_name(&accumulator);
            if p.in_value_body() {
                return p.expression_scope();
            }
            while p.eat("let") {
                let (stmt, initializer_height) = p.let_stmt()?;
                p.expect(";")?;
                height = height.max(initializer_height);
                body.push(stmt);
            }
            let value = p.expr()?;
            p.expect("}")?;
            Ok(value)
        })?;
        self.expect("(")?;
        let seed = self.expr()?;
        self.expect(")")?;
        self.compose(
            height.max(result.height).max(seed.height),
            Expr::UnrollFor(Box::new(UnrollFor {
                index,
                accumulator,
                annotation,
                start: start.value,
                limit: limit.value,
                body,
                result: result.value,
                seed: seed.value,
            })),
        )
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
            self.expression_scope()?
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
    fn expr_at(&mut self, min: u8, braces: Braces) -> Result<ParsedExpr, String> {
        self.nested(|parser| {
            let left = parser.atom(braces)?;
            parser.expr_tail(left, min, braces)
        })
    }

    fn expr_tail(
        &mut self,
        mut left: ParsedExpr,
        min: u8,
        braces: Braces,
    ) -> Result<ParsedExpr, String> {
        loop {
            if self.dslx_value_grammar()
                && left.spelling != ExprSpelling::Parenthesized
                && matches!(left.value, Expr::If { .. })
                && matches!(self.peek(0), "." | "[")
            {
                return Err(
                    self.error("parenthesize a conditional before field access or indexing")
                );
            }
            if self.eat(".") {
                let name = if number_token(self.peek(0)) {
                    let offset = self.tokens[self.pos].start;
                    let index = u32::try_from(self.dimension()?).map_err(|_| {
                        diagnostic(self.source, offset, "tuple index must fit in u32")
                    })?;
                    index.to_string()
                } else if self.value_context == ValueContext::BlockLambda && self.eat("out") {
                    "out".to_owned()
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
                let right = self.expr_at(op.precedence() + 1, braces)?;
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

    fn atom(&mut self, braces: Braces) -> Result<ParsedExpr, String> {
        let text = self.peek(0);
        if self.dslx_value_grammar() && text == "_" {
            return Err(self.error("wildcard pattern '_' cannot be used as a reference"));
        }
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
        if text == "match" && (self.dslx_value_grammar() || self.block_match_ahead()) {
            self.pos += 1;
            return self.match_expr();
        }
        if self.eat("unroll_for!") {
            return self.unroll_for();
        }
        if self.starts_counted_for() {
            self.pos += 1;
            return self.counted_for();
        }
        if text == "zero!" && self.peek(1) == "<" && !self.zero_parametric_call() {
            self.pos += 2;
            let ty = self.ty()?;
            self.expect(">")?;
            self.expect("(")?;
            self.expect(")")?;
            return Ok(ParsedExpr::leaf(Expr::Zero(ty)));
        }
        if self.eat("true") || self.eat("false") {
            return Ok(ParsedExpr::leaf(Expr::Bool(text == "true")));
        }
        if let Some(op) = UnaryOp::from_spelling(text) {
            self.pos += 1;
            let value = self.expr_at(11, braces)?;
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
        if self.parametric_arguments_end(self.pos, &["("]).is_some() {
            self.expect("<")?;
            let mut height = 0;
            let parametrics = self.parametric_arguments(&mut height)?;
            self.expect("(")?;
            let args = self.list(")", |p| {
                let arg = p.expr()?;
                height = height.max(arg.height);
                Ok(arg.value)
            })?;
            return self.compose(
                height,
                Expr::ParametricCall {
                    name,
                    parametrics,
                    args,
                },
            );
        }
        if self.eat("(") {
            if name == "map" && self.peek(0) != ")" {
                let lambda = self.has_lambda_argument();
                if lambda || self.dslx_value_grammar() {
                    return self.map_call(name, lambda, false);
                }
                if !self.block_map_shadowed {
                    if self.forward_map_import {
                        return self.map_before_use(name);
                    }
                    return self.map_call(name, false, false);
                }
            }
            let args = self.list(")", Self::expr)?;
            return self.aggregate(args, |args| Expr::Call { name, args });
        }
        let struct_args = if self
            .struct_arguments_end(self.pos)
            .is_some_and(|end| self.tokens[end].text == ">" && self.tokens[end + 1].text == "{")
        {
            Some(self.struct_arguments()?)
        } else {
            None
        };
        if self.peek(0) == "{" && self.struct_braces(braces) {
            self.pos += 1;
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
            let value = match struct_args {
                Some(args) => Expr::AppliedStruct { name, args, fields },
                None => Expr::Struct { name, fields },
            };
            return self.compose(height, value);
        }
        if struct_args.is_some() {
            return Err(self.error("parenthesize the struct literal in this expression"));
        }
        Ok(ParsedExpr::leaf(self.named_value(name)?))
    }

    /// Read actuals after the opening angle and include their heights in the
    /// parent.
    fn parametric_arguments(&mut self, height: &mut usize) -> Result<Vec<Expr>, String> {
        self.list(">", |p| {
            let actual = p.parametric_argument()?;
            *height = (*height).max(actual.height);
            Ok(actual.value)
        })
    }

    /// DSLX accepts bare names, numbers and typed literals. Braces enclose
    /// other value expressions so their operators cannot consume the closing
    /// angle.
    fn parametric_argument(&mut self) -> Result<ParsedExpr, String> {
        if self.eat("{") {
            // A scoped expression in an explicit DSLX actual is parenthesized,
            // as in f<{({ let k = u32:1; k })}>(x).
            if self.peek(0) == "{" {
                return Err(self.error("parenthesize a scoped parametric argument"));
            }
            let expr = self.with_value_grammar(Self::expr)?;
            self.expect("}")?;
            return Ok(expr);
        }
        let text = self.peek(0);
        if self.eat("true") || self.eat("false") {
            return Ok(ParsedExpr::leaf(Expr::Bool(text == "true")));
        }
        if number_token(text) {
            return self.literal(None);
        }
        if self.starts_bit_literal()? {
            let ty = self.ty()?;
            self.expect(":")?;
            return self.annotated_literal(ty);
        }
        let name = self.name_or_qualified_member()?;
        Ok(ParsedExpr::leaf(self.named_value(name)?))
    }

    /// Find the closing angle of a call or map callback reference, skipping
    /// braced expressions, dependent dimensions and parenthesized typed
    /// literals.
    fn parametric_arguments_end(&self, angle: usize, following: &[&str]) -> Option<usize> {
        if self.tokens[angle].text != "<" || angle == 0 {
            return None;
        }
        // In `pair(x < y, z > (w))` a comma can separate two comparisons.
        // At the start of a standalone expression it can only belong to the
        // parametric list, even if a nearer value shadows the function name.
        let name = self.tokens[angle - 1].text;
        let head = if angle >= 3 && self.tokens[angle - 2].text == "::" {
            angle - 3
        } else {
            angle - 1
        };
        let standalone = head
            .checked_sub(1)
            .is_some_and(|i| matches!(self.tokens[i].text, "{" | ";" | "="));
        let allow_comma = standalone
            || match &self.tokens[..angle] {
                [.., head, separator, _] if separator.text == "::" => self
                    .callable_templates
                    .get(head.text)
                    .is_some_and(|binding| {
                        binding.members.as_ref().is_none_or(|m| m.contains(name))
                    }),
                _ => self
                    .callable_templates
                    .get(name)
                    .is_some_and(|binding| binding.direct.is_some()),
            };
        let mut depth = 0usize;
        for index in angle + 1..self.tokens.len() {
            let token = &self.tokens[index];
            match token.text {
                "" => return None,
                ">" if depth == 0 => {
                    return following
                        .contains(&self.tokens[index + 1].text)
                        .then_some(index);
                }
                "{" if depth != 0 || matches!(self.tokens[index - 1].text, "<" | ",") => depth += 1,
                "(" if depth != 0 || self.tokens[index - 1].text == ":" => depth += 1,
                "[" => depth += 1,
                ")" | "]" | "}" if depth != 0 => depth -= 1,
                _ if depth != 0 => { /* A delimited expression can contain any value operator. */ }
                "," if allow_comma => { /* The template owns this separator. */ }
                text if is_dslx_member_name(text)
                    || number_token(text)
                    || token.character.is_some()
                    || matches!(text, ":" | "::") =>
                { /* Part of an unbraced name or literal. */ }
                _ => return None,
            }
        }
        None
    }

    /// Recognize struct argument lists in expressions without consuming the
    /// comma in a pair of comparisons such as `(x < 1, 2 > y)`.
    fn struct_arguments_end(&self, angle: usize) -> Option<usize> {
        if self.tokens[angle].text != "<" {
            return None;
        }
        let count = self.tokens[angle + 1..]
            .iter()
            .take_while(|token| number_token(token.text) || matches!(token.text, "u32" | ":" | ","))
            .count();
        let end = angle + 1 + count;
        let close = self.tokens[end].text;
        if !matches!(close, ">" | ">>" | ">=") {
            return None;
        }
        let next = self.tokens[end + 1].text;
        if close == ">" && next == "{" {
            return Some(end);
        }
        // Types directly inside zero! or following 'as' can have commas at
        // expression nesting depth zero. Tuples and arrays delimit their own
        // contents in expression_lookahead.
        let (before_name, is_type_param) = match &self.tokens[..angle] {
            [before @ .., _, separator, _] if separator.text == "::" => (before, false),
            [before @ .., name] => (before, self.block_type_params.contains(name.text)),
            [] => return None,
        };
        match before_name {
            [.., zero, open] if zero.text == "zero!" && open.text == "<" => Some(end),
            [.., cast]
                if cast.text == "as"
                    && !is_type_param
                    && (matches!(
                        next,
                        "." | "[" | "as" | "," | ")" | "]" | "}" | ";" | "{" | ".." | "..="
                    ) || BinaryOp::from_spelling(next).is_some()) =>
            {
                Some(end)
            }
            _ => None,
        }
    }

    /// Skip callback and struct argument lists while looking for expression
    /// delimiters. Offsets still refer to the enclosing token stream.
    fn expression_lookahead(&self, start: usize) -> impl Iterator<Item = (usize, &Token<'s>)> {
        let mut index = start;
        std::iter::from_fn(move || {
            let token = self.tokens.get(index)?;
            let offset = index - start;
            index = self
                .parametric_arguments_end(index, &["("])
                .or_else(|| self.struct_arguments_end(index))
                .unwrap_or(index)
                + 1;
            Some((offset, token))
        })
    }

    /// Distinguish the typed zero intrinsic from a visible callback template
    /// with the same name. Without import hints, preserve the intrinsic for a
    /// single type argument and no value arguments.
    fn zero_parametric_call(&self) -> bool {
        let Some(binding) = self.callable_templates.get("zero!") else {
            return false;
        };
        let Some(start) = binding.direct else {
            return false;
        };
        if self.value_context == ValueContext::DslxItem && start > self.tokens[self.pos].start {
            return false;
        }
        if binding.members.is_some() {
            return true;
        }
        let angle = self.pos + 1;
        self.parametric_arguments_end(angle, &["("])
            .is_some_and(|end| {
                let mut depth = 0usize;
                self.tokens[end + 2].text != ")"
                    || matches!(self.tokens[angle + 1].text, "{" | "true" | "false" | ">")
                    || number_token(self.tokens[angle + 1].text)
                    || self.tokens[angle + 1..end].iter().any(|t| {
                        match t.text {
                            "[" | "(" => depth += 1,
                            "]" | ")" => depth = depth.saturating_sub(1),
                            "," | ":" if depth == 0 => return true,
                            _ => { /* A single type can have expressions inside its dimensions. */ }
                        }
                        false
                    })
            })
    }

    /// An imported callable keeps ordinary block argument syntax. An enum
    /// imported later leaves the builtin's pure operand and range syntax
    /// available. Try each interpretation once for the whole call, including
    /// nested calls in its arguments, then let the checker resolve the use.
    fn map_before_use(&mut self, target: String) -> Result<ParsedExpr, String> {
        let start = self.pos;
        // A type argument can consume the first character of >> or >= in
        // place. Save just those tokens inside this argument list so both
        // interpretations see the same input.
        let mut split_candidates = Vec::new();
        let mut depth = 0;
        for (index, &token) in self.tokens.iter().enumerate().skip(start) {
            match token.text {
                "(" => depth += 1,
                ")" if depth == 0 => break,
                ")" => depth -= 1,
                ">>" | ">=" => split_candidates.push((index, token)),
                _ => { /* Other tokens are immutable during parsing. */ }
            }
        }
        self.block_map_shadowed = true;
        let ordinary = self.list(")", Self::expr);
        self.block_map_shadowed = false;
        let ordinary_end = self.pos;
        for &(index, token) in &split_candidates {
            self.tokens[index] = token;
        }
        self.pos = start;
        self.forward_map_import = false;
        let builtin = self.map_call(target.clone(), false, ordinary.is_err());
        self.forward_map_import = true;
        match ordinary {
            Ok(args) => {
                self.pos = ordinary_end;
                for (index, token) in split_candidates {
                    self.tokens[index] = token;
                }
                let height = args.iter().map(|arg| arg.height).max().unwrap_or(0);
                let args: Vec<_> = args.into_iter().map(|arg| arg.value).collect();
                let (builtin_args, builtin_height) = match builtin {
                    Ok(ParsedExpr {
                        value: Expr::Call { args: pure, .. },
                        height,
                        ..
                    }) => {
                        if pure == args {
                            return self.compose(height - 1, Expr::Call { name: target, args });
                        }
                        (Some(pure), height - 1)
                    }
                    // A range operand cannot also parse as an ordinary
                    // argument.
                    Ok(_) => return Err(self.error("expected ordinary map arguments")),
                    Err(_) => (None, 0),
                };
                self.compose(
                    height.max(builtin_height),
                    Expr::MapBeforeUse {
                        name: target,
                        args,
                        builtin_args,
                    },
                )
            }
            Err(_) => builtin,
        }
    }

    /// Accept a range in the first operand and an inline lambda in the second.
    /// Calls with ordinary arguments retain their usual grammar and resolution.
    fn map_call(
        &mut self,
        target: String,
        lambda: bool,
        builtin_only: bool,
    ) -> Result<ParsedExpr, String> {
        let iterable = self.with_value_grammar(|p| p.for_iterable(Braces::Struct))?;
        let specialized = !lambda && self.specialized_map_callback_ahead();
        if !lambda
            && !builtin_only
            && !specialized
            && let ParsedIterable::Array(first) = iterable
        {
            let mut args = vec![first];
            if !self.eat(")") {
                self.expect(",")?;
                args.extend(self.list(")", Self::expr)?);
            }
            return self.aggregate(args, |args| Expr::Call { name: target, args });
        }
        self.expect(",")?;
        let mapped = if specialized {
            self.specialized_named_map(target, iterable)?
        } else if !lambda {
            let offset = self.tokens[self.pos].start;
            let callback = self.expr()?;
            let Expr::Name(callback) = callback.value else {
                return Err(diagnostic(
                    self.source,
                    offset,
                    "expected a named map function",
                ));
            };
            let (iterable, height) = iterable.into_ast();
            self.compose(
                height,
                Expr::NamedMap {
                    target,
                    iterable: Box::new(iterable),
                    callback,
                    parametrics: Vec::new(),
                },
            )?
        } else if self.value_context == ValueContext::Block {
            self.block_map_lambda(iterable)?
        } else {
            self.map_lambda(iterable)?
        };
        self.eat(",");
        self.expect(")")?;
        Ok(mapped)
    }

    /// Only map's second operand permits a specialized function name without
    /// a following call. An invocation here is still an ordinary value
    /// argument.
    fn specialized_map_callback_ahead(&self) -> bool {
        if self.peek(0) != "," {
            return false;
        }
        let mut offset = 1;
        while self.peek(offset) == "(" {
            offset += 1;
        }
        if self.peek(offset + 1) == "::" {
            offset += 2;
        }
        let angle = offset + 1;
        self.peek(angle) == "<"
            && self
                .parametric_arguments_end(self.pos + angle, &[",", ")"])
                .is_some()
    }

    /// Parse a grouped or bare specialization of a map callback.
    fn specialized_named_map(
        &mut self,
        target: String,
        iterable: ParsedIterable,
    ) -> Result<ParsedExpr, String> {
        if self.eat("(") {
            let map = self.nested(|p| p.specialized_named_map(target, iterable))?;
            self.expect(")")?;
            return Ok(map);
        }
        let callback = self.name_or_qualified_member()?;
        self.expect("<")?;
        let (iterable, mut height) = iterable.into_ast();
        let parametrics = self.parametric_arguments(&mut height)?;
        self.compose(
            height,
            Expr::NamedMap {
                target,
                iterable: Box::new(iterable),
                callback,
                parametrics,
            },
        )
    }

    /// The module lexer used BX rules here. Re-lex the callback once using
    /// function rules, including any nested map lambdas, then resume the BX
    /// token stream at the call's closing parenthesis or trailing comma.
    fn block_map_lambda(&mut self, iterable: ParsedIterable) -> Result<ParsedExpr, String> {
        let mut depth = 0usize;
        let end = self.tokens[self.pos..]
            .iter()
            .position(|token| match token.text {
                "(" => {
                    depth += 1;
                    false
                }
                ")" if depth > 0 => {
                    depth -= 1;
                    false
                }
                ")" | "" => true,
                _ => false,
            })
            .expect("the token stream ends with EOF")
            + self.pos;
        let callback_end = if self.tokens[end - 1].text == "," {
            end - 1
        } else {
            end
        };
        let start = self.tokens[self.pos - 1].start + 1; // Immediately after the comma.
        let span = start..self.tokens[callback_end].start;
        let tokens = lex(self.source, Dialect::Bx, Some(span))?;
        let outer_tokens = std::mem::replace(&mut self.tokens, tokens);
        self.pos = 0;
        self.value_context = ValueContext::BlockLambda;
        let mapped = self.map_lambda(iterable).and_then(|value| {
            if self.peek(0).is_empty() {
                Ok(value)
            } else {
                Err(self.error("expected end of map lambda"))
            }
        });
        self.value_context = ValueContext::Block;
        self.tokens = outer_tokens;
        self.pos = callback_end;
        mapped
    }

    /// Find the separator after the array without interpreting commas in
    /// nested calls, aggregate literals, or lexical blocks.
    fn has_lambda_argument(&self) -> bool {
        let mut tokens = self.expression_lookahead(self.pos).map(|(_, token)| token);
        let mut depth = 0;
        while let Some(token) = tokens.next() {
            match token.text {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" if depth == 0 => return false,
                ")" | "]" | "}" => depth -= 1,
                "," if depth == 0 => {
                    return tokens
                        .find(|t| t.text != "(")
                        .is_some_and(|t| matches!(t.text, "|" | "||"));
                }
                _ => { /* Other tokens leave the argument nesting unchanged. */ }
            }
        }
        false
    }

    /// Parse a unary map lambda in DSLX's pure expression grammar. An
    /// explicit return annotation requires a braced body.
    fn map_lambda(&mut self, iterable: ParsedIterable) -> Result<ParsedExpr, String> {
        if self.eat("(") {
            let mapped = self.nested(|parser| parser.map_lambda(iterable))?;
            self.expect(")")?;
            return Ok(mapped);
        }
        self.expect("|")?;
        let parameter = self.take_identifier(is_use_identifier)?;
        let parameter_type = if self.eat(":") {
            Some(self.ty()?)
        } else {
            None
        };
        self.eat(",");
        self.expect("|")?;
        let result_type = if self.eat("->") {
            Some(self.ty()?)
        } else {
            None
        };
        let body = self.with_name_scope(|p| {
            p.hide_pattern_name(&parameter);
            if p.eat("{") {
                p.value_scope()
            } else if result_type.is_some() {
                Err(p.error("lambda with an explicit result type requires a braced body"))
            } else {
                p.expr()
            }
        })?;
        let (iterable, height) = iterable.into_ast();
        self.compose(
            height.max(body.height),
            Expr::Map(Box::new(MapExpr {
                target: "map".into(),
                iterable,
                parameter,
                parameter_type,
                result_type,
                body: body.value,
            })),
        )
    }

    /// Recognize a bit literal's annotation and suggest the DSLX spelling
    /// when a formerly accepted shorthand is followed by a colon.
    fn starts_bit_literal(&self) -> Result<bool, String> {
        let name = self.peek(0);
        if matches!(name, "bits" | "uN" | "sN" | "xN" | "bool")
            || fixed_width(name).is_some()
            || (self.peek(1) == ":" && self.block_type_params.contains(name))
        {
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
        let spelling = self.peek(0);
        let (ty, text) = if self.tokens[self.pos].character.is_some()
            || matches!(spelling, "true" | "false")
        {
            let ty = ty.unwrap_or(if self.tokens[self.pos].character.is_some() {
                Type::Bits {
                    signed: false,
                    width: 8,
                }
            } else {
                Type::Bool
            });
            let text = if matches!(&ty, Type::ParametricBits { .. })
                || matches!(&ty, Type::Named(name) if self.block_type_params.contains(name.as_str()))
            {
                // Preserve the intrinsic width requirement until the call binds
                // this type.
                spelling.to_owned()
            } else {
                character_or_bool_digits(spelling, &ty).map_err(|e| self.error(e))?
            };
            (Some(ty), text)
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

/// Check the intrinsic width of a character or boolean token, then spell its
/// value as a bit pattern so signed annotations retain the same bits.
pub(crate) fn character_or_bool_digits(text: &str, ty: &Type) -> Result<String, &'static str> {
    if matches!(text, "true" | "false") {
        if !matches!(ty, Type::Bool | Type::Bits { width: 1.., .. }) {
            return Err("a boolean literal requires at least one bit");
        }
        return Ok(if text == "true" { "0b1" } else { "0b0" }.to_owned());
    }
    if !matches!(ty, Type::Bits { width: 8.., .. }) {
        return Err("a character literal requires at least eight bits");
    }
    let (byte, _) = scan_character(text)
        .filter(|(_, len)| *len == text.len())
        .ok_or("invalid byte character literal")?;
    Ok(format!("0x{byte:02x}"))
}

/// DSLX retains the outer type spelling, the numeric radix and separators,
/// and whether the literal was parenthesized. Character escapes denote the
/// same literal as the corresponding byte.
fn literal_pattern_spelling(tokens: &[Token]) -> String {
    let mut result = String::new();
    if let Some(colon) = tokens.iter().position(|t| t.text == ":") {
        for token in &tokens[..=colon] {
            result.push_str(token.text);
        }
    }
    let value = tokens
        .iter()
        .rfind(|t| t.text != ")")
        .expect("parsed literal");
    if let Some(byte) = value.character {
        result.push_str(&format!("'{byte}'"));
    } else {
        result.push_str(value.text);
    }
    if tokens.iter().any(|t| t.text == "(") {
        format!("({result})")
    } else {
        result
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

/// Prove that a checked DSLX literal range executes its loop body at least
/// once.
pub(crate) fn nonempty_literal_range(tokens: &[Token<'_>]) -> bool {
    // The DSLX outline lexer keeps each dot and the optional '=' separate.
    let Some(separator) = tokens
        .windows(2)
        .position(|t| t[0].text == "." && t[1].text == ".")
    else {
        return false;
    };
    let upper = &tokens[separator + 2..];
    let inclusive = upper.first().is_some_and(|t| t.text == "=");
    let (Some(lower), Some(upper)) = (
        range_literal(&tokens[..separator]),
        range_literal(&upper[usize::from(inclusive)..]),
    ) else {
        return false;
    };
    let annotation = lower.annotation.or(upper.annotation);
    if annotation.is_none()
        && !(is_canonical_decimal(lower.spelling) && is_canonical_decimal(upper.spelling))
    {
        // A surrounding binder can supply a signed width for prefixed or
        // negated literals. Positive decimal literals must fit either way.
        return false;
    }
    // Each digit needs at most four bits. Once both magnitudes and a sign bit
    // fit, additional leading bits cannot affect the ordering.
    let enough_bits = lower
        .spelling
        .len()
        .max(upper.spelling.len())
        .saturating_mul(4)
        .saturating_add(1);
    let (signed, width) = annotation.unwrap_or((false, enough_bits));
    let width = width.min(enough_bits);
    let (Some(lower), Some(upper)) = (lower.bits(width), upper.bits(width)) else {
        return false;
    };
    if lower.equals(&upper) {
        inclusive
    } else if signed {
        lower.slt(&upper)
    } else {
        lower.ult(&upper)
    }
}

/// Numeric spelling and optional concrete (signedness, width) of a range bound.
struct RangeLiteral<'s> {
    spelling: &'s str,
    annotation: Option<(bool, usize)>,
}

/// Recognize a bare or bit-annotated literal, allowing checked grouping
/// parentheses.
fn range_literal<'s>(tokens: &[Token<'s>]) -> Option<RangeLiteral<'s>> {
    // Parentheses can group the whole range, either bound, or an annotated
    // literal. The typechecker has already checked their placement.
    let words: Vec<_> = tokens
        .iter()
        .map(|t| t.text)
        .filter(|t| !matches!(*t, "(" | ")"))
        .take(7)
        .collect();
    let (&spelling, prefix) = words.split_last()?;
    let annotation = match prefix {
        [] => None,
        [ty, ":"] => Some((ty.starts_with('s'), fixed_width(ty)?)),
        [ty @ ("bits" | "uN" | "sN"), "[", width, "]", ":"] => {
            let width = normalized_integer(width)?;
            let (digits, radix) = radix_digits(&width);
            Some((*ty == "sN", usize::from_str_radix(digits, radix).ok()?))
        }
        _ => return None,
    };
    Some(RangeLiteral {
        spelling,
        annotation,
    })
}

impl RangeLiteral<'_> {
    /// DSLX negates a width-checked literal magnitude, so s3:-7 represents 1.
    fn bits(&self, width: usize) -> Option<IrBits> {
        let text = normalized_integer(self.spelling)?;
        let digits = text.strip_prefix('-').unwrap_or(&text);
        let bits = IrValue::parse_typed(&format!("bits[{width}]:{digits}"))
            .ok()?
            .to_bits()
            .ok()?;
        Some(if text.starts_with('-') {
            bits.negate()
        } else {
            bits
        })
    }
}

/// Recognize DSLX expressions whose inputs are all literals or known constants.
/// The native typechecker has already accepted the syntax.
pub(crate) fn uses_only_const_assert_operands(expr: &str, constant_names: &[String]) -> bool {
    let Ok(tokens) = lex(expr, Dialect::Dslx, None) else {
        return false;
    };
    let operands = &tokens[..tokens.len() - 1]; // Exclude the EOF token.
    !operands.is_empty()
        && operands.iter().all(|token| {
            let word = token.text;
            number_token(word)
                || fixed_width(word).is_some()
                || BinaryOp::from_spelling(word).is_some()
                || matches!(
                    word,
                    "true"
                        | "false"
                        | "as"
                        | "bool"
                        | "bits"
                        | "uN"
                        | "sN"
                        | "!"
                        | ":"
                        | "("
                        | ")"
                        | "["
                        | "]"
                )
                || constant_names.iter().any(|name| name == word)
        })
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

    fn names<const N: usize>(names: [&str; N]) -> [Expr; N] {
        names.map(|name| Expr::Name(name.to_owned()))
    }

    #[test]
    fn numeric_enum_members_retain_expressions_and_module_qualified_references()
    -> Result<(), String> {
        let module = parse_module(
            r#"
const BASE = u8:2;
pub enum Phase' : u8 { First = BASE, Next! = BASE + u8:1, }
enum Inferred { Zero = 0, Two = 2 }
fn from_library(x: lib::Phase') -> lib::Phase' {
    match x { lib::Phase'::First | lib::Phase'::Next! => x, _ => lib::Phase'::First }
}
"#,
        )?;
        assert_eq!(
            module.declaration_order,
            ["BASE", "Phase'", "Inferred", "from_library"]
        );
        let def = &module.enums[0];
        assert!(def.public);
        assert_eq!(
            def.underlying,
            Some(Type::Bits {
                signed: false,
                width: 8
            })
        );
        assert_eq!(def.members[0].value, Expr::Name("BASE".into()));
        assert!(matches!(
            &def.members[1].value,
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!(module.enums[1].underlying.is_none());
        let Expr::Match { arms, .. } = &module.functions[0].body else {
            panic!("expected a match")
        };
        assert_eq!(
            arms[0].patterns[1],
            MatchPattern::Literal(Expr::Name("lib::Phase'::Next!".into()))
        );
        assert_eq!(arms[1].value, Expr::Name("lib::Phase'::First".into()));
        for source in [
            "enum E { A }",
            "enum E { A(u8) }",
            "enum E { }",
            "enum E:u8 { A=0 B=1 }",
            "enum E:u8 { A = /* comment */ 0 }",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
        Ok(())
    }

    #[test]
    fn block_warning_attributes_and_name_positions_are_per_declaration() -> Result<(), String> {
        let source = "/* µ */ block first(clk) -> u8 { u8:1 }\n\
                      #[allow(combinational_block)]\n  pub block second(clk) -> u8 { u8:2 } block third(clk) -> u8 { u8:3 }";
        let module = parse_module(source)?;
        let [first, second, third] = module.blocks.as_slice() else {
            panic!("expected three blocks");
        };
        assert_eq!(
            first.name_position,
            Some(SourcePosition {
                line: 1,
                column: 15
            })
        );
        assert_eq!(
            second.name_position,
            Some(SourcePosition {
                line: 3,
                column: 13
            })
        );
        assert_eq!(
            third.name_position,
            Some(SourcePosition {
                line: 3,
                column: 46
            })
        );
        assert!(first.allowed_warnings.is_empty());
        assert!(second.public);
        assert_eq!(
            second.allowed_warnings,
            BTreeSet::from([BxWarningKind::CombinationalBlock])
        );
        assert!(third.allowed_warnings.is_empty());
        Ok(())
    }

    #[test]
    fn warning_attributes_reject_unknown_names_and_misplaced_allowances() {
        for (source, diagnostic) in [
            (
                "#[allow(typo)] block b(clk) -> u8 { u8:0 }",
                "unknown BX warning in #[allow(...)], found 'typo'",
            ),
            (
                "#[allow()] block b(clk) -> u8 { u8:0 }",
                "expected a BX warning name",
            ),
            (
                "#[allow(combinational_block)] pub fn f() -> u8 { u8:0 }",
                "expected a block definition after #[allow(...)]",
            ),
            (
                "pub #[allow(combinational_block)] block b(clk) -> u8 { u8:0 }",
                "place a BX warning attribute before the block definition",
            ),
            (
                "block b(clk) -> u8 { #[allow(combinational_block)] u8:0 }",
                "place a BX warning attribute before the block definition",
            ),
        ] {
            let error = parse_module(source).unwrap_err();
            assert!(error.contains(diagnostic), "{error}");
        }
        let dslx = "pub #[allow(combinational_block)] fn f() {}";
        assert_eq!(
            crate::parse_dslx_outline(dslx).unwrap_err(),
            "1:5: expected a DSLX module declaration, found '#'"
        );
    }

    #[test]
    fn module_constants_retain_types_visibility_names_and_declaration_order() -> Result<(), String>
    {
        let module = parse_module(
            r#"
const base!=u8:3;
fn read() -> u8 { base! }
pub const Pair': (u8, bool) = { let n!=read(); (n!, n! != base!) };
block top(clk, x:u8) -> bool { /* BX */ x!=Pair'.0 }
"#,
        )?;
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        assert_eq!(module.declaration_order, ["base!", "read", "Pair'", "top"]);
        assert_eq!(
            module.constants[0],
            ConstDef {
                name: "base!".into(),
                public: false,
                ty: None,
                value: Expr::Literal {
                    ty: Some(byte.clone()),
                    text: "3".into()
                },
            }
        );
        let pair = &module.constants[1];
        assert_eq!(pair.name, "Pair'");
        assert!(pair.public);
        assert_eq!(pair.ty, Some(Type::Tuple(vec![byte, Type::Bool])));
        assert!(matches!(&pair.value, Expr::Scope { bindings, .. } if bindings.len() == 1));
        assert_eq!(module.functions[0].body, Expr::Name("base!".into()));
        assert!(matches!(
            &module.blocks[0].output,
            Expr::Binary {
                op: BinaryOp::Ne,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn constant_initializers_keep_dslx_lexical_rules_through_the_semicolon() -> Result<(), String> {
        parse_module(
            "const a = if true { u8:0 } else { u8:1 }; block const(clk, const:u8) -> bool { /* BX */ const!=u8:0 }",
        )?;
        for source in [
            "const (x, y) = (u8:0, u8:1);",
            "pub const fn = u8:1;",
            "const x: = u8:0;",
            "const x = ;",
            "const x = u8:0",
            "const x = if true { u8:0 } else { /* DSLX */ u8:1 };",
            "const x = (if true { u8:0 } else { u8:1 }) + { let a=u8:1; a!=u8:0 };",
            "const x = { reg r: u8; r };",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
        Ok(())
    }

    #[test]
    fn nonempty_ranges_require_ordered_literal_endpoints() -> Result<(), String> {
        for (source, expected) in [
            ("u32:0..u32:1", true),
            ("u32:0..u32:0", false),
            ("(u32:1)..(u32:0)", false),
            ("((u8:(1))..=(u8:1))", true),
            ("u8:255..=u8:255", true),
            ("0..1", true),
            ("0..0x1", false),
            ("s8:0x_ff..0b0", true),
            ("s8:-128..s8:0x80", false),
            ("s3:-7..s3:1", false),
            ("uN[1000000000]:0..uN[1000000000]:1", true),
            (
                "uN[96]:0x10000000000000000..uN[96]:0x10000000000000001",
                true,
            ),
            ("LIMIT..u32:1", false),
            ("u32:0..(u32:1+u32:2)", false),
            ("[u32:1]", false),
            ("", false),
        ] {
            let tokens = lex(source, Dialect::Dslx, None)?;
            assert_eq!(
                nonempty_literal_range(&tokens[..tokens.len() - 1]),
                expected,
                "{source}"
            );
        }
        Ok(())
    }

    #[test]
    fn callback_signatures_and_named_actuals_are_separate_from_data_arguments() -> Result<(), String>
    {
        let module = parse_module(
            r#"
block top(clk, x: u8[2]) -> u8 {
    transform<ops::step, ready, sink,>(x)[u32:0]
}
fn transform<F: fn(u8[2], (bool, u8)) -> u8[2], G: fn() -> bool, H: fn(u8),>(x: u8[2]) -> u8[2] {
    F(x, (G(), u8:1))
}
"#,
        )?;
        let formals = &module.functions[0].callback_params;
        assert_eq!(formals.len(), 3);
        assert_eq!(formals[0].name, "F");
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        let array = Type::Array {
            element: Box::new(byte.clone()),
            size: 2,
        };
        assert_eq!(
            formals[0].params,
            [array.clone(), Type::Tuple(vec![Type::Bool, byte])]
        );
        assert_eq!(formals[0].result, array);
        assert_eq!(formals[1].name, "G");
        assert!(formals[1].params.is_empty());
        assert_eq!(formals[1].result, Type::Bool);
        assert_eq!(formals[2].result, Type::Tuple(Vec::new()));
        let Expr::Index { value, .. } = &module.blocks[0].output else {
            panic!("expected an index into the specialized result");
        };
        let Expr::ParametricCall {
            name,
            parametrics,
            args,
        } = value.as_ref()
        else {
            panic!("expected a callback-parametric call");
        };
        assert_eq!(name, "transform");
        assert_eq!(parametrics, &names(["ops::step", "ready", "sink"]));
        assert_eq!(args, &[Expr::Name("x".into())]);
        assert!(
            matches!(&module.functions[0].body, Expr::Call { name, args } if name == "F" && args.len() == 2)
        );
        Ok(())
    }

    #[test]
    fn callback_arguments_preserve_comparisons_and_visible_zero_intrinsics() -> Result<(), String> {
        let module = parse_module(
            r#"
fn before() -> u8 { zero!<u8>() }
fn zero!<F: fn() -> u8>() -> u8 { F() }
fn get() -> u8 { u8:7 }
fn after() -> u8 { zero!<get>() }
fn cmp(x:u8, y:u8, z:u8, w:u8) -> (bool, bool) { (x < y, z > (w)) }
fn cmp_call(x:u8) -> bool { x < zero!<get>() }
"#,
        )?;
        assert!(matches!(&module.functions[0].body, Expr::Zero(_)));
        assert!(
            matches!(&module.functions[3].body, Expr::ParametricCall { name, parametrics, args } if name == "zero!" && parametrics == &names(["get"]) && args.is_empty())
        );
        let Expr::Tuple(comparisons) = &module.functions[4].body else {
            panic!("expected a pair of comparisons");
        };
        assert!(matches!(
            &comparisons[..],
            [
                Expr::Binary {
                    op: BinaryOp::Lt,
                    ..
                },
                Expr::Binary {
                    op: BinaryOp::Gt,
                    ..
                }
            ]
        ));
        assert!(
            matches!(&module.functions[5].body, Expr::Binary { op: BinaryOp::Lt, right, .. } if matches!(right.as_ref(), Expr::ParametricCall { .. }))
        );
        Ok(())
    }

    #[test]
    fn value_parametrics_retain_order_defaults_and_dependent_types() -> Result<(), String> {
        let module = parse_module(
            r#"
fn transform<N:u32, F:fn(uN[N])->sN[N], S:bool={false}, M:u32={N + u32:1}>
    (xs: (bits[N], xN[S][M])[N + M]) -> xN[S][M] {
    let init: sN[M] = sN[M]:-1;
    init as xN[S][M]
}
fn concrete(x:uN[8][4])->sN[8] { x[u32:0] as s8 }
"#,
        )?;
        let f = &module.functions[0];
        assert_eq!(
            f.value_parametrics
                .iter()
                .map(|p| p.index)
                .collect::<Vec<_>>(),
            [0, 2, 3]
        );
        assert_eq!(
            f.parametric_bindings()
                .map(|p| match p {
                    crate::ast::FunctionParametricRef::Value(v) => v.name.as_str(),
                    crate::ast::FunctionParametricRef::Callback(c) => c.name.as_str(),
                })
                .collect::<Vec<_>>(),
            ["N", "F", "S", "M"]
        );
        assert_eq!(f.value_parametrics[1].default, Some(Expr::Bool(false)));
        assert!(matches!(
            f.value_parametrics[2].default,
            Some(Expr::Binary {
                op: BinaryOp::Add,
                ..
            })
        ));
        let Type::ParametricArray { element, size } = &f.params[0].ty else {
            panic!("expected a dependent array dimension");
        };
        assert!(matches!(
            size.as_ref(),
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!(matches!(element.as_ref(), Type::Tuple(fields) if fields[1] == f.result));
        assert!(matches!(&f.result, Type::ParametricBits { signed, width }
            if signed.as_ref() == &Expr::Name("S".into()) && width.as_ref() == &Expr::Name("M".into())));
        let Expr::Scope { bindings, result } = &f.body else {
            panic!("expected a binding for the dependent literal");
        };
        assert!(
            matches!(&bindings[0].value, Expr::Literal { ty, text } if ty == &bindings[0].ty && text == "-1")
        );
        assert!(matches!(result.as_ref(), Expr::Cast { ty, .. } if *ty == f.result));
        assert!(matches!(
            module.functions[1].params[0].ty,
            Type::Array { size: 4, .. }
        ));
        let self_binding = parse_module("fn f<self:u32>(x:uN[self])->uN[self]{x}")?;
        assert_eq!(self_binding.functions[0].value_parametrics[0].name, "self");
        assert!(parse_module("fn f<N:u32={8}>(x:u8)->u8 { /* still DSLX */ x }").is_err());
        assert!(
            parse_module("fn f<N:u32>(x:uN[N])->uN[N] {x} block bad(clk,x:uN[N])->u8{x}").is_err()
        );
        Ok(())
    }

    #[test]
    fn explicit_parametric_arguments_keep_expressions_and_nested_delimiters() -> Result<(), String>
    {
        let module = parse_module(
            r#"
use lib;
fn f<N:u32>(x:uN[N])->uN[N] { x }
fn g(x:u8)->u8 {
    lib::apply<{if true { f<8,>(x) } else { x }}, u32:8, true, lib::Enum::Member,>(f<>(x))
}
"#,
        )?;
        let Expr::ParametricCall {
            parametrics, args, ..
        } = &module.functions[1].body
        else {
            panic!("expected explicit parametric arguments");
        };
        assert!(matches!(parametrics.as_slice(), [
            Expr::If { then_value, .. }, Expr::Literal { ty: Some(Type::Bits { width: 32, .. }), text },
            Expr::Bool(true), Expr::Name(name),
        ] if matches!(then_value.as_ref(), Expr::ParametricCall { parametrics, .. } if parametrics.len() == 1)
          && text == "8" && name == "lib::Enum::Member"));
        assert!(
            matches!(args.as_slice(), [Expr::ParametricCall { parametrics, .. }] if parametrics.is_empty())
        );
        Ok(())
    }

    #[test]
    fn concrete_struct_arguments_parse_in_aggregate_types_and_constructors() -> Result<(), String> {
        let module = parse_module(
            r#"
use packets::Packet;
fn build(x: u8) -> packets::Packet<u32:0x_f_f, 0b1_0> {
    packets::Packet<255, u32:2> { x }
}
block stage<F: fn(u8) -> Packet<2>>(clk, xs: (Packet<2>, bool)[3]) -> Packet<2> {
    let empty:Packet<2>=zero!<Packet<2>>();
    if (Packet<2> { valid: true }).valid { empty } else { xs[u32:0].0 }
}
"#,
        )?;
        let qualified = Type::Applied {
            name: "packets::Packet".into(),
            args: vec![255, 2],
        };
        let packet = Type::Applied {
            name: "Packet".into(),
            args: vec![2],
        };
        let build = &module.functions[0];
        assert_eq!(build.result, qualified);
        assert_eq!(
            build.body,
            Expr::AppliedStruct {
                name: "packets::Packet".into(),
                args: vec![255, 2],
                fields: vec![("x".into(), Expr::Name("x".into()))],
            }
        );
        let stage = &module.blocks[0];
        assert_eq!(stage.callback_params[0].result, packet);
        assert_eq!(stage.result, packet);
        assert_eq!(
            stage.params[0].ty,
            Type::Array {
                element: Box::new(Type::Tuple(vec![packet.clone(), Type::Bool])),
                size: 3,
            }
        );
        assert!(
            matches!(&stage.statements[..], [Stmt::Let { ty: Some(ty), value: Expr::Zero(zero_ty), .. }] if ty == &packet && zero_ty == &packet)
        );
        let Expr::If { condition, .. } = &stage.output else {
            panic!("expected conditional block output");
        };
        assert!(
            matches!(condition.as_ref(), Expr::Field { value, .. } if matches!(value.as_ref(), Expr::AppliedStruct { .. }))
        );
        Ok(())
    }

    #[test]
    fn struct_arguments_preserve_expression_operators_and_delimiters() -> Result<(), String> {
        let module = parse_module(
            r#"
block compare<T>(clk, x: T, y: T) -> (bool, bool) {
    (x as T < 1, 2 > (y >> u32:1))
}
block choose(clk, xs: u8[2]) -> u8[2] {
    let ys = map(Packet<2,3> { xs }.xs, |v| v);
    let zs = map(zero!<Packet<u32:2,3>>().xs, |v| v);
    match (u8:1) + Packet<2,3> { tag: u8:0 }.tag { 0 => ys, _ => zs }
}
"#,
        )?;
        let Expr::Tuple(pair) = &module.blocks[0].output else {
            panic!("expected a pair of comparisons");
        };
        assert!(
            matches!(&pair[0], Expr::Binary { op: BinaryOp::Lt, left, .. } if matches!(left.as_ref(), Expr::Cast { ty: Type::Named(name), .. } if name == "T"))
        );
        assert!(
            matches!(&pair[1], Expr::Binary { op: BinaryOp::Gt, right, .. } if matches!(right.as_ref(), Expr::Binary { op: BinaryOp::Shr, .. }))
        );
        let choose = &module.blocks[1];
        assert!(matches!(
            &choose.statements[..],
            [
                Stmt::Let {
                    value: Expr::Map(_),
                    ..
                },
                Stmt::Let {
                    value: Expr::Map(_),
                    ..
                }
            ]
        ));
        assert!(matches!(&choose.output, Expr::Match { .. }));
        Ok(())
    }

    #[test]
    fn concrete_struct_lookahead_preserves_separate_comparisons() -> Result<(), String> {
        let module = parse_module(
            r#"
use packets::Packet;
fn compare(Packet:u32, y:u32) -> (bool,bool) {
    (Packet < u32:1, u32:2 > (y))
}
fn both(a:bool, b:bool) -> bool { a && b }
fn call(x:u32, y:u32) -> bool {
    let Packet = x;
    both(Packet < 1, 2 > (y))
}
extern fn match(x:u32) -> u32;
block classify(clk, x:u32, y:u32) -> (bool,bool) {
    (match(x) < 2, 3 > match y { _ => u32:1 })
}
"#,
        )?;
        let Expr::Tuple(pair) = &module.functions[0].body else {
            panic!("expected a pair of comparisons");
        };
        let Expr::Scope { result, .. } = &module.functions[2].body else {
            panic!("expected a shadowing binding");
        };
        let Expr::Call { args, .. } = result.as_ref() else {
            panic!("expected a call with two comparisons");
        };
        let Expr::Tuple(dispatch) = &module.blocks[0].output else {
            panic!("expected the legacy call and a match in separate comparisons");
        };
        for values in [pair, args, dispatch] {
            assert!(matches!(
                &values[..],
                [
                    Expr::Binary {
                        op: BinaryOp::Lt,
                        ..
                    },
                    Expr::Binary {
                        op: BinaryOp::Gt,
                        ..
                    }
                ]
            ));
        }
        Ok(())
    }

    #[test]
    fn struct_applications_require_concrete_u32_literals() {
        for arguments in [
            "N",
            "u8:2",
            "-1",
            "u32:-1",
            "4294967296",
            "0x1_0000_0000",
            "0b2",
            "00",
            "1+2",
            "u32",
            "2,,3",
        ] {
            for source in [
                format!("fn f(x: Packet<{arguments}>) {{ }}"),
                format!("fn f() {{ Packet<{arguments}> {{ x: u8:0 }} }}"),
            ] {
                assert!(parse_module(&source).is_err(), "{source}");
            }
        }
        assert!(parse_module("fn f(x: Packet<u32:0xffff_ffff>) { }").is_ok());
        assert!(
            parse_module("fn f() -> bool { if Packet<2> { v: true }.v { true } else { false } }")
                .is_err()
        );
    }

    #[test]
    fn empty_struct_arguments_preserve_the_request_for_dslx_defaults() -> Result<(), String> {
        let module = parse_module("fn f() -> Packet<> { Packet<> { x: zero!<Tag<>>() } }")?;
        assert_eq!(
            module.functions[0].result,
            Type::Applied {
                name: "Packet".into(),
                args: vec![]
            }
        );
        assert_eq!(
            module.functions[0].body,
            Expr::AppliedStruct {
                name: "Packet".into(),
                args: vec![],
                fields: vec![(
                    "x".into(),
                    Expr::Zero(Type::Applied {
                        name: "Tag".into(),
                        args: vec![]
                    })
                )],
            }
        );
        Ok(())
    }

    #[test]
    fn callback_separators_respect_lexical_shadowing_and_enclosing_expressions()
    -> Result<(), String> {
        let module = parse_module(
            r#"
fn compose<F:fn(u8)->u8, G:fn(u8)->u8>(x:u8)->u8 { F(G(x)) }
fn pair(compose:u8, x:u8, y:u8, z:u8)->(bool, bool) { (compose < x, y > (z)) }
fn id(x:u8)->u8 { x }
fn shadow(x:u8, y:u8, z:u8)->u8 {
    let (_, _) = { let compose = x; (compose < x, y > (z)) };
    compose<id,id>(x)
}
fn compose_arrays<F:fn(u8[2])->u8[2], G:fn(u8[2])->u8[2]>(xs:u8[2])->u8[2] { F(G(xs)) }
fn array_id(xs:u8[2])->u8[2] { xs }
block wrapped(clk, xs:u8[2])->u8[2] {
    let words = map(compose_arrays<array_id,array_id>(xs), |v| v);
    match (u8:0) + compose<id,id>(xs[u32:0]) { u8:0 => words, _ => xs }
}
"#,
        )?;
        let Expr::Tuple(pair) = &module.functions[1].body else {
            panic!("expected tuple with two comparisons");
        };
        assert!(matches!(
            &pair[..],
            [
                Expr::Binary {
                    op: BinaryOp::Lt,
                    ..
                },
                Expr::Binary {
                    op: BinaryOp::Gt,
                    ..
                }
            ]
        ));
        let Expr::Scope { bindings, result } = &module.functions[3].body else {
            panic!("expected nested binding");
        };
        assert!(
            matches!(&bindings[0].value, Expr::Scope { result, .. } if matches!(result.as_ref(), Expr::Tuple(pair) if pair.len() == 2))
        );
        assert!(
            matches!(result.as_ref(), Expr::ParametricCall { parametrics, .. } if parametrics == &names(["id", "id"]))
        );
        assert!(
            matches!(&module.blocks[0].statements[0], Stmt::Let { value: Expr::Map(map), .. } if matches!(&map.iterable, ForIterable::Array(Expr::ParametricCall { .. })))
        );
        assert!(matches!(&module.blocks[0].output, Expr::Match { .. }));
        Ok(())
    }

    #[test]
    fn imported_template_hints_disambiguate_callback_lists_and_constant_comparisons()
    -> Result<(), String> {
        let source = r#"
extern fn hardware::use(x:u8) -> u8;
use lib;
use other::{apply, K};
fn invoke(x: u8) -> u8 { lib::compose<left,right>(apply<left,right>(x)) }
fn compare(a:u8, b:u8, x:u8) -> (bool,bool) { (lib::K<a,b>(x)) }
fn compare_member(a:u8, b:u8, x:u8) -> (bool,bool) { (K < a, b > (x)) }
fn shadow(apply:u8, a:u8, b:u8, x:u8) -> (bool,bool) { (apply < a, b > (x)) }
"#;
        let templates = BTreeSet::from(["lib::compose".to_string(), "apply".to_string()]);
        let module = parse_with_templates(source, &templates)?;
        assert_eq!(module_uses(source)?, module.uses);
        let Expr::ParametricCall { name, args, .. } = &module.functions[0].body else {
            panic!("expected qualified template call");
        };
        assert_eq!(name, "lib::compose");
        assert!(matches!(&args[..], [Expr::ParametricCall { name, .. }] if name == "apply"));
        for function in &module.functions[1..] {
            assert!(matches!(
                &function.body,
                Expr::Tuple(fields) if matches!(&fields[..], [
                    Expr::Binary { op: BinaryOp::Lt, .. },
                    Expr::Binary { op: BinaryOp::Gt, .. },
                ])
            ));
        }
        // Syntax-only parsing can recognize an imported call before loading
        // its file, including when the use is declared after a block.
        let provisional = parse_module("block top(clk, x:u8) -> u8 { lib::f<a,b>(x) } use lib;")?;
        assert!(
            matches!(&provisional.blocks[0].output, Expr::ParametricCall { name, .. } if name == "lib::f")
        );
        Ok(())
    }

    #[test]
    fn syntax_only_imported_zero_preserves_the_intrinsic_and_unambiguous_calls()
    -> Result<(), String> {
        let source = r#"
use wire::zero!;
block probe(clk) -> u8 { zero!<u8>() }
fn named() -> Tag { zero!<Tag>() }
fn pair() -> u8 { zero!<left,right>() }
fn apply(x: u8) -> u8 { zero!<left>(x) }
"#;
        let module = parse_module(source)?;
        assert!(matches!(&module.blocks[0].output, Expr::Zero(_)));
        assert_eq!(
            module.functions[0].body,
            Expr::Zero(Type::Named("Tag".into()))
        );
        assert!(matches!(
            &module.functions[1].body,
            Expr::ParametricCall { parametrics, .. } if parametrics == &names(["left", "right"])
        ));
        assert!(matches!(
            &module.functions[2].body,
            Expr::ParametricCall { parametrics, args, .. } if parametrics == &names(["left"]) && args.len() == 1
        ));
        let ambiguous = "use wire::zero!; fn f() -> u8 { zero!<Tag>() }";
        let templates = BTreeSet::from(["zero!".to_string()]);
        let hinted = parse_with_templates(ambiguous, &templates)?;
        assert!(matches!(
            &hinted.functions[0].body,
            Expr::ParametricCall { parametrics, .. } if parametrics == &names(["Tag"])
        ));
        Ok(())
    }

    #[test]
    fn parametrics_require_well_formed_bindings_and_braced_computed_actuals() {
        for declaration in [
            "fn f<>(x:u8)->u8 {x}",
            "fn f<N:u32=u32:8>(x:u8)->u8 {x}",
            "fn f<F: fn(x:u8) -> u8>(x:u8)->u8 {x}",
            "fn f<F: fn(u8)->fn(u8)->u8>(x:u8)->u8 {x}",
            "fn f(self: fn(u8)->u8)->u8 {u8:0}",
            "extern block f<F: fn(u8)->u8>(clk,x:u8)->u8;",
        ] {
            assert!(parse_module(declaration).is_err(), "{declaration}");
        }
        for call in [
            "f<|v| v>(x)",
            "f<(helper)>(x)",
            "f<helper(x)>(x)",
            "f<if x { a } else { b }>(x)",
            "f<u8>(x)",
            "f<{let a = u32:1; a}>(x)",
            "f<{{let a = u32:1; a}}>(x)",
            "f<,helper>(x)",
            "f<helper,,>(x)",
        ] {
            let source = format!(
                "fn f<F: fn(u8)->u8>(x:u8)->u8 {{F(x)}} block top(clk,x:u8)->u8 {{ {call} }}"
            );
            assert!(parse_module(&source).is_err(), "{call}");
        }
    }

    #[test]
    fn scalar_match_preserves_patterns_arm_order_and_postfix_syntax() {
        let src = r#"
fn f(x: s8) -> s8 {
    match x {
        -2..s8:0 | s8:1 => (s8:3,),
        0x04..=s8:6 => { let v = x; (v,) },
        _ => (x,),
    }.0
}
block b(clk, x: s8) -> s8 { match (x) { _ => { let y = x; y } } }
"#;
        let module = parse_module(src).unwrap();
        let Expr::Field { value, name } = &module.functions[0].body else {
            panic!("expected projection after match");
        };
        assert_eq!(name, "0");
        let Expr::Match { subject, arms } = value.as_ref() else {
            panic!("expected match expression");
        };
        assert_eq!(subject.as_ref(), &Expr::Name("x".into()));
        assert!(matches!(
            &arms[0].patterns[..],
            [
                MatchPattern::Range {
                    inclusive: false,
                    ..
                },
                MatchPattern::Literal(_)
            ]
        ));
        assert!(matches!(
            &arms[1].patterns[..],
            [MatchPattern::Range {
                inclusive: true,
                ..
            }]
        ));
        assert_eq!(arms[2].patterns, [MatchPattern::Wildcard]);
        assert!(matches!(arms[1].value, Expr::Scope { .. }));
        assert!(matches!(module.blocks[0].output, Expr::Match { .. }));
    }

    #[test]
    fn match_is_contextual_in_existing_bx_block_expressions() {
        let src = r#"
extern fn match(x: u8) -> u8;
block existing(clk, match: u8) -> u8 {
    let n = match(match);
    let m = match as u8;
    if match == n { n } else { m }
}
struct Data { match: u8 }
block dispatch(clk, match: u8) -> u8 {
    match Data { match: match }.match {
        0 => match match { 1 => u8:1, _ => u8:2 },
        _ => match (match + u8:1) { _ => match },
    }
}
"#;
        let module = parse_module(src).unwrap();
        let Stmt::Let { value, .. } = &module.blocks[0].statements[0] else {
            panic!("expected binding");
        };
        assert!(
            matches!(value, Expr::Call { name, args } if name == "match" && args == &[Expr::Name("match".into())])
        );
        let Expr::Match { subject, arms } = &module.blocks[1].output else {
            panic!("expected scalar match");
        };
        assert!(
            matches!(subject.as_ref(), Expr::Field { value, .. } if matches!(value.as_ref(), Expr::Struct { .. }))
        );
        assert!(
            matches!(&arms[0].value, Expr::Match { subject, .. } if subject.as_ref() == &Expr::Name("match".into()))
        );
        assert!(matches!(arms[1].value, Expr::Match { .. }));
    }

    #[test]
    fn match_patterns_require_scalar_literals_and_punctuated_arms() {
        for arms in [
            "0 => x 1 => x",
            "0 -> x",
            "0 = > x",
            "_ | => x",
            "0 .. = 1 => x",
            "0 .. x => x",
            "- 1 => x",
            "(0,) => x",
            "[0] => x",
            "NAME .. u8:1 => x",
            "u8[1]:0 => x",
        ] {
            let src = format!("fn f(x:u8) -> u8 {{ match x {{ {arms} }} }}");
            assert!(parse_module(&src).is_err(), "{src}");
        }
    }

    #[test]
    fn scalar_constant_names_are_preserved_in_match_alternatives() -> Result<(), String> {
        let module = parse_module(
            "const Lo=u8:1; const Hi!=u8:2; fn f(x:u8) -> bool { match x { Lo | Hi! => true, _ => false } }",
        )?;
        let Expr::Match { arms, .. } = &module.functions[0].body else {
            panic!("expected match");
        };
        assert_eq!(
            arms[0].patterns,
            [
                MatchPattern::Literal(Expr::Name("Lo".into())),
                MatchPattern::Literal(Expr::Name("Hi!".into())),
            ]
        );
        Ok(())
    }

    #[test]
    fn repeated_range_spelling_is_checked_before_literal_normalization() {
        for (left, right, duplicate) in [
            ("u8:3", "uN[8]:3", false),
            ("0b1_1", "0b11", false),
            ("u8:(3)", "u8:3", false),
            ("u8:(3)", "u8:((3))", true),
            ("u8:u4:3", "u8:3", true),
            ("'A'", "'\\x41'", true),
        ] {
            let src = format!(
                "fn f(x:u8) -> u8 {{ match x {{ {left}..{left} => x, {right}..{right} => x, _ => x }} }}"
            );
            assert_eq!(parse_module(&src).is_err(), duplicate, "{src}");
        }
    }

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
            (
                "pub use std;",
                "expected block, const, enum, fn, or struct after pub",
            ),
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
    fn block_output_cycle_distinguishes_absent_from_explicit_zero() {
        for (suffix, expected) in [
            ("", None),
            (" @0", Some(OutputCycles::Uniform(0))),
            (" @3", Some(OutputCycles::Uniform(3))),
        ] {
            let source = format!("block b(clk, x: u8) -> u8{suffix} {{ x }}");
            let module = parse_module(&source).unwrap();
            assert_eq!(module.blocks[0].output_cycles, expected);
        }
    }

    #[test]
    fn block_output_cycles_parse_named_ports() {
        let source = r#"
struct Ports { saved: u8, live: u8 }
block tap(clk, x: u8) -> Ports @(saved: 1, live: 0) {
    reg q: u8;
    q := x;
    Ports { saved: q, live: x }
}
"#;
        let module = parse_module(source).unwrap();
        assert_eq!(
            module.blocks[0].output_cycles,
            Some(OutputCycles::ByPort(BTreeMap::from([
                ("saved".to_owned(), 1),
                ("live".to_owned(), 0),
            ])))
        );
    }

    #[test]
    fn block_output_cycles_reject_duplicate_ports() {
        let source = "block b(clk, x: u8) -> u8 @(saved: 1, saved: 0) { x }";
        assert_eq!(
            parse_module(source).unwrap_err(),
            "1:49: duplicate output cycle for 'saved', found '{'"
        );
    }

    #[test]
    fn block_input_cycles_distinguish_absent_from_explicit_zero() {
        let source = "block b(clk, x: u8, y: u8 @1, z: u8 @0) -> u8 @2 { x }";
        let module = parse_module(source).unwrap();
        let block = &module.blocks[0];
        assert_eq!(block.input_cycles.get("x"), None);
        assert_eq!(block.input_cycles.get("y"), Some(&1));
        assert_eq!(block.input_cycles.get("z"), Some(&0));
        assert_eq!(block.output_cycles, Some(OutputCycles::Uniform(2)));
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
    fn block_type_parameters_are_retained_in_ports_and_body_annotations() -> Result<(), String> {
        let module = parse_module(
            r#"
            pub block pipe_reg<T>(clk, rst, valid_in: bool, data_in: T) -> (bool, T) {
                reg data: T = zero!<T>();
                let next: T = data_in;
                data := next;
                (valid_in, data)
            }
            block pair<R: type, L>(clk, input: (L, R[2])[3]) -> (L, R[2])[3] { input }
            block plain(clk) -> () { () }
            "#,
        )?;
        let pipe = &module.blocks[0];
        let t = Type::Named("T".into());
        assert_eq!(pipe.type_params, ["T"]);
        assert_eq!(pipe.params[1].ty, t);
        assert_eq!(pipe.result, Type::Tuple(vec![Type::Bool, t.clone()]));
        assert!(matches!(
            &pipe.statements[..],
            [Stmt::Reg { ty, reset: Some(Expr::Zero(zero_ty)), .. }, Stmt::Let { ty: Some(let_ty), .. }, Stmt::Write { .. }]
                if ty == &t && zero_ty == &t && let_ty == &t
        ));
        let pair = &module.blocks[1];
        assert_eq!(pair.type_params, ["R", "L"]);
        assert_eq!(
            pair.params[0].ty,
            Type::Array {
                element: Box::new(Type::Tuple(vec![
                    Type::Named("L".into()),
                    Type::Array {
                        element: Box::new(Type::Named("R".into())),
                        size: 2
                    },
                ])),
                size: 3,
            }
        );
        assert_eq!(pair.result, pair.params[0].ty);
        assert!(module.blocks[2].type_params.is_empty());
        Ok(())
    }

    #[test]
    fn block_callbacks_preserve_symbolic_signatures_and_named_actuals() -> Result<(), String> {
        let module = parse_module(
            r#"
block chain(clk, rst, packet: (bool, u8)) -> (bool, u32) {
    let result = advance<math::widen, square>(packet);
    result
}
block advance<In: type, Mid, F: fn(In) -> Mid, Out, G: fn(Mid) -> Out>(
    clk, rst, prev: (bool, In)
) -> (bool, Out) {
    (prev.0, G(F(prev.1)))
}
block consume<Sink: fn(u8)>(clk, x: u8) -> () { Sink(x) }
block compare(clk, advance: u8, x: u8, y: u8, z: u8) -> (bool, bool) {
    (advance < x, y > (z))
}
"#,
        )?;
        let advance = &module.blocks[1];
        assert_eq!(advance.type_params, ["In", "Mid", "Out"]);
        assert_eq!(
            advance.callback_params,
            [
                CallbackParam {
                    name: "F".into(),
                    params: vec![Type::Named("In".into())],
                    result: Type::Named("Mid".into())
                },
                CallbackParam {
                    name: "G".into(),
                    params: vec![Type::Named("Mid".into())],
                    result: Type::Named("Out".into())
                }
            ]
        );
        assert!(matches!(&module.blocks[0].statements[..],
            [Stmt::Let {value: Expr::ParametricCall {name, parametrics, args}, ..}]
                if name == "advance" && parametrics == &names(["math::widen", "square"]) && args == &[Expr::Name("packet".into())]
        ));
        assert!(module.blocks[2].type_params.is_empty());
        assert_eq!(
            module.blocks[2].callback_params[0].result,
            Type::Tuple(vec![])
        );
        let Expr::Tuple(pair) = &module.blocks[3].output else {
            panic!("expected comparisons using the shadowing data port");
        };
        assert!(matches!(
            &pair[..],
            [
                Expr::Binary {
                    op: BinaryOp::Lt,
                    ..
                },
                Expr::Binary {
                    op: BinaryOp::Gt,
                    ..
                }
            ]
        ));
        Ok(())
    }

    #[test]
    fn callable_formals_preserve_the_lexical_rules_of_the_enclosing_item() -> Result<(), String> {
        parse_module("block compare<F!: fn(u8) -> u8>(clk, x: u8) -> bool { /* BX */ x!=F!(x) }")?;
        let error =
            parse_module("fn compare<F: fn(u8) -> u8>(x: u8) -> bool { /* DSLX */ x != F(x) }")
                .unwrap_err();
        assert!(error.contains("use a // comment in a function"), "{error}");
        Ok(())
    }

    #[test]
    fn block_type_parameter_lists_require_distinct_type_binders() {
        for (params, error) in [
            ("<>", "expected a block type parameter"),
            ("<T, T: type>", "duplicate block type parameter 'T'"),
            (
                "<T, T: fn(u8)->u8>",
                "duplicate block callable parameter 'T'",
            ),
            ("<F: fn(u8), F: type>", "duplicate block type parameter 'F'"),
            (
                "<F: fn(u8), F: fn()>",
                "duplicate block callable parameter 'F'",
            ),
            ("<N: u32>", "expected 'type'"),
            ("<T = u32>", "expected ','"),
            ("<T,,U>", "expected identifier"),
            ("<bool>", "expected identifier"),
            ("<T U>", "expected ','"),
        ] {
            let source = format!("block b{params}(clk) -> bool {{ false }}");
            assert!(
                parse_module(&source).unwrap_err().contains(error),
                "{source}"
            );
        }
        assert!(parse_module("extern block b<T>(clk, x: T) -> T;").is_err());
    }

    #[test]
    fn integer_literal_annotations_use_the_enclosing_block_type_parameters() -> Result<(), String> {
        let module = parse_module(
            r#"
            struct Header { T: u8 }
            block is_zero<T>(clk, x: T) -> bool {
                let n = T:0;
                let y = match Header { T: 0 }.T { 0 => true, _ => false };
                match x { T:0 => y, _ => false }
            }
            "#,
        )?;
        let block = &module.blocks[0];
        let zero = Expr::Literal {
            ty: Some(Type::Named("T".into())),
            text: "0".into(),
        };
        assert!(matches!(&block.statements[0], Stmt::Let { value, .. } if value == &zero));
        let Expr::Match { arms, .. } = &block.output else {
            panic!("expected match");
        };
        assert_eq!(arms[0].patterns, [MatchPattern::Literal(zero)]);
        let subsequent = "block b<T>(clk, x: T) -> T { T:0 } block plain(clk) -> u8 { T:0 }";
        assert!(parse_module(subsequent).is_err());
        Ok(())
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
            "extern block lane(clk, x: u8 @1) -> bool;",
            "extern block lane(clk) -> bool",
        ] {
            assert!(parse_module(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_malformed_paths_and_qualified_declarations() {
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
        assert!(parse_module("struct arithmetic::Payload { x: u8 }").is_err());
    }

    #[test]
    fn qualified_names_can_refer_to_bx_blocks_and_constants() -> Result<(), String> {
        let module = parse_module(
            r#"use library;
block top(clk, x: u8) -> u8 {
    inst child: library::pipe;
    child.x := library::OFFSET;
    match x { library::OFFSET | library::LIMIT => child.out, _ => u8:0 }
}"#,
        )?;
        let block = &module.blocks[0];
        assert!(
            matches!(&block.statements[0], Stmt::Inst { block, .. } if block == "library::pipe")
        );
        assert!(
            matches!(&block.statements[1], Stmt::Connect { value: Expr::Name(name), .. } if name == "library::OFFSET")
        );
        let Expr::Match { arms, .. } = &block.output else {
            panic!("expected a match on the input");
        };
        assert_eq!(
            arms[0].patterns,
            [
                MatchPattern::Literal(Expr::Name("library::OFFSET".into())),
                MatchPattern::Literal(Expr::Name("library::LIMIT".into())),
            ]
        );
        Ok(())
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
    fn map_lambdas_parse_scopes_annotations_and_ranges() -> Result<(), String> {
        let module = parse_module(
            r#"
block wrapper(clk, xs: (u8, u8)[2]) -> u8[2] {
    map({ let ys = xs; ys }, ((|p: (u8, u8),| -> u8 {
        let (a, b) = p;
        a | b
    })),)
}
fn visit() -> u8[3] {
    map(((u8:1..=u8:3)), |_| u8:4)
}
"#,
        )?;
        let Expr::Map(mapped) = &module.blocks[0].output else {
            panic!("expected an inline map")
        };
        assert!(matches!(
            mapped.iterable,
            ForIterable::Array(Expr::Scope { .. })
        ));
        assert_eq!(mapped.parameter, "p");
        assert!(matches!(&mapped.parameter_type, Some(Type::Tuple(items)) if items.len() == 2));
        assert_eq!(
            mapped.result_type,
            Some(Type::Bits {
                signed: false,
                width: 8
            })
        );
        assert!(matches!(mapped.body, Expr::Scope { .. }));
        let Expr::Map(mapped) = &module.functions[0].body else {
            panic!("expected a map over a range")
        };
        assert!(matches!(
            mapped.iterable,
            ForIterable::Range {
                inclusive: true,
                ..
            }
        ));
        assert_eq!(mapped.parameter, "_");
        assert_eq!(mapped.parameter_type, None);
        assert_eq!(mapped.result_type, None);
        assert!(matches!(mapped.body, Expr::Literal { .. }));
        Ok(())
    }

    #[test]
    fn block_map_callbacks_use_function_tokens_only_inside_the_callback() -> Result<(), String> {
        parse_module(
            r#"
block b(clk, xs: u8[2], x: u8) -> u8[2] {
    let mapped = map(if x!=u8:0 { /* BX */ xs } else { xs }, ((|x| {
        let a!=x;
        map([a!], |y| { let b!=y; b! })[u32:0]
    })),);
    if x!=u8:0 { /* BX */ mapped } else { xs }
}
"#,
        )?;
        assert!(
            parse_module("block b(clk, xs:u8[2]) -> bool[2] { map(xs, |x| x!=u8:0) }").is_err()
        );
        let commented = "block b(clk, xs:u8[2]) -> u8[2] {\n    map(xs, |x| {\n        /* comment */ x\n    })\n}";
        assert_eq!(
            parse_module(commented).unwrap_err(),
            "3:9: use a // comment in a function"
        );
        Ok(())
    }

    #[test]
    fn ordinary_calls_named_map_and_bitwise_or_remain_unambiguous() -> Result<(), String> {
        let module = parse_module(
            r#"
fn map(x: u8, y: u8) -> u8 { x + y }
block wrapper(clk, x: u8, y: u8) -> u8 { map(x, (x | y)) }
"#,
        )?;
        let Expr::Call { name, args } = &module.blocks[0].output else {
            panic!("expected an ordinary call")
        };
        assert_eq!(name, "map");
        assert!(matches!(&args[..], [Expr::Name(_), Expr::Binary { .. }]));
        Ok(())
    }

    #[test]
    fn shadowed_map_calls_keep_ordinary_block_operand_grammar() -> Result<(), String> {
        for operand in [
            "unroll_for! (i, a) in u8:0..u8:1 { let v = a + i; v }(x) + u8:1",
            "{ let y = x; { y } }",
            "if true { if true { () }; x } else { x }",
            "map([x], |v| (if true { () }, v).1)[u32:0]",
        ] {
            let source = format!(
                "block m(clk, x: u8) -> u8 {{ map({operand}, x) }}\n\
                 block n(clk, x: u8) -> u8 {{ other({operand}, x) }}\n\
                 fn map(a: u8, b: u8) -> u8 {{ a + b }}"
            );
            let module = parse_module(&source)?;
            let (Expr::Call { args: map, .. }, Expr::Call { args: other, .. }) =
                (&module.blocks[0].output, &module.blocks[1].output)
            else {
                panic!("expected ordinary calls with {operand}")
            };
            assert_eq!(map, other, "{operand}");
        }
        Ok(())
    }

    #[test]
    fn block_map_grammar_follows_imports_externs_and_callback_formals() -> Result<(), String> {
        for (formals, declaration) in [
            ("", "extern fn map(a:u8, b:u8) -> u8;"),
            ("", "block map(clk, a:u8, b:u8) -> u8 { a }"),
            ("<map: fn(u8, u8) -> u8>", ""),
        ] {
            let source = format!(
                "block apply{formals}(clk, x:u8) -> u8 {{ map({{ u8:0 }}, x) }}\n{declaration}"
            );
            assert!(parse_module(&source).is_err(), "{source}");
            parse_module(&source.replace("{ u8:0 }", "u8:0"))?;
        }
        let later_use = "block apply(clk,x:u8)->u8 { map({ u8:0 }, x) } use ops::{other, map};";
        let parsed = parse_module(later_use)?;
        assert!(matches!(parsed.blocks[0].output, Expr::NamedMap { .. }));
        parse_module(
            "extern fn map::step(x:u8) -> u8; block apply(clk,x:u8) -> u8[1] { map({[x]}, map::step) }",
        )?;
        Ok(())
    }

    #[test]
    fn named_maps_preserve_call_syntax_for_arrays_and_grouped_ranges() -> Result<(), String> {
        let module = parse_module(
            r#"
fn by_range() -> u8[3] { map(((u8:1..=u8:3)), (helpers::twice),) }
fn by_array(xs: u8[3]) -> u8[3] { map(map(xs, twice), helpers::twice) }
"#,
        )?;
        let Expr::NamedMap {
            target,
            iterable,
            callback,
            parametrics,
        } = &module.functions[0].body
        else {
            panic!("expected a map over a range")
        };
        assert_eq!(target, "map");
        assert_eq!(callback, "helpers::twice");
        assert!(parametrics.is_empty());
        assert!(matches!(
            iterable.as_ref(),
            ForIterable::Range {
                inclusive: true,
                ..
            }
        ));
        let Expr::Call { name, args } = &module.functions[1].body else {
            panic!("expected an ordinary call")
        };
        assert_eq!(name, "map");
        assert!(
            matches!(&args[..], [Expr::Call { name, args }, Expr::Name(callback)]
            if name == "map" && callback == "helpers::twice"
            && matches!(&args[..], [Expr::Name(xs), Expr::Name(f)] if xs == "xs" && f == "twice"))
        );
        for value in ["other(u8:0..u8:3, f)", "u8:0..u8:3", "map(u8:0..u8:3, f())"] {
            assert!(
                parse_module(&format!("fn g() -> u8[3] {{ {value} }}")).is_err(),
                "{value}"
            );
        }
        Ok(())
    }

    #[test]
    fn named_map_callbacks_accept_explicit_parametric_arguments() -> Result<(), String> {
        let module = parse_module(
            r#"
use lib;
fn f<N:u32>(x:uN[N])->uN[N] { x }
fn by_array<K:u32>(xs:u8[2])->u8[2] { map(xs, f<{K + u32:1}>) }
fn by_range()->u8[2] { map(u8:0..u8:2, ((lib::f<8,>)),) }
block before(clk, xs:u8[2])->u8[2] { map(xs, f<8>) }
"#,
        )?;
        let Expr::NamedMap {
            iterable,
            callback,
            parametrics,
            ..
        } = &module.functions[1].body
        else {
            panic!("expected an explicit specialization on the map callback");
        };
        assert!(matches!(iterable.as_ref(), ForIterable::Array(Expr::Name(name)) if name == "xs"));
        assert_eq!(callback, "f");
        assert!(matches!(
            &parametrics[..],
            [Expr::Binary {
                op: BinaryOp::Add,
                ..
            }]
        ));
        assert!(
            matches!(&module.functions[2].body, Expr::NamedMap { callback, parametrics, .. }
            if callback == "lib::f" && matches!(&parametrics[..], [Expr::Literal { ty: None, text }] if text == "8"))
        );
        assert!(
            matches!(&module.blocks[0].output, Expr::NamedMap { callback, .. } if callback == "f")
        );
        assert!(parse_module("fn g()->u8 { f<8> }").is_err());
        let shadowed =
            parse_module("fn map(x:u8, b:bool)->u8{x} block b(clk,x:u8,y:u8)->u8{map(x, x < y)}")?;
        assert!(matches!(&shadowed.blocks[0].output, Expr::Call { args, .. }
            if matches!(&args[1], Expr::Binary { op: BinaryOp::Lt, .. })));
        Ok(())
    }

    #[test]
    fn map_lambda_grammar_rejects_invalid_binders_and_hardware_statements() {
        for value in [
            "map(xs, || u8:0)",
            "map(xs, |a, b| a)",
            "map(xs, |(a, b)| a)",
            "map(xs, |self| self)",
            "map(xs, |out| out)",
            "map(xs, |_| _)",
            "map(xs, |x| -> u8 x)",
            "map(xs, |x| { reg state: u8; x })",
            "map(xs, |x| { inst child: Child; x })",
            "map(xs, |x| { state := x; x })",
            "map(xs, |x| x, xs)",
            "other(xs, |x| x)",
            "lib::map(xs, |x| x)",
            "|x| x",
        ] {
            assert!(
                parse_module(&format!("block b(clk, xs: u8[2]) -> u8[2] {{ {value} }}")).is_err(),
                "{value}"
            );
        }
        assert!(parse_module("fn f(xs: u8[2]) -> u8[2] { map(xs, |c| c.out) }").is_err());
    }

    #[test]
    fn counted_loops_parse_array_values_and_nested_tuple_binders() -> Result<(), String> {
        let module = parse_module(
            r#"
fn fold(xs: (u8, u16, u8)[2]) -> (u8, u8) {
    for ((x, .., y), (a, b)): ((u8, u16, u8), (u8, u8)) in ((), xs).1 {
        let next = (a + x, b + y);
        next
    }((u8:0, u8:0))
}
block wrapper(clk, xs: u8[2]) -> u8 {
    for (i, acc) in xs { let step = { let j = i; j }; acc + step }(u8:0)
}
"#,
        )?;
        let Expr::For(fold) = &module.functions[0].body else {
            panic!("expected a counted loop")
        };
        assert_eq!(
            fold.element,
            BindingPattern::Tuple(vec![
                BindingPattern::Name("x".into()),
                BindingPattern::Rest,
                BindingPattern::Name("y".into())
            ])
        );
        assert!(matches!(&fold.accumulator, BindingPattern::Tuple(fields) if fields.len() == 2));
        assert!(
            matches!(&fold.annotation, Some((Type::Tuple(items), Type::Tuple(carry))) if items.len() == 3 && carry.len() == 2)
        );
        assert!(
            matches!(&fold.iterable, ForIterable::Array(Expr::Field { name, .. }) if name == "1")
        );
        assert!(matches!(fold.body, Expr::Scope { .. }));
        let Expr::For(fold) = &module.blocks[0].output else {
            panic!("expected a counted loop")
        };
        assert!(matches!(&fold.iterable, ForIterable::Array(Expr::Name(name)) if name == "xs"));
        assert!(matches!(fold.body, Expr::Scope { .. }));
        Ok(())
    }

    #[test]
    fn counted_ranges_preserve_endpoints_and_inclusivity_through_parentheses() -> Result<(), String>
    {
        for (range, inclusive) in [
            ("u8:1..u8:3", false),
            ("(u8:1)..=(u8:3)", true),
            ("((u8:1..u8:3))", false),
            ("((u8:1)..=u8:3)", true),
        ] {
            let module = parse_module(&format!(
                "fn f() -> () {{ for (_, ()) in {range} {{ }}(()) }}"
            ))?;
            let Expr::For(fold) = &module.functions[0].body else {
                panic!("expected a counted loop")
            };
            assert!(matches!(&fold.iterable, ForIterable::Range {
                start: Expr::Literal { text: start, .. },
                limit: Expr::Literal { text: end, .. }, inclusive: flag
            } if start == "1" && end == "3" && *flag == inclusive));
        }
        Ok(())
    }

    #[test]
    fn counted_loop_binders_and_precedence_follow_dslx_functions() {
        for expr in [
            "for pair in xs { pair.0 }(u8:0)",
            "for (i, ..) in xs { i }(u8:0)",
            "for ((i, _), i) in xs { i }(u8:0)",
            "for (i, acc) in xs { acc }",
            "for (i, acc) in xs { acc := i; acc }(u8:0)",
            "for (i, acc) in xs { acc }(u8:0) + u8:1",
        ] {
            assert!(
                parse_module(&format!("fn f(xs:u8[2]) -> u8 {{ {expr} }}")).is_err(),
                "{expr}"
            );
        }
        for expr in [
            "(for (i, acc) in xs { acc }(u8:0)) + u8:1",
            "u8:1 + for (i, acc) in xs { acc }(u8:0)",
        ] {
            assert!(
                parse_module(&format!("fn f(xs:u8[2]) -> u8 {{ {expr} }}")).is_ok(),
                "{expr}"
            );
        }
        // Existing BX block names and calls spelled 'for' remain unambiguous.
        assert!(parse_module("block b(clk, for:u8) -> u8 { for }").is_ok());
        assert!(parse_module("block b(clk, x:u8) -> u8 { let y = for(x); y }").is_ok());
    }

    #[test]
    fn parses_annotated_and_nested_unrolls_with_aggregate_seeds() {
        let module = parse_module(
            r#"
            block fold(clk, bounds: (u8, u8)) -> u8[2] {
                unroll_for! (_, acc): (u8, u8[2]) in bounds.0..bounds.1 {
                    let value: u8 = unroll_for! (i, sum) in 0..2 { sum + i }(u8:0);
                    update(acc, u32:0, value)
                }(zero!<u8[2]>())
            }
            "#,
        )
        .unwrap();
        let Expr::UnrollFor(fold) = &module.blocks[0].output else {
            panic!("expected outer fold");
        };
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        let array = &module.blocks[0].result;
        assert_eq!(fold.index, "_");
        assert_eq!(fold.accumulator, "acc");
        assert_eq!(fold.annotation, Some((byte.clone(), array.clone())));
        assert!(matches!(&fold.start, Expr::Field { name, .. } if name == "0"));
        assert!(matches!(&fold.limit, Expr::Field { name, .. } if name == "1"));
        let [
            Stmt::Let {
                name,
                ty,
                value: Expr::UnrollFor(inner),
            },
        ] = &fold.body[..]
        else {
            panic!("expected a let initialized with the inner fold");
        };
        assert_eq!(name, "value");
        assert_eq!(ty, &Some(byte));
        assert_eq!(inner.index, "i");
        assert_eq!(inner.accumulator, "sum");
        assert_eq!(inner.annotation, None);
        assert!(
            matches!(&fold.result, Expr::Call { name, args } if name == "update" && args.len() == 3)
        );
        assert_eq!(fold.seed, Expr::Zero(array.clone()));
    }

    #[test]
    fn duplicate_unroll_binders_are_invalid_even_in_a_zero_trip_body() {
        let source = r#"block b(clk) -> u8 {
  unroll_for! (_, _) in u32:0..u32:0 {
    unroll_for! (i, i) in u32:0..u32:1 { u8:1 }(u8:0)
  }(u8:0)
}"#;
        assert_eq!(
            parse_module(source).unwrap_err(),
            "3:21: duplicate loop binding 'i'"
        );
        assert!(parse_module(&source.replace("(i, i)", "(_, _)")).is_ok());
    }

    #[test]
    fn rejects_malformed_folds_and_zeros() {
        for expr in [
            "0..2",
            "unroll_for! (i) in 0..2 { i }(0)",
            "unroll_for! (i, a): (u32,) in 0..2 { a }(0)",
            "unroll_for! (i, a) in 0..=2 { a }(0)",
            "unroll_for! (i, a) in 0 . . 2 { a }(0)",
            "unroll_for! (i, a) in 0..2 { a := 0; a }(0)",
            "unroll_for! (i, a) in 0..2 { reg r: u32; r }(0)",
            "unroll_for! (i, a) in 0..2 { let x = a; }(0)",
            "unroll_for! (i, a) in 0..2 { a }",
            "zero!<>()",
            "zero!<u32>",
            "zero!<u32>(1)",
        ] {
            let source = format!("block b(clk) -> u32 {{ {expr} }}");
            assert!(parse_module(&source).is_err(), "{source}");
        }
    }

    #[test]
    fn unroll_height_includes_each_let_initializer() {
        let source = format!(
            "block b(clk) -> u32 {{ unroll_for! (i, a) in 0..1 {{ let x = {}i; a }}(0) }}",
            "i + ".repeat(MAX_SYNTAX_DEPTH - 1),
        );
        assert!(
            parse_module(&source)
                .unwrap_err()
                .contains("expression tree exceeds")
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
