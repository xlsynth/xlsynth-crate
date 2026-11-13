// SPDX-License-Identifier: Apache-2.0

//! Parser for BX's structural declarations and DSLX-style value expressions.

pub mod outline;
pub(crate) mod source_bytes;
mod test_declarations;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use xlsynth_pir::{IrBits, IrValue};

use crate::BxWarningKind;
use crate::ast::{
    BinaryOp, BindingPattern, Block, BlockSource, CallbackParam, ConstDef, EnumDef, EnumMember,
    Expr, ExternBlock, ExternFn, ForIterable, ForLoop, Function, InvariantDecl, InvariantLemma,
    LetBinding, MapExpr, MatchArm, MatchPattern, Module, OutputCycles, Param, ParametricBinding,
    SourceFile, SourcePosition, SourceSpan, Stmt, StructDef, StructSplat, Type, TypeAlias, UnaryOp,
    UnrollFor, UsePath, WidthSliceType,
};
use crate::ast::{ConstAssert, ScopeStatement};

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
                | "const_assert!"
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

/// DSLX permits `self` only at the head of a value path and `_` only after
/// a qualifier. Both spellings are valid in declarations.
pub(crate) fn validate_dslx_value_path(name: &str) -> Result<(), String> {
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
    // Functions and enums end at a closing brace. Constants, assertions and
    // type aliases end at a semicolon outside braced expressions.
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
                    Some("const_assert!") => "use a // comment in a constant assertion",
                    Some("enum") => "use a // comment in an enum",
                    Some("type") => "use a // comment in a type alias",
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
        } else if (dialect == Dialect::Dslx && matches!(b, b'"' | b'`'))
            || (b == b'"' && (brace_depth != 0 || dslx_item.is_some() || callback))
        {
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
        } else if dialect == Dialect::Bx && (rest.starts_with("..=") || rest.starts_with("...")) {
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
                .is_none_or(|t: &Token| matches!(t.text, "pub" | ";" | "}" | "]"));
        match text {
            "fn" | "const" | "const_assert!" | "enum" | "type"
                if item_start && dslx_item.is_none() =>
            {
                dslx_item = Some(text);
            }
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
                (Some("enum"), "}") | (Some("const" | "const_assert!" | "type"), ";")
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

/// Declaration attributes select a test body or configure ordinary warning
/// handling.
enum DeclarationAttributes {
    Test { quickcheck: bool },
    Invariant,
    Warnings(BTreeSet<BxWarningKind>),
}

/// Decode one DSLX byte character and return its value and token length.
fn scan_character(source: &str) -> Option<(u8, usize)> {
    let bytes = source.as_bytes();
    let (value, len) = match bytes.get(1)? {
        b'\\' => scan_byte_escape(&bytes[2..])?,
        value if value.is_ascii() && *value != b'\n' => (*value, 0),
        _ => return None,
    };
    (bytes.get(len + 2) == Some(&b'\'')).then_some((value, len + 3))
}

/// Read a DSLX byte escape after its backslash.
fn scan_byte_escape(bytes: &[u8]) -> Option<(u8, usize)> {
    match bytes {
        [b'x', high, low, ..] => Some((
            (char::from(*high).to_digit(16)? * 16 + char::from(*low).to_digit(16)?) as u8,
            3,
        )),
        [escaped, ..] => {
            let value = match escaped {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'0' => 0,
                b'\\' | b'\'' | b'"' => *escaped,
                _ => return None,
            };
            Some((value, 1))
        }
        [] => None,
    }
}

/// Decode the UTF-8 bytes and escape sequences inside a DSLX string literal.
fn decode_byte_string(text: &str) -> Result<Vec<u8>, &'static str> {
    let mut remaining = &text.as_bytes()[1..text.len() - 1];
    let mut value = Vec::new();
    // The concrete layout charges eight bits per byte plus the array node.
    let max_bytes = (crate::check::MAX_TYPE_SIZE - 1) / 8;
    while let [first, tail @ ..] = remaining {
        remaining = if *first != b'\\' {
            value.push(*first);
            tail
        } else if let Some(unicode) = tail.strip_prefix(b"u{") {
            let close = unicode
                .iter()
                .take(7)
                .position(|byte| *byte == b'}')
                .ok_or("expected 1 to 6 hex digits and '}' in Unicode escape")?;
            let codepoint = std::str::from_utf8(&unicode[..close])
                .ok()
                .filter(|digits| digits.bytes().all(|digit| digit.is_ascii_hexdigit()))
                .and_then(|digits| u32::from_str_radix(digits, 16).ok())
                .and_then(char::from_u32)
                .ok_or("invalid Unicode escape in DSLX string")?;
            value.extend_from_slice(codepoint.encode_utf8(&mut [0; 4]).as_bytes());
            &unicode[close + 1..]
        } else {
            let (byte, len) = scan_byte_escape(tail).ok_or("invalid escape in DSLX string")?;
            value.push(byte);
            &tail[len..]
        };
        if value.len() > max_bytes {
            return Err("DSLX string exceeds the maximum concrete type size");
        }
    }
    Ok(value)
}

/// Parse a source module without loading its imports. Imported spellings of
/// `f<a,b>(x)` are parsed as calls. Use [`crate::parse_module_with_options`] to
/// distinguish these calls from tuples of comparisons on imported constants,
/// distinguish an imported `zero!<callback>()` from the intrinsic
/// `zero!<Type>()`, or parse type arguments to imported BX structs.
/// Syntax diagnostics carry one-based line and column.
pub fn parse_module(source: &str) -> Result<Module, String> {
    Parser::new(source, false)?.module_with_imports(None)
}

/// Parse with the callable and type templates exported by the module's imports.
pub(crate) fn parse_with_imports(
    source: &str,
    templates: &BTreeSet<String>,
    type_structs: &BTreeSet<String>,
) -> Result<Module, String> {
    let mut parser = Parser::new(source, false)?;
    parser.type_structs.extend(type_structs.iter().cloned());
    parser.module_with_imports(Some(templates))
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
    diagnostic_source: Option<Arc<str>>,
    /// Last byte offset and line/column recorded for a diagnostic.
    position_cursor: (usize, SourcePosition),
    dialect: Dialect,
    tokens: Vec<Token<'s>>,
    /// Original angle tokens split during a speculative expression parse.
    saved_angle_tokens: Option<Vec<(usize, Token<'s>)>>,
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
    /// Type parameters that can annotate literals in this function or block.
    type_params: BTreeSet<&'s str>,
    /// Function and struct templates allow full expressions in their type
    /// dimensions.
    dependent_types: bool,
    /// Local template declarations disambiguate angle brackets from
    /// comparisons in a comma-separated expression list. Block callers can
    /// use function and block templates declared later in the source.
    callable_templates: BTreeMap<String, TemplateBinding>,
    /// Local and imported struct templates whose arguments are types.
    type_structs: BTreeSet<String>,
    /// Module aliases and use bindings that can start a named literal
    /// annotation.
    literal_type_heads: BTreeSet<String>,
    /// Module names in discarded literal types of the current item.
    overwritten_literal_types: BTreeSet<String>,
    /// Entries hidden by currently active data or callable bindings.
    hidden_templates: Vec<(String, TemplateBinding)>,
    /// Names in source scope, including imports. An overwritten literal type
    /// needs lexical resolution while its syntax is still available.
    visible_names: HashSet<String>,
    /// Struct and enum names can qualify a member, but cannot be bare values.
    visible_type_names: HashSet<String>,
    /// Names newly introduced in an active lexical scope.
    local_names: Vec<String>,
    literal_type_context: LiteralTypeContext,
}

/// Which parts of a nested literal annotation still require name resolution.
#[derive(Clone, Copy, PartialEq)]
enum LiteralTypeContext {
    /// The checker receives this type and its dimension expressions.
    Live,
    /// The literal's outer annotation supersedes this type.
    Overwritten,
    /// A computed dimension of an overwritten template type is discarded
    /// together with all of its subexpressions and lexical scopes.
    DiscardedExpression,
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

/// Lambdas use DSLX function grammar. Inside a BX block they can also
/// project a BX struct field named `out`.
#[derive(Clone, Copy, PartialEq)]
enum ValueContext {
    Block,
    /// Functions, module constants, enums and type aliases use DSLX value
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

/// Preserve concrete struct applications and track the depth of computed
/// arguments.
enum StructArguments {
    Concrete(Vec<u32>),
    Computed { args: Vec<Expr>, height: usize },
    Types { args: Vec<Type>, height: usize },
}

/// The explicit fields and optional final base in a struct literal.
struct StructMembers {
    fields: Vec<(String, Expr)>,
    base: Option<Box<StructSplat>>,
}

/// Parsed value arguments and their starting tokens for struct-literal
/// validation.
struct ParsedAngleArguments {
    args: Vec<Expr>,
    starts: Vec<usize>,
    height: usize,
}

/// Contents of braces before determining whether they form a guard or a value.
struct Body {
    statements: Vec<Stmt>,
    statement_spans: Vec<SourceSpan>,
    output: Option<(ParsedExpr, SourceSpan)>,
}

/// A let pattern and annotation before introducing its names into scope.
struct LetHead {
    pattern: BindingPattern,
    ty: Option<Type>,
    names: BTreeSet<String>,
}

/// A block's inferred types and explicit value and callable formals.
#[derive(Default)]
struct BlockParametrics {
    type_params: Vec<String>,
    callback_params: Vec<CallbackParam>,
    value_parametrics: Vec<ParametricBinding>,
}

/// Retains the built tree's height until its parent is constructed. Syntactic
/// nesting alone cannot bound a left-associative expression tree.
struct ParsedExpr {
    value: Expr,
    height: usize,
    spelling: ExprSpelling,
}

/// Canonical spellings of tuple and range patterns in one match expression.
#[derive(Default)]
struct MatchPatternSpellings {
    current: String,
    contains_range: bool,
    seen: BTreeSet<String>,
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

/// Find the type formals following a declaration's opening angle. Annotations
/// and defaults can use any of them, including one declared later in the list.
fn declared_type_params<'s>(tokens: &[Token<'s>]) -> Vec<&'s str> {
    let mut next_formal = true;
    let mut groups = 0usize;
    let mut angles = 0usize;
    let mut names = Vec::new();
    for ts in tokens.windows(3) {
        let text = ts[0].text;
        if next_formal && is_use_identifier(text) && (ts[1].text != ":" || ts[2].text == "type") {
            names.push(text);
        }
        next_formal = false;
        match text {
            "(" | "[" | "{" => groups += 1,
            ")" | "]" | "}" => groups = groups.saturating_sub(1),
            "<" if groups == 0 => angles += 1,
            ">" | ">=" | ">>" if groups == 0 => {
                let Some(remaining) = angles.checked_sub(if text == ">>" { 2 } else { 1 }) else {
                    break;
                };
                angles = remaining;
            }
            "," if groups == 0 && angles == 0 => next_formal = true,
            _ => { /* Other tokens do not delimit formals. */ }
        }
    }
    names
}

impl<'s> Parser<'s> {
    fn new(source: &'s str, dslx_type_display: bool) -> Result<Self, String> {
        let tokens = lex(source, Dialect::Bx, None)?;
        let mut callable_templates = BTreeMap::new();
        let mut header = None;
        let mut type_structs = BTreeSet::new();
        let mut literal_type_heads = BTreeSet::new();
        for (index, ts) in tokens.windows(3).enumerate() {
            match (ts[0].text, ts[1].text, ts[2].text) {
                ("type", name, "=") if is_function_identifier(name) => {
                    literal_type_heads.insert(name.to_owned());
                }
                ("fn" | "block", _, "<") => header = Some(ts[1]),
                ("struct", name, "<")
                    if tokens.get(index + 3).is_some_and(|t| is_identifier(t.text))
                        && (tokens.get(index + 4).is_some_and(|t| t.text != ":")
                            || tokens.get(index + 5).is_some_and(|t| t.text == "type")) =>
                {
                    type_structs.insert(name.to_owned());
                }
                (":", annotation, _) if annotation != "type" => {
                    // Only value and callable formals take explicit arguments.
                    if let Some(name) = header.take() {
                        callable_templates
                            .insert(name.text.to_owned(), TemplateBinding::local(name.start));
                    }
                }
                (">", _, _) => header = None,
                _ => { /* Other tokens do not identify a callable template. */ }
            }
        }
        Ok(Self {
            source,
            diagnostic_source: None,
            position_cursor: (0, SOURCE_START),
            dialect: Dialect::Bx,
            tokens,
            saved_angle_tokens: None,
            pos: 0,
            depth: 0,
            dslx_type_display,
            value_context: ValueContext::Block,
            expression_grammar: ExpressionGrammar::Block,
            block_map_shadowed: false,
            forward_map_import: false,
            type_params: BTreeSet::new(),
            dependent_types: false,
            callable_templates,
            type_structs,
            literal_type_heads,
            overwritten_literal_types: BTreeSet::new(),
            hidden_templates: Vec::new(),
            visible_names: HashSet::new(),
            visible_type_names: HashSet::new(),
            local_names: Vec::new(),
            literal_type_context: LiteralTypeContext::Live,
        })
    }

    /// Advance from the preceding diagnostic site so many sites on one line
    /// take linear work in total.
    fn position_at(&mut self, offset: usize) -> SourcePosition {
        let (start, position) = self.position_cursor;
        let position = advance_position(position, &self.source[start..offset]);
        self.position_cursor = (offset, position);
        position
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
            if let Some(saved) = &mut self.saved_angle_tokens {
                saved.push((self.pos, *token));
            }
            token.text = &token.text[1..];
            token.start += 1;
            true
        } else {
            false
        }
    }

    /// Leave the opening angle for the same argument parser as bare `<...>`.
    fn eat_turbofish(&mut self) -> bool {
        self.peek(1) == "<" && self.eat("::")
    }

    /// Whether this opening angle is explicitly prefixed with `::`.
    fn turbofish_at(&self, angle: usize) -> bool {
        angle > 0 && self.tokens[angle].text == "<" && self.tokens[angle - 1].text == "::"
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
                "place a BX warning attribute before the declaration, including its optional 'pub'",
            );
        }
        if self.dialect == Dialect::Bx && token.text == "#" && self.peek(1) == "!" {
            return diagnostic(
                self.source,
                token.start,
                "write #![...] at the top level of the BX file",
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
        let first_name = self.local_names.len();
        let map_shadowed = self.block_map_shadowed;
        let result = parse(self);
        self.block_map_shadowed = map_shadowed;
        self.callable_templates
            .extend(self.hidden_templates.drain(first..));
        for name in self.local_names.drain(first_name..) {
            self.visible_names.remove(&name);
        }
        result
    }

    fn bind_local_name(&mut self, name: &str) {
        if !self.visible_names.contains(name) {
            self.visible_names.insert(name.to_owned());
            self.local_names.push(name.to_owned());
        }
    }

    fn shadow_callable(&mut self, name: &str) {
        self.bind_local_name(name);
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
                "#" if depth == 0
                    && self.peek(0) == "["
                    && matches!(self.peek(1), "test" | "quickcheck" | "invariant")
                    && self.peek(2) == "]"
                    && matches!(self.peek(3), "fn" | "block") =>
                {
                    // Test names never bind call targets, including the map
                    // builtin.
                    self.pos += 4;
                }
                "fn" | "block" | "struct"
                    if depth == 0
                        && self.peek(0) == "map"
                        && matches!(self.peek(1), "(" | "<" | "{") =>
                {
                    self.block_map_shadowed = true;
                }
                // A use can start the file or follow a declaration or module
                // attribute.
                "use"
                    if depth == 0
                        && (self.pos == 1
                            || matches!(self.tokens[self.pos - 2].text, ";" | "}" | "]")) =>
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
            .any(|token| matches!(token.text, "<" | "map" | "use"))
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
            self.literal_type_heads.extend(imports.keys().cloned());
            self.callable_templates.extend(imports);
        }
        self.module()
    }

    fn module(&mut self) -> Result<Module, String> {
        let mut module = Module::default();
        let mut use_bindings = BTreeSet::new();
        while !self.peek(0).is_empty() {
            let first_declaration = module.declaration_order.len();
            let mut allowed_warnings = match self.attributes(&mut module)? {
                DeclarationAttributes::Test { quickcheck } => {
                    let test = self.with_name_scope(|p| p.test_declaration(quickcheck))?;
                    module.declaration_order.push(test.name.clone());
                    module.tests.push(test);
                    continue;
                }
                DeclarationAttributes::Invariant => {
                    if !self.eat("block") {
                        return Err(self.error("expected private block after #[invariant]"));
                    }
                    let position = self.position_at(self.tokens[self.pos].start);
                    let mut lemmas = Vec::new();
                    let block = self.with_name_scope(|p| {
                        p.block(false, position, BTreeSet::new(), Some(&mut lemmas))
                    })?;
                    let mut labels = HashSet::new();
                    for lemma in &lemmas {
                        if !labels.insert(&lemma.name) {
                            return Err(format!(
                                "{}:{}: duplicate lemma '{}'",
                                lemma.position.line, lemma.position.column, lemma.name
                            ));
                        }
                    }
                    module.declaration_order.push(block.name.clone());
                    module.invariants.push(InvariantDecl {
                        block,
                        position,
                        lemmas,
                        overwritten_literal_types: std::mem::take(
                            &mut self.overwritten_literal_types,
                        ),
                    });
                    continue;
                }
                DeclarationAttributes::Warnings(warnings) => warnings,
            };
            if self.peek(0).is_empty() && allowed_warnings.is_empty() {
                break;
            }
            let public = self.eat("pub");
            let declares_type = matches!(self.peek(0), "enum" | "struct" | "type");
            let is_const_assert = self.peek(0) == "const_assert!";
            if self.eat("block") {
                let name_offset = self.tokens[self.pos].start;
                let position = self.position_at(name_offset);
                let block = self.with_name_scope(|p| {
                    p.block(
                        public,
                        position,
                        std::mem::take(&mut allowed_warnings),
                        None,
                    )
                })?;
                module.declaration_order.push(block.name.clone());
                module.blocks.push(block);
            } else if allowed_warnings
                .iter()
                .any(|&kind| kind != BxWarningKind::UselessStructSplat)
            {
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
            } else if self.eat("type") {
                self.value_context = ValueContext::DslxItem;
                let def = self.with_name_scope(|p| p.type_alias(public))?;
                self.value_context = ValueContext::Block;
                self.block_map_shadowed |= def.name == "map";
                module.declaration_order.push(def.name.clone());
                module.type_aliases.push(def);
            } else if self.eat("struct") {
                let def = self.with_name_scope(|p| p.struct_def(public))?;
                module.declaration_order.push(def.name.clone());
                module.structs.push(def);
            } else if self.eat("enum") {
                self.value_context = ValueContext::DslxItem;
                let def = self.with_name_scope(|p| p.enum_def(public))?;
                self.value_context = ValueContext::Block;
                self.block_map_shadowed |= def.name == "map";
                module.declaration_order.push(def.name.clone());
                module.enums.push(def);
            } else if public {
                return Err(
                    self.error("expected block, const, enum, fn, struct, or type after pub")
                );
            } else if is_const_assert {
                if !allowed_warnings.is_empty() {
                    return Err(self.error("a const_assert! declaration cannot have an attribute"));
                }
                self.pos += 1;
                self.value_context = ValueContext::DslxItem;
                let (assertion, _) = self.with_name_scope(|p| p.const_assert_predicate())?;
                self.value_context = ValueContext::Block;
                let position = assertion.position;
                let label = format!("const_assert! at {}:{}", position.line, position.column);
                module.declaration_order.push(label.clone());
                module.const_asserts.insert(label, assertion);
            } else if self.peek(0) == "use" && !allowed_warnings.is_empty() {
                return Err(self.error("a use declaration cannot have a warning allowance"));
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
                    "expected a block, const, const_assert!, enum, fn, struct, type, extern, or use declaration",
                ));
            }
            if !allowed_warnings.is_empty() {
                let name = &module.declaration_order[first_declaration];
                module
                    .declaration_allowances
                    .insert(name.clone(), allowed_warnings);
            }
            if !self.overwritten_literal_types.is_empty() {
                module.overwritten_literal_types.insert(
                    module.declaration_order[first_declaration].clone(),
                    std::mem::take(&mut self.overwritten_literal_types),
                );
            }
            if is_const_assert {
                continue;
            }
            let names = if declares_type {
                &mut self.visible_type_names
            } else {
                &mut self.visible_names
            };
            names.extend(
                module.declaration_order[first_declaration..]
                    .iter()
                    .cloned(),
            );
        }
        if !module.tests.is_empty() || !module.invariants.is_empty() {
            let test_positions: BTreeMap<_, _> = module
                .tests
                .iter()
                .map(|t| (t.name.as_str(), ("test", t.position)))
                .chain(
                    module
                        .invariants
                        .iter()
                        .map(|i| (i.block.name.as_str(), ("invariant", i.position))),
                )
                .collect();
            let mut seen = HashSet::new();
            for name in &module.declaration_order {
                if let Some((kind, position)) = test_positions.get(name.as_str())
                    && !seen.insert(name)
                {
                    return Err(format!(
                        "{}:{}: duplicate declaration for BX {kind} '{name}'",
                        position.line, position.column
                    ));
                }
            }
        }
        Ok(module)
    }

    /// Record file-wide attributes and identify warning allowances or a test
    /// declaration.
    fn attributes(&mut self, module: &mut Module) -> Result<DeclarationAttributes, String> {
        let mut allowed = BTreeSet::new();
        while self.eat("#") {
            let module_attribute = self.eat("!");
            self.expect("[")?;
            let name = self.peek(0);
            if module_attribute {
                match name {
                    "allow" => {
                        if !module.declaration_order.is_empty() {
                            return Err(self.error(
                                "place #![allow(block_reset_signal)] before the first declaration",
                            ));
                        }
                        self.pos += 1;
                        self.expect("(")?;
                        self.expect("block_reset_signal")?;
                        self.eat(",");
                        self.expect(")")?;
                        module.allow_block_reset_signal = true;
                    }
                    "require_block_timing" => {
                        self.pos += 1;
                        module.require_block_timing = true;
                    }
                    "require_define_before_use" => {
                        if !module.declaration_order.is_empty() {
                            return Err(self.error(
                                "place #![require_define_before_use] before the first declaration",
                            ));
                        }
                        self.pos += 1;
                        module.require_define_before_use = true;
                    }
                    _ => return Err(self.error("unknown BX module attribute")),
                }
                self.expect("]")?;
                continue;
            }
            if matches!(name, "require_block_timing" | "require_define_before_use") {
                return Err(self.error(&format!("use #![{name}] at module scope")));
            }
            if matches!(name, "test" | "quickcheck" | "invariant") {
                self.pos += 1;
                if !allowed.is_empty() {
                    return Err(self.error("BX warning allowances apply to hardware declarations"));
                }
                self.expect("]")?;
                if name == "invariant" {
                    return Ok(DeclarationAttributes::Invariant);
                }
                return Ok(DeclarationAttributes::Test {
                    quickcheck: name == "quickcheck",
                });
            }
            self.expect("allow")?;
            self.expect("(")?;
            if self.peek(0) == ")" {
                return Err(self.error("expected a BX warning name in #[allow(...)]"));
            }
            loop {
                let name = self.peek(0);
                if name == "block_reset_signal" {
                    return Err(self.error("use #![allow(block_reset_signal)] at module scope"));
                }
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
        Ok(DeclarationAttributes::Warnings(allowed))
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
        if self.peek(1) != "<" && self.eat("::") {
            let member = self.take_identifier(is_dslx_member_name)?;
            return Ok(format!("{head}::{member}"));
        }
        Ok(head)
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

    /// Parse an aggregate declaration and its type or value parametrics.
    fn struct_def(&mut self, public: bool) -> Result<StructDef, String> {
        let name = self.ident()?;
        let mut type_params = Vec::new();
        let mut value_parametrics = Vec::new();
        if self.eat("<") {
            self.dependent_types = true;
            if self.peek(0) == ">" {
                return Err(self.error("expected a struct parameter"));
            }
            let mut names = BTreeSet::new();
            self.with_value_grammar(|p| {
                p.list(">", |p| {
                    let token = p.tokens[p.pos];
                    let is_type = p.peek(1) != ":" || p.peek(2) == "type";
                    if !names.insert(token.text) {
                        return Err(
                            p.error(&format!("duplicate struct parameter '{}'", token.text))
                        );
                    }
                    if is_type {
                        if !value_parametrics.is_empty() {
                            return Err(p.error("a struct cannot mix type and value parameters"));
                        }
                        let name = p.ident()?;
                        if p.eat(":") {
                            p.expect("type")?;
                        }
                        p.type_params.insert(token.text);
                        type_params.push(name);
                    } else {
                        if !type_params.is_empty() {
                            return Err(p.error("a struct cannot mix type and value parameters"));
                        }
                        value_parametrics.push(p.value_parametric(value_parametrics.len())?);
                    }
                    Ok(())
                })
            })?;
        }
        self.expect("{")?;
        let fields = if self.dependent_types {
            self.with_value_grammar(|p| p.list("}", Self::param))?
        } else {
            self.list("}", Self::param)?
        };
        self.dependent_types = false;
        self.type_params.clear();
        Ok(StructDef {
            name,
            public,
            type_params,
            value_parametrics,
            fields,
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

    /// Parse a concrete type alias whose dimensions may use module constants.
    fn type_alias(&mut self, public: bool) -> Result<TypeAlias, String> {
        let name = self.ident()?;
        self.expect("=")?;
        self.dependent_types = true;
        let ty = self.ty()?;
        self.dependent_types = false;
        self.expect(";")?;
        Ok(TypeAlias { name, public, ty })
    }

    /// Parse pure local lets and a trailing value. A body ending in a
    /// semicolon or an empty body returns unit, as in DSLX.
    fn function(&mut self, public: bool) -> Result<Function, String> {
        let name = self.ident()?;
        let mut type_params = Vec::new();
        let mut value_parametrics = Vec::new();
        let mut callback_params = Vec::new();
        if self.eat("<") {
            // An early callback annotation may use a module constant even
            // when the first value parametric is declared later in this list.
            self.dependent_types = true;
            if self.peek(0) == ">" {
                return Err(self.error("expected a function parameter"));
            }
            for formal in declared_type_params(&self.tokens[self.pos..]) {
                self.type_params.insert(formal);
                self.shadow_callable(formal);
            }
            self.list(">", |p| {
                if p.peek(1) != ":" || p.peek(2) == "type" {
                    type_params.push(p.take_identifier(is_use_identifier)?);
                    if p.eat(":") {
                        p.expect("type")?;
                    }
                } else if p.peek(2) == "fn" {
                    let param = p.callback_param()?;
                    p.shadow_callable(&param.name);
                    callback_params.push(param);
                } else {
                    value_parametrics
                        .push(p.value_parametric(value_parametrics.len() + callback_params.len())?);
                }
                Ok(())
            })?;
        }
        self.expect("(")?;
        let params = self.list(")", |p| {
            p.bind_local_name(p.peek(0));
            p.param()
        })?;
        let result = if self.eat("->") {
            self.ty()?
        } else {
            Type::Tuple(Vec::new())
        };
        self.expect("{")?;
        self.bind_local_name(&name);
        for param in &params {
            self.shadow_callable(&param.name);
        }
        let body = self.value_scope()?.value;
        self.type_params.clear();
        self.dependent_types = false;
        Ok(Function {
            name,
            public,
            type_params,
            callback_params,
            value_parametrics,
            params,
            result,
            body,
        })
    }

    /// Parse a value formal and its optional braced default.
    fn value_parametric(&mut self, index: usize) -> Result<ParametricBinding, String> {
        // DSLX permits `self` as a value parametric even though a data
        // parameter named `self` requires an impl method.
        let name = self.ident()?;
        self.expect(":")?;
        self.bind_local_name(&name);
        let ty = self.ty()?;
        let default = if self.eat("=") {
            self.expect("{")?;
            let value = self.expr()?.value;
            self.expect("}")?;
            Some(value)
        } else {
            None
        };
        self.shadow_callable(&name);
        Ok(ParametricBinding {
            name,
            ty,
            default,
            index,
        })
    }

    /// Parse a named callable in a function or block's compile-time parameter
    /// list.
    fn callback_param(&mut self) -> Result<CallbackParam, String> {
        let name = self.take_identifier(is_use_identifier)?;
        self.bind_local_name(&name);
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
        let mut statements = Vec::new();
        let mut height = 0;
        let output = loop {
            if self.eat("}") {
                break ParsedExpr::leaf(Expr::Tuple(Vec::new()));
            }
            if self.eat("let") {
                let (binding, initializer_height) = self.let_binding(dslx_tuple_patterns)?;
                height = height.max(initializer_height);
                statements.push(ScopeStatement::Let(binding));
                self.expect(";")?;
            } else if self.eat("const_assert!") {
                let (assertion, predicate_height) = self.const_assert_predicate()?;
                height = height.max(predicate_height);
                statements.push(ScopeStatement::ConstAssert(assertion));
            } else {
                let value = self.expr()?;
                if self.eat("}") {
                    break value;
                }
                self.expect(";")?;
                height = height.max(value.height);
                statements.push(ScopeStatement::Expression(value.value));
            }
        };
        if statements.is_empty() {
            return Ok(output);
        }
        self.compose(
            height.max(output.height),
            Expr::Scope {
                statements,
                result: Box::new(output.value),
            },
        )
    }

    /// Parse the single predicate and terminating semicolon after
    /// `const_assert!`.
    fn const_assert_predicate(&mut self) -> Result<(ConstAssert, usize), String> {
        let position = self.position_at(self.tokens[self.pos - 1].start);
        self.expect("(")?;
        let predicate = self.expr()?;
        self.expect(")")?;
        self.expect(";")?;
        Ok((
            ConstAssert {
                predicate: predicate.value,
                position,
            },
            predicate.height,
        ))
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

    /// Read block type, value, and callable formals in one namespace.
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
            let annotation = (p.peek(1) == ":").then(|| p.peek(2));
            if !names.insert(token.text) {
                let kind = match annotation {
                    None | Some("type") => "type",
                    Some("fn") => "callable",
                    Some(_) => "value",
                };
                return Err(diagnostic(
                    p.source,
                    token.start,
                    &format!("duplicate block {kind} parameter '{}'", token.text),
                ));
            }
            match annotation {
                Some("fn") => params.callback_params.push(p.callback_param()?),
                None | Some("type") => {
                    let name = p.ident()?;
                    if p.eat(":") {
                        p.expect("type")?;
                    }
                    p.type_params.insert(token.text);
                    params.type_params.push(name);
                }
                Some(_) => {
                    let name = p.ident()?;
                    p.expect(":")?;
                    let ty = p.ty()?;
                    if !matches!(
                        ty,
                        Type::Bits {
                            signed: false,
                            width: 32
                        } | Type::Named(_)
                    ) {
                        return Err(p.error("block value parameters require type 'u32'"));
                    }
                    if p.peek(0) == "=" {
                        return Err(p.error("block value parameters require explicit arguments"));
                    }
                    params.value_parametrics.push(ParametricBinding {
                        name,
                        ty,
                        default: None,
                        index: params.value_parametrics.len() + params.callback_params.len(),
                    });
                }
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
        lemmas: Option<&mut Vec<InvariantLemma>>,
    ) -> Result<Block, String> {
        let name = self.ident()?;
        if lemmas.is_some() && self.peek(0) == "<" {
            return Err(self.error("BX invariants cannot have parametric bindings"));
        }
        let params = self.block_parametrics()?;
        let mut input_cycles = BTreeMap::new();
        let signature = self.block_signature(name, |parser| {
            let param = parser.param()?;
            if parser.eat("@") {
                input_cycles.insert(param.name.clone(), parser.cycle_number("input")?);
            }
            Ok(param)
        })?;
        if lemmas.is_some()
            && crate::typed::bits(&signature.result) != Some((false, 1))
            && !matches!(
                signature.result,
                Type::Named(_) | Type::ParametricBits { .. }
            )
        {
            return Err(diagnostic(
                self.source,
                self.tokens[self.pos - 1].start,
                &format!(
                    "BX invariant blocks must return bool, got {}",
                    signature.result
                ),
            ));
        }
        let output_cycles = self.output_cycles()?;
        self.expect("{")?;
        if lemmas.is_none() {
            // Only a design block introduces a callable hardware name.
            self.bind_local_name(&signature.name);
        }
        for name in &params.type_params {
            self.shadow_callable(name);
        }
        for formal in &params.callback_params {
            self.shadow_callable(&formal.name);
        }
        for formal in &params.value_parametrics {
            self.shadow_callable(&formal.name);
        }
        self.shadow_callable(&signature.clock);
        if let Some(reset) = &signature.reset {
            self.shadow_callable(reset);
        }
        for port in &signature.params {
            self.shadow_callable(&port.name);
        }
        self.dependent_types = !params.value_parametrics.is_empty();
        let body = self.with_name_scope(|p| p.block_body_contents(lemmas))?;
        self.dependent_types = false;
        self.type_params.clear();
        let (output, output_span) = body
            .output
            .ok_or_else(|| self.error("block requires an output expression"))?;
        let source = body
            .statements
            .iter()
            .any(|stmt| matches!(stmt, Stmt::Wire { .. }))
            .then(|| BlockSource {
                file: SourceFile {
                    text: self
                        .diagnostic_source
                        .get_or_insert_with(|| Arc::from(self.source))
                        .clone(),
                    path: None,
                },
                statements: body.statement_spans,
                output: output_span,
            });
        Ok(Block {
            name: signature.name,
            name_position: Some(name_position),
            source,
            allowed_warnings,
            public,
            type_params: params.type_params,
            callback_params: params.callback_params,
            value_parametrics: params.value_parametrics,
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
        let (signed, bit_count) = self.bit_type_expressions(Self::expr)?;
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
    fn bit_type_expressions(
        &mut self,
        nonliteral_dimension: fn(&mut Self) -> Result<ParsedExpr, String>,
    ) -> Result<(ParsedExpr, ParsedExpr), String> {
        let spelling = self.peek(0);
        self.pos += 1;
        let signed = if spelling == "xN" {
            self.expect("[")?;
            let expr = if matches!(self.peek(0), "true" | "false") && self.peek(1) == "]" {
                self.expr()?
            } else {
                nonliteral_dimension(self)?
            };
            self.expect("]")?;
            expr
        } else {
            ParsedExpr::leaf(Expr::Bool(spelling == "sN"))
        };
        self.expect("[")?;
        let bit_count = if self.plain_dimension(0) {
            self.expr()?
        } else {
            nonliteral_dimension(self)?
        };
        self.expect("]")?;
        Ok((signed, bit_count))
    }

    /// Nonliteral dimensions name constants. Templates can also
    /// compute dimensions from their value parametrics.
    fn type_dimension(&mut self) -> Result<ParsedExpr, String> {
        let start = self.tokens[self.pos].start;
        let mut dimension = if self.dependent_types {
            // Template expressions introduce their own scopes. Only a bare
            // dimension name here needs the lookup for an overwritten type.
            let previous = self.literal_type_context;
            if previous == LiteralTypeContext::Overwritten {
                self.literal_type_context = LiteralTypeContext::DiscardedExpression;
            }
            let expr = self.expr();
            self.literal_type_context = previous;
            expr?
        } else {
            ParsedExpr::leaf(Expr::Name(self.name_or_qualified_member()?))
        };
        if let Expr::Name(name) = &mut dimension.value {
            if self.literal_type_context == LiteralTypeContext::Overwritten {
                // DSLX resolves only the head of a path in a type that an
                // outer literal annotation overwrites.
                while self.eat("::") {
                    name.push_str("::");
                    name.push_str(&self.take_identifier(is_function_identifier)?);
                }
                let head = name
                    .split_once("::")
                    .map_or(name.as_str(), |(head, _)| head);
                let type_member = head != name.as_str()
                    && (self.visible_type_names.contains(head)
                        || crate::check::is_dslx_builtin_type(head));
                if !type_member
                    && !self.visible_names.contains(head)
                    && !self.visible_names.contains(name.as_str())
                    && !crate::check::is_dslx_builtin_value(head)
                {
                    return Err(diagnostic(
                        self.source,
                        start,
                        &format!("unknown name '{head}' in literal annotation"),
                    ));
                }
            }
            validate_dslx_value_path(name).map_err(|e| self.error(&e))?;
        }
        Ok(dimension)
    }

    fn parsed_ty(&mut self, singleton_tuples: bool) -> Result<ParsedType, String> {
        let mut ty = self.type_atom(singleton_tuples)?;
        while self.eat("[") {
            let element = Box::new(ty.value);
            ty.value = if !self.dslx_type_display && !self.plain_dimension(0) {
                let size = self.type_dimension()?;
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

    /// Read struct arguments using the declaration's parameter kind.
    fn struct_arguments(&mut self, name: &str) -> Result<StructArguments, String> {
        if self.type_structs.contains(name) {
            self.expect("<")?;
            let singleton_tuples = self.dslx_value_grammar();
            let mut height = 0;
            let args = self.nested(|p| {
                p.list(">", |p| {
                    let arg = p.parsed_ty(singleton_tuples)?;
                    height = height.max(arg.depth);
                    Ok(arg.value)
                })
            })?;
            return Ok(StructArguments::Types { args, height });
        }
        let arguments = self.parsed_angle_arguments()?;
        self.classify_struct_arguments(arguments)
    }

    /// Read explicit value arguments before their enclosing syntax is known.
    fn parsed_angle_arguments(&mut self) -> Result<ParsedAngleArguments, String> {
        self.expect("<")?;
        let mut starts = Vec::new();
        let mut height = 0;
        let args = self.list(">", |p| {
            starts.push(p.pos);
            let actual = p.parametric_argument()?;
            height = height.max(actual.height);
            Ok(actual.value)
        })?;
        Ok(ParsedAngleArguments {
            args,
            starts,
            height,
        })
    }

    /// Compact unannotated decimal arguments. Keep the radix of bit-pattern
    /// literals because a signed formal can accept `0xff` and reject `255`.
    fn classify_struct_arguments(
        &self,
        ParsedAngleArguments {
            args,
            starts,
            height,
        }: ParsedAngleArguments,
    ) -> Result<StructArguments, String> {
        let mut concrete = Vec::new();
        for (arg, start) in args.iter().zip(starts) {
            let start = self.tokens[start];
            if start.text == "{" {
                continue;
            }
            if let Expr::Name(name) = arg {
                validate_dslx_value_path(name)
                    .map_err(|e| diagnostic(self.source, start.start, &e))?;
                continue;
            }
            if let Expr::Literal { ty: None, text } = arg
                && let Ok(value) = text.parse::<u32>()
            {
                concrete.push(value);
            }
        }
        Ok(if concrete.len() == args.len() {
            StructArguments::Concrete(concrete)
        } else {
            StructArguments::Computed { args, height }
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
            if !self.dslx_type_display && !concrete {
                let (signed, width) = self.bit_type_expressions(Self::type_dimension)?;
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
            if spelling == "self"
                && self.literal_type_heads.contains(spelling)
                && !self.type_params.contains(spelling)
            {
                return Err(self.error("the keyword 'self' cannot refer to a type alias"));
            }
            if spelling == "_"
                && self.literal_type_heads.contains(spelling)
                && !self.type_params.contains(spelling)
                && !self.visible_literal_type(spelling, false)
            {
                return Err(self.error("'_' does not name a type in this scope"));
            }
            // A cast can be followed immediately by a less-than comparison.
            // Type parameters can name scalar cast targets, including
            // in a tuple of comparisons such as (x as T < 1, 2 > (y)).
            let after_as = self.pos > 0 && self.tokens[self.pos - 1].text == "as";
            let name = self.name_or_qualified_member()?;
            if self.literal_type_context == LiteralTypeContext::Overwritten {
                self.record_overwritten_type(&name)?;
            }
            if self.eat_turbofish()
                || (self.peek(0) == "<"
                    && (!after_as || self.struct_arguments_end(self.pos).is_some()))
            {
                match self.struct_arguments(&name)? {
                    StructArguments::Concrete(args) => Type::Applied { name, args },
                    StructArguments::Computed { args, height } => {
                        if height >= MAX_SYNTAX_DEPTH {
                            return Err(self.error("type nesting exceeds 64 levels"));
                        }
                        return Ok(ParsedType {
                            value: Type::ParametricStruct { name, args },
                            depth: height + 1,
                        });
                    }
                    StructArguments::Types { args, height } => {
                        if height >= MAX_SYNTAX_DEPTH {
                            return Err(self.error("type nesting exceeds 64 levels"));
                        }
                        return Ok(ParsedType {
                            value: Type::TypeApplied { name, args },
                            depth: height + 1,
                        });
                    }
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
        self.with_name_scope(|p| p.block_body_contents(None))
    }

    fn block_body_contents(
        &mut self,
        mut lemmas: Option<&mut Vec<InvariantLemma>>,
    ) -> Result<Body, String> {
        let mut statements = Vec::new();
        let mut statement_spans = Vec::new();
        while !self.eat("}") {
            let start = self.tokens[self.pos].start;
            // A DSLX use leaf can be named reg. A following name distinguishes
            // a register declaration from a call or constructor.
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
            } else if self.peek(0) == "#" && self.peek(1) == "[" && self.peek(2) == "lemma" {
                self.pos += 3;
                self.expect("]")?;
                let lemmas = lemmas.as_deref_mut().ok_or_else(|| {
                    self.error("#[lemma] requires the top level of an #[invariant] block")
                })?;
                self.expect("let")?;
                let position = self.position_at(self.tokens[self.pos].start);
                let (mut statement, _) = self.let_stmt(false)?;
                let Stmt::Let { name, ty, .. } = &mut statement else {
                    return Err(self.error("#[lemma] requires a single named let binding"));
                };
                ty.get_or_insert(Type::Bool);
                let observation = Stmt::Let {
                    name: InvariantLemma::observation_name(lemmas.len()),
                    ty: Some(Type::Bool),
                    value: Expr::Name(name.clone()),
                };
                lemmas.push(InvariantLemma {
                    name: name.clone(),
                    position,
                });
                statements.push(statement);
                statement_spans.push(self.span_since(start));
                statements.push(observation);
            } else if self.eat("let") {
                statements.push(self.let_stmt(/* allow_wire= */ true)?.0);
            } else if self.peek(0) == "inst" && is_identifier(self.peek(1)) && self.peek(2) == ":" {
                return Err(self.error("use `let name = child(args);` to instantiate a block"));
            } else if self.peek(1) == ":=" {
                let name = self.ident()?;
                self.expect(":=")?;
                statements.push(Stmt::Write {
                    name,
                    value: self.expr()?.value,
                });
            } else if self.peek(1) == "=" {
                let name = self.ident()?;
                self.expect("=")?;
                statements.push(Stmt::DriveWire {
                    name,
                    value: self.expr()?.value,
                });
            } else if self.peek(1) == "." && self.peek(3) == ":=" {
                return Err(self.error(
                    "pass child inputs as call arguments; use a deferred let for feedback",
                ));
            } else if self.eat("if") {
                let condition = self.condition()?;
                if self.arm_has_else() {
                    let then_value = self.nested(Self::expression_scope)?;
                    self.expect("else")?;
                    let expr = self.if_tail(condition, then_value)?;
                    let output = self.expr_tail(expr, 0, Braces::Struct)?;
                    let span = self.span_since(start);
                    self.expect("}")?;
                    return Ok(Body {
                        statements,
                        statement_spans,
                        output: Some((output, span)),
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
            } else {
                let output = self.expr()?;
                let span = self.span_since(start);
                self.expect("}")?;
                return Ok(Body {
                    statements,
                    statement_spans,
                    output: Some((output, span)),
                });
            }
            if !matches!(statements.last(), Some(Stmt::If { .. })) {
                self.expect(";")?;
            }
            statement_spans.push(self.span_since(start));
        }
        Ok(Body {
            statements,
            statement_spans,
            output: None,
        })
    }

    /// Cover the tokens consumed since the given source offset.
    fn span_since(&self, start: usize) -> SourceSpan {
        let last = self.tokens[self.pos - 1];
        SourceSpan {
            start,
            end: last.start + last.text.len(),
        }
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
    fn let_stmt(&mut self, allow_wire: bool) -> Result<(Stmt, usize), String> {
        let head = self.let_head(true)?;
        if allow_wire && self.peek(0) == ";" {
            return match (head.pattern, head.ty) {
                (BindingPattern::Name(name), Some(ty)) if name != "_" => {
                    self.shadow_callable(&name);
                    Ok((Stmt::Wire { name, ty }, 0))
                }
                _ => Err(self.error("deferred let requires a single name and an explicit type")),
            };
        }
        let (binding, height) = self.let_initializer(head)?;
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
        let head = self.let_head(dslx_tuple_patterns)?;
        self.let_initializer(head)
    }

    /// Parse a let pattern and its optional type without binding its names.
    fn let_head(&mut self, dslx_tuple_patterns: bool) -> Result<LetHead, String> {
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
        Ok(LetHead { pattern, ty, names })
    }

    /// Parse the initializer in the enclosing scope and introduce the names.
    fn let_initializer(&mut self, head: LetHead) -> Result<(LetBinding, usize), String> {
        self.expect("=")?;
        let value = self.expr()?;
        // DSLX's single-name let hides a type named '_'. A wildcard inside a
        // destructuring pattern discards the value without hiding the type.
        if self.literal_type_heads.contains("_")
            && matches!(&head.pattern, BindingPattern::Name(name) if name == "_")
        {
            self.shadow_callable("_");
        }
        for name in head.names {
            self.shadow_callable(&name);
        }
        Ok((
            LetBinding {
                pattern: head.pattern,
                ty: head.ty,
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
        let mut spellings = MatchPatternSpellings::default();
        let arms = self.list("}", |parser| {
            let mut patterns = vec![parser.match_arm_pattern(&mut spellings)?];
            while parser.eat("|") {
                patterns.push(parser.match_arm_pattern(&mut spellings)?);
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

    /// Detect repeated spellings even when the pattern has an empty domain.
    fn match_arm_pattern(
        &mut self,
        spellings: &mut MatchPatternSpellings,
    ) -> Result<MatchPattern, String> {
        let first = self.pos;
        spellings.current.clear();
        spellings.contains_range = false;
        let pattern = self.match_pattern(spellings)?;
        let tuple = matches!(pattern, MatchPattern::Tuple(_));
        if (tuple || spellings.contains_range) && !spellings.seen.insert(spellings.current.clone())
        {
            let message = if tuple {
                "exact-duplicate match pattern"
            } else {
                "duplicate range pattern"
            };
            return Err(diagnostic(self.source, self.tokens[first].start, message));
        }
        Ok(pattern)
    }

    /// Parse tuple patterns recursively, building a key for the full pattern.
    fn match_pattern(
        &mut self,
        spellings: &mut MatchPatternSpellings,
    ) -> Result<MatchPattern, String> {
        if self.eat("_") {
            spellings.current.push('_');
            return Ok(MatchPattern::Wildcard);
        }
        if self.eat("(") {
            spellings.current.push('(');
            return self.nested(|parser| {
                let fields = parser.list(")", |parser| {
                    let field = parser.match_pattern(spellings)?;
                    spellings.current.push(',');
                    Ok(field)
                })?;
                spellings.current.push(')');
                Ok(MatchPattern::Tuple(fields))
            });
        }
        let first = self.pos;
        let mut end = 1;
        while end < 5 && self.peek(end) == "::" && is_function_identifier(self.peek(end + 1)) {
            end += 2;
        }
        if is_function_identifier(self.peek(0)) && matches!(self.peek(end), "=>" | "|" | "," | ")")
        {
            // The checker resolves a scalar constant, value parametric, or enum
            // member. DSLX requires numeric literals on both sides of a range.
            let name = self.name_or_qualified_member()?;
            let value = self.named_value(name)?;
            spellings
                .current
                .extend(self.tokens[first..self.pos].iter().map(|token| token.text));
            return Ok(MatchPattern::Literal(value));
        }
        if self.named_literal_type_at(0) && !self.type_params.contains(self.peek(0)) {
            return Err(self.error("a match pattern cannot start with an alias-annotated literal"));
        }
        let start = self.pattern_literal()?;
        append_literal_pattern_spelling(&mut spellings.current, &self.tokens[first..self.pos]);
        let separator = self.pos;
        let inclusive = self.eat("..=");
        if inclusive || self.eat("..") {
            let end = self.pattern_literal()?;
            spellings.contains_range = true;
            spellings.current.push_str(self.tokens[separator].text);
            append_literal_pattern_spelling(
                &mut spellings.current,
                &self.tokens[separator + 1..self.pos],
            );
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
            let ty = self.parsed_ty(self.dslx_value_grammar())?;
            if !matches!(
                ty.value,
                Type::Bool | Type::Bits { .. } | Type::ParametricBits { .. } | Type::Named(_)
            ) {
                return Err(self.error("match patterns require a scalar literal type"));
            }
            self.expect(":")?;
            self.annotated_literal(ty)?
        } else if number_token(self.peek(0)) || self.tokens[self.pos].character.is_some() {
            self.literal(None)?
        } else {
            return Err(self.error("expected a scalar match pattern (literal, range, or _)"));
        };
        if !matches!(&literal.value, Expr::Literal { .. }) {
            return Err(self.error("match patterns require a scalar literal"));
        }
        Ok(literal.value)
    }

    /// In a match subject, distinguish a struct constructor from the pattern
    /// list by what follows the opening brace.
    fn struct_braces(&self, braces: Braces) -> bool {
        match braces {
            Braces::Struct => true,
            Braces::Body => false,
            Braces::MatchArms => {
                let first = self.peek(1);
                let next = self.peek(2);
                if first == ".." {
                    return true;
                }
                if first == "}" {
                    return matches!(next, "." | "[" | "as" | "{")
                        || BinaryOp::from_spelling(next).is_some();
                }
                let field = is_function_identifier(first)
                    || !self.dslx_value_grammar() && is_identifier(first);
                if next == ":" && self.named_literal_type_at(1) {
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
    fn loop_annotation<'n>(
        &mut self,
        binders: impl IntoIterator<Item = &'n str>,
    ) -> Result<Option<(Type, Type)>, String> {
        if !self.eat(":") {
            return Ok(None);
        }
        // The names are in scope while parsing the binder annotation. The
        // iterable and seed are parsed in the enclosing scope.
        self.with_name_scope(|p| {
            for name in binders {
                p.bind_local_name(name);
            }
            p.expect("(")?;
            let element = p.ty()?;
            p.expect(",")?;
            let accumulator = p.ty()?;
            p.eat(",");
            p.expect(")")?;
            Ok(Some((element, accumulator)))
        })
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
        let annotation = self.loop_annotation(names.iter().map(String::as_str))?;
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
        let start = if self.peek(0) == "(" && !self.starts_array_literal_type() && self.eat("(") {
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
        let annotation = self.loop_annotation([index.as_str(), accumulator.as_str()])?;
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
                let (stmt, initializer_height) = p.let_stmt(/* allow_wire= */ false)?;
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
            if left.spelling == ExprSpelling::AnnotatedLiteral
                && matches!(left.value, Expr::ArrayLiteral { .. })
                && matches!(self.peek(0), "." | "[")
            {
                return Err(self
                    .error("parenthesize a typed array literal before field access or indexing"));
            }
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
        if self.eat("[") {
            return self.array_literal(None);
        }
        if self.starts_bit_literal()? || self.starts_array_literal_type() {
            return self.parse_annotated_literal();
        }
        if self.dslx_value_grammar()
            && text == "_"
            && !(self.peek(1) == "{" && self.visible_literal_type(text, false))
        {
            return Err(self.error("wildcard pattern '_' cannot be used as a reference"));
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
        let angle_offset = if self.peek(1) == "::" { 2 } else { 1 };
        if self.peek(angle_offset) == "<" {
            let angle = self.pos + angle_offset;
            if text == "zero!" && !self.zero_parametric_call(angle) {
                self.pos = angle + 1;
                let ty = self.ty()?;
                self.expect(">")?;
                self.expect("(")?;
                self.expect(")")?;
                return Ok(ParsedExpr::leaf(Expr::Zero(ty)));
            }
            if self.type_builtin_at(self.pos) {
                return self.type_builtin_call(angle);
            }
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
        if number_token(text) || self.tokens[self.pos].character.is_some() {
            return self.literal(None);
        }
        if text.starts_with('"') {
            let bytes = decode_byte_string(text)
                .map_err(|error| diagnostic(self.source, self.tokens[self.pos].start, error))?;
            if bytes.is_empty() {
                return Err(diagnostic(
                    self.source,
                    self.tokens[self.pos].start,
                    "zero-length DSLX strings are not supported",
                ));
            }
            self.pos += 1;
            return Ok(ParsedExpr::leaf(Expr::ByteString(bytes)));
        }
        let name = self.name_or_qualified_member()?;
        let turbofish = self.eat_turbofish();
        let type_struct = self.type_structs.contains(&name);
        let typed_struct_args = (turbofish && type_struct)
            .then(|| self.struct_arguments(&name))
            .transpose()?;
        let mut explicit = (turbofish && !type_struct)
            .then(|| self.parsed_angle_arguments())
            .transpose()?;
        if self.peek(0) == "("
            && let Some(arguments) = explicit.take()
        {
            return self.parametric_call(name, arguments.args, arguments.height);
        }
        if !turbofish && self.parametric_arguments_end(self.pos, &["("]).is_some() {
            self.expect("<")?;
            let mut height = 0;
            let parametrics = self.parametric_arguments(&mut height)?;
            return self.parametric_call(name, parametrics, height);
        }
        if typed_struct_args.is_none() && self.eat("(") {
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
        let struct_args = if typed_struct_args.is_some() {
            if self.peek(0) != "{" {
                return Err(self.error("expected '{' after struct type arguments"));
            }
            typed_struct_args
        } else if let Some(arguments) = explicit {
            if self.peek(0) != "{" {
                return Err(self.error("expected '(' or '{' after explicit value arguments"));
            }
            Some(self.classify_struct_arguments(arguments)?)
        } else if self.struct_arguments_end(self.pos).is_some_and(|end| {
            (self.tokens[end].text == ">" || type_struct && self.tokens[end].text == ">>")
                && self.tokens[end + 1].text == "{"
        }) {
            Some(self.struct_arguments(&name)?)
        } else {
            None
        };
        if self.peek(0) == "{" && self.struct_braces(braces) {
            if name == "self" && self.literal_type_heads.contains("self") {
                return Err(self.error("the keyword 'self' cannot introduce a struct literal"));
            }
            self.pos += 1;
            let mut height = match &struct_args {
                Some(
                    StructArguments::Computed { height, .. }
                    | StructArguments::Types { height, .. },
                ) => *height,
                _ => 0,
            };
            let StructMembers { fields, base } = self.struct_members(&mut height)?;
            let value = match struct_args {
                Some(StructArguments::Concrete(args)) => Expr::AppliedStruct {
                    name,
                    args,
                    fields,
                    base,
                },
                Some(StructArguments::Computed { args, .. }) => Expr::ParametricStruct {
                    name,
                    args,
                    fields,
                    base,
                },
                Some(StructArguments::Types { args, .. }) => Expr::TypeAppliedStruct {
                    name,
                    args,
                    fields,
                    base,
                },
                None => Expr::Struct { name, fields, base },
            };
            return self.compose(height, value);
        }
        if struct_args.is_some() {
            return Err(self.error("parenthesize the struct literal in this expression"));
        }
        Ok(ParsedExpr::leaf(self.named_value(name)?))
    }

    /// Parse fields and an optional DSLX-style splat before the closing brace.
    fn struct_members(&mut self, height: &mut usize) -> Result<StructMembers, String> {
        let mut fields = Vec::new();
        while !self.eat("}") {
            if self.eat("..") {
                let position = self.position_at(self.tokens[self.pos - 1].start);
                let base = self.expr()?;
                *height = (*height).max(base.height);
                self.expect("}")?;
                return Ok(StructMembers {
                    fields,
                    base: Some(Box::new(StructSplat {
                        value: base.value,
                        position: Some(position),
                    })),
                });
            }
            let field = self.take_identifier(is_dslx_member_name)?;
            let value = if self.eat(":") {
                self.expr()?
            } else {
                ParsedExpr::leaf(Expr::Name(field.clone()))
            };
            *height = (*height).max(value.height);
            fields.push((field, value.value));
            if self.eat("}") {
                break;
            }
            self.expect(",")?;
        }
        Ok(StructMembers { fields, base: None })
    }

    /// Parse the runtime arguments of a call with explicit value arguments.
    fn parametric_call(
        &mut self,
        name: String,
        parametrics: Vec<Expr>,
        mut height: usize,
    ) -> Result<ParsedExpr, String> {
        self.expect("(")?;
        let args = self.list(")", |p| {
            let arg = p.expr()?;
            height = height.max(arg.height);
            Ok(arg.value)
        })?;
        self.compose(
            height,
            Expr::ParametricCall {
                name,
                parametrics,
                args,
            },
        )
    }

    /// A lexical binding or callable template can shadow a typed builtin.
    fn type_builtin_at(&self, name: usize) -> bool {
        let token = self.tokens[name];
        matches!(
            token.text,
            "all_ones!"
                | "bit_count"
                | "element_count"
                | "decode"
                | "widening_cast"
                | "checked_cast"
        ) && !self.visible_names.contains(token.text)
            && !self.visible_type_names.contains(token.text)
            && !self
                .callable_templates
                .get(token.text)
                .and_then(|binding| binding.direct)
                .is_some_and(|start| {
                    self.value_context != ValueContext::DslxItem || start <= token.start
                })
    }

    /// Parse the type list and value arguments of a typed builtin.
    fn type_builtin_call(&mut self, angle: usize) -> Result<ParsedExpr, String> {
        let name = self.peek(0).to_owned();
        self.pos = angle + 1;
        let mut height = 0;
        let types = self.list(">", |parser| {
            let ty = parser.parsed_ty(true)?;
            height = height.max(ty.depth);
            Ok(ty.value)
        })?;
        self.expect("(")?;
        let args = self.list(")", |parser| {
            let arg = parser.expr()?;
            height = height.max(arg.height);
            Ok(arg.value)
        })?;
        self.compose(height, Expr::TypeBuiltin { name, types, args })
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
        if self.starts_bit_literal()? || self.starts_array_literal_type() {
            return self.nested(Self::parse_annotated_literal);
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
        if self.turbofish_at(angle) {
            let end = self.value_arguments_end(
                angle, /* allow_comma= */ true, /* allow_combined_close= */ false,
            )?;
            return following
                .contains(&self.tokens[end + 1].text)
                .then_some(end);
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
        let end = self.value_arguments_end(angle, allow_comma, false)?;
        following
            .contains(&self.tokens[end + 1].text)
            .then_some(end)
    }

    /// Find the first top-level closing angle without scanning into the
    /// enclosing expression.
    fn value_arguments_end(
        &self,
        angle: usize,
        allow_comma: bool,
        allow_combined_close: bool,
    ) -> Option<usize> {
        let mut depth = 0usize;
        for index in angle + 1..self.tokens.len() {
            let token = &self.tokens[index];
            match token.text {
                "" => return None,
                ">" if depth == 0 => return Some(index),
                ">>" | ">=" if depth == 0 && allow_combined_close => return Some(index),
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
        let type_struct = self.type_struct_before(angle);
        let end = if type_struct {
            self.type_arguments_end(angle)?
        } else {
            self.value_arguments_end(
                angle, /* allow_comma= */ true, /* allow_combined_close= */ true,
            )?
        };
        let close = self.tokens[end].text;
        let next = self.tokens[end + 1].text;
        if next == "[" && self.dimensions_before_colon(end + 1) {
            // An annotated array can start with a struct application whose
            // arguments have their own commas, e.g. Pair<A, B>[2]:[p, ...].
            return Some(end);
        }
        let (before_name, bare_name) = match &self.tokens[..angle] {
            [before @ .., _, separator, _] if separator.text == "::" => (before, None),
            [before @ .., name] => (before, Some(name.text)),
            [] => return None,
        };
        if (close == ">" || type_struct && close == ">>") && next == "{" {
            // Imports have template placeholders even when they could name a
            // type. A known bare value cannot start a symbolic constructor.
            let known_value = bare_name.is_some_and(|name| {
                self.visible_names.contains(name) && !self.callable_templates.contains_key(name)
            });
            if !known_value
                || self.tokens[angle + 1..end].iter().all(|token| {
                    number_token(token.text) || matches!(token.text, "u32" | ":" | ",")
                })
            {
                return Some(end);
            }
        }
        // Types directly inside zero! or following 'as' can have commas at
        // expression nesting depth zero. Tuples and arrays delimit their own
        // contents in expression_lookahead.
        match before_name {
            [.., zero, Token { text: "::", .. }, open] | [.., zero, open]
                if zero.text == "zero!" && open.text == "<" =>
            {
                Some(end)
            }
            [.., cast]
                if cast.text == "as"
                    && !bare_name.is_some_and(|name| self.type_params.contains(name))
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

    /// Whether this angle follows a known type-parametric struct name.
    fn type_struct_before(&self, angle: usize) -> bool {
        let end = angle - usize::from(self.turbofish_at(angle));
        let visible_type = |name: &str| {
            !self.visible_names.contains(name) || self.callable_templates.contains_key(name)
        };
        match &self.tokens[..end] {
            [.., head, separator, member] if separator.text == "::" => {
                visible_type(head.text)
                    && self
                        .type_structs
                        .contains(&format!("{}::{}", head.text, member.text))
            }
            [.., name] => visible_type(name.text) && self.type_structs.contains(name.text),
            [] => false,
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
                .type_builtin_arguments_end(index)
                .or_else(|| {
                    self.turbofish_at(index)
                        .then(|| {
                            if self.type_struct_before(index) {
                                self.type_arguments_end(index)
                            } else {
                                self.value_arguments_end(
                                    index, /* allow_comma= */ true,
                                    /* allow_combined_close= */ true,
                                )
                            }
                        })
                        .flatten()
                })
                .or_else(|| self.parametric_arguments_end(index, &["("]))
                .or_else(|| self.struct_arguments_end(index))
                .unwrap_or(index)
                + 1;
            Some((offset, token))
        })
    }

    /// Skip a typed builtin's annotation list when looking for outer commas.
    fn type_builtin_arguments_end(&self, angle: usize) -> Option<usize> {
        let name = angle.checked_sub(if self.turbofish_at(angle) { 2 } else { 1 })?;
        if self.tokens[angle].text != "<"
            || !self.type_builtin_at(name)
            || (name > 0 && self.tokens[name - 1].text == "::")
        {
            return None;
        }
        let end = self.type_arguments_end(angle)?;
        (self.tokens[end + 1].text == "(").then_some(end)
    }

    /// Find the closing angle of a type argument list, including nested types.
    fn type_arguments_end(&self, angle: usize) -> Option<usize> {
        let mut angles = 1usize;
        let mut groups = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(angle + 1) {
            match token.text {
                "(" | "[" | "{" => groups += 1,
                ")" | "]" | "}" => groups = groups.checked_sub(1)?,
                "<" if groups == 0 => angles += 1,
                close @ (">" | ">>") if groups == 0 => {
                    angles = angles.checked_sub(if close == ">>" { 2 } else { 1 })?;
                    if angles == 0 {
                        return Some(index);
                    }
                }
                "" | ";" if groups == 0 => break,
                _ => { /* Other tokens do not delimit the type argument list. */ }
            }
        }
        None
    }

    /// Distinguish the typed zero intrinsic from a visible callback template
    /// with the same name. Without import hints, preserve the intrinsic for a
    /// single type argument and no value arguments.
    fn zero_parametric_call(&self, angle: usize) -> bool {
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
        let position_start = self.position_cursor;
        let ordinary = self.list(")", Self::expr);
        self.block_map_shadowed = false;
        let ordinary_end = self.pos;
        let ordinary_position = self.position_cursor;
        for &(index, token) in &split_candidates {
            self.tokens[index] = token;
        }
        self.pos = start;
        self.position_cursor = position_start;
        self.forward_map_import = false;
        let builtin = self.map_call(target.clone(), false, ordinary.is_err());
        self.forward_map_import = true;
        match ordinary {
            Ok(args) => {
                self.pos = ordinary_end;
                self.position_cursor = ordinary_position;
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
        if self.peek(offset + 1) == "::" && self.peek(offset + 2) != "<" {
            offset += 2;
        }
        let angle = offset + if self.peek(offset + 1) == "::" { 2 } else { 1 };
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
        self.eat_turbofish();
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
        let outer_angles = self.saved_angle_tokens.take();
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
        self.saved_angle_tokens = outer_angles;
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
        let (parameter_type, result_type, body) = self.with_name_scope(|p| {
            p.bind_local_name(&parameter);
            let parameter_type = if p.eat(":") { Some(p.ty()?) } else { None };
            p.eat(",");
            p.expect("|")?;
            let result_type = if p.eat("->") { Some(p.ty()?) } else { None };
            p.hide_pattern_name(&parameter);
            let body = if p.eat("{") {
                p.value_scope()?
            } else if result_type.is_some() {
                return Err(p.error("lambda with an explicit result type requires a braced body"));
            } else {
                p.expr()?
            };
            Ok((parameter_type, result_type, body))
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

    /// Recognize a type formal, module alias or imported name before a literal
    /// colon.
    fn named_literal_type_at(&self, offset: usize) -> bool {
        let head = self.peek(offset);
        let bare = self.peek(offset + 1) == ":";
        if bare && self.type_params.contains(head) {
            return true;
        }
        let qualified = self.peek(offset + 1) == "::" && self.peek(offset + 3) == ":";
        (bare || qualified) && self.visible_literal_type(head, qualified)
    }

    /// A lexical binding can hide a module alias or an imported type name.
    fn visible_literal_type(&self, head: &str, qualified: bool) -> bool {
        if !self.literal_type_heads.contains(head) {
            return false;
        }
        let allow_forward = self.value_context != ValueContext::DslxItem;
        if self.callable_templates.contains_key(head) {
            // All use bindings have a template entry until a local shadows
            // them.
            allow_forward || self.visible_names.contains(head)
        } else {
            !qualified
                && !self.visible_names.contains(head)
                && (allow_forward || self.visible_type_names.contains(head))
        }
    }

    /// Retain module references from each type atom, including the elements
    /// of discarded arrays and tuples. Lexical binders are resolved here.
    fn record_overwritten_type(&mut self, name: &str) -> Result<(), String> {
        let (head, qualified) = name
            .split_once("::")
            .map_or((name, false), |(h, _)| (h, true));
        if self.type_params.contains(head) || crate::check::is_dslx_builtin_type(head) {
            return Ok(());
        }
        if !self.visible_literal_type(head, qualified) {
            // A qualified reference only needs its head to be bound. A bare
            // name must be a type and a local value can shadow a module type.
            if qualified && crate::check::is_dslx_builtin_value(head) {
                return Ok(());
            }
            if self.visible_names.contains(head) {
                return if qualified {
                    Ok(())
                } else {
                    Err(self.error(&format!(
                        "'{head}' does not name a type in literal annotation"
                    )))
                };
            }
            if !self.visible_type_names.contains(head)
                && self.value_context == ValueContext::DslxItem
            {
                return Err(self.error(&format!("unknown name '{head}' in literal annotation")));
            }
        }
        self.overwritten_literal_types.insert(name.to_owned());
        Ok(())
    }

    /// Recognize a bit literal's annotation and suggest the DSLX spelling
    /// when a formerly accepted shorthand is followed by a colon.
    fn starts_bit_literal(&self) -> Result<bool, String> {
        let name = self.peek(0);
        if matches!(name, "bits" | "uN" | "sN" | "xN" | "bool")
            || fixed_width(name).is_some()
            || self.named_literal_type_at(0)
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

    /// Skip a delimited type atom or dimension during literal lookahead.
    fn after_group(&self, start: usize, open: &str, close: &str) -> Option<usize> {
        let mut depth = 1;
        for (index, token) in self.tokens.iter().enumerate().skip(start + 1) {
            if token.text == open {
                depth += 1;
            } else if token.text == close {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
            }
        }
        None
    }

    /// The colon distinguishes an array type's dimensions from indexing.
    fn dimensions_before_colon(&self, mut index: usize) -> bool {
        while self.tokens[index].text == "[" {
            let Some(after) = self.after_group(index, "[", "]") else {
                return false;
            };
            index = after;
        }
        self.tokens[index].text == ":"
    }

    /// A tuple, nominal type or type parameter can start a typed array, but
    /// the same opening tokens also occur in ordinary values and subscripts.
    fn starts_array_literal_type(&self) -> bool {
        let mut index = self.pos;
        if self.peek(0) == "(" {
            let Some(after) = self.after_group(index, "(", ")") else {
                return false;
            };
            index = after;
        } else if is_value_identifier(self.peek(0)) {
            index += 1;
            if self.tokens[index].text == "::" && self.tokens[index + 1].text != "<" {
                index += 2;
            }
            if self.tokens.get(index).is_some_and(|t| t.text == "::")
                && self.tokens[index + 1].text == "<"
            {
                index += 1;
            }
            if self.tokens.get(index).is_some_and(|t| t.text == "<") {
                let Some(end) = self.struct_arguments_end(index) else {
                    return false;
                };
                index = end + 1;
            }
        } else {
            return false;
        }
        match self.tokens.get(index).map(|t| t.text) {
            Some("[") => self.dimensions_before_colon(index),
            // A type parameter can itself denote an array.
            Some(":") => self.tokens[index + 1..]
                .iter()
                .find(|t| t.text != "(")
                .is_some_and(|t| t.text == "["),
            _ => false,
        }
    }

    fn parse_annotated_literal(&mut self) -> Result<ParsedExpr, String> {
        let ty = self.parsed_ty(self.dslx_value_grammar())?;
        self.expect(":")?;
        self.annotated_literal(ty)
    }

    /// Parse explicit array members once, retaining an optional trailing fill.
    fn array_literal(&mut self, annotation: Option<ParsedType>) -> Result<ParsedExpr, String> {
        let mut height = annotation.as_ref().map_or(0, |ty| ty.depth);
        let mut members = Vec::new();
        let mut has_ellipsis = false;
        while !self.eat("]") {
            if self.eat("...") {
                if members.is_empty() {
                    return Err(self.error("array ellipsis requires a preceding member"));
                }
                has_ellipsis = true;
                self.eat(",");
                self.expect("]")?;
                break;
            }
            let member = self.expr()?;
            height = height.max(member.height);
            members.push(member.value);
            if self.eat("]") {
                break;
            }
            self.expect(",")?;
        }
        let annotated = annotation.is_some();
        let value = if annotated || has_ellipsis {
            Expr::ArrayLiteral {
                annotation: annotation.map(|ty| ty.value),
                members,
                has_ellipsis,
            }
        } else {
            Expr::Array(members)
        };
        let mut literal = self.compose(height, value)?;
        if annotated {
            literal.spelling = ExprSpelling::AnnotatedLiteral;
        }
        Ok(literal)
    }

    /// Parse a literal term after a type annotation. Parentheses group the
    /// literal, and the outermost annotation supplies its final type.
    fn annotated_literal(&mut self, ty: ParsedType) -> Result<ParsedExpr, String> {
        if self.eat("[") {
            return self.array_literal(Some(ty));
        }
        let compound_type = self.starts_array_literal_type();
        if !compound_type && self.eat("(") {
            let literal = self.nested(|parser| parser.annotated_literal(ty))?;
            self.expect(")")?;
            return Ok(literal);
        }
        if self.starts_bit_literal()? || compound_type {
            // DSLX resolves the head of each name as it parses this type.
            // The outer annotation subsequently replaces it, so its members,
            // signedness and widths never undergo type checking.
            let previous = self.literal_type_context;
            if previous == LiteralTypeContext::Live {
                self.literal_type_context = LiteralTypeContext::Overwritten;
            }
            let inner = self.ty();
            self.literal_type_context = previous;
            inner?;
            self.expect(":")?;
            return self.nested(|parser| parser.annotated_literal(ty));
        }
        self.literal(Some(ty.value))
    }

    /// Preserve arbitrary-width numbers as text for checking against their
    /// type.
    fn literal(&mut self, ty: Option<Type>) -> Result<ParsedExpr, String> {
        let annotated = ty.is_some();
        let spelling = self.peek(0);
        let (ty, text) =
            if self.tokens[self.pos].character.is_some() || matches!(spelling, "true" | "false") {
                let ty = ty.unwrap_or(if self.tokens[self.pos].character.is_some() {
                    Type::unsigned(8)
                } else {
                    Type::Bool
                });
                let text = if matches!(&ty, Type::ParametricBits { .. } | Type::Named(_)) {
                    // Check the intrinsic width once the annotation resolves.
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
fn append_literal_pattern_spelling(result: &mut String, tokens: &[Token]) {
    let parenthesized = tokens.iter().any(|t| t.text == "(");
    if parenthesized {
        result.push('(');
    }
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
    if parenthesized {
        result.push(')');
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
    use crate::ast::FunctionParametricRef;

    fn names<const N: usize>(names: [&str; N]) -> [Expr; N] {
        names.map(|name| Expr::Name(name.to_owned()))
    }

    #[test]
    fn typed_builtins_keep_aggregate_annotations_and_shadowed_comparisons() -> Result<(), String> {
        let module = parse_module(
            r#"
fn custom<N: u32>(x: u8) -> u8 { x + (N as u8) }
fn widths(x: u8) -> (u32, s16) {
    (bit_count<(u8, lib::Packet<2, 3>)>(), widening_cast<s16, u8>(x))
}
fn shadow(bit_count: u8, x: u8) -> bool { bit_count < x }
fn decode<N: u32>(x: u8) -> u8 { custom<N>(x) }
fn local(x: u8) -> u8 { decode<1>(x) }
block mapped(clk) -> lib::Packet<2, 3>[2] {
    map(all_ones!<lib::Packet<2, 3>[2]>(), |x| x)
}
"#,
        )?;
        let Expr::Tuple(values) = &module.functions[1].body else {
            panic!("expected a tuple of builtins");
        };
        let Expr::TypeBuiltin { name, types, args } = &values[0] else {
            panic!("expected a typed builtin");
        };
        assert_eq!(name, "bit_count");
        assert!(args.is_empty());
        let [Type::Tuple(fields)] = types.as_slice() else {
            panic!("expected a tuple type argument");
        };
        assert_eq!(
            fields[1],
            Type::Applied {
                name: "lib::Packet".into(),
                args: vec![2, 3]
            }
        );
        assert!(matches!(&values[1], Expr::TypeBuiltin { name, types, args }
            if name == "widening_cast" && types.len() == 2 && args == &names(["x"])));
        assert!(matches!(
            module.functions[2].body,
            Expr::Binary {
                op: BinaryOp::Lt,
                ..
            }
        ));
        assert!(
            matches!(&module.functions[4].body, Expr::ParametricCall { name, .. } if name == "decode")
        );
        assert!(matches!(module.blocks[0].output, Expr::Map(_)));
        Ok(())
    }

    #[test]
    fn strings_preserve_escaped_and_utf8_bytes() -> Result<(), String> {
        let module = parse_module(
            r#"
const STRING = "é,\n\x00\"\u{1F410}";
block top(clk) -> u8 { configured_value_or("a\\b", u8:0) }
"#,
        )?;
        assert_eq!(
            module.constants[0].value,
            Expr::ByteString("é,\n\0\"🐐".as_bytes().to_vec())
        );
        let Expr::Call { args, .. } = &module.blocks[0].output else {
            panic!("expected a call");
        };
        assert_eq!(args[0], Expr::ByteString(b"a\\b".to_vec()));
        assert!(parse_module(r#"const BAD = "\x1";"#).is_err());
        assert!(parse_module(r#"const EMPTY = "";"#).is_err());
        Ok(())
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
    fn module_attributes_record_each_requirement_independently() -> Result<(), String> {
        let default = parse_module("const X = u8:0;")?;
        assert!(!default.require_define_before_use);
        assert!(!default.require_block_timing);
        assert!(!default.allow_block_reset_signal);
        let order = parse_module("#![require_define_before_use]")?;
        assert!(order.require_define_before_use);
        assert!(!order.require_block_timing);
        let timing = parse_module("#![require_block_timing]")?;
        assert!(!timing.require_define_before_use);
        assert!(timing.require_block_timing);
        for attrs in [
            "#![allow(block_reset_signal)]\n#![require_define_before_use]\n#![require_block_timing]",
            "#![require_block_timing]\n#![require_define_before_use]\n#![allow(block_reset_signal,)]",
        ] {
            let module = parse_module(&format!("{attrs}\nuse lib;"))?;
            assert!(module.require_define_before_use);
            assert!(module.require_block_timing);
            assert!(module.allow_block_reset_signal);
            assert_eq!(module.declaration_order, ["lib"]);
        }
        Ok(())
    }

    #[test]
    fn block_reset_signal_allowance_requires_a_module_header() {
        for (source, diagnostic) in [
            (
                "use lib;\n#![allow(block_reset_signal)]",
                "2:4: place #![allow(block_reset_signal)] before the first declaration, found 'allow'",
            ),
            (
                "#[allow(block_reset_signal)] block b(clk, rst) -> bool { rst }",
                "1:9: use #![allow(block_reset_signal)] at module scope, found 'block_reset_signal'",
            ),
            (
                "#![allow(other_signal)]",
                "1:10: expected 'block_reset_signal', found 'other_signal'",
            ),
        ] {
            assert_eq!(parse_module(source).unwrap_err(), diagnostic);
        }
    }

    #[test]
    fn define_before_use_attribute_requires_module_header() {
        for (source, expected) in [
            (
                "const X = u8:0;\n#![require_define_before_use]",
                "2:4: place #![require_define_before_use] before the first declaration, found 'require_define_before_use'",
            ),
            (
                "use lib;\n#![require_define_before_use]",
                "2:4: place #![require_define_before_use] before the first declaration, found 'require_define_before_use'",
            ),
            (
                "#[require_define_before_use] const X = u8:0;",
                "1:3: use #![require_define_before_use] at module scope, found 'require_define_before_use'",
            ),
            (
                "block b(clk) -> u8 { #![require_define_before_use] u8:0 }",
                "1:22: write #![...] at the top level of the BX file",
            ),
        ] {
            assert_eq!(parse_module(source).unwrap_err(), expected);
        }
    }

    #[test]
    fn define_before_use_attribute_preserves_call_syntax() -> Result<(), String> {
        let source = r#"
block before(clk) -> (u32, u8, u8) {
    (bit_count<1>(), zero!<2>(), map(u8:1, u8:2))
}
fn bit_count<N: u32>() -> u32 { N }
fn zero!<N: u32>() -> u8 { N as u8 }
fn map(x: u8, y: u8) -> u8 { x + y }
"#;
        let plain = parse_module(source)?;
        let ordered = parse_module(&format!("#![require_define_before_use]\n{source}"))?;
        assert_eq!(ordered.blocks[0].output, plain.blocks[0].output);
        Ok(())
    }

    #[test]
    fn warning_attributes_reject_unknown_names_and_misplaced_allowances() {
        for (source, diagnostic) in [
            (
                "#[require_block_timing] block b(clk) -> u8 { u8:0 }",
                "use #![require_block_timing] at module scope",
            ),
            (
                "block b(clk) -> u8 { #![require_block_timing] u8:0 }",
                "write #![...] at the top level of the BX file",
            ),
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
                "#[allow(useless_struct_splat, missing_block_timing)] const X = u8:0;",
                "expected a block definition after #[allow(...)]",
            ),
            (
                "#[allow(useless_struct_splat)] use lib;",
                "a use declaration cannot have a warning allowance",
            ),
            (
                "pub #[allow(combinational_block)] block b(clk) -> u8 { u8:0 }",
                "place a BX warning attribute before the declaration",
            ),
            (
                "block b(clk) -> u8 { #[allow(combinational_block)] u8:0 }",
                "place a BX warning attribute before the declaration",
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
    fn type_aliases_retain_visibility_order_and_rhs_syntax() -> Result<(), String> {
        let module = parse_module(
            r#"
const WIDTH = u32:4;
type Word!=uN[WIDTH + u32:1];
pub type Pair' = (Word!, lib::Lane)[2];
"#,
        )?;
        assert_eq!(module.declaration_order, ["WIDTH", "Word!", "Pair'"]);
        let word = &module.type_aliases[0];
        assert_eq!(word.name, "Word!");
        assert!(!word.public);
        assert!(matches!(&word.ty, Type::ParametricBits { width, .. }
            if matches!(width.as_ref(), Expr::Binary { op: BinaryOp::Add, .. })));
        assert_eq!(
            module.type_aliases[1],
            TypeAlias {
                name: "Pair'".into(),
                public: true,
                ty: Type::Array {
                    element: Box::new(Type::Tuple(vec![
                        Type::Named("Word!".into()),
                        Type::Named("lib::Lane".into()),
                    ])),
                    size: 2,
                },
            }
        );
        for malformed in [
            "type Word u8;",
            "type Word = ;",
            "type Word = u8",
            "pub type Word<T> = T;",
            "type Word<N:u32> = uN[N];",
            "type u8 = bool;",
            "fn local() { type Word = u8; }",
        ] {
            assert!(parse_module(malformed).is_err(), "{malformed}");
        }
        Ok(())
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
        assert!(matches!(&pair.value, Expr::Scope { statements, .. } if statements.len() == 1));
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
    fn function_type_parameters_are_visible_in_signatures_defaults_and_bodies() -> Result<(), String>
    {
        let module = parse_module(
            r#"
fn combine<A, F: fn(A, B) -> R, B: type, N: R = { R:0 }, R>(x: A, y: B) -> R {
    let result: R = F(x, y);
    match result { R:0 => zero!<R>(), _ => result }
}
fn apply(x:u8,y:u16)->u32 { combine<add, u32:1>(x, y) }
fn plain<T>(x: T) -> T { x }
fn compare(x:u8, y:u8)->(bool,bool) { (plain < x, y > (x)) }
"#,
        )?;
        let f = &module.functions[0];
        let r = Type::Named("R".into());
        assert_eq!(f.type_params, ["A", "B", "R"]);
        assert!(f.has_inferred_parametrics());
        assert_eq!(
            f.callback_params[0].params,
            [Type::Named("A".into()), Type::Named("B".into())]
        );
        assert_eq!(f.callback_params[0].result, r);
        assert_eq!(f.value_parametrics[0].index, 1);
        assert_eq!(
            f.value_parametrics[0].default,
            Some(Expr::Literal {
                ty: Some(r.clone()),
                text: "0".into()
            })
        );
        let Expr::Scope { statements, result } = &f.body else {
            panic!("expected local let");
        };
        let ScopeStatement::Let(binding) = &statements[0] else {
            panic!("expected local let");
        };
        assert_eq!(binding.ty.as_ref(), Some(&r));
        let Expr::Match { arms, .. } = result.as_ref() else {
            panic!("expected match on the result");
        };
        assert_eq!(
            arms[0].patterns,
            [MatchPattern::Literal(
                f.value_parametrics[0].default.clone().unwrap()
            )]
        );
        assert_eq!(arms[0].value, Expr::Zero(r));
        assert!(
            matches!(&module.functions[1].body, Expr::ParametricCall { parametrics, .. } if parametrics.len() == 2)
        );
        assert!(
            matches!(&module.functions[3].body, Expr::Tuple(pair) if matches!(&pair[..], [
                Expr::Binary { op: BinaryOp::Lt, .. }, Expr::Binary { op: BinaryOp::Gt, .. }
            ]))
        );
        assert!(parse_module("fn f<T>(x:T)->T{x} fn g()->u8{T:0}").is_err());
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
        let Expr::Scope { statements, result } = &f.body else {
            panic!("expected a binding for the dependent literal");
        };
        let ScopeStatement::Let(binding) = &statements[0] else {
            panic!("expected local let");
        };
        assert!(
            matches!(&binding.value, Expr::Literal { ty, text } if ty == &binding.ty && text == "-1")
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
            parse_module("fn f<N:u32>(x:uN[N])->uN[N] {x} block bad(clk,x:uN[N + 1])->u8{x}")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn module_constants_in_block_and_function_dimensions_retain_their_names() -> Result<(), String>
    {
        let module = parse_module(
            r#"
use sizes;
const WORDS = u32:2;
fn pass(xs: (sN[sizes::WIDTH], bool)[WORDS]) -> (sN[sizes::WIDTH], bool)[WORDS] { xs }
block wires(clk, xs: (sN[sizes::WIDTH], bool)[WORDS]) -> (sN[sizes::WIDTH], bool)[WORDS] { xs }
"#,
        )?;
        let expected = Type::ParametricArray {
            element: Box::new(Type::Tuple(vec![
                Type::ParametricBits {
                    signed: Box::new(Expr::Bool(true)),
                    width: Box::new(Expr::Name("sizes::WIDTH".into())),
                },
                Type::Bool,
            ])),
            size: Box::new(Expr::Name("WORDS".into())),
        };
        let function = &module.functions[0];
        let block = &module.blocks[0];
        for annotation in [
            &function.params[0].ty,
            &function.result,
            &block.params[0].ty,
            &block.result,
        ] {
            assert_eq!(annotation, &expected);
        }
        for ty in [
            "xN[sizes::SIGNED][8]",
            "xN[true][WORDS]",
            "uN[self]",
            "u8[sizes::_]",
        ] {
            parse_module(&format!("fn pass(x:{ty})->{ty} {{ x }}"))?;
        }
        for (ty, message) in [
            ("u8[_]", "wildcard pattern"),
            ("uN[sizes::self]", "expected an identifier after '::'"),
        ] {
            for source in [
                format!("struct Bus {{ words: {ty} }}"),
                format!("fn f<F: fn({ty}) -> u8>(x: {ty}) -> u8 {{ F(x) }}"),
            ] {
                let error = parse_module(&source).unwrap_err();
                assert!(error.contains(message), "{source}\n{error}");
            }
        }
        for ty in ["u8[WORDS + 1]", "uN[WORDS * 2]", "xN[!SIGNED][WORDS]"] {
            assert!(
                parse_module(&format!("fn bad(x:{ty}) {{ }}")).is_err(),
                "{ty}"
            );
        }
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
fn build(x: u8) -> packets::Packet<0x_f_f, 0b1_0> {
    packets::Packet<255, 2> { x }
}
block stage<F: fn(u8) -> Packet<2>>(clk, xs: (Packet<2>, bool)[3]) -> Packet<2> {
    let empty:Packet<2>=zero!<Packet<2>>();
    if (Packet<2> { valid: true }).valid { empty } else { xs[u32:0].0 }
}
"#,
        )?;
        let qualified = Type::ParametricStruct {
            name: "packets::Packet".into(),
            args: vec![
                Expr::Literal {
                    ty: None,
                    text: "0xff".into(),
                },
                Expr::Literal {
                    ty: None,
                    text: "0b10".into(),
                },
            ],
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
                base: None,
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
    fn native_struct_parametrics_preserve_defaults_and_dependent_fields() -> Result<(), String> {
        let module = parse_module(
            r#"
pub struct Packet<N: u32, M: u32 = {N + u32:1},> {
    data: uN[N], nested: Other<M>, array: u8[M + u32:1],
}
fn wrap<N: u32>(data: uN[N], nested: Other<{N + u32:1}>) -> Packet<N> {
    Packet<N, {N + u32:1}> { data, nested, array: zero!<u8[N + u32:2]>() }
}
"#,
        )?;
        let definition = &module.structs[0];
        assert!(definition.public);
        assert!(matches!(&definition.value_parametrics[..], [
            ParametricBinding { name: n, index: 0, default: None, .. },
            ParametricBinding { name: m, index: 1, default: Some(Expr::Binary { op: BinaryOp::Add, .. }), .. },
        ] if n == "N" && m == "M"));
        assert!(
            matches!(&definition.fields[0].ty, Type::ParametricBits { width, .. }
            if width.as_ref() == &Expr::Name("N".into()))
        );
        assert_eq!(
            definition.fields[1].ty,
            Type::ParametricStruct {
                name: "Other".into(),
                args: vec![Expr::Name("M".into())],
            }
        );
        assert!(
            matches!(&definition.fields[2].ty, Type::ParametricArray { size, .. }
            if matches!(size.as_ref(), Expr::Binary { op: BinaryOp::Add, .. }))
        );
        let wrap = &module.functions[0];
        assert_eq!(
            wrap.result,
            Type::ParametricStruct {
                name: "Packet".into(),
                args: vec![Expr::Name("N".into())],
            }
        );
        assert!(
            matches!(&wrap.body, Expr::ParametricStruct { name, args, fields, base: None }
            if name == "Packet"
                && matches!(&args[..], [Expr::Name(n), Expr::Binary { op: BinaryOp::Add, .. }] if n == "N")
                && fields.len() == 3)
        );
        Ok(())
    }

    #[test]
    fn struct_arguments_preserve_expression_operators_and_delimiters() -> Result<(), String> {
        let module = parse_module(
            r#"
block compare<T>(clk, x: T, y: T) -> (bool, bool) {
    (x as T < WIDTH, 2 > (y >> u32:1))
}
block choose(clk, xs: u8[2]) -> u8[2] {
    let ys = map(Packet<2,3> { xs }.xs, |v| v);
    let zs = map(zero!<Packet<u32:2,{N + u32:1}>>().xs, |v| v);
    match (u8:1) + Packet<N,M> { tag: u8:0 }.tag { 0 => ys, _ => zs }
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
    fn symbolic_constructor_lookahead_respects_known_values() -> Result<(), String> {
        for (declarations, head) in [("", "a"), ("const LIMIT = u32:7;", "LIMIT")] {
            let signature = "fn compare(a: u32, b: u32, c: u32) -> (bool, bool)";
            let missing_parens =
                format!("{declarations} {signature} {{ ({head} < b, b > {{ c }}) }}");
            assert!(parse_module(&missing_parens).is_err(), "{missing_parens}");
            let valid = format!("{declarations} {signature} {{ ({head} < b, b > ({{ c }})) }}");
            let module = parse_module(&valid)?;
            assert!(matches!(&module.functions[0].body, Expr::Tuple(fields)
                if matches!(&fields[..], [Expr::Binary { op: BinaryOp::Lt, .. }, Expr::Binary { op: BinaryOp::Gt, .. }])));
        }

        let source = r#"
use packets;
use packets::Packet;
struct Local<W: u32> { data: uN[W] }
fn make<W: u32>(data: uN[W]) -> (Local<W>, Packet<W>, packets::Packet<W>) {
    (Local<W> { data }, Packet<W> { data }, packets::Packet<W> { data })
}
"#;
        let module = parse_with_imports(source, &BTreeSet::new(), &BTreeSet::new())?;
        assert!(matches!(&module.functions[0].body, Expr::Tuple(fields)
            if fields.len() == 3 && fields.iter().all(|field| matches!(field, Expr::ParametricStruct { .. }))));
        Ok(())
    }

    #[test]
    fn turbofish_preserves_struct_types_constructors_and_intrinsics() -> Result<(), String> {
        let module = parse_module(
            r#"
use wire;
struct Packet<N: u32> { data: uN[N] }
fn wrap<N: u32>(data: uN[N]) -> (Packet::<N>, wire::Packet::<N>) {
    (Packet::<N> { data }, wire::Packet::<{N + u32:0}> { data })
}
fn empty() -> Packet::<> { zero!::<Packet::<>>() }
fn width() -> u32 { bit_count::<uN[8]>() }
"#,
        )?;
        let wrap = &module.functions[0];
        assert!(
            matches!(&wrap.result, Type::Tuple(types) if matches!(&types[..], [
            Type::ParametricStruct { name: local, .. },
            Type::ParametricStruct { name: qualified, .. },
        ] if local == "Packet" && qualified == "wire::Packet"))
        );
        assert!(
            matches!(&wrap.body, Expr::Tuple(fields) if matches!(&fields[..], [
            Expr::ParametricStruct { name: local, .. },
            Expr::ParametricStruct { name: qualified, args, .. },
        ] if local == "Packet" && qualified == "wire::Packet" && matches!(&args[..], [Expr::Binary { op: BinaryOp::Add, .. }])))
        );
        assert!(
            matches!(&module.functions[1].body, Expr::Zero(Type::Applied { name, args })
            if name == "Packet" && args.is_empty())
        );
        assert!(
            matches!(&module.functions[2].body, Expr::TypeBuiltin { name, .. }
            if name == "bit_count")
        );
        Ok(())
    }

    #[test]
    fn struct_type_arguments_parse_nested_applications_and_constructors() -> Result<(), String> {
        let module = parse_module(
            r#"
struct FlowOut<T: type> { pop_valid: bool, pop_data: T }
struct Pair<A, B> { first: A, second: B }
fn compare(x: u8) -> bool {
    FlowOut<Pair<u8, u8>> { pop_valid: true, pop_data: Pair { first: x, second: x } }.pop_data.first < x
}
block stage<T>(clk, x: T) -> (FlowOut<Pair<T[2], (bool, u8)>>, Pair::<T, u8>) {
    (FlowOut<Pair<T[2], (bool, u8)>> { pop_valid: true, pop_data: x },
     Pair::<T, u8> { first: x, second: u8:0 })
}
"#,
        )?;
        assert_eq!(module.structs[0].type_params, ["T"]);
        assert_eq!(module.structs[1].type_params, ["A", "B"]);
        assert!(matches!(
            &module.functions[0].body,
            Expr::Binary {
                op: BinaryOp::Lt,
                ..
            }
        ));
        let expected = [
            Type::TypeApplied {
                name: "FlowOut".into(),
                args: vec![Type::TypeApplied {
                    name: "Pair".into(),
                    args: vec![
                        Type::Array {
                            element: Box::new(Type::Named("T".into())),
                            size: 2,
                        },
                        Type::Tuple(vec![
                            Type::Bool,
                            Type::Bits {
                                signed: false,
                                width: 8,
                            },
                        ]),
                    ],
                }],
            },
            Type::TypeApplied {
                name: "Pair".into(),
                args: vec![
                    Type::Named("T".into()),
                    Type::Bits {
                        signed: false,
                        width: 8,
                    },
                ],
            },
        ];
        let block = &module.blocks[0];
        assert_eq!(block.result, Type::Tuple(expected.to_vec()));
        let Expr::Tuple(values) = &block.output else {
            panic!("expected a pair of constructors");
        };
        for (value, expected) in values.iter().zip(expected) {
            assert!(matches!((value, expected), (
                Expr::TypeAppliedStruct { name, args, .. },
                Type::TypeApplied { name: expected_name, args: expected_args }
            ) if name == &expected_name && args == &expected_args));
        }
        Ok(())
    }

    #[test]
    fn struct_type_lookahead_respects_local_value_shadowing() -> Result<(), String> {
        let mut parser = Parser::new("struct Wrap<T> { payload: T } Wrap < x", false)?;
        let angle = parser
            .tokens
            .iter()
            .rposition(|token| token.text == "<")
            .unwrap();
        assert!(parser.type_struct_before(angle));
        parser.shadow_callable("Wrap");
        assert!(!parser.type_struct_before(angle));
        Ok(())
    }

    #[test]
    fn imported_struct_type_parameters_are_disambiguated() -> Result<(), String> {
        let module = parse_with_imports(
            r#"
use library;
use library::FlowOut;
fn wrap<T>(x: T) -> (FlowOut<T>, library::FlowOut<(T, u8)>) {
    (FlowOut<T> { pop_data: x }, library::FlowOut::<(T, u8)> { pop_data: (x, u8:0) })
}
"#,
            &BTreeSet::new(),
            &BTreeSet::from(["FlowOut".into(), "library::FlowOut".into()]),
        )?;
        let function = &module.functions[0];
        assert!(matches!(&function.result, Type::Tuple(types)
            if matches!(&types[..], [Type::TypeApplied { name: local, .. }, Type::TypeApplied { name: qualified, .. }]
                if local == "FlowOut" && qualified == "library::FlowOut")));
        assert!(matches!(&function.body, Expr::Tuple(values)
            if matches!(&values[..], [Expr::TypeAppliedStruct { name: local, .. }, Expr::TypeAppliedStruct { name: qualified, .. }]
                if local == "FlowOut" && qualified == "library::FlowOut")));
        Ok(())
    }

    #[test]
    fn structs_reject_mixed_type_and_value_parametrics() {
        for params in ["T, N: u32", "N: u32, T: type"] {
            let source = format!("struct Mixed<{params}> {{ field: T }}");
            let error = parse_module(&source).unwrap_err();
            assert!(
                error.contains("cannot mix type and value parameters"),
                "{error}"
            );
        }
    }

    #[test]
    fn turbofish_disambiguates_calls_and_named_map_callbacks() -> Result<(), String> {
        let module = parse_with_imports(
            r#"
use wire;
fn apply(x: u8) -> (u8, u8) { (local::<u8:3, true>(x), wire::apply::<u8:3, true>(x)) }
block sample(clk, xs: u8[2]) -> (u8[2], u8[2]) {
    let ys = map(Packet::<2,3> { xs }.xs, |v| v);
    (map(ys, local::<u8:3, true>), map(ys, wire::apply::<u8:3, true>))
}
"#,
            &BTreeSet::new(),
            &BTreeSet::new(),
        )?;
        let Expr::Tuple(calls) = &module.functions[0].body else {
            panic!("expected two calls");
        };
        let Expr::Tuple(maps) = &module.blocks[0].output else {
            panic!("expected two maps");
        };
        for ((call, map), expected) in calls.iter().zip(maps).zip(["local", "wire::apply"]) {
            assert!(
                matches!(call, Expr::ParametricCall { name, parametrics, args }
                if name == expected && parametrics.len() == 2 && args.len() == 1)
            );
            assert!(matches!(map, Expr::NamedMap { callback, parametrics, .. }
                if callback == expected && parametrics.len() == 2));
        }
        assert!(matches!(
            &module.blocks[0].statements[0],
            Stmt::Let {
                value: Expr::Map(_),
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn map_lambda_accepts_mixed_angle_spellings_in_zero_type() -> Result<(), String> {
        for zero in [
            "zero!<S<2,3>>()",
            "zero!<S::<2,3>>()",
            "zero!::<S<2,3>>()",
            "zero!::<S::<2,3>>()",
        ] {
            let source = format!(
                r#"
struct S<N: u32, M: u32> {{ data: u8[N], tag: uN[M] }}
block sample(clk) -> u8[2] {{ map({zero}.data, |x| x) }}
"#,
            );
            let module = parse_module(&source).map_err(|error| format!("{zero}: {error}"))?;
            let Expr::Map(mapped) = &module.blocks[0].output else {
                panic!("expected a map lambda for {zero}");
            };
            assert!(
                matches!(&mapped.iterable, ForIterable::Array(Expr::Field { value, name })
                if name == "data" && matches!(value.as_ref(), Expr::Zero(Type::Applied { name, args })
                    if name == "S" && args == &[2, 3]))
            );
        }
        Ok(())
    }

    #[test]
    fn struct_arguments_preserve_literal_types_and_arbitrary_widths() -> Result<(), String> {
        let module = parse_module(
            r#"
enum Mode: u8 { FIRST = 0, SECOND = 1 }
struct Tagged<S: bool, W: u8, I: s8, D: u32, X: uN[96], M: Mode> { data: xN[S][W] }
fn tag(data: s5) -> Tagged<true, u8:5, -1, u32:5, 4294967296, Mode::SECOND> {
    Tagged<true, u8:5, -1, u32:5, 4294967296, Mode::SECOND> { data }
}
struct Aggregates<T: (u8, bool), A: u8[2], R: Tagged<false, 5, 0, 5, 0, Mode::FIRST>> { v: u8 }
"#,
        )?;
        let value_types: Vec<_> = module.structs[0]
            .value_parametrics
            .iter()
            .map(|formal| &formal.ty)
            .collect();
        assert!(matches!(&value_types[..], [
            Type::Bool,
            Type::Bits { signed: false, width: 8 },
            Type::Bits { signed: true, width: 8 },
            Type::Bits { signed: false, width: 32 },
            Type::Bits { signed: false, width: 96 },
            Type::Named(name),
        ] if name == "Mode"));
        let function = &module.functions[0];
        let Type::ParametricStruct { args, .. } = &function.result else {
            panic!("expected value arguments in the result type");
        };
        assert!(matches!(&args[..], [
            Expr::Bool(true),
            Expr::Literal { ty: Some(Type::Bits { signed: false, width: 8 }), text: width },
            Expr::Literal { ty: None, text: negative },
            Expr::Literal { ty: Some(Type::Bits { signed: false, width: 32 }), text: dimension },
            Expr::Literal { ty: None, text: wide },
            Expr::Name(member),
        ] if width == "5" && negative == "-1" && dimension == "5" && wide == "4294967296" && member == "Mode::SECOND"));
        assert!(
            matches!(&function.body, Expr::ParametricStruct { args: actual, .. } if actual == args)
        );
        assert!(matches!(
            &module.structs[1].value_parametrics[..],
            [
                ParametricBinding {
                    ty: Type::Tuple(_),
                    ..
                },
                ParametricBinding {
                    ty: Type::Array { .. },
                    ..
                },
                ParametricBinding {
                    ty: Type::ParametricStruct { .. },
                    ..
                },
            ]
        ));
        Ok(())
    }

    #[test]
    fn unbraced_struct_arguments_require_names_or_literals() {
        for arguments in ["0b2", "00", "1+2", "N+2", "_", "u32", "2,,3"] {
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
                base: None,
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
        let Expr::Scope { statements, result } = &module.functions[3].body else {
            panic!("expected nested binding");
        };
        let ScopeStatement::Let(binding) = &statements[0] else {
            panic!("expected local let");
        };
        assert!(
            matches!(&binding.value, Expr::Scope { result, .. } if matches!(result.as_ref(), Expr::Tuple(pair) if pair.len() == 2))
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
        let module = parse_with_imports(source, &templates, &BTreeSet::new())?;
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
        let hinted = parse_with_imports(ambiguous, &templates, &BTreeSet::new())?;
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
    fn struct_splat_disambiguates_the_match_subject() -> Result<(), String> {
        let module = parse_module(
            r#"
struct S { flag: bool }
fn test(x: S) -> u8 {
    match S { ..x }.flag { true => u8:1, false => u8:0 }
}
"#,
        )?;
        let Expr::Match { subject, .. } = &module.functions[0].body else {
            panic!("expected match");
        };
        let Expr::Field { name, value } = subject.as_ref() else {
            panic!("expected field projection");
        };
        assert_eq!(name, "flag");
        let Expr::Struct {
            name,
            fields,
            base: Some(base),
        } = value.as_ref()
        else {
            panic!("expected struct update");
        };
        assert_eq!(name, "S");
        assert!(fields.is_empty());
        assert_eq!(base.value, Expr::Name("x".into()));
        assert_eq!(
            base.position,
            Some(SourcePosition {
                line: 4,
                column: 15
            })
        );
        Ok(())
    }

    #[test]
    fn struct_splat_must_end_the_constructor() {
        for tail in ["..x,", "..x, a: u8:0", "..x, ..x", "a: u8:0 ..x"] {
            let source = format!("struct S {{ a: u8 }} fn test(x: S) -> S {{ S {{ {tail} }} }}");
            assert!(parse_module(&source).is_err(), "{source}");
        }
    }

    #[test]
    fn struct_splat_position_survives_reparsing_a_forward_map_import() -> Result<(), String> {
        let module = parse_module(
            r#"struct S { a: u8 }
block first(clk, x: S) -> S { map(S { a: u8:3, ..x }, x) }
use helpers::map;
block second(clk, x: S) -> S { S { ..x } }
"#,
        )?;
        let Expr::Struct {
            base: Some(base), ..
        } = &module.blocks[1].output
        else {
            panic!("expected a struct update");
        };
        assert_eq!(
            base.position,
            Some(SourcePosition {
                line: 4,
                column: 36
            })
        );
        Ok(())
    }

    #[test]
    fn match_patterns_require_punctuated_arms_and_literal_range_endpoints() {
        for arms in [
            "0 => x 1 => x",
            "0 -> x",
            "0 = > x",
            "_ | => x",
            "0 .. = 1 => x",
            "0 .. x => x",
            "- 1 => x",
            "(0 | 1,) => x",
            "(_, ..) => x",
            "[0] => x",
            "NAME .. u8:1 => x",
            "u8[1]:0 => x",
        ] {
            let src = format!("fn f(x:u8) -> u8 {{ match x {{ {arms} }} }}");
            assert!(parse_module(&src).is_err(), "{src}");
        }
    }

    #[test]
    fn tuple_match_patterns_preserve_shape_and_leaf_order() -> Result<(), String> {
        let module = parse_module(
            r#"
fn f(x: ((bool,), (), u8)) -> bool {
    match x {
        ((false), (), u8:1..u8:2) | ((true,), (), limits::MAX,) => true,
        (_, _, _) => false,
    }
}
"#,
        )?;
        let Expr::Match { arms, .. } = &module.functions[0].body else {
            panic!("expected match");
        };
        let literal = |text: &str| Expr::Literal {
            ty: Some(Type::Bits {
                signed: false,
                width: 8,
            }),
            text: text.into(),
        };
        assert_eq!(
            arms[0].patterns,
            [
                MatchPattern::Tuple(vec![
                    MatchPattern::Tuple(vec![MatchPattern::Literal(Expr::Bool(false))]),
                    MatchPattern::Tuple(vec![]),
                    MatchPattern::Range {
                        start: literal("1"),
                        end: literal("2"),
                        inclusive: false,
                    },
                ]),
                MatchPattern::Tuple(vec![
                    MatchPattern::Tuple(vec![MatchPattern::Literal(Expr::Bool(true))]),
                    MatchPattern::Tuple(vec![]),
                    MatchPattern::Literal(Expr::Name("limits::MAX".into())),
                ]),
            ]
        );
        assert_eq!(
            arms[0].patterns[0].literals().cloned().collect::<Vec<_>>(),
            [Expr::Bool(false), literal("1"), literal("2")]
        );
        Ok(())
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
    fn range_spelling_is_compared_as_part_of_the_whole_tuple_pattern() {
        let left = "((u2:1..u2:1, _), false)";
        for (right, duplicate) in [
            (left, true),
            ("((u2:1 .. u2:1, _,), false,)", true),
            ("((uN[2]:1..uN[2]:1, _), false)", false),
            ("((u2:1..u2:1, _), true)", false),
        ] {
            for between in ["=> true,", "|"] {
                let src = format!(
                    "fn f(x: ((u2, bool), bool)) -> bool {{ match x {{ {left} {between} {right} => false, _ => true }} }}"
                );
                assert_eq!(parse_module(&src).is_err(), duplicate, "{src}");
            }
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
                "expected block, const, enum, fn, struct, or type after pub",
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
        assert!(parse_module("block import(clk, import: u8) -> u8 { import }").is_ok());
        assert!(parse_module("block inst(clk, x: u8) -> u8 { let inst = x; inst }").is_ok());
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
            "uN[WIDTH]",
            "uN[8][WORDS]",
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
    fn block_value_and_callback_parameters_share_explicit_argument_order() -> Result<(), String> {
        let module = parse_module(
            r#"
const BASE = u32:3;
fn identity(x: u8) -> u8 { x }
block caller(clk, x: u8) -> (u8, u8) {
    let a = stage<2, identity, {BASE + u32:1}>(x);
    let b = stage::<BASE, identity, u32:6>(x);
    (a, b)
}
block stage<T, N: u32, F: fn(T) -> T, M: u32>(clk, x: T) -> T {
    unroll_for! (_, acc) in N..M { F(acc) }(x)
}
"#,
        )?;
        let stage = &module.blocks[1];
        assert!(stage.is_parametric());
        assert_eq!(stage.type_params, ["T"]);
        assert_eq!(stage.value_parametrics[0].index, 0);
        assert_eq!(stage.value_parametrics[1].index, 2);
        assert_eq!(
            stage
                .parametric_bindings()
                .map(|formal| match formal {
                    FunctionParametricRef::Value(value) => value.name.as_str(),
                    FunctionParametricRef::Callback(callback) => callback.name.as_str(),
                })
                .collect::<Vec<_>>(),
            ["N", "F", "M"]
        );
        let [Stmt::Let { value: a, .. }, Stmt::Let { value: b, .. }] =
            module.blocks[0].statements.as_slice()
        else {
            panic!("expected two call bindings");
        };
        assert!(matches!(a, Expr::ParametricCall { name, parametrics, .. }
            if name == "stage"
                && matches!(parametrics.as_slice(),
                    [Expr::Literal { ty: None, .. }, Expr::Name(f), Expr::Binary { .. }] if f == "identity")));
        assert!(matches!(b, Expr::ParametricCall { name, parametrics, .. }
            if name == "stage"
                && matches!(parametrics.as_slice(),
                    [Expr::Name(base), Expr::Name(f), Expr::Literal { ty: Some(_), .. }]
                        if base == "BASE" && f == "identity")));
        Ok(())
    }

    #[test]
    fn block_parameter_lists_require_distinct_binders() {
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
            ("<T, T: u32>", "duplicate block value parameter 'T'"),
            ("<N: u8>", "block value parameters require type 'u32'"),
            (
                "<N: u32 = {u32:1}>",
                "block value parameters require explicit arguments",
            ),
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
    fn keeps_aggregate_types_and_child_call_arguments() {
        let module = parse_module(
            r#"
            struct Result { value: s8, spare: (), }
            extern fn invert(value: s8) -> s8 = "__logic__invert";
            block wrapper(clk, value: s8) -> (Result,) {
                let lane = child(invert(value));
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
        let [
            Stmt::Let {
                name,
                value: Expr::Call { name: callee, args },
                ..
            },
        ] = &block.statements[..]
        else {
            panic!("expected a child call binding");
        };
        assert_eq!(name, "lane");
        assert_eq!(callee, "child");
        assert!(matches!(&args[..], [Expr::Call { name, .. }] if name == "invert"));
    }

    #[test]
    fn removed_instance_forms_direct_callers_to_child_calls_and_deferred_lets() {
        for (stmt, diagnostic) in [
            (
                "inst lane: child;",
                "use `let name = child(args);` to instantiate a block, found 'inst'",
            ),
            (
                "lane.input := x;",
                "pass child inputs as call arguments; use a deferred let for feedback, found 'lane'",
            ),
        ] {
            let source = format!("block wrapper(clk, x: u8) -> u8 {{\n    {stmt}\n    x\n}}");
            assert_eq!(
                parse_module(&source).unwrap_err(),
                format!("2:5: {diagnostic}")
            );
        }
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
    let child = library::pipe(library::OFFSET);
    match x { library::OFFSET | library::LIMIT => child, _ => u8:0 }
}"#,
        )?;
        let block = &module.blocks[0];
        let Stmt::Let {
            value: Expr::Call { name, args },
            ..
        } = &block.statements[0]
        else {
            panic!("expected a child call");
        };
        assert_eq!(name, "library::pipe");
        assert_eq!(args, &[Expr::Name("library::OFFSET".into())]);
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
    fn typed_array_literals_retain_the_annotation_and_explicit_members() -> Result<(), String> {
        let module = parse_module(
            r#"
            fn grid<T, N: u32>(x: T) -> T[N][2] { T[N][2]:[[x, ...], ...] }
            fn repeated(x: u8) -> u8[4] { u8[4]:(u8[3]:[1, x, ...,]) }
            block pair(clk, p: (bool, u8)) -> (bool, u8)[2] {
                (bool, u8)[2]:[p, p,]
            }
            "#,
        )?;
        let grid = &module.functions[0];
        assert_eq!(
            grid.body,
            Expr::ArrayLiteral {
                annotation: Some(grid.result.clone()),
                members: vec![Expr::ArrayLiteral {
                    annotation: None,
                    members: names(["x"]).into(),
                    has_ellipsis: true,
                }],
                has_ellipsis: true,
            }
        );
        let repeated = &module.functions[1];
        assert_eq!(
            repeated.body,
            Expr::ArrayLiteral {
                annotation: Some(repeated.result.clone()),
                members: vec![
                    Expr::Literal {
                        ty: None,
                        text: "1".into()
                    },
                    Expr::Name("x".into())
                ],
                has_ellipsis: true,
            }
        );
        let pair = &module.blocks[0];
        assert_eq!(
            pair.output,
            Expr::ArrayLiteral {
                annotation: Some(pair.result.clone()),
                members: names(["p", "p"]).into(),
                has_ellipsis: false,
            }
        );
        Ok(())
    }

    #[test]
    fn typed_array_map_operands_handle_tuple_and_struct_type_delimiters() -> Result<(), String> {
        for ty in [
            "(u8, bool)",
            "Pair<u8, bool>",
            "Pair::<u8, bool>",
            "lib::Packet<1,2>",
        ] {
            let source = format!(
                r#"
                struct Pair<A,B> {{ first: A, second: B }}
                block b(clk, x: {ty}) -> {ty}[2] {{ map({ty}[2]:[x, ...], |e| e) }}
            "#
            );
            let module = parse_module(&source)?;
            let block = &module.blocks[0];
            let Expr::Map(map) = &block.output else {
                panic!("expected map over {ty}")
            };
            assert_eq!(
                map.iterable,
                ForIterable::Array(Expr::ArrayLiteral {
                    annotation: Some(block.result.clone()),
                    members: names(["x"]).into(),
                    has_ellipsis: true,
                })
            );
        }
        Ok(())
    }

    #[test]
    fn array_ellipses_require_a_preceding_member_and_terminate_the_literal() {
        for expr in [
            "[...]",
            "u8[0]:[...,]",
            "[x ...]",
            "[x, .. .]",
            "[x, ..., x]",
            "[x, ..., ...]",
            "u8[4]:[x, ...][0]",
            "library::",
            "library::Packet::<",
            "library::Packet<1>[",
        ] {
            let source = format!("fn f(x:u8) -> u8 {{ {expr} }}");
            assert!(parse_module(&source).is_err(), "{expr}");
        }
        assert!(parse_module("fn f(x:u8)->u8 { (u8[4]:[x,...])[0] }").is_ok());
        assert!(parse_module("fn f() -> u8 { library::").is_err());
    }

    #[test]
    fn array_annotation_lookahead_keeps_sibling_comparisons_separate() -> Result<(), String> {
        let comparisons = "x < y,".repeat(4096);
        let source = format!("fn f(x:u8,y:u8)->() {{ let _ = ({comparisons}); () }}");
        let module = parse_module(&source)?;
        let Expr::Scope { statements, .. } = &module.functions[0].body else {
            panic!("expected a tuple binding")
        };
        let ScopeStatement::Let(binding) = &statements[0] else {
            panic!("expected local let");
        };
        assert!(matches!(&binding.value, Expr::Tuple(fields) if fields.len() == 4096));
        Ok(())
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
        assert!(parse_module(
            "struct Sample { out: u8 } block b(clk, xs: Sample[2]) -> u8[2] { map(xs, |c| c.out) }"
        ).is_ok());
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
            "block b(clk, x: u8[8_]) -> u8 { x[u1:0] }",
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
