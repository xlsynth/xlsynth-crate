// SPDX-License-Identifier: Apache-2.0

//! Parse the module declarations needed to resolve imported DSLX types.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    Dialect, ExpressionGrammar, MAX_SYNTAX_DEPTH, Parser, Token, ValueContext, fixed_width,
    is_dslx_member_name, is_use_identifier, lex,
};

/// Names, types, and import bindings written in one DSLX source module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DslxOutline {
    /// Local import binding to the full dotted module subject.
    pub imports: BTreeMap<String, String>,
    /// Full paths of the leaves selected by DSLX use trees.
    pub uses: Vec<Vec<String>>,
    pub declarations: BTreeMap<String, OutlineDecl>,
    /// Public functions and constants, including functions with no return
    /// annotation.
    pub public_values: BTreeSet<String>,
    /// Directly declared public functions, including implicit unit results.
    pub public_functions: BTreeSet<String>,
    /// Ordered formal annotations for public functions with explicit bindings.
    pub function_parametrics: BTreeMap<String, Vec<OutlineFunctionParametric>>,
    /// Explicit function return annotations in their declaration scope.
    pub functions: BTreeMap<String, OutlineType>,
}

/// A function parametric and its written annotation in the declaring module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlineFunctionParametric {
    pub name: String,
    pub ty: OutlineType,
}

/// A module-level DSLX type declaration and its visibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlineDecl {
    pub public: bool,
    /// The declaration has its own explicit parametric bindings.
    pub parametric: bool,
    pub kind: OutlineKind,
}

impl OutlineDecl {
    /// DSLX can declare `self`, but a qualified reference requires an
    /// identifier.
    pub(crate) fn has_public_spelling(&self, name: &str) -> bool {
        self.public && is_use_identifier(name)
    }

    pub(crate) fn is_public_struct_template(&self, name: &str) -> bool {
        self.parametric
            && self.has_public_spelling(name)
            && matches!(self.kind, OutlineKind::Struct { .. })
    }
}

/// Type declarations and ordered members as written, before typechecking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutlineKind {
    Struct {
        fields: Vec<(String, OutlineType)>,
    },
    /// A struct named by a DSLX fuzz-domain attribute. Its fields are derived
    /// by the native frontend.
    GeneratedStruct,
    Enum {
        underlying: Option<OutlineType>,
        members: Vec<(String, String)>,
    },
    Alias(OutlineType),
    /// A declaration such as a proc or semantic sum outside BX value types.
    Opaque,
}

/// A written type and the structure needed to follow its nominal leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlineType {
    /// Original source between the first and last tokens of the annotation.
    pub syntax: String,
    pub shape: OutlineShape,
}

/// Written names and aggregate components. DSLX evaluates concrete widths,
/// array sizes and parametric bindings when the module is checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutlineShape {
    Name(Vec<String>),
    Array(Box<OutlineType>),
    Tuple(Vec<OutlineType>),
    Other,
}

/// Parse declaration syntax while retaining expressions for DSLX to evaluate.
/// Function bodies are skipped as balanced tokens in the same source snapshot.
pub fn parse_dslx_outline(source: &str) -> Result<DslxOutline, String> {
    parse_outline_with_functions(source, |_, _, _, _| {}, |_| {}, |_, _, _| {})
}

/// Index semantic sums for resolving checked constructor and pattern paths.
/// They retain their existing opaque representation in the public outline.
pub(crate) fn outline_with_sum_names(
    source: &str,
    sums: &mut BTreeSet<String>,
) -> Result<DslxOutline, String> {
    parse_outline_with_functions(
        source,
        |_, _, _, _| {},
        |name| {
            sums.insert(name.to_owned());
        },
        |_, _, _| {},
    )
}

/// Visit numeric enum names and the byte after each declaration's closing
/// brace.
pub(crate) fn visit_numeric_enums(
    source: &str,
    visit: impl FnMut(Token<'_>, usize, bool),
) -> Result<(), String> {
    parse_outline_with_functions(source, |_, _, _, _| {}, |_| {}, visit)?;
    Ok(())
}

/// Original declaration tokens for a checked DSLX function. Defaults and
/// signature types can contain calls that affect its calling convention.
pub(crate) struct DslxFunctionTokens<'s> {
    pub(crate) bindings: Vec<Token<'s>>,
    pub(crate) signature: Vec<Token<'s>>,
    pub(crate) body: Vec<Token<'s>>,
}

/// Index the original tokens of top-level functions on the diagnostic
/// path.
pub(crate) fn dslx_function_tokens(
    source: &str,
) -> Result<BTreeMap<String, DslxFunctionTokens<'_>>, String> {
    let mut functions = BTreeMap::new();
    let mut ambiguous = false;
    parse_outline_with_functions(
        source,
        |name, bindings, signature, body| {
            let tokens = DslxFunctionTokens {
                bindings: bindings.to_vec(),
                signature: signature.to_vec(),
                body: body.to_vec(),
            };
            ambiguous |= functions.insert(name.to_owned(), tokens).is_some();
        },
        |_| {},
        |_, _, _| {},
    )?;
    if ambiguous {
        return Err("multiple DSLX source declarations for one function name".into());
    }
    Ok(functions)
}

/// Copy a public function's binding and parameter declarations for a type
/// witness that returns one operand under the original inference constraints.
pub(crate) fn argument_identity(
    source: &str,
    function: &str,
    index: usize,
    witness_name: &str,
) -> Result<(String, OutlineType), String> {
    let functions = dslx_function_tokens(source)?;
    let declaration = functions
        .get(function)
        .ok_or("missing DSLX function declaration")?;
    let start = declaration
        .bindings
        .first()
        .or(declaration.signature.first())
        .ok_or("missing DSLX function signature")?
        .start;
    let end = declaration
        .signature
        .last()
        .ok_or("missing DSLX function parameters")?;
    let mut p = outline_parser(&source[start..end.start + end.text.len()])?;
    let bindings = p.outline_bindings(None)?;
    p.expect("(")?;
    let params = p.list(")", |p| {
        let name = p.tokens[p.pos];
        p.outline_identifier()?;
        p.expect(":")?;
        Ok((name, p.outline_type(&bindings)?))
    })?;
    let (name, annotation) = params
        .get(index)
        .ok_or("missing native argument annotation")?;
    let mut signature = p.outline_syntax(0);
    // A discarded parameter can still constrain another operand. Give it a
    // fresh binding when this witness needs to return it.
    let mut value_name = format!("{witness_name}_value");
    let name = if name.text == "_" {
        while p.tokens.iter().any(|t| t.text == value_name) {
            value_name.push('_');
        }
        signature.replace_range(name.start..name.start + 1, &value_name);
        &value_name
    } else {
        name.text
    };
    Ok((
        format!(
            "pub fn {witness_name}{signature} -> {} {{ {name} }}",
            annotation.syntax
        ),
        annotation.clone(),
    ))
}

fn outline_parser(source: &str) -> Result<Parser<'_>, String> {
    Ok(Parser {
        source,
        dialect: Dialect::Dslx,
        tokens: lex(source, Dialect::Dslx, None)?,
        pos: 0,
        depth: 0,
        dslx_type_display: false,
        value_context: ValueContext::Block,
        expression_grammar: ExpressionGrammar::Block,
        block_type_params: BTreeSet::new(),
        dependent_types: false,
        callable_templates: Default::default(),
        hidden_templates: Vec::new(),
        block_map_shadowed: false,
        forward_map_import: false,
    })
}

/// Source ranges needed to defer a declaration and guard its comparisons in
/// an isolated, typecheck-only copy of a DSLX module.
pub(crate) struct ConversionItem<'a, 's> {
    pub kind: &'s str,
    pub name: Token<'s>,
    pub associated: bool,
    pub parametric: bool,
    pub fuzz_domain: bool,
    pub test_attributes: &'a [std::ops::Range<usize>],
    pub body: &'a [Token<'s>],
}

/// Visit functions, impl methods, struct definitions and module constant
/// initializers without resolving references or typechecking their bodies.
/// Return the insertion point for a declaration after leading module
/// attributes.
pub(crate) fn visit_conversion_items<'s>(
    source: &'s str,
    mut visit: impl FnMut(ConversionItem<'_, 's>),
) -> Result<usize, String> {
    let mut p = outline_parser(source)?;
    while p.peek(0) == "#" && p.peek(1) == "!" {
        p.pos += 2;
        p.outline_group("[")?;
    }
    let declarations_start = p.tokens.get(p.pos).map_or(source.len(), |t| t.start);
    let mut scope_depth = 0usize;
    let mut test_attributes = Vec::new();
    let mut fuzz_domain = false;
    while !p.peek(0).is_empty() {
        let kind = p.tokens[p.pos].text;
        let start = p.tokens[p.pos].start;
        p.pos += 1;
        if kind == "#" {
            let inner = p.eat("!");
            let test = !inner && matches!(p.peek(1), "test" | "quickcheck" | "test_proc");
            fuzz_domain |= !inner && p.peek(1) == "fuzz_domain";
            p.outline_group("[")?;
            if test {
                let last = &p.tokens[p.pos - 1];
                test_attributes.push(start..last.start + last.text.len());
            }
            continue;
        }
        match kind {
            "{" => scope_depth += 1,
            "}" => scope_depth = scope_depth.saturating_sub(1),
            "pub" => continue,
            _ => { /* Other declaration tokens do not open an impl scope. */ }
        }
        if !(matches!(kind, "fn" | "struct" | "impl" | "proc")
            || kind == "const" && scope_depth == 0)
            || !is_dslx_member_name(p.peek(0))
        {
            test_attributes.clear();
            fuzz_domain = false;
            continue;
        }
        let name = p.tokens[p.pos];
        p.pos += 1;
        let parametric = p.peek(0) == "<";
        let mut body = None;
        if kind == "const" {
            while !p.eat("=") {
                match p.peek(0) {
                    "(" | "[" | "{" | "<" => p.outline_group(p.peek(0))?,
                    "" | ";" => return Err(p.error("expected DSLX constant initializer")),
                    _ => p.pos += 1,
                }
            }
            body = Some(p.pos);
            p.skip_outline_item(false)?;
        }
        while !matches!(kind, "impl" | "const") && !p.peek(0).is_empty() {
            match p.peek(0) {
                ";" => {
                    p.pos += 1;
                    break;
                }
                "{" => {
                    body = Some(p.pos);
                    p.outline_group("{")?;
                    break;
                }
                "(" | "[" | "<" => p.outline_group(p.peek(0))?,
                _ => p.pos += 1,
            }
        }
        visit(ConversionItem {
            kind,
            name,
            associated: scope_depth > 0,
            parametric,
            fuzz_domain,
            test_attributes: &test_attributes,
            body: &p.tokens[body.unwrap_or(p.pos)..p.pos - usize::from(kind == "const")],
        });
        test_attributes.clear();
        fuzz_domain = false;
    }
    Ok(declarations_start)
}

fn parse_outline_with_functions<'s>(
    source: &'s str,
    mut visit: impl FnMut(&str, &[Token<'s>], &[Token<'s>], &[Token<'s>]),
    mut visit_sum: impl FnMut(&str),
    mut visit_enum: impl FnMut(Token<'s>, usize, bool),
) -> Result<DslxOutline, String> {
    let mut parser = outline_parser(source)?;
    let mut module = DslxOutline::default();
    let mut uses = Vec::new();
    let mut pending_domain = None;
    while !parser.peek(0).is_empty() {
        if parser.eat("#") {
            let inner = parser.eat("!");
            if !inner
                && parser.peek(1) == "fuzz_domain"
                && parser.peek(2) == "("
                && parser.peek(4) == ")"
                && parser.peek(5) == "]"
            {
                pending_domain = fuzz_domain_name(parser.peek(3));
            }
            parser.outline_group("[")?;
            continue;
        }
        let domain = pending_domain.take();
        let public = parser.eat("pub");
        match parser.peek(0) {
            "import" => {
                parser.pos += 1;
                let mut parts = vec![parser.outline_identifier()?];
                while parser.eat(".") {
                    parts.push(parser.outline_identifier()?);
                }
                let binding = if parser.eat("as") {
                    parser.outline_identifier()?
                } else {
                    parts.last().unwrap().clone()
                };
                parser.expect(";")?;
                module.imports.insert(binding, parts.join("."));
            }
            "use" => {
                parser.pos += 1;
                parser.use_tree(&mut uses, None)?;
                parser.expect(";")?;
            }
            "type" | "struct" | "enum" | "proc" | "fn" => {
                let keyword = parser.peek(0);
                parser.pos += 1;
                let name = parser.outline_identifier()?;
                let name_token = parser.tokens[parser.pos - 1];
                let bindings_start = parser.pos;
                let mut function_parametrics = Vec::new();
                let bindings = parser.outline_bindings(
                    (public && keyword == "fn").then_some(&mut function_parametrics),
                )?;
                let kind = match keyword {
                    "type" => {
                        parser.expect("=")?;
                        let ty = parser.outline_type(&bindings)?;
                        parser.expect(";")?;
                        OutlineKind::Alias(ty)
                    }
                    "struct" => {
                        parser.expect("{")?;
                        let fields = parser.list("}", |p| {
                            let field = p.outline_identifier()?;
                            p.expect(":")?;
                            Ok((field, p.outline_type(&bindings)?))
                        })?;
                        if let Some(domain) = domain
                            && bindings.is_empty()
                        {
                            module.declarations.insert(
                                domain,
                                OutlineDecl {
                                    public,
                                    parametric: false,
                                    kind: OutlineKind::GeneratedStruct,
                                },
                            );
                        }
                        OutlineKind::Struct { fields }
                    }
                    "enum" => {
                        let kind = parser.outline_enum(&bindings)?;
                        if matches!(kind, OutlineKind::Opaque) {
                            visit_sum(&name);
                        } else {
                            let close = parser.tokens[parser.pos - 1];
                            visit_enum(name_token, close.start + close.text.len(), public);
                        }
                        kind
                    }
                    "fn" => {
                        if public {
                            module.public_values.insert(name.clone());
                            module.public_functions.insert(name.clone());
                            if !function_parametrics.is_empty() {
                                module
                                    .function_parametrics
                                    .insert(name.clone(), function_parametrics);
                            }
                        }
                        let signature_start = parser.pos;
                        parser.outline_group("(")?;
                        let result = parser
                            .eat("->")
                            .then(|| parser.outline_type(&bindings))
                            .transpose()?;
                        let body_start = parser.pos;
                        parser.outline_group("{")?;
                        visit(
                            &name,
                            &parser.tokens[bindings_start..signature_start],
                            &parser.tokens[signature_start..body_start],
                            &parser.tokens[body_start..parser.pos],
                        );
                        if let Some(result) = result {
                            module.functions.insert(name, result);
                        }
                        continue;
                    }
                    "proc" => {
                        parser.skip_outline_item(true)?;
                        OutlineKind::Opaque
                    }
                    _ => unreachable!(),
                };
                module.declarations.insert(
                    name,
                    OutlineDecl {
                        public,
                        parametric: !bindings.is_empty(),
                        kind,
                    },
                );
            }
            "impl" | "trait" => parser.skip_outline_item(true)?,
            "const" => {
                parser.pos += 1;
                let name = parser.outline_identifier()?;
                if public {
                    module.public_values.insert(name);
                }
                parser.skip_outline_item(false)?;
            }
            "const_assert!" => parser.skip_outline_item(false)?,
            _ => return Err(parser.error("expected a DSLX module declaration")),
        }
    }
    module.uses = uses.into_iter().map(|path| path.components).collect();
    Ok(module)
}

/// DSLX accepts arbitrary strings for derived fuzz-domain names. Keep the
/// decoded names that can be selected by a DSLX identifier. Of the string
/// escapes, only hex, Unicode, and a quoted prime can produce such a name.
fn fuzz_domain_name(token: &str) -> Option<String> {
    let raw = token.strip_prefix('"')?.strip_suffix('"')?;
    let mut chars = raw.chars();
    let mut name = String::new();
    while let Some(c) = chars.next() {
        let c = if c == '\\' {
            match chars.next()? {
                '\'' => '\'',
                'x' => {
                    let value = chars.next()?.to_digit(16)? * 16 + chars.next()?.to_digit(16)?;
                    char::from_u32(value)?
                }
                'u' => {
                    let (digits, rest) = chars.as_str().strip_prefix('{')?.split_once('}')?;
                    if digits.is_empty()
                        || digits.len() > 6
                        || !digits.bytes().all(|b| b.is_ascii_hexdigit())
                    {
                        return None;
                    }
                    chars = rest.chars();
                    char::from_u32(u32::from_str_radix(digits, 16).ok()?)?
                }
                _ => return None,
            }
        } else {
            c
        };
        name.push(c);
    }
    is_use_identifier(&name).then_some(name)
}

impl Parser<'_> {
    fn outline_identifier(&mut self) -> Result<String, String> {
        self.take_identifier(is_dslx_member_name)
    }

    /// Copy the exact source of consumed tokens, including internal trivia.
    fn outline_syntax(&self, start: usize) -> String {
        let last = self.tokens[self.pos - 1];
        self.source[self.tokens[start].start..last.start + last.text.len()].to_owned()
    }

    /// Skip a delimited group without parsing executable expressions. Angles
    /// nest only in type-argument context, never inside a dimension or body.
    fn outline_group(&mut self, open: &str) -> Result<(), String> {
        let close = match open {
            "(" => ")",
            "[" => "]",
            "{" => "}",
            "<" => ">",
            _ => unreachable!(),
        };
        self.expect(open)?;
        let mut pending = vec![close];
        while let Some(&expected) = pending.last() {
            let token = self.peek(0);
            if token == expected {
                pending.pop();
            } else {
                match token {
                    "(" => pending.push(")"),
                    "[" => pending.push("]"),
                    "{" => pending.push("}"),
                    "<" if expected == ">" => pending.push(">"),
                    "" | ")" | "]" | "}" => {
                        return Err(self.error(&format!("expected '{expected}'")));
                    }
                    _ => { /* Other tokens cannot close this group. */ }
                }
            }
            self.pos += 1;
        }
        Ok(())
    }

    /// Parametric bindings shadow module-level names in annotations.
    fn outline_bindings(
        &mut self,
        mut formals: Option<&mut Vec<OutlineFunctionParametric>>,
    ) -> Result<BTreeSet<String>, String> {
        let mut names = BTreeSet::new();
        if self.eat("<") {
            self.list(">", |p| {
                let name = p.outline_identifier()?;
                p.expect(":")?;
                if let Some(formals) = &mut formals {
                    let start = p.pos;
                    let ty = match p.outline_type(&names) {
                        Ok(ty) if matches!(p.peek(0), "=" | "," | ">") => ty,
                        _ => {
                            // Preserve annotations outside the nominal-value
                            // grammar,
                            // including function types. DSLX checks them when
                            // used.
                            p.pos = start;
                            while !matches!(p.peek(0), "=" | "," | ">") {
                                p.skip_binding_token()?;
                            }
                            if p.pos == start {
                                return Err(p.error("expected a parametric binding type"));
                            }
                            OutlineType {
                                syntax: p.outline_syntax(start),
                                shape: OutlineShape::Other,
                            }
                        }
                    };
                    formals.push(OutlineFunctionParametric {
                        name: name.clone(),
                        ty,
                    });
                }
                names.insert(name);
                while !matches!(p.peek(0), "," | ">") {
                    p.skip_binding_token()?;
                }
                Ok(())
            })?;
        }
        Ok(names)
    }

    /// A grouped type or default may contain its own commas and comparisons.
    fn skip_binding_token(&mut self) -> Result<(), String> {
        match self.peek(0) {
            "(" | "[" | "{" | "<" => self.outline_group(self.peek(0)),
            "" => Err(self.error("expected end of parametric bindings")),
            _ => {
                self.pos += 1;
                Ok(())
            }
        }
    }

    /// Read a type using DSLX's tuple, reference, parametric and dimension
    /// productions. The first dimensions of parametric bit types describe a
    /// scalar; subsequent dimensions describe arrays of that scalar.
    fn outline_type(&mut self, bindings: &BTreeSet<String>) -> Result<OutlineType, String> {
        self.nested(|p| {
            let start = p.pos;
            let mut is_channel = false;
            let shape = if p.eat("(") {
                OutlineShape::Tuple(p.list(")", |p| p.outline_type(bindings))?)
            } else {
                let head = p.outline_identifier()?;
                let mut bit_dims = match head.as_str() {
                    "uN" | "sN" | "bits" => 1,
                    "xN" => 2,
                    _ => 0,
                };
                is_channel = head == "chan";
                let builtin = bit_dims > 0
                    || fixed_width(&head).is_some()
                    || matches!(head.as_str(), "bool" | "token" | "chan" | "Self" | "type");
                let local = bindings.contains(&head);
                let mut path = vec![head];
                while p.eat("::") {
                    path.push(p.outline_identifier()?);
                }
                if p.peek(0) == "<" {
                    p.outline_group("<")?;
                }
                while bit_dims > 0 && p.peek(0) == "[" {
                    p.outline_group("[")?;
                    bit_dims -= 1;
                }
                if builtin || local {
                    OutlineShape::Other
                } else {
                    OutlineShape::Name(path)
                }
            };
            let mut ty = OutlineType {
                syntax: p.outline_syntax(start),
                shape,
            };
            let mut dimensions = 0;
            while p.peek(0) == "[" {
                dimensions += 1;
                if dimensions + p.depth > MAX_SYNTAX_DEPTH {
                    // DSLX permits arbitrarily many array suffixes. A module
                    // can define such a type without using it in BX. Keep its
                    // syntax and let the normal checker bound used layouts.
                    while p.peek(0) == "[" {
                        p.outline_group("[")?;
                    }
                    ty = OutlineType {
                        syntax: p.outline_syntax(start),
                        shape: OutlineShape::Other,
                    };
                    break;
                }
                p.outline_group("[")?;
                ty = OutlineType {
                    syntax: p.outline_syntax(start),
                    shape: OutlineShape::Array(Box::new(ty)),
                };
            }
            if is_channel {
                if !p.eat("in") && !p.eat("out") {
                    return Err(p.error("expected channel direction"));
                }
                ty = OutlineType {
                    syntax: p.outline_syntax(start),
                    shape: OutlineShape::Other,
                };
            }
            Ok(ty)
        })
    }

    /// Keep ordinary enum discriminants in source order. A payload-bearing
    /// variant makes this declaration a DSLX semantic sum.
    fn outline_enum(&mut self, bindings: &BTreeSet<String>) -> Result<OutlineKind, String> {
        let underlying = if self.eat(":") {
            Some(self.outline_type(bindings)?)
        } else {
            None
        };
        let open = self.pos;
        self.expect("{")?;
        let mut members = Vec::new();
        while !self.eat("}") {
            let name = self.outline_identifier()?;
            if !self.eat("=") {
                self.pos = open;
                self.outline_group("{")?;
                return Ok(OutlineKind::Opaque);
            }
            let start = self.pos;
            loop {
                match self.peek(0) {
                    "}" => break,
                    // A comma can also separate parametric arguments in an
                    // enum's value expression, e.g. f<A, B>().
                    "," if self.peek(1) == "}"
                        || (is_dslx_member_name(self.peek(1))
                            && matches!(self.peek(2), "=" | "(" | "{")) =>
                    {
                        break;
                    }
                    "(" | "[" | "{" => self.outline_group(self.peek(0))?,
                    "" => return Err(self.error("expected end of enum")),
                    _ => self.pos += 1,
                }
            }
            if self.pos == start {
                return Err(self.error("expected enum discriminant"));
            }
            members.push((name, self.outline_syntax(start)));
            if !self.eat(",") {
                self.expect("}")?;
                break;
            }
        }
        if underlying.is_none() && members.is_empty() {
            Ok(OutlineKind::Opaque)
        } else {
            Ok(OutlineKind::Enum {
                underlying,
                members,
            })
        }
    }

    /// Skip a declaration with a brace body, or a semicolon-terminated
    /// constant/alias. Braces inside its signature or expression stay nested.
    fn skip_outline_item(&mut self, body: bool) -> Result<(), String> {
        loop {
            match self.peek(0) {
                ";" => {
                    self.pos += 1;
                    return Ok(());
                }
                "{" if body => return self.outline_group("{"),
                "(" | "[" | "{" => self.outline_group(self.peek(0))?,
                "<" if body => self.outline_group("<")?,
                "" => return Err(self.error("expected end of DSLX declaration")),
                _ => self.pos += 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};

    #[test]
    fn outline_retains_declaration_order_scopes_and_type_expressions() -> Result<(), String> {
        let source = r#"
#![feature(type_inference_v1)]
import net.protocol as wire;
import common.format;
#[sv_type("nested } \" type Fake = u32;")]
pub struct Frame {
    header: wire::Header,
    values: (wire::Header, uN[width()])[N // computed
        + u32:1],
}
type Local = Frame;
struct T { x: u8 }
pub enum Kind : uN[width()] { A = f<u32:1, u32:2>(), B = if x < y { u8:1 } else { u8:0 }, }
pub enum Choice { Empty, Word(u32), }
pub fn build<T: type, N: u32 = {if X < Y {u32:2} else {u32:3}}>(x: T) -> (T, Local) {
    let s = " pub struct Fake { \" }"; let b = ` }`; (x, zero!<Frame>())
}
const X = if true { u8:0 } else { u8:1 };
"#;
        let parsed = parse_dslx_outline(source)?;
        assert_eq!(
            parsed.imports,
            BTreeMap::from([
                ("wire".into(), "net.protocol".into()),
                ("format".into(), "common.format".into()),
            ])
        );
        assert_eq!(parsed.declarations.len(), 5);
        assert!(!parsed.declarations["Local"].public);
        assert!(parsed.declarations["Frame"].public);
        let OutlineKind::Struct { fields } = &parsed.declarations["Frame"].kind else {
            panic!("expected struct")
        };
        assert_eq!(
            fields
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["header", "values"]
        );
        assert_eq!(
            fields[0].1.shape,
            OutlineShape::Name(vec!["wire".into(), "Header".into()])
        );
        assert_eq!(
            fields[1].1.syntax,
            "(wire::Header, uN[width()])[N // computed\n        + u32:1]"
        );
        let OutlineShape::Array(element) = &fields[1].1.shape else {
            panic!("expected array");
        };
        let OutlineShape::Tuple(items) = &element.shape else {
            panic!("expected tuple");
        };
        assert_eq!(
            items,
            &[
                fields[0].1.clone(),
                OutlineType {
                    syntax: "uN[width()]".into(),
                    shape: OutlineShape::Other
                }
            ]
        );
        let OutlineKind::Enum {
            underlying,
            members,
        } = &parsed.declarations["Kind"].kind
        else {
            panic!("expected enum")
        };
        assert_eq!(underlying.as_ref().unwrap().syntax, "uN[width()]");
        assert_eq!(
            members,
            &[
                ("A".into(), "f<u32:1, u32:2>()".into()),
                ("B".into(), "if x < y { u8:1 } else { u8:0 }".into()),
            ]
        );
        assert_eq!(parsed.declarations["Choice"].kind, OutlineKind::Opaque);
        assert_eq!(
            parsed.function_parametrics["build"]
                .iter()
                .map(|p| (p.name.as_str(), p.ty.syntax.as_str()))
                .collect::<Vec<_>>(),
            [("T", "type"), ("N", "u32")]
        );
        let OutlineShape::Tuple(result) = &parsed.functions["build"].shape else {
            panic!("expected tuple");
        };
        assert_eq!(result[0].shape, OutlineShape::Other);
        assert_eq!(result[1].shape, OutlineShape::Name(vec!["Local".into()]));
        Ok(())
    }

    #[test]
    fn outline_shares_use_trees_and_handles_nested_type_arguments() -> Result<(), String> {
        let module = parse_dslx_outline(
            "use pkg::{a::Outer, b::{Inner, Item}}; pub type Alias = Outer<Inner<Item>>[u32:2];",
        )?;
        assert_eq!(
            module.uses,
            [
                vec!["pkg", "a", "Outer"],
                vec!["pkg", "b", "Inner"],
                vec!["pkg", "b", "Item"],
            ]
        );
        let OutlineKind::Alias(ty) = &module.declarations["Alias"].kind else {
            panic!("expected alias")
        };
        let OutlineShape::Array(element) = &ty.shape else {
            panic!("expected array")
        };
        assert_eq!(element.syntax, "Outer<Inner<Item>>");
        assert_eq!(element.shape, OutlineShape::Name(vec!["Outer".into()]));
        // Native DSLX permits two equal leaf names in one tree and resolves
        // uses of that name to the final leaf.
        assert_eq!(
            parse_dslx_outline("use pkg::{a::Item, b::Item};")?.uses,
            [vec!["pkg", "a", "Item"], vec!["pkg", "b", "Item"]]
        );
        Ok(())
    }

    #[test]
    fn bang_names_before_equals_match_native_dslx_declarations()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = "pub type Byte!=u8; pub enum E:u8{A!=0}";
        let outline = parse_dslx_outline(source)?;
        let checked = parse_and_typecheck(source, "bang.x", "bang", &mut ImportData::default())?;
        let module = checked.get_module();
        let alias = module.get_type_definition_as_type_alias(0).unwrap();
        let enumeration = module.get_type_definition_as_enum_def(1).unwrap();
        assert_eq!(alias.get_identifier(), "Byte!");
        assert!(outline.declarations.contains_key(&alias.get_identifier()));
        let OutlineKind::Enum { members, .. } =
            &outline.declarations[&enumeration.get_identifier()].kind
        else {
            panic!("expected enum");
        };
        assert_eq!(members[0].0, enumeration.get_member(0).get_name());
        assert_eq!(members[0].0, "A!");
        Ok(())
    }

    #[test]
    fn outlined_fuzz_domains_match_native_names_and_visibility()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = r#"#[fuzz_domain("\x50ub\u{44}omain\'!")]
#[sv_type("pkg::packet_t")]
pub struct Packet { value: u8 }
#[fuzz_domain("PrivateDomain")]
struct Hidden { value: u8 }
#[fuzz_domain("bad name")]
pub struct Odd { value: u8 }
#[fuzz_domain("fn")]
pub struct Keyword { value: u8 }
#[fuzz_domain("u8")]
pub struct Builtin { value: u8 }"#;
        let outline = parse_dslx_outline(source)?;
        let mut data = ImportData::default();
        let checked = parse_and_typecheck(source, "domain.x", "domain", &mut data)?;
        let module = checked.get_module();
        let names: BTreeSet<_> = (0..module.get_type_definition_count())
            .map(|i| {
                module
                    .get_type_definition_as_struct_def(i)
                    .unwrap()
                    .get_identifier()
            })
            .collect();
        for (name, public) in [("PubDomain'!", true), ("PrivateDomain", false)] {
            assert!(names.contains(name));
            assert_eq!(
                outline.declarations[name],
                OutlineDecl {
                    public,
                    parametric: false,
                    kind: OutlineKind::GeneratedStruct,
                }
            );
        }
        for unusable in ["bad name", "fn", "u8"] {
            assert!(names.contains(unusable));
            assert!(!outline.declarations.contains_key(unusable));
        }
        parse_and_typecheck(
            "import domain; fn public_consumer(x: domain::PubDomain'!) {}",
            "public_user.x",
            "public_user",
            &mut data,
        )?;
        assert!(
            parse_and_typecheck(
                "import domain; fn private_consumer(x: domain::PrivateDomain) {}",
                "private_user.x",
                "private_user",
                &mut data,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn unused_deep_array_retains_syntax_without_a_deep_shape() -> Result<(), String> {
        let deep = format!("u1{}", "[1]".repeat(MAX_SYNTAX_DEPTH + 1));
        let outline = parse_dslx_outline(&format!("type Deep = {deep}; fn id(x:u8) -> u8 {{x}}"))?;
        assert_eq!(
            outline.declarations["Deep"].kind,
            OutlineKind::Alias(OutlineType {
                syntax: deep,
                shape: OutlineShape::Other,
            })
        );
        assert_eq!(outline.functions["id"].syntax, "u8");
        Ok(())
    }

    #[test]
    fn outline_parses_bundled_dslx_standard_library() -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(xlsynth::default_dslx_stdlib_path())? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "x") {
                parse_dslx_outline(&fs::read_to_string(&path)?)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        Ok(())
    }
}
