// SPDX-License-Identifier: Apache-2.0

//! Locate checked DSLX functions and their calls in the frozen import graph.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};

use xlsynth::dslx::{Function, MatchableModuleMember, Module};

use super::ImportedStructs;
use super::outline_graph::SourceModule;
use crate::effect_diagnostic::{EffectLocation, const_assert_replacement};
use crate::parser::outline::{DslxFunctionTokens, dslx_function_tokens};
use crate::parser::{Token, is_dslx_member_name, is_use_identifier};

/// Lookup by native module identity, built only for a rejected signature probe.
pub(crate) struct EffectSources<'a> {
    modules: BTreeMap<String, CapturedModule<'a>>,
}

/// Original bytes and a lazily built index of checked top-level declarations.
struct CapturedModule<'a> {
    module: &'a SourceModule,
    index: OnceCell<Option<FunctionIndex<'a>>>,
}

/// Written bodies and actual native declarations share a unique name per
/// module.
struct FunctionIndex<'a> {
    source: &'a str,
    declarations: BTreeMap<String, DslxFunctionTokens<'a>>,
    definitions: BTreeMap<String, Function>,
    constants: BTreeSet<String>,
}

/// Source candidates for the caller's checked outgoing edges.
pub(crate) struct EffectFunctionSource<'a> {
    pub(crate) subject: &'a str,
    pub(crate) effects: Vec<EffectSourceOperation<'a>>,
    pub(crate) calls: Vec<EffectSourceCall<'a>>,
    /// Earliest possible effect for which the native graph lacks source
    /// ownership (e.g. a method or an assertion inside a lambda).
    pub(crate) opaque_offset: Option<usize>,
}

/// An original DSLX operation and an optional compile-time assertion rewrite.
pub(crate) struct EffectSourceOperation<'a> {
    pub(crate) location: EffectLocation<'a>,
    pub(crate) const_assert_replacement: Option<String>,
}

/// A written call, resolved in the declaration scope for ordering native edges.
pub(crate) struct EffectSourceCall<'a> {
    pub(crate) subject: &'a str,
    pub(crate) name: &'a str,
    pub(crate) offset: usize,
    pub(crate) opaque: bool,
    pub(crate) unconditional: bool,
}

impl ImportedStructs {
    pub(crate) fn effect_sources(&self) -> EffectSources<'_> {
        let files = self
            .files
            .as_ref()
            .expect("DSLX imports have a source snapshot");
        EffectSources {
            modules: self
                .graph
                .sources()
                .map(|module| {
                    (
                        files.native_module_name(&module.subject),
                        CapturedModule {
                            module,
                            index: OnceCell::new(),
                        },
                    )
                })
                .collect(),
        }
    }
}

impl<'a> EffectSources<'a> {
    pub(crate) fn subject(&self, owner: &Module) -> Option<&'a str> {
        Some(&self.modules.get(&owner.get_name())?.module.subject)
    }

    /// Match the checked top-level declaration without scanning its body.
    fn declaration(
        &self,
        function: &Function,
    ) -> Option<(&FunctionIndex<'a>, &'a SourceModule, String)> {
        let owner = function.get_body().get_owner_module();
        let captured = self.modules.get(&owner.get_name())?;
        let module = captured.module;
        let index = captured
            .index
            .get_or_init(|| FunctionIndex::new(module, &owner))
            .as_ref()?;
        let name = function.get_identifier();
        if !index.definitions.get(&name)?.is_same_definition(function) {
            return None;
        }
        Some((index, module, name))
    }

    pub(crate) fn is_named_definition(&self, function: &Function) -> bool {
        self.declaration(function).is_some()
    }

    /// Check declaration identity before pairing a native function with source.
    pub(crate) fn function(&self, function: &Function) -> Option<EffectFunctionSource<'a>> {
        let (index, module, name) = self.declaration(function)?;
        let declaration = index.declarations.get(&name)?;
        Some(index.sites(module, function, declaration))
    }
}

impl<'a> FunctionIndex<'a> {
    fn new(module: &'a SourceModule, owner: &Module) -> Option<Self> {
        let source = std::str::from_utf8(&module.source).ok()?;
        let declarations = dslx_function_tokens(source).ok()?;
        let mut definitions = BTreeMap::new();
        let mut constants = BTreeSet::new();
        for i in 0..owner.get_member_count() {
            match owner.get_member(i).to_matchable() {
                Some(MatchableModuleMember::Function(f)) => {
                    definitions.insert(f.get_identifier(), f);
                }
                Some(MatchableModuleMember::ConstantDef(c)) => {
                    constants.insert(c.get_name());
                }
                _ => { /* Other declarations have no callable value here. */ }
            }
        }
        Some(Self {
            source,
            declarations,
            definitions,
            constants,
        })
    }

    /// Recover source order for ordinary calls; native edges confirm their
    /// identity.
    fn sites(
        &self,
        module: &'a SourceModule,
        function: &Function,
        declaration: &DslxFunctionTokens<'a>,
    ) -> EffectFunctionSource<'a> {
        let mut sites = EffectFunctionSource {
            subject: &module.subject,
            effects: Vec::new(),
            calls: Vec::new(),
            opaque_offset: None,
        };
        let shadowed = shadowed_names(function, &declaration.body);
        let compile_time_names: Vec<_> = function
            .parametric_bindings()
            .into_iter()
            .map(|p| p.get_identifier())
            .chain(self.constants.iter().cloned())
            .filter(|name| !shadowed.contains(name))
            .collect();
        for (body, call) in declaration_calls(declaration) {
            let end = body[call.close].start + 1;
            let Some(name) = call.name else {
                sites.opaque_offset.get_or_insert(end);
                continue;
            };
            let callee = body[name];
            let qualifier = (call.start != name).then(|| body[call.start].text);
            if matches!(body[call.start - 1].text, "::" | ".")
                || shadowed.contains(body[call.start].text)
            {
                // The typechecker can omit edges for receiver methods and
                // indirect calls. A later effect may not be the first one.
                sites.opaque_offset.get_or_insert(end);
                continue;
            }
            let Some(subject) = module.call_subject(qualifier, callee.text) else {
                // An associated function such as Foo::check is not an import.
                sites.opaque_offset.get_or_insert(end);
                continue;
            };
            if qualifier.is_some()
                || subject != module.subject
                || self.definitions.contains_key(callee.text)
            {
                sites.calls.push(EffectSourceCall {
                    subject,
                    name: callee.text,
                    offset: end,
                    opaque: call.opaque,
                    unconditional: call.unconditional,
                });
                continue;
            }
            if callee.text == "map" && !self.constants.contains("map") {
                // DSLX applies a named map callback after evaluating the array.
                // Other callback expressions need native source information.
                let actual = map_callback_argument(&body[call.open + 1..call.close]);
                if let Some((qualifier, name)) = named_map_callback(actual)
                    && !shadowed.contains(qualifier.unwrap_or(name))
                    && let Some(subject) = module.call_subject(qualifier, name)
                    && (qualifier.is_some()
                        || subject != module.subject
                        || self.definitions.contains_key(name))
                {
                    sites.calls.push(EffectSourceCall {
                        subject,
                        name,
                        offset: end,
                        opaque: call.opaque,
                        // A map over an empty array does not call its callback.
                        unconditional: false,
                    });
                } else if !actual.first().is_some_and(|t| matches!(t.text, "|" | "||")) {
                    // Calls inside an anonymous callback are tracked above;
                    // a different indirect callback has no known definition.
                    sites.opaque_offset.get_or_insert(end);
                }
                continue;
            }
            if !is_token_builtin(callee.text) || self.constants.contains(callee.text) {
                continue;
            }
            if call.opaque {
                sites.opaque_offset.get_or_insert(end);
                continue;
            }
            // const_assert! is a statement, even when assert! can appear in
            // an unconditional let initializer or tail expression.
            let replacement = if call.unconditional
                && matches!(body[call.start - 1].text, "{" | ";")
                && body.get(call.close + 1).is_some_and(|t| t.text == ";")
            {
                const_assert_replacement(
                    callee.text,
                    &arguments(self.source, &body[call.open + 1..call.close]),
                    &compile_time_names,
                )
            } else {
                None
            };
            sites.effects.push(EffectSourceOperation {
                location: EffectLocation {
                    path: &module.source_path,
                    source: self.source,
                    range: callee.start..end,
                    operation: callee.text,
                },
                const_assert_replacement: replacement,
            });
        }
        sites
    }
}

/// Local names whose spelling cannot reliably identify a module declaration or
/// builtin.
fn shadowed_names(function: &Function, body: &[Token<'_>]) -> BTreeSet<String> {
    let mut names: BTreeSet<_> = (0..function.get_param_count())
        .map(|i| function.get_param(i).get_name())
        .collect();
    for (i, token) in body.iter().enumerate() {
        if token.text == "let" {
            names.extend(
                body[i + 1..]
                    .iter()
                    .take_while(|t| !matches!(t.text, ":" | "="))
                    .filter(|t| is_dslx_member_name(t.text))
                    .map(|t| t.text.to_owned()),
            );
        } else if (is_token_builtin(token.text) || token.text == "map")
            && body.get(i + 1).is_none_or(|next| next.text != "(")
        {
            // E.g. a pattern binding or an indirectly applied function.
            names.insert(token.text.to_owned());
        }
    }
    names
}

/// Read the callback argument after the last top-level comma of a checked map.
fn map_callback_argument<'s, 'a>(args: &'s [Token<'a>]) -> &'s [Token<'a>] {
    let args = match args {
        [args @ .., last] if last.text == "," => args,
        args => args,
    };
    let mut depth = 0usize;
    for (i, token) in args.iter().enumerate().rev() {
        match token.text {
            ")" | "}" | "]" => depth += 1,
            "(" | "{" | "[" => depth -= 1,
            "," if depth == 0 => return &args[i + 1..],
            _ => { /* Only the separator before the callback matters. */ }
        }
    }
    &[]
}

/// Match a bare or module-qualified function reference with optional grouping.
fn named_map_callback<'a>(mut actual: &[Token<'a>]) -> Option<(Option<&'a str>, &'a str)> {
    while let [open, rest @ .., close] = actual
        && open.text == "("
        && close.text == ")"
    {
        actual = rest;
    }
    match actual {
        [module, sep, name]
            if sep.text == "::"
                && is_dslx_member_name(module.text)
                && is_dslx_member_name(name.text) =>
        {
            Some((Some(module.text), name.text))
        }
        [name] if is_dslx_member_name(name.text) => Some((None, name.text)),
        _ => None,
    }
}

/// Token indices of a written function call, including its optional module
/// qualifier.
struct WrittenCall {
    start: usize,
    name: Option<usize>,
    open: usize,
    close: usize,
    unconditional: bool,
    opaque: bool,
}

/// A default binding or signature dimension can propagate a token through the
/// native graph. Check these before the body without attributing a runtime
/// span.
fn declaration_calls<'d, 's>(
    function: &'d DslxFunctionTokens<'s>,
) -> impl Iterator<Item = (&'d [Token<'s>], WrittenCall)> {
    [
        (function.bindings.as_slice(), true),
        (function.signature.as_slice(), true),
        (function.body.as_slice(), false),
    ]
    .into_iter()
    .flat_map(|(tokens, header)| {
        written_calls(tokens).into_iter().map(move |mut call| {
            if header {
                call.opaque = true;
                call.unconditional = false;
            }
            (tokens, call)
        })
    })
}

/// Collect calls in argument evaluation order. The checked graph can merge
/// calls from lambdas, loops and constexpr contexts into the caller's edges.
fn written_calls(body: &[Token<'_>]) -> Vec<WrittenCall> {
    let mut calls: Vec<WrittenCall> = Vec::new();
    let mut stack = Vec::new();
    let mut first_call = 0;
    let mut statement_start = 1;
    let mut opaque_statement = false;
    for (i, token) in body.iter().enumerate() {
        let ambiguous = match token.text {
            "for" | "unroll_for!" | "const" | "const_assert!" => true,
            "|" | "||" => !body[..i].last().is_some_and(|prev| {
                let text = prev.text;
                matches!(text, ")" | "]" | "}")
                    || (is_dslx_member_name(text)
                        && !matches!(text, "if" | "else" | "match" | "in"))
                    || text
                        .starts_with(|c: char| c.is_ascii_digit() || matches!(c, '-' | '\'' | '"'))
            }),
            _ => false,
        };
        opaque_statement |= ambiguous;
        match token.text {
            "(" | "[" | "{" => {
                if token.text == "("
                    && i > 0
                    && body[i - 1].text == ">"
                    && let Some(name) = call_name(body, i)
                {
                    // DSLX can propagate an implicit token from a value
                    // parameter even though its expression is evaluated at
                    // compile time. Keep those calls as potential effects.
                    for call in calls.iter_mut().rev().take_while(|c| c.close > name) {
                        call.opaque = true;
                        call.unconditional = false;
                    }
                }
                stack.push(i);
            }
            ")" | "]" | "}" => {
                let open = stack.pop().expect("DSLX outline groups are balanced");
                if token.text == ")"
                    && let Some(name) = call_name(body, open)
                {
                    let start = if name >= 2 && body[name - 1].text == "::" {
                        name - 2
                    } else {
                        name
                    };
                    if start > 0 {
                        calls.push(WrittenCall {
                            start,
                            name: Some(name),
                            open,
                            close: i,
                            opaque: false,
                            unconditional: stack.len() == 1
                                && (start == statement_start
                                    || (body[start - 1].text == "="
                                        && body[statement_start].text == "let"))
                                && body.get(i + 1).is_some_and(|t| matches!(t.text, "}" | ";")),
                        });
                    }
                } else if token.text == ")"
                    && open > 0
                    && matches!(body[open - 1].text, ")" | "]" | ">")
                {
                    // An argument list on an expression, such as (f)(x) or
                    // callbacks[i](x), is not a direct named invocation.
                    calls.push(WrittenCall {
                        start: open - 1,
                        name: None,
                        open,
                        close: i,
                        opaque: true,
                        unconditional: false,
                    });
                }
            }
            _ => { /* Other tokens cannot complete an invocation. */ }
        }
        if (token.text == ";" && stack.len() == 1) || stack.is_empty() {
            if opaque_statement {
                for call in &mut calls[first_call..] {
                    call.opaque = true;
                    call.unconditional = false;
                }
            }
            first_call = calls.len();
            statement_start = i + 1;
            opaque_statement = false;
        }
    }
    calls
}

/// Recognize a bare name or skip explicit DSLX value/type arguments. Angle
/// operators inside braced expressions and array dimensions are ordinary
/// tokens.
fn call_name(body: &[Token<'_>], open: usize) -> Option<usize> {
    let mut head = open.checked_sub(1)?;
    if body[head].text == ">" {
        let mut groups = 0usize;
        let mut angles = 1usize;
        while angles != 0 {
            head = head.checked_sub(1)?;
            let token = body[head].text;
            match token {
                ")" | "]" | "}" => groups += 1,
                "(" | "[" | "{" => groups = groups.checked_sub(1)?,
                _ if groups != 0 => { /* The entire balanced group is one argument. */ }
                ">" => angles += 1,
                "<" => angles -= 1,
                ":" | "," | "::" => { /* Punctuation within a value or type argument. */ }
                _ if is_dslx_member_name(token)
                    || token.starts_with(|c: char| c.is_ascii_digit()) =>
                { /* Name or literal. */ }
                _ => return None,
            }
        }
        head = head.checked_sub(1)?;
    }
    let name = body[head].text;
    // const_assert! is a declaration. Its predicate can still contain calls.
    (name != "const_assert!" && (is_use_identifier(name) || name == "self")).then_some(head)
}

/// The builtins that can contribute an implicit token in the pinned DSLX
/// frontend.
pub(crate) fn is_token_builtin(name: &str) -> bool {
    matches!(
        name,
        "assert!"
            | "assert_eq"
            | "assert_ne"
            | "assert_lt"
            | "assert_le"
            | "assert_gt"
            | "assert_ge"
            | "assert_fmt!"
            | "cover!"
            | "fail!"
            | "trace!"
            | "trace_fmt!"
            | "vtrace_fmt!"
    )
}

/// Slice arguments at top-level commas; whitespace and comments remain intact.
fn arguments<'a>(source: &'a str, tokens: &[Token<'a>]) -> Vec<&'a str> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (end, t) in tokens.iter().enumerate() {
        match t.text {
            "(" | "{" | "[" => depth += 1,
            ")" | "}" | "]" => depth -= 1,
            "," if depth == 0 => {
                if start < end {
                    let last = tokens[end - 1];
                    result.push(&source[tokens[start].start..last.start + last.text.len()]);
                }
                start = end + 1;
            }
            _ => { /* The token belongs to the current argument. */ }
        }
    }
    if let Some(last) = tokens.get(start..).and_then(|tail| tail.last()) {
        result.push(&source[tokens[start].start..last.start + last.text.len()]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs;

    #[test]
    fn source_calls_mark_opaque_bodies_and_visit_arguments_first() -> Result<(), String> {
        let source = r#"fn f<N: u32, M: u32 = {width(u8:1)}>(x: uN[{dim(N)}]) -> uN[{dim(M)}] {
    let callback = |y: u8| { fail!("nested", y) };
    for (_, acc): (u32, u8) in u32:0..u32:0 { trace!(acc) }(x);
    const_assert!(true);
    const K = assert!(true, "constant");
    const_assert!(helper(u8:1) != u8:0);
    assert_eq(helper(x | u8:1), u8:1);
    let y = helper<{width(x) + u32:1}>(x);
    assert!(x != u8:0, "data");
    ((helper))(x);
    helper(y)
}"#;
        let functions = dslx_function_tokens(source)?;
        let calls: Vec<_> = declaration_calls(&functions["f"])
            .map(|(tokens, c)| {
                (
                    c.name.map_or("<expression>", |i| tokens[i].text),
                    c.opaque,
                    c.unconditional,
                )
            })
            .collect();
        assert_eq!(
            calls,
            [
                ("width", true, false),
                ("dim", true, false),
                ("dim", true, false),
                ("fail!", true, false),
                ("trace!", true, false),
                ("assert!", true, false),
                ("helper", true, false),
                ("helper", false, false),
                ("assert_eq", false, true),
                ("width", true, false),
                ("helper", false, true),
                ("assert!", false, true),
                ("<expression>", true, false),
                ("helper", false, true),
            ]
        );
        Ok(())
    }

    #[test]
    fn diagnostic_uses_frozen_original_tokens_of_the_checked_declaration()
    -> Result<(), Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let math = root.path().join("math");
        fs::create_dir(&math)?;
        fs::write(
            root.path().join("identity.x"),
            "pub fn id(x: u8) -> u8 { x }",
        )?;
        let path = math.join("stage.x");
        let source = r#"// UTF-8 µ. assert!(false, "decoy")
import identity as h; const MIN_WIDTH = u32:2; pub fn checked<N: u32>(x: uN[N]) -> u8 { const_assert!(N >= u32:1); assert!(N >= MIN_WIDTH, "width");
    let _ = "assert!(false)";
    assert! // Source trivia between the callee and its arguments.
        (x == uN[N]:0 || x == uN[N]:1, "data");
    let _unit = assert!(N >= u32:1, "initializer");
    let _mapped = map([x as u8], (h::id));
    h::id(x as u8)
}
fn unused() { assert!(false, "unused"); }
"#;
        fs::write(&path, source)?;
        let uses = crate::parse_module("use math::stage;")?.uses;
        let (structs, mut imports) =
            ImportedStructs::load(&uses, xlsynth::default_dslx_stdlib_path(), &[root.path()])?;
        // Source offsets must survive both import relocation and a later edit.
        fs::write(&path, "pub fn checked(x: u8) -> u8 { x }")?;
        let probe = structs.check_probe(
            "import math.stage; fn probe(x: u8) -> u8 { stage::checked(x) }",
            "0bx_diagnostic_test",
            imports.as_mut().ok_or("missing native snapshot")?,
        )?;
        let graph = probe
            .get_type_info()
            .build_function_call_graph(&probe.get_module())?;
        let caller = graph.get_function(0).ok_or("missing probe")?;
        let checked = (0..graph.callee_count(&caller))
            .filter_map(|i| graph.get_callee(&caller, i))
            .find(|f| f.get_identifier() == "checked")
            .ok_or("missing checked function")?;
        let sources = structs.effect_sources();
        let sites = sources
            .function(&checked)
            .ok_or("missing checked declaration source")?;
        assert_eq!(sites.subject, "math.stage");
        assert_eq!(sites.opaque_offset, None);
        let expected = [
            (
                r#"assert!(N >= MIN_WIDTH, "width")"#,
                Some("const_assert!(N >= MIN_WIDTH)"),
            ),
            (
                r#"assert! // Source trivia between the callee and its arguments.
        (x == uN[N]:0 || x == uN[N]:1, "data")"#,
                None,
            ),
            (r#"assert!(N >= u32:1, "initializer")"#, None),
        ];
        assert_eq!(sites.effects.len(), expected.len());
        for (op, (text, replacement)) in sites.effects.iter().zip(expected) {
            let start = source.find(text).ok_or("expected operation in fixture")?;
            assert_eq!(op.location.range, start..start + text.len());
            assert_eq!(op.location.path, path);
            assert_eq!(op.location.source, source);
            assert_eq!(op.const_assert_replacement.as_deref(), replacement);
        }
        let calls: Vec<_> = sites
            .calls
            .iter()
            .map(|c| (c.subject, c.name, c.offset))
            .collect();
        let call = "h::id(x as u8)";
        let mapped = "map([x as u8], (h::id))";
        assert_eq!(
            calls,
            [
                (
                    "identity",
                    "id",
                    source.find(mapped).unwrap() + mapped.len()
                ),
                ("identity", "id", source.find(call).unwrap() + call.len())
            ]
        );
        Ok(())
    }
}
