// SPDX-License-Identifier: Apache-2.0

//! The pinned converter aborts on ordered enum comparisons. Ask DSLX to
//! typecheck a private copy where comparisons require bits and declarations
//! are instantiated by the selected calls.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Range;

use xlsynth::dslx;
use xlsynth::ir_value::IrFormatPreference;

use super::{ImportedStructs, NativeSourceFiles};
use crate::OutlineKind;
use crate::parser::outline::visit_conversion_items;
use crate::parser::source_bytes::{
    SourceColumns, SourceEdit, apply_source_edits, decode_dslx_source_with_columns,
    native_display_tokens_source,
};
use crate::parser::{Token, is_dslx_member_name, is_identifier_word};

fn ordered(token: &Token<'_>) -> bool {
    matches!(token.text, "<" | ">" | "<=" | ">=")
}

/// Without a pair of generic angle delimiters, each ordering token in the
/// written body is a comparison. This also works after a native display was
/// truncated by a NUL character literal.
fn source_orderings(body: &[Token<'_>]) -> Option<Vec<usize>> {
    let mut open = false;
    let mut offsets = Vec::new();
    let mut tokens = body.iter().peekable();
    while let Some(token) = tokens.next() {
        match token.text {
            "<" => open = true,
            ">" | ">=" | ">>" if open => return None,
            ">" if tokens
                .next_if(|next| next.text == ">" && next.start == token.start + 1)
                .is_some() =>
            {
                // The DSLX outline scanner splits >> for nested type arguments.
                // Without an opening angle these adjacent tokens are a shift.
                continue;
            }
            _ => { /* These tokens cannot close a type argument list. */ }
        }
        if ordered(token) {
            offsets.push(token.start);
        }
    }
    Some(offsets)
}

/// Edits to one captured module, pending a comparison that needs guarding.
struct GuardedModule<'a> {
    subject: &'a str,
    source: Cow<'a, str>,
    columns: SourceColumns,
    edits: Vec<(Range<usize>, String)>,
    constant_calls: BTreeMap<String, ConstantCall>,
    declarations_start: usize,
    checked: Option<dslx::TypecheckedModule>,
}

/// Initializer containing a call and the name boundary for an unused private
/// declaration whose annotation can be elided along with its value.
struct ConstantCall {
    expression: Range<usize>,
    unused_name_end: Option<usize>,
}

/// v0.61 permits cross-module use trees to name private constants. Only
/// discard an initializer if its name appears exactly once in the whole graph.
fn unused_private_constants(modules: &[GuardedModule<'_>]) -> BTreeSet<String> {
    let mut unused: BTreeSet<String> = modules
        .iter()
        .flat_map(|m| &m.constant_calls)
        .filter(|(_, call)| call.unused_name_end.is_some())
        .map(|(name, _)| name.clone())
        .collect();
    let mut seen = BTreeSet::new();
    for module in modules {
        if unused.is_empty() {
            break;
        }
        for word in module
            .source
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        {
            if unused.contains(word) && !seen.insert(word) {
                unused.remove(word);
            }
        }
    }
    unused
}

impl ImportedStructs {
    /// Native inference supplies the call graph, including map, loops, local
    /// aliases and impl dispatch. IR conversion uses the original comparisons.
    pub(crate) fn check_enum_order(&self, source: &str, module: &str) -> Result<(), String> {
        let files = self
            .enum_order_snapshot
            .get_or_init(|| self.make_enum_order_snapshot())
            .as_ref()
            .map_err(Clone::clone)?;
        let Some(files) = files else { return Ok(()) };
        let native_source = files.native_source(source);
        let path = format!("{module}.x");
        dslx::parse_and_typecheck(&native_source, &path, module, &mut files.import_data())
            .map_err(|error| {
                let diagnostic = files.remap_diagnostic(error.to_string());
                format!(
                    "cannot verify ordered comparisons in imported DSLX with libxls v0.61. \
                     For an ordered comparison, cast enum operands to their backing bits type.\n{diagnostic}"
                )
            })?;
        Ok(())
    }

    /// Cache the transformed sources across all call specializations. Modules
    /// without plausible comparison operators need no native pretty-print.
    fn make_enum_order_snapshot(&self) -> Result<Option<NativeSourceFiles>, String> {
        if !self.graph.sources().any(|s| {
            s.outline
                .declarations
                .values()
                .any(|decl| matches!(decl.kind, OutlineKind::Enum { .. }))
        }) {
            return Ok(None);
        }
        let mut data = self.fresh_import_data();
        let mut modules = Vec::new();
        let mut any_guarded = false;
        for (i, original) in self.graph.sources().enumerate() {
            let (written, columns) = decode_dslx_source_with_columns(&original.source)?;
            let mut possible_order = false;
            let mut impls = BTreeSet::new();
            let mut constant_calls = BTreeMap::new();
            let declarations_start = visit_conversion_items(&written, |item| {
                if item.kind == "impl" {
                    impls.insert(item.name.text);
                }
                possible_order |= item.kind == "fn" && item.body.iter().any(ordered);
                if item.kind == "const"
                    && item.body.windows(2).any(|pair| {
                        pair[1].text == "("
                            && (is_dslx_member_name(pair[0].text)
                                || matches!(pair[0].text, ">" | ")" | "]" | "}"))
                    })
                {
                    let last = item.body.last().expect("nonempty call");
                    constant_calls.insert(
                        item.name.text.to_owned(),
                        ConstantCall {
                            expression: item.body[0].start..last.start + last.text.len(),
                            // Splitting at punctuation overcounts references to
                            // ordinary names; leave names containing primes or
                            // bangs to reflection.
                            unused_name_end: (is_identifier_word(item.name.text)
                                && !original.outline.public_values.contains(item.name.text))
                            .then_some(item.name.start + item.name.text.len()),
                        },
                    );
                }
            })?;
            let mut bodies = NativeComparisons::default();
            let mut checked_constants = None;
            if possible_order {
                let module = format!("0bx_enum_order_{i}");
                let relocated = self
                    .files
                    .as_ref()
                    .expect("captured DSLX")
                    .relocate_source(&written);
                let native = String::from_utf8(relocated.bytes).map_err(|e| e.to_string())?;
                let checked =
                    dslx::parse_and_typecheck(&native, &format!("{module}.x"), &module, &mut data)
                        .map_err(|e| self.remap_diagnostic(e.to_string()))?;
                bodies = NativeComparisons::collect(
                    &checked.get_module(),
                    &written,
                    &module,
                    &relocated.columns,
                );
                if !constant_calls.is_empty() {
                    checked_constants = Some(checked);
                }
            }
            let binding = fresh_identifier(&written, "_BX_ENUM_ORDER");
            let param = format!("<{binding}:u32={{u32:0}}>");
            let mut edits = Vec::new();

            visit_conversion_items(&written, |item| {
                if matches!(item.kind, "fn" | "proc") && !item.associated && !item.parametric
                    || item.kind == "struct"
                        && !item.parametric
                        && !item.fuzz_domain
                        && impls.contains(item.name.text)
                {
                    // Parametric tests are illegal. Their test-only attributes
                    // have no role in checking pure imported call signatures.
                    for attr in item.test_attributes {
                        edits.push((attr.clone(), String::new()));
                    }
                    let at = item.name.start + item.name.text.len();
                    edits.push((at..at, param.clone()));
                }
                if possible_order && item.kind == "fn" && !item.body.is_empty() {
                    let roles = bodies
                        .functions
                        .get_mut(&(item.name.text.to_owned(), item.associated))
                        .and_then(VecDeque::pop_front);
                    if item.body.iter().any(ordered) {
                        // Leave source spelling, including lambda captures,
                        // intact. Unpaired roles fail closed if this function
                        // is instantiated.
                        match roles
                            .and_then(|roles| bodies.insertions(item.body, &roles))
                            .or_else(|| source_orderings(item.body))
                        {
                            Some(offsets) => {
                                any_guarded |= !offsets.is_empty();
                                edits.extend(
                                    offsets.into_iter().map(|at| (at..at, " << u1:0 ".into())),
                                );
                            }
                            None => {
                                any_guarded = true;
                                let at = item.body[0].start + 1;
                                edits.push((at..at, " let _:u1 = u2:0; ".into()));
                            }
                        }
                    }
                }
            })?;
            modules.push(GuardedModule {
                subject: &original.subject,
                source: written,
                columns,
                edits,
                constant_calls,
                declarations_start,
                checked: checked_constants,
            });
        }
        if !any_guarded {
            return Ok(None);
        }
        let unused = unused_private_constants(&modules);
        let mut overrides = BTreeMap::new();
        for (i, mut module) in modules.into_iter().enumerate() {
            module.constant_calls.retain(|name, call| {
                if let Some(start) = call.unused_name_end
                    && unused.contains(name)
                {
                    module
                        .edits
                        .push((start..call.expression.end, " = ()".into()));
                    false
                } else {
                    true
                }
            });
            if !module.constant_calls.is_empty() {
                let checked = match module.checked {
                    Some(checked) => checked,
                    None => self.check_probe(
                        &module.source,
                        &format!("0bx_enum_constants_{i}"),
                        &mut data,
                    )?,
                };
                freeze_scalar_constants(
                    &checked,
                    &module.constant_calls,
                    &module.source,
                    module.declarations_start,
                    &mut module.edits,
                );
            }
            if !module.edits.is_empty() {
                module.edits.sort_by_key(|(range, _)| range.start);
                let edits = module
                    .edits
                    .into_iter()
                    .map(|(range, text)| SourceEdit { range, text })
                    .collect();
                let mut rewritten = apply_source_edits(module.source.as_bytes(), edits);
                rewritten.columns = rewritten.columns.with_original(module.columns);
                overrides.insert(module.subject.to_owned(), rewritten);
            }
        }
        NativeSourceFiles::capture_with_overrides(&self.graph, overrides).map(Some)
    }
}

/// Preserve native scalar constants without instantiating their initializer's
/// callees in the guarded module. Nominal and aggregate types retain their
/// original expressions.
fn freeze_scalar_constants(
    checked: &dslx::TypecheckedModule,
    calls: &BTreeMap<String, ConstantCall>,
    source: &str,
    declarations_start: usize,
    edits: &mut Vec<(Range<usize>, String)>,
) {
    let info = checked.get_type_info();
    let module = checked.get_module();
    let mut missing_value_enum = None;
    for i in 0..module.get_member_count() {
        let Some(dslx::MatchableModuleMember::ConstantDef(constant)) =
            module.get_member(i).to_matchable()
        else {
            continue;
        };
        let Some(call) = calls.get(&constant.get_name()) else {
            continue;
        };
        let Some((signed, width)) = info
            .try_get_type_for_constant_def(&constant)
            .and_then(|ty| ty.is_bits_like())
        else {
            continue;
        };
        let prefix = if signed { "sN" } else { "uN" };
        let ty = format!("{prefix}[{width}]");
        let initializer = match info.get_const_expr(&constant.get_value()) {
            Ok(value) => {
                let Ok(bits) = value.convert_to_ir().and_then(|v| v.to_bits()) else {
                    continue;
                };
                format!(
                    "{ty}:{}",
                    bits.to_string_fmt(IrFormatPreference::Hex, false)
                )
            }
            // v0.61 may typecheck a declaration without obtaining a value
            // (e.g. its bytecode interpreter cannot compare enum operands).
            // Casting zero to a fresh enum whose only member is one likewise
            // typechecks without producing a constexpr value. IR conversion
            // uses the original expression and diagnoses it if required.
            Err(error) if error.0.starts_with("NOT_FOUND:") => {
                let name = missing_value_enum
                    .get_or_insert_with(|| fresh_identifier(source, "BxUnavailableConstant"));
                format!("(u1:0 as {name}) as {ty}")
            }
            Err(_) => continue,
        };
        edits.push((call.expression.clone(), initializer));
    }
    if let Some(name) = missing_value_enum {
        edits.push((
            declarations_start..declarations_start,
            format!("enum {name}:u1 {{ Present=1 }} "),
        ));
    }
}

/// Choose a name that cannot shadow any identifier in this source.
fn fresh_identifier(source: &str, prefix: &str) -> String {
    let occupied: BTreeSet<_> = source
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|id| id.starts_with(prefix))
        .collect();
    (0usize..)
        .map(|i| format!("{prefix}{i}"))
        .find(|n| !occupied.contains(n.as_str()))
        .expect("finite input contains finitely many names")
}

/// Native syntax distinguishes comparisons from type/instantiation angle
/// brackets. Typechecking hoists each lambda into an impl with a source span.
#[derive(Default)]
struct NativeComparisons {
    functions: BTreeMap<(String, bool), VecDeque<Vec<Angle>>>,
    lambdas: BTreeMap<usize, (usize, Vec<Angle>)>,
}

/// The role and spelling of one angle token in a native expression body.
struct Angle {
    spelling: String,
    compare: bool,
}

impl Angle {
    fn from_token(printed: &str, token: Token<'_>) -> Self {
        let bytes = printed.as_bytes();
        Self {
            spelling: token.text.to_owned(),
            compare: bytes.get(token.start.wrapping_sub(1)) == Some(&b' ')
                && bytes.get(token.start + token.text.len()) == Some(&b' '),
        }
    }
}

impl NativeComparisons {
    fn collect(
        native: &dslx::Module,
        written: &str,
        module: &str,
        columns: &SourceColumns,
    ) -> Self {
        let display = native.to_text();
        let printed = native_display_tokens_source(&display);
        let mut result = Self::default();
        let synthetic = format!("lambda_capture_struct_at_{module}.x:");
        let line_starts: Vec<_> = std::iter::once(0)
            .chain(written.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        let offset = |position: &str| -> Option<usize> {
            let (line, col) = position.split_once(':')?;
            let line = line.parse::<usize>().ok()?.checked_sub(1)?;
            let col = col.parse::<usize>().ok()?.checked_sub(1)?;
            Some(line_starts.get(line)? + columns.original_column(line, col))
        };
        let mut lambda = None;
        // The C API ends its display at the first NUL. Keep complete
        // declarations from the prefix. The caller checks any unpaired body
        // against its original tokens and fails closed on ambiguous angles.
        let complete = visit_conversion_items(&printed, |item| {
            if item.kind == "impl" {
                lambda = printed[item.name.start..]
                    .strip_prefix(&synthetic)
                    .and_then(|tail| {
                        let span = tail
                            .split(|c: char| !(c.is_ascii_digit() || c == ':' || c == '-'))
                            .next()?;
                        let (start, end) = span.split_once('-')?;
                        Some(offset(start)?..offset(end)?)
                    });
            } else if item.kind == "fn" && !item.body.is_empty() {
                let mut roles = Vec::new();
                let mut i = 0;
                while i < item.body.len() {
                    let token = item.body[i];
                    if printed[token.start..].starts_with(&synthetic) {
                        // The closure constructor can contain synthetic generic
                        // arguments. The original source has the lambda
                        // instead.
                        if let Some(end) = item.body[i..].windows(3).position(|w| {
                            w[0].text == "}" && w[1].text == "." && w[2].text == "call"
                        }) {
                            i += end + 3;
                            continue;
                        }
                    }
                    if ordered(&token) {
                        roles.push(Angle::from_token(&printed, token));
                    }
                    i += 1;
                }
                if let Some(range) = lambda.take() {
                    result.lambdas.insert(range.start, (range.end, roles));
                } else {
                    result
                        .functions
                        .entry((item.name.text.to_owned(), item.associated))
                        .or_default()
                        .push_back(roles);
                }
            }
        });
        if complete.is_err() {
            // A NUL in one member must not obscure the angle roles of an
            // independently printable sibling. Lambdas need their module spans.
            for i in 0..native.get_member_count() {
                let Some(dslx::MatchableModuleMember::Function(function)) =
                    native.get_member(i).to_matchable()
                else {
                    continue;
                };
                let key = (function.get_identifier(), false);
                if result.functions.contains_key(&key) {
                    continue;
                }
                let display = function.to_text();
                let text = native_display_tokens_source(&display);
                if text.contains(&synthetic) {
                    continue;
                }
                let mut body = None;
                let parsed = visit_conversion_items(&text, |item| {
                    if item.kind == "fn" && item.name.text == key.0 && !item.body.is_empty() {
                        body = Some(
                            item.body
                                .iter()
                                .copied()
                                .filter(ordered)
                                .map(|t| Angle::from_token(&text, t))
                                .collect(),
                        );
                    }
                });
                if parsed.is_ok()
                    && let Some(roles) = body
                {
                    result.functions.insert(key, VecDeque::from([roles]));
                }
            }
        }
        result
    }

    /// Pair the original token stream with each native body separately.
    /// A mismatch cannot authorize conversion of a possibly unsafe operator.
    fn insertions(&self, body: &[Token<'_>], roles: &[Angle]) -> Option<Vec<usize>> {
        let mut offsets = Vec::new();
        let mut pending = vec![(body, roles)];
        while let Some((tokens, roles)) = pending.pop() {
            let mut i = 0;
            let mut roles = roles.iter();
            while i < tokens.len() {
                let token = tokens[i];
                if let Some((end, lambda_roles)) = self.lambdas.get(&token.start) {
                    let stop = tokens.partition_point(|t| t.start < *end);
                    if stop <= i || *end > tokens.last()?.start + tokens.last()?.text.len() {
                        return None;
                    }
                    let lambda = tokens.get(i..stop)?;
                    let start = lambda_body_start(lambda)?;
                    pending.push((lambda.get(start..)?, lambda_roles.as_slice()));
                    i = stop;
                } else {
                    if ordered(&token) {
                        let role = roles.next()?;
                        if token.text != role.spelling {
                            return None;
                        }
                        if role.compare {
                            offsets.push(token.start);
                        }
                    }
                    i += 1;
                }
            }
            if roles.next().is_some() {
                return None;
            }
        }
        offsets.sort_unstable();
        Some(offsets)
    }
}

/// A typed lambda ends in a brace-delimited body. Inferred lambdas may instead
/// have a single expression immediately following the parameter list.
fn lambda_body_start(tokens: &[Token<'_>]) -> Option<usize> {
    let close_pipe = match tokens.first()?.text {
        "||" => 0,
        "|" => tokens.get(1..)?.iter().position(|t| t.text == "|")? + 1,
        _ => return None,
    };
    if tokens.get(close_pipe + 1)?.text != "->" {
        return Some(close_pipe + 1);
    }
    let mut depth = 0usize;
    for (i, token) in tokens.iter().enumerate().rev() {
        match token.text {
            "}" => depth += 1,
            "{" => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => { /* Skip the body back to its opening brace. */ }
        }
    }
    None
}
