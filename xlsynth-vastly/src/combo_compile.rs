// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::Result;
use crate::Signedness;
use crate::ast::Expr;
use crate::ast_spanned::SpannedExpr;
use crate::ast_spanned::SpannedExprKind;
use crate::module_compile::DeclInfo;
use crate::parser::parse_expr;
use crate::parser_spanned::parse_expr_spanned;
use crate::sv_ast::ComboFunctionBody;
use crate::sv_ast::ComboItem;
use crate::sv_ast::ComboModule;
use crate::sv_ast::Decl;
use crate::sv_ast::PortDir as SvPortDir;
use crate::sv_ast::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortDir {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub dir: PortDir,
    pub name: String,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct ComboAssign {
    pub lhs: String,
    pub rhs: Expr,
    pub rhs_span: Span,
    pub rhs_spanned: SpannedExpr,
}

#[derive(Debug, Clone)]
pub struct CasezPattern {
    pub width: u32,
    /// MSB-first chars, with '?' and 'z' as don't-care.
    pub bits_msb: String,
}

#[derive(Debug, Clone)]
pub struct CasezArm {
    pub pat: Option<CasezPattern>, // None => default
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct FunctionVar {
    pub name: String,
    pub width: u32,
    pub signedness: Signedness,
}

#[derive(Debug, Clone)]
pub struct FunctionAssign {
    pub lhs: String,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct ComboFunction {
    pub name: String,
    pub ret_width: u32,
    pub ret_signedness: Signedness,
    pub args: Vec<FunctionVar>,
    pub locals: BTreeMap<String, DeclInfo>,
    pub body: ComboFunctionImpl,
}

#[derive(Debug, Clone)]
pub enum ComboFunctionImpl {
    Casez {
        selector: Expr,
        arms: Vec<CasezArm>,
    },
    Expr {
        expr: Expr,
        expr_spanned: Option<SpannedExpr>,
    },
    Procedure {
        assigns: Vec<FunctionAssign>,
    },
}

#[derive(Debug, Clone)]
pub struct CompiledComboModule {
    pub module_name: String,
    pub input_ports: Vec<Port>,
    pub output_ports: Vec<Port>,
    pub decls: BTreeMap<String, DeclInfo>,
    pub assigns: Vec<ComboAssign>,
    pub functions: BTreeMap<String, ComboFunction>,
}

pub fn compile_combo_module(src: &str) -> Result<CompiledComboModule> {
    let normalized = normalize_generated_unpacked_arrays(src);
    let parse_src = normalized.normalized_src.as_str();
    let parsed: ComboModule = crate::sv_parser::parse_combo_module(parse_src)?;

    let module_name = parsed.name;

    let mut input_ports: Vec<Port> = Vec::new();
    let mut output_ports: Vec<Port> = Vec::new();
    let mut decls: BTreeMap<String, DeclInfo> = BTreeMap::new();

    for p in &parsed.ports {
        let dir = match p.dir {
            SvPortDir::Input => PortDir::Input,
            SvPortDir::Output => PortDir::Output,
        };
        let dir_for_vecs = dir.clone();
        let port = Port {
            dir,
            name: p.name.clone(),
            width: p.width,
        };
        match dir_for_vecs {
            PortDir::Input => input_ports.push(port),
            PortDir::Output => output_ports.push(port),
        }
        decls.insert(
            normalized.denormalize_ident(&p.name),
            DeclInfo {
                width: p.width,
                signedness: if p.signed {
                    Signedness::Signed
                } else {
                    Signedness::Unsigned
                },
            },
        );
    }

    let mut functions: BTreeMap<String, ComboFunction> = BTreeMap::new();
    let mut assigns: Vec<ComboAssign> = Vec::new();

    for it in &parsed.items {
        match it {
            ComboItem::WireDecl(d) => {
                decls.insert(
                    normalized.denormalize_ident(&d.name),
                    DeclInfo {
                        width: d.width,
                        signedness: if d.signed {
                            Signedness::Signed
                        } else {
                            Signedness::Unsigned
                        },
                    },
                );
            }
            ComboItem::Assign { lhs_ident, rhs } => {
                let rhs_src = parse_src[rhs.start..rhs.end].trim();
                let rhs_expr =
                    denormalize_expr(parse_expr(rhs_src)?, &normalized.placeholder_to_original);
                let mut rhs_spanned = parse_expr_spanned(rhs_src)?;
                denormalize_spanned_expr(&mut rhs_spanned, &normalized.placeholder_to_original);
                rhs_spanned.shift_spans(rhs.start);
                assigns.push(ComboAssign {
                    lhs: normalized.denormalize_ident(lhs_ident),
                    rhs: rhs_expr,
                    rhs_span: *rhs,
                    rhs_spanned,
                });
            }
            ComboItem::Function(f) => {
                let args: Vec<FunctionVar> =
                    f.args.iter().map(lower_decl_to_function_var).collect();
                let locals: BTreeMap<String, DeclInfo> = f
                    .locals
                    .iter()
                    .map(|d| {
                        (
                            normalized.denormalize_ident(&d.name),
                            decl_info_from_decl(d),
                        )
                    })
                    .collect();

                let body = match &f.body {
                    ComboFunctionBody::UniqueCasez { selector, arms, .. } => {
                        let selector_src = parse_src[selector.start..selector.end].trim();
                        let selector_expr = denormalize_expr(
                            parse_expr(selector_src)?,
                            &normalized.placeholder_to_original,
                        );

                        let mut out_arms: Vec<CasezArm> = Vec::new();
                        for a in arms {
                            let value_src = parse_src[a.value.start..a.value.end].trim();
                            let value_expr = denormalize_expr(
                                parse_expr(value_src)?,
                                &normalized.placeholder_to_original,
                            );
                            let pat = a.pat.as_ref().map(|p| CasezPattern {
                                width: p.width,
                                bits_msb: p.bits_msb.clone(),
                            });
                            out_arms.push(CasezArm {
                                pat,
                                value: value_expr,
                            });
                        }
                        ComboFunctionImpl::Casez {
                            selector: selector_expr,
                            arms: out_arms,
                        }
                    }
                    ComboFunctionBody::Assign { value } => {
                        let value_src = parse_src[value.start..value.end].trim();
                        let expr = denormalize_expr(
                            parse_expr(value_src)?,
                            &normalized.placeholder_to_original,
                        );
                        let mut expr_spanned = parse_expr_spanned(value_src)?;
                        denormalize_spanned_expr(
                            &mut expr_spanned,
                            &normalized.placeholder_to_original,
                        );
                        expr_spanned.shift_spans(value.start);
                        ComboFunctionImpl::Expr {
                            expr,
                            expr_spanned: Some(expr_spanned),
                        }
                    }
                    ComboFunctionBody::Procedure { assigns } => {
                        let mut out_assigns = Vec::with_capacity(assigns.len());
                        for a in assigns {
                            let value_src = parse_src[a.value.start..a.value.end].trim();
                            let expr = denormalize_expr(
                                parse_expr(value_src)?,
                                &normalized.placeholder_to_original,
                            );
                            out_assigns.push(FunctionAssign {
                                lhs: normalized.denormalize_ident(&a.lhs),
                                expr,
                            });
                        }
                        ComboFunctionImpl::Procedure {
                            assigns: out_assigns,
                        }
                    }
                };

                functions.insert(
                    f.name.clone(),
                    ComboFunction {
                        name: f.name.clone(),
                        ret_width: f.ret_width,
                        ret_signedness: if f.ret_signed {
                            Signedness::Signed
                        } else {
                            Signedness::Unsigned
                        },
                        args: args
                            .into_iter()
                            .map(|mut a| {
                                a.name = normalized.denormalize_ident(&a.name);
                                a
                            })
                            .collect(),
                        locals,
                        body,
                    },
                );
            }
        }
    }

    Ok(CompiledComboModule {
        module_name,
        input_ports,
        output_ports,
        decls,
        assigns,
        functions,
    })
}

fn decl_info_from_decl(d: &Decl) -> DeclInfo {
    DeclInfo {
        width: d.width,
        signedness: if d.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
    }
}

fn lower_decl_to_function_var(d: &Decl) -> FunctionVar {
    FunctionVar {
        name: d.name.clone(),
        width: d.width,
        signedness: if d.signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
    }
}

#[derive(Debug, Default)]
struct ArrayNormalization {
    normalized_src: String,
    placeholder_to_original: BTreeMap<String, String>,
}

impl ArrayNormalization {
    fn denormalize_ident(&self, name: &str) -> String {
        self.placeholder_to_original
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
}

fn normalize_generated_unpacked_arrays(src: &str) -> ArrayNormalization {
    let mut decls_by_base = BTreeMap::new();
    let mut replacement_pairs: Vec<(String, String)> = Vec::new();
    let mut out_lines: Vec<String> = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        if let Some(decl) = parse_generated_unpacked_array_decl_line(line) {
            for indices in decl.full_index_tuples() {
                let placeholder = decl.placeholder(&indices);
                replacement_pairs.push((decl.original_ref(&indices), placeholder.clone()));
                out_lines.push(render_flat_decl_line(&decl, &placeholder));
            }
            decls_by_base.insert(decl.base_name.clone(), decl);
            i += 1;
            continue;
        }

        if let Some(loop_info) = parse_generated_genvar_for_header(line) {
            let mut body_lines: Vec<&str> = Vec::new();
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim() != "end" {
                body_lines.push(lines[j]);
                j += 1;
            }
            if j == lines.len() {
                out_lines.push(line.to_string());
                out_lines.extend(body_lines.into_iter().map(str::to_string));
                break;
            }
            for iter in loop_info.start..loop_info.limit {
                let iter_text = iter.to_string();
                for body_line in &body_lines {
                    let substituted = replace_ident_token(body_line, &loop_info.var, &iter_text);
                    out_lines.extend(expand_generated_unpacked_array_assign_line(
                        &substituted,
                        &decls_by_base,
                    ));
                }
            }
            i = j + 1;
        } else {
            out_lines.extend(expand_generated_unpacked_array_assign_line(
                line,
                &decls_by_base,
            ));
            i += 1;
        }
    }

    let mut normalized_src = out_lines.join("\n");
    if src.ends_with('\n') {
        normalized_src.push('\n');
    }
    replacement_pairs.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut placeholder_to_original = BTreeMap::new();
    for (original, placeholder) in replacement_pairs {
        normalized_src = normalized_src.replace(&original, &placeholder);
        placeholder_to_original.insert(placeholder, original);
    }

    ArrayNormalization {
        normalized_src,
        placeholder_to_original,
    }
}

#[derive(Debug, Clone)]
struct GeneratedUnpackedArrayDecl {
    indent: String,
    kind: String,
    signed: bool,
    packed_range: Option<String>,
    base_name: String,
    dimensions: Vec<Vec<u32>>,
}

impl GeneratedUnpackedArrayDecl {
    fn rank(&self) -> usize {
        self.dimensions.len()
    }

    fn full_index_tuples(&self) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        collect_index_tuples(&self.dimensions, &mut prefix, &mut out);
        out
    }

    fn original_ref(&self, indices: &[u32]) -> String {
        let mut s = self.base_name.clone();
        for idx in indices {
            s.push('[');
            s.push_str(&idx.to_string());
            s.push(']');
        }
        s
    }

    fn placeholder(&self, indices: &[u32]) -> String {
        make_array_placeholder(&self.base_name, indices)
    }
}

fn parse_generated_unpacked_array_decl_line(line: &str) -> Option<GeneratedUnpackedArrayDecl> {
    let indent_len = line.len() - line.trim_start_matches(char::is_whitespace).len();
    let indent = line[..indent_len].to_string();
    let mut rest = line[indent_len..].trim_end();
    rest = rest.strip_suffix(';')?.trim_end();

    let (kind, after_kind) = if let Some(r) = rest.strip_prefix("wire ") {
        ("wire".to_string(), r)
    } else if let Some(r) = rest.strip_prefix("logic ") {
        ("logic".to_string(), r)
    } else {
        return None;
    };

    let (signed, after_signed) = if let Some(r) = after_kind.strip_prefix("signed ") {
        (true, r)
    } else {
        (false, after_kind)
    };

    let (packed_range, after_packed) = if after_signed.starts_with('[') {
        let (range, rest) = take_bracket_group(after_signed)?;
        (Some(range.to_string()), rest.trim_start())
    } else {
        (None, after_signed.trim_start())
    };

    let name_end = after_packed
        .find(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .unwrap_or(after_packed.len());
    if name_end == 0 {
        return None;
    }
    let base_name = after_packed[..name_end].to_string();
    let mut after_name = after_packed[name_end..].trim_start();
    if !after_name.starts_with('[') {
        return None;
    }
    let mut dimensions = Vec::new();
    while after_name.starts_with('[') {
        let (group, tail) = take_bracket_group(after_name)?;
        dimensions.push(parse_unpacked_indices(group)?);
        after_name = tail.trim_start();
    }
    if !after_name.is_empty() {
        return None;
    }

    Some(GeneratedUnpackedArrayDecl {
        indent,
        kind,
        signed,
        packed_range,
        base_name,
        dimensions,
    })
}

fn take_bracket_group(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with('[') {
        return None;
    }
    let mut depth = 0u32;
    for (idx, ch) in s.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some((&s[..=idx], &s[idx + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_unpacked_indices(group: &str) -> Option<Vec<u32>> {
    let inner = group.strip_prefix('[')?.strip_suffix(']')?.trim();
    if let Some((lhs, rhs)) = inner.split_once(':') {
        let start = lhs.trim().parse::<u32>().ok()?;
        let end = rhs.trim().parse::<u32>().ok()?;
        if start <= end {
            Some((start..=end).collect())
        } else {
            Some((end..=start).rev().collect())
        }
    } else {
        let count = inner.parse::<u32>().ok()?;
        Some((0..count).collect())
    }
}

fn render_flat_decl_line(decl: &GeneratedUnpackedArrayDecl, placeholder: &str) -> String {
    let mut s = String::new();
    s.push_str(&decl.indent);
    s.push_str(&decl.kind);
    s.push(' ');
    if decl.signed {
        s.push_str("signed ");
    }
    if let Some(range) = &decl.packed_range {
        s.push_str(range);
        s.push(' ');
    }
    s.push_str(placeholder);
    s.push(';');
    s
}

fn make_array_placeholder(base: &str, indices: &[u32]) -> String {
    let mut s = String::from(base);
    for idx in indices {
        s.push_str("__vastly_idx_");
        s.push_str(&idx.to_string());
    }
    s
}

fn collect_index_tuples(dims: &[Vec<u32>], prefix: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
    if dims.is_empty() {
        out.push(prefix.clone());
        return;
    }
    for &idx in &dims[0] {
        prefix.push(idx);
        collect_index_tuples(&dims[1..], prefix, out);
        prefix.pop();
    }
}

fn expand_generated_unpacked_array_assign_line(
    line: &str,
    decls_by_base: &BTreeMap<String, GeneratedUnpackedArrayDecl>,
) -> Vec<String> {
    let Some((indent, lhs, rhs)) = parse_generated_assign_line(line) else {
        return vec![line.to_string()];
    };
    let (_lhs_base, lhs_prefix, lhs_decl) =
        match parse_array_ref(&lhs).and_then(|(lhs_base, lhs_prefix)| {
            decls_by_base
                .get(&lhs_base)
                .map(|d| (lhs_base, lhs_prefix, d))
        }) {
            Some(found) => found,
            None => {
                let rhs_ref = rewrite_array_refs_for_suffix(&rhs, decls_by_base, &[]);
                if rhs_ref == rhs {
                    return vec![line.to_string()];
                }
                return vec![format!("{indent}assign {lhs} = {rhs_ref};")];
            }
        };
    if lhs_prefix.len() > lhs_decl.rank() {
        return vec![line.to_string()];
    }

    let suffix_dims = &lhs_decl.dimensions[lhs_prefix.len()..];
    if suffix_dims.is_empty() {
        let rhs_ref = rewrite_array_refs_for_suffix(&rhs, decls_by_base, &[]);
        if rhs_ref == rhs {
            return vec![line.to_string()];
        }
        return vec![format!("{indent}assign {lhs} = {rhs_ref};")];
    }

    let mut suffixes = Vec::new();
    collect_index_tuples(suffix_dims, &mut Vec::new(), &mut suffixes);
    let mut out = Vec::with_capacity(suffixes.len());
    for suffix in suffixes {
        let mut full_lhs = lhs_prefix.clone();
        full_lhs.extend_from_slice(&suffix);
        let lhs_ref = lhs_decl.original_ref(&full_lhs);
        let rhs_ref = rewrite_array_refs_for_suffix(&rhs, decls_by_base, &suffix);
        out.push(format!("{indent}assign {lhs_ref} = {rhs_ref};"));
    }
    out
}

fn parse_generated_assign_line(line: &str) -> Option<(String, String, String)> {
    let indent_len = line.len() - line.trim_start_matches(char::is_whitespace).len();
    let indent = line[..indent_len].to_string();
    let mut rest = line[indent_len..].trim_end();
    rest = rest.strip_suffix(';')?.trim_end();
    let rest = rest.strip_prefix("assign ")?;
    let eq = rest.find(" = ")?;
    let lhs = rest[..eq].trim().to_string();
    let rhs = rest[eq + 3..].trim().to_string();
    Some((indent, lhs, rhs))
}

fn parse_array_ref(s: &str) -> Option<(String, Vec<u32>)> {
    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }
    let mut name_end = first.len_utf8();
    for (idx, ch) in chars {
        if ch == '_' || ch.is_ascii_alphanumeric() {
            name_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    let base = s[..name_end].to_string();
    let mut rest = &s[name_end..];
    let mut indices = Vec::new();
    while rest.starts_with('[') {
        let (group, tail) = take_bracket_group(rest)?;
        let inner = group.strip_prefix('[')?.strip_suffix(']')?.trim();
        indices.push(inner.parse::<u32>().ok()?);
        rest = tail;
    }
    if !rest.is_empty() {
        return None;
    }
    Some((base, indices))
}

fn rewrite_array_refs_for_suffix(
    expr: &str,
    decls_by_base: &BTreeMap<String, GeneratedUnpackedArrayDecl>,
    suffix: &[u32],
) -> String {
    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if is_ident_start(c) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let base = &expr[start..i];
            if let Some(decl) = decls_by_base.get(base) {
                let mut full = String::from(base);
                let mut groups: Vec<String> = Vec::new();
                let mut j = i;
                while let Some((group, tail)) = take_bracket_group(&expr[j..]) {
                    groups.push(group.to_string());
                    full.push_str(group);
                    j = expr.len() - tail.len();
                }
                if let Some(lowered) = lower_array_ref(decl, base, &groups, suffix) {
                    out.push_str(&lowered);
                    i = j;
                    continue;
                }
                out.push_str(&full);
                i = j;
                continue;
            }
            out.push_str(base);
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

#[derive(Debug)]
struct GeneratedGenvarLoop {
    var: String,
    start: u32,
    limit: u32,
}

fn parse_generated_genvar_for_header(line: &str) -> Option<GeneratedGenvarLoop> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("for (genvar ")?;
    let var_end = rest
        .find(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .unwrap_or(rest.len());
    if var_end == 0 {
        return None;
    }
    let var = rest[..var_end].to_string();
    let rest = &rest[var_end..];
    let rest = rest.strip_prefix(" = ")?;
    let (start_text, rest) = rest.split_once(';')?;
    let start = start_text.trim().parse::<u32>().ok()?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(&var)?;
    let rest = rest.strip_prefix(" < ")?;
    let (limit_text, rest) = rest.split_once(';')?;
    let limit = limit_text.trim().parse::<u32>().ok()?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(&var)?;
    let rest = rest.strip_prefix(" = ")?;
    let rest = rest.strip_prefix(&var)?;
    let rest = rest.strip_prefix(" + 1) begin")?;
    if !rest.trim_start().starts_with(':') {
        return None;
    }
    Some(GeneratedGenvarLoop { var, start, limit })
}

fn replace_ident_token(s: &str, target: &str, replacement: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = &s[start..i];
            if ident == target {
                out.push_str(replacement);
            } else {
                out.push_str(ident);
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn lower_array_ref(
    decl: &GeneratedUnpackedArrayDecl,
    _base: &str,
    groups: &[String],
    suffix: &[u32],
) -> Option<String> {
    let mut numeric_prefix = Vec::new();
    let mut dynamic_expr: Option<&str> = None;
    let mut trailing_numeric = Vec::new();

    for group in groups {
        let inner = group.strip_prefix('[')?.strip_suffix(']')?.trim();
        if dynamic_expr.is_none() {
            if let Ok(idx) = inner.parse::<u32>() {
                numeric_prefix.push(idx);
            } else {
                dynamic_expr = Some(inner);
            }
        } else if let Ok(idx) = inner.parse::<u32>() {
            trailing_numeric.push(idx);
        } else {
            return None;
        }
    }

    if dynamic_expr.is_none() {
        if numeric_prefix.len() > decl.rank() {
            return None;
        }
        let remaining_rank = decl.rank() - numeric_prefix.len();
        if remaining_rank != suffix.len() {
            return None;
        }
        let mut indices = numeric_prefix;
        indices.extend_from_slice(suffix);
        return Some(decl.original_ref(&indices));
    }

    let dyn_expr = dynamic_expr?;
    let dyn_dim = decl.dimensions.get(numeric_prefix.len())?;
    let resolved_len = numeric_prefix.len() + 1 + trailing_numeric.len() + suffix.len();
    if resolved_len != decl.rank() {
        return None;
    }

    let mut fallback = String::from("'x");
    for &candidate in dyn_dim.iter().rev() {
        let mut indices = numeric_prefix.clone();
        indices.push(candidate);
        indices.extend_from_slice(&trailing_numeric);
        indices.extend_from_slice(suffix);
        let candidate_ref = decl.original_ref(&indices);
        fallback = format!(
            "(({}) === {} ? {} : {})",
            dyn_expr, candidate, candidate_ref, fallback
        );
    }
    Some(fallback)
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

fn denormalize_expr(expr: Expr, placeholders: &BTreeMap<String, String>) -> Expr {
    match expr {
        Expr::Ident(name) => Expr::Ident(placeholders.get(&name).cloned().unwrap_or(name)),
        Expr::Literal(v) => Expr::Literal(v),
        Expr::UnbasedUnsized(b) => Expr::UnbasedUnsized(b),
        Expr::Call { name, args } => Expr::Call {
            name,
            args: args
                .into_iter()
                .map(|a| denormalize_expr(a, placeholders))
                .collect(),
        },
        Expr::Concat(parts) => Expr::Concat(
            parts
                .into_iter()
                .map(|p| denormalize_expr(p, placeholders))
                .collect(),
        ),
        Expr::Replicate { count, expr } => Expr::Replicate {
            count: Box::new(denormalize_expr(*count, placeholders)),
            expr: Box::new(denormalize_expr(*expr, placeholders)),
        },
        Expr::Index { expr, index } => Expr::Index {
            expr: Box::new(denormalize_expr(*expr, placeholders)),
            index: Box::new(denormalize_expr(*index, placeholders)),
        },
        Expr::Slice { expr, msb, lsb } => Expr::Slice {
            expr: Box::new(denormalize_expr(*expr, placeholders)),
            msb: Box::new(denormalize_expr(*msb, placeholders)),
            lsb: Box::new(denormalize_expr(*lsb, placeholders)),
        },
        Expr::IndexedSlice {
            expr,
            base,
            width,
            upward,
        } => Expr::IndexedSlice {
            expr: Box::new(denormalize_expr(*expr, placeholders)),
            base: Box::new(denormalize_expr(*base, placeholders)),
            width: Box::new(denormalize_expr(*width, placeholders)),
            upward,
        },
        Expr::Unary { op, expr } => Expr::Unary {
            op,
            expr: Box::new(denormalize_expr(*expr, placeholders)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op,
            lhs: Box::new(denormalize_expr(*lhs, placeholders)),
            rhs: Box::new(denormalize_expr(*rhs, placeholders)),
        },
        Expr::Ternary { cond, t, f } => Expr::Ternary {
            cond: Box::new(denormalize_expr(*cond, placeholders)),
            t: Box::new(denormalize_expr(*t, placeholders)),
            f: Box::new(denormalize_expr(*f, placeholders)),
        },
    }
}

fn denormalize_spanned_expr(expr: &mut SpannedExpr, placeholders: &BTreeMap<String, String>) {
    match &mut expr.kind {
        SpannedExprKind::Ident(name) => {
            if let Some(original) = placeholders.get(name) {
                *name = original.clone();
            }
        }
        SpannedExprKind::Literal(_) | SpannedExprKind::UnbasedUnsized(_) => {}
        SpannedExprKind::Call { args, .. } => {
            for a in args {
                denormalize_spanned_expr(a, placeholders);
            }
        }
        SpannedExprKind::Concat(parts) => {
            for p in parts {
                denormalize_spanned_expr(p, placeholders);
            }
        }
        SpannedExprKind::Replicate { count, expr } => {
            denormalize_spanned_expr(count, placeholders);
            denormalize_spanned_expr(expr, placeholders);
        }
        SpannedExprKind::Index { expr, index } => {
            denormalize_spanned_expr(expr, placeholders);
            denormalize_spanned_expr(index, placeholders);
        }
        SpannedExprKind::Slice { expr, msb, lsb } => {
            denormalize_spanned_expr(expr, placeholders);
            denormalize_spanned_expr(msb, placeholders);
            denormalize_spanned_expr(lsb, placeholders);
        }
        SpannedExprKind::IndexedSlice {
            expr, base, width, ..
        } => {
            denormalize_spanned_expr(expr, placeholders);
            denormalize_spanned_expr(base, placeholders);
            denormalize_spanned_expr(width, placeholders);
        }
        SpannedExprKind::Unary { expr, .. } => {
            denormalize_spanned_expr(expr, placeholders);
        }
        SpannedExprKind::Binary { lhs, rhs, .. } => {
            denormalize_spanned_expr(lhs, placeholders);
            denormalize_spanned_expr(rhs, placeholders);
        }
        SpannedExprKind::Ternary { cond, t, f } => {
            denormalize_spanned_expr(cond, placeholders);
            denormalize_spanned_expr(t, placeholders);
            denormalize_spanned_expr(f, placeholders);
        }
    }
}
