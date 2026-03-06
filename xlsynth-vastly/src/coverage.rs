// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::LogicBit;
use crate::Value4;
use crate::ast_spanned::SpannedExpr;
use crate::ast_spanned::SpannedExprKind;
use crate::pipeline_compile::FunctionMeta;
use crate::sv_ast::Span;

#[derive(Debug, Clone)]
pub struct SourceText {
    pub text: String,
    line_starts: Vec<usize>, // 0-based byte offsets
    coverable: Vec<bool>,    // 1-based indexed via line number - 1
}

impl SourceText {
    pub fn new(text: String) -> Self {
        let mut line_starts: Vec<usize> = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        let line_count = line_starts.len();
        let mut coverable: Vec<bool> = vec![false; line_count];
        for (idx0, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.is_empty() {
                continue;
            }
            if t.starts_with("//") {
                continue;
            }
            coverable[idx0] = true;
        }
        Self {
            text,
            line_starts,
            coverable,
        }
    }

    pub fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }

    pub fn is_coverable_line(&self, line_1b: u32) -> bool {
        let idx0 = (line_1b as usize).saturating_sub(1);
        self.coverable.get(idx0).cloned().unwrap_or(false)
    }

    pub fn line_of_offset(&self, offset: usize) -> u32 {
        // Returns 1-based line number.
        match self.line_starts.binary_search(&offset) {
            Ok(i) => (i + 1) as u32,
            Err(i) => i as u32, /* insertion point is next line start; previous line is i-1 => 1b
                                 * is i */
        }
    }

    pub fn lines_for_span(&self, span: Span) -> Vec<u32> {
        let start_line = self.line_of_offset(span.start);
        let end_line = self.line_of_offset(span.end.saturating_sub(1));
        let mut out: Vec<u32> = Vec::new();
        for l in start_line..=end_line {
            out.push(l);
        }
        out
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpanKey {
    pub start: usize,
    pub end: usize,
}

impl From<Span> for SpanKey {
    fn from(s: Span) -> Self {
        Self {
            start: s.start,
            end: s.end,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct TernaryBranchCounts {
    pub t_taken: u64,
    pub f_taken: u64,
    pub cond_unknown: u64,
    pub t_span: SpanKey,
    pub f_span: SpanKey,
}

#[derive(Debug, Default, Clone)]
pub struct CoverageCounters {
    pub line_hits: BTreeMap<u32, u64>,
    pub ternary_branches: BTreeMap<SpanKey, TernaryBranchCounts>,
    pub toggle_counts: BTreeMap<String, Vec<u64>>,
    pub function_calls: BTreeMap<String, u64>,
    pub executed_spans: BTreeMap<SpanKey, u64>,
    pub selected_arm_spans: BTreeMap<SpanKey, u64>,
    pub functions: BTreeMap<String, FunctionMeta>,
    pub defines: BTreeSet<String>,
}

impl CoverageCounters {
    pub fn register_ternaries_from_spanned_expr(&mut self, expr: &SpannedExpr) {
        collect_ternaries_spanned(expr, &mut self.ternary_branches);
    }

    pub fn register_functions(&mut self, fn_meta: &BTreeMap<String, FunctionMeta>) {
        for (name, meta) in fn_meta {
            self.functions.insert(name.clone(), meta.clone());
            // Prepopulate maps so misses can be rendered.
            self.function_calls.entry(name.clone()).or_insert(0);
            self.executed_spans.entry(meta.def_span.into()).or_insert(0);
            self.executed_spans
                .entry(meta.body_span.into())
                .or_insert(0);
            for s in &meta.scaffold_spans {
                self.executed_spans.entry((*s).into()).or_insert(0);
            }
            if let Some(s) = meta.assign_expr_span {
                self.executed_spans.entry(s.into()).or_insert(0);
            }
            for a in &meta.arms {
                self.selected_arm_spans
                    .entry(a.arm_span.into())
                    .or_insert(0);
                self.executed_spans.entry(a.value_span.into()).or_insert(0);
            }
        }
    }

    pub fn hit_span(&mut self, src: &SourceText, span: Span) {
        for l in src.lines_for_span(span) {
            *self.line_hits.entry(l).or_insert(0) += 1;
        }
    }

    pub fn record_ternary_decision(&mut self, ternary_span: SpanKey, cond: LogicBit) {
        let e = self.ternary_branches.entry(ternary_span).or_default();
        match cond {
            LogicBit::One => e.t_taken += 1,
            LogicBit::Zero => e.f_taken += 1,
            LogicBit::X | LogicBit::Z => e.cond_unknown += 1,
        }
    }

    pub fn observe_toggles(&mut self, prev: Option<&Value4>, cur: &Value4, name: &str) {
        let counts = self
            .toggle_counts
            .entry(name.to_string())
            .or_insert_with(|| vec![0u64; cur.width as usize]);
        if counts.len() != cur.width as usize {
            *counts = vec![0u64; cur.width as usize];
        }
        let Some(prev) = prev else {
            return;
        };
        if prev.width != cur.width {
            return;
        }
        for i in 0..(cur.width as usize) {
            let a = prev.bits_lsb_first()[i];
            let b = cur.bits_lsb_first()[i];
            if a.is_known_01() && b.is_known_01() && a != b {
                counts[i] += 1;
            }
        }
    }

    pub fn bump_span(&mut self, span: Span) {
        *self.executed_spans.entry(span.into()).or_insert(0) += 1;
    }

    pub fn bump_selected_arm(&mut self, span: Span) {
        *self.selected_arm_spans.entry(span.into()).or_insert(0) += 1;
    }
}

fn collect_ternaries_spanned(expr: &SpannedExpr, out: &mut BTreeMap<SpanKey, TernaryBranchCounts>) {
    match &expr.kind {
        SpannedExprKind::Ident(_) => {}
        SpannedExprKind::Literal(_) => {}
        SpannedExprKind::UnbasedUnsized(_) => {}
        SpannedExprKind::Call { args, .. } => {
            for a in args {
                collect_ternaries_spanned(a, out);
            }
        }
        SpannedExprKind::Concat(ps) => {
            for p in ps {
                collect_ternaries_spanned(p, out);
            }
        }
        SpannedExprKind::Replicate { count, expr } => {
            collect_ternaries_spanned(count, out);
            collect_ternaries_spanned(expr, out);
        }
        SpannedExprKind::Index { expr, index } => {
            collect_ternaries_spanned(expr, out);
            collect_ternaries_spanned(index, out);
        }
        SpannedExprKind::Slice { expr, msb, lsb } => {
            collect_ternaries_spanned(expr, out);
            collect_ternaries_spanned(msb, out);
            collect_ternaries_spanned(lsb, out);
        }
        SpannedExprKind::IndexedSlice {
            expr, base, width, ..
        } => {
            collect_ternaries_spanned(expr, out);
            collect_ternaries_spanned(base, out);
            collect_ternaries_spanned(width, out);
        }
        SpannedExprKind::Unary { expr, .. } => {
            collect_ternaries_spanned(expr, out);
        }
        SpannedExprKind::Binary { lhs, rhs, .. } => {
            collect_ternaries_spanned(lhs, out);
            collect_ternaries_spanned(rhs, out);
        }
        SpannedExprKind::Ternary { cond, t, f } => {
            let id: SpanKey = expr.span.into();
            out.entry(id).or_insert_with(|| TernaryBranchCounts {
                t_span: t.span.into(),
                f_span: f.span.into(),
                ..TernaryBranchCounts::default()
            });
            collect_ternaries_spanned(cond, out);
            collect_ternaries_spanned(t, out);
            collect_ternaries_spanned(f, out);
        }
    }
}
