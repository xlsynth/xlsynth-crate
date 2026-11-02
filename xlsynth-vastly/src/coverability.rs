// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use crate::sv_ast::ModuleItem;
use crate::sv_ast::Span;
use crate::{CoverageCounters, Error, SourceText};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LineCoverability {
    NonCoverableStructural,
    CoverableExecutable,
    CoverableUnknown,
    BlankOrComment,
}

#[derive(Debug, Clone)]
pub struct CoverabilityMap {
    // 1-based line number stored at index line-1.
    per_line: Vec<LineCoverability>,
}

impl CoverabilityMap {
    pub fn line(&self, line_1b: u32) -> LineCoverability {
        let idx0 = (line_1b as usize).saturating_sub(1);
        self.per_line
            .get(idx0)
            .copied()
            .unwrap_or(LineCoverability::BlankOrComment)
    }

    pub fn is_coverable(&self, line_1b: u32) -> bool {
        matches!(
            self.line(line_1b),
            LineCoverability::CoverableExecutable | LineCoverability::CoverableUnknown
        )
    }
}

pub fn compute_coverability(src: &SourceText) -> crate::Result<CoverabilityMap> {
    compute_coverability_with_defines(src, &std::collections::BTreeSet::new())
}

pub fn compute_coverability_with_defines(
    src: &SourceText,
    defines: &std::collections::BTreeSet<String>,
) -> crate::Result<CoverabilityMap> {
    compute_coverability_with_options(src, defines, None)
}

impl CoverageCounters {
    /// Classifies source lines for the pipeline whose coverage was collected.
    pub fn coverability(&self, src: &SourceText) -> CoverabilityMap {
        compute_coverability_with_options(src, &self.defines, self.top.as_deref())
            .unwrap_or_else(|_| baseline_coverability(src))
    }
}

fn compute_coverability_with_options(
    src: &SourceText,
    defines: &BTreeSet<String>,
    top: Option<&str>,
) -> crate::Result<CoverabilityMap> {
    // Baseline: nonblank/non-comment => unknown coverable.
    let mut per_line: Vec<LineCoverability> = Vec::with_capacity(src.line_count() as usize);
    for line in src.text.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with("//") {
            per_line.push(LineCoverability::BlankOrComment);
        } else if t.starts_with('`') {
            // Preprocessor directives are structural, not executable.
            per_line.push(LineCoverability::NonCoverableStructural);
        } else {
            per_line.push(LineCoverability::CoverableUnknown);
        }
    }

    let modules = crate::sv_parser::parse_pipeline_modules(&src.text, defines)?;
    let mut selected = BTreeSet::new();
    if let Some(name) = top {
        let module = modules
            .get(name)
            .ok_or_else(|| Error::Parse(format!("unknown top module `{name}`")))?;
        selected.insert(module.name.as_str());
        // The pipeline compiler supports direct instances of combinational leaf
        // modules.
        for item in &module.items {
            if let ModuleItem::Instance { module, .. } = item {
                selected.insert(module.as_str());
            }
        }
    }
    for (name, module) in &modules {
        if top.is_none() || selected.contains(name.as_str()) {
            mark_module(src, &mut per_line, module)?;
        } else {
            let span = Span {
                start: module.header_span.start,
                end: module.endmodule_span.end,
            };
            mark_span(
                src,
                &mut per_line,
                span,
                LineCoverability::NonCoverableStructural,
            );
        }
    }
    Ok(CoverabilityMap { per_line })
}

/// Classifies each module using spans in the original multi-module source.
fn mark_module(
    src: &SourceText,
    per_line: &mut [LineCoverability],
    pm: &crate::sv_ast::ParsedModule,
) -> crate::Result<()> {
    let items =
        crate::generate_constructs::elaborate_pipeline_items(&src.text, &pm.params, &pm.items)?;

    // Structural: module header.
    mark_span(
        src,
        per_line,
        pm.header_span,
        LineCoverability::NonCoverableStructural,
    );
    mark_span(
        src,
        per_line,
        pm.endmodule_span,
        LineCoverability::NonCoverableStructural,
    );

    // Structural and executable spans from items.
    for it in items {
        match it {
            ModuleItem::Instance {
                span, connections, ..
            } => {
                let expression_lines: BTreeSet<_> = connections
                    .values()
                    .flat_map(|connection| src.lines_for_span(connection.spanned.span))
                    .collect();
                for line in src.lines_for_span(span) {
                    let class = if expression_lines.contains(&line) {
                        LineCoverability::CoverableExecutable
                    } else {
                        LineCoverability::NonCoverableStructural
                    };
                    mark_lines(per_line, line, line, class);
                }
            }
            ModuleItem::Decl { span, .. } => {
                mark_span(
                    src,
                    per_line,
                    span,
                    LineCoverability::NonCoverableStructural,
                );
            }
            ModuleItem::Assign { span, .. } => {
                mark_span(src, per_line, span, LineCoverability::CoverableExecutable);
            }
            ModuleItem::AlwaysFf { span, .. } => {
                mark_span(src, per_line, span, LineCoverability::CoverableExecutable);
            }
            ModuleItem::Function {
                span, body_span, ..
            } => {
                // The signature/definition is structural; the body is
                // executable when invoked.
                //
                // We classify by line ranges (vs raw byte spans) so indentation
                // on the `begin`/`end` lines does not get
                // incorrectly swept into the structural
                // signature span.
                let def_start_line = src.line_of_offset(span.start);
                let def_end_line = src.line_of_offset(span.end.saturating_sub(1));
                let body_start_line = src.line_of_offset(body_span.start);
                let body_end_line = src.line_of_offset(body_span.end.saturating_sub(1));

                mark_lines(
                    per_line,
                    def_start_line,
                    body_start_line.saturating_sub(1),
                    LineCoverability::NonCoverableStructural,
                );
                mark_lines(
                    per_line,
                    body_start_line,
                    body_end_line,
                    LineCoverability::CoverableExecutable,
                );
                mark_lines(
                    per_line,
                    body_end_line + 1,
                    def_end_line,
                    LineCoverability::NonCoverableStructural,
                );
            }
            ModuleItem::GenerateFor { .. } | ModuleItem::GenerateIf { .. } => {
                unreachable!("pipeline items should be elaborated")
            }
        }
    }

    Ok(())
}

pub fn compute_coverability_or_fallback(src: &SourceText) -> CoverabilityMap {
    compute_coverability(src).unwrap_or_else(|_| baseline_coverability(src))
}

pub fn compute_coverability_or_fallback_with_defines(
    src: &SourceText,
    defines: &std::collections::BTreeSet<String>,
) -> CoverabilityMap {
    compute_coverability_with_defines(src, defines).unwrap_or_else(|_| baseline_coverability(src))
}

fn baseline_coverability(src: &SourceText) -> CoverabilityMap {
    let mut per_line: Vec<LineCoverability> = Vec::with_capacity(src.line_count() as usize);
    for line in src.text.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with("//") {
            per_line.push(LineCoverability::BlankOrComment);
        } else if t.starts_with('`') {
            per_line.push(LineCoverability::NonCoverableStructural);
        } else {
            per_line.push(LineCoverability::CoverableUnknown);
        }
    }
    CoverabilityMap { per_line }
}

fn mark_span(
    src: &SourceText,
    per_line: &mut [LineCoverability],
    span: Span,
    class: LineCoverability,
) {
    for l in src.lines_for_span(span) {
        let idx0 = (l as usize).saturating_sub(1);
        if idx0 >= per_line.len() {
            continue;
        }
        // Preserve blank/comment classification.
        if per_line[idx0] == LineCoverability::BlankOrComment {
            continue;
        }
        // Structural wins over everything; executable wins over unknown.
        per_line[idx0] = match (per_line[idx0], class) {
            (LineCoverability::NonCoverableStructural, _) => {
                LineCoverability::NonCoverableStructural
            }
            (_, LineCoverability::NonCoverableStructural) => {
                LineCoverability::NonCoverableStructural
            }
            (_, LineCoverability::CoverableExecutable) => LineCoverability::CoverableExecutable,
            (cur, LineCoverability::CoverableUnknown) => cur,
            (cur, LineCoverability::BlankOrComment) => cur,
        };
    }
}

fn mark_lines(
    per_line: &mut [LineCoverability],
    start_1b: u32,
    end_1b: u32,
    class: LineCoverability,
) {
    if start_1b == 0 || end_1b == 0 || start_1b > end_1b {
        return;
    }
    let len = per_line.len();
    for l in start_1b..=end_1b {
        let idx0 = (l as usize).saturating_sub(1);
        if idx0 >= len {
            continue;
        }
        if per_line[idx0] == LineCoverability::BlankOrComment {
            continue;
        }
        per_line[idx0] = match (per_line[idx0], class) {
            (LineCoverability::NonCoverableStructural, _) => {
                LineCoverability::NonCoverableStructural
            }
            (_, LineCoverability::NonCoverableStructural) => {
                LineCoverability::NonCoverableStructural
            }
            (_, LineCoverability::CoverableExecutable) => LineCoverability::CoverableExecutable,
            (cur, LineCoverability::CoverableUnknown) => cur,
            (cur, LineCoverability::BlankOrComment) => cur,
        };
    }
}
