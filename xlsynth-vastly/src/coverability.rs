// SPDX-License-Identifier: Apache-2.0

use crate::SourceText;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::Span;

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

    let pm = crate::sv_parser::parse_pipeline_module_with_defines(&src.text, defines)?;
    let items =
        crate::generate_constructs::elaborate_pipeline_items(&src.text, &pm.params, &pm.items)?;

    // Structural: module header.
    mark_span(
        src,
        &mut per_line,
        pm.header_span,
        LineCoverability::NonCoverableStructural,
    );
    mark_span(
        src,
        &mut per_line,
        pm.endmodule_span,
        LineCoverability::NonCoverableStructural,
    );

    // Structural and executable spans from items.
    for it in items {
        match it {
            ModuleItem::Decl { span, .. } => {
                mark_span(
                    src,
                    &mut per_line,
                    span,
                    LineCoverability::NonCoverableStructural,
                );
            }
            ModuleItem::Assign { span, .. } => {
                mark_span(
                    src,
                    &mut per_line,
                    span,
                    LineCoverability::CoverableExecutable,
                );
            }
            ModuleItem::AlwaysFf { span, .. } => {
                mark_span(
                    src,
                    &mut per_line,
                    span,
                    LineCoverability::CoverableExecutable,
                );
            }
            ModuleItem::Function {
                span, body_span, ..
            } => {
                // The signature/definition is structural; the body is executable when invoked.
                //
                // We classify by line ranges (vs raw byte spans) so indentation on the
                // `begin`/`end` lines does not get incorrectly swept into the
                // structural signature span.
                let def_start_line = src.line_of_offset(span.start);
                let def_end_line = src.line_of_offset(span.end.saturating_sub(1));
                let body_start_line = src.line_of_offset(body_span.start);
                let body_end_line = src.line_of_offset(body_span.end.saturating_sub(1));

                mark_lines(
                    &mut per_line,
                    def_start_line,
                    body_start_line.saturating_sub(1),
                    LineCoverability::NonCoverableStructural,
                );
                mark_lines(
                    &mut per_line,
                    body_start_line,
                    body_end_line,
                    LineCoverability::CoverableExecutable,
                );
                mark_lines(
                    &mut per_line,
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

    Ok(CoverabilityMap { per_line })
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
