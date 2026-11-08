// SPDX-License-Identifier: Apache-2.0

//! Source diagnostics for DSLX functions that require an implicit token.

use annotate_snippets::{AnnotationKind, Group, Level, Renderer, Snippet};
use std::fmt;
use std::ops::Range;
use std::path::Path;

use crate::parser::uses_only_const_assert_operands;

/// A verified operation in the original, captured DSLX source.
pub(crate) struct EffectLocation<'a> {
    pub(crate) path: &'a Path,
    pub(crate) source: &'a str,
    pub(crate) range: Range<usize>,
    pub(crate) operation: &'a str,
}

/// One reachable effect and the concrete DSLX call path leading to it.
pub(crate) struct EffectDiagnostic<'a> {
    pub(crate) call_path: Vec<String>,
    pub(crate) operation: Option<String>,
    pub(crate) locations: Vec<EffectLocation<'a>>,
    pub(crate) const_assert_replacement: Option<String>,
}

/// Spell an unconditional assertion of known compile-time values as a DSLX
/// compile-time check.
pub(crate) fn const_assert_replacement(
    operation: &str,
    arguments: &[&str],
    constant_names: &[String],
) -> Option<String> {
    let constant = |expr| uses_only_const_assert_operands(expr, constant_names);
    let predicate = match (operation, arguments) {
        ("assert!" | "assert_fmt!", [predicate, _]) if constant(predicate) => {
            predicate.trim().to_owned()
        }
        (_, [lhs, rhs]) => {
            let comparison = match operation {
                "assert_eq" => "==",
                "assert_ne" => "!=",
                "assert_lt" => "<",
                "assert_le" => "<=",
                "assert_gt" => ">",
                "assert_ge" => ">=",
                _ => return None,
            };
            if !constant(lhs) || !constant(rhs) {
                return None;
            }
            format!("({}) {comparison} ({})", lhs.trim(), rhs.trim())
        }
        _ => return None,
    };
    Some(format!("const_assert!({predicate})"))
}

impl fmt::Display for EffectDiagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = match &self.operation {
            Some(operation) => format!(
                "cannot import an effectful DSLX function into BX (`{operation}` requires an implicit token)"
            ),
            None => {
                "cannot import this DSLX function into BX because it requires an implicit token"
                    .to_owned()
            }
        };
        let mut report = Group::with_title(Level::ERROR.primary_title(title));
        if let Some((first, related)) = self.locations.split_first() {
            let mut snippet = Snippet::source(first.source)
                .path(first.path.to_string_lossy())
                .fold(true)
                .annotation(
                    AnnotationKind::Primary
                        .span(first.range.clone())
                        .label(format!("`{}` introduces the effect", first.operation)),
                );
            for site in related {
                snippet = snippet.annotation(
                    AnnotationKind::Context
                        .span(site.range.clone())
                        .label(format!("another `{}` in this function", site.operation)),
                );
            }
            report = report.element(snippet);
        }
        if !self.call_path.is_empty() {
            report = report.element(
                Level::NOTE.message(format!("DSLX call path: {}", self.call_path.join(" -> "))),
            );
        }
        if let Some(replacement) = &self.const_assert_replacement {
            report = report.element(
                Level::HELP.message(format!("for a compile-time invariant, use `{replacement}`")),
            );
        }
        write!(f, "{}", Renderer::plain().render(&[report]))
    }
}

#[cfg(test)]
mod tests {
    use super::const_assert_replacement;

    #[test]
    fn comparison_hints_preserve_grouping_and_exclude_runtime_values() {
        let names = ["N".to_owned(), "M".to_owned()];
        assert_eq!(
            const_assert_replacement("assert_eq", &["N | M", "u32:3"], &names).as_deref(),
            Some("const_assert!((N | M) == (u32:3))"),
        );
        for (builtin, operator) in [
            ("assert_lt", "<"),
            ("assert_le", "<="),
            ("assert_gt", ">"),
            ("assert_ge", ">="),
            ("assert_ne", "!="),
        ] {
            assert_eq!(
                const_assert_replacement(builtin, &["N", "M"], &names),
                Some(format!("const_assert!((N) {operator} (M))")),
            );
        }
        for runtime_operand in ["payload", "u100", "read_width(N)"] {
            assert_eq!(
                const_assert_replacement("assert_eq", &["N", runtime_operand], &names),
                None,
            );
        }
    }
}
