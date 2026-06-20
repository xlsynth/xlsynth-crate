// SPDX-License-Identifier: Apache-2.0

use crate::liberty_model::Library;
use anyhow::{Context, Result, bail};
use regex::Regex;
use std::path::Path;

/// Action applied when a cell name matches a policy rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellFilterAction {
    Include,
    Exclude,
}

/// One compiled, ordered cell-filter rule.
#[derive(Clone, Debug)]
pub struct CellFilterRule {
    pub action: CellFilterAction,
    pub pattern: String,
    regex: Regex,
}

/// Ordered cell-filter policy; the last matching rule wins.
#[derive(Clone, Debug, Default)]
pub struct CellFilterPolicy {
    rules: Vec<CellFilterRule>,
}

/// Counts describing one cell-filter application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellFilterStats {
    pub input_cells: usize,
    pub native_dont_use_cells: usize,
    pub retained_cells: usize,
    pub removed_cells: usize,
}

impl CellFilterPolicy {
    /// Parses `include REGEX` and `exclude REGEX` lines in source order.
    pub fn parse(text: &str, source_name: &str) -> Result<Self> {
        let mut rules = Vec::new();
        for (line_index, raw_line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some(separator) = line.find(char::is_whitespace) else {
                bail!(
                    "{}:{}: expected 'include REGEX' or 'exclude REGEX'",
                    source_name,
                    line_number
                );
            };
            let action_text = &line[..separator];
            let pattern = line[separator..].trim();
            if pattern.is_empty() {
                bail!("{}:{}: regex must be non-empty", source_name, line_number);
            }
            let action = match action_text {
                "include" => CellFilterAction::Include,
                "exclude" => CellFilterAction::Exclude,
                _ => {
                    bail!(
                        "{}:{}: unknown action {:?}; expected 'include' or 'exclude'",
                        source_name,
                        line_number,
                        action_text
                    )
                }
            };
            let regex = Regex::new(pattern).with_context(|| {
                format!(
                    "{}:{}: invalid {} regex {:?}",
                    source_name, line_number, action_text, pattern
                )
            })?;
            rules.push(CellFilterRule {
                action,
                pattern: pattern.to_string(),
                regex,
            });
        }
        Ok(Self { rules })
    }

    /// Reads and parses a policy file.
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading cell-filter policy '{}'", path.display()))?;
        Self::parse(&text, &path.display().to_string())
    }

    pub fn rules(&self) -> &[CellFilterRule] {
        &self.rules
    }

    /// Applies native `dont_use` followed by ordered policy rules.
    pub fn apply(&self, library: &mut Library) -> CellFilterStats {
        let input_cells = library.cells.len();
        let native_dont_use_cells = library
            .cells
            .iter()
            .filter(|cell| cell.dont_use == Some(true))
            .count();
        let mut removed_cells = 0;
        library.cells.retain_mut(|cell| {
            let mut dont_use = cell.dont_use.unwrap_or(false);
            for rule in &self.rules {
                if rule.regex.is_match(&cell.name) {
                    dont_use = rule.action == CellFilterAction::Exclude;
                }
            }
            if dont_use {
                removed_cells += 1;
                false
            } else {
                // The filtered library records final availability, including
                // explicit policy overrides of native Liberty dont_use.
                cell.dont_use = Some(false);
                true
            }
        });
        CellFilterStats {
            input_cells,
            native_dont_use_cells,
            retained_cells: library.cells.len(),
            removed_cells,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_model::Cell;

    #[test]
    fn policy_is_ordered_and_can_override_native_dont_use() {
        let mut library = Library {
            cells: vec![
                Cell {
                    name: "DLY".to_string(),
                    ..Default::default()
                },
                Cell {
                    name: "CLKBUF".to_string(),
                    dont_use: Some(true),
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    ..Default::default()
                },
                Cell {
                    name: "BUF_TEST".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let policy = CellFilterPolicy::parse(
            r#"
exclude ^DLY$
exclude ^CLK
include ^CLKBUF$
exclude _TEST$
"#,
            "test.policy",
        )
        .unwrap();

        let stats = policy.apply(&mut library);

        assert_eq!(
            stats,
            CellFilterStats {
                input_cells: 4,
                native_dont_use_cells: 1,
                retained_cells: 2,
                removed_cells: 2,
            }
        );
        assert_eq!(
            library
                .cells
                .iter()
                .map(|cell| cell.name.as_str())
                .collect::<Vec<_>>(),
            vec!["CLKBUF", "INV"]
        );
        assert!(
            library
                .cells
                .iter()
                .all(|cell| cell.dont_use == Some(false))
        );
    }

    #[test]
    fn policy_reports_line_for_invalid_action_and_regex() {
        let action_error = CellFilterPolicy::parse("drop INV", "bad.policy").unwrap_err();
        assert!(format!("{action_error:#}").contains("bad.policy:1: unknown action"));

        let regex_error = CellFilterPolicy::parse("exclude [", "bad.policy").unwrap_err();
        assert!(format!("{regex_error:#}").contains("bad.policy:1: invalid exclude regex"));
    }
}
