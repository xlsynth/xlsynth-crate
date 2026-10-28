// SPDX-License-Identifier: Apache-2.0

//! Run selected DSLX QuickCheck properties directly or as a tactic obligation.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::dslx_tactics::obligations::{
    FileWithHistory, ObligationPayload, ProverObligation, QcObligation, SourceFile,
};
use crate::dslx_tactics::script::{OblTree, OblTreeConfig};
use crate::dslx_utils::mangle_dslx_uf_map;
use crate::proof_outcome::{ProofResultHeader, ProofStatus, aggregate_proof_statuses};
use crate::prover::quickcheck::{QuickCheckError, QuickCheckErrorKind, load_quickcheck_context};
use crate::prover::types::{
    AssertionViolation, BoolPropertyResult, FnInput, QuickCheckAssertionSemantics,
    QuickCheckOptions, QuickCheckRunResult,
};
use crate::prover::uf::validate_uf_symbols;
use crate::prover::{SolverChoice, prover_for_choice};

/// The selected properties and their interpretation, shared by direct and
/// scripted proofs.
pub struct DslxQuickCheckRequest {
    pub entry_file: PathBuf,
    pub dslx_stdlib_path: Option<PathBuf>,
    pub dslx_paths: Vec<PathBuf>,
    pub solver: Option<SolverChoice>,
    pub test_filter: Option<String>,
    pub assertion_semantics: QuickCheckAssertionSemantics,
    pub assert_label_filter: Option<String>,
    /// DSLX helper names mapped to shared uninterpreted symbols.
    pub uf_map: HashMap<String, String>,
}

/// A progress boundary for one property; preparation precedes all events.
pub enum QuickCheckProgress<'a> {
    Started(&'a str),
    Finished {
        location: Option<(&'a str, Range<usize>)>,
        run: &'a QuickCheckRunResult,
    },
}

/// The counterexample in DSLX terms, without the lowering's token and
/// activation parameters.
pub struct DslxQuickCheckCounterexample<'a> {
    pub inputs: &'a [FnInput],
    pub return_value: xlsynth_pir::IrValue,
    pub assertion_violation: Option<&'a AssertionViolation>,
}

impl QuickCheckRunResult {
    /// Projects an IR counterexample onto the arguments and return value of the
    /// DSLX property.
    pub fn dslx_counterexample(&self) -> Option<DslxQuickCheckCounterexample<'_>> {
        let BoolPropertyResult::Disproved { inputs, output } = &self.result else {
            return None;
        };
        let (inputs, return_value) = if self.implicit_token {
            (
                inputs.get(2..).unwrap_or_default(),
                output
                    .value
                    .get_element(1)
                    .unwrap_or_else(|_| output.value.clone()),
            )
        } else {
            (inputs.as_slice(), output.value.clone())
        };
        Some(DslxQuickCheckCounterexample {
            inputs,
            return_value,
            assertion_violation: output.assertion_violation.as_ref(),
        })
    }
}

/// Report for a single property, with a counterexample only when one was found.
#[derive(Debug, Clone, Serialize)]
pub struct QuickCheckTestOutcome {
    pub name: String,
    pub time_micros: u128,
    #[serde(flatten)]
    pub conclusion: ProofResultHeader,
    pub counterexample: Option<String>,
    pub error_str: Option<String>,
}

/// Conjunction of the selected properties and individual reports in source
/// order.
#[derive(Debug, Clone, Serialize)]
pub struct QuickCheckOutcome {
    #[serde(flatten)]
    pub conclusion: ProofResultHeader,
    pub tests: Vec<QuickCheckTestOutcome>,
}

impl From<QuickCheckRunResult> for QuickCheckTestOutcome {
    fn from(run: QuickCheckRunResult) -> Self {
        let status = ProofStatus::from(&run.result);
        let (counterexample, error_str) = match run.result {
            BoolPropertyResult::Proved => (None, None),
            BoolPropertyResult::Disproved { inputs, output } => (
                Some(format!("inputs: {:?}, output: {:?}", inputs, output)),
                None,
            ),
            BoolPropertyResult::Inconclusive(msg)
            | BoolPropertyResult::ToolchainDisproved(msg)
            | BoolPropertyResult::Error(msg) => (None, Some(msg)),
        };
        Self {
            name: run.name,
            time_micros: run.duration.as_micros(),
            conclusion: status.into(),
            counterexample,
            error_str,
        }
    }
}

impl DslxQuickCheckRequest {
    /// Runs every selected property, including those following a disproof or a
    /// solver error.
    pub fn run(
        &self,
        options: QuickCheckOptions,
        tool_path: Option<&Path>,
        mut progress: impl FnMut(QuickCheckProgress<'_>),
    ) -> Result<QuickCheckOutcome, QuickCheckError> {
        options.validate_uf_optimization(&self.uf_map)?;
        let module_name = self
            .entry_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                QuickCheckError::new(
                    QuickCheckErrorKind::Configuration,
                    "Invalid DSLX module name",
                )
            })?;
        let uf_map = mangle_dslx_uf_map(module_name, &self.uf_map)
            .map_err(|e| QuickCheckError::new(QuickCheckErrorKind::Configuration, e))?;
        let prover = prover_for_choice(self.solver.unwrap_or(SolverChoice::Bitwuzla), tool_path);
        let suite = prover.prepare_dslx_quickchecks(
            &self.entry_file,
            self.dslx_stdlib_path.as_deref(),
            &self
                .dslx_paths
                .iter()
                .map(PathBuf::as_path)
                .collect::<Vec<_>>(),
            self.test_filter.as_deref(),
            self.assertion_semantics,
            self.assert_label_filter.as_deref(),
            &uf_map,
            options,
        )?;
        if suite.properties().is_empty() {
            return Err(QuickCheckError::no_matching_properties());
        }
        let mut tests = Vec::with_capacity(suite.properties().len());
        for property in suite.properties() {
            progress(QuickCheckProgress::Started(&property.name));
            let run = suite.prove(property.id)?;
            progress(QuickCheckProgress::Finished {
                location: property
                    .name_span
                    .clone()
                    .map(|span| (suite.source(), span)),
                run: &run,
            });
            tests.push(QuickCheckTestOutcome::from(run));
        }
        let status = aggregate_proof_statuses(tests.iter().map(|test| test.conclusion.status));
        Ok(QuickCheckOutcome {
            conclusion: status.into(),
            tests,
        })
    }

    /// Captures the discovered source snapshot and the same property
    /// interpretation for tactic workers.
    pub fn tactic_tree(&self) -> Result<OblTree, QuickCheckError> {
        validate_uf_symbols(&self.uf_map)
            .map_err(|e| QuickCheckError::new(QuickCheckErrorKind::Configuration, e))?;
        let context = load_quickcheck_context(
            &self.entry_file,
            self.dslx_stdlib_path.as_deref(),
            &self
                .dslx_paths
                .iter()
                .map(PathBuf::as_path)
                .collect::<Vec<_>>(),
            self.test_filter.as_deref(),
        )?;
        if context.properties.is_empty() {
            return Err(QuickCheckError::no_matching_properties());
        }
        let root = QcObligation {
            file: FileWithHistory {
                base_source: SourceFile::Path(context.path),
                edits: Vec::new(),
                text: context.source,
            },
            tests: context.properties.into_iter().map(|p| p.name).collect(),
            uf_map: self.uf_map.clone().into_iter().collect(),
            assertion_semantics: self.assertion_semantics,
            assert_label_filter: self.assert_label_filter.clone(),
        };
        Ok(OblTree::new(
            ProverObligation {
                selector_segment: String::new(),
                description: None,
                payload: ObligationPayload::QuickCheck(root),
            },
            OblTreeConfig {
                dslx_stdlib_path: self.dslx_stdlib_path.clone(),
                dslx_paths: self.dslx_paths.clone(),
                solver: self.solver,
                timeout_ms: None,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::types::FnOutput;
    use std::time::Duration;
    use xlsynth_pir::IrValue;

    #[test]
    fn invalid_uf_symbol_rejects_direct_and_tactic_quickchecks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.x");
        std::fs::write(
            &path,
            r#"
fn helper(x: u1) -> u1 { x }
#[quickcheck] fn p(x: u1) -> bool { helper(x) == x }
"#,
        )
        .unwrap();
        let request = DslxQuickCheckRequest {
            entry_file: path,
            dslx_stdlib_path: None,
            dslx_paths: vec![],
            solver: Some(SolverChoice::Bitwuzla),
            test_filter: None,
            assertion_semantics: QuickCheckAssertionSemantics::Assume,
            assert_label_filter: None,
            uf_map: HashMap::from([("helper".into(), "F\0x".into())]),
        };
        let mut events = 0;
        let error = request
            .run(QuickCheckOptions { optimize: false }, None, |_| events += 1)
            .unwrap_err();
        assert_eq!(error.kind, QuickCheckErrorKind::Configuration);
        assert_eq!(events, 0);
        assert_eq!(
            request.tactic_tree().unwrap_err().kind,
            QuickCheckErrorKind::Configuration
        );
    }

    #[test]
    fn quickcheck_rejects_nul_in_names_before_starting_a_proof() {
        for (path, helper, invalid_name) in [
            ("bad\0name.x", "helper", "module"),
            ("valid.x", "bad\0helper", "function"),
        ] {
            let request = DslxQuickCheckRequest {
                entry_file: path.into(),
                dslx_stdlib_path: None,
                dslx_paths: vec![],
                solver: None,
                test_filter: None,
                assertion_semantics: QuickCheckAssertionSemantics::Assume,
                assert_label_filter: None,
                uf_map: HashMap::from([(helper.into(), "F".into())]),
            };
            let error = request
                .run(QuickCheckOptions { optimize: false }, None, |_| {
                    panic!("a property started with an invalid name")
                })
                .unwrap_err();
            assert_eq!(error.kind, QuickCheckErrorKind::Configuration);
            assert_eq!(
                error.message,
                format!("xlsynth error: DSLX {invalid_name} name contains a NUL byte")
            );
        }
    }

    #[test]
    fn quickcheck_json_does_not_conflate_errors_with_counterexamples() {
        for (result, status, counterexample, error) in [
            (BoolPropertyResult::Proved, "proved", None, None),
            (
                BoolPropertyResult::Disproved {
                    inputs: vec![],
                    output: FnOutput {
                        value: IrValue::make_ubits(1, 0).unwrap(),
                        assertion_violation: None,
                    },
                },
                "disproved",
                Some(
                    "inputs: [], output: FnOutput { value: bits[1]:0, assertion_violation: None }",
                ),
                None,
            ),
            (
                BoolPropertyResult::Inconclusive("unknown".into()),
                "inconclusive",
                None,
                Some("unknown"),
            ),
            (
                BoolPropertyResult::Error("solver failed".into()),
                "execution-error",
                None,
                Some("solver failed"),
            ),
            (
                BoolPropertyResult::ToolchainDisproved("tool failed".into()),
                "execution-error",
                None,
                Some("tool failed"),
            ),
        ] {
            let outcome = QuickCheckTestOutcome::from(QuickCheckRunResult {
                name: "property".into(),
                implicit_token: false,
                duration: Duration::ZERO,
                result,
            });
            assert_eq!(
                serde_json::to_value(outcome).unwrap(),
                serde_json::json!({
                    "name": "property",
                    "time_micros": 0,
                    "status": status,
                    "success": status == "proved",
                    "counterexample": counterexample,
                    "error_str": error,
                }),
            );
        }
    }
}
