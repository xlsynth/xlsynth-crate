// SPDX-License-Identifier: Apache-2.0

//! Implements the `prove-quickcheck` sub-command – prove that every
//! `#[quickcheck]` function in a DSLX file (or a selected one) always
//! returns `true` for every possible input using an SMT solver.

use crate::common::parse_dslx_uf_spec;
use crate::proofs::report_setup_error;
use crate::proofs::script::TacticScript;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{
    ToolchainConfig, get_dslx_search_paths, get_dslx_stdlib_path, proof_driver,
};
use annotate_snippets::{AnnotationKind, Group, Level, Origin, Renderer, Snippet};
use std::io::{self, IsTerminal, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use xlsynth_prover::dslx_quickcheck::{DslxQuickCheckRequest, QuickCheckProgress};
use xlsynth_prover::proof_outcome::ProofSummary;
use xlsynth_prover::prover::QuickCheckErrorKind;
use xlsynth_prover::prover::types::{
    BoolPropertyResult, QuickCheckAssertionSemantics, QuickCheckOptions, QuickCheckRunResult,
};

const SUBCOMMAND: &str = "prove-quickcheck";

/// Flushes the start boundary before the caller begins optimization/solving.
fn write_quickcheck_started(writer: &mut impl Write, name: &str) -> io::Result<()> {
    writeln!(writer, "[ RUN QUICKCHECK        ] {name}")?;
    writer.flush()
}

/// Renders a proof result; only the optional function-name span is highlighted.
fn write_quickcheck_finished(
    writer: &mut impl Write,
    path: &Path,
    location: Option<(&str, Range<usize>)>,
    run: &QuickCheckRunResult,
    renderer: &Renderer,
) -> io::Result<()> {
    if matches!(run.result, BoolPropertyResult::Proved) {
        writeln!(writer, "[                    OK ] {}", run.name)?;
    } else {
        let (description, label) = match &run.result {
            BoolPropertyResult::Disproved { output, .. }
                if output.assertion_violation.is_some() =>
            {
                ("can fail an assertion", "property fails for these inputs")
            }
            BoolPropertyResult::Disproved { .. } => {
                ("has a counterexample", "property fails for these inputs")
            }
            BoolPropertyResult::Inconclusive(_) => ("could not be proved", "proof is inconclusive"),
            BoolPropertyResult::Error(_) => ("failed to run", "proof could not be completed"),
            BoolPropertyResult::ToolchainDisproved(_) => {
                ("external proof failed", "external tool reported failure")
            }
            BoolPropertyResult::Proved => unreachable!("success handled above"),
        };
        let title = format!("QuickCheck `{}` {description}", run.name);
        let mut report = Group::with_title(Level::ERROR.primary_title(title));
        let path = path.to_string_lossy();
        // Never guess a span. Validate even library-supplied ranges so the
        // renderer cannot panic if location metadata is absent or inconsistent.
        if let Some((source, span)) =
            location.filter(|(source, span)| source.get(span.clone()) == Some(run.name.as_str()))
        {
            report = report.element(
                Snippet::source(source)
                    .path(path)
                    .fold(true)
                    .annotation(AnnotationKind::Primary.span(span).label(label)),
            );
        } else {
            report = report.element(Origin::path(path));
        }
        if let Some(counterexample) = run.dslx_counterexample() {
            let inputs = counterexample.inputs;
            if !inputs.is_empty() {
                report = report.element(
                    Level::NOTE.message(
                        inputs
                            .iter()
                            .map(|input| format!("{} = {}", input.name, input.value))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                );
            }
            let return_value = counterexample.return_value;
            let returned = if return_value.bits_equals_u64_value(0) {
                "false".into()
            } else if return_value.bits_equals_u64_value(1) {
                "true".into()
            } else {
                return_value.to_string()
            };
            report = report.element(Level::NOTE.message(format!("returned {returned}")));
            if let Some(violation) = counterexample.assertion_violation {
                report = report
                    .element(
                        Level::NOTE.message(format!("assertion failed: {}", violation.message)),
                    )
                    .element(Level::NOTE.message(format!("assertion label: {}", violation.label)));
            }
        } else if let BoolPropertyResult::Inconclusive(message)
        | BoolPropertyResult::ToolchainDisproved(message)
        | BoolPropertyResult::Error(message) = &run.result
        {
            report = report.element(Level::NOTE.message(message));
        }
        writeln!(writer, "{}", renderer.render(&[report]))?;
        writeln!(writer, "[                FAILED ] {}", run.name)?;
    }
    writer.flush()
}

/// Reports setup errors separately because no property has started yet.
fn write_quickcheck_preparation_error(
    writer: &mut impl Write,
    path: &Path,
    message: &str,
    renderer: &Renderer,
) -> io::Result<()> {
    let report = Level::ERROR
        .primary_title("Could not prepare QuickCheck proofs")
        .element(Origin::path(path.to_string_lossy()))
        .element(Level::NOTE.message(message));
    writeln!(writer, "{}", renderer.render(&[report]))?;
    writer.flush()
}

/// Implements the `prove-quickcheck` sub-command.
pub fn handle_prove_quickcheck(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file_str = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let input_path = std::path::Path::new(input_file_str);
    let start = std::time::Instant::now();
    let output_json = matches.get_one::<String>("output_json").map(String::as_str);

    let request = DslxQuickCheckRequest {
        entry_file: input_path.to_path_buf(),
        dslx_stdlib_path: get_dslx_stdlib_path(matches, config).map(PathBuf::from),
        dslx_paths: get_dslx_search_paths(matches, config),
        solver: matches
            .get_one::<String>("solver")
            .map(|s| s.parse().unwrap()),
        test_filter: matches.get_one::<String>("test_filter").cloned(),
        assertion_semantics: *matches
            .get_one::<QuickCheckAssertionSemantics>("assertion_semantics")
            .expect("assertion semantics default"),
        assert_label_filter: matches.get_one::<String>("assert_label_filter").cloned(),
        uf_map: parse_dslx_uf_spec(matches.get_many::<String>("uf"))
            .unwrap_or_else(|message| report_setup_error(start, output_json, SUBCOMMAND, &message)),
    };
    if let Some(script) = TacticScript::from_matches(matches) {
        script.execute_and_exit(
            request.tactic_tree().map_err(|e| e.to_string()),
            SUBCOMMAND,
            "QuickCheck obligations proved",
            output_json,
            proof_driver(matches),
        );
    }

    let optimize = *matches.get_one::<bool>("opt").expect("opt default");
    let tool_path = config
        .as_ref()
        .and_then(|c| c.tool_path.as_deref())
        .map(Path::new);
    let mut stderr = io::stderr().lock();
    let renderer = if stderr.is_terminal() && std::env::var_os("NO_COLOR").is_none() {
        Renderer::styled()
    } else {
        Renderer::plain()
    };
    let result = request.run(QuickCheckOptions { optimize }, tool_path, |event| {
        let written = match event {
            QuickCheckProgress::Started(name) => write_quickcheck_started(&mut stderr, name),
            QuickCheckProgress::Finished { location, run } => {
                write_quickcheck_finished(&mut stderr, input_path, location, run, &renderer)
            }
        };
        if let Err(error) = written {
            report_cli_error_and_exit(
                &format!("Failed to report QuickCheck proof: {error}"),
                Some(SUBCOMMAND),
                vec![],
            );
        }
    });
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            if let Some(path) = output_json {
                let report = ProofSummary::execution_error(start.elapsed(), error.to_string());
                std::fs::write(path, serde_json::to_vec(&report).unwrap()).unwrap();
            }
            match error.kind {
                QuickCheckErrorKind::OptimizingUninterpretedFunctions => report_cli_error_and_exit(
                    "--uf cannot be combined with --opt=true\n  XLS optimization inlines calls and would erase UF boundaries.\n  Use --opt=false to preserve uninterpreted functions.",
                    Some(SUBCOMMAND),
                    vec![],
                ),
                QuickCheckErrorKind::NoMatchingProperties => report_cli_error_and_exit(
                    &error.to_string(),
                    Some(SUBCOMMAND),
                    vec![("file", input_file_str)],
                ),
                _ => {
                    if let Err(write_error) = write_quickcheck_preparation_error(
                        &mut stderr,
                        input_path,
                        &error.to_string(),
                        &renderer,
                    ) {
                        report_cli_error_and_exit(
                            &format!("Failed to write proof diagnostic: {write_error}"),
                            Some(SUBCOMMAND),
                            vec![],
                        );
                    }
                    std::process::exit(1);
                }
            }
        }
    };
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap();
    }
    let status = outcome.conclusion.status;
    if status.is_proved() {
        println!("Success: All QuickChecks proved");
    } else {
        println!("Failure: Some QuickChecks did not succeed");
    }
    std::process::exit(status.exit_code());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use xlsynth_pir::IrValue;
    use xlsynth_prover::prover::types::{
        AssertionViolation, FnInput, FnOutput, QuickCheckRunResult,
    };
    use xlsynth_test_helpers::compare_golden_text;

    /// Golden output covers all result kinds without solver-dependent models.
    #[test]
    fn quickcheck_diagnostic_goldens() {
        let results = [
            ("passing", BoolPropertyResult::Proved),
            (
                "always_fail",
                BoolPropertyResult::Disproved {
                    inputs: vec![
                        FnInput {
                            name: "a".into(),
                            value: IrValue::parse_typed("bits[1]:0").unwrap(),
                        },
                        FnInput {
                            name: "b".into(),
                            value: IrValue::parse_typed("bits[2]:0").unwrap(),
                        },
                    ],
                    output: FnOutput {
                        value: IrValue::make_ubits(1, 0).unwrap(),
                        assertion_violation: None,
                    },
                },
            ),
            (
                "assertion_failure",
                BoolPropertyResult::Disproved {
                    inputs: vec![],
                    output: FnOutput {
                        value: IrValue::make_ubits(1, 1).unwrap(),
                        assertion_violation: Some(AssertionViolation {
                            message: "expected nonzero input".into(),
                            label: "nonzero".into(),
                        }),
                    },
                },
            ),
            (
                "aggregate_failure",
                BoolPropertyResult::Disproved {
                    inputs: vec![
                        FnInput {
                            name: "pair".into(),
                            value: IrValue::parse_typed(
                                "(bits[8]:7, bits[129]:0x100000000000000000000000000000001)",
                            )
                            .unwrap(),
                        },
                        FnInput {
                            name: "values".into(),
                            value: IrValue::parse_typed("[bits[4]:2, bits[4]:3]").unwrap(),
                        },
                    ],
                    output: FnOutput {
                        value: IrValue::make_ubits(1, 0).unwrap(),
                        assertion_violation: None,
                    },
                },
            ),
            (
                "unknown",
                BoolPropertyResult::Inconclusive("solver returned unknown".into()),
            ),
            (
                "broken",
                BoolPropertyResult::Error("solver unavailable\nrebuild with solver support".into()),
            ),
            (
                "external_failure",
                BoolPropertyResult::ToolchainDisproved("external tool exited with status 1".into()),
            ),
        ];
        let mut output = Vec::new();
        for (name, result) in results {
            let run = QuickCheckRunResult {
                name: name.into(),
                implicit_token: false,
                duration: Duration::ZERO,
                result,
            };
            write_quickcheck_started(&mut output, name).unwrap();
            write_quickcheck_finished(
                &mut output,
                Path::new("qc.x"),
                None,
                &run,
                &Renderer::plain(),
            )
            .unwrap();
        }
        compare_golden_text(
            &String::from_utf8(output).unwrap(),
            "src/testdata/quickcheck/diagnostics.golden.txt",
        );
    }

    #[test]
    fn quickcheck_diagnostic_hides_implicit_ir_plumbing() {
        let run = QuickCheckRunResult {
            name: "assertion_failure".into(),
            implicit_token: true,
            duration: Duration::ZERO,
            result: BoolPropertyResult::Disproved {
                inputs: vec![
                    FnInput {
                        name: "__token".into(),
                        value: IrValue::make_token(),
                    },
                    FnInput {
                        name: "__activated".into(),
                        value: IrValue::make_ubits(1, 1).unwrap(),
                    },
                    FnInput {
                        name: "x".into(),
                        value: IrValue::make_ubits(8, 0).unwrap(),
                    },
                ],
                output: FnOutput {
                    value: IrValue::make_tuple(&[
                        IrValue::make_token(),
                        IrValue::make_ubits(1, 1).unwrap(),
                    ]),
                    assertion_violation: Some(AssertionViolation {
                        message: "expected nonzero input".into(),
                        label: "nonzero".into(),
                    }),
                },
            },
        };
        let mut output = Vec::new();
        write_quickcheck_started(&mut output, &run.name).unwrap();
        write_quickcheck_finished(
            &mut output,
            Path::new("qc.x"),
            None,
            &run,
            &Renderer::plain(),
        )
        .unwrap();
        compare_golden_text(
            &String::from_utf8(output).unwrap(),
            "src/testdata/quickcheck/implicit_token.golden.txt",
        );
    }

    /// Unicode/CRLF and multiline signatures preserve byte-based locations;
    /// invalid metadata deliberately falls back to a filename-only diagnostic.
    #[test]
    fn quickcheck_source_location_goldens() {
        let source =
            "// λ🌍\r\n#[quickcheck]\r\nfn property(\r\n    x: u8,\r\n) -> bool { false }\r\n";
        let run = QuickCheckRunResult {
            name: "property".into(),
            implicit_token: false,
            duration: Duration::ZERO,
            result: BoolPropertyResult::Inconclusive("solver returned unknown".into()),
        };
        let start = source.find("property(").unwrap();
        let mut output = Vec::new();
        for span in [start..start + run.name.len(), 1..2, 5..usize::MAX] {
            write_quickcheck_started(&mut output, &run.name).unwrap();
            write_quickcheck_finished(
                &mut output,
                Path::new("qc.x"),
                Some((source, span)),
                &run,
                &Renderer::plain(),
            )
            .unwrap();
        }
        compare_golden_text(
            &String::from_utf8(output).unwrap(),
            "src/testdata/quickcheck/source_locations.golden.txt",
        );
    }

    #[test]
    fn quickcheck_preparation_error_golden() {
        let mut output = Vec::new();
        write_quickcheck_preparation_error(&mut output, Path::new("qc.x"),
            "invalid regular expression in quickcheck test filter: regex parse error:\n    [\n    ^\nerror: unclosed character class",
            &Renderer::plain()).unwrap();
        compare_golden_text(
            &String::from_utf8(output).unwrap(),
            "src/testdata/quickcheck/preparation.golden.txt",
        );
    }

    #[derive(Default)]
    struct FlushRecorder {
        bytes: Vec<u8>,
        flushed: Vec<usize>,
    }

    impl Write for FlushRecorder {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushed.push(self.bytes.len());
            Ok(())
        }
    }

    #[test]
    fn quickcheck_progress_flushes_each_boundary() {
        let mut writer = FlushRecorder::default();
        let run = QuickCheckRunResult {
            name: "p".into(),
            implicit_token: false,
            duration: Duration::ZERO,
            result: BoolPropertyResult::Proved,
        };
        write_quickcheck_started(&mut writer, "p").unwrap();
        let first_length = writer.bytes.len();
        assert_eq!(writer.flushed, [first_length]);
        write_quickcheck_finished(
            &mut writer,
            Path::new("qc.x"),
            None,
            &run,
            &Renderer::plain(),
        )
        .unwrap();
        assert_eq!(writer.flushed, [first_length, writer.bytes.len()]);
    }
}
