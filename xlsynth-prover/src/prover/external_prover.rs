// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::Prover;
use super::quickcheck::{
    PreparedQuickCheckBackend, PreparedQuickCheckSuite, QuickCheckError, QuickCheckErrorKind,
    QuickCheckProofResult, QuickCheckProperty, load_quickcheck_context,
};
use super::types::{
    AssertionSemantics, BoolPropertyResult, EquivParallelism, EquivResult, ProverFn,
    QuickCheckAssertionSemantics, QuickCheckOptions,
};
use crate::dslx_utils::join_dslx_search_paths;
use crate::proof_phases::{Phase, PhaseCollector};
use crate::toolchain_shim::run_prove_quickcheck_main_with_phases;
use regex::escape;
use xlsynth_pir::ir::emit_fn;
use xlsynth_pir::ir_inline::package_with_reachable_callees;
use xlsynth_pir::prove_equiv_via_toolchain::{self, ToolchainEquivResult};

const MAX_TOOLCHAIN_MESSAGE: usize = 512;

pub enum ExternalProver {
    ToolExe(PathBuf),
    ToolDir(PathBuf),
    Toolchain,
}

fn resolve_tool_path(prover: &ExternalProver, exe_name: &str) -> Result<PathBuf, String> {
    let (candidate, label) = match prover {
        ExternalProver::ToolExe(path) => (path.clone(), "tool executable".to_string()),
        ExternalProver::ToolDir(dir) => (dir.join(exe_name), exe_name.to_string()),
        ExternalProver::Toolchain => match std::env::var("XLSYNTH_TOOLS") {
            Ok(dir) => (PathBuf::from(dir).join(exe_name), exe_name.to_string()),
            Err(_) => {
                return Err("XLSYNTH_TOOLS is not set; cannot locate external tool".to_string());
            }
        },
    };

    if candidate.exists() {
        Ok(candidate)
    } else {
        Err(format!("{label} not found at {}", candidate.display()))
    }
}

impl Prover for ExternalProver {
    fn prove_ir_fn_equiv(
        self: &Self,
        lhs: &xlsynth_pir::ir::Fn,
        rhs: &xlsynth_pir::ir::Fn,
    ) -> EquivResult {
        let lhs = ProverFn::new(lhs, None);
        let rhs = ProverFn::new(rhs, None);
        self.prove_ir_equiv(
            &lhs,
            &rhs,
            EquivParallelism::SingleThreaded,
            AssertionSemantics::Ignore,
            None,
            false,
        )
    }

    fn prove_ir_pkg_text_equiv(
        self: &Self,
        lhs_pkg_text: &str,
        rhs_pkg_text: &str,
        top: Option<&str>,
    ) -> EquivResult {
        self.prove_text_with_phases(lhs_pkg_text, rhs_pkg_text, top, &PhaseCollector::default())
    }

    fn prove_ir_equiv<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        strategy: EquivParallelism,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult {
        self.prove_ir_equiv_with_phases(
            lhs,
            rhs,
            strategy,
            assertion_semantics,
            assert_label_filter,
            allow_flatten,
            &PhaseCollector::default(),
        )
    }

    fn prove_ir_equiv_with_phases<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        strategy: EquivParallelism,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
        phases: &PhaseCollector,
    ) -> EquivResult {
        if !lhs.uf_map.is_empty() || !rhs.uf_map.is_empty() {
            return EquivResult::Error("External provers do not support UFs".to_string());
        }
        match strategy {
            EquivParallelism::SingleThreaded => {
                if lhs.fixed_implicit_activation || rhs.fixed_implicit_activation {
                    return EquivResult::Error(
                        "External provers do not support fixed implicit activation".to_string(),
                    );
                }
                if lhs.has_domains() || rhs.has_domains() {
                    return EquivResult::Error(
                        "External provers do not support parameter domains".to_string(),
                    );
                }
                if assertion_semantics != AssertionSemantics::Ignore {
                    return EquivResult::Error(
                        "External provers require assertion_semantics=ignore".to_string(),
                    );
                }
                if assert_label_filter.is_some() {
                    return EquivResult::Error(
                        "External provers do not support assertion label filters".to_string(),
                    );
                }
                if allow_flatten {
                    return EquivResult::Error(
                        "External provers do not support flattening".to_string(),
                    );
                }
                let inputs = phases.prepare(|| (toolchain_input(lhs), toolchain_input(rhs)));
                match inputs {
                    (Ok(lhs_pkg), Ok(rhs_pkg)) => {
                        // Each package marks its own entry, even when the names
                        // differ.
                        self.prove_text_with_phases(&lhs_pkg, &rhs_pkg, None, phases)
                    }
                    (Err(e), _) | (_, Err(e)) => EquivResult::Error(e),
                }
            }
            EquivParallelism::OutputBits => {
                if assert_label_filter.is_some() {
                    return EquivResult::Error(
                        "External provers do not support assertion label filters".to_string(),
                    );
                }
                EquivResult::Error(
                    "External provers do not support output-bits parallel strategy".to_string(),
                )
            }
            EquivParallelism::InputBitSplit => {
                if assert_label_filter.is_some() {
                    return EquivResult::Error(
                        "External provers do not support assertion label filters".to_string(),
                    );
                }
                EquivResult::Error(
                    "External provers do not support input-bit split strategy".to_string(),
                )
            }
        }
    }

    fn prove_ir_quickcheck<'a>(
        self: &Self,
        _ir_fn: &ProverFn<'a>,
        _assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult {
        if assert_label_filter.is_some() {
            return BoolPropertyResult::Error(
                "External provers do not support assertion label filters".to_string(),
            );
        }
        BoolPropertyResult::Error(
            "External provers do not support IR-level quickcheck; use prove_dslx_quickcheck"
                .to_string(),
        )
    }

    fn prepare_dslx_quickchecks(
        &self,
        entry_file: &Path,
        dslx_stdlib_path: Option<&Path>,
        additional_search_paths: &[&Path],
        test_filter: Option<&str>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
        options: QuickCheckOptions,
    ) -> Result<PreparedQuickCheckSuite<'_>, QuickCheckError> {
        let exe = if assertion_semantics != QuickCheckAssertionSemantics::Never {
            Err("External quickcheck requires assertion_semantics=never".to_string())
        } else if assert_label_filter.is_some() {
            Err("External quickcheck does not support assertion label filters".to_string())
        } else if !uf_map.is_empty() {
            Err("External quickcheck does not support uninterpreted functions".to_string())
        } else if !options.optimize {
            Err("The XLS toolchain always optimizes QuickCheck IR; use an in-process solver to disable optimization".to_string())
        } else {
            resolve_tool_path(self, "prove_quickcheck_main")
        }.map_err(|message| QuickCheckError::new(QuickCheckErrorKind::Configuration, message))?;
        join_dslx_search_paths(additional_search_paths, ':')
            .map_err(|message| QuickCheckError::new(QuickCheckErrorKind::Configuration, message))?;
        let context = load_quickcheck_context(
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
        )?;
        let backend = ExternalQuickChecks {
            exe,
            entry_file: entry_file.into(),
            source: context.source.clone(),
            stdlib: dslx_stdlib_path.map(Path::to_path_buf),
            search_paths: additional_search_paths
                .iter()
                .map(|p| p.to_path_buf())
                .collect(),
        };
        Ok(PreparedQuickCheckSuite::new(context, Box::new(backend)))
    }
}

impl ExternalProver {
    /// Records only a launched checker, after all IR preparation has succeeded.
    fn prove_text_with_phases(
        &self,
        lhs: &str,
        rhs: &str,
        top: Option<&str>,
        phases: &PhaseCollector,
    ) -> EquivResult {
        let exe = match self {
            Self::ToolExe(path) => path.clone(),
            Self::ToolDir(dir) => {
                let exe = dir.join("check_ir_equivalence_main");
                if !exe.exists() {
                    return EquivResult::Error(format!(
                        "check_ir_equivalence_main not found in {}",
                        dir.display()
                    ));
                }
                exe
            }
            Self::Toolchain => match std::env::var("XLSYNTH_TOOLS") {
                Ok(dir) => {
                    return Self::ToolDir(dir.into()).prove_text_with_phases(lhs, rhs, top, phases);
                }
                Err(_) => {
                    return EquivResult::Error(
                        "XLSYNTH_TOOLS is not set; cannot run toolchain equivalence".into(),
                    );
                }
            },
        };
        prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_exe_timed(
            lhs,
            rhs,
            top,
            exe,
            None,
            |duration| phases.record(Phase::ExternalTool, duration),
        )
        .into()
    }
}

struct ExternalQuickChecks {
    exe: PathBuf,
    entry_file: PathBuf,
    source: String,
    stdlib: Option<PathBuf>,
    search_paths: Vec<PathBuf>,
}

impl PreparedQuickCheckBackend for ExternalQuickChecks {
    fn prove(
        &self,
        property: &QuickCheckProperty,
        phases: &PhaseCollector,
    ) -> QuickCheckProofResult {
        let prepare = phases.start_prepare();
        let result = match std::fs::read_to_string(&self.entry_file) {
            Ok(source) if source == self.source => {
                let filter = format!("^{}$", escape(&property.name));
                let paths: Vec<_> = self.search_paths.iter().map(PathBuf::as_path).collect();
                drop(prepare);
                match run_prove_quickcheck_main_with_phases(
                    &self.exe,
                    &self.entry_file,
                    self.stdlib.as_deref(),
                    &paths,
                    &filter,
                    phases,
                ) {
                    Ok(_) => BoolPropertyResult::Proved,
                    // The tool's textual failure does not distinguish a
                    // counterexample from a failed invocation or solver.
                    Err(message) => BoolPropertyResult::Error(
                        message.chars().take(MAX_TOOLCHAIN_MESSAGE).collect(),
                    ),
                }
            }
            Ok(_) => BoolPropertyResult::Error(
                "DSLX input changed after QuickCheck preparation; prepare the suite again".into(),
            ),
            Err(error) => {
                BoolPropertyResult::Error(format!("Failed to read prepared DSLX input: {error}"))
            }
        };
        QuickCheckProofResult {
            implicit_token: false,
            result,
        }
    }
}

impl From<ToolchainEquivResult> for EquivResult {
    fn from(result: ToolchainEquivResult) -> Self {
        match result {
            ToolchainEquivResult::Proved => EquivResult::Proved,
            // This legacy result is inferred from a nonzero process exit, not
            // a structured counterexample. It cannot establish disproof.
            ToolchainEquivResult::Disproved(msg) => EquivResult::Error(msg),
            ToolchainEquivResult::TimedOutOrInterrupted(msg) => EquivResult::Inconclusive(msg),
            ToolchainEquivResult::Error(msg) => EquivResult::Error(msg),
        }
    }
}

/// Sends the requested function, including signature edits, with its helper
/// definitions.
fn toolchain_input(entry: &ProverFn<'_>) -> Result<String, String> {
    match entry.pkg_ref {
        Some(pkg) => Ok(package_with_reachable_callees(pkg, entry.fn_ref)?.to_string()),
        None => Ok(format!("package proof\n\n{}", emit_fn(entry.fn_ref, true))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use test_case::test_case;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn external_checker_receives_the_edited_entry_and_its_callees() {
        let ir = r#"package source
fn helper(x: bits[1] id=1) -> bits[1] {
  ret y: bits[1] = not(x, id=2)
}
fn selected(x: bits[1] id=3, unused: bits[1] id=4) -> bits[1] {
  ret y: bits[1] = invoke(x, to_apply=helper, id=5)
}
top fn original(x: bits[1] id=6) -> bits[1] {
  ret y: bits[1] = invoke(x, x, to_apply=selected, id=7)
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let selected = pkg
            .get_fn("selected")
            .unwrap()
            .clone()
            .drop_params(&["unused".to_string()])
            .unwrap();
        let input = toolchain_input(&ProverFn::new(&selected, Some(&pkg))).unwrap();
        let sent = Parser::new(&input).parse_and_validate_package().unwrap();
        assert_eq!(sent.get_top_fn().unwrap().to_string(), selected.to_string());
        assert_eq!(
            sent.get_fn("helper").unwrap().to_string(),
            pkg.get_fn("helper").unwrap().to_string()
        );
        assert!(sent.get_fn("original").is_none());
    }

    #[test_case(None; "ir_quickcheck")]
    #[test_case(Some("label"); "assertion_filter")]
    fn unsupported_ir_quickchecks_are_errors(assert_label_filter: Option<&str>) {
        let ir = r#"package test

fn f(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = ProverFn::new(pkg.get_fn("f").unwrap(), Some(&pkg));
        let prover = ExternalProver::Toolchain;
        let result = prover.prove_ir_quickcheck(
            &f,
            QuickCheckAssertionSemantics::Never,
            assert_label_filter,
        );
        assert!(matches!(result, BoolPropertyResult::Error(_)), "{result:?}");
    }

    #[test]
    fn legacy_toolchain_exit_failure_is_not_disproof() {
        let message = "tool exited with status 1".to_string();
        assert_eq!(
            EquivResult::from(ToolchainEquivResult::Disproved(message.clone())),
            EquivResult::Error(message),
        );
        assert_eq!(
            EquivResult::from(ToolchainEquivResult::Proved),
            EquivResult::Proved,
        );
        let message = "tool timed out".to_string();
        assert_eq!(
            EquivResult::from(ToolchainEquivResult::TimedOutOrInterrupted(message.clone())),
            EquivResult::Inconclusive(message),
        );
    }

    #[cfg(unix)]
    #[test_case("exit 0\n", true; "success")]
    #[test_case("echo service unavailable >&2\nexit 1\n", false; "ordinary_failure")]
    #[test_case("exit 23\n", false; "unexpected_exit_code")]
    #[test_case("kill -TERM $$\n", false; "signal")]
    fn external_quickcheck_process_outcomes_require_success(script: &str, expect_success: bool) {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("prove_quickcheck_main");
        std::fs::write(&exe, format!("#!/bin/sh\n{script}")).unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        let entry_file = dir.path().join("property.x");
        let source = "#[quickcheck]\nfn property(x: u1) -> bool { x == x }\n";
        std::fs::write(&entry_file, source).unwrap();
        let backend = ExternalQuickChecks {
            exe,
            entry_file,
            source: source.into(),
            stdlib: None,
            search_paths: Vec::new(),
        };
        let property = QuickCheckProperty {
            id: 0,
            name: "property".into(),
            name_span: None,
            requires_implicit_token: false,
        };
        let phases = PhaseCollector::default();
        let result = backend.prove(&property, &phases).result;
        let measured = phases.snapshot();
        assert!(measured.prepare_work_micros.is_some());
        assert!(measured.external_tool_work_micros.is_some());
        assert_eq!(measured.solver_check_work_micros, None);
        if expect_success {
            assert_eq!(result, BoolPropertyResult::Proved);
        } else {
            assert!(matches!(result, BoolPropertyResult::Error(_)), "{result:?}");
        }
    }

    #[cfg(unix)]
    #[test_case(false; "missing_executable")]
    #[test_case(true; "permission_denied")]
    fn external_quickcheck_spawn_failure_is_error(create_nonexecutable: bool) {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("prove_quickcheck_main");
        if create_nonexecutable {
            std::fs::write(&exe, "#!/bin/sh\nexit 0\n").unwrap();
            std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let entry_file = dir.path().join("property.x");
        let source = "#[quickcheck]\nfn property(x: u1) -> bool { x == x }\n";
        std::fs::write(&entry_file, source).unwrap();
        let backend = ExternalQuickChecks {
            exe,
            entry_file,
            source: source.into(),
            stdlib: None,
            search_paths: Vec::new(),
        };
        let property = QuickCheckProperty {
            id: 0,
            name: "property".into(),
            name_span: None,
            requires_implicit_token: false,
        };
        let phases = PhaseCollector::default();
        let result = backend.prove(&property, &phases).result;
        let measured = phases.snapshot();
        assert!(measured.prepare_work_micros.is_some());
        assert_eq!(measured.external_tool_work_micros, None);
        assert!(matches!(result, BoolPropertyResult::Error(_)), "{result:?}");
    }

    #[test]
    fn fn_equiv_convenience_uses_supported_assertion_semantics() {
        let ir = r#"package test

fn f(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap();
        let tool_dir = tempfile::tempdir().unwrap();
        let prover = ExternalProver::ToolDir(tool_dir.path().to_path_buf());

        let result = prover.prove_ir_fn_equiv(f, f);
        assert!(matches!(
            result,
            EquivResult::Error(msg) if msg.contains("check_ir_equivalence_main not found")
        ));
    }
}
