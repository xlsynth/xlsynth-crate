// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::Prover;
use super::quickcheck::load_quickcheck_context;
use super::types::{
    AssertionSemantics, BoolPropertyResult, EquivParallelism, EquivResult, ProverFn,
    QuickCheckAssertionSemantics, QuickCheckRunResult,
};
use crate::toolchain_shim::run_prove_quickcheck_main;
use regex::escape;
use xlsynth_pir::prove_equiv_via_toolchain::{self, ToolchainEquivResult};

const MAX_TOOLCHAIN_MESSAGE: usize = 512;

pub enum ExternalProver {
    ToolExe(PathBuf),
    ToolDir(PathBuf),
    Toolchain,
}

impl Prover for ExternalProver {
    fn prove_ir_pkg_text_equiv(
        self: &Self,
        lhs_pkg_text: &str,
        rhs_pkg_text: &str,
        top: Option<&str>,
    ) -> EquivResult {
        match self {
            ExternalProver::ToolExe(path) => {
                prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_exe(
                    lhs_pkg_text,
                    rhs_pkg_text,
                    top,
                    path,
                )
                .into()
            }
            ExternalProver::ToolDir(path) => {
                prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_dir(
                    lhs_pkg_text,
                    rhs_pkg_text,
                    top,
                    path,
                )
                .into()
            }
            ExternalProver::Toolchain => match std::env::var("XLSYNTH_TOOLS") {
                Ok(dir) => ExternalProver::ToolDir(PathBuf::from(dir)).prove_ir_pkg_text_equiv(
                    lhs_pkg_text,
                    rhs_pkg_text,
                    top,
                ),
                Err(_) => EquivResult::Error(
                    "XLSYNTH_TOOLS is not set; cannot run toolchain equivalence".to_string(),
                ),
            },
        }
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
                if (lhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false))
                    || (rhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false))
                {
                    println!(
                        "Warning: External provers do not support domains for arguments. Enums will be treated as possibly out of bounds."
                    );
                }
                if assertion_semantics != AssertionSemantics::Same {
                    return EquivResult::Error(
                        "External provers do not support assertion semantics".to_string(),
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
                let lhs_pkg = match lhs.pkg_ref {
                    Some(pkg) => pkg.to_string(),
                    None => format!("package lhs\n\ntop {}\n", lhs.fn_ref.to_string()),
                };
                let rhs_pkg = match rhs.pkg_ref {
                    Some(pkg) => pkg.to_string(),
                    None => format!("package rhs\n\ntop {}\n", rhs.fn_ref.to_string()),
                };
                let (lhs_unified, rhs_unified, unified_top) =
                    unify_toolchain_tops(&lhs_pkg, &rhs_pkg, &lhs.fn_ref.name, &rhs.fn_ref.name);

                self.prove_ir_pkg_text_equiv(&lhs_unified, &rhs_unified, Some(&unified_top))
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
            return BoolPropertyResult::ToolchainDisproved(
                "External provers do not support assertion label filters".to_string(),
            );
        }
        BoolPropertyResult::ToolchainDisproved(
            "External provers do not support IR-level quickcheck; use prove_dslx_quickcheck"
                .to_string(),
        )
    }

    fn prove_dslx_quickcheck(
        &self,
        entry_file: &Path,
        dslx_stdlib_path: Option<&Path>,
        additional_search_paths: &[&Path],
        test_filter: Option<&str>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult> {
        let (_, quickchecks) = load_quickcheck_context(
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
        );
        if quickchecks.is_empty() {
            return Vec::new();
        }

        if assert_label_filter.is_some() {
            return quickchecks
                .into_iter()
                .map(|(name, _)| QuickCheckRunResult {
                    name,
                    duration: std::time::Duration::default(),
                    result: BoolPropertyResult::ToolchainDisproved(
                        "External quickcheck does not support assertion label filters".to_string(),
                    ),
                })
                .collect();
        }
        if !uf_map.is_empty() {
            return quickchecks
                .into_iter()
                .map(|(name, _)| QuickCheckRunResult {
                    name,
                    duration: std::time::Duration::default(),
                    result: BoolPropertyResult::ToolchainDisproved(
                        "External quickcheck does not support uninterpreted functions".to_string(),
                    ),
                })
                .collect();
        }

        let mut results = Vec::with_capacity(quickchecks.len());
        for (quickcheck_name, _) in quickchecks {
            let start_time = std::time::Instant::now();
            let filter = format!("^{}$", escape(quickcheck_name.as_str()));
            let limit = |msg: &str| msg.chars().take(MAX_TOOLCHAIN_MESSAGE).collect::<String>();
            let run_with_exe = |exe: &Path| match run_prove_quickcheck_main(
                exe,
                entry_file,
                dslx_stdlib_path,
                additional_search_paths,
                filter.as_str(),
            ) {
                Ok(_) => BoolPropertyResult::Proved,
                Err(msg) => BoolPropertyResult::ToolchainDisproved(limit(&msg)),
            };

            let result = match self {
                ExternalProver::ToolExe(path) => run_with_exe(path),
                ExternalProver::ToolDir(dir) => {
                    let exe = dir.join("prove_quickcheck_main");
                    if !exe.exists() {
                        BoolPropertyResult::ToolchainDisproved(format!(
                            "prove_quickcheck_main not found in {}",
                            dir.display()
                        ))
                    } else {
                        run_with_exe(&exe)
                    }
                }
                ExternalProver::Toolchain => match std::env::var("XLSYNTH_TOOLS") {
                    Ok(dir) => {
                        let dir = PathBuf::from(dir);
                        let exe = dir.join("prove_quickcheck_main");
                        if !exe.exists() {
                            BoolPropertyResult::ToolchainDisproved(format!(
                                "prove_quickcheck_main not found in {}",
                                dir.display()
                            ))
                        } else {
                            run_with_exe(&exe)
                        }
                    }
                    Err(_) => BoolPropertyResult::ToolchainDisproved(
                        "XLSYNTH_TOOLS is not set; cannot run toolchain quickcheck".to_string(),
                    ),
                },
            };

            results.push(QuickCheckRunResult {
                name: quickcheck_name,
                duration: start_time.elapsed(),
                result,
            });
        }

        let _ = assertion_semantics;

        results
    }
}

impl From<ToolchainEquivResult> for EquivResult {
    fn from(result: ToolchainEquivResult) -> Self {
        match result {
            ToolchainEquivResult::Proved => EquivResult::Proved,
            ToolchainEquivResult::Disproved(msg) => EquivResult::ToolchainDisproved(msg),
            ToolchainEquivResult::Error(msg) => EquivResult::Error(msg),
        }
    }
}

fn unify_toolchain_tops<'a>(
    lhs_ir: &'a str,
    rhs_ir: &'a str,
    lhs_top: &str,
    rhs_top: &str,
) -> (Cow<'a, str>, Cow<'a, str>, String) {
    if lhs_top == rhs_top {
        return (
            Cow::Borrowed(lhs_ir),
            Cow::Borrowed(rhs_ir),
            lhs_top.to_string(),
        );
    }
    let unified = lhs_top.to_string();
    let rhs_rewritten = rhs_ir.replace(rhs_top, &unified);
    (Cow::Borrowed(lhs_ir), Cow::Owned(rhs_rewritten), unified)
}
