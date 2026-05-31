// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, fmt, path::Path};

use serde::{Deserialize, Serialize};

pub mod assertion_filter;
pub mod corner_prover;
pub mod enum_in_bound;
pub mod external_prover;
pub mod ir_equiv;
pub mod quickcheck;
pub mod translate;
pub mod types;
pub mod uf;

pub use external_prover::ExternalProver;
pub use quickcheck::discover_quickcheck_tests;

use self::quickcheck::build_assert_label_regex;
use self::types::{
    AssertionSemantics, BoolPropertyResult, EquivParallelism, EquivResult, ProverFn,
    QuickCheckAssertionSemantics, QuickCheckRunResult,
};
use crate::solver::SolverConfig;
use std::str::FromStr;

/// Optional resource limits for solver-backed proofs.
///
/// Limits default to disabled. The in-process Bitwuzla backend currently
/// enforces both fields per satisfiability check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverLimits {
    pub time_limit_per_ms: Option<u64>,
    pub memory_limit_mb: Option<u64>,
}

impl SolverLimits {
    /// Creates limits with a per-satisfiability-check time limit.
    pub fn with_time_limit_per_ms(time_limit_per_ms: u64) -> Self {
        Self {
            time_limit_per_ms: Some(time_limit_per_ms),
            memory_limit_mb: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverChoice {
    #[cfg(feature = "has-easy-smt")]
    Z3Binary,
    #[cfg(feature = "has-easy-smt")]
    BitwuzlaBinary,
    #[cfg(feature = "has-easy-smt")]
    BoolectorBinary,

    /// Use the preferred in-process Bitwuzla backend.
    ///
    /// This intentionally does not fall back to slower solvers or XLS tools.
    Bitwuzla,

    #[cfg(feature = "has-boolector")]
    Boolector,

    /// Use the external XLS tool-chain binaries (configured via
    /// `xlsynth-toolchain.toml`).
    Toolchain,
}

impl fmt::Display for SolverChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::Z3Binary => "z3-binary",
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::BitwuzlaBinary => "bitwuzla-binary",
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::BoolectorBinary => "boolector-binary",
            SolverChoice::Bitwuzla => "bitwuzla",
            #[cfg(feature = "has-boolector")]
            SolverChoice::Boolector => "boolector",
            SolverChoice::Toolchain => "toolchain",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for SolverChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            #[cfg(feature = "has-easy-smt")]
            "z3-binary" => Ok(Self::Z3Binary),
            #[cfg(feature = "has-easy-smt")]
            "bitwuzla-binary" => Ok(Self::BitwuzlaBinary),
            #[cfg(feature = "has-easy-smt")]
            "boolector-binary" => Ok(Self::BoolectorBinary),

            "bitwuzla" => Ok(Self::Bitwuzla),

            #[cfg(feature = "has-boolector")]
            "boolector" => Ok(Self::Boolector),

            "toolchain" => Ok(Self::Toolchain),
            _ => Err(format!("invalid solver: {}", s)),
        }
    }
}

use xlsynth_pir::{ir, ir_parser};

pub trait Prover {
    fn prove_ir_fn_equiv(self: &Self, lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
        let lhs_fn = ProverFn::new(lhs, None);
        let rhs_fn = ProverFn::new(rhs, None);
        self.prove_ir_equiv(
            &lhs_fn,
            &rhs_fn,
            EquivParallelism::SingleThreaded,
            AssertionSemantics::Same,
            None,
            false,
        )
    }
    fn prove_ir_pkg_text_equiv(
        self: &Self,
        lhs_pkg_text: &str,
        rhs_pkg_text: &str,
        top: Option<&str>,
    ) -> EquivResult;
    fn prove_ir_equiv<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        strategy: EquivParallelism,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult;

    fn prove_ir_fn_quickcheck<'a>(self: &Self, ir_fn: &ProverFn<'a>) -> BoolPropertyResult {
        self.prove_ir_quickcheck(ir_fn, QuickCheckAssertionSemantics::Never, None)
    }

    fn prove_ir_quickcheck<'a>(
        self: &Self,
        prover_fn: &ProverFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult;

    fn prove_dslx_quickcheck(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&str>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult>;
}

#[cfg(not(feature = "has-bitwuzla"))]
struct UnavailableProver {
    message: String,
}

#[cfg(not(feature = "has-bitwuzla"))]
impl UnavailableProver {
    fn for_bitwuzla_selection() -> Self {
        Self {
            message: "--solver=bitwuzla requires in-process Bitwuzla support; rebuild with \
                      --features with-bitwuzla-system or --features with-bitwuzla-built, or select \
                      an alternate solver explicitly"
                .to_string(),
        }
    }
}

#[cfg(not(feature = "has-bitwuzla"))]
impl Prover for UnavailableProver {
    fn prove_ir_pkg_text_equiv(
        self: &Self,
        _lhs_pkg_text: &str,
        _rhs_pkg_text: &str,
        _top: Option<&str>,
    ) -> EquivResult {
        EquivResult::Error(self.message.clone())
    }

    fn prove_ir_equiv<'a>(
        self: &Self,
        _lhs: &ProverFn<'a>,
        _rhs: &ProverFn<'a>,
        _strategy: EquivParallelism,
        _assertion_semantics: AssertionSemantics,
        _assert_label_filter: Option<&str>,
        _allow_flatten: bool,
    ) -> EquivResult {
        EquivResult::Error(self.message.clone())
    }

    fn prove_ir_quickcheck<'a>(
        self: &Self,
        _prover_fn: &ProverFn<'a>,
        _assertion_semantics: QuickCheckAssertionSemantics,
        _assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult {
        BoolPropertyResult::Error(self.message.clone())
    }

    fn prove_dslx_quickcheck(
        &self,
        _entry_file: &std::path::Path,
        _dslx_stdlib_path: Option<&std::path::Path>,
        _additional_search_paths: &[&std::path::Path],
        _test_filter: Option<&str>,
        _assertion_semantics: QuickCheckAssertionSemantics,
        _assert_label_filter: Option<&str>,
        _uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult> {
        vec![QuickCheckRunResult {
            name: "solver-selection".to_string(),
            duration: std::time::Duration::ZERO,
            result: BoolPropertyResult::Error(self.message.clone()),
        }]
    }
}

impl<S: SolverConfig> Prover for S {
    fn prove_ir_pkg_text_equiv(
        self: &Self,
        lhs_pkg_text: &str,
        rhs_pkg_text: &str,
        top: Option<&str>,
    ) -> EquivResult {
        let lhs_pkg = ir_parser::Parser::new(lhs_pkg_text)
            .parse_package()
            .expect("Failed to parse LHS IR package");
        let rhs_pkg = ir_parser::Parser::new(rhs_pkg_text)
            .parse_package()
            .expect("Failed to parse RHS IR package");

        let lhs_top = match top {
            Some(name) => lhs_pkg
                .get_fn(name)
                .expect("Top function not found in LHS package"),
            None => lhs_pkg.get_top_fn().expect("No functions in LHS package"),
        };
        let rhs_top = match top {
            Some(name) => rhs_pkg
                .get_fn(name)
                .expect("Top function not found in RHS package"),
            None => rhs_pkg.get_top_fn().expect("No functions in RHS package"),
        };

        let lhs = ProverFn::new(lhs_top, Some(&lhs_pkg));
        let rhs = ProverFn::new(rhs_top, Some(&rhs_pkg));

        ir_equiv::prove_ir_fn_equiv::<S::Solver>(
            self,
            &lhs,
            &rhs,
            AssertionSemantics::Same,
            None,
            false,
        )
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
        if strategy != EquivParallelism::SingleThreaded
            && (!lhs.uf_map.is_empty() || !rhs.uf_map.is_empty())
        {
            return EquivResult::Error(
                "UF mappings are only supported with single-threaded equivalence".to_string(),
            );
        }
        if strategy != EquivParallelism::SingleThreaded && (lhs.has_domains() || rhs.has_domains())
        {
            return EquivResult::Error(format!(
                "Only single-threaded equivalence supports parameter domains",
            ));
        }
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        match strategy {
            EquivParallelism::SingleThreaded => ir_equiv::prove_ir_fn_equiv::<S::Solver>(
                self,
                lhs,
                rhs,
                assertion_semantics,
                assert_label_regex.as_ref(),
                allow_flatten,
            ),
            EquivParallelism::OutputBits => {
                ir_equiv::prove_ir_fn_equiv_output_bits_parallel::<S::Solver>(
                    self,
                    lhs,
                    rhs,
                    assertion_semantics,
                    assert_label_regex.as_ref(),
                    allow_flatten,
                )
            }
            EquivParallelism::InputBitSplit => {
                ir_equiv::prove_ir_fn_equiv_split_input_bit::<S::Solver>(
                    self,
                    lhs,
                    rhs,
                    0,
                    0,
                    assertion_semantics,
                    assert_label_regex.as_ref(),
                    allow_flatten,
                )
            }
        }
    }

    fn prove_ir_quickcheck<'a>(
        self: &Self,
        ir_fn: &ProverFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        quickcheck::prove_ir_quickcheck::<S::Solver>(
            self,
            ir_fn,
            assertion_semantics,
            assert_label_regex.as_ref(),
        )
    }

    fn prove_dslx_quickcheck(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&str>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult> {
        quickcheck::prove_dslx_quickcheck::<S>(
            self,
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
            assertion_semantics,
            assert_label_filter,
            uf_map,
        )
    }
}

pub fn default_prover() -> Box<dyn Prover> {
    default_prover_with_limits(SolverLimits::default())
}

/// Creates the preferred Bitwuzla prover and applies supported resource limits.
pub fn default_prover_with_limits(limits: SolverLimits) -> Box<dyn Prover> {
    prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, limits)
}

pub fn prove_ir_fn_equiv(lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
    default_prover().prove_ir_fn_equiv(lhs, rhs)
}

pub fn prove_ir_pkg_text_equiv(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
) -> EquivResult {
    default_prover().prove_ir_pkg_text_equiv(lhs_pkg_text, rhs_pkg_text, top)
}

pub fn prove_ir_equiv<'a>(
    lhs: &ProverFn<'a>,
    rhs: &ProverFn<'a>,
    strategy: EquivParallelism,
    assertion_semantics: AssertionSemantics,
    assert_label_filter: Option<&str>,
    allow_flatten: bool,
) -> EquivResult {
    default_prover().prove_ir_equiv(
        lhs,
        rhs,
        strategy,
        assertion_semantics,
        assert_label_filter,
        allow_flatten,
    )
}

pub fn prove_ir_fn_quickcheck(ir_fn: &ProverFn<'_>) -> BoolPropertyResult {
    default_prover().prove_ir_fn_quickcheck(ir_fn)
}

pub fn prove_ir_quickcheck(
    prover_fn: &ProverFn<'_>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
) -> BoolPropertyResult {
    default_prover().prove_ir_quickcheck(prover_fn, assertion_semantics, assert_label_filter)
}

pub fn prove_dslx_quickcheck(
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    test_filter: Option<&str>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
    uf_map: &HashMap<String, String>,
) -> Vec<QuickCheckRunResult> {
    default_prover().prove_dslx_quickcheck(
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
        assertion_semantics,
        assert_label_filter,
        uf_map,
    )
}

pub fn prover_for_choice(choice: SolverChoice, tool_path: Option<&Path>) -> Box<dyn Prover> {
    prover_for_choice_with_limits(choice, tool_path, SolverLimits::default())
}

/// Creates the requested prover and applies backend-supported resource limits.
pub fn prover_for_choice_with_limits(
    choice: SolverChoice,
    tool_path: Option<&Path>,
    limits: SolverLimits,
) -> Box<dyn Prover> {
    match choice {
        SolverChoice::Bitwuzla => {
            #[cfg(feature = "has-bitwuzla")]
            {
                let mut options = crate::solver::bitwuzla::BitwuzlaOptions::new();
                apply_bitwuzla_limits(&mut options, limits);
                Box::new(options)
            }
            #[cfg(not(feature = "has-bitwuzla"))]
            {
                let _ = limits;
                Box::new(UnavailableProver::for_bitwuzla_selection())
            }
        }
        #[cfg(feature = "has-boolector")]
        SolverChoice::Boolector => Box::new(crate::solver::boolector::BoolectorConfig::new()),
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::Z3Binary => Box::new(crate::solver::easy_smt::EasySmtConfig::z3()),
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::BitwuzlaBinary => {
            Box::new(crate::solver::easy_smt::EasySmtConfig::bitwuzla())
        }
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::BoolectorBinary => {
            Box::new(crate::solver::easy_smt::EasySmtConfig::boolector())
        }
        SolverChoice::Toolchain => match tool_path {
            Some(path) => {
                let path_buf = path.to_path_buf();
                if path.is_dir() {
                    Box::new(ExternalProver::ToolDir(path_buf))
                } else {
                    Box::new(ExternalProver::ToolExe(path_buf))
                }
            }
            None => Box::new(ExternalProver::Toolchain),
        },
    }
}

#[cfg(feature = "has-bitwuzla")]
fn apply_bitwuzla_limits(
    options: &mut crate::solver::bitwuzla::BitwuzlaOptions,
    limits: SolverLimits,
) {
    if let Some(time_limit_per_ms) = limits.time_limit_per_ms {
        options.set_time_limit_per(time_limit_per_ms);
    }
    if let Some(memory_limit_mb) = limits.memory_limit_mb {
        options.set_memory_limit(memory_limit_mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitwuzla_choice_roundtrips_through_cli_spelling() {
        assert_eq!(SolverChoice::Bitwuzla.to_string(), "bitwuzla");
        assert_eq!(
            "bitwuzla".parse::<SolverChoice>().unwrap(),
            SolverChoice::Bitwuzla
        );
    }

    #[cfg(not(feature = "has-bitwuzla"))]
    #[test]
    fn default_prover_reports_missing_bitwuzla_feature() {
        let result = default_prover().prove_ir_pkg_text_equiv("", "", None);
        assert!(matches!(
            result,
            EquivResult::Error(msg) if msg.contains("--solver=bitwuzla requires")
        ));
    }
}
