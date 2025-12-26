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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverChoice {
    /// Let the library select an appropriate prover based on available
    /// features.
    Auto,
    #[cfg(feature = "has-easy-smt")]
    Z3Binary,
    #[cfg(feature = "has-easy-smt")]
    BitwuzlaBinary,
    #[cfg(feature = "has-easy-smt")]
    BoolectorBinary,

    #[cfg(feature = "has-bitwuzla")]
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
            SolverChoice::Auto => "auto",
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::Z3Binary => "z3-binary",
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::BitwuzlaBinary => "bitwuzla-binary",
            #[cfg(feature = "has-easy-smt")]
            SolverChoice::BoolectorBinary => "boolector-binary",
            #[cfg(feature = "has-bitwuzla")]
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
            "auto" => Ok(Self::Auto),
            #[cfg(feature = "has-easy-smt")]
            "z3-binary" => Ok(Self::Z3Binary),
            #[cfg(feature = "has-easy-smt")]
            "bitwuzla-binary" => Ok(Self::BitwuzlaBinary),
            #[cfg(feature = "has-easy-smt")]
            "boolector-binary" => Ok(Self::BoolectorBinary),

            #[cfg(feature = "has-bitwuzla")]
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

pub fn auto_selected_prover() -> Box<dyn Prover> {
    #[cfg(feature = "has-bitwuzla")]
    {
        use crate::solver::bitwuzla::BitwuzlaOptions;
        return Box::new(BitwuzlaOptions::new());
    }
    #[cfg(all(feature = "has-boolector", not(feature = "has-bitwuzla")))]
    {
        use crate::solver::boolector::BoolectorConfig;
        return Box::new(BoolectorConfig::new());
    }
    #[cfg(all(
        feature = "has-easy-smt",
        not(feature = "has-bitwuzla"),
        not(feature = "has-boolector")
    ))]
    {
        use crate::solver::{
            Response, Solver,
            easy_smt::{EasySmtConfig, EasySmtSolver},
        };

        fn is_usable(config: &EasySmtConfig) -> bool {
            match EasySmtSolver::new(config) {
                Ok(mut solver) => {
                    // Minimal sanity: declare a symbol, assert a trivial constraint, check SAT.
                    if solver.declare("probe_x", 1).is_err() {
                        return false;
                    }
                    let one = solver.one(1);
                    let a = solver.numerical(1, 1);
                    let eq = solver.eq(&one, &a);
                    if solver.assert(&eq).is_err() {
                        return false;
                    }
                    match solver.check() {
                        Ok(Response::Sat) => true,
                        _ => false,
                    }
                }
                Err(_) => false,
            }
        }

        let candidates = [
            EasySmtConfig::z3(),
            EasySmtConfig::boolector(),
            EasySmtConfig::bitwuzla(),
        ];

        for cfg in candidates.into_iter() {
            if is_usable(&cfg) {
                return Box::new(cfg);
            }
        }
    }
    #[cfg(all(not(feature = "has-bitwuzla"), not(feature = "has-boolector")))]
    Box::new(ExternalProver::Toolchain)
}

pub fn prove_ir_fn_equiv(lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
    auto_selected_prover().prove_ir_fn_equiv(lhs, rhs)
}

pub fn prove_ir_pkg_text_equiv(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
) -> EquivResult {
    auto_selected_prover().prove_ir_pkg_text_equiv(lhs_pkg_text, rhs_pkg_text, top)
}

pub fn prove_ir_equiv<'a>(
    lhs: &ProverFn<'a>,
    rhs: &ProverFn<'a>,
    strategy: EquivParallelism,
    assertion_semantics: AssertionSemantics,
    assert_label_filter: Option<&str>,
    allow_flatten: bool,
) -> EquivResult {
    auto_selected_prover().prove_ir_equiv(
        lhs,
        rhs,
        strategy,
        assertion_semantics,
        assert_label_filter,
        allow_flatten,
    )
}

pub fn prove_ir_fn_quickcheck(ir_fn: &ProverFn<'_>) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_fn_quickcheck(ir_fn)
}

pub fn prove_ir_quickcheck(
    prover_fn: &ProverFn<'_>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_quickcheck(prover_fn, assertion_semantics, assert_label_filter)
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
    auto_selected_prover().prove_dslx_quickcheck(
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
    match choice {
        SolverChoice::Auto => auto_selected_prover(),
        #[cfg(feature = "has-bitwuzla")]
        SolverChoice::Bitwuzla => Box::new(crate::solver::bitwuzla::BitwuzlaOptions::new()),
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
