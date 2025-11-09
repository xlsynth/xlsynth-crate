// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::types::{
    AssertionSemantics, BoolPropertyResult, EquivParallelism, EquivResult, IrFn, ProverFn,
    QuickCheckAssertionSemantics, QuickCheckRunResult, UfSignature,
};
use crate::{prove_quickcheck::build_assert_label_regex, solver_interface::SolverConfig};
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

use xlsynth_pir::prove_equiv_via_toolchain::{self, ToolchainEquivResult};
use xlsynth_pir::{ir, ir_parser};

pub trait Prover {
    fn prove_ir_fn_equiv(self: &Self, lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
        self.prove_ir_equiv(
            &ProverFn {
                ir_fn: &IrFn {
                    fn_ref: lhs,
                    pkg_ref: None,
                    fixed_implicit_activation: false,
                },
                domains: None,
                uf_map: HashMap::new(),
            },
            &ProverFn {
                ir_fn: &IrFn {
                    fn_ref: rhs,
                    pkg_ref: None,
                    fixed_implicit_activation: false,
                },
                domains: None,
                uf_map: HashMap::new(),
            },
            EquivParallelism::SingleThreaded,
            AssertionSemantics::Same,
            None,
            false,
            &HashMap::new(),
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
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult;

    fn prove_ir_fn_quickcheck<'a>(self: &Self, ir_fn: &IrFn<'a>) -> BoolPropertyResult {
        let prover_fn = ProverFn {
            ir_fn,
            domains: None,
            uf_map: HashMap::new(),
        };
        self.prove_ir_quickcheck(
            &prover_fn,
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
        )
    }

    fn prove_ir_quickcheck<'a>(
        self: &Self,
        prover_fn: &ProverFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_signatures: &HashMap<String, UfSignature>,
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

        let lhs = ProverFn {
            ir_fn: &IrFn {
                fn_ref: lhs_top,
                pkg_ref: Some(&lhs_pkg),
                fixed_implicit_activation: false,
            },
            domains: None,
            uf_map: HashMap::new(),
        };
        let rhs = ProverFn {
            ir_fn: &IrFn {
                fn_ref: rhs_top,
                pkg_ref: Some(&rhs_pkg),
                fixed_implicit_activation: false,
            },
            domains: None,
            uf_map: HashMap::new(),
        };

        crate::prove_equiv::prove_ir_fn_equiv_full::<S::Solver>(
            self,
            &lhs,
            &rhs,
            AssertionSemantics::Same,
            None,
            false,
            &HashMap::new(),
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
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        match strategy {
            EquivParallelism::SingleThreaded => {
                crate::prove_equiv::prove_ir_fn_equiv_full::<S::Solver>(
                    self,
                    lhs,
                    rhs,
                    assertion_semantics,
                    assert_label_regex.as_ref(),
                    allow_flatten,
                    uf_signatures,
                )
            }
            EquivParallelism::OutputBits => {
                if !uf_signatures.is_empty() {
                    return EquivResult::Error(
                        "Output-bits strategy does not support UF signatures".to_string(),
                    );
                }
                if lhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false)
                    || rhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false)
                {
                    return EquivResult::Error(
                        "Output-bits strategy does not support parameter domains".to_string(),
                    );
                }
                crate::prove_equiv::prove_ir_fn_equiv_output_bits_parallel::<S::Solver>(
                    self,
                    lhs.ir_fn,
                    rhs.ir_fn,
                    assertion_semantics,
                    assert_label_regex.as_ref(),
                    allow_flatten,
                )
            }
            EquivParallelism::InputBitSplit => {
                if !uf_signatures.is_empty() {
                    return EquivResult::Error(
                        "Input-bit-split strategy does not support UF signatures".to_string(),
                    );
                }
                if lhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false)
                    || rhs.domains.as_ref().map(|d| !d.is_empty()).unwrap_or(false)
                {
                    return EquivResult::Error(
                        "Input-bit-split strategy does not support parameter domains".to_string(),
                    );
                }
                crate::prove_equiv::prove_ir_fn_equiv_split_input_bit::<S::Solver>(
                    self,
                    lhs.ir_fn,
                    rhs.ir_fn,
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
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        crate::prove_quickcheck::prove_ir_quickcheck::<S::Solver>(
            self,
            ir_fn,
            assertion_semantics,
            assert_label_regex.as_ref(),
            uf_signatures,
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
        crate::prove_quickcheck::prove_dslx_quickcheck::<S>(
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

impl From<ToolchainEquivResult> for EquivResult {
    fn from(result: ToolchainEquivResult) -> Self {
        match result {
            ToolchainEquivResult::Proved => EquivResult::Proved,
            ToolchainEquivResult::Disproved(msg) => EquivResult::ToolchainDisproved(msg),
            ToolchainEquivResult::Error(msg) => EquivResult::Error(msg),
        }
    }
}

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
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult {
        match strategy {
            EquivParallelism::SingleThreaded => {
                if lhs.ir_fn.fixed_implicit_activation || rhs.ir_fn.fixed_implicit_activation {
                    return EquivResult::Error(
                        "External provers do not support fixed implicit activation".to_string(),
                    );
                }
                if (lhs.domains.is_some() && lhs.domains.as_ref().unwrap().len() != 0)
                    || (rhs.domains.is_some() && rhs.domains.as_ref().unwrap().len() != 0)
                {
                    println!(
                        "Warning: External provers do not support domains for arguments. Enums will be treated as possibly out of bounds."
                    );
                }
                if lhs.uf_map.len() != 0 || rhs.uf_map.len() != 0 {
                    return EquivResult::Error("External provers do not support UFs".to_string());
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
                if !uf_signatures.is_empty() {
                    return EquivResult::Error("External provers do not support UFs".to_string());
                }
                let lhs_pkg = match lhs.ir_fn.pkg_ref {
                    Some(pkg) => pkg.to_string(),
                    None => format!("package lhs\n\ntop {}\n", lhs.ir_fn.fn_ref.to_string()),
                };
                let rhs_pkg = match rhs.ir_fn.pkg_ref {
                    Some(pkg) => pkg.to_string(),
                    None => format!("package rhs\n\ntop {}\n", rhs.ir_fn.fn_ref.to_string()),
                };
                fn unify_toolchain_tops<'a>(
                    lhs_ir: &'a str,
                    rhs_ir: &'a str,
                    lhs_top: &str,
                    rhs_top: &str,
                ) -> (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>, String)
                {
                    if lhs_top == rhs_top {
                        return (
                            std::borrow::Cow::Borrowed(lhs_ir),
                            std::borrow::Cow::Borrowed(rhs_ir),
                            lhs_top.to_string(),
                        );
                    }
                    let unified = lhs_top.to_string();
                    let rhs_rewritten = rhs_ir.replace(rhs_top, &unified);
                    (
                        std::borrow::Cow::Borrowed(lhs_ir),
                        std::borrow::Cow::Owned(rhs_rewritten),
                        unified,
                    )
                }

                let (lhs_unified, rhs_unified, unified_top) = unify_toolchain_tops(
                    &lhs_pkg,
                    &rhs_pkg,
                    &lhs.ir_fn.fn_ref.name,
                    &rhs.ir_fn.fn_ref.name,
                );

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
        _uf_signatures: &HashMap<String, UfSignature>,
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
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&str>,
        _assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult> {
        crate::toolchain::prove_dslx_quickcheck_full_via_toolchain(
            self,
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
            _assertion_semantics,
            assert_label_filter,
            uf_map,
        )
    }
}

pub fn auto_selected_prover() -> Box<dyn Prover> {
    #[cfg(feature = "has-bitwuzla")]
    {
        use crate::bitwuzla_backend::BitwuzlaOptions;
        return Box::new(BitwuzlaOptions::new());
    }
    #[cfg(all(feature = "has-boolector", not(feature = "has-bitwuzla")))]
    {
        use crate::boolector_backend::BoolectorConfig;
        return Box::new(BoolectorConfig::new());
    }
    #[cfg(all(
        feature = "has-easy-smt",
        not(feature = "has-bitwuzla"),
        not(feature = "has-boolector")
    ))]
    {
        use crate::easy_smt_backend::{EasySmtConfig, EasySmtSolver};
        use crate::solver_interface::{Response, Solver};

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
    uf_signatures: &HashMap<String, UfSignature>,
) -> EquivResult {
    auto_selected_prover().prove_ir_equiv(
        lhs,
        rhs,
        strategy,
        assertion_semantics,
        assert_label_filter,
        allow_flatten,
        uf_signatures,
    )
}

pub fn prove_ir_fn_quickcheck(ir_fn: &IrFn<'_>) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_fn_quickcheck(ir_fn)
}

pub fn prove_ir_quickcheck(
    prover_fn: &ProverFn<'_>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_quickcheck(
        prover_fn,
        assertion_semantics,
        assert_label_filter,
        uf_signatures,
    )
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
        SolverChoice::Bitwuzla => Box::new(crate::bitwuzla_backend::BitwuzlaOptions::new()),
        #[cfg(feature = "has-boolector")]
        SolverChoice::Boolector => Box::new(crate::boolector_backend::BoolectorConfig::new()),
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::Z3Binary => Box::new(crate::easy_smt_backend::EasySmtConfig::z3()),
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::BitwuzlaBinary => {
            Box::new(crate::easy_smt_backend::EasySmtConfig::bitwuzla())
        }
        #[cfg(feature = "has-easy-smt")]
        SolverChoice::BoolectorBinary => {
            Box::new(crate::easy_smt_backend::EasySmtConfig::boolector())
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
