// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::PathBuf};

use crate::equiv::solver_interface::SolverConfig;
use crate::equiv::types::{
    AssertionSemantics, BoolPropertyResult, EquivResult, EquivSide, IrFn,
    QuickCheckAssertionSemantics, UfSignature,
};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{ir, ir_parser};

pub trait Prover {
    fn prove_ir_fn_equiv(self: &Self, lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
        self.prove_ir_fn_equiv_full(
            &EquivSide {
                ir_fn: &IrFn {
                    fn_ref: lhs,
                    pkg_ref: None,
                    fixed_implicit_activation: false,
                },
                domains: None,
                uf_map: HashMap::new(),
            },
            &EquivSide {
                ir_fn: &IrFn {
                    fn_ref: rhs,
                    pkg_ref: None,
                    fixed_implicit_activation: false,
                },
                domains: None,
                uf_map: HashMap::new(),
            },
            AssertionSemantics::Same,
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
    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &EquivSide<'a>,
        rhs: &EquivSide<'a>,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult;
    fn prove_ir_fn_equiv_output_bits_parallel(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
    ) -> EquivResult;
    fn prove_ir_fn_equiv_split_input_bit(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        start_input: usize,
        start_bit: usize,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
    ) -> EquivResult;

    // --- QuickCheck-style proving: f() always returns bits[1]:1 ---
    fn prove_ir_fn_always_true<'a>(
        self: &Self,
        ir_fn: &IrFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
    ) -> BoolPropertyResult {
        self.prove_ir_fn_always_true_with_ufs(
            ir_fn,
            assertion_semantics,
            &HashMap::new(),
            &HashMap::new(),
        )
    }

    fn prove_ir_fn_always_true_with_ufs<'a>(
        self: &Self,
        ir_fn: &IrFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        uf_map: &HashMap<String, String>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult;

    /// Prove a DSLX quickcheck function (by name) directly.
    fn prove_dslx_quickcheck(
        &self,
        entry_file: &std::path::Path,
        quickcheck_name: &str,
        assertion_semantics: QuickCheckAssertionSemantics,
    ) -> BoolPropertyResult {
        self.prove_dslx_quickcheck_with_ufs(
            entry_file,
            quickcheck_name,
            assertion_semantics,
            &HashMap::new(),
            &HashMap::new(),
        )
    }

    fn prove_dslx_quickcheck_with_ufs(
        &self,
        entry_file: &std::path::Path,
        quickcheck_name: &str,
        assertion_semantics: QuickCheckAssertionSemantics,
        uf_map: &HashMap<String, String>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult;
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
            None => lhs_pkg.get_top().expect("No functions in LHS package"),
        };
        let rhs_top = match top {
            Some(name) => rhs_pkg
                .get_fn(name)
                .expect("Top function not found in RHS package"),
            None => rhs_pkg.get_top().expect("No functions in RHS package"),
        };

        let lhs = EquivSide {
            ir_fn: &IrFn {
                fn_ref: lhs_top,
                pkg_ref: Some(&lhs_pkg),
                fixed_implicit_activation: false,
            },
            domains: None,
            uf_map: HashMap::new(),
        };
        let rhs = EquivSide {
            ir_fn: &IrFn {
                fn_ref: rhs_top,
                pkg_ref: Some(&rhs_pkg),
                fixed_implicit_activation: false,
            },
            domains: None,
            uf_map: HashMap::new(),
        };

        crate::equiv::prove_equiv::prove_ir_fn_equiv_full::<S::Solver>(
            self,
            &lhs,
            &rhs,
            AssertionSemantics::Same,
            false,
            &HashMap::new(),
        )
    }

    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &EquivSide<'a>,
        rhs: &EquivSide<'a>,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult {
        crate::equiv::prove_equiv::prove_ir_fn_equiv_full::<S::Solver>(
            self,
            lhs,
            rhs,
            assertion_semantics,
            allow_flatten,
            uf_signatures,
        )
    }

    fn prove_ir_fn_equiv_output_bits_parallel(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
    ) -> EquivResult {
        crate::equiv::prove_equiv::prove_ir_fn_equiv_output_bits_parallel::<S::Solver>(
            self,
            lhs,
            rhs,
            assertion_semantics,
            allow_flatten,
        )
    }

    fn prove_ir_fn_equiv_split_input_bit(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        start_input: usize,
        start_bit: usize,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
    ) -> EquivResult {
        crate::equiv::prove_equiv::prove_ir_fn_equiv_split_input_bit::<S::Solver>(
            self,
            lhs,
            rhs,
            start_input,
            start_bit,
            assertion_semantics,
            allow_flatten,
        )
    }

    fn prove_ir_fn_always_true_with_ufs<'a>(
        self: &Self,
        ir_fn: &IrFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        uf_map: &HashMap<String, String>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        crate::equiv::prove_quickcheck::prove_ir_fn_always_true_with_ufs::<S::Solver>(
            self,
            ir_fn,
            assertion_semantics,
            uf_map,
            uf_signatures,
        )
    }

    fn prove_dslx_quickcheck_with_ufs(
        &self,
        entry_file: &std::path::Path,
        quickcheck_name: &str,
        assertion_semantics: QuickCheckAssertionSemantics,
        uf_map: &HashMap<String, String>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        // Convert DSLX module to IR text using the runtime API with default options.
        let dslx_contents = std::fs::read_to_string(entry_file)
            .expect("Failed to read DSLX input file for quickcheck");
        let options = xlsynth::DslxConvertOptions {
            dslx_stdlib_path: None,
            additional_search_paths: Vec::new(),
            enable_warnings: None,
            disable_warnings: None,
        };
        let ir_text = xlsynth::convert_dslx_to_ir_text(&dslx_contents, entry_file, &options)
            .expect("DSLX->IR conversion failed for quickcheck")
            .ir;

        // Parse IR and locate the quickcheck function, accounting for calling
        // convention.
        let pkg = Parser::new(&ir_text)
            .parse_package()
            .expect("Failed to parse IR package for quickcheck");
        let module_name = entry_file
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("valid module name");
        let mangled_itok = xlsynth::mangle_dslx_name_with_calling_convention(
            module_name,
            quickcheck_name,
            xlsynth::DslxCallingConvention::ImplicitToken,
        )
        .expect("mangle itok");
        let mangled_normal = xlsynth::mangle_dslx_name_with_calling_convention(
            module_name,
            quickcheck_name,
            xlsynth::DslxCallingConvention::Typical,
        )
        .expect("mangle normal");

        let (fn_ref, fixed_implicit_activation) = if let Some(f) = pkg.get_fn(&mangled_itok) {
            (f, true)
        } else if let Some(f) = pkg.get_fn(&mangled_normal) {
            (f, false)
        } else {
            panic!(
                "quickcheck function '{}' not found (module '{}')",
                quickcheck_name, module_name
            );
        };

        let ir_fn = IrFn {
            fn_ref,
            pkg_ref: Some(&pkg),
            fixed_implicit_activation,
        };

        crate::equiv::prove_quickcheck::prove_ir_fn_always_true_with_ufs::<S::Solver>(
            self,
            &ir_fn,
            assertion_semantics,
            uf_map,
            uf_signatures,
        )
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
                crate::equiv::prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_exe(
                    lhs_pkg_text,
                    rhs_pkg_text,
                    top,
                    path,
                )
            }
            ExternalProver::ToolDir(path) => {
                crate::equiv::prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_dir(
                    lhs_pkg_text,
                    rhs_pkg_text,
                    top,
                    path,
                )
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

    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &EquivSide<'a>,
        rhs: &EquivSide<'a>,
        assertion_semantics: AssertionSemantics,
        allow_flatten: bool,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult {
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
        if allow_flatten {
            return EquivResult::Error("External provers do not support flattening".to_string());
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
        ) -> (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>, String) {
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

    fn prove_ir_fn_equiv_output_bits_parallel(
        self: &Self,
        _lhs: &IrFn,
        _rhs: &IrFn,
        _assertion_semantics: AssertionSemantics,
        _allow_flatten: bool,
    ) -> EquivResult {
        EquivResult::Error(
            "External provers do not support output-bits parallel strategy".to_string(),
        )
    }

    fn prove_ir_fn_equiv_split_input_bit(
        self: &Self,
        _lhs: &IrFn,
        _rhs: &IrFn,
        _start_input: usize,
        _start_bit: usize,
        _assertion_semantics: AssertionSemantics,
        _allow_flatten: bool,
    ) -> EquivResult {
        EquivResult::Error("External provers do not support input-bit split strategy".to_string())
    }

    fn prove_ir_fn_always_true_with_ufs<'a>(
        self: &Self,
        _ir_fn: &IrFn<'a>,
        _assertion_semantics: QuickCheckAssertionSemantics,
        _uf_map: &HashMap<String, String>,
        _uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        BoolPropertyResult::ToolchainDisproved(
            "External provers do not support IR-level quickcheck; use prove_dslx_quickcheck"
                .to_string(),
        )
    }

    fn prove_dslx_quickcheck_with_ufs(
        &self,
        entry_file: &std::path::Path,
        quickcheck_name: &str,
        _assertion_semantics: QuickCheckAssertionSemantics,
        uf_map: &HashMap<String, String>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        if !uf_map.is_empty() || !uf_signatures.is_empty() {
            return BoolPropertyResult::ToolchainDisproved(
                "External quickcheck does not support uninterpreted functions".to_string(),
            );
        }
        match self {
            ExternalProver::ToolExe(path) => {
                crate::equiv::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_with_tool_exe(
                    entry_file,
                    quickcheck_name,
                    path,
                )
            }
            ExternalProver::ToolDir(path) => {
                crate::equiv::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_with_tool_dir(
                    entry_file,
                    quickcheck_name,
                    path,
                )
            }
            ExternalProver::Toolchain => {
                crate::equiv::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_via_toolchain(
                    entry_file,
                    quickcheck_name,
                )
            }
        }
    }
}

pub fn auto_selected_prover() -> Box<dyn Prover> {
    #[cfg(feature = "has-bitwuzla")]
    {
        use crate::equiv::bitwuzla_backend::BitwuzlaOptions;
        return Box::new(BitwuzlaOptions::new());
    }
    #[cfg(all(feature = "has-boolector", not(feature = "has-bitwuzla")))]
    {
        use crate::equiv::boolector_backend::BoolectorConfig;
        return Box::new(BoolectorConfig::new());
    }
    #[cfg(all(
        feature = "has-easy-smt",
        not(feature = "has-bitwuzla"),
        not(feature = "has-boolector")
    ))]
    {
        use crate::equiv::easy_smt_backend::{EasySmtConfig, EasySmtSolver};
        use crate::equiv::solver_interface::{Response, Solver};

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

pub fn prove_ir_fn_equiv_full(
    lhs: &EquivSide<'_>,
    rhs: &EquivSide<'_>,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
    uf_signatures: &HashMap<String, UfSignature>,
) -> EquivResult {
    auto_selected_prover().prove_ir_fn_equiv_full(
        lhs,
        rhs,
        assertion_semantics,
        allow_flatten,
        uf_signatures,
    )
}

pub fn prove_ir_pkg_text_equiv(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
) -> EquivResult {
    auto_selected_prover().prove_ir_pkg_text_equiv(lhs_pkg_text, rhs_pkg_text, top)
}

pub fn prove_ir_fn_always_true(
    ir_fn: &IrFn,
    assertion_semantics: QuickCheckAssertionSemantics,
) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_fn_always_true(ir_fn, assertion_semantics)
}

pub fn prove_ir_fn_always_true_with_ufs(
    ir_fn: &IrFn,
    assertion_semantics: QuickCheckAssertionSemantics,
    uf_map: &HashMap<String, String>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> BoolPropertyResult {
    auto_selected_prover().prove_ir_fn_always_true_with_ufs(
        ir_fn,
        assertion_semantics,
        uf_map,
        uf_signatures,
    )
}

pub fn prove_dslx_quickcheck(
    entry_file: &std::path::Path,
    quickcheck_name: &str,
    assertion_semantics: QuickCheckAssertionSemantics,
) -> BoolPropertyResult {
    auto_selected_prover().prove_dslx_quickcheck(entry_file, quickcheck_name, assertion_semantics)
}

pub fn prove_dslx_quickcheck_with_ufs(
    entry_file: &std::path::Path,
    quickcheck_name: &str,
    assertion_semantics: QuickCheckAssertionSemantics,
    uf_map: &HashMap<String, String>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> BoolPropertyResult {
    auto_selected_prover().prove_dslx_quickcheck_with_ufs(
        entry_file,
        quickcheck_name,
        assertion_semantics,
        uf_map,
        uf_signatures,
    )
}
