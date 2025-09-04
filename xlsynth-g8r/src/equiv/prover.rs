// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::PathBuf};

use crate::{
    equiv::{
        prove_equiv::{AssertionSemantics, EquivResult, EquivSide, IrFn, UfSignature},
        solver_interface::SolverConfig,
    },
    xls_ir::{ir, ir_parser::Parser},
};

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
}

impl<S: SolverConfig> Prover for S {
    fn prove_ir_pkg_text_equiv(
        self: &Self,
        lhs_pkg_text: &str,
        rhs_pkg_text: &str,
        top: Option<&str>,
    ) -> EquivResult {
        let lhs_pkg = Parser::new(lhs_pkg_text)
            .parse_package()
            .expect("Failed to parse LHS IR package");
        let rhs_pkg = Parser::new(rhs_pkg_text)
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
        if lhs.domains.is_some() || rhs.domains.is_some() {
            return EquivResult::Error("External provers do not support domains".to_string());
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
            None => format!("package lhs\n\ntop {}\n", lhs.ir_fn.fn_ref.name),
        };
        let rhs_pkg = match rhs.ir_fn.pkg_ref {
            Some(pkg) => pkg.to_string(),
            None => format!("package rhs\n\ntop {}\n", rhs.ir_fn.fn_ref.name),
        };
        if lhs.ir_fn.fn_ref.name != rhs.ir_fn.fn_ref.name {
            return EquivResult::Error(
                "External provers do not support different function names".to_string(),
            );
        }
        self.prove_ir_pkg_text_equiv(&lhs_pkg, &rhs_pkg, Some(&lhs.ir_fn.fn_ref.name))
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
