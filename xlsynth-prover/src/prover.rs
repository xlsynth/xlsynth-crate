// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::PathBuf};

use crate::solver_interface::SolverConfig;
use crate::types::{
    AssertionSemantics, BoolPropertyResult, EquivResult, IrFn, ProverFn,
    QuickCheckAssertionSemantics, QuickCheckRunResult, UfSignature,
};
use regex::Regex;
use xlsynth::dslx::{ImportData, MatchableModuleMember};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::prove_equiv_via_toolchain::{self, ToolchainEquivResult};
use xlsynth_pir::{ir, ir_parser};

fn build_assert_label_regex(filter: Option<&str>) -> Option<Regex> {
    match filter {
        None => None,
        Some(pattern) => {
            Some(Regex::new(pattern).expect("invalid regular expression in assert label filter"))
        }
    }
}

fn load_quickcheck_context(
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    test_filter: Option<&Regex>,
) -> (String, Vec<(String, bool)>) {
    let dslx_contents = std::fs::read_to_string(entry_file)
        .expect("Failed to read DSLX input file for quickcheck discovery");
    let module_name = entry_file
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("valid module name");

    let mut import_data = ImportData::new(dslx_stdlib_path, additional_search_paths);
    let type_checked = xlsynth::dslx::parse_and_typecheck(
        &dslx_contents,
        entry_file
            .to_str()
            .expect("DSLX quickcheck entry file must be valid UTF-8"),
        module_name,
        &mut import_data,
    )
    .expect("DSLX parse/type-check failed for quickcheck discovery");

    let module = type_checked.get_module();
    let type_info = type_checked.get_type_info();
    let mut quickchecks = Vec::new();
    for idx in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Quickcheck(qc)) = module.get_member(idx).to_matchable() {
            let function = qc.get_function();
            let fn_ident = function.get_identifier().to_string();
            if test_filter
                .map(|re| re.is_match(fn_ident.as_str()))
                .unwrap_or(true)
            {
                let requires_itok = type_info
                    .requires_implicit_token(&function)
                    .expect("requires_implicit_token query");
                quickchecks.push((fn_ident, requires_itok));
            }
        }
    }

    (dslx_contents, quickchecks)
}

fn infer_uf_signatures_from_map(
    pkg: &ir::Package,
    uf_map: &HashMap<String, String>,
) -> HashMap<String, UfSignature> {
    let mut uf_sigs = HashMap::new();
    for (fn_name, uf_sym) in uf_map {
        let (ir_fn, skip_implicit) = match pkg.get_fn(fn_name) {
            Some(f) => (f, false),
            None => {
                let itok_name = format!("__itok{}", fn_name);
                match pkg.get_fn(&itok_name) {
                    Some(f) => (f, true),
                    None => {
                        panic!(
                            "Unknown function '{}' when inferring UF signature for symbol '{}'",
                            fn_name, uf_sym
                        );
                    }
                }
            }
        };

        let arg_widths: Vec<usize> = if skip_implicit {
            ir_fn
                .params
                .iter()
                .skip(2)
                .map(|p| p.ty.bit_count())
                .collect()
        } else {
            ir_fn.params.iter().map(|p| p.ty.bit_count()).collect()
        };
        let ret_width = ir_fn.ret_ty.bit_count();
        let sig = UfSignature {
            arg_widths,
            ret_width,
        };

        if let Some(prev) = uf_sigs.get(uf_sym) {
            if prev != &sig {
                panic!(
                    "Conflicting UF signature for symbol '{}': {:?} vs {:?}",
                    uf_sym, prev, sig
                );
            }
        } else {
            uf_sigs.insert(uf_sym.clone(), sig);
        }
    }

    uf_sigs
}

pub trait Prover {
    fn prove_ir_fn_equiv(self: &Self, lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
        self.prove_ir_fn_equiv_full(
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
    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult;
    fn prove_ir_fn_equiv_output_bits_parallel(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult;
    fn prove_ir_fn_equiv_split_input_bit(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        start_input: usize,
        start_bit: usize,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult;

    // --- QuickCheck-style proving: f() always returns bits[1]:1 ---
    fn prove_ir_fn_always_true<'a>(
        self: &Self,
        ir_fn: &IrFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult {
        let prover_fn = ProverFn {
            ir_fn,
            domains: None,
            uf_map: HashMap::new(),
        };
        let empty_signatures: HashMap<String, UfSignature> = HashMap::new();
        self.prove_ir_fn_always_true_full(
            &prover_fn,
            assertion_semantics,
            assert_label_filter,
            &empty_signatures,
        )
    }

    fn prove_ir_fn_always_true_full<'a>(
        self: &Self,
        ir_fn: &ProverFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult;

    /// Prove a DSLX quickcheck function (by name) directly.
    fn prove_dslx_quickcheck(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        quickcheck_name: &str,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
    ) -> BoolPropertyResult {
        let exact_pattern = format!("^{}$", regex::escape(quickcheck_name));
        let filter = Regex::new(&exact_pattern).expect("invalid quickcheck name regex");
        let empty_map: HashMap<String, String> = HashMap::new();
        let results = self.prove_dslx_quickcheck_full(
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            Some(&filter),
            assertion_semantics,
            assert_label_filter,
            &empty_map,
        );
        results
            .into_iter()
            .find(|r| r.name == quickcheck_name)
            .map(|r| r.result)
            .unwrap_or_else(|| panic!("quickcheck function '{}' not found", quickcheck_name))
    }

    fn prove_dslx_quickcheck_full(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&Regex>,
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
            None => lhs_pkg.get_top().expect("No functions in LHS package"),
        };
        let rhs_top = match top {
            Some(name) => rhs_pkg
                .get_fn(name)
                .expect("Top function not found in RHS package"),
            None => rhs_pkg.get_top().expect("No functions in RHS package"),
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

    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> EquivResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
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

    fn prove_ir_fn_equiv_output_bits_parallel(
        self: &Self,
        lhs: &IrFn,
        rhs: &IrFn,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        crate::prove_equiv::prove_ir_fn_equiv_output_bits_parallel::<S::Solver>(
            self,
            lhs,
            rhs,
            assertion_semantics,
            assert_label_regex.as_ref(),
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
        assert_label_filter: Option<&str>,
        allow_flatten: bool,
    ) -> EquivResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        crate::prove_equiv::prove_ir_fn_equiv_split_input_bit::<S::Solver>(
            self,
            lhs,
            rhs,
            start_input,
            start_bit,
            assertion_semantics,
            assert_label_regex.as_ref(),
            allow_flatten,
        )
    }

    fn prove_ir_fn_always_true_full<'a>(
        self: &Self,
        ir_fn: &ProverFn<'a>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> BoolPropertyResult {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        crate::prove_quickcheck::prove_ir_fn_always_true_full::<S::Solver>(
            self,
            ir_fn,
            assertion_semantics,
            assert_label_regex.as_ref(),
            uf_signatures,
        )
    }

    fn prove_dslx_quickcheck_full(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&Regex>,
        assertion_semantics: QuickCheckAssertionSemantics,
        assert_label_filter: Option<&str>,
        uf_map: &HashMap<String, String>,
    ) -> Vec<QuickCheckRunResult> {
        let assert_label_regex = build_assert_label_regex(assert_label_filter);
        let (dslx_contents, quickchecks) = load_quickcheck_context(
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
        );
        if quickchecks.is_empty() {
            return Vec::new();
        }

        let options = xlsynth::DslxConvertOptions {
            dslx_stdlib_path,
            additional_search_paths: additional_search_paths.iter().copied().collect(),
            enable_warnings: None,
            disable_warnings: None,
        };
        let ir_text = xlsynth::convert_dslx_to_ir_text(&dslx_contents, entry_file, &options)
            .expect("DSLX->IR conversion failed for quickcheck")
            .ir;

        let pkg = Parser::new(&ir_text)
            .parse_package()
            .expect("Failed to parse IR package for quickcheck");
        let module_name = entry_file
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("valid module name");

        let uf_signatures = infer_uf_signatures_from_map(&pkg, uf_map);

        let mut results = Vec::with_capacity(quickchecks.len());
        for (quickcheck_name, requires_itok) in quickchecks {
            let start_time = std::time::Instant::now();
            let mangled_itok = xlsynth::mangle_dslx_name_with_calling_convention(
                module_name,
                quickcheck_name.as_str(),
                xlsynth::DslxCallingConvention::ImplicitToken,
            )
            .expect("mangle itok");
            let mangled_normal = xlsynth::mangle_dslx_name_with_calling_convention(
                module_name,
                quickcheck_name.as_str(),
                xlsynth::DslxCallingConvention::Typical,
            )
            .expect("mangle normal");

            let (fn_ref, fixed_implicit_activation) = if requires_itok {
                if let Some(f) = pkg.get_fn(&mangled_itok) {
                    (f, true)
                } else if let Some(f) = pkg.get_fn(&mangled_normal) {
                    (f, false)
                } else {
                    panic!(
                        "quickcheck function '{}' not found (module '{}')",
                        quickcheck_name, module_name
                    );
                }
            } else if let Some(f) = pkg.get_fn(&mangled_normal) {
                (f, false)
            } else if let Some(f) = pkg.get_fn(&mangled_itok) {
                (f, true)
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

            let prover_fn = ProverFn {
                ir_fn: &ir_fn,
                domains: None,
                uf_map: uf_map.clone(),
            };

            let result = crate::prove_quickcheck::prove_ir_fn_always_true_full::<S::Solver>(
                self,
                &prover_fn,
                assertion_semantics,
                assert_label_regex.as_ref(),
                &uf_signatures,
            );

            results.push(QuickCheckRunResult {
                name: quickcheck_name,
                duration: start_time.elapsed(),
                result,
            });
        }

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

    fn prove_ir_fn_equiv_full<'a>(
        self: &Self,
        lhs: &ProverFn<'a>,
        rhs: &ProverFn<'a>,
        assertion_semantics: AssertionSemantics,
        assert_label_filter: Option<&str>,
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
        if assert_label_filter.is_some() {
            return EquivResult::Error(
                "External provers do not support assertion label filters".to_string(),
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
        assert_label_filter: Option<&str>,
        _allow_flatten: bool,
    ) -> EquivResult {
        if assert_label_filter.is_some() {
            return EquivResult::Error(
                "External provers do not support assertion label filters".to_string(),
            );
        }
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
        assert_label_filter: Option<&str>,
        _allow_flatten: bool,
    ) -> EquivResult {
        if assert_label_filter.is_some() {
            return EquivResult::Error(
                "External provers do not support assertion label filters".to_string(),
            );
        }
        EquivResult::Error("External provers do not support input-bit split strategy".to_string())
    }

    fn prove_ir_fn_always_true_full<'a>(
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

    fn prove_dslx_quickcheck_full(
        &self,
        entry_file: &std::path::Path,
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
        test_filter: Option<&Regex>,
        _assertion_semantics: QuickCheckAssertionSemantics,
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
            let filter = format!("^{}$", regex::escape(quickcheck_name.as_str()));
            let result = match self {
                ExternalProver::ToolExe(path) => {
                    crate::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_with_tool_exe(
                        path,
                        entry_file,
                        dslx_stdlib_path,
                        additional_search_paths,
                        filter.as_str(),
                    )
                }
                ExternalProver::ToolDir(path) => {
                    crate::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_with_tool_dir(
                        path,
                        entry_file,
                        dslx_stdlib_path,
                        additional_search_paths,
                        filter.as_str(),
                    )
                }
                ExternalProver::Toolchain => {
                    crate::prove_quickcheck_via_toolchain::prove_dslx_quickcheck_via_toolchain(
                        entry_file,
                        dslx_stdlib_path,
                        additional_search_paths,
                        filter.as_str(),
                    )
                }
            };

            results.push(QuickCheckRunResult {
                name: quickcheck_name,
                duration: start_time.elapsed(),
                result,
            });
        }
        results
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

pub fn prove_ir_fn_equiv_full(
    lhs: &ProverFn<'_>,
    rhs: &ProverFn<'_>,
    assertion_semantics: AssertionSemantics,
    allow_flatten: bool,
    uf_signatures: &HashMap<String, UfSignature>,
) -> EquivResult {
    auto_selected_prover().prove_ir_fn_equiv_full(
        lhs,
        rhs,
        assertion_semantics,
        None,
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
    auto_selected_prover().prove_ir_fn_always_true(ir_fn, assertion_semantics, None)
}

pub fn prove_ir_fn_always_true_with_ufs(
    ir_fn: &IrFn,
    assertion_semantics: QuickCheckAssertionSemantics,
    uf_map: &HashMap<String, String>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> BoolPropertyResult {
    let prover_fn = ProverFn {
        ir_fn,
        domains: None,
        uf_map: uf_map.clone(),
    };
    auto_selected_prover().prove_ir_fn_always_true_full(
        &prover_fn,
        assertion_semantics,
        None,
        uf_signatures,
    )
}

pub fn prove_dslx_quickcheck(
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    quickcheck_name: &str,
    assertion_semantics: QuickCheckAssertionSemantics,
) -> BoolPropertyResult {
    auto_selected_prover().prove_dslx_quickcheck(
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        quickcheck_name,
        assertion_semantics,
        None,
    )
}

pub fn prove_dslx_quickcheck_with_ufs(
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    quickcheck_name: &str,
    assertion_semantics: QuickCheckAssertionSemantics,
    uf_map: &HashMap<String, String>,
) -> BoolPropertyResult {
    let exact_pattern = format!("^{}$", regex::escape(quickcheck_name));
    let filter = Regex::new(&exact_pattern).expect("invalid quickcheck name regex");
    let results = auto_selected_prover().prove_dslx_quickcheck_full(
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        Some(&filter),
        assertion_semantics,
        None,
        uf_map,
    );
    results
        .into_iter()
        .find(|r| r.name == quickcheck_name)
        .map(|r| r.result)
        .unwrap_or_else(|| panic!("quickcheck function '{}' not found", quickcheck_name))
}
