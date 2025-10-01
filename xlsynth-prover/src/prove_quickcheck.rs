// SPDX-License-Identifier: Apache-2.0

use crate::{
    solver_interface::{BitVec, Response, Solver, SolverConfig},
    translate::{get_fn_inputs, ir_to_smt, ir_value_to_bv},
    types::{
        AssertionViolation, BoolPropertyResult, FnInput, FnOutput, IrFn, ProverFn,
        QuickCheckAssertionSemantics, QuickCheckRunResult, UfRegistry, UfSignature,
    },
};

use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use xlsynth::dslx::{ImportData, MatchableModuleMember};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;

/// Prove that a given `IrFn` always returns boolean `true` (`bits[1] == 1`) for
/// all possible inputs.
///
/// * `solver_config` – backend-specific solver configuration.
/// * `ir_fn`          – function to analyse – must return `bits[1]`.
/// * `assertion_semantics` – semantics used for in-function `assert` handling
///   (see [`QuickCheckAssertionSemantics`]).  Most callers will want
///   `QuickCheckAssertionSemantics::Assume` so that the property is only
///   required to hold when the function itself does **not** raise an assertion.
///
/// Returns [`BoolPropertyResult`] describing the outcome.
pub fn prove_ir_fn_always_true<'a, S>(
    solver_config: &S::Config,
    ir_fn: &IrFn<'a>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_include: Option<&Regex>,
) -> BoolPropertyResult
where
    S: Solver,
    S::Term: 'a,
{
    let prover_fn = ProverFn {
        ir_fn,
        domains: None,
        uf_map: HashMap::new(),
    };
    let empty_signatures: HashMap<String, UfSignature> = HashMap::new();
    prove_ir_fn_always_true_full::<S>(
        solver_config,
        &prover_fn,
        assertion_semantics,
        assert_label_include,
        &empty_signatures,
    )
}

/// UF-enabled variant of `prove_ir_fn_always_true`.
pub fn prove_ir_fn_always_true_full<'a, S>(
    solver_config: &S::Config,
    prover_fn: &ProverFn<'a>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_include: Option<&Regex>,
    uf_signatures: &HashMap<String, UfSignature>,
) -> BoolPropertyResult
where
    S: Solver,
    S::Term: 'a,
{
    let ir_fn = prover_fn.ir_fn;

    // Ensure the function indeed returns a single-bit value.
    assert_eq!(
        ir_fn.fn_ref.ret_ty.bit_count(),
        1,
        "Function must return a single-bit value"
    );

    let mut solver = S::new(solver_config).unwrap();

    // Generate SMT representation with UF mapping/registry.
    let fn_inputs = get_fn_inputs(&mut solver, ir_fn.clone(), None);

    if let Some(domains) = &prover_fn.domains {
        for param in fn_inputs.params().iter() {
            if let Some(allowed) = domains.get(&param.name) {
                if let Some(sym) = fn_inputs.inputs.get(&param.name) {
                    let mut domain_constraint: Option<BitVec<S::Term>> = None;
                    for value in allowed {
                        let value_bv = ir_value_to_bv(&mut solver, value, &param.ty).bitvec;
                        let eq = solver.eq(&sym.bitvec, &value_bv);
                        domain_constraint = Some(match domain_constraint {
                            None => eq,
                            Some(prev) => solver.or(&prev, &eq),
                        });
                    }
                    if let Some(expr) = domain_constraint {
                        solver.assert(&expr).unwrap();
                    }
                }
            }
        }
    }

    let uf_registry = UfRegistry::from_uf_signatures(&mut solver, uf_signatures);
    let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &prover_fn.uf_map, &uf_registry);

    // Optionally filter assertions by label before applying semantics.
    let filtered_assertions =
        crate::assertion_filter::filter_assertions(&smt_fn.assertions, assert_label_include);

    // Build a 1-bit flag that is `1` iff *all* selected in-function assertions
    // pass.
    let success_flag: BitVec<S::Term> = if filtered_assertions.is_empty() {
        solver.numerical(1, 1)
    } else {
        let mut acc_opt: Option<BitVec<S::Term>> = None;
        for a in filtered_assertions.iter() {
            acc_opt = Some(match acc_opt {
                None => a.active.clone(),
                Some(prev) => solver.and(&prev, &a.active),
            });
        }
        acc_opt.expect("acc populated")
    };

    let output_is_false = {
        let false_bv = solver.zero(1);
        solver.eq(&smt_fn.output.bitvec, &false_bv)
    };

    // Build condition according to assertion semantics.
    let condition = match assertion_semantics {
        QuickCheckAssertionSemantics::Ignore => output_is_false.clone(),
        QuickCheckAssertionSemantics::Never => {
            // We require no assertion to fail, so any failure is a counter-example.
            let failed = solver.not(&success_flag);
            solver.or(&failed, &output_is_false)
        }
        QuickCheckAssertionSemantics::Assume => {
            // If the function succeeds (all assertions pass) AND output is false.
            solver.and(&success_flag, &output_is_false)
        }
    };

    // Ask solver for a model that satisfies the *negation* of the property.
    solver.assert(&condition).unwrap();

    match solver.check().unwrap() {
        Response::Unsat => BoolPropertyResult::Proved,
        Response::Sat => {
            // Extract counter-example values.
            let inputs: Vec<FnInput> = smt_fn
                .fn_ref
                .params
                .iter()
                .zip(smt_fn.inputs.iter())
                .map(|(p, i)| FnInput {
                    name: p.name.clone(),
                    value: solver.get_value(&i.bitvec, &i.ir_type).unwrap(),
                })
                .collect();

            // Determine if any assertion violated and build FnOutput accordingly
            let mut violation: Option<(String, String)> = None;
            for a in filtered_assertions.iter() {
                let val = solver.get_value(&a.active, &ir::Type::Bits(1)).unwrap();
                let bits = val.to_bits().unwrap();
                if !bits.get_bit(0).unwrap() {
                    violation = Some((a.message.to_string(), a.label.to_string()));
                    break;
                }
            }

            let output: FnOutput = {
                FnOutput {
                    value: solver
                        .get_value(&smt_fn.output.bitvec, &smt_fn.output.ir_type)
                        .unwrap(),
                    assertion_violation: violation.map(|(msg, lbl)| AssertionViolation {
                        message: msg,
                        label: lbl,
                    }),
                }
            };

            BoolPropertyResult::Disproved { inputs, output }
        }
        Response::Unknown => panic!("Solver returned unknown"),
    }
}

pub(crate) fn build_assert_label_regex(filter: Option<&str>) -> Option<Regex> {
    filter.map(|pattern| {
        Regex::new(pattern).expect("invalid regular expression in assert label filter")
    })
}

pub(crate) fn load_quickcheck_context(
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: Option<&str>,
) -> (String, Vec<(String, bool)>) {
    let dslx_contents = fs::read_to_string(entry_file)
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
    let test_regex = test_filter.map(|pattern| {
        Regex::new(pattern).expect("invalid regular expression in quickcheck test filter")
    });
    let mut quickchecks = Vec::new();
    for idx in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Quickcheck(qc)) = module.get_member(idx).to_matchable() {
            let function = qc.get_function();
            let fn_ident = function.get_identifier().to_string();
            if test_regex
                .as_ref()
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

pub(crate) fn infer_uf_signatures_from_map(
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

pub(crate) fn prove_dslx_quickcheck_full<SConfig>(
    solver_config: &SConfig,
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: Option<&str>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
    uf_map: &HashMap<String, String>,
) -> Vec<QuickCheckRunResult>
where
    SConfig: SolverConfig,
{
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
        ..Default::default()
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

        let result = prove_ir_fn_always_true_full::<SConfig::Solver>(
            solver_config,
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

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::types::{IrFn, ParamDomains, ProverFn};
    use xlsynth::IrValue;
    use xlsynth_pir::ir_parser::Parser;

    /// Assert that `prove_ir_fn_always_true` returns `Proved`.
    pub fn assert_proved<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        sem: QuickCheckAssertionSemantics,
    ) {
        assert_proved_with_fixed_implicit_activation_choice::<S>(solver_config, ir_text, sem, false)
    }

    pub fn assert_proved_with_fixed_implicit_activation_choice<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        sem: QuickCheckAssertionSemantics,
        fixed_implicit_activation: bool,
    ) {
        let mut parser = Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR function");
        let ir_fn = IrFn {
            fn_ref: &f,
            pkg_ref: None,
            fixed_implicit_activation,
        };
        let res = super::prove_ir_fn_always_true::<S>(solver_config, &ir_fn, sem, None);
        assert!(matches!(res, BoolPropertyResult::Proved));
    }

    /// Assert that `prove_ir_fn_always_true` returns `Disproved`.
    pub fn assert_disproved<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        sem: QuickCheckAssertionSemantics,
        expect_violation: bool,
    ) {
        assert_disproved_with_fixed_implicit_activation_choice::<S>(
            solver_config,
            ir_text,
            sem,
            expect_violation,
            false,
        )
    }

    pub fn assert_disproved_with_fixed_implicit_activation_choice<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        sem: QuickCheckAssertionSemantics,
        expect_violation: bool,
        fixed_implicit_activation: bool,
    ) {
        let mut parser = Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR function");
        let ir_fn = IrFn {
            fn_ref: &f,
            pkg_ref: None,
            fixed_implicit_activation,
        };
        let res = super::prove_ir_fn_always_true::<S>(solver_config, &ir_fn, sem, None);
        match res {
            BoolPropertyResult::Disproved { output, .. } => match (expect_violation, output) {
                (
                    true,
                    FnOutput {
                        assertion_violation: Some(_),
                        ..
                    },
                ) => {}
                (
                    false,
                    FnOutput {
                        assertion_violation: None,
                        ..
                    },
                ) => {}
                (true, other) => panic!("Expected AssertionViolation, got {:?}", other),
                (false, other) => panic!("Expected Value, got {:?}", other),
            },
            _ => panic!("Expected Disproved result"),
        }
    }

    // === IR snippets ===

    pub const ALWAYS_TRUE: &str = r#"
        fn f() -> bits[1] {
            ret lit1: bits[1] = literal(value=1, id=1)
        }
    "#;

    pub const IDENTITY_BOOL: &str = r#"
        fn f(x: bits[1]) -> bits[1] {
            ret p: bits[1] = param(name=x, id=1)
        }
    "#;

    pub const ASSERT_ON_PARAM: &str = r#"
        fn f(__token: token, ok: bits[1]) -> bits[1] {
            assert.1: token = assert(__token, ok, message="fail", label="a", id=1)
            ret p: bits[1] = param(name=ok, id=2)
        }
    "#;

    // Assert may fail but function always returns true.
    pub const ASSERT_ON_PARAM_RET_TRUE: &str = r#"
        fn f(__token: token, ok: bits[1]) -> bits[1] {
            assert.1: token = assert(__token, ok, message="fail", label="a", id=1)
            ret lit1: bits[1] = literal(value=1, id=2)
        }
    "#;

    // Function with implicit token & activation param, returns activation param.
    pub const TOKEN_ACT_RET_ACT: &str = r#"
        fn f(__token: token, __act: bits[1]) -> (token, bits[1]) {
            literal.2: bits[1] = literal(value=1, id=2)
            assert.3: token = assert(__token, __act, message="fail", label="a", id=3)
            ret p: (token, bits[1]) = tuple(assert.3, literal.2, id=4)
        }
    "#;

    /// Ensure that counter-example input ordering matches function parameter
    /// order.
    ///
    /// This mirrors the ordering check performed in
    /// `prove_equiv::test_utils::test_counterexample_input_order`,
    /// but for the QuickCheck path where we prove a single function is always
    /// true. We create a trivially falsifiable function with multiple
    /// parameters of differing widths and confirm that the returned
    /// `inputs` vector from `BoolPropertyResult::Disproved` preserves the
    /// declared parameter order and widths.
    pub fn test_counterexample_input_order<S: Solver>(solver_config: &S::Config) {
        // Intentionally not always-true: returns param `c`, so setting c=0 falsifies
        // the property.
        let ir_text = r#"
            fn f(a: bits[8], b: bits[4], c: bits[1]) -> bits[1] {
                ret pc: bits[1] = identity(c, id=1)
            }
        "#;

        let mut parser = Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR function");
        let ir_fn = IrFn {
            fn_ref: &f,
            pkg_ref: None,
            fixed_implicit_activation: false,
        };

        let res = super::prove_ir_fn_always_true::<S>(
            solver_config,
            &ir_fn,
            QuickCheckAssertionSemantics::Ignore,
            None,
        );

        match res {
            BoolPropertyResult::Disproved { inputs, .. } => {
                assert_eq!(inputs.len(), f.params.len());
                for (idx, param) in f.params.iter().enumerate() {
                    assert_eq!(
                        inputs[idx].name, param.name,
                        "param name mismatch at index {idx}"
                    );
                    assert_eq!(
                        inputs[idx].value.bit_count().unwrap(),
                        param.ty.bit_count(),
                        "param bit width mismatch at index {idx}"
                    );
                }
            }
            other => panic!(
                "Expected Disproved result with counterexample, got {:?}",
                other
            ),
        }
    }

    /// QuickCheck with UF mapping: prove f(x) == (invoke g x) == (invoke h x)
    /// is always true when g/h are abstracted to the same UF.
    pub fn test_quickcheck_uf_basic<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_qc_uf

            fn g(x: bits[8] id=1) -> bits[8] {
              ret add.2: bits[8] = add(x, x, id=2)
            }

            fn h(x: bits[8] id=3) -> bits[8] {
              ret sub.4: bits[8] = sub(x, x, id=4)
            }

            fn f(x: bits[8] id=5) -> bits[1] {
              a: bits[8] = invoke(x, to_apply=g, id=6)
              b: bits[8] = invoke(x, to_apply=h, id=7)
              ret eq.8: bits[1] = eq(a, b, id=8)
            }
        "#;

        let pkg = Parser::new(ir_pkg_text)
            .parse_and_validate_package()
            .expect("parse package");
        let f = pkg.get_fn("f").expect("f not found");
        let ir_fn = IrFn {
            fn_ref: f,
            pkg_ref: Some(&pkg),
            fixed_implicit_activation: false,
        };

        let mut uf_map: HashMap<String, String> = HashMap::new();
        uf_map.insert("g".to_string(), "F".to_string());
        uf_map.insert("h".to_string(), "F".to_string());
        let mut uf_sigs: HashMap<String, crate::types::UfSignature> = HashMap::new();
        uf_sigs.insert(
            "F".to_string(),
            crate::types::UfSignature {
                arg_widths: vec![8],
                ret_width: 8,
            },
        );

        let prover_fn = ProverFn {
            ir_fn: &ir_fn,
            domains: None,
            uf_map: uf_map.clone(),
        };

        let res = super::prove_ir_fn_always_true_full::<S>(
            solver_config,
            &prover_fn,
            QuickCheckAssertionSemantics::Assume,
            None,
            &uf_sigs,
        );
        assert!(matches!(res, super::BoolPropertyResult::Proved));
    }

    /// Assertion-label include filter for QuickCheck: without filter, 'Never'
    /// fails due to red assertion depending on input; with include('blue'),
    /// no included assertions can fail, so the property is proved.
    pub fn test_qc_assert_label_filter<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn f(__token: token, a: bits[1]) -> bits[1] {
              assert.1: token = assert(__token, a, message="rf", label="red", id=1)
              t: bits[1] = literal(value=1, id=2)
              assert.3: token = assert(assert.1, t, message="bf", label="blue", id=3)
              ret lit1: bits[1] = literal(value=1, id=4)
            }
        "#;

        let mut parser = Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR function");
        let ir_fn = IrFn {
            fn_ref: &f,
            pkg_ref: None,
            fixed_implicit_activation: false,
        };

        // No filter: 'Never' should be disproved (red may fail for a=0)
        let prover_fn = ProverFn {
            ir_fn: &ir_fn,
            domains: None,
            uf_map: std::collections::HashMap::new(),
        };
        let empty_signatures =
            std::collections::HashMap::<String, crate::types::UfSignature>::new();

        let res_no_filter = super::prove_ir_fn_always_true_full::<S>(
            solver_config,
            &prover_fn,
            QuickCheckAssertionSemantics::Never,
            None,
            &empty_signatures,
        );
        assert!(matches!(
            res_no_filter,
            super::BoolPropertyResult::Disproved { .. }
        ));

        // Filter include only 'blue': all included asserts hold -> Proved
        let include = Regex::new(r"^(?:blue)$").unwrap();
        let res_filtered = super::prove_ir_fn_always_true_full::<S>(
            solver_config,
            &prover_fn,
            QuickCheckAssertionSemantics::Never,
            Some(&include),
            &empty_signatures,
        );
        assert!(matches!(res_filtered, super::BoolPropertyResult::Proved));
    }

    /// Parameter-domain restriction: without domains the property is falsified,
    /// but constraining the argument to the valid enum set makes it provable.
    pub fn test_quickcheck_param_domains<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn f(x: bits[2]) -> bits[1] {
              literal.1: bits[2] = literal(value=1, id=1)
              ret eq.2: bits[1] = eq(x, literal.1, id=2)
            }
        "#;

        let mut parser = Parser::new(ir_text);
        let f = parser.parse_fn().expect("Failed to parse IR function");
        let ir_fn = IrFn {
            fn_ref: &f,
            pkg_ref: None,
            fixed_implicit_activation: false,
        };

        // Without domains, the property is disproved (e.g. x = 0).
        let res_no_domains = super::prove_ir_fn_always_true::<S>(
            solver_config,
            &ir_fn,
            QuickCheckAssertionSemantics::Ignore,
            None,
        );
        assert!(matches!(
            res_no_domains,
            super::BoolPropertyResult::Disproved { .. }
        ));

        // Restrict x to {1}; the property now holds.
        let mut domains: ParamDomains = HashMap::new();
        domains.insert(
            "x".to_string(),
            vec![IrValue::make_ubits(2, 1).expect("make ubits")],
        );

        let prover_fn = ProverFn {
            ir_fn: &ir_fn,
            domains: Some(domains),
            uf_map: HashMap::new(),
        };
        let empty_signatures: HashMap<String, crate::types::UfSignature> = HashMap::new();

        let res_with_domains = super::prove_ir_fn_always_true_full::<S>(
            solver_config,
            &prover_fn,
            QuickCheckAssertionSemantics::Ignore,
            None,
            &empty_signatures,
        );
        assert!(matches!(
            res_with_domains,
            super::BoolPropertyResult::Proved
        ));
    }
}

#[cfg(test)]
macro_rules! quickcheck_test_with_solver {
    ($mod_ident:ident, $solver_type:ty, $solver_config:expr) => {
        #[cfg(test)]
        mod $mod_ident {
            use super::*;
            use crate::prove_quickcheck::test_utils;

            #[test]
            fn always_true_proved_ignore() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ALWAYS_TRUE,
                    QuickCheckAssertionSemantics::Ignore,
                );
            }

            #[test]
            fn always_true_proved_never() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ALWAYS_TRUE,
                    QuickCheckAssertionSemantics::Never,
                );
            }

            #[test]
            fn always_true_proved_assume() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ALWAYS_TRUE,
                    QuickCheckAssertionSemantics::Assume,
                );
            }

            #[test]
            fn identity_bool_disproved() {
                test_utils::assert_disproved::<$solver_type>(
                    $solver_config,
                    test_utils::IDENTITY_BOOL,
                    QuickCheckAssertionSemantics::Ignore,
                    false,
                );
            }

            #[test]
            fn assert_param_ignore_disproved_violation() {
                test_utils::assert_disproved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM,
                    QuickCheckAssertionSemantics::Ignore,
                    true,
                );
            }

            #[test]
            fn assert_param_never_disproved_violation() {
                test_utils::assert_disproved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM,
                    QuickCheckAssertionSemantics::Never,
                    true,
                );
            }

            #[test]
            fn assert_param_assume_proved() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM,
                    QuickCheckAssertionSemantics::Assume,
                );
            }

            // Tests for function always returning true but having assertion.

            #[test]
            fn assert_param_ret_true_ignore_proved() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM_RET_TRUE,
                    QuickCheckAssertionSemantics::Ignore,
                );
            }

            #[test]
            fn assert_param_ret_true_never_disproved_violation() {
                test_utils::assert_disproved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM_RET_TRUE,
                    QuickCheckAssertionSemantics::Never,
                    true,
                );
            }

            #[test]
            fn assert_param_ret_true_assume_proved() {
                test_utils::assert_proved::<$solver_type>(
                    $solver_config,
                    test_utils::ASSERT_ON_PARAM_RET_TRUE,
                    QuickCheckAssertionSemantics::Assume,
                );
            }

            // ---------- Fixed implicit activation tests ----------

            // No fixed implicit activation: activation param free, function returns
            // activation -> should be disproved.
            #[test]
            fn token_no_ret_act_disproved() {
                test_utils::assert_disproved_with_fixed_implicit_activation_choice::<$solver_type>(
                    $solver_config,
                    test_utils::TOKEN_ACT_RET_ACT,
                    QuickCheckAssertionSemantics::Never,
                    true,
                    false,
                );
            }

            // Fixed implicit activation: activation implicitly fixed to 1, returning
            // activation should prove.
            #[test]
            fn token_fixed_ret_act_proved() {
                test_utils::assert_proved_with_fixed_implicit_activation_choice::<$solver_type>(
                    $solver_config,
                    test_utils::TOKEN_ACT_RET_ACT,
                    QuickCheckAssertionSemantics::Never,
                    true,
                );
            }

            // New: ensure counterexample input ordering matches parameter order.
            #[test]
            fn counterexample_input_order() {
                test_utils::test_counterexample_input_order::<$solver_type>($solver_config);
            }

            #[test]
            fn quickcheck_uf_basic() {
                test_utils::test_quickcheck_uf_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn quickcheck_assert_label_filter() {
                test_utils::test_qc_assert_label_filter::<$solver_type>($solver_config);
            }
            #[test]
            fn quickcheck_param_domains() {
                test_utils::test_quickcheck_param_domains::<$solver_type>($solver_config);
            }
        }
    };
}

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-binary-test")]
quickcheck_test_with_solver!(
    bitwuzla_qc_tests,
    crate::easy_smt_backend::EasySmtSolver,
    &crate::easy_smt_backend::EasySmtConfig::bitwuzla()
);

#[cfg(test)]
#[cfg(feature = "with-boolector-binary-test")]
quickcheck_test_with_solver!(
    boolector_qc_tests,
    crate::easy_smt_backend::EasySmtSolver,
    &crate::easy_smt_backend::EasySmtConfig::boolector()
);

#[cfg(test)]
#[cfg(feature = "with-z3-binary-test")]
quickcheck_test_with_solver!(
    z3_qc_tests,
    crate::easy_smt_backend::EasySmtSolver,
    &crate::easy_smt_backend::EasySmtConfig::z3()
);

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-built")]
quickcheck_test_with_solver!(
    bitwuzla_built_qc_tests,
    crate::bitwuzla_backend::Bitwuzla,
    &crate::bitwuzla_backend::BitwuzlaOptions::new()
);
