// SPDX-License-Identifier: Apache-2.0

use crate::{
    solver_interface::{BitVec, Response, Solver},
    translate::{get_fn_inputs, ir_to_smt},
    types::{
        AssertionViolation, BoolPropertyResult, FnInput, FnOutput, IrFn,
        QuickCheckAssertionSemantics, UfRegistry, UfSignature,
    },
};

use std::collections::HashMap;
use xlsynth_pir::ir;

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
) -> BoolPropertyResult
where
    S: Solver,
    S::Term: 'a,
{
    prove_ir_fn_always_true_with_ufs::<S>(
        solver_config,
        ir_fn,
        assertion_semantics,
        &HashMap::new(),
        &HashMap::new(),
    )
}

/// UF-enabled variant of `prove_ir_fn_always_true`.
pub fn prove_ir_fn_always_true_with_ufs<'a, S>(
    solver_config: &S::Config,
    ir_fn: &IrFn<'a>,
    assertion_semantics: QuickCheckAssertionSemantics,
    uf_map: &HashMap<String, String>,
    uf_sigs: &HashMap<String, UfSignature>,
) -> BoolPropertyResult
where
    S: Solver,
    S::Term: 'a,
{
    // Ensure the function indeed returns a single-bit value.
    assert_eq!(
        ir_fn.fn_ref.ret_ty.bit_count(),
        1,
        "Function must return a single-bit value"
    );

    let mut solver = S::new(solver_config).unwrap();

    // Generate SMT representation with UF mapping/registry.
    let fn_inputs = get_fn_inputs(&mut solver, ir_fn, None);
    let uf_registry = UfRegistry::from_uf_signatures(&mut solver, uf_sigs);
    let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &uf_map, &uf_registry);

    // Build a 1-bit flag that is `1` iff *all* in-function assertions pass.
    let success_flag: BitVec<S::Term> = if smt_fn.assertions.is_empty() {
        solver.numerical(1, 1)
    } else {
        let mut acc_opt: Option<BitVec<S::Term>> = None;
        for a in &smt_fn.assertions {
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
            for a in &smt_fn.assertions {
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

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::types::IrFn;
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
        let res = super::prove_ir_fn_always_true::<S>(solver_config, &ir_fn, sem);
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
        let res = super::prove_ir_fn_always_true::<S>(solver_config, &ir_fn, sem);
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

        let res = super::prove_ir_fn_always_true_with_ufs::<S>(
            solver_config,
            &ir_fn,
            QuickCheckAssertionSemantics::Assume,
            &uf_map,
            &uf_sigs,
        );
        assert!(matches!(res, super::BoolPropertyResult::Proved));
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
