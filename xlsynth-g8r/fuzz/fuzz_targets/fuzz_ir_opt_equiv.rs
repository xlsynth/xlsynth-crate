// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_g8r_fuzz::fuzz_bitwuzla_options;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_random::{
    generate_fn, DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy,
};
use xlsynth_prover::prover::ir_equiv::{prove_ir_fn_equiv, prove_ir_fn_equiv_output_bits_parallel};
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult, ProverFn};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::solver::bitwuzla::Bitwuzla;
#[cfg(feature = "has-boolector")]
use xlsynth_prover::solver::boolector::{Boolector, BoolectorConfig};
#[cfg(feature = "has-easy-smt")]
use xlsynth_prover::solver::easy_smt::{EasySmtConfig, EasySmtSolver};

// Checks that an enabled solver proves the optimizer preserved semantics.
fn validate_equiv_result(
    solver_result: EquivResult,
    solver_name: &str,
    orig_ir: &str,
    opt_ir: &str,
) -> bool {
    if let EquivResult::Inconclusive(msg) = &solver_result {
        log::debug!("{solver_name} equivalence inconclusive: {msg}");
        return false;
    }
    match solver_result {
        EquivResult::Proved => true,
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            ..
        } => {
            log::info!("==== IR disagreement detected ====");
            log::info!("Original IR:\n{}", orig_ir);
            log::info!("Optimized IR:\n{}", opt_ir);
            panic!(
                "Optimizer equivalence failed under {}: lhs_inputs={:?} rhs_inputs={:?}",
                solver_name, lhs_inputs, rhs_inputs
            );
        }
        EquivResult::Error(msg) => {
            panic!("{solver_name} optimizer equivalence check failed to run: {msg}");
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    // Construct valid PIR directly, then load it through libxls because this
    // target exercises the XLS optimizer.
    let operations = OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
        !matches!(
            operation,
            RandomOperation::Umulp
                | RandomOperation::Smulp
                | RandomOperation::ExtCarryOut
                | RandomOperation::ExtPrioEncode
                | RandomOperation::ExtClz
                | RandomOperation::ExtNormalizeLeft
                | RandomOperation::ExtMaskLow
                | RandomOperation::ExtNaryAdd
        )
    }));
    let options = RandomFnOptions {
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: operations,
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(data);
    let generated = generate_fn(
        &mut entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed random PIR options should construct a valid function");
    let orig_ir = generated.into_top_package("fuzz_pkg").to_string();
    let pkg = xlsynth::IrPackage::parse_ir(&orig_ir, None)
        .expect("PIR-emitted standard XLS IR should parse in libxls");
    let top_fn_name = "random_fn";

    // Optimize the IR
    let optimized_pkg = match xlsynth::optimize_ir(&pkg, top_fn_name) {
        Ok(p) => p,
        Err(e) => {
            log::error!("optimize_ir failed: {}", e);
            return;
        }
    };

    // Parse both packages using the xlsynth-g8r parser
    let orig_pkg = ir_parser::Parser::new(&orig_ir)
        .parse_and_validate_package()
        .unwrap();
    let opt_pkg = ir_parser::Parser::new(&optimized_pkg.to_string())
        .parse_and_validate_package()
        .unwrap();

    let orig_fn = orig_pkg.get_top_fn().unwrap();
    let opt_fn = opt_pkg.get_top_fn().unwrap();

    let opt_ir = optimized_pkg.to_string();
    #[cfg(feature = "has-bitwuzla")]
    {
        let bitwuzla_result = prove_ir_fn_equiv::<Bitwuzla>(
            &fuzz_bitwuzla_options(),
            &ProverFn::new(orig_fn, None),
            &ProverFn::new(opt_fn, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        if !validate_equiv_result(
            bitwuzla_result,
            "Bitwuzla",
            &orig_ir,
            &opt_ir,
        ) {
            // A configured solver limit is expected to make some fuzz samples
            // inconclusive; those samples are not optimizer failures.
            return;
        }
    }

    #[cfg(feature = "has-boolector")]
    {
        let boolector_result = prove_ir_fn_equiv::<Boolector>(
            &BoolectorConfig::new(),
            &ProverFn::new(orig_fn, None),
            &ProverFn::new(opt_fn, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        if !validate_equiv_result(
            boolector_result,
            "Boolector",
            &orig_ir,
            &opt_ir,
        ) {
            // An inconclusive solver result is not an optimizer failure.
            return;
        }
    }

    #[cfg(feature = "with-boolector-binary-test")]
    {
        let boolector_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::boolector(),
            &ProverFn::new(orig_fn, None),
            &ProverFn::new(opt_fn, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        if !validate_equiv_result(
            boolector_result,
            "Boolector binary",
            &orig_ir,
            &opt_ir,
        ) {
            // An inconclusive solver result is not an optimizer failure.
            return;
        }
    }

    #[cfg(feature = "with-bitwuzla-binary-test")]
    {
        let bitwuzla_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::bitwuzla(),
            &ProverFn::new(orig_fn, None),
            &ProverFn::new(opt_fn, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        if !validate_equiv_result(
            bitwuzla_result,
            "Bitwuzla binary",
            &orig_ir,
            &opt_ir,
        ) {
            // An inconclusive solver result is not an optimizer failure.
            return;
        }
    }

    #[cfg(feature = "with-z3-binary-test")]
    {
        let z3_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::z3(),
            &ProverFn::new(orig_fn, None),
            &ProverFn::new(opt_fn, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        if !validate_equiv_result(z3_result, "Z3 binary", &orig_ir, &opt_ir) {
            // An inconclusive solver result is not an optimizer failure.
            return;
        }
    }

    #[cfg(feature = "has-bitwuzla")]
    {
        let output_bit_count = orig_fn.ret_ty.bit_count();
        if output_bit_count <= 64 {
            let bitwuzla_parallel_result = prove_ir_fn_equiv_output_bits_parallel::<Bitwuzla>(
                &fuzz_bitwuzla_options(),
                &ProverFn::new(orig_fn, None),
                &ProverFn::new(opt_fn, None),
                AssertionSemantics::Same,
                None,
                false,
            );
            if !validate_equiv_result(
                bitwuzla_parallel_result,
                "Bitwuzla-parallel",
                &orig_ir,
                &opt_ir,
            ) {
                // A configured solver limit is expected to make some fuzz
                // samples inconclusive; those samples are not optimizer
                // failures.
                return;
            }
        }
    }
});
