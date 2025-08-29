// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::check_equivalence;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_g8r::equiv::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
#[cfg(feature = "has-boolector")]
use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
#[cfg(feature = "has-easy-smt")]
use xlsynth_g8r::equiv::easy_smt_backend::{EasySmtConfig, EasySmtSolver};
use xlsynth_g8r::equiv::prove_equiv::{
    AssertionSemantics, EquivResult, IrFn, prove_ir_fn_equiv,
    prove_ir_fn_equiv_output_bits_parallel,
};
use xlsynth_g8r::xls_ir::ir_parser;
use xlsynth_test_helpers::ir_fuzz::{FuzzSample, generate_ir_fn};

// Insert helper that checks consistency among the external tool, a primary
// solver result, and an optional per-bit parallel solver result.
fn validate_equiv_result(
    ext_equiv: Result<(), String>,
    solver_result: EquivResult,
    solver_name: &str,
    orig_ir: &str,
    opt_ir: &str,
) {
    match ext_equiv {
        Ok(()) => {
            // External tool says equivalent – solver must prove equivalence.
            if let EquivResult::Disproved { lhs_inputs, rhs_inputs, .. } = solver_result {
                log::info!("==== IR disagreement detected ====");
                log::info!("Original IR:\n{}", orig_ir);
                log::info!("Optimized IR:\n{}", opt_ir);
                panic!(
                    "Disagreement: external tool says equivalent, but {} solver disproves: lhs_inputs={:?} rhs_inputs={:?}",
                    solver_name, lhs_inputs, rhs_inputs
                );
            }
        }
        Err(ext_err) => {
            // External tool says not equivalent – solver should also find inequivalence.
            if let EquivResult::Proved = solver_result {
                log::info!("==== IR disagreement detected ====");
                log::info!("Original IR:\n{}", orig_ir);
                log::info!("Optimized IR:\n{}", opt_ir);
                panic!(
                    "Disagreement: external tool says NOT equivalent, but {} solver proves equivalence. External error: {}",
                    solver_name, ext_err
                );
            }
        }
    }
}

fuzz_target!(|sample: FuzzSample| {
    // Ensure XLSYNTH_TOOLS is set for equivalence checking
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    if sample.ops.is_empty() || sample.input_bits == 0 {
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // Build an XLS IR package from the fuzz sample
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").unwrap();
    if let Err(e) = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg) {
        log::info!("IR generation failed: {}", e);
        return;
    }

    let top_fn = pkg.get_function("fuzz_test").unwrap();

    // Optimize the IR
    let optimized_pkg = match xlsynth::optimize_ir(&pkg, top_fn.get_name().as_str()) {
        Ok(p) => p,
        Err(e) => {
            log::error!("optimize_ir failed: {}", e);
            return;
        }
    };

    // Parse both packages using the xlsynth-g8r parser
    let orig_pkg = ir_parser::Parser::new(&pkg.to_string())
        .parse_and_validate_package()
        .unwrap();
    let opt_pkg = ir_parser::Parser::new(&optimized_pkg.to_string())
        .parse_and_validate_package()
        .unwrap();

    let orig_fn = orig_pkg.get_top().unwrap();
    let opt_fn = opt_pkg.get_top().unwrap();

    // Check equivalence using the external tool first, specifying the top function
    let orig_ir = pkg.to_string();
    let opt_ir = optimized_pkg.to_string();
    let top_fn_name = "fuzz_test";
    let ext_equiv =
        check_equivalence::check_equivalence_with_top(&orig_ir, &opt_ir, Some(top_fn_name), false);
    #[cfg(feature = "has-bitwuzla")]
    {
        let bitwuzla_result = prove_ir_fn_equiv::<Bitwuzla>(
            &BitwuzlaOptions::new(),
            &IrFn::new(orig_fn, None),
            &IrFn::new(opt_fn, None),
            AssertionSemantics::Same,
            false,
        );
        validate_equiv_result(
            ext_equiv.clone(),
            bitwuzla_result,
            "Bitwuzla",
            &orig_ir,
            &opt_ir,
        );
    }

    #[cfg(feature = "has-boolector")]
    {
        let boolector_result = prove_ir_fn_equiv::<Boolector>(
            &BoolectorConfig::new(),
            &IrFn::new(orig_fn, None),
            &IrFn::new(opt_fn, None),
            AssertionSemantics::Same,
            false,
        );
        validate_equiv_result(
            ext_equiv.clone(),
            boolector_result,
            "Boolector",
            &orig_ir,
            &opt_ir,
        );
    }

    #[cfg(feature = "with-boolector-binary-test")]
    {
        let boolector_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::boolector(),
            &IrFn::new(orig_fn, None),
            &IrFn::new(opt_fn, None),
            AssertionSemantics::Same,
            false,
        );
        validate_equiv_result(
            ext_equiv.clone(),
            boolector_result,
            "Boolector binary",
            &orig_ir,
            &opt_ir,
        );
    }

    #[cfg(feature = "with-bitwuzla-binary-test")]
    {
        let bitwuzla_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::bitwuzla(),
            &IrFn::new(orig_fn, None),
            &IrFn::new(opt_fn, None),
            AssertionSemantics::Same,
            false,
        );
        validate_equiv_result(
            ext_equiv.clone(),
            bitwuzla_result,
            "Bitwuzla binary",
            &orig_ir,
            &opt_ir,
        );
    }

    #[cfg(feature = "with-z3-binary-test")]
    {
        let z3_result = prove_ir_fn_equiv::<EasySmtSolver>(
            &EasySmtConfig::z3(),
            &IrFn::new(orig_fn, None),
            &IrFn::new(opt_fn, None),
            AssertionSemantics::Same,
            false,
        );
        validate_equiv_result(ext_equiv.clone(), z3_result, "Z3 binary", &orig_ir, &opt_ir);
    }

    #[cfg(feature = "has-bitwuzla")]
    {
        let output_bit_count = orig_fn.ret_ty.bit_count();
        if output_bit_count <= 64 {
            let bitwuzla_parallel_result = prove_ir_fn_equiv_output_bits_parallel::<Bitwuzla>(
                &BitwuzlaOptions::new(),
                &IrFn::new(orig_fn, None),
                &IrFn::new(opt_fn, None),
                AssertionSemantics::Same,
                false,
            );
            validate_equiv_result(
                ext_equiv,
                bitwuzla_parallel_result,
                "Bitwuzla-parallel",
                &orig_ir,
                &opt_ir,
            );
        }
    }
});
