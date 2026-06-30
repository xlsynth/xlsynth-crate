// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use xlsynth::IrBits;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r_fuzz::generate_full_g8r_fuzz_case;
use xlsynth_pir::ir::{Fn, Package};
use xlsynth_pir::ir_eval::{self, FnEvalResult};
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::random_inputs::generate_argument_sets_with_rng;

const SIMULATION_SAMPLE_COUNT: usize = 1024;

/// Simulates both forms using inputs seeded only by the generated IR text.
fn check_simulation_equivalence(
    package: &Package,
    source_fn: &Fn,
    gate_fn: &GateFn,
    ir_text: &str,
) {
    let mut seed = [0_u8; 32];
    seed.copy_from_slice(blake3::hash(ir_text.as_bytes()).as_bytes());
    let mut rng = StdRng::from_seed(seed);
    let argument_sets =
        generate_argument_sets_with_rng(source_fn, &mut rng, SIMULATION_SAMPLE_COUNT);

    for (sample_index, args) in argument_sets.into_iter().enumerate() {
        let pir_output = match ir_eval::eval_fn_in_package(package, source_fn, &args) {
            FnEvalResult::Success(success) => success.value,
            failure @ FnEvalResult::Failure(_) => panic!(
                "source PIR evaluation failed for g8r fuzz sample {sample_index}:\n\
                 IR:\n{ir_text}\nargs={args:?}\nresult={failure:?}"
            ),
        };

        let gate_inputs: Vec<IrBits> = args
            .iter()
            .zip(source_fn.params.iter())
            .map(|(value, param)| {
                let mut bits = Vec::with_capacity(param.ty.bit_count());
                flatten_ir_value_to_lsb0_bits_for_type(value, &param.ty, &mut bits)
                    .expect("generated argument should match its PIR parameter type");
                IrBits::from_lsb_is_0(&bits)
            })
            .collect();
        let gate_outputs = gate_sim::eval(gate_fn, &gate_inputs, Collect::None).outputs;
        assert_eq!(
            gate_outputs.len(),
            1,
            "gatified function should have one flattened output"
        );

        let mut expected_output_bits = Vec::with_capacity(source_fn.ret_ty.bit_count());
        flatten_ir_value_to_lsb0_bits_for_type(
            &pir_output,
            &source_fn.ret_ty,
            &mut expected_output_bits,
        )
        .expect("PIR evaluator result should match the function return type");
        let expected_output = IrBits::from_lsb_is_0(&expected_output_bits);
        assert_eq!(
            gate_outputs[0], expected_output,
            "g8r simulation mismatch for sample {sample_index}:\n\
             IR:\n{ir_text}\nargs={args:?}\npir_output={pir_output:?}"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let case = generate_full_g8r_fuzz_case(data, "fuzz_g8r_evaluation")
        .unwrap_or_else(|error| panic!("{error}"));
    let source_fn = case
        .source_package
        .get_fn(&case.source_top)
        .expect("generated package should retain its top function");
    check_simulation_equivalence(
        &case.source_package,
        source_fn,
        &case.gate_fn,
        &case.source_ir,
    );
});
