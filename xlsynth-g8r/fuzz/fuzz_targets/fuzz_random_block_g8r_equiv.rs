// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially checks generated block IR against sequential G8R lowering.

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::IrValue;
use xlsynth_g8r::aig_sim::sequential::{self, SequentialState};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r_fuzz::random_block::{
    block_output_types, evaluate_block_cycle, flatten_value,
};
use xlsynth_pir::ir::{BlockMetadata, Fn};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomFnOptions, RandomOperation,
    RandomBlockResetTiming, StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

const CYCLE_COUNT: usize = 32;

fn fuzz_block_options() -> RandomBlockOptions {
    let operations = OperationSet::new(
        OperationSet::all_supported()
            .iter()
            .filter(|operation| {
                !matches!(
                    operation,
                    RandomOperation::Umulp | RandomOperation::Smulp
                )
            }),
    );
    RandomBlockOptions {
        max_input_ports: 6,
        max_output_ports: 4,
        max_registers: 4,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_nodes: 64,
            max_bit_width: 16,
            allow_arbitrary_width_multiply: true,
            enabled_operations: operations,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    }
}

fn generate_initial_state(metadata: &BlockMetadata, rng: &mut StdRng) -> Vec<IrValue> {
    metadata
        .registers
        .iter()
        .map(|register| {
            register
                .reset_value
                .clone()
                .unwrap_or_else(|| generate_uniform_value_with_rng(rng, &register.ty))
        })
        .collect()
}

fn generate_cycle_inputs(
    block: &Fn,
    metadata: &BlockMetadata,
    rng: &mut StdRng,
    cycle: usize,
) -> Vec<IrValue> {
    block
        .params
        .iter()
        .map(|param| {
            if let Some(reset) = metadata.reset.as_ref()
                && param.name == reset.port_name
            {
                let asserted = if cycle == 0 {
                    rng.gen_bool(0.5)
                } else {
                    rng.gen_ratio(1, 10)
                };
                let signal_high = if reset.active_low {
                    !asserted
                } else {
                    asserted
                };
                return IrValue::make_ubits(1, u64::from(signal_high))
                    .expect("bits[1] reset input should construct");
            }
            generate_uniform_value_with_rng(rng, &param.ty)
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let mut entropy = DepletableBytes::new(data);
    let generated = generate_block_package(
        &mut entropy,
        &fuzz_block_options(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed random block options should construct a valid package");
    let block_ir = generated.package.to_string();
    let block = generated
        .package
        .get_top_block()
        .expect("generated package should have a top block");
    let xlsynth_pir::ir::PackageMember::Block { func, metadata } = block else {
        unreachable!("generated package top should be a block");
    };

    if metadata
        .reset
        .as_ref()
        .is_some_and(|reset| reset.asynchronous)
    {
        panic!("synchronous-only block generation emitted an asynchronous reset:\n{block_ir}");
    }

    let design = block_package_to_sequential_gate_fn(
        &generated.package,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap_or_else(|error| panic!("random block G8R lowering failed:\n{block_ir}\n{error}"));
    let mut seed = [0_u8; 32];
    seed.copy_from_slice(blake3::hash(block_ir.as_bytes()).as_bytes());
    let mut rng = StdRng::from_seed(seed);
    let mut block_state = generate_initial_state(metadata, &mut rng);
    let initial_g8r_state = block_state
        .iter()
        .zip(&metadata.registers)
        .map(|(value, register)| flatten_value(value, &register.ty))
        .collect();
    let mut g8r_state = SequentialState::from_register_values(&design, initial_g8r_state)
        .expect("generated initial register state should match lowered G8R");
    let output_types = block_output_types(func, metadata);

    for cycle in 0..CYCLE_COUNT {
        let inputs = generate_cycle_inputs(func, metadata, &mut rng, cycle);
        let g8r_inputs = inputs
            .iter()
            .zip(&func.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        let (expected_outputs, next_block_state) =
            evaluate_block_cycle(func, metadata, &inputs, &block_state, &block_ir);
        let expected_output_bits = expected_outputs
            .iter()
            .zip(&output_types)
            .map(|(value, ty)| flatten_value(value, ty))
            .collect::<Vec<_>>();
        let trace = sequential::simulate(&design, &[g8r_inputs], g8r_state)
            .unwrap_or_else(|error| panic!("random block G8R simulation failed:\n{block_ir}\n{error}"));
        assert_eq!(
            trace.external_outputs()[0],
            expected_output_bits,
            "random block output mismatch at cycle {cycle}:\n{block_ir}"
        );
        let expected_state_bits = next_block_state
            .iter()
            .zip(&metadata.registers)
            .map(|(value, register)| flatten_value(value, &register.ty))
            .collect::<Vec<_>>();
        assert_eq!(
            trace.final_state().values(),
            expected_state_bits,
            "random block register-state mismatch at cycle {cycle}:\n{block_ir}"
        );
        block_state = next_block_state;
        g8r_state = trace.final_state().clone();
    }
});
