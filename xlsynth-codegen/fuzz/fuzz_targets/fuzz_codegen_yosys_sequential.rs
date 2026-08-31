// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially checks post-reset behavior after sequential Yosys mapping.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::yosys::{map_sequential, map_sequential_inputs, map_sequential_outputs};
use xlsynth_codegen_fuzz::{
    CYCLE_COUNT, Profile, deterministic_rng, generate, generate_cycle_inputs, top_block,
};
use xlsynth_g8r::aig_sim::sequential::{self, SequentialState};
use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;
use xlsynth_g8r_fuzz::random_block::{block_output_types, evaluate_block_cycle, flatten_value};
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::YosysSequential) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let context = required_external_yosys_context()
        .expect("startup validated Yosys/Liberty setup");
    let mut options = Profile::StockXls.block_options(false, true);
    options.allow_zero_width_ports_and_registers = false;
    options.function_options.allow_zero_width_bits = false;
    options.require_reset_on_all_registers = true;
    options.function_options.max_nodes = 24;
    options.function_options.max_bit_width = 8;
    let package = generate(data, &options);
    let ir = package.to_string();
    let mapped = xlsynth_codegen_fuzz::fuzz_tool!(map_sequential(&package, context));
    let (block, metadata) = top_block(&package);
    let output_types = block_output_types(block, metadata);
    let design = &mapped.mapped.sequential_gate_fn;
    let mut rng = deterministic_rng(&ir);
    let mut state = metadata
        .registers
        .iter()
        .map(|register| generate_uniform_value_with_rng(&mut rng, &register.ty))
        .collect::<Vec<_>>();
    let mut stimuli = Vec::with_capacity(CYCLE_COUNT);
    let mut expected_outputs = Vec::with_capacity(CYCLE_COUNT);
    for cycle in 0..CYCLE_COUNT {
        let inputs = generate_cycle_inputs(block, metadata, &mut rng, cycle, true);
        let bits = inputs
            .iter()
            .zip(&block.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        stimuli.push(map_sequential_inputs(design, block, &bits));
        let (outputs, next_state) = evaluate_block_cycle(block, metadata, &inputs, &state, &ir);
        let output_bits = outputs
            .iter()
            .zip(output_types.iter())
            .map(|(value, ty)| flatten_value(value, ty))
            .collect::<Vec<_>>();
        expected_outputs.push(map_sequential_outputs(design, metadata, &output_bits));
        state = next_state;
    }
    let trace = sequential::simulate(design, &stimuli, SequentialState::all_zeros(design))
        .unwrap_or_else(|error| {
            panic!(
                "mapped sequential simulation failed:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}\n{error}",
                mapped.rtl, mapped.netlist
            )
        });
    for (cycle, (actual, expected)) in trace
        .external_outputs()
        .iter()
        .zip(expected_outputs.iter())
        .enumerate()
        .skip(1)
    {
        assert_eq!(
            actual, expected,
            "mapped sequential outputs differ after reset at cycle {cycle}:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}",
            mapped.rtl, mapped.netlist
        );
    }
});
