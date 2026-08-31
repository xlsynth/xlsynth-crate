// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially evaluates native block SystemVerilog after Yosys mapping.

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::yosys::{map_combinational, map_combinational_inputs};
use xlsynth_codegen_fuzz::{
    INPUT_SAMPLE_COUNT, Profile, deterministic_rng, generate, generate_inputs, top_block,
};
use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;
use xlsynth_g8r_fuzz::random_block::{block_output_types, evaluate_block_outputs, flatten_value};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::YosysCombinational) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let context = required_external_yosys_context()
        .expect("startup validated Yosys/Liberty setup");
    let mut options = Profile::StockXls.block_options(false, false);
    options.allow_zero_width_ports_and_registers = false;
    options.function_options.allow_zero_width_bits = false;
    options.function_options.max_nodes = 20;
    options.function_options.max_bit_width = 8;
    let package = generate(data, &options);
    let ir = package.to_string();
    let mapped = xlsynth_codegen_fuzz::fuzz_tool!(map_combinational(&package, context));
    let (block, metadata) = top_block(&package);
    let output_types = block_output_types(block, metadata);
    let mut rng = deterministic_rng(&ir);
    for sample in 0..INPUT_SAMPLE_COUNT {
        let inputs = generate_inputs(block, &mut rng);
        let source_inputs = inputs
            .iter()
            .zip(&block.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        let mapped_inputs = map_combinational_inputs(&mapped.mapped, block, &source_inputs);
        let expected = evaluate_block_outputs(block, metadata, &inputs, &ir)
            .iter()
            .zip(output_types.iter())
            .zip(metadata.output_names.iter())
            .map(|((value, ty), name)| (name.as_str(), flatten_value(value, ty)))
            .collect::<BTreeMap<_, _>>();
        let actual = mapped
            .mapped
            .evaluate_bits(&mapped_inputs)
            .unwrap_or_else(|error| {
                panic!("mapped combinational evaluation failed at sample {sample}:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}\n{error}", mapped.rtl, mapped.netlist)
            });
        for (port, value) in mapped.mapped.gate_fn.outputs.iter().zip(actual) {
            assert_eq!(
                value,
                *expected
                    .get(port.name.as_str())
                    .unwrap_or_else(|| panic!("mapped output `{}` is missing", port.name)),
                "mapped output `{}` differs at sample {sample}:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}",
                port.name,
                mapped.rtl,
                mapped.netlist
            );
        }
    }
});
