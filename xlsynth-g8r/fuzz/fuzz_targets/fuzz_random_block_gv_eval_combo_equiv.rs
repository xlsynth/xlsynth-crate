// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially checks combinational block IR against Yosys-mapped gv-eval.

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::SeedableRng;
use xlsynth::IrValue;
use xlsynth_g8r::aig_serdes::emit_netlist::{
    NetlistPortStyle, emit_netlist_with_version_and_port_style,
};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_netlist_aig_with_liberty,
};
use xlsynth_g8r::verilog_version::VerilogVersion;
use xlsynth_g8r_fuzz::external_yosys::external_yosys_context;
use xlsynth_g8r_fuzz::random_block::{
    block_output_types, evaluate_block_outputs, flatten_value,
};
use xlsynth_pir::ir::Fn;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomFnOptions, RandomOperation,
    StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

const INPUT_SAMPLE_COUNT: usize = 16;

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
        max_input_ports: 4,
        max_output_ports: 4,
        max_registers: 0,
        allow_load_enable: false,
        allow_reset: false,
        function_options: RandomFnOptions {
            max_nodes: 32,
            max_bit_width: 8,
            allow_arbitrary_width_multiply: true,
            enabled_operations: operations,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    }
}

fn generate_inputs(block: &Fn, rng: &mut StdRng) -> Vec<IrValue> {
    block
        .params
        .iter()
        .map(|param| generate_uniform_value_with_rng(rng, &param.ty))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    // Missing Yosys or Liberty files are infrastructure conditions, not
    // properties of an individual fuzz sample.
    let Some(context) = external_yosys_context() else {
        return;
    };

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
    assert!(
        metadata.registers.is_empty(),
        "combinational-only generation emitted registers:\n{block_ir}"
    );

    let design = block_package_to_sequential_gate_fn(
        &generated.package,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap_or_else(|error| panic!("random block G8R lowering failed:\n{block_ir}\n{error}"));
    assert!(
        design.registers.is_empty() && design.clock.is_none(),
        "combinational block lowered to sequential G8R:\n{block_ir}"
    );
    let rtl = emit_netlist_with_version_and_port_style(
        &design,
        VerilogVersion::Verilog,
        NetlistPortStyle::PackedBits,
    )
    .unwrap_or_else(|error| panic!("random block RTL emission failed:\n{block_ir}\n{error}"));
    let mapped_gv = context
        .yosys
        .synthesize_verilog_to_gv(&rtl, &design.name)
        .unwrap_or_else(|error| {
        panic!("random block Yosys mapping failed:\nIR:\n{block_ir}\nRTL:\n{rtl}\n{error}")
    });
    let netlist_dir = tempfile::tempdir().expect("create temporary mapped-netlist directory");
    let netlist_path = netlist_dir.path().join("mapped.gv");
    std::fs::write(&netlist_path, &mapped_gv).expect("write temporary mapped netlist");
    let model = load_labeled_netlist_aig_with_liberty(
        &netlist_path,
        &context.liberty,
        &GvEvalOptions {
            module_name: Some(design.name.clone()),
            ..GvEvalOptions::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!(
            "random block gv-eval loading failed:\nIR:\n{block_ir}\nRTL:\n{rtl}\nGV:\n{mapped_gv}\n{error}"
        )
    });

    let expected_input_widths = design
        .inputs
        .iter()
        .map(|input| design.transition.inputs[input.index()].get_bit_count())
        .collect::<Vec<_>>();
    let actual_input_widths = model
        .gate_fn
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .collect::<Vec<_>>();
    assert_eq!(
        actual_input_widths, expected_input_widths,
        "Yosys changed combinational block input shape:\nIR:\n{block_ir}\nGV:\n{mapped_gv}"
    );
    let expected_output_widths = design
        .outputs
        .iter()
        .map(|output| design.transition.outputs[output.index()].get_bit_count())
        .collect::<Vec<_>>();
    let actual_output_widths = model
        .gate_fn
        .outputs
        .iter()
        .map(|output| output.get_bit_count())
        .collect::<Vec<_>>();
    assert_eq!(
        actual_output_widths, expected_output_widths,
        "Yosys changed combinational block output shape:\nIR:\n{block_ir}\nGV:\n{mapped_gv}"
    );

    let mut seed = [0_u8; 32];
    seed.copy_from_slice(blake3::hash(block_ir.as_bytes()).as_bytes());
    let mut rng = StdRng::from_seed(seed);
    let output_types = block_output_types(func, metadata);
    for sample_index in 0..INPUT_SAMPLE_COUNT {
        let inputs = generate_inputs(func, &mut rng);
        let input_bits = inputs
            .iter()
            .zip(&func.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        let expected_output_bits = evaluate_block_outputs(func, metadata, &inputs, &block_ir)
            .iter()
            .zip(&output_types)
            .map(|(value, ty)| flatten_value(value, ty))
            .collect::<Vec<_>>();
        let actual_output_bits = model.evaluate_bits(&input_bits).unwrap_or_else(|error| {
            panic!(
                "random block gv-eval failed at sample {sample_index}:\nIR:\n{block_ir}\nGV:\n{mapped_gv}\n{error}"
            )
        });
        assert_eq!(
            actual_output_bits, expected_output_bits,
            "random block gv-eval mismatch at sample {sample_index}:\nIR:\n{block_ir}\nRTL:\n{rtl}\nGV:\n{mapped_gv}"
        );
    }
});
