// SPDX-License-Identifier: Apache-2.0

//! Icarus oracles with block-IR interfaces and public hierarchy/foreign models.

use std::process::Command;
use std::time::Duration;

use rand::Rng;
use xlsynth::external_tool::{ToolError, run_checked_detailed};
use xlsynth_g8r_fuzz::random_block::block_output_types;
use xlsynth_pir::ir::{Package, Type};
use xlsynth_test_helpers::rtl_sim::{Icarus, Interface, Port, StateSignal, identifier};

use crate::semantics::Trace;
use crate::{INPUT_SAMPLE_COUNT, deterministic_rng, top_block};

#[derive(Clone, Copy)]
pub enum StateLayout {
    Packed,
    StockUnpacked,
}

/// Derives testbench bindings from block IR without parsing the emitted RTL.
pub fn interface(package: &Package, module_name: Option<&str>, layout: StateLayout) -> Interface {
    let (block, metadata) = top_block(package);
    let types = block_output_types(block, metadata);
    Interface {
        module: module_name.unwrap_or(&block.name).to_owned(),
        inputs: block
            .params
            .iter()
            .map(|p| Port {
                name: p.name.clone(),
                width: p.ty.bit_count(),
            })
            .collect(),
        outputs: metadata
            .output_names
            .iter()
            .zip(types)
            .map(|(name, ty)| Port {
                name: name.clone(),
                width: ty.bit_count(),
            })
            .collect(),
        clock: metadata.clock_port_name.clone(),
        state: metadata
            .registers
            .iter()
            .filter(|r| r.ty.bit_count() != 0)
            .map(|r| {
                let name = format!("dut.{}", identifier(&r.name).unwrap());
                StateSignal {
                    name: r.name.clone(),
                    width: r.ty.bit_count(),
                    expression: match layout {
                        StateLayout::Packed => name,
                        StateLayout::StockUnpacked => stock_state_lvalue(&name, &r.ty),
                    },
                }
            })
            .collect(),
    }
}

/// Checks concrete outputs and next state with a real event-driven simulator.
pub fn assert_rtl_semantics(
    package: &Package,
    rtl: &str,
    module_name: Option<&str>,
    layout: StateLayout,
) -> Result<(), ToolError> {
    let trace = Trace::for_package(package);
    assert_rtl_trace(package, rtl, module_name, layout, &trace)
}

/// Replays exactly the same precomputed inputs and expectations as other
/// oracles.
pub fn assert_rtl_trace(
    package: &Package,
    rtl: &str,
    module_name: Option<&str>,
    layout: StateLayout,
    trace: &Trace,
) -> Result<(), ToolError> {
    let ir = package.to_string();
    let mut simulator = Icarus::new(rtl, interface(package, module_name, layout))
        .map_err(|e| e.with_context(format!("Icarus compilation\nIR:\n{ir}")))?;
    simulator
        .set_state(&trace.initial_state)
        .map_err(|e| e.with_context(format!("initialize state\nIR:\n{ir}\nRTL:\n{rtl}")))?;
    for (index, sample) in trace.samples.iter().enumerate() {
        let actual = simulator.evaluate(&sample.inputs).map_err(|e| {
            e.with_context(format!("Icarus sample {index}\nIR:\n{ir}\nRTL:\n{rtl}"))
        })?;
        assert_eq!(
            actual.outputs, sample.outputs,
            "outputs, sample {index}:\nIR:\n{ir}\nRTL:\n{rtl}\ninputs: {:?}",
            sample.inputs
        );
        if let Some(next_state) = &sample.next_state {
            let after = simulator.cycle(&sample.inputs).map_err(|e| {
                e.with_context(format!("Icarus edge {index}\nIR:\n{ir}\nRTL:\n{rtl}"))
            })?;
            assert_eq!(
                &after.state, next_state,
                "state, sample {index}:\nIR:\n{ir}\nRTL:\n{rtl}\ninputs: {:?}",
                sample.inputs
            );
        }
    }
    Ok(())
}
fn stock_state_lvalue(name: &str, ty: &Type) -> String {
    if let Type::Array(array) = ty {
        format!(
            "{{{}}}",
            (0..array.element_count)
                .rev()
                .map(|i| stock_state_lvalue(&format!("{name}[{i}]"), &array.element_type))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        name.to_owned()
    }
}

/// Verifies two independently connected child instances by concrete simulation.
pub fn assert_hierarchy_semantics(ir: &str, rtl: &str, width: usize) -> Result<(), ToolError> {
    let mut testbench = format!(
        "`timescale 1ns/1ps\nmodule public_fuzz_testbench;\n  reg [{last}:0] left;\n  reg [{last}:0] right;\n  wire [{last}:0] result;\n  parent dut (.left(left), .right(right), .result(result));\n  initial begin\n",
        last = width - 1
    );
    let mask = (1_u64 << width) - 1;
    let mut rng = deterministic_rng(ir);
    for sample in 0..INPUT_SAMPLE_COUNT {
        let left = rng.gen_range(0..=mask);
        let right = rng.gen_range(0..=mask);
        let expected = ((left.wrapping_add(1) & mask) ^ (right.wrapping_add(1) & mask)) & mask;
        testbench.push_str(&format!(
            "    left = {width}'h{left:x}; right = {width}'h{right:x}; #1;\n    if (result !== {width}'h{expected:x}) $fatal(1, \"hierarchy sample {sample} mismatch: actual=%h expected={expected:x}\", result);\n"
        ));
    }
    testbench.push_str("    $finish;\n  end\nendmodule\n");
    simulate_public_design(ir, rtl, &testbench)
}

/// Verifies a foreign instance using its independently authored identity model.
pub fn assert_extern_semantics(
    ir: &str,
    rtl: &str,
    width: usize,
    external_module: &str,
) -> Result<(), ToolError> {
    let leaf = format!(
        "\nmodule {external_module}(input wire [{last}:0] x, output wire [{last}:0] y);\n  assign y = x;\nendmodule\n",
        last = width - 1
    );
    let design = format!("{rtl}{leaf}");
    let mut testbench = format!(
        "`timescale 1ns/1ps\nmodule public_fuzz_testbench;\n  reg [{last}:0] x;\n  wire [{last}:0] result;\n  wrapper dut (.x(x), .result(result));\n  initial begin\n",
        last = width - 1
    );
    let mask = (1_u64 << width) - 1;
    let mut rng = deterministic_rng(ir);
    for sample in 0..INPUT_SAMPLE_COUNT {
        let input = rng.gen_range(0..=mask);
        testbench.push_str(&format!(
            "    x = {width}'h{input:x}; #1;\n    if (result !== {width}'h{input:x}) $fatal(1, \"external instance sample {sample} mismatch: actual=%h expected={input:x}\", result);\n"
        ));
    }
    testbench.push_str("    $finish;\n  end\nendmodule\n");
    simulate_public_design(ir, &design, &testbench)
}

/// Compiles and executes a self-checking SystemVerilog reference testbench.
fn simulate_public_design(ir: &str, rtl: &str, testbench: &str) -> Result<(), ToolError> {
    let directory = tempfile::tempdir().expect("create public Icarus simulation directory");
    let design_path = directory.path().join("public_design.sv");
    let testbench_path = directory.path().join("public_testbench.sv");
    let output_path = directory.path().join("public_simulator");
    std::fs::write(&design_path, rtl).expect("write generated public SystemVerilog");
    std::fs::write(&testbench_path, testbench).expect("write public simulation testbench");

    run_checked_detailed(
        Command::new(
            xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
                .expect("Icarus compiler/runtime required")
                .iverilog_path(),
        )
        .args(["-g2012", "-s", "public_fuzz_testbench", "-o"])
        .arg(&output_path)
        .arg(&design_path)
        .arg(&testbench_path),
        directory.path(),
        "compile",
        Duration::from_secs(60),
    )
    .map_err(|e| {
        e.with_context(format!(
            "public hierarchy/external compilation\nIR:\n{ir}\nRTL:\n{rtl}\ntestbench:\n{testbench}"
        ))
    })?;

    run_checked_detailed(
        Command::new(
            xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
                .expect("Icarus compiler/runtime required")
                .vvp_path(),
        )
        .arg(&output_path),
        directory.path(),
        "simulate",
        Duration::from_secs(60),
    )
    .map_err(|e| {
        e.with_context(format!(
            "public hierarchy/external simulation\nIR:\n{ir}\nRTL:\n{rtl}\ntestbench:\n{testbench}"
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use xlsynth_codegen::BlockCodegenOptions;

    use crate::{emit, parse_reference, references};

    use super::{assert_extern_semantics, assert_hierarchy_semantics};

    #[test]
    fn public_hierarchy_is_semantically_correct() {
        let package = parse_reference(references::HIERARCHY);
        let rtl = emit(&package, &BlockCodegenOptions::default());
        assert_hierarchy_semantics(references::HIERARCHY, &rtl, 8).unwrap();
    }

    #[test]
    fn public_extern_is_semantically_correct() {
        let package = parse_reference(references::EXTERN);
        let rtl = emit(&package, &BlockCodegenOptions::default());
        assert_extern_semantics(references::EXTERN, &rtl, 8, "external_cell").unwrap();
    }
}
