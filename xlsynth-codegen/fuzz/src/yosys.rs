// SPDX-License-Identifier: Apache-2.0

//! Shared external Yosys/Liberty oracles for public block codegen targets.

use xlsynth_pir::IrBits;
use std::collections::BTreeMap;

use xlsynth::external_tool::ToolError;

use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, LabeledNetlistAig, LabeledSequentialNetlistAig,
    load_labeled_netlist_aig_with_liberty, load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::yosys::{YosysInputLanguage, YosysMappingContext, YosysMappingKind};
use xlsynth_pir::ir::{BlockMetadata, Fn, Package};

use crate::emit;

/// Emitted SystemVerilog and its independently synthesized gate graph.
pub struct MappedCombinational {
    pub source: SequentialGateFn,
    pub mapped: LabeledNetlistAig,
    pub rtl: String,
    pub netlist: String,
}

/// Emitted synchronous SystemVerilog and its Liberty-backed mapped transition.
pub struct MappedSequential {
    pub source: SequentialGateFn,
    pub mapped: LabeledSequentialNetlistAig,
    pub rtl: String,
    pub netlist: String,
}

/// Maps emitted combinational SystemVerilog through external Yosys and Liberty.
pub fn map_combinational(
    package: &Package,
    context: &YosysMappingContext,
) -> Result<MappedCombinational, ToolError> {
    let ir = package.to_string();
    let rtl = emit(package, &BlockCodegenOptions::default());
    let source = block_package_to_sequential_gate_fn(package, two_valued_gatify_options())
        .unwrap_or_else(|error| panic!("independent block G8R lowering failed:\n{ir}\n{error}"));
    assert!(
        source.registers.is_empty(),
        "combinational mapping received registers:\n{ir}"
    );
    let netlist = context
        .yosys
        .synthesize_to_gv(
            &rtl,
            &source.name,
            YosysInputLanguage::SystemVerilog,
            YosysMappingKind::Combinational,
        )
        .map_err(|error| error.with_context(format!("IR:\n{ir}\nRTL:\n{rtl}")))?;
    let directory = tempfile::tempdir().expect("create mapped-netlist directory");
    let path = directory.path().join("mapped.gv");
    std::fs::write(&path, &netlist).expect("write temporary mapped netlist");
    let mapped = load_labeled_netlist_aig_with_liberty(
        &path,
        &context.liberty,
        &GvEvalOptions {
            module_name: Some(source.name.clone()),
            ..GvEvalOptions::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!("mapped combinational netlist could not load:\nIR:\n{ir}\nRTL:\n{rtl}\nGV:\n{netlist}\n{error}")
    });
    Ok(MappedCombinational {
        source,
        mapped,
        rtl,
        netlist,
    })
}

/// Maps emitted synchronous SystemVerilog through external Yosys and Liberty.
pub fn map_sequential(
    package: &Package,
    context: &YosysMappingContext,
) -> Result<MappedSequential, ToolError> {
    let ir = package.to_string();
    let rtl = emit(package, &BlockCodegenOptions::default());
    let source = block_package_to_sequential_gate_fn(package, two_valued_gatify_options())
        .unwrap_or_else(|error| panic!("independent block G8R lowering failed:\n{ir}\n{error}"));
    assert!(
        !source.registers.is_empty(),
        "sequential mapping requires registers:\n{ir}"
    );
    let netlist = context
        .yosys
        .synthesize_sequential_verilog_to_gv_detailed(&rtl, &source.name)
        .map_err(|error| error.with_context(format!("IR:\n{ir}\nRTL:\n{rtl}")))?;
    let directory = tempfile::tempdir().expect("create mapped-netlist directory");
    let path = directory.path().join("mapped.gv");
    std::fs::write(&path, &netlist).expect("write temporary mapped netlist");
    let mapped = load_labeled_sequential_netlist_aig_with_liberty(
        &path,
        &context.liberty,
        &GvEvalOptions {
            module_name: Some(source.name.clone()),
            clock_port_name: source.clock.as_ref().map(|clock| clock.name.clone()),
        },
    )
    .unwrap_or_else(|error| {
        panic!("mapped sequential netlist could not load:\nIR:\n{ir}\nRTL:\n{rtl}\nGV:\n{netlist}\n{error}")
    });
    Ok(MappedSequential {
        source,
        mapped,
        rtl,
        netlist,
    })
}

/// Enables standard XLS gate lowering only for concrete two-valued oracles.
fn two_valued_gatify_options() -> GatifyOptions {
    GatifyOptions {
        unsafe_gatify_gate_operation: true,
        ..GatifyOptions::all_opts_disabled()
    }
}

/// Reorders source port values to match mapped combinational input ordering.
pub fn map_combinational_inputs(
    model: &LabeledNetlistAig,
    block: &Fn,
    values: &[IrBits],
) -> Vec<IrBits> {
    let by_name = block
        .params
        .iter()
        .zip(values)
        .map(|(param, value)| (param.name.as_str(), value))
        .collect::<BTreeMap<_, _>>();
    model
        .gate_fn
        .inputs
        .iter()
        .map(|input| {
            (*by_name
                .get(input.name.as_str())
                .unwrap_or_else(|| panic!("mapped input `{}` is missing", input.name)))
            .clone()
        })
        .collect()
}

/// Reorders source values to match mapped sequential external-input ordering.
pub fn map_sequential_inputs(
    design: &SequentialGateFn,
    block: &Fn,
    values: &[IrBits],
) -> Vec<IrBits> {
    let by_name = block
        .params
        .iter()
        .zip(values)
        .map(|(param, value)| (param.name.as_str(), value))
        .collect::<BTreeMap<_, _>>();
    design
        .inputs
        .iter()
        .map(|input| {
            let name = design.transition.inputs[input.index()].name.as_str();
            (*by_name
                .get(name)
                .unwrap_or_else(|| panic!("mapped sequential input `{name}` is missing")))
            .clone()
        })
        .collect()
}

/// Reorders source values to match mapped sequential external-output ordering.
pub fn map_sequential_outputs(
    design: &SequentialGateFn,
    metadata: &BlockMetadata,
    values: &[IrBits],
) -> Vec<IrBits> {
    let by_name = metadata
        .output_names
        .iter()
        .zip(values)
        .map(|(name, value)| (name.as_str(), value))
        .collect::<BTreeMap<_, _>>();
    design
        .outputs
        .iter()
        .map(|output| {
            let name = design.transition.outputs[output.index()].name.as_str();
            (*by_name
                .get(name)
                .unwrap_or_else(|| panic!("mapped sequential output `{name}` is missing")))
            .clone()
        })
        .collect()
}
