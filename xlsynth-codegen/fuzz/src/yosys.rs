// SPDX-License-Identifier: Apache-2.0

//! Shared external Yosys/Liberty oracles for public block codegen targets.

use std::collections::BTreeMap;
use std::process::Command;
use std::time::Duration;

use xlsynth::external_tool::{ToolError, run_checked_detailed};

use xlsynth::IrBits;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, LabeledNetlistAig, LabeledSequentialNetlistAig,
    load_labeled_netlist_aig_with_liberty, load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r_fuzz::external_yosys::ExternalYosysContext;
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
    context: &ExternalYosysContext,
) -> Result<MappedCombinational, ToolError> {
    let ir = package.to_string();
    let rtl = emit(package, &BlockCodegenOptions::default());
    let source = block_package_to_sequential_gate_fn(package, two_valued_gatify_options())
        .unwrap_or_else(|error| panic!("independent block G8R lowering failed:\n{ir}\n{error}"));
    assert!(
        source.registers.is_empty(),
        "combinational mapping received registers:\n{ir}"
    );
    let netlist = synthesize_combinational_system_verilog(&rtl, &source.name, context)
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

/// Uses Yosys's SystemVerilog frontend for a combinational technology map.
fn synthesize_combinational_system_verilog(
    rtl: &str,
    top: &str,
    context: &ExternalYosysContext,
) -> Result<String, ToolError> {
    let directory =
        tempfile::tempdir().map_err(|error| format!("create Yosys directory: {error}"))?;
    let input = directory.path().join("dut.sv");
    let mapped = directory.path().join("mapped.gv");
    std::fs::write(&input, rtl).map_err(|error| format!("write Yosys SystemVerilog: {error}"))?;

    let mut read_libraries = String::new();
    let mut abc_libraries = String::new();
    for path in context.yosys.liberty_files().paths() {
        let quoted = path
            .display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        read_libraries.push_str(&format!("read_liberty -lib \"{quoted}\"\n"));
        abc_libraries.push_str(&format!(" -liberty \"{quoted}\""));
    }
    let program = format!(
        "{read_libraries}read_verilog -sv dut.sv\nhierarchy -check -top {top}\nproc\nflatten\nopt\ntechmap\nopt\nabc{abc_libraries}\nclean -purge\nwrite_verilog -noattr mapped.gv\n"
    );
    run_checked_detailed(
        Command::new(context.yosys.yosys_path())
            .current_dir(directory.path())
            .args(["-Q", "-p", &program]),
        directory.path(),
        "yosys-mapping",
        Duration::from_secs(120),
    )
    .map_err(|error| error.with_context(format!("Yosys program:\n{program}")))?;

    std::fs::read_to_string(&mapped).map_err(|error| {
        ToolError::failure(format!("read mapped SystemVerilog gate netlist: {error}"))
    })
}

/// Maps emitted synchronous SystemVerilog through external Yosys and Liberty.
pub fn map_sequential(
    package: &Package,
    context: &ExternalYosysContext,
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

#[cfg(test)]
mod tests {
    use xlsynth_g8r::prove_gate_fn_equiv_sat::{
        EquivResult, GateFormalBackend, prove_gate_fn_equiv_with_backend_and_options,
    };
    use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;
    use xlsynth_g8r_fuzz::fuzz_gate_formal_options;

    use crate::{parse_reference, references};

    use super::{map_combinational, map_sequential, synthesize_combinational_system_verilog};

    #[test]
    fn narrow_slice_update_is_formally_equivalent_after_yosys_mapping() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(references::NARROW_SLICE_UPDATE);
        let mapped = map_combinational(&package, context).unwrap();
        let mut source = mapped.source.transition;
        let mut target = mapped.mapped.gate_fn;
        for design in [&mut source, &mut target] {
            design.inputs.sort_by(|a, b| a.name.cmp(&b.name));
            design.outputs.sort_by(|a, b| a.name.cmp(&b.name));
        }
        assert_eq!(
            prove_gate_fn_equiv_with_backend_and_options(
                &source,
                &target,
                GateFormalBackend::Cadical,
                fuzz_gate_formal_options(),
            )
            .unwrap(),
            EquivResult::Proved,
        );
    }

    #[test]
    fn public_combinational_reference_maps() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(references::COMBINATIONAL);
        let mapped = map_combinational(&package, context).unwrap();
        assert_eq!(mapped.source.name, "arithmetic");
        assert_eq!(mapped.mapped.gate_fn.outputs.len(), 2);
    }

    #[test]
    fn public_sequential_reference_maps() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(references::SEQUENTIAL);
        let mapped = map_sequential(&package, context).unwrap();
        assert_eq!(mapped.source.name, "accumulator");
        assert!(!mapped.mapped.sequential_gate_fn.registers.is_empty());
    }

    #[test]
    fn public_combinational_mapper_accepts_system_verilog_width_casts() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let rtl = r#"module public_cast(input wire [3:0] value, output wire result);
  assign result = 1'(value >> 1);
endmodule
"#;
        let mapped = synthesize_combinational_system_verilog(rtl, "public_cast", context)
            .expect("Yosys combinational mapper must enable SystemVerilog parsing");
        assert!(mapped.contains("module public_cast("));
    }
}
