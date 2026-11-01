// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use xlsynth_vast::{Expr, LiteralFormat, LogicRef, VastDataType, VastFile, VastModule};

use super::layout::PipelineLayout;
use super::names::{
    flop_reg_name, slice_wire_name, stage_instance_name, stage_output_wire_name, valid_reg_name,
};

/// Identifies the combinational module and result port for a pipeline stage.
#[derive(Clone, Debug)]
pub(crate) struct StageModuleInterface {
    pub(crate) name: String,
    pub(crate) output_port: String,
}

/// Describes the packed Verilog wrapper and the modules it instantiates.
pub struct PipelineConfig<'a> {
    pub top_module_name: String,
    pub clk_port_name: String,

    pub stage_modules: Vec<StageModuleInterface>,

    pub layout: &'a PipelineLayout,

    pub input_valid_signal: Option<&'a str>,
    pub output_valid_signal: Option<&'a str>,
    pub reset_signal: Option<&'a str>,
    pub reset_active_low: bool,
}

/// Tracks the current data and valid signals on either side of a flop layer.
struct NetBundle {
    name_to_ref: HashMap<String, (LogicRef, VastDataType)>,
    valid_signal: Option<LogicRef>,
}

impl std::fmt::Debug for NetBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_to_ref_str = self
            .name_to_ref
            .iter()
            .map(|(name, (logic_ref, _vast_type))| format!("{:?} => {:?}", name, logic_ref))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "NetBundle {{ name_to_ref: [{}], valid_signal: {:?} }}",
            name_to_ref_str, self.valid_signal
        )
    }
}

/// Creates a layer of flops using the input combinational signals given in
/// `current_inputs`. Returns the new `current_inputs` bundle.
fn make_flop_layer(
    file: &mut VastFile,
    outer_module: VastModule,
    posedge_clk: &Expr,
    bit_type: &VastDataType,
    current_inputs: NetBundle,
    layer: usize,
    reset_signal: Option<LogicRef>,
    reset_active_low: bool,
) -> NetBundle {
    // First create register declarations so they appear before procedural
    // blocks.
    let mut new_name_to_ref = HashMap::new();

    // Collect and sort for deterministic emission order.
    let mut entries: Vec<(String, (LogicRef, VastDataType))> =
        current_inputs.name_to_ref.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    #[derive(PartialEq)]
    enum IsValidReg {
        Yes,
        No,
    }

    // Vector of tuples used later to emit assignments.
    let mut assign_info: Vec<(LogicRef, LogicRef, IsValidReg)> = Vec::new();

    for (name, (logic_ref, data_type)) in &entries {
        let reg = file
            .add_reg(outer_module, &flop_reg_name(layer, name), data_type)
            .unwrap();
        new_name_to_ref.insert(name.clone(), (reg, *data_type));
        assign_info.push((reg, *logic_ref, IsValidReg::No));
    }

    // Handle valid signal register declaration (if any) before always.
    let flopped_valid_signal = if let Some(valid_signal) = current_inputs.valid_signal {
        let reg = file
            .add_reg(outer_module, &valid_reg_name(layer), bit_type)
            .unwrap();
        assign_info.push((reg, valid_signal, IsValidReg::Yes));
        Some(reg)
    } else {
        None
    };

    // Prepare zero literal once if we need reset gating.
    let zero_expr = file
        .make_literal("bits[1]:0", &LiteralFormat::Binary)
        .unwrap();

    // Now create the `always` block and emit assignments into it.
    //
    // Note that we guard this so that we don't emit an empty `always` block if
    // there are no assignments to perform within it.
    if !assign_info.is_empty() {
        let always = file.add_always_at(outer_module, &[posedge_clk]).unwrap();
        let sb = file.statement_block(always);

        for (reg, src_logic, is_valid_reg) in assign_info {
            let rhs = if is_valid_reg == IsValidReg::No {
                // Data regs: include ternary guard if a valid signal exists.
                if let Some(ref valid_signal) = current_inputs.valid_signal {
                    file.make_ternary(
                        &valid_signal.to_expr(),
                        &src_logic.to_expr(),
                        &reg.to_expr(),
                    )
                } else {
                    src_logic.to_expr()
                }
            } else {
                // The valid reg itself just captures the signal directly, but
                // respect reset gating.
                if let Some(ref rst) = reset_signal {
                    if reset_active_low {
                        file.make_ternary(&rst.to_expr(), &src_logic.to_expr(), &zero_expr)
                    } else {
                        file.make_ternary(&rst.to_expr(), &zero_expr, &src_logic.to_expr())
                    }
                } else {
                    src_logic.to_expr()
                }
            };
            file.block_add_nonblocking_assignment(sb, &reg.to_expr(), &rhs);
        }
    }

    NetBundle {
        name_to_ref: new_name_to_ref,
        valid_signal: flopped_valid_signal,
    }
}

/// Creates I/O ports on the outer (encapsulating) module.
fn create_outer_io_ports(
    file: &mut VastFile,
    outer_module: VastModule,
    config: &PipelineConfig,
) -> (NetBundle, LogicRef) {
    log::debug!("Creating outer I/O ports");

    let mut name_to_ref = HashMap::new();
    for port in &config.layout.stage_inputs[0].ports {
        let vast_type = file.make_bit_vector_type(port.bits.len() as i64, false);
        let outer_module_input = file.add_input(outer_module, &port.name, &vast_type);
        name_to_ref.insert(port.name.clone(), (outer_module_input, vast_type));
    }

    // A LogicRef that represents the input valid signal coming combinationally
    // into the outer module.
    let bit_type = file.make_bit_vector_type(1, false);
    let outer_module_input_valid_signal: Option<LogicRef> =
        if let Some(ref input_valid_signal_name) = config.input_valid_signal {
            let outer_module_input_valid_signal =
                file.add_input(outer_module, input_valid_signal_name, &bit_type);
            Some(outer_module_input_valid_signal)
        } else {
            None
        };

    let output = &config.layout.output.ports[0];
    let vast_type = file.make_bit_vector_type(output.bits.len() as i64, false);
    let output_port = file.add_output(outer_module, &output.name, &vast_type);
    (
        NetBundle {
            name_to_ref,
            valid_signal: outer_module_input_valid_signal,
        },
        output_port,
    )
}

/// Emits the pipeline wrapper and its stage instances from the resolved layout.
pub fn build_pipeline(
    file: &mut VastFile,
    config: &PipelineConfig,
) -> Result<String, xlsynth::XlsynthError> {
    if config.stage_modules.is_empty() {
        return Err(xlsynth::XlsynthError(
            "Cannot build pipeline: no stage modules provided".to_string(),
        ));
    }

    let outer_module: VastModule = file.add_module(&config.top_module_name);

    let bit_type = file.make_bit_vector_type(1, false);
    let clk_port = file.add_input(outer_module, &config.clk_port_name, &bit_type);
    // Optional synchronous reset input comes immediately after clk.
    let reset_port = if let Some(ref rst_name) = config.reset_signal {
        Some(file.add_input(outer_module, rst_name, &bit_type))
    } else {
        None
    };
    let posedge_clk = file.make_pos_edge(&clk_port.to_expr());

    let (mut current_inputs, output_port) = create_outer_io_ports(file, outer_module, config);
    log::debug!("starting with current_inputs: {:?}", current_inputs);

    if let Some(layer) = config.layout.stage_inputs[0].register_layer {
        current_inputs = make_flop_layer(
            file,
            outer_module,
            &posedge_clk,
            &bit_type,
            current_inputs,
            layer,
            reset_port,
            config.reset_active_low,
        );
    }

    for (i, stage_module) in config.stage_modules.iter().enumerate() {
        log::debug!(
            "building stage {}; module name: {}; current_inputs: {:?}",
            i,
            stage_module.name,
            current_inputs
        );

        let destination = config.layout.after_stage(i);
        let stage_output_name = stage_output_wire_name(i);
        let stage_output_type = file.make_bit_vector_type(destination.width() as i64, false);
        let stage_output_wire = file.add_wire(outer_module, &stage_output_name, &stage_output_type);

        // Instantiate the stage module with the current inputs.
        let mut stage_port_names = Vec::new();
        let mut stage_expressions = Vec::new();
        for port in &config.layout.stage_inputs[i].ports {
            stage_port_names.push(port.name.as_str());
            let (input, _) = current_inputs.name_to_ref[&port.name];
            stage_expressions.push(input.to_expr());
        }
        stage_port_names.push(&stage_module.output_port);
        stage_expressions.push(stage_output_wire.to_expr());
        let stage_expression_refs = stage_expressions.iter().map(Some).collect::<Vec<_>>();

        // Instantiate the combinational stage module.
        let instantiation = file.make_instantiation(
            &stage_module.name,
            &stage_instance_name(i),
            &[],
            &[],
            &stage_port_names,
            &stage_expression_refs,
        );
        file.add_member_instantiation(outer_module, instantiation);

        // If the next stage wants more than one signal (i.e. it has arity > 1),
        // we break out the signals by slicing here.
        current_inputs.name_to_ref = if destination.ports.len() > 1 {
            let layer = destination
                .register_layer
                .expect("interstage register layer");
            let mut name_to_ref = HashMap::new();
            for port in &destination.ports {
                let vast_type = file.make_bit_vector_type(port.bits.len() as i64, false);
                let wire_name = slice_wire_name(layer, &port.name);
                let wire = file.add_wire(outer_module, &wire_name, &vast_type);
                let slice_expr = file
                    .make_slice(
                        &stage_output_wire.to_indexable_expr(),
                        (port.bits.end - 1) as i64,
                        port.bits.start as i64,
                    )
                    .to_expr();
                let assign = file.make_continuous_assignment(&wire.to_expr(), &slice_expr);
                file.add_member_continuous_assignment(outer_module, assign);
                name_to_ref.insert(port.name.clone(), (wire, vast_type));
            }
            name_to_ref
        } else {
            HashMap::from([(
                destination.ports[0].name.clone(),
                (stage_output_wire, stage_output_type),
            )])
        };

        log::debug!("current_inputs now: {:?}", current_inputs);

        if let Some(layer) = destination.register_layer {
            current_inputs = make_flop_layer(
                file,
                outer_module,
                &posedge_clk,
                &bit_type,
                current_inputs,
                layer,
                reset_port,
                config.reset_active_low,
            );
        }
    }

    let (result, _) = current_inputs.name_to_ref[&config.layout.output.ports[0].name];
    let assignment = file.make_continuous_assignment(&output_port.to_expr(), &result.to_expr());
    file.add_member_continuous_assignment(outer_module, assignment);

    // Wire up the output valid signal if one is present on the module.
    if let Some(ref output_valid_signal_name) = config.output_valid_signal {
        let value = current_inputs
            .valid_signal
            .expect("flop layers must have a valid signal when module has an output-valid signal");

        // Get the output port LogicRef so we can assign to it.
        let output_port = file.add_output(outer_module, output_valid_signal_name, &bit_type);
        let assignment = file.make_continuous_assignment(&output_port.to_expr(), &value.to_expr());
        file.add_member_continuous_assignment(outer_module, assignment);
    }

    Ok(file.emit())
}

#[cfg(test)]
mod tests {
    use xlsynth_vast::VastFileType;

    use xlsynth_test_helpers::compare_golden_sv;

    use super::*;

    /// Builds the single-stage adder used by the wrapper emission goldens.
    fn build_add32_pipeline(
        flop_inputs: bool,
        flop_outputs: bool,
        has_valid: bool,
    ) -> Result<String, xlsynth::XlsynthError> {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("add32");
        let u32 = file.make_bit_vector_type(32, false);
        let a = file.add_input(module, "a", &u32);
        let b = file.add_input(module, "b", &u32);
        let c = file.add_output(module, "c", &u32);
        let add = file.make_add(&a.to_expr(), &b.to_expr());
        let assignment = file.make_continuous_assignment(&c.to_expr(), &add);
        file.add_member_continuous_assignment(module, assignment);

        let stage = StageModuleInterface {
            name: file.module_name(module),
            output_port: "c".into(),
        };
        let layout = PipelineLayout::new(
            vec![vec![("a".into(), 32), ("b".into(), 32)]],
            ("c".into(), 32),
            flop_inputs,
            flop_outputs,
            has_valid,
        )?;
        let config = PipelineConfig {
            top_module_name: "top".to_string(),
            clk_port_name: "clk".to_string(),
            stage_modules: vec![stage],
            layout: &layout,
            input_valid_signal: has_valid.then_some("in_valid"),
            output_valid_signal: has_valid.then_some("out_valid"),
            reset_signal: has_valid.then_some("rst"),
            reset_active_low: false,
        };
        build_pipeline(&mut file, &config)
    }

    /// Builds a pipeline that is one stage with input/output IO flops.
    #[test]
    fn test_build_pipeline() {
        let pipeline = build_add32_pipeline(true, true, false).unwrap();

        compare_golden_sv(&pipeline, "tests/goldens/build_pipeline_simple.golden.v");
    }

    /// Builds a pipeline that is one stage with input/output IO flops and
    /// input/output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals() {
        let pipeline = build_add32_pipeline(true, true, true).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals.golden.v",
        );
    }

    /// Builds a pipeline that has only input flops and input /
    /// output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals_and_no_output_flops() {
        let pipeline = build_add32_pipeline(true, false, true).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals_and_no_output_flops.golden.v",
        );
    }

    /// Builds a pipeline that has only output flops (no input flops) and input
    /// / output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals_and_no_input_flops() {
        let pipeline = build_add32_pipeline(false, true, true).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals_and_no_input_flops.golden.v",
        );
    }
}
