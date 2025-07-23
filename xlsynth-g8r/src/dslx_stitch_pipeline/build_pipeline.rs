// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use xlsynth::ir_value::IrFormatPreference;
use xlsynth::vast::{Expr, LogicRef, ModulePortDirection, VastDataType, VastFile, VastModule};

use crate::xls_ir::ir::FunctionType;

pub struct PipelineConfig<'a> {
    pub top_module_name: String,
    pub clk_port_name: String,

    pub stage_modules: Vec<&'a VastModule>,

    pub flop_inputs: bool,
    pub flop_outputs: bool,

    pub input_valid_signal: Option<String>,
    pub output_valid_signal: Option<String>,
    pub reset_signal: Option<String>,
    pub reset_active_low: bool,
}

struct NetBundle {
    name_to_ref: HashMap<String, (LogicRef, VastDataType)>,
    valid_signal: Option<LogicRef>,
}

impl std::fmt::Debug for NetBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_to_ref_str = self
            .name_to_ref
            .iter()
            .map(|(name, (logic_ref, _vast_type))| format!("{:?} => {}", name, logic_ref.name()))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "NetBundle {{ name_to_ref: [{}], valid_signal: {:?} }}",
            name_to_ref_str,
            self.valid_signal.as_ref().map(|v| v.name())
        )
    }
}

fn to_net_bundle(
    names: &[String],
    ports: &[(LogicRef, VastDataType)],
    valid_signal: Option<LogicRef>,
) -> NetBundle {
    let mut name_to_ref = HashMap::new();
    for (name, (logic_ref, data_type)) in names.iter().zip(ports) {
        name_to_ref.insert(name.to_string(), (logic_ref.clone(), data_type.clone()));
    }
    NetBundle {
        name_to_ref,
        valid_signal,
    }
}

/// Creates a layer of flops using the input combinational signals given in
/// `current_inputs`. Returns the new `current_inputs` bundle.
fn make_flop_layer(
    file: &mut VastFile,
    outer_module: &mut VastModule,
    posedge_clk: &Expr,
    bit_type: &VastDataType,
    current_inputs: NetBundle,
    next_pipe_stage_number: u32,
    reset_signal: Option<LogicRef>,
    reset_active_low: bool,
) -> NetBundle {
    // First create register declarations so they appear before procedural blocks.
    let mut new_name_to_ref = HashMap::new();
    // Collect and sort for deterministic emission order.
    let mut entries: Vec<(String, (LogicRef, VastDataType))> =
        current_inputs.name_to_ref.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Vector of tuples used later to emit assignments. Bool indicates this row is
    // the valid-reg assignment.
    let mut assign_info: Vec<(LogicRef, LogicRef, bool)> = Vec::new();

    for (name, (logic_ref, data_type)) in &entries {
        let reg = outer_module
            .add_reg(&format!("p{next_pipe_stage_number}_{}", name), &data_type)
            .unwrap();
        new_name_to_ref.insert(name.clone(), (reg.clone(), data_type.clone()));
        assign_info.push((reg, logic_ref.clone(), false));
    }

    // Handle valid signal register declaration (if any) before always_ff.
    let flopped_valid_signal = if let Some(ref valid_signal) = current_inputs.valid_signal {
        let reg = outer_module
            .add_reg(&format!("p{next_pipe_stage_number}_valid"), &bit_type)
            .unwrap();
        assign_info.push((reg.clone(), valid_signal.clone(), true));
        Some(reg)
    } else {
        None
    };

    // Prepare zero literal once if we need reset gating.
    let zero_expr = file
        .make_literal("bits[1]:0", &IrFormatPreference::Binary)
        .unwrap();

    // Now create the always_ff and emit assignments.
    let always_ff = outer_module.add_always_ff(&[posedge_clk]).unwrap();
    let mut sb = always_ff.get_statement_block();

    for (reg, src_logic, is_valid_reg) in assign_info {
        let rhs = if !is_valid_reg {
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
            // The valid reg itself just captures the signal directly, but respect reset
            // gating.
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
        sb.add_nonblocking_assignment(&reg.to_expr(), &rhs);
    }

    NetBundle {
        name_to_ref: new_name_to_ref,
        valid_signal: flopped_valid_signal,
    }
}

/// Creates I/O ports on the outer (encapsulating) module.
fn create_outer_io_ports(
    file: &mut VastFile,
    outer_module: &mut VastModule,
    config: &PipelineConfig,
) -> (NetBundle, Vec<(LogicRef, VastDataType)>) {
    log::debug!("Creating outer I/O ports");

    // Create I/O ports based on the stage modules.
    // Start with inputs from the first stage module.
    let stage_first_module = config.stage_modules[0];
    let mut outer_module_inputs: Vec<(LogicRef, VastDataType)> = Vec::new();
    let outer_module_input_names = stage_first_module
        .input_ports()
        .into_iter()
        .map(|port| port.name().to_string())
        .collect::<Vec<_>>();
    for port in stage_first_module.input_ports() {
        let vast_type = file.make_bit_vector_type(port.width(), false);
        let outer_module_input = outer_module.add_input(&port.name(), &vast_type);
        outer_module_inputs.push((outer_module_input, vast_type));
    }

    // A LogicRef that represents the input valid signal coming combinationally into
    // the outer module.
    let bit_type = file.make_bit_vector_type(1, false);
    let outer_module_input_valid_signal: Option<LogicRef> =
        if let Some(ref input_valid_signal_name) = config.input_valid_signal {
            let outer_module_input_valid_signal =
                outer_module.add_input(&input_valid_signal_name, &bit_type);
            Some(outer_module_input_valid_signal)
        } else {
            None
        };

    // Now do outputs on the outer module, which come from the last stage module.
    let stage_last_module = config.stage_modules.last().unwrap();
    let mut outer_module_outputs: Vec<(LogicRef, VastDataType)> = Vec::new();
    for port in stage_last_module.output_ports() {
        let vast_type = file.make_bit_vector_type(port.width(), false);
        let outer_module_output = outer_module.add_output(&port.name(), &vast_type);
        outer_module_outputs.push((outer_module_output, vast_type));
    }

    // The bundle we give back just has the input-side of the equation so we can
    // feed it to the first stage.
    let net_bundle = to_net_bundle(
        &outer_module_input_names,
        &outer_module_inputs,
        outer_module_input_valid_signal,
    );

    (net_bundle, outer_module_outputs)
}

pub fn build_pipeline(
    file: &mut VastFile,
    config: &PipelineConfig,
) -> Result<String, xlsynth::XlsynthError> {
    if config.stage_modules.is_empty() {
        return Err(xlsynth::XlsynthError(
            "Cannot build pipeline: no stage modules provided".to_string(),
        ));
    }

    let mut outer_module: VastModule = file.add_module(&config.top_module_name);

    let bit_type = file.make_bit_vector_type(1, false);
    let clk_port = outer_module.add_input(&config.clk_port_name, &bit_type);
    // Optional synchronous reset input comes immediately after clk.
    let _reset_port = if let Some(ref rst_name) = config.reset_signal {
        Some(outer_module.add_input(rst_name, &bit_type))
    } else {
        None
    };
    let posedge_clk = file.make_pos_edge(&clk_port.to_expr());

    let (mut current_inputs, outer_module_outputs) =
        create_outer_io_ports(file, &mut outer_module, &config);
    log::debug!("starting with current_inputs: {:?}", current_inputs);

    let mut next_pipe_stage_number = 0;
    if config.flop_inputs {
        current_inputs = make_flop_layer(
            file,
            &mut outer_module,
            &posedge_clk,
            &bit_type,
            current_inputs,
            next_pipe_stage_number,
            _reset_port.clone(),
            config.reset_active_low,
        );
        next_pipe_stage_number += 1;
    }

    for (i, stage_module) in config.stage_modules.iter().enumerate() {
        log::debug!(
            "building stage {}; module name: {}; current_inputs: {:?}",
            i,
            stage_module.name(),
            current_inputs
        );

        assert_eq!(
            stage_module.output_ports().len(),
            1,
            "Stage module must have exactly one output port"
        );
        let stage_output_port = &stage_module.output_ports()[0];
        let stage_output_name = format!("stage_{}_out_comb", i);
        let stage_output_width = stage_output_port.width();
        let stage_output_type = file.make_bit_vector_type(stage_output_width, false);
        let stage_output_wire = outer_module.add_wire(&stage_output_name, &stage_output_type);

        // -- "Smart" slicing: the stage has exactly one output port (because that's how
        // XLS does combinational module generation today), but the next stage
        // (or the final outer-module connection) may require multiple
        // signals wired up to input ports: create slice wires for each required
        // destination.

        assert_eq!(
            stage_module.output_ports().len(),
            1,
            "Stage module must have exactly one output port"
        );

        // Determine the required destinations to potentially slice the output wire
        // into. This is a vector of (next_stage_name, start, limit)
        let mut dest_ports: Vec<(String, i64, i64)> = Vec::new();
        let mut dest_bit_index: i64 = 0;

        let last_stage = i + 1 == config.stage_modules.len();
        if !last_stage {
            // Not the last stage – look at next stage's input ports.
            for p in config.stage_modules[i + 1].input_ports() {
                dest_ports.push((
                    p.name().to_string(),
                    dest_bit_index,
                    dest_bit_index + p.width(),
                ));
                dest_bit_index += p.width();
            }
        } else {
            // Last stage – slice according to outer-module outputs.
            for (logic_ref, dt) in &outer_module_outputs {
                dest_ports.push((
                    logic_ref.name().to_string(),
                    dest_bit_index,
                    dest_bit_index + dt.width_as_int64().unwrap(),
                ));
                dest_bit_index += dt.width_as_int64().unwrap();
            }
        }

        log::debug!(
            "determined required destinations; dest_ports: {:?}",
            dest_ports
        );

        // Instantiate the stage module with the current inputs.
        let mut stage_port_names: Vec<String> = Vec::new();
        let mut stage_expressions: Vec<Option<Expr>> = Vec::new();
        for port in stage_module.ports() {
            match port.direction() {
                ModulePortDirection::Input => {
                    stage_port_names.push(port.name());
                    let (input_logic_ref, _) =
                        current_inputs
                            .name_to_ref
                            .get(&port.name())
                            .expect(&format!(
                                "module {} input port {} not found in current_inputs",
                                stage_module.name(),
                                port.name()
                            ));
                    let input_expr = input_logic_ref.to_expr();
                    stage_expressions.push(Some(input_expr));
                }
                ModulePortDirection::Output => {
                    stage_port_names.push(port.name());
                    stage_expressions.push(Some(stage_output_wire.to_expr()));
                }
            }
        }

        let stage_port_name_refs = stage_port_names
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>();
        let stage_expressions = stage_expressions
            .iter()
            .map(|expr| expr.as_ref())
            .collect::<Vec<_>>();

        // Instantiate the combinational stage module.
        let instantiation = file.make_instantiation(
            &stage_module.name(),
            &format!("stage_{i}"),
            &[],
            &[],
            &stage_port_name_refs,
            &stage_expressions,
        );
        outer_module.add_member_instantiation(instantiation);

        // If the next stage wants more than one signal (i.e. it has arity > 1), we
        // break out the signals by slicing here.
        current_inputs = if dest_ports.len() > 1 {
            let total_bits: i64 = dest_bit_index;
            assert_eq!(
                stage_output_width, total_bits,
                "Bit width mismatch between sole output ({stage_output_width}) and concatenated destination ports ({}).",
                total_bits
            );

            let mut name_to_ref = HashMap::new();
            for (dest_name, start, limit) in dest_ports {
                let vast_type = file.make_bit_vector_type(limit - start, false);
                let wire_name = format!("p{}_{}_comb", next_pipe_stage_number, dest_name);
                let wire = outer_module.add_wire(&wire_name, &vast_type);
                let slice_expr = file
                    .make_slice(&stage_output_wire.to_indexable_expr(), limit - 1, start)
                    .to_expr();
                let assign = file.make_continuous_assignment(&wire.to_expr(), &slice_expr);
                outer_module.add_member_continuous_assignment(assign);
                name_to_ref.insert(dest_name.clone(), (wire, vast_type));
            }

            NetBundle {
                name_to_ref,
                valid_signal: current_inputs.valid_signal,
            }
        } else {
            let name_to_ref = HashMap::from([(
                dest_ports[0].0.clone(),
                (stage_output_wire, stage_output_type),
            )]);
            NetBundle {
                name_to_ref,
                valid_signal: current_inputs.valid_signal,
            }
        };

        log::debug!("current_inputs now: {:?}", current_inputs);

        // If this is not the last stage, then we need to make a intra-pipeline flop
        // layer.
        if !last_stage {
            current_inputs = make_flop_layer(
                file,
                &mut outer_module,
                &posedge_clk,
                &bit_type,
                current_inputs,
                next_pipe_stage_number,
                _reset_port.clone(),
                config.reset_active_low,
            );
            next_pipe_stage_number += 1;
        }
    }

    if config.flop_outputs {
        current_inputs = make_flop_layer(
            file,
            &mut outer_module,
            &posedge_clk,
            &bit_type,
            current_inputs,
            next_pipe_stage_number,
            _reset_port.clone(),
            config.reset_active_low,
        );
        next_pipe_stage_number += 1;
    }

    assert_eq!(
        next_pipe_stage_number as usize,
        (config.stage_modules.len() - 1)
            + config.flop_inputs as usize
            + config.flop_outputs as usize
    );

    // Now we need to wire up the output ports for the outer module using the
    // `current_inputs` bundle.
    for (logic_ref, _data_type) in outer_module_outputs.iter() {
        // Make an assignment to this output from the corresponding `current_inputs`
        // value.
        let current_input = &current_inputs.name_to_ref.get(&logic_ref.name()).unwrap().0;
        let assignment =
            file.make_continuous_assignment(&logic_ref.to_expr(), &current_input.to_expr());
        outer_module.add_member_continuous_assignment(assignment);
    }

    // Wire up the output valid signal if one is present on the module.
    if let Some(ref output_valid_signal_name) = config.output_valid_signal {
        let value = current_inputs
            .valid_signal
            .expect("flop layers must have a valid signal when module has an output-valid signal");

        // Get the output port LogicRef so we can assign to it.
        let output_port = outer_module.add_output(output_valid_signal_name, &bit_type);
        let assignment = file.make_continuous_assignment(&output_port.to_expr(), &value.to_expr());
        outer_module.add_member_continuous_assignment(assignment);
    }

    Ok(file.emit())
}

#[cfg(test)]
mod tests {
    use xlsynth::vast::VastFileType;

    use xlsynth_test_helpers::compare_golden_sv;

    use super::*;

    fn add_add32_module(file: &mut VastFile) -> VastModule {
        let mut module = file.add_module("add32");
        let u32 = file.make_bit_vector_type(32, false);
        let a = module.add_input("a", &u32);
        let b = module.add_input("b", &u32);
        let c = module.add_output("c", &u32);
        let add = file.make_add(&a.to_expr(), &b.to_expr());
        let assignment = file.make_continuous_assignment(&c.to_expr(), &add);
        module.add_member_continuous_assignment(assignment);
        module
    }

    /// Builds a pipeline that is one stage with input/output IO flops.
    #[test]
    fn test_build_pipeline() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut file = VastFile::new(VastFileType::Verilog);
        let add32 = add_add32_module(&mut file);
        let config = PipelineConfig {
            top_module_name: "top".to_string(),
            clk_port_name: "clk".to_string(),
            stage_modules: vec![&add32],
            flop_inputs: true,
            flop_outputs: true,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        };
        let pipeline = build_pipeline(&mut file, &config).unwrap();

        compare_golden_sv(&pipeline, "tests/goldens/build_pipeline_simple.golden.v");
    }

    /// Builds a pipeline that is one stage with input/output IO flops and
    /// input/output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut file = VastFile::new(VastFileType::Verilog);
        let add32 = add_add32_module(&mut file);
        let config = PipelineConfig {
            top_module_name: "top".to_string(),
            clk_port_name: "clk".to_string(),
            stage_modules: vec![&add32],
            flop_inputs: true,
            flop_outputs: true,
            input_valid_signal: Some("in_valid".to_string()),
            output_valid_signal: Some("out_valid".to_string()),
            reset_signal: Some("rst".to_string()),
            reset_active_low: false,
        };
        let pipeline = build_pipeline(&mut file, &config).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals.golden.v",
        );
    }

    /// Builds a pipline that has only input flops no output flops and input /
    /// output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals_and_no_output_flops() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut file = VastFile::new(VastFileType::Verilog);
        let add32 = add_add32_module(&mut file);
        let config = PipelineConfig {
            top_module_name: "top".to_string(),
            clk_port_name: "clk".to_string(),
            stage_modules: vec![&add32],
            flop_inputs: true,
            flop_outputs: false,
            input_valid_signal: Some("in_valid".to_string()),
            output_valid_signal: Some("out_valid".to_string()),
            reset_signal: Some("rst".to_string()),
            reset_active_low: false,
        };
        let pipeline = build_pipeline(&mut file, &config).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals_and_no_output_flops.golden.v",
        );
    }

    /// Builds a pipeline that has only output flops (no input flops) and input
    /// / output valid signals.
    #[test]
    fn test_build_pipeline_with_valid_signals_and_no_input_flops() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut file = VastFile::new(VastFileType::Verilog);
        let add32 = add_add32_module(&mut file);
        let config = PipelineConfig {
            top_module_name: "top".to_string(),
            clk_port_name: "clk".to_string(),
            stage_modules: vec![&add32],
            flop_inputs: false,
            flop_outputs: true,
            input_valid_signal: Some("in_valid".to_string()),
            output_valid_signal: Some("out_valid".to_string()),
            reset_signal: Some("rst".to_string()),
            reset_active_low: false,
        };
        let pipeline = build_pipeline(&mut file, &config).unwrap();

        compare_golden_sv(
            &pipeline,
            "tests/goldens/build_pipeline_with_valid_signals_and_no_input_flops.golden.v",
        );
    }
}
