// SPDX-License-Identifier: Apache-2.0

//! Emits a Verilog-like RTL netlist for a synchronous gate-level design.

use std::collections::BTreeMap;
use std::rc::Rc;

use xlsynth::vast;

use crate::aig::gate;
use crate::aig::{SequentialGateFn, TransitionInputId};
use crate::verilog_version::VerilogVersion;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NetlistPortStyle {
    ScalarBits,
    PackedBits,
}

#[derive(Clone)]
struct SignalRef {
    logic_ref: vast::LogicRef,
}

#[derive(Clone)]
struct DirectAssignment {
    sort_key: String,
    lhs_expr: Rc<vast::Expr>,
    rhs_expr: Rc<vast::Expr>,
}

#[derive(Clone)]
struct ProceduralAssignment {
    sort_key: String,
    lhs_expr: Rc<vast::Expr>,
    rhs_expr: Rc<vast::Expr>,
}

struct NetlistEmitConfig<'a> {
    design: &'a SequentialGateFn,
    port_style: NetlistPortStyle,
}

struct NetlistEmitState {
    gate_ref_to_vast_expr: BTreeMap<gate::AigRef, Rc<vast::Expr>>,
    transition_output_bit_targets: BTreeMap<(usize, usize), Rc<vast::Expr>>,
    packed_external_inputs: BTreeMap<usize, SignalRef>,
    register_bit_exprs: BTreeMap<(usize, usize), Rc<vast::Expr>>,
    packed_registers: BTreeMap<usize, SignalRef>,
    packed_next_state_wires: BTreeMap<usize, SignalRef>,
    internal_wire_assignments: Vec<DirectAssignment>,
    direct_output_assignments: Vec<DirectAssignment>,
    final_output_assignments: Vec<DirectAssignment>,
    procedural_assignments: Vec<ProceduralAssignment>,
}

impl NetlistEmitState {
    fn new() -> Self {
        Self {
            gate_ref_to_vast_expr: BTreeMap::new(),
            transition_output_bit_targets: BTreeMap::new(),
            packed_external_inputs: BTreeMap::new(),
            register_bit_exprs: BTreeMap::new(),
            packed_registers: BTreeMap::new(),
            packed_next_state_wires: BTreeMap::new(),
            internal_wire_assignments: Vec::new(),
            direct_output_assignments: Vec::new(),
            final_output_assignments: Vec::new(),
            procedural_assignments: Vec::new(),
        }
    }
}

fn port_data_type(
    file: &mut vast::VastFile,
    bit_type: &vast::VastDataType,
    bit_count: usize,
) -> Result<vast::VastDataType, String> {
    match bit_count {
        0 => Err("emit_netlist: packed ports cannot have zero width".to_string()),
        1 => Ok(bit_type.clone()),
        width => Ok(file.make_bit_vector_type(width as i64, false)),
    }
}

fn packed_bit_expr(
    file: &mut vast::VastFile,
    logic_ref: &vast::LogicRef,
    bit_count: usize,
    bit_index: usize,
) -> Rc<vast::Expr> {
    if bit_count == 1 {
        return Rc::new(logic_ref.to_expr());
    }
    Rc::new(
        file.make_index(&logic_ref.to_indexable_expr(), bit_index as i64)
            .to_expr(),
    )
}

fn scalar_bit_name(name: &str, bit_count: usize, bit_index: usize) -> String {
    if bit_count == 1 {
        name.to_string()
    } else {
        format!("{name}_{bit_index}")
    }
}

fn scalar_next_state_name(name: &str, bit_count: usize, bit_index: usize) -> String {
    if bit_count == 1 {
        return name.to_string();
    }
    match name.strip_suffix("_comb") {
        Some(base) => format!("{base}_{bit_index}_comb"),
        None => format!("{name}_{bit_index}"),
    }
}

fn synthetic_output_value_index(name: &str) -> Option<usize> {
    let suffix = name.strip_prefix("output_value_")?;
    if suffix.is_empty() {
        return None;
    }
    suffix.parse::<usize>().ok()
}

fn compare_packed_output_port_order(lhs: &gate::Output, rhs: &gate::Output) -> std::cmp::Ordering {
    match (
        synthetic_output_value_index(&lhs.name),
        synthetic_output_value_index(&rhs.name),
    ) {
        (Some(lhs_index), Some(rhs_index)) => lhs_index
            .cmp(&rhs_index)
            .then_with(|| lhs.name.cmp(&rhs.name)),
        _ => lhs.name.cmp(&rhs.name),
    }
}

fn bit_vectors_equal(lhs: &gate::AigBitVector, rhs: &gate::AigBitVector) -> bool {
    lhs.get_bit_count() == rhs.get_bit_count()
        && lhs
            .iter_lsb_to_msb()
            .zip(rhs.iter_lsb_to_msb())
            .all(|(lhs_bit, rhs_bit)| lhs_bit == rhs_bit)
}

fn direct_external_input_for_d(
    design: &SequentialGateFn,
    d_output: &gate::Output,
) -> Option<TransitionInputId> {
    if d_output.name.ends_with("_comb") {
        return None;
    }
    design.inputs.iter().copied().find(|id| {
        bit_vectors_equal(
            &design.transition.inputs[id.index()].bit_vector,
            &d_output.bit_vector,
        )
    })
}

fn driving_register_for_output(design: &SequentialGateFn, output: &gate::Output) -> Option<usize> {
    design.registers.iter().position(|register| {
        bit_vectors_equal(
            &design.transition.inputs[register.q.index()].bit_vector,
            &output.bit_vector,
        )
    })
}

fn generate_external_input_ports(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) -> Result<(), String> {
    for input_id in &config.design.inputs {
        let input = &config.design.transition.inputs[input_id.index()];
        let bit_count = input.get_bit_count();
        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                for (bit_index, aig_bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                    let name = scalar_bit_name(&input.name, bit_count, bit_index);
                    let logic_ref = module.add_input(&name, bit_type);
                    state
                        .gate_ref_to_vast_expr
                        .insert(aig_bit.node, Rc::new(logic_ref.to_expr()));
                }
            }
            NetlistPortStyle::PackedBits => {
                let data_type = port_data_type(file, bit_type, bit_count)?;
                let logic_ref = module.add_input(&input.name, &data_type);
                for (bit_index, aig_bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                    state.gate_ref_to_vast_expr.insert(
                        aig_bit.node,
                        packed_bit_expr(file, &logic_ref, bit_count, bit_index),
                    );
                }
                state
                    .packed_external_inputs
                    .insert(input_id.index(), SignalRef { logic_ref });
            }
        }
    }
    Ok(())
}

fn generate_registers_and_next_state_wires(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) -> Result<(), String> {
    for (register_index, register) in config.design.registers.iter().enumerate() {
        let q_input = &config.design.transition.inputs[register.q.index()];
        let d_output = &config.design.transition.outputs[register.d.index()];
        let bit_count = q_input.get_bit_count();
        let d_is_external_input = direct_external_input_for_d(config.design, d_output).is_some();

        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                if !d_is_external_input {
                    for bit_index in 0..bit_count {
                        let d_name = scalar_next_state_name(&d_output.name, bit_count, bit_index);
                        let wire_ref = module.add_wire(&d_name, bit_type);
                        state
                            .transition_output_bit_targets
                            .insert((register.d.index(), bit_index), Rc::new(wire_ref.to_expr()));
                    }
                }
                for (bit_index, aig_bit) in q_input.bit_vector.iter_lsb_to_msb().enumerate() {
                    let reg_name = scalar_bit_name(&register.name, bit_count, bit_index);
                    let reg_ref = module.add_reg(&reg_name, bit_type).unwrap();
                    let expr = Rc::new(reg_ref.to_expr());
                    state
                        .gate_ref_to_vast_expr
                        .insert(aig_bit.node, expr.clone());
                    state
                        .register_bit_exprs
                        .insert((register_index, bit_index), expr);
                }
            }
            NetlistPortStyle::PackedBits => {
                let data_type = port_data_type(file, bit_type, bit_count)?;
                if !d_is_external_input {
                    let wire_ref = module.add_wire(&d_output.name, &data_type);
                    for bit_index in 0..bit_count {
                        state.transition_output_bit_targets.insert(
                            (register.d.index(), bit_index),
                            packed_bit_expr(file, &wire_ref, bit_count, bit_index),
                        );
                    }
                    state.packed_next_state_wires.insert(
                        register_index,
                        SignalRef {
                            logic_ref: wire_ref,
                        },
                    );
                }
                let reg_ref = module.add_reg(&register.name, &data_type).unwrap();
                for (bit_index, aig_bit) in q_input.bit_vector.iter_lsb_to_msb().enumerate() {
                    let expr = packed_bit_expr(file, &reg_ref, bit_count, bit_index);
                    state
                        .gate_ref_to_vast_expr
                        .insert(aig_bit.node, expr.clone());
                    state
                        .register_bit_exprs
                        .insert((register_index, bit_index), expr);
                }
                state
                    .packed_registers
                    .insert(register_index, SignalRef { logic_ref: reg_ref });
            }
        }
    }
    Ok(())
}

fn generate_external_output_ports(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) -> Result<(), String> {
    let mut outputs = config
        .design
        .outputs
        .iter()
        .map(|id| (id.index(), &config.design.transition.outputs[id.index()]))
        .collect::<Vec<(usize, &gate::Output)>>();
    if config.port_style == NetlistPortStyle::PackedBits {
        outputs.sort_by(|(_, lhs), (_, rhs)| compare_packed_output_port_order(lhs, rhs));
    }

    for (output_index, output) in outputs {
        let bit_count = output.get_bit_count();
        let q_register_index = driving_register_for_output(config.design, output);
        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                for bit_index in 0..bit_count {
                    let name = scalar_bit_name(&output.name, bit_count, bit_index);
                    let output_ref = module.add_output(&name, bit_type);
                    if let Some(register_index) = q_register_index {
                        let rhs_expr = state
                            .register_bit_exprs
                            .get(&(register_index, bit_index))
                            .expect("register Q expression was generated")
                            .clone();
                        state.final_output_assignments.push(DirectAssignment {
                            sort_key: name,
                            lhs_expr: Rc::new(output_ref.to_expr()),
                            rhs_expr,
                        });
                    } else {
                        state
                            .transition_output_bit_targets
                            .insert((output_index, bit_index), Rc::new(output_ref.to_expr()));
                    }
                }
            }
            NetlistPortStyle::PackedBits => {
                let data_type = port_data_type(file, bit_type, bit_count)?;
                let output_ref = module.add_output(&output.name, &data_type);
                if let Some(register_index) = q_register_index {
                    let register_ref = state
                        .packed_registers
                        .get(&register_index)
                        .expect("register Q declaration was generated");
                    state.final_output_assignments.push(DirectAssignment {
                        sort_key: output.name.clone(),
                        lhs_expr: Rc::new(output_ref.to_expr()),
                        rhs_expr: Rc::new(register_ref.logic_ref.to_expr()),
                    });
                } else {
                    for bit_index in 0..bit_count {
                        state.transition_output_bit_targets.insert(
                            (output_index, bit_index),
                            packed_bit_expr(file, &output_ref, bit_count, bit_index),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn generate_internal_combinational_logic(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) {
    for (index, gate_node) in config.design.transition.gates.iter().enumerate() {
        let aig_ref = gate::AigRef { id: index };
        match gate_node {
            gate::AigNode::Input { .. } => {}
            gate::AigNode::Literal { value, .. } => {
                let name = format!("G{index}");
                let wire_ref = module.add_wire(&name, bit_type);
                state
                    .gate_ref_to_vast_expr
                    .insert(aig_ref, Rc::new(wire_ref.to_expr()));
                let value_str = if *value { "bits[1]:1" } else { "bits[1]:0" };
                let rhs_expr = Rc::new(
                    file.make_literal(value_str, &xlsynth::ir_value::IrFormatPreference::Binary)
                        .unwrap(),
                );
                state.internal_wire_assignments.push(DirectAssignment {
                    sort_key: name,
                    lhs_expr: Rc::new(wire_ref.to_expr()),
                    rhs_expr,
                });
            }
            gate::AigNode::And2 { a, b, .. } => {
                let name = format!("G{index}");
                let wire_ref = module.add_wire(&name, bit_type);
                state
                    .gate_ref_to_vast_expr
                    .insert(aig_ref, Rc::new(wire_ref.to_expr()));
                let a_expr = operand_expr(file, state, a);
                let b_expr = operand_expr(file, state, b);
                state.internal_wire_assignments.push(DirectAssignment {
                    sort_key: name,
                    lhs_expr: Rc::new(wire_ref.to_expr()),
                    rhs_expr: Rc::new(file.make_bitwise_and(&a_expr, &b_expr)),
                });
            }
        }
    }
}

fn operand_expr(
    file: &mut vast::VastFile,
    state: &NetlistEmitState,
    operand: &gate::AigOperand,
) -> vast::Expr {
    let base = state
        .gate_ref_to_vast_expr
        .get(&operand.node)
        .unwrap_or_else(|| panic!("missing emitted expression for AIG node {:?}", operand.node));
    if operand.negated {
        file.make_not(base)
    } else {
        (**base).clone()
    }
}

fn connect_transition_output_targets(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
) {
    for (output_index, output) in config.design.transition.outputs.iter().enumerate() {
        for (bit_index, operand) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            let Some(target_expr) = state
                .transition_output_bit_targets
                .get(&(output_index, bit_index))
            else {
                continue;
            };
            state.direct_output_assignments.push(DirectAssignment {
                sort_key: format!("{}[{bit_index:020}]", output.name),
                lhs_expr: target_expr.clone(),
                rhs_expr: Rc::new(operand_expr(file, state, operand)),
            });
        }
    }
}

fn generate_register_assignments(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
) {
    for (register_index, register) in config.design.registers.iter().enumerate() {
        let d_output = &config.design.transition.outputs[register.d.index()];
        let direct_input = direct_external_input_for_d(config.design, d_output);
        let bit_count = d_output.get_bit_count();
        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                for (bit_index, operand) in d_output.bit_vector.iter_lsb_to_msb().enumerate() {
                    let rhs_expr = match direct_input {
                        Some(_) => Rc::new(operand_expr(file, state, operand)),
                        None => state
                            .transition_output_bit_targets
                            .get(&(register.d.index(), bit_index))
                            .expect("register D wire expression was generated")
                            .clone(),
                    };
                    state.procedural_assignments.push(ProceduralAssignment {
                        sort_key: scalar_bit_name(&register.name, bit_count, bit_index),
                        lhs_expr: state
                            .register_bit_exprs
                            .get(&(register_index, bit_index))
                            .expect("register Q expression was generated")
                            .clone(),
                        rhs_expr,
                    });
                }
            }
            NetlistPortStyle::PackedBits => {
                let lhs = state
                    .packed_registers
                    .get(&register_index)
                    .expect("register declaration was generated");
                let rhs_expr = match direct_input {
                    Some(input_id) => Rc::new(
                        state
                            .packed_external_inputs
                            .get(&input_id.index())
                            .expect("external input declaration was generated")
                            .logic_ref
                            .to_expr(),
                    ),
                    None => Rc::new(
                        state
                            .packed_next_state_wires
                            .get(&register_index)
                            .expect("register D wire declaration was generated")
                            .logic_ref
                            .to_expr(),
                    ),
                };
                state.procedural_assignments.push(ProceduralAssignment {
                    sort_key: register.name.clone(),
                    lhs_expr: Rc::new(lhs.logic_ref.to_expr()),
                    rhs_expr,
                });
            }
        }
    }
}

fn emit_assignments(
    assignments: &mut [DirectAssignment],
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
) {
    assignments.sort_by(|lhs, rhs| lhs.sort_key.cmp(&rhs.sort_key));
    for assignment in assignments {
        module.add_member_continuous_assignment(
            file.make_continuous_assignment(&assignment.lhs_expr, &assignment.rhs_expr),
        );
    }
}

fn generate_sequential_block(
    design: &SequentialGateFn,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    clock_ref: Option<&vast::LogicRef>,
) {
    if state.procedural_assignments.is_empty() {
        return;
    }
    let clock_ref = clock_ref.expect("validated registered designs must have a clock");
    state
        .procedural_assignments
        .sort_by(|lhs, rhs| lhs.sort_key.cmp(&rhs.sort_key));
    let posedge_clock = file.make_pos_edge(&clock_ref.to_expr());
    let mut block = module
        .add_always_ff(&[&posedge_clock])
        .unwrap()
        .get_statement_block();
    for assignment in &state.procedural_assignments {
        block.add_nonblocking_assignment(&assignment.lhs_expr, &assignment.rhs_expr);
    }
    debug_assert!(!design.registers.is_empty());
}

/// Emits a netlist from a validated sequential design with configurable port
/// packing.
pub fn emit_netlist_with_version_and_port_style(
    design: &SequentialGateFn,
    version: VerilogVersion,
    port_style: NetlistPortStyle,
) -> Result<String, String> {
    design
        .validate()
        .map_err(|e| format!("emit_netlist: invalid SequentialGateFn: {e}"))?;
    if let Some(register) = design
        .registers
        .iter()
        .find(|register| register.initial_value.is_some())
    {
        return Err(format!(
            "emit_netlist: register '{}' has an initial value, which RTL emission does not currently represent",
            register.name
        ));
    }
    if let Some(clock) = &design.clock
        && design
            .inputs
            .iter()
            .any(|id| design.transition.inputs[id.index()].name == clock.name)
    {
        return Err(format!(
            "emit_netlist: clock name '{}' collides with an external input name",
            clock.name
        ));
    }

    let file_type = if version.is_system_verilog() {
        vast::VastFileType::SystemVerilog
    } else {
        vast::VastFileType::Verilog
    };
    let mut file = vast::VastFile::new(file_type);
    let mut module = file.add_module(&design.name);
    let bit_type = file.make_bit_vector_type(1, false);
    let config = NetlistEmitConfig { design, port_style };
    let mut state = NetlistEmitState::new();

    let clock_ref = design
        .clock
        .as_ref()
        .map(|clock| module.add_input(&clock.name, &bit_type));
    generate_external_input_ports(&config, &mut state, &mut file, &mut module, &bit_type)?;
    generate_registers_and_next_state_wires(
        &config,
        &mut state,
        &mut file,
        &mut module,
        &bit_type,
    )?;
    generate_external_output_ports(&config, &mut state, &mut file, &mut module, &bit_type)?;
    generate_internal_combinational_logic(&config, &mut state, &mut file, &mut module, &bit_type);
    connect_transition_output_targets(&config, &mut state, &mut file);
    generate_register_assignments(&config, &mut state, &mut file);

    emit_assignments(&mut state.internal_wire_assignments, &mut file, &mut module);
    emit_assignments(&mut state.direct_output_assignments, &mut file, &mut module);
    generate_sequential_block(
        design,
        &mut state,
        &mut file,
        &mut module,
        clock_ref.as_ref(),
    );
    emit_assignments(&mut state.final_output_assignments, &mut file, &mut module);
    Ok(file.emit())
}

/// Emits a scalar-port netlist for a sequential design.
pub fn emit_netlist_with_version(
    design: &SequentialGateFn,
    version: VerilogVersion,
) -> Result<String, String> {
    emit_netlist_with_version_and_port_style(design, version, NetlistPortStyle::ScalarBits)
}

/// Emits a scalar-port netlist, retaining the existing boolean version
/// selection convenience API.
pub fn emit_netlist(design: &SequentialGateFn, use_system_verilog: bool) -> Result<String, String> {
    let version = if use_system_verilog {
        VerilogVersion::SystemVerilog
    } else {
        VerilogVersion::Verilog
    };
    emit_netlist_with_version(design, version)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::aig::{ClockPort, add_input_registers, add_output_registers};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn wrap(gate_fn: gate::GateFn, flop_inputs: bool, flop_outputs: bool) -> SequentialGateFn {
        let clock = ClockPort {
            name: "clk".to_string(),
        };
        let mut design = SequentialGateFn::from_gate_fn(gate_fn);
        if flop_inputs {
            design = add_input_registers(&design, clock.clone()).unwrap();
        }
        if flop_outputs {
            design = add_output_registers(&design, clock.clone()).unwrap();
        }
        design
    }

    #[test]
    fn test_emit_inverter_no_flops() {
        let mut builder = GateBuilder::new("my_inverter".to_string(), GateBuilderOptions::no_opt());
        let i_ref = builder.add_input("i".to_string(), 1);
        let gate_ref = builder.add_not(*i_ref.get_lsb(0));
        builder.add_output("o".to_string(), gate::AigBitVector::from_bit(gate_ref));

        let netlist =
            emit_netlist(&SequentialGateFn::from_gate_fn(builder.build()), false).unwrap();
        let expected = r#"module my_inverter(
  input wire i,
  output wire o
);
  wire G0;
  assign G0 = 1'b0;
  assign o = ~i;
endmodule
"#;
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_and_gate_no_flops() {
        let mut builder = GateBuilder::new("my_and_gate".to_string(), GateBuilderOptions::no_opt());
        let i_val = builder.add_input("i".to_string(), 1);
        let j_val = builder.add_input("j".to_string(), 1);
        let o_val = builder.add_and_binary(*i_val.get_lsb(0), *j_val.get_lsb(0));
        builder.add_output("o".to_string(), gate::AigBitVector::from_bit(o_val));

        let netlist =
            emit_netlist(&SequentialGateFn::from_gate_fn(builder.build()), false).unwrap();
        let expected = "module my_and_gate(\n  input wire i,\n  input wire j,\n  output wire o\n);\n  wire G0;\n  wire G3;\n  assign G0 = 1'b0;\n  assign G3 = i & j;\n  assign o = G3;\nendmodule\n";
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_packed_vector_ports_no_flops() {
        let mut builder = GateBuilder::new("packed_and".to_string(), GateBuilderOptions::no_opt());
        let arg0 = builder.add_input("arg0".to_string(), 4);
        let arg1 = builder.add_input("arg1".to_string(), 4);
        let output_value = builder.add_and_vec(&arg0, &arg1);
        builder.add_output("output_value".to_string(), output_value);
        let design = SequentialGateFn::from_gate_fn(builder.build());

        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        let expected = r#"module packed_and(
  input wire [3:0] arg0,
  input wire [3:0] arg1,
  output wire [3:0] output_value
);
  wire G0;
  wire G9;
  wire G10;
  wire G11;
  wire G12;
  assign G0 = 1'b0;
  assign G10 = arg0[1] & arg1[1];
  assign G11 = arg0[2] & arg1[2];
  assign G12 = arg0[3] & arg1[3];
  assign G9 = arg0[0] & arg1[0];
  assign output_value[0] = G9;
  assign output_value[1] = G10;
  assign output_value[2] = G11;
  assign output_value[3] = G12;
endmodule
"#;
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_packed_tuple_outputs_use_numeric_port_order() {
        let mut builder = GateBuilder::new(
            "packed_tuple_order".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let arg0 = builder.add_input("arg0".to_string(), 1);
        let output_bit = *arg0.get_lsb(0);

        for index in (0..11).rev() {
            builder.add_output(
                format!("output_value_{index}"),
                gate::AigBitVector::from_bit(output_bit),
            );
        }
        let design = SequentialGateFn::from_gate_fn(builder.build());
        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        let module_header = netlist.split(");\n").next().expect("module header");
        let output_ports = module_header
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix("output wire ")
                    .map(|name| name.trim_end_matches(',').to_string())
            })
            .collect::<Vec<String>>();
        let expected_ports = (0..11)
            .map(|index| format!("output_value_{index}"))
            .collect::<Vec<String>>();

        assert_eq!(output_ports, expected_ports, "netlist:\n{netlist}");
    }

    #[test]
    fn test_emit_packed_vector_ports_flop_outputs() {
        let mut builder = GateBuilder::new("packed_flop".to_string(), GateBuilderOptions::no_opt());
        let arg0 = builder.add_input("arg0".to_string(), 2);
        let output_bits = arg0
            .iter_lsb_to_msb()
            .map(|bit| builder.add_not(*bit))
            .collect::<Vec<gate::AigOperand>>();
        builder.add_output(
            "output_value".to_string(),
            gate::AigBitVector::from_lsb_is_index_0(&output_bits),
        );
        let design = wrap(builder.build(), false, true);
        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        let expected = r#"module packed_flop(
  input wire clk,
  input wire [1:0] arg0,
  output wire [1:0] output_value
);
  wire [1:0] output_value_comb;
  reg [1:0] p0_output_value;
  wire G0;
  assign G0 = 1'b0;
  assign output_value_comb[0] = ~arg0[0];
  assign output_value_comb[1] = ~arg0[1];
  always_ff @ (posedge clk) begin
    p0_output_value <= output_value_comb;
  end
  assign output_value = p0_output_value;
endmodule
"#;
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_inverter_flop_input_output() {
        let mut builder = GateBuilder::new("my_flop_inv".to_string(), GateBuilderOptions::no_opt());
        let i_val = builder.add_input("i".to_string(), 1);
        let o_val = builder.add_not(*i_val.get_lsb(0));
        builder.add_output("o".to_string(), gate::AigBitVector::from_bit(o_val));
        let design = wrap(builder.build(), true, true);
        let netlist = emit_netlist(&design, false).unwrap();

        let expected = r#"module my_flop_inv(
  input wire clk,
  input wire i,
  output wire o
);
  reg p0_i;
  wire o_comb;
  reg p0_o;
  wire G0;
  assign G0 = 1'b0;
  assign o_comb = ~p0_i;
  always_ff @ (posedge clk) begin
    p0_i <= i;
    p0_o <= o_comb;
  end
  assign o = p0_o;
endmodule
"#;
        assert_eq!(netlist, expected.to_string());
    }
}
