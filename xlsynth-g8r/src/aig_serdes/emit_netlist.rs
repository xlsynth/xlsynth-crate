// SPDX-License-Identifier: Apache-2.0

//! Uses the xlsynth VAST APIs to build a netlist verilog representation of the
//! given gates.

use crate::aig::gate;
use crate::verilog_version::VerilogVersion;
use std::collections::BTreeMap;
use std::rc::Rc;
use xlsynth::vast;

#[derive(Clone)] // vast::LogicRef is Clone, String is Clone
struct RegDecl {
    name: String,
    logic_ref: vast::LogicRef,
}

#[derive(Clone)] // vast::Expr is Clone, String is Clone
struct FinalOutputAssignment {
    port_name: String,
    port_expr: Rc<vast::Expr>,
    reg_expr: Rc<vast::Expr>,
}

#[derive(Clone)] // String, Rc<vast::Expr> are Clone
struct DirectOutputAssignment {
    target_name: String, // Name of the output port or wire being assigned
    lhs_expr: Rc<vast::Expr>,
    rhs_expr: Rc<vast::Expr>,
}

#[derive(Clone)] // String, vast::Expr are Clone
struct ProceduralAssignment {
    sort_key: String, // Typically the name of the register being assigned (LHS)
    lhs_expr: Rc<vast::Expr>,
    rhs_expr: Rc<vast::Expr>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NetlistPortStyle {
    ScalarBits,
    PackedBits,
}

// Configuration for netlist emission (immutable after creation)
struct NetlistEmitConfig<'g> {
    gate_fn: &'g gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    clk_name: Option<String>,
    port_style: NetlistPortStyle,
}

// Mutable state collected during netlist emission
struct NetlistEmitState {
    // Maps an AIG reference to the expression that reads its value. Packed-port
    // mode maps primary input leaves to bit-select expressions such as
    // `arg0[3]`, while internal gates still map to scalar wires.
    gate_ref_to_vast_expr: BTreeMap<gate::AigRef, Rc<vast::Expr>>,
    // Maps each logical output bit occurrence to the assignment target
    // expression (output port, output bit-select, or _comb bit-select). This is
    // keyed by output position, not driver operand, because valid AIGER can
    // expose the same literal through multiple output ports.
    output_bit_to_combinational_target_expr: BTreeMap<(usize, usize), Rc<vast::Expr>>,
    // Collects procedural assignments for always_ff blocks (e.g., p0_input <= input_port,
    // p0_output_reg <= o_comb_wire).
    procedural_assignments: Vec<ProceduralAssignment>,
    // Collects assignments for final output ports when outputs are flopped (e.g., assign
    // output_port = p0_output_reg).
    final_output_port_assignments: Vec<FinalOutputAssignment>,
    // Collects declarations for all registers (p0_ inputs/outputs).
    reg_decls: Vec<RegDecl>,
    // Collects (lhs_expr, rhs_expr, wire_name) for internal gates (literals/ANDs) for sorted
    // emission.
    internal_wire_assignments: Vec<(Rc<vast::Expr>, Rc<vast::Expr>, String)>,
    // Collects direct assignments to output ports (for non-flopped outputs) or intermediate _comb
    // wires.
    direct_output_assignments: Vec<DirectOutputAssignment>,
}

impl NetlistEmitState {
    fn new() -> Self {
        Self {
            gate_ref_to_vast_expr: BTreeMap::new(),
            output_bit_to_combinational_target_expr: BTreeMap::new(),
            procedural_assignments: Vec::new(),
            final_output_port_assignments: Vec::new(),
            reg_decls: Vec::new(),
            internal_wire_assignments: Vec::new(),
            direct_output_assignments: Vec::new(),
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

fn generate_module_ports_and_registers(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) -> Result<Option<vast::LogicRef>, String> {
    let mut clk_input_ref: Option<vast::LogicRef> = None;
    if config.flop_inputs || config.flop_outputs || config.clk_name.is_some() {
        let final_clk_name = config.clk_name.as_deref().unwrap_or("clk");
        clk_input_ref = Some(module.add_input(final_clk_name, bit_type));
    }

    // Add all the inputs to the module.
    for input_spec in config.gate_fn.inputs.iter() {
        let bit_count = input_spec.bit_vector.get_bit_count();
        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                for (i, aig_bit) in input_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                    let base_name = if bit_count == 1 {
                        input_spec.name.clone()
                    } else {
                        format!("{}_{}", input_spec.name, i)
                    };

                    let input_port_ref = module.add_input(base_name.as_str(), bit_type);

                    if config.flop_inputs {
                        let reg_name = format!("p0_{}", base_name);
                        let input_reg_ref = module.add_reg(reg_name.as_str(), bit_type).unwrap();
                        state
                            .gate_ref_to_vast_expr
                            .insert(aig_bit.node, Rc::new(input_reg_ref.to_expr()));
                        state.reg_decls.push(RegDecl {
                            name: reg_name.clone(),
                            logic_ref: input_reg_ref.clone(),
                        });

                        if clk_input_ref.is_some() {
                            state.procedural_assignments.push(ProceduralAssignment {
                                sort_key: reg_name, // Sort by register name
                                lhs_expr: Rc::new(input_reg_ref.to_expr()),
                                rhs_expr: Rc::new(input_port_ref.to_expr()),
                            });
                        }
                    } else {
                        state
                            .gate_ref_to_vast_expr
                            .insert(aig_bit.node, Rc::new(input_port_ref.to_expr()));
                    }
                }
            }
            NetlistPortStyle::PackedBits => {
                let port_type = port_data_type(file, bit_type, bit_count)?;
                let input_port_ref = module.add_input(input_spec.name.as_str(), &port_type);
                let readable_ref = if config.flop_inputs {
                    let reg_name = format!("p0_{}", input_spec.name);
                    let input_reg_ref = module.add_reg(reg_name.as_str(), &port_type).unwrap();
                    state.reg_decls.push(RegDecl {
                        name: reg_name.clone(),
                        logic_ref: input_reg_ref.clone(),
                    });
                    if clk_input_ref.is_some() {
                        state.procedural_assignments.push(ProceduralAssignment {
                            sort_key: reg_name,
                            lhs_expr: Rc::new(input_reg_ref.to_expr()),
                            rhs_expr: Rc::new(input_port_ref.to_expr()),
                        });
                    }
                    input_reg_ref
                } else {
                    input_port_ref
                };
                for (i, aig_bit) in input_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                    state.gate_ref_to_vast_expr.insert(
                        aig_bit.node,
                        packed_bit_expr(file, &readable_ref, bit_count, i),
                    );
                }
            }
        }
    }

    // Add all the outputs to the module.
    let mut output_specs = config
        .gate_fn
        .outputs
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &gate::Output)>>();
    if config.port_style == NetlistPortStyle::PackedBits {
        output_specs.sort_by(|(_, lhs), (_, rhs)| compare_packed_output_port_order(lhs, rhs));
    }
    for (output_index, output_spec) in output_specs {
        let bit_count = output_spec.bit_vector.get_bit_count();
        match config.port_style {
            NetlistPortStyle::ScalarBits => {
                for (i, _aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                    let base_name = if bit_count == 1 {
                        output_spec.name.clone()
                    } else {
                        format!("{}_{}", output_spec.name, i)
                    };

                    let output_port_ref = module.add_output(base_name.as_str(), bit_type);

                    if config.flop_outputs {
                        let comb_wire_name = format!("{}_comb", base_name);
                        let comb_wire_ref = module.add_wire(&comb_wire_name, bit_type);
                        state
                            .output_bit_to_combinational_target_expr
                            .insert((output_index, i), Rc::new(comb_wire_ref.to_expr()));

                        let reg_name = format!("p0_{}", base_name);
                        let output_reg_ref = module.add_reg(reg_name.as_str(), bit_type).unwrap();
                        state.reg_decls.push(RegDecl {
                            name: reg_name.clone(),
                            logic_ref: output_reg_ref.clone(),
                        });

                        state
                            .final_output_port_assignments
                            .push(FinalOutputAssignment {
                                port_name: base_name.clone(),
                                port_expr: Rc::new(output_port_ref.to_expr()),
                                reg_expr: Rc::new(output_reg_ref.to_expr()),
                            });
                    } else {
                        state
                            .output_bit_to_combinational_target_expr
                            .insert((output_index, i), Rc::new(output_port_ref.to_expr()));
                    }
                }
            }
            NetlistPortStyle::PackedBits => {
                let port_type = port_data_type(file, bit_type, bit_count)?;
                let output_port_ref = module.add_output(output_spec.name.as_str(), &port_type);

                if config.flop_outputs {
                    let comb_wire_name = format!("{}_comb", output_spec.name);
                    let comb_wire_ref = module.add_wire(&comb_wire_name, &port_type);
                    for (i, _aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                        state.output_bit_to_combinational_target_expr.insert(
                            (output_index, i),
                            packed_bit_expr(file, &comb_wire_ref, bit_count, i),
                        );
                    }

                    let reg_name = format!("p0_{}", output_spec.name);
                    let output_reg_ref = module.add_reg(reg_name.as_str(), &port_type).unwrap();
                    state.reg_decls.push(RegDecl {
                        name: reg_name.clone(),
                        logic_ref: output_reg_ref.clone(),
                    });
                    state.procedural_assignments.push(ProceduralAssignment {
                        sort_key: reg_name.clone(),
                        lhs_expr: Rc::new(output_reg_ref.to_expr()),
                        rhs_expr: Rc::new(comb_wire_ref.to_expr()),
                    });
                    state
                        .final_output_port_assignments
                        .push(FinalOutputAssignment {
                            port_name: output_spec.name.clone(),
                            port_expr: Rc::new(output_port_ref.to_expr()),
                            reg_expr: Rc::new(output_reg_ref.to_expr()),
                        });
                } else {
                    for (i, _aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
                        state.output_bit_to_combinational_target_expr.insert(
                            (output_index, i),
                            packed_bit_expr(file, &output_port_ref, bit_count, i),
                        );
                    }
                }
            }
        }
    }
    Ok(clk_input_ref)
}

fn generate_internal_combinational_logic(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) {
    for (idx, gate_node) in config.gate_fn.gates.iter().enumerate() {
        let current_gate_aig_ref = gate::AigRef { id: idx };
        match gate_node {
            gate::AigNode::Input { .. } => {
                continue;
            }
            gate::AigNode::Literal { value, .. } => {
                let gate_name = format!("G{}", idx);
                let actual_wire_ref = module.add_wire(&gate_name, bit_type);
                state
                    .gate_ref_to_vast_expr
                    .insert(current_gate_aig_ref, Rc::new(actual_wire_ref.to_expr()));

                let lhs_expr = Rc::new(actual_wire_ref.to_expr());
                let value_str = if *value { "bits[1]:1" } else { "bits[1]:0" };
                let rhs_literal_expr = Rc::new(
                    file.make_literal(value_str, &xlsynth::ir_value::IrFormatPreference::Binary)
                        .unwrap(),
                );
                state
                    .internal_wire_assignments
                    .push((lhs_expr, rhs_literal_expr, gate_name));
            }
            gate::AigNode::And2 {
                a, b, tags: _tags, ..
            } => {
                let gate_name = format!("G{}", idx);
                let actual_wire_ref = module.add_wire(&gate_name, bit_type);
                state
                    .gate_ref_to_vast_expr
                    .insert(current_gate_aig_ref, Rc::new(actual_wire_ref.to_expr()));

                let lhs_expr = Rc::new(actual_wire_ref.to_expr());

                let expr_a_base = state.gate_ref_to_vast_expr.get(&a.node).unwrap_or_else(|| {
                    panic!(
                        "Missing expression for AND input a: {:?}. Node type: {:?}. Gate_fn.gates len: {}. Current gate being processed G{}",
                        a.node,
                        config.gate_fn.gates.get(a.node.id),
                        config.gate_fn.gates.len(),
                        idx,
                    )
                });
                let expr_b_base = state.gate_ref_to_vast_expr.get(&b.node).unwrap_or_else(|| {
                    panic!(
                        "Missing expression for AND input b: {:?}. Node type: {:?}. Gate_fn.gates len: {}. Current gate being processed G{}",
                        b.node,
                        config.gate_fn.gates.get(b.node.id),
                        config.gate_fn.gates.len(),
                        idx,
                    )
                });

                let final_expr_a = if a.negated {
                    file.make_not(expr_a_base)
                } else {
                    (**expr_a_base).clone()
                };
                let final_expr_b = if b.negated {
                    file.make_not(expr_b_base)
                } else {
                    (**expr_b_base).clone()
                };

                let rhs_and_expr = Rc::new(file.make_bitwise_and(&final_expr_a, &final_expr_b));
                state
                    .internal_wire_assignments
                    .push((lhs_expr, rhs_and_expr, gate_name));
            }
        }
    }
}

fn connect_combinational_logic_to_outputs(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
) {
    for (output_index, output_spec) in config.gate_fn.outputs.iter().enumerate() {
        let bit_count = output_spec.bit_vector.get_bit_count();
        for (i, output_aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
            let assignment_sort_key = match config.port_style {
                NetlistPortStyle::ScalarBits => {
                    if bit_count == 1 {
                        output_spec.name.clone()
                    } else {
                        format!("{}_{}", output_spec.name, i)
                    }
                }
                NetlistPortStyle::PackedBits => format!("{}[{i:020}]", output_spec.name),
            };

            let source_expr_base = state.gate_ref_to_vast_expr.get(&output_aig_bit_ref.node)
                .unwrap_or_else(|| panic!("Missing expression for output source gate: {:?}. Ensure wires are processed before this.", output_aig_bit_ref.node));
            let rhs_expr_from_source = if output_aig_bit_ref.negated {
                file.make_not(source_expr_base)
            } else {
                (**source_expr_base).clone()
            };

            let combinational_target_expr = state.output_bit_to_combinational_target_expr
                .get(&(output_index, i))
                .unwrap_or_else(|| {
                    panic!(
                        "Missing combinational target for output bit: ({}, {}). Ensure output_bit_to_combinational_target_expr is populated correctly for it.",
                        output_index, i
                    )
                });

            state
                .direct_output_assignments
                .push(DirectOutputAssignment {
                    target_name: assignment_sort_key,
                    lhs_expr: combinational_target_expr.clone(),
                    rhs_expr: Rc::new(rhs_expr_from_source),
                });
        }
    }
}

fn generate_sequential_block(
    module: &mut vast::VastModule,
    file: &mut vast::VastFile,
    clk_input_ref: &vast::LogicRef,
    resolved_procedural_assignments: &[ProceduralAssignment],
) {
    if resolved_procedural_assignments.is_empty() {
        return;
    }
    let posedge_clk_expr = file.make_pos_edge(&clk_input_ref.to_expr());
    let ff_block_base = module.add_always_ff(&[&posedge_clk_expr]).unwrap();
    let mut stmt_block = ff_block_base.get_statement_block();
    for pa in resolved_procedural_assignments {
        stmt_block.add_nonblocking_assignment(&pa.lhs_expr, &pa.rhs_expr);
    }
}

pub fn emit_netlist_with_version_and_port_style(
    name: &str,
    gate_fn: &gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    version: VerilogVersion,
    clk_name: Option<String>,
    port_style: NetlistPortStyle,
) -> Result<String, String> {
    if (flop_inputs || flop_outputs) && clk_name.is_none() {
        return Err(
            "emit_netlist: flop_inputs or flop_outputs is true, but no clk_name was provided"
                .to_string(),
        );
    }
    if let Some(ref clk) = clk_name {
        if gate_fn.inputs.iter().any(|input| input.name == *clk) {
            return Err(format!(
                "emit_netlist: requested clock name '{}' collides with an existing input name",
                clk
            ));
        }
    }

    let file_type = if version.is_system_verilog() {
        vast::VastFileType::SystemVerilog
    } else {
        vast::VastFileType::Verilog
    };
    let mut file = vast::VastFile::new(file_type);
    let mut module = file.add_module(name);
    let bit_type = file.make_bit_vector_type(1, false);

    let config = NetlistEmitConfig {
        gate_fn,
        flop_inputs,
        flop_outputs,
        clk_name: clk_name.clone(),
        port_style,
    };
    let mut state = NetlistEmitState::new();

    let clk_input_ref = generate_module_ports_and_registers(
        &config,
        &mut state,
        &mut file,
        &mut module,
        &bit_type,
    )?;
    generate_internal_combinational_logic(&config, &mut state, &mut file, &mut module, &bit_type);

    // Phase 3: Deterministic Emission
    state.reg_decls.sort_by(|a, b| a.name.cmp(&b.name));
    // Regs already added to module. This loop is for ordered side-effects if any
    // (e.g., comments).

    state
        .internal_wire_assignments
        .sort_by(|a, b| a.2.cmp(&b.2)); // Sort by wire name
    for (lhs_expr, rhs_expr, _wire_name) in &state.internal_wire_assignments {
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(lhs_expr, rhs_expr));
    }

    connect_combinational_logic_to_outputs(&config, &mut state, &mut file);

    state
        .direct_output_assignments
        .sort_by(|a, b| a.target_name.cmp(&b.target_name));
    for doa in &state.direct_output_assignments {
        module.add_member_continuous_assignment(
            file.make_continuous_assignment(&doa.lhs_expr, &doa.rhs_expr),
        );
    }

    let mut resolved_procedural_assignments: Vec<ProceduralAssignment> =
        state.procedural_assignments.iter().cloned().collect();

    if config.flop_outputs && config.port_style == NetlistPortStyle::ScalarBits {
        for (output_index, output_spec) in config.gate_fn.outputs.iter().enumerate() {
            let bit_count = output_spec.bit_vector.get_bit_count();
            for (i, _aig_bit_ref_output_driver) in
                output_spec.bit_vector.iter_lsb_to_msb().enumerate()
            {
                let base_output_name = if bit_count == 1 {
                    output_spec.name.clone()
                } else {
                    format!("{}_{}", output_spec.name, i)
                };
                let reg_name = format!("p0_{}", base_output_name);
                let _comb_wire_name = format!("{}_comb", base_output_name);

                if let Some(output_reg_decl) = state.reg_decls.iter().find(|rd| rd.name == reg_name)
                {
                    if let Some(comb_wire_logic_ref) = state
                        .output_bit_to_combinational_target_expr
                        .get(&(output_index, i))
                    {
                        resolved_procedural_assignments.push(ProceduralAssignment {
                            sort_key: reg_name.clone(),
                            lhs_expr: Rc::new(output_reg_decl.logic_ref.to_expr()),
                            rhs_expr: comb_wire_logic_ref.clone(),
                        });
                    } else {
                        panic!(
                            "Could not find combinational target expression for output bit ({}, {}) driving output reg {} to create procedural assignment.",
                            output_index, i, reg_name
                        );
                    }
                }
            }
        }
    }

    resolved_procedural_assignments.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

    if (config.flop_inputs || config.flop_outputs) && !resolved_procedural_assignments.is_empty() {
        if let Some(ref clk_ref) = clk_input_ref {
            generate_sequential_block(
                &mut module,
                &mut file,
                clk_ref,
                &resolved_procedural_assignments,
            );
        }
    }

    state
        .final_output_port_assignments
        .sort_by(|a, b| a.port_name.cmp(&b.port_name));
    for foa in &state.final_output_port_assignments {
        module.add_member_continuous_assignment(
            file.make_continuous_assignment(&foa.port_expr, &foa.reg_expr),
        );
    }

    Ok(file.emit())
}

pub fn emit_netlist_with_version(
    name: &str,
    gate_fn: &gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    version: VerilogVersion,
    clk_name: Option<String>,
) -> Result<String, String> {
    emit_netlist_with_version_and_port_style(
        name,
        gate_fn,
        flop_inputs,
        flop_outputs,
        version,
        clk_name,
        NetlistPortStyle::ScalarBits,
    )
}

/// Backwards-compatibility wrapper: accepts the old `use_system_verilog` bool.
pub fn emit_netlist(
    name: &str,
    gate_fn: &gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    use_system_verilog: bool,
    clk_name: Option<String>,
) -> Result<String, String> {
    let version = if use_system_verilog {
        VerilogVersion::SystemVerilog
    } else {
        VerilogVersion::Verilog
    };
    emit_netlist_with_version(name, gate_fn, flop_inputs, flop_outputs, version, clk_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_inverter_no_flops() {
        let mut g8_builder =
            GateBuilder::new("my_inverter".to_string(), GateBuilderOptions::no_opt());
        let i_ref = g8_builder.add_input("i".to_string(), 1);
        let gate_ref = g8_builder.add_not(*i_ref.get_lsb(0));
        g8_builder.add_output("o".to_string(), gate::AigBitVector::from_bit(gate_ref));

        let netlist = emit_netlist(
            "my_inverter",
            &g8_builder.build(),
            false,
            false,
            false,
            None,
        )
        .unwrap();

        // After refactor, if the NOT operation directly drives an output
        // and the input to NOT is a primary input, no intermediate wire is strictly
        // needed by the AIG structure itself. GateBuilder::add_not returns a
        // negated operand.
        let expected = "module my_inverter(
  input wire i,
  output wire o
);
  wire G0;
  assign G0 = 1'b0;
  assign o = ~i;
endmodule
";
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_and_gate_no_flops() {
        let mut g8_builder =
            GateBuilder::new("my_and_gate".to_string(), GateBuilderOptions::no_opt());
        let i_val = g8_builder.add_input("i".to_string(), 1);
        let j_val = g8_builder.add_input("j".to_string(), 1);
        let o_val = g8_builder.add_and_binary(*i_val.get_lsb(0), *j_val.get_lsb(0));
        g8_builder.add_output("o".to_string(), gate::AigBitVector::from_bit(o_val));

        let netlist = emit_netlist(
            "my_and_gate",
            &g8_builder.build(),
            false,
            false,
            false,
            None,
        )
        .unwrap();
        // G0 = literal false.
        // AND gate (i & j) is, say, G3.
        // assign o = G3.
        // Wire declarations are typically grouped.
        let expected = "module my_and_gate(\n  input wire i,\n  input wire j,\n  output wire o\n);\n  wire G0;\n  wire G3;\n  assign G0 = 1'b0;\n  assign G3 = i & j;\n  assign o = G3;\nendmodule\n";
        assert_eq!(netlist, expected.to_string());
    }

    #[test]
    fn test_emit_packed_vector_ports_no_flops() {
        let mut g8_builder =
            GateBuilder::new("packed_and".to_string(), GateBuilderOptions::no_opt());
        let arg0 = g8_builder.add_input("arg0".to_string(), 4);
        let arg1 = g8_builder.add_input("arg1".to_string(), 4);
        let output_value = g8_builder.add_and_vec(&arg0, &arg1);
        g8_builder.add_output("output_value".to_string(), output_value);

        let netlist = emit_netlist_with_version_and_port_style(
            "packed_and",
            &g8_builder.build(),
            false,
            false,
            VerilogVersion::SystemVerilog,
            None,
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
        let mut g8_builder = GateBuilder::new(
            "packed_tuple_order".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let arg0 = g8_builder.add_input("arg0".to_string(), 1);
        let output_bit = *arg0.get_lsb(0);

        for index in (0..11).rev() {
            g8_builder.add_output(
                format!("output_value_{index}"),
                gate::AigBitVector::from_bit(output_bit),
            );
        }

        let netlist = emit_netlist_with_version_and_port_style(
            "packed_tuple_order",
            &g8_builder.build(),
            false,
            false,
            VerilogVersion::SystemVerilog,
            None,
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
        let mut g8_builder =
            GateBuilder::new("packed_flop".to_string(), GateBuilderOptions::no_opt());
        let arg0 = g8_builder.add_input("arg0".to_string(), 2);
        let output_bits = arg0
            .iter_lsb_to_msb()
            .map(|bit| g8_builder.add_not(*bit))
            .collect::<Vec<gate::AigOperand>>();
        g8_builder.add_output(
            "output_value".to_string(),
            gate::AigBitVector::from_lsb_is_index_0(&output_bits),
        );

        let netlist = emit_netlist_with_version_and_port_style(
            "packed_flop",
            &g8_builder.build(),
            false,
            true,
            VerilogVersion::SystemVerilog,
            Some("clk".to_string()),
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
        let mut g8_builder =
            GateBuilder::new("my_flop_inv".to_string(), GateBuilderOptions::no_opt());
        let i_val = g8_builder.add_input("i".to_string(), 1);
        let o_val = g8_builder.add_not(*i_val.get_lsb(0));
        g8_builder.add_output("o".to_string(), gate::AigBitVector::from_bit(o_val));
        let netlist = emit_netlist(
            "my_flop_inv",
            &g8_builder.build(),
            true,
            true,
            false,
            Some("clk".to_string()),
        )
        .unwrap();

        let expected_netlist = r#"module my_flop_inv(
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
        assert_eq!(netlist, expected_netlist.to_string());
    }
}
