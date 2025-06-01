// SPDX-License-Identifier: Apache-2.0

//! Uses the xlsynth VAST APIs to build a netlist verilog representation of the
//! given gates.

use crate::gate;
use std::collections::BTreeMap;
use std::rc::Rc;
use xlsynth::vast;

#[derive(Clone)] // vast::LogicRef is Clone, String is Clone
struct RegDecl {
    name: String,
    logic_ref: vast::LogicRef,
}

#[derive(Clone)]
struct WireInfo {
    name: String,
    rhs_expr: Option<Rc<vast::Expr>>, // For literals or pre-computed RHS
    op_a: Option<gate::AigOperand>,   // For AND gates
    op_b: Option<gate::AigOperand>,   // For AND gates
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

// Configuration for netlist emission (immutable after creation)
struct NetlistEmitConfig<'g> {
    gate_fn: &'g gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    clk_name: Option<String>,
}

// Mutable state collected during netlist emission
struct NetlistEmitState {
    // Maps an AIG reference (typically an internal gate or flopped input) to its VAST logic
    // representation (wire/reg).
    gate_ref_to_vast_logic: BTreeMap<gate::AigRef, vast::LogicRef>,
    // Maps an AIG operand driving an output to its VAST logic representation (output port or _comb
    // wire).
    output_bit_to_combinational_target_ref: BTreeMap<gate::AigOperand, vast::LogicRef>,
    // Collects procedural assignments for always_ff blocks (e.g., p0_input <= input_port,
    // p0_output_reg <= o_comb_wire).
    procedural_assignments: Vec<ProceduralAssignment>,
    // Collects assignments for final output ports when outputs are flopped (e.g., assign
    // output_port = p0_output_reg).
    final_output_port_assignments: Vec<FinalOutputAssignment>,
    // Collects declarations for all registers (p0_ inputs/outputs).
    reg_decls: Vec<RegDecl>,
    // Collects information needed to declare wires and their driving expressions (literals or AND
    // gates).
    wire_info_collector: Vec<WireInfo>,
    // Collects direct assignments to output ports (for non-flopped outputs) or intermediate _comb
    // wires.
    direct_output_assignments: Vec<DirectOutputAssignment>,
    // Maps an AIG reference (for an internal gate like G0, G1) to its intended wire name.
    aig_ref_to_wire_name: BTreeMap<gate::AigRef, String>,
}

impl NetlistEmitState {
    fn new() -> Self {
        Self {
            gate_ref_to_vast_logic: BTreeMap::new(),
            output_bit_to_combinational_target_ref: BTreeMap::new(),
            procedural_assignments: Vec::new(),
            final_output_port_assignments: Vec::new(),
            reg_decls: Vec::new(),
            wire_info_collector: Vec::new(),
            direct_output_assignments: Vec::new(),
            aig_ref_to_wire_name: BTreeMap::new(),
        }
    }
}

fn generate_module_ports_and_registers(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    module: &mut vast::VastModule,
    bit_type: &vast::VastDataType,
) -> Option<vast::LogicRef> {
    let mut clk_input_ref: Option<vast::LogicRef> = None;
    if config.flop_inputs || config.flop_outputs || config.clk_name.is_some() {
        let final_clk_name = config.clk_name.as_deref().unwrap_or("clk");
        clk_input_ref = Some(module.add_input(final_clk_name, bit_type));
    }

    // Add all the inputs to the module.
    for input_spec in config.gate_fn.inputs.iter() {
        let bit_count = input_spec.bit_vector.get_bit_count();
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
                    .gate_ref_to_vast_logic
                    .insert(aig_bit.node, input_reg_ref.clone());
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
                    .gate_ref_to_vast_logic
                    .insert(aig_bit.node, input_port_ref);
            }
        }
    }

    // Add all the outputs to the module.
    for output_spec in config.gate_fn.outputs.iter() {
        let bit_count = output_spec.bit_vector.get_bit_count();
        for (i, aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
            let base_name = if bit_count == 1 {
                output_spec.name.clone()
            } else {
                format!("{}_{}", output_spec.name, i)
            };

            let output_port_ref = module.add_output(base_name.as_str(), bit_type);

            if config.flop_outputs {
                let comb_wire_name = format!("{}_comb", base_name);
                state.wire_info_collector.push(WireInfo {
                    name: comb_wire_name.clone(),
                    rhs_expr: None,
                    op_a: None,
                    op_b: None,
                });

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
                    .output_bit_to_combinational_target_ref
                    .insert(*aig_bit_ref, output_port_ref.clone());
            }
        }
    }
    clk_input_ref
}

fn generate_internal_combinational_logic(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
) {
    for (idx, gate_node) in config.gate_fn.gates.iter().enumerate() {
        let current_gate_aig_ref = gate::AigRef { id: idx };
        match gate_node {
            gate::AigNode::Input { .. } => {
                continue;
            }
            gate::AigNode::Literal(value) => {
                let gate_name = format!("G{}", idx);
                let value_str = if *value { "bits[1]:1" } else { "bits[1]:0" };
                let literal_node = file
                    .make_literal(value_str, &xlsynth::ir_value::IrFormatPreference::Binary)
                    .unwrap();
                state.wire_info_collector.push(WireInfo {
                    name: gate_name.clone(),
                    rhs_expr: Some(Rc::new(literal_node)),
                    op_a: None,
                    op_b: None,
                });
                state
                    .aig_ref_to_wire_name
                    .insert(current_gate_aig_ref, gate_name);
            }
            gate::AigNode::And2 { a, b, tags: _tags } => {
                let gate_name = format!("G{}", idx);
                state.wire_info_collector.push(WireInfo {
                    name: gate_name.clone(),
                    rhs_expr: None,
                    op_a: Some(*a),
                    op_b: Some(*b),
                });
                state
                    .aig_ref_to_wire_name
                    .insert(current_gate_aig_ref, gate_name);
            }
        }
    }
}

fn connect_combinational_logic_to_outputs(
    config: &NetlistEmitConfig,
    state: &mut NetlistEmitState,
    file: &mut vast::VastFile,
) {
    for output_spec in config.gate_fn.outputs.iter() {
        let bit_count = output_spec.bit_vector.get_bit_count();
        for (i, output_aig_bit_ref) in output_spec.bit_vector.iter_lsb_to_msb().enumerate() {
            let base_name = if bit_count == 1 {
                output_spec.name.clone()
            } else {
                format!("{}_{}", output_spec.name, i)
            };

            let source_gate_logic_ref = state.gate_ref_to_vast_logic.get(&output_aig_bit_ref.node)
                .unwrap_or_else(|| panic!("Missing LogicRef for output source gate: {:?}. Ensure wires are processed before this.", output_aig_bit_ref.node));

            let source_expr_base = source_gate_logic_ref.to_expr();
            let rhs_expr_from_source = if output_aig_bit_ref.negated {
                file.make_not(&source_expr_base)
            } else {
                source_expr_base
            };

            let combinational_target_ref = state.output_bit_to_combinational_target_ref
                .get(output_aig_bit_ref)
                .unwrap_or_else(|| {
                    panic!(
                        "Missing combinational target for output bit: {:?}. Ensure output_bit_to_combinational_target_ref is populated correctly for it.",
                        output_aig_bit_ref
                    )
                });

            state
                .direct_output_assignments
                .push(DirectOutputAssignment {
                    target_name: base_name,
                    lhs_expr: Rc::new(combinational_target_ref.to_expr()),
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

pub fn emit_netlist(
    name: &str,
    gate_fn: &gate::GateFn,
    flop_inputs: bool,
    flop_outputs: bool,
    use_system_verilog: bool,
    clk_name: Option<String>,
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

    let file_type = if use_system_verilog {
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
    };
    let mut state = NetlistEmitState::new();

    let clk_input_ref =
        generate_module_ports_and_registers(&config, &mut state, &mut module, &bit_type);
    generate_internal_combinational_logic(&config, &mut state, &mut file);

    // Phase 3: Deterministic Emission
    state.reg_decls.sort_by(|a, b| a.name.cmp(&b.name));
    // Regs already added to module. This loop is for ordered side-effects if any
    // (e.g., comments).

    state
        .wire_info_collector
        .sort_by(|a, b| a.name.cmp(&b.name));
    let mut wire_assignments_to_emit: Vec<(Rc<vast::Expr>, Rc<vast::Expr>)> = Vec::new();

    for winfo in &state.wire_info_collector {
        let actual_wire_ref = module.add_wire(&winfo.name, &bit_type);

        if let Some(aig_ref) = state.aig_ref_to_wire_name.iter().find_map(|(key, val)| {
            if val == &winfo.name {
                Some(*key)
            } else {
                None
            }
        }) {
            state
                .gate_ref_to_vast_logic
                .insert(aig_ref, actual_wire_ref.clone());
        }

        if winfo.name.ends_with("_comb") {
            for output_spec in config.gate_fn.outputs.iter() {
                let bit_count = output_spec.bit_vector.get_bit_count();
                for (i, aig_bit_ref_in_output) in
                    output_spec.bit_vector.iter_lsb_to_msb().enumerate()
                {
                    let base_output_name = if bit_count == 1 {
                        output_spec.name.clone()
                    } else {
                        format!("{}_{}", output_spec.name, i)
                    };
                    if format!("{}_comb", base_output_name) == winfo.name {
                        state
                            .output_bit_to_combinational_target_ref
                            .insert(*aig_bit_ref_in_output, actual_wire_ref.clone());
                    }
                }
            }
        }

        let lhs_expr = Rc::new(actual_wire_ref.to_expr());

        if let Some(ref rhs_literal_expr) = winfo.rhs_expr {
            wire_assignments_to_emit.push((lhs_expr.clone(), rhs_literal_expr.clone()));
        } else if let (Some(op_a), Some(op_b)) = (winfo.op_a, winfo.op_b) {
            let ref_a = state
                .gate_ref_to_vast_logic
                .get(&op_a.node)
                .unwrap_or_else(|| panic!("Missing LogicRef for AND input a: {:?}", op_a.node));
            let ref_b = state
                .gate_ref_to_vast_logic
                .get(&op_b.node)
                .unwrap_or_else(|| panic!("Missing LogicRef for AND input b: {:?}", op_b.node));

            let expr_a_base = ref_a.to_expr();
            let expr_b_base = ref_b.to_expr();

            let final_expr_a = if op_a.negated {
                file.make_not(&expr_a_base)
            } else {
                expr_a_base
            };
            let final_expr_b = if op_b.negated {
                file.make_not(&expr_b_base)
            } else {
                expr_b_base
            };

            let rhs_and_expr = Rc::new(file.make_bitwise_and(&final_expr_a, &final_expr_b));
            wire_assignments_to_emit.push((lhs_expr.clone(), rhs_and_expr));
        }
    }

    connect_combinational_logic_to_outputs(&config, &mut state, &mut file);

    for (lhs_expr, rhs_expr) in wire_assignments_to_emit {
        module.add_member_continuous_assignment(
            file.make_continuous_assignment(&lhs_expr, &rhs_expr),
        );
    }

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

    if config.flop_outputs {
        for output_spec in config.gate_fn.outputs.iter() {
            let bit_count = output_spec.bit_vector.get_bit_count();
            for (i, aig_bit_ref_output_driver) in
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
                        .output_bit_to_combinational_target_ref
                        .get(aig_bit_ref_output_driver)
                    {
                        resolved_procedural_assignments.push(ProceduralAssignment {
                            sort_key: reg_name.clone(),
                            lhs_expr: Rc::new(output_reg_decl.logic_ref.to_expr()),
                            rhs_expr: Rc::new(comb_wire_logic_ref.to_expr()),
                        });
                    } else {
                        panic!("Could not find combinational target LogicRef for AigOperand {:?} (driving output for reg {}) to create procedural assignment.", aig_bit_ref_output_driver, reg_name);
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
  reg p0_o;
  wire G0;
  wire o_comb;
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
