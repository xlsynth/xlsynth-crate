// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use xlsynth_g8r::aig::{
        self, ClockPort, GateFn, SequentialGateFn, add_input_registers, add_output_registers,
    };
    use xlsynth_g8r::aig_serdes::emit_netlist::{
        NetlistPortStyle, emit_netlist, emit_netlist_with_version_and_port_style,
    };
    use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
    use xlsynth_g8r::test_utils::{Opt, load_bf16_add_sample, load_bf16_mul_sample};
    use xlsynth_g8r::verilog_version::VerilogVersion;
    use xlsynth_test_helpers::compare_golden_sv;
    // Use pretty_assertions if detailed diffs are needed in the future, but remove
    // for now as not strictly used. use pretty_assertions::assert_eq;

    fn make_packed_inverter_gate_fn(bit_count: usize) -> GateFn {
        let mut g8_builder =
            GateBuilder::new("packed_inverter".to_string(), GateBuilderOptions::no_opt());
        let arg0 = g8_builder.add_input("arg0".to_string(), bit_count);
        let output_bits = arg0
            .iter_lsb_to_msb()
            .map(|bit| g8_builder.add_not(*bit))
            .collect::<Vec<aig::AigOperand>>();
        g8_builder.add_output(
            "output_value".to_string(),
            aig::AigBitVector::from_lsb_is_index_0(&output_bits),
        );
        g8_builder.build()
    }

    fn wrap(
        mut design: SequentialGateFn,
        name: &str,
        flop_inputs: bool,
        flop_outputs: bool,
    ) -> SequentialGateFn {
        design.name = name.to_string();
        let clock = ClockPort {
            name: "clk".to_string(),
        };
        if flop_inputs {
            design = add_input_registers(&design, clock.clone()).unwrap();
        }
        if flop_outputs {
            design = add_output_registers(&design, clock).unwrap();
        }
        design
    }

    #[test]
    fn test_emit_packed_flop_inputs_golden() {
        let design = wrap(
            SequentialGateFn::from_gate_fn(make_packed_inverter_gate_fn(2)),
            "packed_flop_inputs",
            true,
            false,
        );
        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        compare_golden_sv(
            &netlist,
            "tests/goldens/emit_netlist_packed_flop_inputs.golden.sv",
        );
    }

    #[test]
    fn test_emit_packed_flop_outputs_golden() {
        let design = wrap(
            SequentialGateFn::from_gate_fn(make_packed_inverter_gate_fn(2)),
            "packed_flop_outputs",
            false,
            true,
        );
        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        compare_golden_sv(
            &netlist,
            "tests/goldens/emit_netlist_packed_flop_outputs.golden.sv",
        );
    }

    #[test]
    fn test_emit_packed_flop_inputs_outputs_golden() {
        let design = wrap(
            SequentialGateFn::from_gate_fn(make_packed_inverter_gate_fn(2)),
            "packed_flop_inputs_outputs",
            true,
            true,
        );
        let netlist = emit_netlist_with_version_and_port_style(
            &design,
            VerilogVersion::SystemVerilog,
            NetlistPortStyle::PackedBits,
        )
        .unwrap();

        compare_golden_sv(
            &netlist,
            "tests/goldens/emit_netlist_packed_flop_inputs_outputs.golden.sv",
        );
    }

    #[test]
    fn test_emit_bf16_add_with_flops() {
        let gate_fn = load_bf16_add_sample(Opt::No).gate_fn;
        let design = wrap(
            SequentialGateFn::from_gate_fn(gate_fn),
            "bf16_add_flopped",
            true,
            true,
        );
        let result = emit_netlist(&design, false);

        assert!(
            result.is_ok(),
            "emit_netlist for bf16_add failed: {:?}",
            result.err()
        );
        if let Ok(verilog) = result {
            // println!("bf16_add_flopped Verilog length: {}", verilog.len()); // Optional:
            // for debugging
            assert!(
                verilog.len() > 100,
                "Generated Verilog for bf16_add_flopped seems too short"
            );
            assert!(
                verilog.contains("module bf16_add_flopped("),
                "Missing module definition for bf16_add_flopped"
            );
            assert!(
                verilog.contains("input wire clk,"),
                "Missing clk input for bf16_add_flopped"
            );
            assert!(
                verilog.contains("always_ff @ (posedge clk)"),
                "Missing always_ff block for bf16_add_flopped"
            );
        }
    }

    #[test]
    fn test_emit_bf16_mul_with_flops() {
        let gate_fn = load_bf16_mul_sample(Opt::No).gate_fn;
        let design = wrap(
            SequentialGateFn::from_gate_fn(gate_fn),
            "bf16_mul_flopped",
            true,
            true,
        );
        let result = emit_netlist(&design, false);

        assert!(
            result.is_ok(),
            "emit_netlist for bf16_mul failed: {:?}",
            result.err()
        );
        if let Ok(verilog) = result {
            // println!("bf16_mul_flopped Verilog length: {}", verilog.len()); // Optional:
            // for debugging
            assert!(
                verilog.len() > 100,
                "Generated Verilog for bf16_mul_flopped seems too short"
            );
            assert!(
                verilog.contains("module bf16_mul_flopped("),
                "Missing module definition for bf16_mul_flopped"
            );
            assert!(
                verilog.contains("input wire clk,"),
                "Missing clk input for bf16_mul_flopped"
            );
            assert!(
                verilog.contains("always_ff @ (posedge clk)"),
                "Missing always_ff block for bf16_mul_flopped"
            );
        }
    }
}
