// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use xlsynth_g8r::emit_netlist::emit_netlist;
    use xlsynth_g8r::test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt};
    // Use pretty_assertions if detailed diffs are needed in the future, but remove
    // for now as not strictly used. use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_bf16_add_with_flops() {
        let gate_fn = load_bf16_add_sample(Opt::No).gate_fn;
        let result = emit_netlist(
            "bf16_add_flopped",
            &gate_fn,
            true,  // flop_inputs
            true,  // flop_outputs
            false, // use_system_verilog
            Some("clk".to_string()),
        );

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
        let result = emit_netlist(
            "bf16_mul_flopped",
            &gate_fn,
            true,  // flop_inputs
            true,  // flop_outputs
            false, // use_system_verilog
            Some("clk".to_string()),
        );

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
