// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "support/cases.rs"]
mod cases;
mod support;
use cases::*;
use std::time::Duration;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_g8r::netlist::yosys::YosysToolchain;

#[test]
fn one_hot_case_functions_prove_equivalent_to_prefix_logic_with_yosys() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    let directory = tempfile::tempdir().unwrap();
    for width in [1, 4, 8, 65, 256] {
        let ir = format!(
            r#"package public_hot_proof

top block hot_proof(value: bits[{width}], lsb: bits[{out_width}], msb: bits[{out_width}]) {{
  value: bits[{width}] = input_port(name=value, id=1)
  low: bits[{out_width}] = one_hot(value, lsb_prio=true, id=2)
  high: bits[{out_width}] = one_hot(value, lsb_prio=false, id=3)
  lsb: () = output_port(low, name=lsb, id=4)
  msb: () = output_port(high, name=msb, id=5)
}}
"#,
            out_width = width + 1
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let mut source = format!(
            r#"{generated}
module proof(input logic [{last}:0] value, output logic equal);
  wire [{width}:0] actual_lsb, actual_msb, expected_lsb, expected_msb;
  hot_proof dut(.value(value), .lsb(actual_lsb), .msb(actual_msb));
  assign expected_lsb[{width}] = ~(|value);
  assign expected_msb[{width}] = ~(|value);
  assign equal = actual_lsb == expected_lsb && actual_msb == expected_msb;
"#,
            last = width - 1
        );
        for bit in 0..width {
            let low = if bit == 0 {
                format!("value[{bit}]")
            } else {
                format!("value[{bit}] && !(|value[{}:0])", bit - 1)
            };
            let high = if bit == width - 1 {
                format!("value[{bit}]")
            } else {
                format!("value[{bit}] && !(|value[{}:{}])", width - 1, bit + 1)
            };
            source.push_str(&format!(
                "assign expected_lsb[{bit}] = {low};\nassign expected_msb[{bit}] = {high};\n"
            ));
        }
        source.push_str("endmodule\n");
        std::fs::write(directory.path().join("proof.sv"), source).unwrap();
        yosys.run_script(directory.path(), "read_verilog -sv proof.sv; hierarchy -check -top proof; proc; flatten; opt; sat -verify -prove equal 1 -show-inputs -show-outputs", Duration::from_secs(60)).unwrap_or_else(|e| panic!("one_hot width={width}: {e}"));
    }
}
