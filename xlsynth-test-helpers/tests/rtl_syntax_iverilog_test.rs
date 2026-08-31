// SPDX-License-Identifier: Apache-2.0

//! Direct Icarus evaluation of SystemVerilog constructs and port spelling.

use xlsynth_test_helpers::rtl_sim::{Bindings, Icarus, Interface, LogicValue, Port};

/// Supplies the testbench interface independently of the SystemVerilog parser.
fn evaluate(source: &str, module: &str, inputs: &Bindings, output: &str, width: usize) -> Bindings {
    let interface = Interface {
        module: module.into(),
        inputs: inputs
            .iter()
            .map(|(name, value)| Port {
                name: name.clone(),
                width: value.width(),
            })
            .collect(),
        outputs: vec![Port {
            name: output.into(),
            width,
        }],
        clock: None,
        state: vec![],
    };
    Icarus::new(source, interface)
        .unwrap()
        .evaluate(inputs)
        .unwrap()
        .outputs
}

#[test]
fn icarus_accepts_systemverilog_width_casts() {
    let source = r#"module selection(
  input wire [3:0] value,
  output wire result
);
  assign result = 1'(value >> 2);
endmodule
"#;
    let inputs = [("value".to_owned(), LogicValue::from_u64(4, 0b0110))]
        .into_iter()
        .collect();
    let outputs = evaluate(source, "selection", &inputs, "result", 1);
    assert_eq!(outputs["result"].to_u64_if_known(), Some(1));
}

#[test]
fn icarus_preserves_input_and_output_names_with_underscores() {
    let source = r#"module public_oracle(
  input wire [7:0] left_value,
  input wire [7:0] _right__2,
  output wire [7:0] combined_value_3
);
  assign combined_value_3 = left_value + _right__2;
endmodule
"#;
    let inputs = [
        ("left_value".to_owned(), LogicValue::from_u64(8, 0x40)),
        ("_right__2".to_owned(), LogicValue::from_u64(8, 0x12)),
    ]
    .into_iter()
    .collect();
    let outputs = evaluate(source, "public_oracle", &inputs, "combined_value_3", 8);
    assert_eq!(outputs["combined_value_3"].to_u64_if_known(), Some(0x52));
}

#[test]
fn icarus_preserves_legal_embedded_dollar_identifiers() {
    let source = r#"module public_oracle(
  input wire [7:0] left$value_1,
  output wire [7:0] result$value_2
);
  assign result$value_2 = left$value_1 + 8'h01;
endmodule
"#;
    let inputs = [("left$value_1".to_owned(), LogicValue::from_u64(8, 0x41))]
        .into_iter()
        .collect();
    let outputs = evaluate(source, "public_oracle", &inputs, "result$value_2", 8);
    assert_eq!(outputs["result$value_2"].to_u64_if_known(), Some(0x42));
}
