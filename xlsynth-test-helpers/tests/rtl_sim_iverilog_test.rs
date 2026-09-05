// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::IrBits;
use xlsynth_test_helpers::rtl_sim::{Bindings, Icarus, Interface, LogicValue, Port, StateSignal};

#[test]
fn icarus_persistent_vectors_preserve_wide_packed_values_and_four_states() {
    let interface = Interface {
        module: "wide".into(),
        inputs: vec![Port {
            name: "data".into(),
            width: 129,
        }],
        outputs: vec![Port {
            name: "result".into(),
            width: 129,
        }],
        clock: None,
        state: vec![],
    };
    let source = "module wide(input logic [2:0][42:0] data, output logic [2:0][42:0] result); assign result = data; endmodule";
    let mut simulator = Icarus::new(source, interface).unwrap();
    for value in [
        LogicValue::parse_binary(&format!("1{}01", "0".repeat(126)), 129).unwrap(),
        LogicValue::parse_binary(&format!("z{}x0", "1".repeat(126)), 129).unwrap(),
        LogicValue::from_u64(129, 42),
    ] {
        let output = simulator
            .evaluate(&Bindings::from([("data".into(), value.clone())]))
            .unwrap();
        assert_eq!(output.outputs["result"], value);
        assert!(output.state.is_empty());
    }
    let bad_width = Bindings::from([("data".into(), LogicValue::from_u64(1, 0))]);
    assert!(
        simulator
            .evaluate(&bad_width)
            .unwrap_err()
            .to_string()
            .contains("wrong width for `data`")
    );
    assert!(
        simulator
            .cycle(&bad_width)
            .unwrap_err()
            .to_string()
            .contains("requires a clock")
    );
}

#[test]
fn icarus_array_indices_keep_self_determined_expression_widths() {
    let source = r#"module indices(input logic bit_index, output logic [15:0] result);
      wire [7:0] elements [0:1];
      assign elements[0] = 8'h35;
      assign elements[1] = 8'ha7;
      assign result = {elements[~bit_index], elements[bit_index + 1'b1]};
    endmodule"#;
    let interface = Interface {
        module: "indices".into(),
        inputs: vec![Port {
            name: "bit_index".into(),
            width: 1,
        }],
        outputs: vec![Port {
            name: "result".into(),
            width: 16,
        }],
        clock: None,
        state: vec![],
    };
    let mut simulator = Icarus::new(source, interface).unwrap();
    for (index, expected) in [(0, 0xa7a7), (1, 0x3535)] {
        let actual = simulator
            .evaluate(&Bindings::from([(
                "bit_index".into(),
                LogicValue::from_u64(1, index),
            )]))
            .unwrap();
        assert_eq!(actual.outputs["result"], LogicValue::from_u64(16, expected));
    }
}

#[test]
fn dut_stdout_cannot_impersonate_harness_responses() {
    let source = r#"module noisy(input logic data, output logic result);
      initial $display("RTL_RESULT 0");
      assign result = data;
    endmodule"#;
    let interface = Interface {
        module: "noisy".into(),
        inputs: vec![Port {
            name: "data".into(),
            width: 1,
        }],
        outputs: vec![Port {
            name: "result".into(),
            width: 1,
        }],
        clock: None,
        state: vec![],
    };
    let mut simulator = Icarus::new(source, interface).unwrap();
    let snapshot = simulator
        .evaluate(&Bindings::from([(
            "data".into(),
            LogicValue::from_u64(1, 1),
        )]))
        .unwrap();
    assert_eq!(snapshot.outputs["result"], LogicValue::from_u64(1, 1));
}

#[test]
fn unterminated_dut_stdout_does_not_hide_harness_responses() {
    let source = r#"module noisy_write(input logic data, output logic result);
      always @(data) $write("diagnostic without newline");
      assign result = data;
    endmodule"#;
    let interface = Interface {
        module: "noisy_write".into(),
        inputs: vec![Port {
            name: "data".into(),
            width: 1,
        }],
        outputs: vec![Port {
            name: "result".into(),
            width: 1,
        }],
        clock: None,
        state: vec![],
    };
    let mut simulator = Icarus::new(source, interface).unwrap();
    for value in [0, 1] {
        let snapshot = simulator
            .evaluate(&Bindings::from([(
                "data".into(),
                LogicValue::from_u64(1, value),
            )]))
            .unwrap();
        assert_eq!(snapshot.outputs["result"], LogicValue::from_u64(1, value));
    }
}

#[test]
fn icarus_state_initialization_and_sampling_follow_nonblocking_clock_updates() {
    let source = r#"module stateful(input logic clk, input logic [7:0] data,
      input logic enable, input logic reset, output logic [7:0] result);
      logic [7:0] r;
      assign result = r;
      always @(posedge clk) if (reset) r <= 8'h42; else if (enable) r <= data;
    endmodule"#;
    let interface = Interface {
        module: "stateful".into(),
        inputs: vec![
            Port {
                name: "data".into(),
                width: 8,
            },
            Port {
                name: "enable".into(),
                width: 1,
            },
            Port {
                name: "reset".into(),
                width: 1,
            },
        ],
        outputs: vec![Port {
            name: "result".into(),
            width: 8,
        }],
        clock: Some("clk".into()),
        state: vec![StateSignal {
            name: "r".into(),
            width: 8,
            expression: "dut.r".into(),
        }],
    };
    let mut simulator = Icarus::new(source, interface).unwrap();
    simulator
        .set_state(&Bindings::from([(
            "r".into(),
            LogicValue::from_u64(8, 0x31),
        )]))
        .unwrap();
    let mut previous = 0x31;
    for (data, enable, reset, expected) in
        [(0x55, 1, 0, 0x55), (0xff, 0, 0, 0x55), (0xaa, 1, 1, 0x42)]
    {
        let inputs = Bindings::from([
            ("data".into(), LogicValue::from_u64(8, data)),
            ("enable".into(), LogicValue::from_u64(1, enable)),
            ("reset".into(), LogicValue::from_u64(1, reset)),
        ]);
        assert_eq!(
            simulator.evaluate(&inputs).unwrap().outputs["result"],
            LogicValue::from_u64(8, previous)
        );
        let after = simulator.cycle(&inputs).unwrap();
        assert_eq!(after.outputs["result"], LogicValue::from_u64(8, expected));
        assert_eq!(after.state["r"], LogicValue::from_u64(8, expected));
        previous = expected;
    }
}

#[test]
fn icarus_compile_errors_include_source_and_diagnostics() {
    let interface = Interface {
        module: "broken".into(),
        inputs: vec![],
        outputs: vec![],
        clock: None,
        state: vec![],
    };
    let error = Icarus::new("module broken(; endmodule", interface)
        .err()
        .unwrap();
    assert!(error.to_string().contains("iverilog failed"), "{error}");
    assert!(
        error.to_string().contains("module broken(; endmodule"),
        "{error}"
    );
}

#[test]
fn icarus_preserves_wide_ports_without_parsing_sv_in_rust() {
    let source = r#"module port_names(input logic [128:0] data_in, output logic [128:0] result$out);
      assign result$out = ~data_in;
    endmodule"#;
    let interface = Interface {
        module: "port_names".into(),
        inputs: vec![Port {
            name: "data_in".into(),
            width: 129,
        }],
        outputs: vec![Port {
            name: "result$out".into(),
            width: 129,
        }],
        clock: None,
        state: vec![],
    };
    let input_bits = IrBits::from_lsb_is_0(&(0..129).map(|i| i % 5 == 0).collect::<Vec<_>>());
    let expected_bits = IrBits::from_lsb_is_0(&(0..129).map(|i| i % 5 != 0).collect::<Vec<_>>());
    let inputs = Bindings::from([("data_in".into(), LogicValue::from_bits(&input_bits))]);
    let actual = Icarus::new(source, interface.clone())
        .unwrap()
        .evaluate(&inputs)
        .unwrap()
        .outputs;
    assert_eq!(actual["result$out"], LogicValue::from_bits(&expected_bits));
}
