// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;
use xlsynth_g8r::aig::{
    AigBitVector, ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId,
    TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::blif::{emit_blif, emit_gate_fn_blif, parse_blif, parse_gate_fn_blif};
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::test_utils::{
    interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case, structurally_equivalent,
};

fn make_pipeline_design() -> SequentialGateFn {
    let mut builder = GateBuilder::new(
        "pipeline__transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let data = builder.add_input("data".to_string(), 2);
    let state_q = builder.add_input("state__q".to_string(), 2);
    let next_low = builder.add_and_binary(*data.get_lsb(0), *state_q.get_lsb(0));
    let state_d = AigBitVector::from_lsb_is_index_0(&[next_low, *data.get_lsb(1)]);
    builder.add_output("out".to_string(), state_q);
    builder.add_output("state__d".to_string(), state_d);
    SequentialGateFn::new(
        "pipeline".to_string(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort {
            name: "clk_main".to_string(),
        }),
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(1),
            initial_value: Some(IrBits::make_ubits(2, 3).unwrap()),
        }],
    )
    .unwrap()
}

#[test]
fn sequential_blif_emission_matches_golden_and_roundtrips() {
    let design = make_pipeline_design();
    let text = emit_blif(&design).unwrap();
    assert!(!text.contains("xlsynth-g8r"));
    assert_eq!(
        text,
        include_str!("goldens/sequential_pipeline.golden.blif")
    );

    let parsed = parse_blif(&text).unwrap();
    assert_eq!(parsed.name, design.name);
    assert_eq!(parsed.inputs, design.inputs);
    assert_eq!(parsed.outputs, design.outputs);
    assert_eq!(parsed.clock, design.clock);
    assert_eq!(parsed.registers, design.registers);
    assert!(structurally_equivalent(
        &parsed.transition,
        &design.transition
    ));
}

#[test]
fn sequential_blif_imports_abc_compact_latches() {
    let design = make_pipeline_design();
    let text = emit_blif(&design)
        .unwrap()
        .replace("state_next[0]", "n12")
        .replace("state_next[1]", "n17")
        .replace(
            ".latch n12 state_reg[0] re clk_main 1",
            ".latch n12 state_reg[0] 1",
        )
        .replace(
            ".latch n17 state_reg[1] re clk_main 1",
            ".latch n17 state_reg[1] 1",
        );

    let parsed = parse_blif(&text).unwrap();
    assert_eq!(parsed.clock, design.clock);
    assert_eq!(parsed.registers, design.registers);
    assert!(structurally_equivalent(
        &parsed.transition,
        &design.transition
    ));
}

#[test]
fn block_lowering_and_blif_import_share_generated_endpoint_names() {
    let block_ir = r#"package endpoint_names

top block pipe(clk: clock, state__q: bits[1], state__d: bits[1]) {
  reg state(bits[1])
  state__q: bits[1] = input_port(name=state__q, id=1)
  state_q: bits[1] = register_read(register=state, id=2)
  write: () = register_write(state__q, register=state, id=3)
  state__d: () = output_port(state_q, name=state__d, id=4)
}
"#;
    let lowered = block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled())
        .expect("block lowering should succeed");
    assert_eq!(lowered.transition.name, "pipe__transition");
    assert_eq!(
        lowered
            .transition
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        vec!["state__q", "state__q__1"]
    );
    assert_eq!(
        lowered
            .transition
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["state__d", "state__d__1"]
    );

    let parsed = parse_blif(&emit_blif(&lowered).unwrap()).unwrap();
    assert_eq!(parsed.transition.name, lowered.transition.name);
    assert_eq!(
        parsed
            .transition
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        vec!["state__q", "state__q__1"]
    );
    assert_eq!(
        parsed
            .transition
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["state__d", "state__d__1"]
    );
    assert_eq!(parsed.registers, lowered.registers);
    assert!(structurally_equivalent(
        &parsed.transition,
        &lowered.transition
    ));
}

#[test]
fn sequential_blif_imports_supported_latch_initial_values() {
    let design = make_pipeline_design();
    let cases = [
        ("0", Some(IrBits::make_ubits(2, 0).unwrap())),
        ("1", Some(IrBits::make_ubits(2, 3).unwrap())),
        ("2", None),
    ];
    for (initial, expected) in cases {
        let text = emit_blif(&design)
            .unwrap()
            .replace("clk_main 1", &format!("clk_main {initial}"));
        let parsed = parse_blif(&text).unwrap();
        assert_eq!(
            parsed.registers[0].initial_value, expected,
            "initial value {initial}"
        );
    }
}

#[test]
fn sequential_blif_rejects_unsupported_latch_semantics() {
    let text = emit_blif(&make_pipeline_design()).unwrap();
    let cases = [
        (
            text.replace("state_reg[0] re clk_main 1", "state_reg[0] re clk_main 3"),
            "unsupported initial value '3'",
        ),
        (
            text.replace("state_reg[1] re clk_main 1", "state_reg[1] re clk_main 2"),
            "mixes specified and unspecified initial bits",
        ),
        (
            text.replace("state_reg[0] re clk_main 1", "state_reg[0] fe clk_main 1"),
            "only rising-edge 're' latches are supported",
        ),
        (
            text.replace("state_reg[1] re clk_main 1", "state_reg[1] re clk_alt 1"),
            "does not match inferred clock primary input 'clk_main'",
        ),
    ];
    for (variant, expected) in cases {
        let error = parse_blif(&variant).unwrap_err();
        assert!(error.contains(expected), "got: {error}");
    }
}

#[test]
fn combinational_blif_roundtrips_interesting_signatures() {
    for case in interesting_ir_roundtrip_cases() {
        let sample = load_interesting_ir_roundtrip_case(case);
        let contains_zero_width_port = sample
            .gate_fn
            .inputs
            .iter()
            .any(|input| input.get_bit_count() == 0)
            || sample
                .gate_fn
                .outputs
                .iter()
                .any(|output| output.get_bit_count() == 0);
        let emitted = emit_gate_fn_blif(&sample.gate_fn);
        if contains_zero_width_port {
            assert!(
                emitted
                    .unwrap_err()
                    .contains("plain BLIF has no net with which to recover it"),
                "{}",
                case.name
            );
            continue;
        }
        let text = emitted.unwrap();
        let parsed = parse_gate_fn_blif(&text).unwrap();
        assert_eq!(
            parsed.inputs.len(),
            sample.gate_fn.inputs.len(),
            "{}",
            case.name
        );
        assert_eq!(
            parsed.outputs.len(),
            sample.gate_fn.outputs.len(),
            "{}",
            case.name
        );
        assert!(
            structurally_equivalent(&parsed, &sample.gate_fn),
            "{}",
            case.name
        );
    }
}

#[test]
fn blif_parser_lowers_multi_input_on_set_covers() {
    let mut builder = GateBuilder::new("cover".to_string(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".to_string(), 1);
    let b = builder.add_input("b".to_string(), 1);
    let c = builder.add_input("c".to_string(), 1);
    let ab = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
    let result = builder.add_or_binary(ab, *c.get_lsb(0));
    builder.add_output("o".to_string(), result.into());
    let expected = builder.build();

    let emitted = emit_gate_fn_blif(&expected).unwrap();
    let interface = emitted
        .split(".names")
        .next()
        .expect("emitted BLIF includes logic boundary");
    let text = format!(
        "{interface}.names a_input0[0] b_input1[0] c_input2[0] o_output0[0]\n11- 1\n--1 1\n.end\n"
    );
    let parsed = parse_gate_fn_blif(&text).unwrap();
    assert!(structurally_equivalent(&parsed, &expected));
}

#[test]
fn blif_parser_lowers_abc_off_set_covers() {
    let mut builder = GateBuilder::new("offset".to_string(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".to_string(), 1);
    let b = builder.add_input("b".to_string(), 1);
    let result = builder.add_or_binary(*a.get_lsb(0), *b.get_lsb(0));
    builder.add_output("o".to_string(), result.into());
    let expected = builder.build();
    let text = r#".model offset
.inputs a_input0[0] b_input1[0]
.outputs o_output0[0]
.names a_input0[0] b_input1[0] o_output0[0]
00 0
.end
"#;
    let parsed = parse_gate_fn_blif(text).unwrap();
    assert!(structurally_equivalent(&parsed, &expected));
}

#[test]
fn blif_parser_rejects_unrecognized_port_names_and_mixed_cover_polarity() {
    let unrecognized_ports = ".model x\n.inputs a\n.outputs o\n.names a o\n1 1\n.end\n";
    assert!(
        parse_blif(unrecognized_ports)
            .unwrap_err()
            .contains("does not match the g8r flattened-net convention")
    );

    let mut builder = GateBuilder::new("bad_cover".to_string(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".to_string(), 1);
    builder.add_output("o".to_string(), a);
    let emitted = emit_gate_fn_blif(&builder.build()).unwrap();
    let interface = emitted.split(".names").next().unwrap();
    let text = format!("{interface}.names a_input0[0] o_output0[0]\n0 0\n1 1\n.end\n");
    assert!(
        parse_blif(&text)
            .unwrap_err()
            .contains("mixes on-set and off-set rows")
    );
}

#[test]
fn blif_parser_accepts_abc_line_continuations() {
    let mut builder = GateBuilder::new("continuation".to_string(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".to_string(), 1);
    let b = builder.add_input("b".to_string(), 1);
    let result = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
    builder.add_output("o".to_string(), result.into());
    let expected = builder.build();
    let text = r#".model continuation
.inputs a_input0[0] \
 b_input1[0]
.outputs o_output0[0]
.names a_input0[0] b_input1[0] o_output0[0]
11 1
.end
"#;
    let parsed = parse_gate_fn_blif(text).unwrap();
    assert!(structurally_equivalent(&parsed, &expected));
}

#[test]
fn sequential_blif_parser_accepts_inline_comments() {
    let design = make_pipeline_design();
    let text = emit_blif(&design)
        .unwrap()
        .replace(".model pipeline", ".model pipeline # generated design")
        .replace(
            ".inputs data_input0[0] data_input0[1] clk_main",
            ".inputs data_input0[0] data_input0[1] \\ # continue inputs\n clk_main # retained clock",
        )
        .replace("11 1", "11 1 # product term")
        .replace(
            ".latch state_next[0] state_reg[0] re clk_main 1",
            ".latch state_next[0] state_reg[0] re clk_main 1 # first register bit",
        )
        .replace(".end", ".end # done");
    let parsed = parse_blif(&text).unwrap();
    assert_eq!(parsed.clock, design.clock);
    assert_eq!(parsed.registers, design.registers);
    assert!(structurally_equivalent(
        &parsed.transition,
        &design.transition
    ));
}

#[test]
fn blif_parser_rejects_unsupported_directives_and_unterminated_continuations() {
    let unsupported_subckt = r#".model x
.inputs a_input0[0]
.outputs o_output0[0]
.subckt BUF A=a_input0[0] Y=o_output0[0]
.end
"#;
    assert!(
        parse_blif(unsupported_subckt)
            .unwrap_err()
            .contains("unsupported BLIF directive '.subckt'")
    );

    let unterminated_continuation = r#".model x
.inputs a_input0[0] \
"#;
    assert!(
        parse_blif(unterminated_continuation)
            .unwrap_err()
            .contains("unterminated BLIF line continuation")
    );
}
