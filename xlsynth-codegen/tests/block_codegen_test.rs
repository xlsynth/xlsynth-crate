// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
#[allow(dead_code)]
#[path = "support/cases.rs"]
mod cases;
mod support;
use cases::*;
use xlsynth_codegen::{BlockCodegenError, BlockCodegenOptions, Layout, emit_system_verilog};
use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn simple_passthrough_has_stable_exact_systemverilog() {
    assert_eq!(
        emit(PASSTHROUGH, &BlockCodegenOptions::default()),
        r#"module passthrough(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn top_selection_uses_the_marked_block() {
    let ir = r#"package public_top

block unused(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block selected(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module selected(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn explicit_top_selection_overrides_the_package_top() {
    let ir = r#"package public_top

block alternate(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block marked(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    let options = BlockCodegenOptions {
        top: Some("alternate".to_owned()),
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit(ir, &options),
        r#"module alternate(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn sole_unmarked_block_is_selected_automatically() {
    let ir = PASSTHROUGH.replace("top block", "block");
    assert_eq!(
        emit(&ir, &BlockCodegenOptions::default()),
        emit(PASSTHROUGH, &BlockCodegenOptions::default())
    );
}

#[test]
fn missing_requested_top_has_actionable_error() {
    let options = BlockCodegenOptions {
        top: Some("missing".to_owned()),
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit_system_verilog(&package(PASSTHROUGH), &options),
        Err(BlockCodegenError::TopSelection(
            "requested top block `missing` does not exist".to_owned()
        ))
    );
}

#[test]
fn ambiguous_unmarked_top_has_actionable_error() {
    let ir = r#"package public_top

block first(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

block second(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::TopSelection(
            "package contains multiple blocks (first, second) without a block top; specify --top"
                .to_owned()
        ))
    );
}

#[test]
fn function_only_package_reports_missing_block() {
    let ir = r#"package public_functions

top fn identity(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}
"#;
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::TopSelection(
            "package contains no block suitable for SystemVerilog generation".to_owned()
        ))
    );
}

#[test]
fn top_module_name_can_be_overridden_without_changing_behavior() {
    let options = BlockCodegenOptions {
        module_name: Some("renamed_block".to_owned()),
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit(PASSTHROUGH, &options),
        r#"module renamed_block(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn top_module_name_override_cannot_collide_with_a_transitive_child() {
    let options = BlockCodegenOptions {
        module_name: Some("child".to_owned()),
        ..BlockCodegenOptions::default()
    };
    match emit_system_verilog(&package(HIERARCHY), &options) {
        Err(BlockCodegenError::InvalidBlock(message)) => {
            assert!(
                message.contains("child") && message.contains("module"),
                "collision error must identify the conflicting module: {message}"
            );
        }
        other => panic!("module-name collision must produce an actionable error: {other:?}"),
    }
}

#[test]
fn top_module_name_override_cannot_begin_with_a_dollar() {
    let options = BlockCodegenOptions {
        module_name: Some("$bad".to_owned()),
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit_system_verilog(&package(PASSTHROUGH), &options),
        Err(BlockCodegenError::InvalidBlock(
            "invalid SystemVerilog module identifier `$bad`: identifier is reserved, malformed, or requires escaping"
                .to_owned()
        ))
    );
}

#[test]
fn top_module_name_override_can_contain_an_embedded_dollar() {
    let options = BlockCodegenOptions {
        module_name: Some("public$block".to_owned()),
        ..BlockCodegenOptions::default()
    };
    let generated = emit(PASSTHROUGH, &options);
    assert!(generated.starts_with("module public$block(\n"));
}

#[test]
fn repeated_emission_is_byte_for_byte_deterministic() {
    let parsed = package(ADD);
    let options = BlockCodegenOptions::default();
    let reference = emit_system_verilog(&parsed, &options).unwrap();
    for _ in 0..32 {
        assert_eq!(emit_system_verilog(&parsed, &options).unwrap(), reference);
    }
    let reparsed = package(&parsed.to_string());
    assert_eq!(emit_system_verilog(&reparsed, &options).unwrap(), reference);
}

#[test]
fn interleaved_clock_and_port_order_are_preserved() {
    let ir = r#"package public_ports

top block ordered(data: bits[8], result: bits[8], clk: clock, enable: bits[1]) {
  reg state(bits[8])
  data: bits[8] = input_port(name=data, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  write_state: () = register_write(data, register=state, load_enable=enable, id=3)
  read_state: bits[8] = register_read(register=state, id=4)
  result: () = output_port(read_state, name=result, id=5)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(
        generated.lines().take(6).collect::<Vec<_>>(),
        [
            "module ordered(",
            "  input logic [7:0] data,",
            "  output logic [7:0] result,",
            "  input logic clk,",
            "  input logic enable",
            ");",
        ]
    );
}

#[test]
fn optional_systemverilog_port_types_can_be_preserved_or_flattened() {
    let ir = r#"package public_types

top block typed(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, sv_type="types::input_t", id=1)
  result: () = output_port(value, name=result, sv_type="types::output_t", id=2)
}
"#;
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module typed(
  input types::input_t value,
  output types::output_t result
);
  assign result = value;
endmodule
"#
    );
    let flattened = BlockCodegenOptions {
        emit_sv_types: false,
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit(ir, &flattened),
        r#"module typed(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn reserved_systemverilog_port_names_are_rejected_clearly() {
    let ir = r#"package public_names

top block public_block(initial: bits[8], result: bits[8]) {
  initial: bits[8] = input_port(name=initial, id=1)
  result: () = output_port(initial, name=result, id=2)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::InvalidBlock(
            "invalid SystemVerilog port identifier `initial`: identifier is reserved, malformed, or requires escaping"
                .to_owned()
        ))
    );
}

#[test]
fn reserved_invoked_function_names_are_rejected_clearly() {
    let ir = r#"package public_names

fn initial(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}

top block caller(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=2)
  called: bits[8] = invoke(value, to_apply=initial, id=3)
  result: () = output_port(called, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::Unsupported(
            "invoke is not supported in `caller`; inline function calls before block2sv code generation"
                .to_owned()
        ))
    );
}

#[test]
fn reserved_invoked_function_parameters_are_rejected_clearly() {
    let ir = r#"package public_names

fn helper(initial: bits[8] id=1) -> bits[8] {
  ret initial: bits[8] = param(name=initial, id=1)
}

top block caller(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=2)
  called: bits[8] = invoke(value, to_apply=helper, id=3)
  result: () = output_port(called, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::Unsupported(
            "invoke is not supported in `caller`; inline function calls before block2sv code generation"
                .to_owned()
        ))
    );
}

#[test]
fn reserved_block_instance_names_are_rejected_clearly() {
    let ir = r#"package public_names

block child(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block parent(value: bits[8], result: bits[8]) {
  instantiation initial(block=child, kind=block)
  value: bits[8] = input_port(name=value, id=3)
  connected: () = instantiation_input(value, instantiation=initial, port_name=value, id=4)
  received: bits[8] = instantiation_output(instantiation=initial, port_name=result, id=5)
  result: () = output_port(received, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::InvalidBlock(
            "invalid SystemVerilog instance identifier `initial`: identifier is reserved, malformed, or requires escaping"
                .to_owned()
        ))
    );
}

#[test]
fn zero_width_interface_ports_are_omitted() {
    let ir = r#"package public_empty

top block empty_ports(empty: bits[0], value: bits[8], empty_result: bits[0], result: bits[8]) {
  empty: bits[0] = input_port(name=empty, id=1)
  value: bits[8] = input_port(name=value, id=2)
  empty_result: () = output_port(empty, name=empty_result, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module empty_ports(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn zero_width_sign_extension_agrees_with_upstream_validity() {
    let ir = r#"package public_empty

top block sign_extend_empty(empty: bits[0], result: bits[8]) {
  empty: bits[0] = input_port(name=empty, id=1)
  expanded: bits[8] = sign_ext(empty, new_bit_count=8, id=2)
  result: () = output_port(expanded, name=result, id=3)
}
"#;
    let upstream = xlsynth::IrPackage::parse_ir(ir, None);
    assert!(
        upstream.is_err(),
        "upstream XLS unexpectedly accepts sign extension without a sign bit"
    );
    assert!(
        Parser::new(ir).parse_and_verify_package().is_err(),
        "PIR accepted a zero-width sign extension that upstream XLS rejects"
    );
}

#[test]
fn arrays_of_empty_elements_disappear_without_affecting_live_values() {
    let ir = r#"package public_empty

top block empty_array(empty: bits[0][3], value: bits[8], ignored: bits[0][3], result: bits[8]) {
  empty: bits[0][3] = input_port(name=empty, id=1)
  value: bits[8] = input_port(name=value, id=2)
  ignored: () = output_port(empty, name=ignored, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module empty_array(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn fully_empty_nested_aggregates_disappear_from_the_interface() {
    let ir = r#"package public_empty

top block empty_nested(empty: (bits[0], ((), bits[0][3])), value: bits[8], ignored: (bits[0], ((), bits[0][3])), result: bits[8]) {
  empty: (bits[0], ((), bits[0][3])) = input_port(name=empty, id=1)
  value: bits[8] = input_port(name=value, id=2)
  ignored: () = output_port(empty, name=ignored, id=3)
  result: () = output_port(value, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module empty_nested(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn zero_width_register_with_reset_and_enable_is_omitted() {
    let ir = r#"package public_empty

top block empty_register(clk: clock, rst: bits[1], enable: bits[1], empty: bits[0], value: bits[8], result: bits[8]) {
  #![reset(port="rst", asynchronous=true, active_low=false)]
  reg state(bits[0], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  empty: bits[0] = input_port(name=empty, id=3)
  value: bits[8] = input_port(name=value, id=4)
  current: bits[0] = register_read(register=state, id=5)
  state_write: () = register_write(empty, register=state, load_enable=enable, reset=rst, id=6)
  result: () = output_port(value, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module empty_register(
  input logic clk,
  input logic rst,
  input logic enable,
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
"#
    );
}

#[test]
fn empty_block_emits_valid_module() {
    let ir = "package public_empty\n\ntop block empty() {\n}\n";
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        "module empty;\n\nendmodule\n"
    );
}

#[test]
fn wide_array_slices_keep_generated_source_compact() {
    let ir = r#"package public_arrays

top block wide_slice(values: bits[8][4], start: bits[3], result: bits[8][64]) {
  values: bits[8][4] = input_port(name=values, id=1)
  start: bits[3] = input_port(name=start, id=2)
  selected: bits[8][64] = array_slice(values, start, width=64, id=3)
  result: () = output_port(selected, name=result, id=4)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(generated.matches("for (genvar ").count(), 1);
    assert!(
        generated.len() < 1_200,
        "wide slices should use one generated element loop, not expanded assignments: {} bytes\n{generated}",
        generated.len()
    );
}

#[test]
fn wide_array_updates_keep_generated_source_compact() {
    let make_ir = |count: usize| {
        format!(
            r#"package public_arrays

top block update(values: bits[8][{count}], index: bits[9], replacement: bits[8], result: bits[8][{count}]) {{
  values: bits[8][{count}] = input_port(name=values, id=1)
  index: bits[9] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  updated: bits[8][{count}] = array_update(values, replacement, indices=[index], id=4)
  result: () = output_port(updated, name=result, id=5)
}}
"#
        )
    };
    let small = emit(&make_ir(4), &BlockCodegenOptions::default());
    let large = emit(&make_ir(256), &BlockCodegenOptions::default());
    assert_eq!(large.matches("for (genvar ").count(), 1);
    assert!(
        large.len() < small.len() + 100,
        "generated updates should grow only with printed widths and bounds: small={} bytes, large={} bytes",
        small.len(),
        large.len()
    );
}

#[test]
fn multiple_register_writes_are_rejected_for_every_layout() {
    let ir = multiwrite_register_ir(None);
    assert_stock_xls_accepts(&ir);
    for layout in [Layout::None, Layout::Pipeline] {
        let error = emit_system_verilog(
            &package(&ir),
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        )
        .expect_err("multiple register writes must not emit ambiguous SystemVerilog");
        assert_eq!(
            error.to_string(),
            "register `state` in block `multiwrite` requires exactly one write, found 2",
            "layout={layout:?}"
        );
    }
}

#[test]
fn multiple_register_writes_are_rejected_with_every_reset_configuration() {
    for asynchronous in [false, true] {
        for active_low in [false, true] {
            let ir = multiwrite_register_ir(Some((asynchronous, active_low)));
            assert_stock_xls_accepts(&ir);
            for layout in [Layout::None, Layout::Pipeline] {
                let error = emit_system_verilog(
                    &package(&ir),
                    &BlockCodegenOptions {
                        layout,
                        ..BlockCodegenOptions::default()
                    },
                )
                .expect_err("reset metadata cannot make multiple register writes unambiguous");
                assert_eq!(
                    error.to_string(),
                    "register `state` in block `multiwrite` requires exactly one write, found 2",
                    "layout={layout:?}, asynchronous={asynchronous}, active_low={active_low}"
                );
            }
        }
    }
}

#[test]
fn asynchronous_resets_use_the_correct_reset_edge_and_priority() {
    for active_low in [false, true] {
        let ir = register_ir(true, active_low, false);
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let edge = generated
            .lines()
            .find(|line| line.trim_start().starts_with("always_ff"))
            .expect("asynchronous register must have a clocked process");
        let expected_edge = if active_low {
            "  always_ff @ (posedge clk or negedge rst_n) begin"
        } else {
            "  always_ff @ (posedge clk or posedge rst) begin"
        };
        assert_eq!(edge, expected_edge);
        let condition = generated
            .lines()
            .find(|line| line.trim_start().starts_with("if ("))
            .expect("resettable register must check reset before enable");
        let expected_condition = if active_low {
            "    if (!rst_n) begin"
        } else {
            "    if (rst) begin"
        };
        assert_eq!(condition, expected_condition);
    }
}

#[test]
fn register_writes_keep_their_own_reset_operands() {
    let ir = r#"package public_derived_reset

top block derived_reset(clk: clock, rst: bits[1], gate: bits[1], data: bits[8], direct_out: bits[8], gated_out: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg direct(bits[8], reset_value=0)
  reg gated(bits[8], reset_value=1)
  rst: bits[1] = input_port(name=rst, id=1)
  gate: bits[1] = input_port(name=gate, id=2)
  data: bits[8] = input_port(name=data, id=3)
  direct_q: bits[8] = register_read(register=direct, id=4)
  gated_q: bits[8] = register_read(register=gated, id=5)
  gated_reset: bits[1] = and(rst, gate, id=6)
  direct_d: () = register_write(data, register=direct, reset=rst, id=7)
  gated_d: () = register_write(data, register=gated, reset=gated_reset, id=8)
  direct_out: () = output_port(direct_q, name=direct_out, id=9)
  gated_out: () = output_port(gated_q, name=gated_out, id=10)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    assert_eq!(generated.matches("always_ff @ (posedge clk)").count(), 2);
    assert!(generated.contains("if (rst) begin"));
    assert!(generated.contains("assign gated_reset = rst & gate;"));
    assert!(generated.contains("if (gated_reset) begin"));
    assert_eq!(generated.matches("if (rst) begin").count(), 1);
}

#[test]
fn backend_rejects_a_declared_register_without_a_read() {
    let ir = r#"package public_invalid_register

top block missing_read(clk: clock, value: bits[8], result: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  written: () = register_write(value, register=state, id=2)
    result: () = output_port(value, name=result, id=3)
}
"#;
    assert!(xlsynth::IrPackage::parse_ir(ir, None).is_err());
    let package = Parser::new(ir).parse_package().unwrap();
    match emit_system_verilog(&package, &BlockCodegenOptions::default()) {
        Err(BlockCodegenError::InvalidBlock(message)) => {
            assert!(
                message.contains("state") && message.contains("read"),
                "missing-read diagnostic must identify the register: {message}"
            );
        }
        other => panic!("missing register_read must be rejected: {other:?}"),
    }
}

#[test]
fn backend_rejects_a_declared_register_without_a_write() {
    let ir = r#"package public_invalid_register

top block missing_write(clk: clock, value: bits[8], result: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  current: bits[8] = register_read(register=state, id=2)
  result: () = output_port(current, name=result, id=3)
}
"#;
    assert!(xlsynth::IrPackage::parse_ir(ir, None).is_err());
    match emit_system_verilog(&package(ir), &BlockCodegenOptions::default()) {
        Err(BlockCodegenError::InvalidBlock(message)) => {
            assert!(
                message.contains("state") && message.contains("write"),
                "missing-write diagnostic must identify the register: {message}"
            );
        }
        other => panic!("missing register_write must be rejected: {other:?}"),
    }
}

#[test]
fn backend_rejects_reset_metadata_without_its_declared_input_port() {
    let ir = r#"package public_invalid_reset

top block missing_reset(clk: clock, value: bits[8], result: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}
"#;
    let parsed = Parser::new(ir).parse_package().unwrap();
    match emit_system_verilog(&parsed, &BlockCodegenOptions::default()) {
        Err(BlockCodegenError::InvalidBlock(message)) => {
            assert!(
                message.contains("reset") && message.contains("rst"),
                "missing-reset diagnostic must identify the port: {message}"
            );
        }
        other => panic!("missing reset input must be rejected: {other:?}"),
    }
}

#[test]
fn backend_rejects_resettable_register_without_block_reset_metadata() {
    let ir = register_ir(false, false, false).replace(
        "  #![reset(port=\"rst\", asynchronous=false, active_low=false)]\n",
        "",
    );
    assert!(xlsynth::IrPackage::parse_ir(&ir, None).is_err());
    match emit_system_verilog(&package(&ir), &BlockCodegenOptions::default()) {
        Err(BlockCodegenError::InvalidBlock(message)) => {
            assert!(
                message.contains("state") && message.contains("reset"),
                "missing block reset must identify its state register: {message}"
            );
        }
        other => panic!("resettable register without block reset must be rejected: {other:?}"),
    }
}

#[test]
fn pipeline_layout_exposes_every_reconstructed_stage() {
    let options = BlockCodegenOptions {
        layout: Layout::Pipeline,
        ..BlockCodegenOptions::default()
    };
    let generated = emit(PIPELINE, &options);
    let stages = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("// ===== Pipe stage"))
        .collect::<Vec<_>>();
    assert_eq!(
        stages,
        ["  // ===== Pipe stage 0:", "  // ===== Pipe stage 1:",]
    );
    let register_sections = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("// Registers for pipe stage"))
        .collect::<Vec<_>>();
    assert_eq!(
        register_sections,
        [
            "  // Registers for pipe stage 0:",
            "  // Registers for pipe stage 1:",
        ]
    );
}

#[test]
fn constant_data_does_not_hide_register_enable_feedback_or_dynamic_bypasses() {
    let constant =
        pipeline_with_shared_data_ir("  shared: bits[8] = literal(value=90, id=5)", "dead");
    let feedback = constant.replace("bit_slice(r0_q,", "bit_slice(r1_q,");
    let bypass = pipeline_with_shared_data_ir("  shared: bits[8] = identity(data, id=5)", "dead");
    let options = BlockCodegenOptions {
        layout: Layout::Pipeline,
        ..Default::default()
    };
    for ir in [feedback, bypass] {
        assert_stock_xls_accepts(&ir);
        emit(&ir, &BlockCodegenOptions::default());
        assert!(matches!(
            emit_system_verilog(&package(&ir), &options),
            Err(BlockCodegenError::NotPipeline(_))
        ));
    }
}

#[test]
fn combinational_child_remains_between_its_parent_register_stages() {
    let ir = child_pipeline_ir(false);
    assert_stock_xls_accepts(&ir);
    let generated = emit(
        &ir,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    assert_eq!(
        pipeline_stage_comments(&generated, "parent"),
        ["  // ===== Pipe stage 0:", "  // ===== Pipe stage 1:"]
    );
}

#[test]
fn registered_child_adds_its_latency_between_parent_register_stages() {
    let ir = child_pipeline_ir(true);
    assert_stock_xls_accepts(&ir);
    let generated = emit(
        &ir,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    assert_eq!(
        pipeline_stage_comments(&generated, "child"),
        ["  // ===== Pipe stage 0:"]
    );
    assert_eq!(
        pipeline_stage_comments(&generated, "parent"),
        [
            "  // ===== Pipe stage 0:",
            "  // ===== Pipe stage 1:",
            "  // ===== Pipe stage 2:",
        ],
        "generated heterogeneous hierarchy:\n{generated}"
    );
}

#[test]
fn heterogeneous_child_outputs_keep_their_individual_pipeline_latencies() {
    let ir = r#"package public_hierarchy

block child(clk: clock, value: bits[8], fast: bits[8], slow: bits[8]) {
  reg delayed(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  delayed_write: () = register_write(value, register=delayed, id=2)
  delayed_read: bits[8] = register_read(register=delayed, id=3)
  fast: () = output_port(value, name=fast, id=4)
  slow: () = output_port(delayed_read, name=slow, id=5)
}

top block parent(clk: clock, value: bits[8], fast: bits[8], slow: bits[8]) {
  reg incoming(bits[8])
  reg faststate(bits[8])
  reg slowstate(bits[8])
  instantiation component(block=child, kind=block)
  value: bits[8] = input_port(name=value, id=6)
  incoming_write: () = register_write(value, register=incoming, id=7)
  incoming_read: bits[8] = register_read(register=incoming, id=8)
  connected: () = instantiation_input(incoming_read, instantiation=component, port_name=value, id=9)
  fast_value: bits[8] = instantiation_output(instantiation=component, port_name=fast, id=10)
  slow_value: bits[8] = instantiation_output(instantiation=component, port_name=slow, id=11)
  fast_write: () = register_write(fast_value, register=faststate, id=12)
  fast_read: bits[8] = register_read(register=faststate, id=13)
  slow_write: () = register_write(slow_value, register=slowstate, id=14)
  slow_read: bits[8] = register_read(register=slowstate, id=15)
  fast: () = output_port(fast_read, name=fast, id=16)
  slow: () = output_port(slow_read, name=slow, id=17)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(
        ir,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    assert_eq!(
        pipeline_stage_comments(&generated, "parent"),
        [
            "  // ===== Pipe stage 0:",
            "  // ===== Pipe stage 1:",
            "  // ===== Pipe stage 2:",
        ],
        "generated heterogeneous hierarchy:\n{generated}"
    );
    let parent = generated
        .split("module parent(")
        .nth(1)
        .expect("parent module must be emitted");
    let fast_stage = parent.find("// ===== Pipe stage 1:").unwrap();
    let fast_register = parent.find("logic [7:0] faststate;").unwrap();
    let slow_stage = parent.find("// ===== Pipe stage 2:").unwrap();
    let slow_register = parent.find("logic [7:0] slowstate;").unwrap();
    assert!(fast_stage < fast_register);
    assert!(fast_register < slow_stage);
    assert!(slow_stage < slow_register);
}

#[test]
fn external_function_remains_between_parent_register_stages() {
    let ir = r#"package public_external

#[ffi_proto("""code_template: "external_cell {fn} (.value({value}), .result({return}));"
""")]
fn external_identity(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}

top block parent(clk: clock, value: bits[8], result: bits[8]) {
  reg before(bits[8])
  reg after(bits[8])
  instantiation component(foreign_function=external_identity, kind=extern)
  value: bits[8] = input_port(name=value, id=2)
  before_write: () = register_write(value, register=before, id=3)
  before_read: bits[8] = register_read(register=before, id=4)
  connected: () = instantiation_input(before_read, instantiation=component, port_name=value, id=5)
  received: bits[8] = instantiation_output(instantiation=component, port_name=return, id=6)
  after_write: () = register_write(received, register=after, id=7)
  after_read: bits[8] = register_read(register=after, id=8)
  result: () = output_port(after_read, name=result, id=9)
}
"#;
    let generated = emit(
        ir,
        &BlockCodegenOptions {
            layout: Layout::Pipeline,
            ..BlockCodegenOptions::default()
        },
    );
    assert_eq!(
        pipeline_stage_comments(&generated, "parent"),
        ["  // ===== Pipe stage 0:", "  // ===== Pipe stage 1:"]
    );
    let instances = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("external_cell "))
        .collect::<Vec<_>>();
    assert_eq!(
        instances,
        ["  external_cell component (.value(before_), .result(received));"]
    );
}

#[test]
fn feedback_registers_are_supported_flat_but_rejected_as_pipelines() {
    let ir = register_ir(false, false, true);
    emit(&ir, &BlockCodegenOptions::default());
    let options = BlockCodegenOptions {
        layout: Layout::Pipeline,
        ..BlockCodegenOptions::default()
    };
    assert_eq!(
        emit_system_verilog(&package(&ir), &options),
        Err(BlockCodegenError::NotPipeline(
            "block `register_block` cannot use layout=pipeline: register feedback or uneven register layering prevents feed-forward stage reconstruction; use layout=none".to_owned()
        ))
    );
}

#[test]
fn hierarchy_emits_each_child_once_before_its_parent() {
    let generated = emit(HIERARCHY, &BlockCodegenOptions::default());
    let modules = generated
        .lines()
        .filter(|line| line.starts_with("module "))
        .collect::<Vec<_>>();
    assert_eq!(modules, ["module child(", "module parent("]);
    let instances = generated
        .lines()
        .filter(|line| line.starts_with("  child "))
        .collect::<Vec<_>>();
    assert_eq!(instances, ["  child left (", "  child right ("]);
}

#[test]
fn nested_hierarchy_automatically_connects_parent_clocks() {
    let ir = r#"package public_hierarchy

block leaf(clk: clock, value: bits[8], result: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  written: () = register_write(value, register=state, id=2)
  current: bits[8] = register_read(register=state, id=3)
  result: () = output_port(current, name=result, id=4)
}

block middle(clk: clock, value: bits[8], result: bits[8]) {
  instantiation first(block=leaf, kind=block)
  instantiation second(block=leaf, kind=block)
  value: bits[8] = input_port(name=value, id=5)
  first_input: () = instantiation_input(value, instantiation=first, port_name=value, id=6)
  first_value: bits[8] = instantiation_output(instantiation=first, port_name=result, id=7)
  second_input: () = instantiation_input(first_value, instantiation=second, port_name=value, id=8)
  second_value: bits[8] = instantiation_output(instantiation=second, port_name=result, id=9)
  result: () = output_port(second_value, name=result, id=10)
}

top block parent(clk: clock, value: bits[8], result: bits[8]) {
  instantiation nested(block=middle, kind=block)
  value: bits[8] = input_port(name=value, id=11)
  connected: () = instantiation_input(value, instantiation=nested, port_name=value, id=12)
  received: bits[8] = instantiation_output(instantiation=nested, port_name=result, id=13)
  result: () = output_port(received, name=result, id=14)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let modules = generated
        .lines()
        .filter(|line| line.starts_with("module "))
        .collect::<Vec<_>>();
    assert_eq!(
        modules,
        ["module leaf(", "module middle(", "module parent("]
    );
    let clocks = generated
        .lines()
        .filter(|line| line.trim() == ".clk(clk),")
        .count();
    assert_eq!(clocks, 3);
}

#[test]
fn external_templates_substitute_function_names_and_connected_ports() {
    let generated = emit(EXTERNAL, &BlockCodegenOptions::default());
    let instances = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("external_cell "))
        .collect::<Vec<_>>();
    assert_eq!(
        instances,
        ["  external_cell external_instance (.source(data), .destination(received));"]
    );
}

#[test]
fn external_templates_substitute_nested_aggregate_port_paths() {
    let ir = r#"package public_external

#[ffi_proto("""code_template: "external_pair {fn} (.low({pair.0}), .high({pair.1.0}), .result({return.1.0}));"
""")]
fn external_pair(pair: (bits[8], (bits[16])) id=1) -> (bits[8], (bits[16])) {
  ret pair: (bits[8], (bits[16])) = param(name=pair, id=1)
}

top block wrapper(low: bits[8], high: bits[16], result: bits[16]) {
  instantiation external_instance(foreign_function=external_pair, kind=extern)
  low: bits[8] = input_port(name=low, id=2)
  high: bits[16] = input_port(name=high, id=3)
  connected_low: () = instantiation_input(low, instantiation=external_instance, port_name=pair.0, id=4)
  connected_high: () = instantiation_input(high, instantiation=external_instance, port_name=pair.1.0, id=5)
  received: bits[16] = instantiation_output(instantiation=external_instance, port_name=return.1.0, id=6)
  result: () = output_port(received, name=result, id=7)
}
"#;
    let generated = emit(ir, &BlockCodegenOptions::default());
    let instances = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("external_pair "))
        .collect::<Vec<_>>();
    assert_eq!(
        instances,
        ["  external_pair external_instance (.low(low), .high(high), .result(received));"]
    );
}

#[test]
fn external_function_without_code_template_has_actionable_error() {
    let ir = EXTERNAL.replace(
        "#[ffi_proto(\"\"\"code_template: \"external_cell {fn} (.source({value}), .destination({return}));\"\n\"\"\")]\n",
        "",
    );
    assert_eq!(
        emit_system_verilog(&package(&ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::Unsupported(
            "external function `external_identity` has no ffi_proto code_template".to_owned()
        ))
    );
}

#[test]
fn unresolved_external_template_placeholder_has_actionable_error() {
    let ir = EXTERNAL.replace(".source({value})", ".source({missing})");
    assert_eq!(
        emit_system_verilog(&package(&ir), &BlockCodegenOptions::default()),
        Err(BlockCodegenError::InvalidBlock(
            "external instance `external_instance` leaves ffi_proto placeholder `{missing}` unresolved"
                .to_owned()
        ))
    );
}

#[test]
fn clocked_assertions_coverage_and_tracing_preserve_debug_controls() {
    let ir = r#"package public_debug

top block debug_events(clk: clock, predicate: bits[1], value: bits[8], result: bits[8]) {
  predicate: bits[1] = input_port(name=predicate, id=1)
  value: bits[8] = input_port(name=value, id=2)
  start_token: token = after_all(id=3)
  checked: token = assert(start_token, predicate, message="bad value", label="value_is_valid", id=4)
  observed: token = trace(checked, predicate, format="value={}", data_operands=[value], verbosity=0, id=5)
  covered: () = cover(predicate, label="value_was_observed", id=6)
  result: () = output_port(value, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    let enabled = emit(ir, &BlockCodegenOptions::default());
    let assertions = enabled
        .lines()
        .filter(|line| line.contains("assert property"))
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(
        assertions,
        [
            "value_is_valid: assert property (@(posedge clk) disable iff ($sampled($isunknown(predicate))) predicate) else $error(\"bad value\");"
        ]
    );
    let covers = enabled
        .lines()
        .filter(|line| line.contains("cover property"))
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(
        covers,
        ["value_was_observed: cover property (@(posedge clk) predicate);"]
    );
    let traces = enabled
        .lines()
        .filter(|line| line.contains("$display"))
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(traces, ["if (predicate) $display(\"value=%0d\", value);"]);

    let disabled = emit(
        ir,
        &BlockCodegenOptions {
            add_invariant_assertions: false,
            ..BlockCodegenOptions::default()
        },
    );
    assert_eq!(disabled.matches("assert property").count(), 0);
    assert_eq!(disabled.matches("cover property").count(), 1);
    assert_eq!(disabled.matches("$display").count(), 1);
}

#[test]
fn unlabeled_assertions_emit_valid_clocked_and_combinational_statements() {
    for clocked in [false, true] {
        let clock = if clocked { "clk: clock, " } else { "" };
        let ir = format!(
            r#"package public_debug

top block assertion({clock}predicate: bits[1], result: bits[1]) {{
  predicate: bits[1] = input_port(name=predicate, id=1)
  start_token: token = after_all(id=2)
  checked: token = assert(start_token, predicate, message="condition failed", id=3)
  result: () = output_port(predicate, name=result, id=4)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        let generated = emit(&ir, &BlockCodegenOptions::default());
        if clocked {
            let assertions = generated
                .lines()
                .filter(|line| line.contains("assert property"))
                .map(str::trim)
                .collect::<Vec<_>>();
            assert_eq!(
                assertions,
                [
                    "assert property (@(posedge clk) disable iff ($sampled($isunknown(predicate))) predicate) else $error(\"condition failed\");"
                ]
            );
        } else {
            assert!(generated.lines().any(|line| line == "  always_comb begin"));
            assert!(
                generated
                    .lines()
                    .any(|line| line.trim()
                        == "if (!($isunknown(predicate))) assert (predicate) else $error(\"condition failed\");")
            );
        }
        assert!(!generated.contains("begin : \n"));
        assert!(!generated.contains("\n  : assert"));
    }
}

#[test]
fn trace_formats_preserve_decimal_hex_binary_braces_and_literal_percent() {
    let ir = r#"package public_debug

top block formatted_trace(clk: clock, enabled: bits[1], value: bits[8], result: bits[8]) {
  enabled: bits[1] = input_port(name=enabled, id=1)
  value: bits[8] = input_port(name=value, id=2)
  start_token: token = after_all(id=3)
  observed: token = trace(start_token, enabled, format="default={} unsigned={:u} signed={:d} hex={:x} padded={:0x} prefix={:#x} binary={:b} bpadded={:0b} bprefix={:#b} braces={{value}} percent=50%", data_operands=[value, value, value, value, value, value, value, value, value], verbosity=0, id=4)
  result: () = output_port(value, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    let generated = emit(ir, &BlockCodegenOptions::default());
    let traces = generated
        .lines()
        .filter(|line| line.contains("$display"))
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(
        traces,
        [
            "if (enabled) $display(\"default=%0d unsigned=%0d signed=%0d hex=%0h padded=%h prefix=0x%h binary=%0b bpadded=%b bprefix=0b%b braces={value} percent=50%%\", value, value, $signed(value), value, value, value, value, value, value);"
        ]
    );
}

#[test]
fn resettable_assertions_disable_on_reset_and_unknown_predicates() {
    for asynchronous in [false, true] {
        for active_low in [false, true] {
            let reset = if active_low { "rst_n" } else { "rst" };
            let ir = format!(
                r#"package public_debug

top block guarded(clk: clock, {reset}: bits[1], predicate: bits[1], result: bits[1]) {{
  #![reset(port="{reset}", asynchronous={asynchronous}, active_low={active_low})]
  {reset}: bits[1] = input_port(name={reset}, id=1)
  predicate: bits[1] = input_port(name=predicate, id=2)
  start_token: token = after_all(id=3)
  checked: token = assert(start_token, predicate, message="condition failed", id=4)
  result: () = output_port(predicate, name=result, id=5)
}}
"#
            );
            assert_stock_xls_accepts(&ir);
            let generated = emit(&ir, &BlockCodegenOptions::default());
            let assertions = generated
                .lines()
                .filter(|line| line.contains("assert property"))
                .map(str::trim)
                .collect::<Vec<_>>();
            let inactive = if active_low { "1'b1" } else { "1'b0" };
            let disabled = format!("{reset} !== {inactive} || $isunknown(predicate)");
            let disabled = if asynchronous {
                disabled
            } else {
                format!("$sampled({disabled})")
            };
            assert_eq!(
                assertions,
                [format!(
                    "assert property (@(posedge clk) disable iff ({disabled}) predicate) else $error(\"condition failed\");"
                )],
                "asynchronous={asynchronous}, active_low={active_low}"
            );
        }
    }
}

#[test]
fn invokes_with_debug_operations_are_rejected() {
    let ir = r#"package public_helper_debug

fn checked_helper(value: bits[8] id=1) -> bits[8] {
  zero: bits[8] = literal(value=0, id=2)
  local_condition: bits[1] = ne(value, zero, id=3)
  start_token: token = after_all(id=4)
  checked: token = assert(start_token, local_condition, message="helper condition", label="helper_valid", id=5)
  observed: token = trace(checked, local_condition, format="helper={:x}", data_operands=[value], verbosity=0, id=6)
  ret forwarded: bits[8] = identity(value, id=7)
}

top block wrapper(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=8)
  called: bits[8] = invoke(value, to_apply=checked_helper, id=9)
  result: () = output_port(called, name=result, id=10)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "wrapper");
}

#[test]
fn invokes_with_coverage_are_rejected() {
    let ir = r#"package public_helper_debug

fn covered_helper(value: bits[8] id=1) -> bits[8] {
  condition: bits[1] = literal(value=1, id=2)
  observed: () = cover(condition, label="helper_covered", id=3)
  ret forwarded: bits[8] = identity(value, id=4)
}

top block wrapper(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=5)
  called: bits[8] = invoke(value, to_apply=covered_helper, id=6)
  result: () = output_port(called, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "wrapper");
}

#[test]
fn fixed_names_reject_port_instance_and_debug_label_collisions() {
    let base = r#"package public_fixed_names

block child(value: bits[1], result: bits[1]) {
  value: bits[1] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block fixed_names(clk: clock, input_value: bits[1], result: bits[1]) {
  instantiation child_instance(block=child, kind=block)
  input_value: bits[1] = input_port(name=input_value, id=3)
  connected: () = instantiation_input(input_value, instantiation=child_instance, port_name=value, id=4)
  received: bits[1] = instantiation_output(instantiation=child_instance, port_name=result, id=5)
  start_token: token = after_all(id=6)
EVENTS
  result: () = output_port(received, name=result, id=10)
}
"#;
    let mut cases = Vec::new();
    for (label, existing) in [
        ("clk", "port"),
        ("input_value", "port"),
        ("result", "port"),
        ("child_instance", "instance"),
    ] {
        for (event, kind) in [
            (
                format!(
                    "  event: token = assert(start_token, input_value, message=\"check\", label=\"{label}\", id=7)"
                ),
                "assertion label",
            ),
            (
                format!("  event: () = cover(input_value, label=\"{label}\", id=7)"),
                "coverage label",
            ),
        ] {
            cases.push((base.replace("EVENTS", &event), label, existing, kind));
        }
    }
    for (first, second) in [(true, true), (false, false), (true, false), (false, true)] {
        let mut events = Vec::new();
        for (id, assertion) in [(7, first), (8, second)] {
            events.push(if assertion {
                format!("  event{id}: token = assert(start_token, input_value, message=\"check\", label=\"check\", id={id})")
            } else {
                format!("  event{id}: () = cover(input_value, label=\"check\", id={id})")
            });
        }
        cases.push((
            base.replace("EVENTS", &events.join("\n")),
            "check",
            if first {
                "assertion label"
            } else {
                "coverage label"
            },
            if second {
                "assertion label"
            } else {
                "coverage label"
            },
        ));
    }
    cases.push((
        base.replace("EVENTS", "")
            .replace("child_instance", "input_value"),
        "input_value",
        "port",
        "instance",
    ));
    for (ir, name, first, second) in cases {
        let parsed = package(&ir);
        for layout in [Layout::None, Layout::Pipeline] {
            assert_eq!(
                emit_system_verilog(
                    &parsed,
                    &BlockCodegenOptions {
                        layout,
                        ..BlockCodegenOptions::default()
                    }
                ),
                Err(BlockCodegenError::InvalidBlock(format!(
                    "SystemVerilog name collision in block `fixed_names`: `{name}` is used by both {first} and {second}"
                ))),
                "layout={layout:?}\n{ir}"
            );
        }
    }
}

#[test]
fn fixed_name_validation_also_checks_programmatically_constructed_instances() {
    let mut parsed = package(HIERARCHY);
    let PackageMember::Block { metadata, .. } = parsed.members.last_mut().unwrap() else {
        panic!("expected parent block");
    };
    metadata
        .instantiations
        .push(metadata.instantiations[0].clone());
    assert_eq!(emit_system_verilog(&parsed, &BlockCodegenOptions::default()),
        Err(BlockCodegenError::InvalidBlock(
            "SystemVerilog name collision in block `parent`: `left` is used by both instance and instance".to_owned()
        )));
}

#[test]
fn fixed_instance_names_are_reserved_before_generated_signals() {
    let ir = r#"package public_fixed_instance

block child(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block parent(data: bits[8], out: bits[8]) {
  instantiation collision(block=child, kind=block)
  data: bits[8] = input_port(name=data, id=3)
  connected: () = instantiation_input(data, instantiation=collision, port_name=value, id=4)
  collision: bits[8] = instantiation_output(instantiation=collision, port_name=result, id=5)
  out: () = output_port(collision, name=out, id=6)
}
"#;
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module child(
  input logic [7:0] value,
  output logic [7:0] result
);
  assign result = value;
endmodule
module parent(
  input logic [7:0] data,
  output logic [7:0] out
);
  logic [7:0] collision__1;
  assign out = collision__1;
  child collision (
    .value(data),
    .result(collision__1)
  );
endmodule
"#
    );
}

#[test]
fn fixed_debug_labels_are_reserved_before_generated_signals() {
    let ir = r#"package public_fixed_label

top block fixed_label(data: bits[1], out: bits[1]) {
  data: bits[1] = input_port(name=data, id=1)
  check: bits[1] = not(data, id=2)
  start_token: token = after_all(id=3)
  event: token = assert(start_token, check, message="check", label="check", id=4)
  out: () = output_port(check, name=out, id=5)
}
"#;
    assert_eq!(
        emit(ir, &BlockCodegenOptions::default()),
        r#"module fixed_label(
  input logic data,
  output logic out
);
  logic check__1;
  assign check__1 = ~data;
  `ifndef SYNTHESIS
  always_comb begin : check
    if (!($isunknown(check__1))) assert (check__1) else $error("check");
  end
  `endif
  assign out = check__1;
endmodule
"#
    );
}

#[test]
fn omitted_declarations_do_not_reserve_fixed_names() {
    let ir = r#"package public_omitted_names

top block omitted_names(data: bits[1], out: bits[1]) {
  data: bits[1] = input_port(name=data, id=1)
  start_token: token = after_all(id=2)
  event: token = assert(start_token, data, message="check", label="data", id=3)
  out: () = output_port(data, name=out, id=4)
}
"#;
    assert_eq!(
        emit(
            ir,
            &BlockCodegenOptions {
                add_invariant_assertions: false,
                ..BlockCodegenOptions::default()
            }
        ),
        r#"module omitted_names(
  input logic data,
  output logic out
);
  assign out = data;
endmodule
"#
    );
    let with_empty = ir
        .replace("label=\"data\"", "label=\"empty\"")
        .replace("(data: bits[1],", "(empty: bits[0], data: bits[1],")
        .replace(
            "  data: bits[1] =",
            "  empty: bits[0] = input_port(name=empty, id=5)\n  data: bits[1] =",
        );
    let without_empty = ir.replace("label=\"data\"", "label=\"empty\"");
    assert_eq!(
        emit(&with_empty, &BlockCodegenOptions::default()),
        emit(&without_empty, &BlockCodegenOptions::default())
    );
}

#[test]
fn fixed_debug_labels_can_repeat_in_different_modules() {
    let ir = r#"package public_scoped_labels

block child(value: bits[1], result: bits[1]) {
  value: bits[1] = input_port(name=value, id=1)
  event: () = cover(value, label="check", id=2)
  result: () = output_port(value, name=result, id=3)
}

top block parent(value: bits[1], result: bits[1]) {
  instantiation nested(block=child, kind=block)
  value: bits[1] = input_port(name=value, id=4)
  connected: () = instantiation_input(value, instantiation=nested, port_name=value, id=5)
  received: bits[1] = instantiation_output(instantiation=nested, port_name=result, id=6)
  event: () = cover(received, label="check", id=7)
  result: () = output_port(received, name=result, id=8)
}
"#;
    for layout in [Layout::None, Layout::Pipeline] {
        emit_system_verilog(
            &package(ir),
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        )
        .expect("identical labels in separate module scopes are legal");
    }
}

#[test]
fn reserved_assertion_and_coverage_labels_are_rejected() {
    for is_assertion in [false, true] {
        let (event, category) = if is_assertion {
            (
                "  start_token: token = after_all(id=2)\n  event: token = assert(start_token, predicate, message=\"failed\", label=\"initial\", id=3)\n",
                "assertion label",
            )
        } else {
            (
                "  event: () = cover(predicate, label=\"initial\", id=3)\n",
                "coverage label",
            )
        };
        let ir = format!(
            r#"package public_debug

top block invalid_label(clk: clock, predicate: bits[1], result: bits[1]) {{
  predicate: bits[1] = input_port(name=predicate, id=1)
{event}  result: () = output_port(predicate, name=result, id=4)
}}
"#
        );
        assert_stock_xls_accepts(&ir);
        assert_eq!(
            emit_system_verilog(&package(&ir), &BlockCodegenOptions::default()),
            Err(BlockCodegenError::InvalidBlock(format!(
                "invalid SystemVerilog {category} identifier `initial`: identifier is reserved, malformed, or requires escaping"
            )))
        );
    }
}

#[test]
fn invoked_ir_functions_are_rejected() {
    let ir = r#"package public_invoke

fn add_one(value: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  ret sum: bits[8] = add(value, one, id=3)
}

top block invoking(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=4)
  called: bits[8] = invoke(value, to_apply=add_one, id=5)
  result: () = output_port(called, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "invoking");
}

#[test]
fn invokes_with_aggregate_results_are_rejected() {
    let ir = r#"package public_invoke_arrays

fn update(values: bits[8][3] id=1, index: bits[2] id=2, replacement: bits[8] id=3) -> bits[8][3] {
  ret result: bits[8][3] = array_update(values, replacement, indices=[index], id=4)
}

top block invoking(values: bits[8][3], index: bits[2], replacement: bits[8], result: bits[8][3]) {
  values: bits[8][3] = input_port(name=values, id=5)
  index: bits[2] = input_port(name=index, id=6)
  replacement: bits[8] = input_port(name=replacement, id=7)
  called: bits[8][3] = invoke(values, index, replacement, to_apply=update, id=8)
  result: () = output_port(called, name=result, id=9)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "invoking");
}

#[test]
fn invokes_with_zero_width_results_are_rejected() {
    let ir = r#"package public_empty_functions

fn empty_result(value: bits[8] id=1) -> bits[0] {
  ret empty: bits[0] = literal(value=0, id=2)
}

top block ignore_result(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=3)
  ignored: bits[0] = invoke(value, to_apply=empty_result, id=4)
  result: () = output_port(value, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "ignore_result");
}

#[test]
fn invokes_with_empty_tuple_results_are_rejected() {
    let ir = r#"package public_empty_functions

fn empty_tuple(value: bits[8] id=1) -> (bits[0], ()) {
  empty_bits: bits[0] = literal(value=0, id=2)
  tuple.3: () = tuple(id=3)
  ret combined: (bits[0], ()) = tuple(empty_bits, tuple.3, id=4)
}

top block ignore_tuple(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=5)
  ignored: (bits[0], ()) = invoke(value, to_apply=empty_tuple, id=6)
  result: () = output_port(value, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "ignore_tuple");
}

#[test]
fn invokes_with_zero_width_parameters_are_rejected() {
    let ir = r#"package public_empty_functions

fn ignore_empty(value: bits[8] id=1, empty: bits[0] id=2) -> bits[8] {
  ret result: bits[8] = identity(value, id=3)
}

top block invoke_empty(value: bits[8], empty: bits[0], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=4)
  empty: bits[0] = input_port(name=empty, id=5)
  called: bits[8] = invoke(value, empty, to_apply=ignore_empty, id=6)
  result: () = output_port(called, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "invoke_empty");
}

#[test]
fn invokes_with_only_zero_width_arguments_are_rejected() {
    let ir = r#"package public_empty_functions

fn public_constant(empty: bits[0] id=1) -> bits[8] {
  ret constant: bits[8] = literal(value=42, id=2)
}

top block call_constant(empty: bits[0], result: bits[8]) {
  empty: bits[0] = input_port(name=empty, id=3)
  called: bits[8] = invoke(empty, to_apply=public_constant, id=4)
  result: () = output_port(called, name=result, id=5)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "call_constant");
}

#[test]
fn counted_for_loops_are_rejected_in_both_layouts() {
    let ir = r#"package public_counted_for

fn body(index: bits[3] id=1, carry: bits[8] id=2, increment: bits[8] id=3) -> bits[8] {
  ret sum: bits[8] = add(carry, increment, id=4)
}

top block repeated(seed: bits[8], increment: bits[8], result: bits[8]) {
  seed: bits[8] = input_port(name=seed, id=5)
  increment: bits[8] = input_port(name=increment, id=6)
  loop_result: bits[8] = counted_for(seed, trip_count=4, stride=1, body=body, invariant_args=[increment], id=7)
  result: () = output_port(loop_result, name=result, id=8)
}
"#;
    for layout in [Layout::None, Layout::Pipeline] {
        let error = emit_system_verilog(
            &package(ir),
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, BlockCodegenError::Unsupported(
            "counted_for is not supported in `repeated`; unroll loops before block2sv code generation".to_owned()
        ));
    }
}

#[test]
fn counted_for_with_zero_width_invariants_is_rejected() {
    let ir = r#"package public_empty_functions

fn body(index: bits[1] id=1, carry: bits[8] id=2, empty: bits[0] id=3) -> bits[8] {
  one: bits[8] = literal(value=1, id=4)
  ret advanced: bits[8] = add(carry, one, id=5)
}

top block once(seed: bits[8], empty: bits[0], result: bits[8]) {
  seed: bits[8] = input_port(name=seed, id=6)
  empty: bits[0] = input_port(name=empty, id=7)
  advanced: bits[8] = counted_for(seed, trip_count=1, stride=1, body=body, invariant_args=[empty], id=8)
  result: () = output_port(advanced, name=result, id=9)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()).unwrap_err(),
        BlockCodegenError::Unsupported(
            "counted_for is not supported in `once`; unroll loops before block2sv code generation"
                .to_owned()
        )
    );
}

#[test]
fn counted_for_with_unused_zero_width_carry_is_rejected() {
    let ir = r#"package public_empty_functions

fn body(index: bits[2] id=1, carry: bits[0] id=2) -> bits[0] {
  ret carry: bits[0] = param(name=carry, id=2)
}

top block empty_loop(empty: bits[0], value: bits[8], result: bits[8]) {
  empty: bits[0] = input_port(name=empty, id=3)
  value: bits[8] = input_port(name=value, id=4)
  ignored: bits[0] = counted_for(empty, trip_count=3, stride=1, body=body, id=5)
  result: () = output_port(value, name=result, id=6)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()).unwrap_err(),
        BlockCodegenError::Unsupported(
            "counted_for is not supported in `empty_loop`; unroll loops before block2sv code generation".to_owned()
        )
    );
}

#[test]
fn counted_for_with_zero_iterations_is_rejected() {
    let ir = r#"package public_counted_for

fn body(index: bits[1] id=1, carry: bits[8] id=2) -> bits[8] {
  one: bits[8] = literal(value=1, id=3)
  ret advanced: bits[8] = add(carry, one, id=4)
}

top block no_iterations(seed: bits[8], result: bits[8]) {
  seed: bits[8] = input_port(name=seed, id=5)
  unchanged: bits[8] = counted_for(seed, trip_count=0, stride=1, body=body, id=6)
  result: () = output_port(unchanged, name=result, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_eq!(
        emit_system_verilog(&package(ir), &BlockCodegenOptions::default()).unwrap_err(),
        BlockCodegenError::Unsupported(
            "counted_for is not supported in `no_iterations`; unroll loops before block2sv code generation".to_owned()
        )
    );
}

#[test]
fn invokes_of_zero_width_counted_loops_are_rejected() {
    let ir = r#"package nested_counted_loop

fn body(index: bits[2] id=1, carry: bits[0] id=2) -> bits[0] {
  ret carry: bits[0] = param(name=carry, id=2)
}

fn helper(empty: bits[0] id=3) -> bits[0] {
  ret loop_result: bits[0] = counted_for(empty, trip_count=3, stride=1, body=body, id=4)
}

top block caller(empty: bits[0], out: bits[0]) {
  empty: bits[0] = input_port(name=empty, id=5)
  result: bits[0] = invoke(empty, to_apply=helper, id=6)
  out: () = output_port(result, name=out, id=7)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "caller");
}

#[test]
fn nested_invocations_are_rejected() {
    let ir = r#"package public_invoke

fn add_one(value: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  ret incremented: bits[8] = add(value, one, id=3)
}

fn add_two(value: bits[8] id=4) -> bits[8] {
  once: bits[8] = invoke(value, to_apply=add_one, id=5)
  ret twice: bits[8] = invoke(once, to_apply=add_one, id=6)
}

top block nested_helpers(first: bits[8], second: bits[8], result: bits[8]) {
  first: bits[8] = input_port(name=first, id=7)
  second: bits[8] = input_port(name=second, id=8)
  left: bits[8] = invoke(first, to_apply=add_two, id=9)
  right: bits[8] = invoke(second, to_apply=add_two, id=10)
  combined: bits[8] = add(left, right, id=11)
  result: () = output_port(combined, name=result, id=12)
}
"#;
    assert_stock_xls_accepts(ir);
    assert_invoke_rejected(ir, "nested_helpers");
}
