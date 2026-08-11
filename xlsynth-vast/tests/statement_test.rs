// SPDX-License-Identifier: Apache-2.0

//! Module, statement, and procedural golden tests adapted from XLS VAST.

use xlsynth_vast::{DataKind, Expr, LiteralFormat, ModulePortDirection, VastFile, VastFileType};

/// Creates the unsized decimal literals used throughout upstream VAST tests.
fn plain(file: &mut VastFile, value: i32) -> Expr {
    file.make_unsized_decimal_literal(value)
}

/// Creates a width-preserving hexadecimal literal without relying on XLS.
fn hex(file: &mut VastFile, value: &str) -> Expr {
    file.make_literal(value, &LiteralFormat::Hex)
        .expect("valid hexadecimal typed bits literal")
}

#[test]
fn empty_modules_preserve_file_order_and_internal_blank_lines() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.add_module("first");
    file.add_module("second");

    let expected = r#"module first;

endmodule
module second;

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn includes_multiline_comments_and_blank_lines_preserve_file_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.add_include("defs/common.svh");
    file.add_comment_text("generated header\n\nsecond paragraph");
    let separator = file.make_blank_line();
    file.add_blank_line(separator);
    let module = file.add_module("first");
    let scalar = file.make_scalar_type();
    file.add_wire(module, "ready", &scalar);
    file.add_include("defs/trailer.svh");
    file.add_module("second");

    let expected = r#"`include "defs/common.svh"
// generated header
//
// second paragraph

module first;
  wire ready;
endmodule
`include "defs/trailer.svh"
module second;

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn mixed_port_directions_data_kinds_and_external_types_keep_declaration_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("interfaces");
    let scalar = file.make_scalar_type();
    let signed_byte = file.make_bit_vector_type(8, true);
    let external = file.make_extern_package_type("bus_pkg", "payload_t");

    file.add_logic_input(module, "clk", &scalar);
    file.add_output(module, "sample", &signed_byte);
    file.add_inout(module, "pad", &scalar);
    file.add_input(module, "payload", &external);
    file.add_logic_output(module, "valid", &scalar);

    let directions: Vec<_> = file
        .module_ports(module)
        .into_iter()
        .map(|port| file.port_direction(port))
        .collect();
    assert_eq!(
        directions,
        vec![
            ModulePortDirection::Input,
            ModulePortDirection::Output,
            ModulePortDirection::InOut,
            ModulePortDirection::Input,
            ModulePortDirection::Output,
        ]
    );

    let expected = r#"module interfaces(
  input logic clk,
  output wire signed [7:0] sample,
  inout wire pad,
  input bus_pkg::payload_t payload,
  output logic valid
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn typed_and_untyped_parameter_ports_emit_without_io_ports() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("parameters_only");
    let byte = file.make_bit_vector_type(8, false);
    let five = plain(&mut file, 5);
    let seven = plain(&mut file, 7);
    file.add_typed_parameter_port(module, "TypedParam", &byte, &five);
    file.add_parameter_port(module, "UntypedParam", &seven);

    let expected = r#"module parameters_only #(
  parameter logic [7:0] TypedParam = 5,
  parameter UntypedParam = 7
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn parameter_ports_and_symbolic_io_widths_share_one_module_header() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("parameterized");
    let width_default = plain(&mut file, 12);
    let width = file.add_parameter_port(module, "Width", &width_default);
    let signed_word = file.make_bit_vector_type(16, true);
    let limit_default = plain(&mut file, 5);
    file.add_typed_parameter_port(module, "Limit", &signed_word, &limit_default);
    let dynamic = file.make_bit_vector_type_expr(&width.to_expr(), false);
    file.add_logic_input(module, "payload", &dynamic);
    file.add_logic_output(module, "result", &dynamic);

    let expected = r#"module parameterized #(
  parameter Width = 12,
  parameter logic signed [15:0] Limit = 5
) (
  input logic [Width - 1:0] payload,
  output logic [Width - 1:0] result
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn mixed_declarations_keep_insertion_order_signedness_and_packed_dimensions() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("definitions");
    let scalar = file.make_scalar_type();
    let signed_byte = file.make_bit_vector_type(8, true);
    let packed = file.make_packed_array_type(signed_byte, &[2, 3]);
    let external = file.make_extern_package_type("types", "state_t");

    file.add_wire(module, "first", &scalar);
    file.add_reg(module, "counter", &signed_byte)
        .expect("unique register");
    file.add_logic(module, "state", &signed_byte)
        .expect("unique logic variable");
    file.add_wire(module, "matrix", &packed);
    file.add_wire(module, "external", &external);

    let expected = r#"module definitions;
  wire first;
  reg signed [7:0] counter;
  logic signed [7:0] state;
  wire signed [1:0][2:0][7:0] matrix;
  wire types::state_t external;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn verilog_unpacked_register_arrays_retain_packed_dimensions_and_ranges() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("verilog_arrays");
    let nibble = file.make_bit_vector_type(4, false);
    let packed = file.make_packed_array_type(nibble, &[3, 2]);
    let array = file.make_unpacked_array_type(packed, &[4, 5]);
    file.add_reg(module, "values", &array)
        .expect("unique array register");

    let expected = r#"module verilog_arrays;
  reg [2:0][1:0][3:0] values[0:3][0:4];
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn systemverilog_unpacked_register_arrays_retain_packed_dimensions_and_sizes() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("systemverilog_arrays");
    let nibble = file.make_bit_vector_type(4, false);
    let packed = file.make_packed_array_type(nibble, &[3, 2]);
    let array = file.make_unpacked_array_type(packed, &[4, 5]);
    file.add_reg(module, "values", &array)
        .expect("unique array register");

    let expected = r#"module systemverilog_arrays;
  reg [2:0][1:0][3:0] values[4][5];
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn parameters_and_localparams_preserve_integer_signedness_and_declared_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("constants");

    let unsigned_integer = file.make_integer_type(false);
    let integer_definition = file.make_def("UnsignedInteger", DataKind::Integer, &unsigned_integer);
    let seven = plain(&mut file, 7);
    file.add_parameter_with_def(module, &integer_definition, &seven);

    let signed_int = file.make_int_type(true);
    let int_definition = file.make_def("SignedInt", DataKind::Int, &signed_int);
    let nine = plain(&mut file, 9);
    file.add_typed_localparam(module, &int_definition, &nine);

    let byte = file.make_bit_vector_type(8, false);
    let byte_definition = file.make_def("Mask", DataKind::Logic, &byte);
    let mask = hex(&mut file, "bits[8]:0xA5");
    file.add_typed_localparam(module, &byte_definition, &mask);

    let eleven = plain(&mut file, 11);
    file.add_int_localparam(module, "AutoInt", &eleven);
    let all_ones = file.make_unsized_one_literal();
    file.add_localparam(module, "AllOnes", &all_ones);

    let expected = r#"module constants;
  parameter integer unsigned UnsignedInteger = 7;
  localparam int SignedInt = 9;
  localparam logic [7:0] Mask = 8'ha5;
  localparam int AutoInt = 11;
  localparam AllOnes = '1;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn always_ff_supports_positive_clock_negative_reset_and_nested_reset_logic() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("async_reset");
    let scalar = file.make_scalar_type();
    let byte = file.make_bit_vector_type(8, false);
    let clock = file.add_input(module, "clock", &scalar);
    let reset_n = file.add_input(module, "reset_n", &scalar);
    let data = file.add_input(module, "data", &byte);
    let register = file
        .add_reg(module, "state", &byte)
        .expect("unique state register");

    let posedge = file.make_pos_edge(&clock.to_expr());
    let negedge = file.make_neg_edge(&reset_n.to_expr());
    let always = file
        .add_always_ff(module, &[&posedge, &negedge])
        .expect("edge sensitivity list");
    let block = file.statement_block(always);
    let reset_asserted = file.make_logical_not(&reset_n.to_expr());
    let conditional = file.block_add_cond(block, &reset_asserted);
    let reset_block = file.conditional_then_block(conditional);
    let zero = hex(&mut file, "bits[8]:0");
    file.block_add_nonblocking_assignment(reset_block, &register.to_expr(), &zero);
    let update_block = file.conditional_add_else(conditional);
    file.block_add_nonblocking_assignment(update_block, &register.to_expr(), &data.to_expr());

    let expected = r#"module async_reset(
  input wire clock,
  input wire reset_n,
  input wire [7:0] data
);
  reg [7:0] state;
  always_ff @ (posedge clock or negedge reset_n) begin
    if (!reset_n) begin
      state <= 8'h00;
    end else begin
      state <= data;
    end
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn always_at_joins_multiple_level_sensitive_logic_references() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("level_sensitive");
    let word = file.make_bit_vector_type(32, false);
    let first = file.add_input(module, "first", &word);
    let second = file.add_input(module, "second", &word);
    let result = file
        .add_reg(module, "result", &word)
        .expect("unique result register");
    let first_expression = first.to_expr();
    let second_expression = second.to_expr();
    let always = file
        .add_always_at(module, &[&first_expression, &second_expression])
        .expect("level-sensitive signal references");
    let block = file.statement_block(always);
    let sum = file.make_add(&first_expression, &second_expression);
    file.block_add_blocking_assignment(block, &result.to_expr(), &sum);

    let expected = r#"module level_sensitive(
  input wire [31:0] first,
  input wire [31:0] second
);
  reg [31:0] result;
  always @ (first or second) begin
    result = first + second;
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn always_comb_interleaves_comments_blank_lines_inline_and_both_assignment_kinds() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("mixed_block");
    let scalar = file.make_scalar_type();
    let input = file.add_input(module, "input_bit", &scalar);
    let first = file
        .add_reg(module, "first", &scalar)
        .expect("unique first register");
    let second = file
        .add_reg(module, "second", &scalar)
        .expect("unique second register");
    let always = file.add_always_comb(module).expect("combinational block");
    let block = file.statement_block(always);

    file.block_add_comment_text(block, "first stage\nsecond line");
    file.block_add_blocking_assignment(block, &first.to_expr(), &input.to_expr());
    file.block_add_blank_line(block);
    file.block_add_inline_text(block, "/* synthesis keep */");
    file.block_add_nonblocking_assignment(block, &second.to_expr(), &first.to_expr());

    let expected = r#"module mixed_block(
  input wire input_bit
);
  reg first;
  reg second;
  always_comb begin
    // first stage
    // second line
    first = input_bit;

    /* synthesis keep */
    second <= first;
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn reusable_assignment_handles_remain_distinct_and_stable_during_block_construction() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("assignment_handles");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clock", &scalar);
    let data = file.add_input(module, "data", &scalar);
    let combinational = file
        .add_reg(module, "comb", &scalar)
        .expect("unique combinational register");
    let sequential = file
        .add_reg(module, "seq", &scalar)
        .expect("unique sequential register");

    let reusable_blocking =
        file.make_blocking_assignment(&combinational.to_expr(), &data.to_expr());
    let reusable_nonblocking =
        file.make_nonblocking_assignment(&sequential.to_expr(), &data.to_expr());
    let saved_blocking = reusable_blocking;
    let saved_nonblocking = reusable_nonblocking;

    let comb_always = file.add_always_comb(module).expect("combinational block");
    let comb_block = file.statement_block(comb_always);
    file.block_add_blocking_assignment(comb_block, &combinational.to_expr(), &data.to_expr());
    let posedge = file.make_pos_edge(&clock.to_expr());
    let seq_always = file
        .add_always_ff(module, &[&posedge])
        .expect("clocked block");
    let seq_block = file.statement_block(seq_always);
    file.block_add_nonblocking_assignment(seq_block, &sequential.to_expr(), &data.to_expr());

    assert_ne!(reusable_blocking, reusable_nonblocking);
    assert_eq!(reusable_blocking, saved_blocking);
    assert_eq!(reusable_nonblocking, saved_nonblocking);

    let expected = r#"module assignment_handles(
  input wire clock,
  input wire data
);
  reg comb;
  reg seq;
  always_comb begin
    comb = data;
  end
  always_ff @ (posedge clock) begin
    seq <= data;
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn combinational_conditionals_emit_multiple_else_if_arms_and_ordered_else_statements() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("priority");
    let scalar = file.make_scalar_type();
    let first = file.add_input(module, "first", &scalar);
    let second = file.add_input(module, "second", &scalar);
    let third = file.add_input(module, "third", &scalar);
    let primary = file
        .add_reg(module, "primary", &scalar)
        .expect("unique primary register");
    let fallback = file
        .add_reg(module, "fallback", &scalar)
        .expect("unique fallback register");
    let one = hex(&mut file, "bits[1]:1");
    let zero = hex(&mut file, "bits[1]:0");

    let always = file.add_always_comb(module).expect("combinational block");
    let block = file.statement_block(always);
    let conditional = file.block_add_cond(block, &first.to_expr());
    let consequent = file.conditional_then_block(conditional);
    file.block_add_blocking_assignment(consequent, &primary.to_expr(), &one);

    let combined = file.make_bitwise_and(&first.to_expr(), &second.to_expr());
    let first_alternate = file.conditional_add_else_if(conditional, &combined);
    file.block_add_blocking_assignment(first_alternate, &primary.to_expr(), &zero);

    let second_alternate = file.conditional_add_else_if(conditional, &third.to_expr());
    file.block_add_blocking_assignment(second_alternate, &primary.to_expr(), &one);

    let default = file.conditional_add_else(conditional);
    file.block_add_blocking_assignment(default, &primary.to_expr(), &zero);
    file.block_add_blocking_assignment(default, &fallback.to_expr(), &one);

    let expected = r#"module priority(
  input wire first,
  input wire second,
  input wire third
);
  reg primary;
  reg fallback;
  always_comb begin
    if (first) begin
      primary = 1'h1;
    end else if (first & second) begin
      primary = 1'h0;
    end else if (third) begin
      primary = 1'h1;
    end else begin
      primary = 1'h0;
      fallback = 1'h1;
    end
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn conditional_blocks_can_nest_inside_unconditional_else_arms() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("nested_conditionals");
    let scalar = file.make_scalar_type();
    let first = file.add_input(module, "first", &scalar);
    let second = file.add_input(module, "second", &scalar);
    let output = file
        .add_reg(module, "output_bit", &scalar)
        .expect("unique output register");
    let one = hex(&mut file, "bits[1]:1");
    let zero = hex(&mut file, "bits[1]:0");

    let always = file.add_always_comb(module).expect("combinational block");
    let root = file.statement_block(always);
    let outer = file.block_add_cond(root, &first.to_expr());
    let outer_then = file.conditional_then_block(outer);
    file.block_add_blocking_assignment(outer_then, &output.to_expr(), &one);
    let outer_else = file.conditional_add_else(outer);
    let inner = file.block_add_cond(outer_else, &second.to_expr());
    let inner_then = file.conditional_then_block(inner);
    file.block_add_blocking_assignment(inner_then, &output.to_expr(), &one);
    let inner_else = file.conditional_add_else(inner);
    file.block_add_blocking_assignment(inner_else, &output.to_expr(), &zero);

    let expected = r#"module nested_conditionals(
  input wire first,
  input wire second
);
  reg output_bit;
  always_comb begin
    if (first) begin
      output_bit = 1'h1;
    end else begin
      if (second) begin
        output_bit = 1'h1;
      end else begin
        output_bit = 1'h0;
      end
    end
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn case_statements_emit_multiple_arms_default_comments_and_blocking_assignments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("state_case");
    let state_type = file.make_bit_vector_type(2, false);
    let scalar = file.make_scalar_type();
    let selector = file.add_input(module, "state", &state_type);
    let value = file.add_input(module, "value", &scalar);
    let output = file
        .add_reg(module, "next", &scalar)
        .expect("unique next register");
    let zero_state = hex(&mut file, "bits[2]:0");
    let one_state = hex(&mut file, "bits[2]:1");
    let zero = hex(&mut file, "bits[1]:0");

    let always = file.add_always_comb(module).expect("combinational block");
    let block = file.statement_block(always);
    let case = file.block_add_case(block, &selector.to_expr());
    let idle = file.case_add_item(case, &zero_state);
    file.block_add_blocking_assignment(idle, &output.to_expr(), &zero);
    let active = file.case_add_item(case, &one_state);
    file.block_add_comment_text(active, "forward payload");
    file.block_add_blocking_assignment(active, &output.to_expr(), &value.to_expr());
    let default = file.case_add_default(case);
    let unknown = file.make_unsized_x_literal();
    file.block_add_blocking_assignment(default, &output.to_expr(), &unknown);

    let expected = r#"module state_case(
  input wire [1:0] state,
  input wire value
);
  reg next;
  always_comb begin
    case (state)
      2'h0: begin
        next = 1'h0;
      end
      2'h1: begin
        // forward payload
        next = value;
      end
      default: begin
        next = 'X;
      end
    endcase
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn module_scope_conditionals_can_drive_continuous_assignments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("parameter_branch");
    let one = plain(&mut file, 1);
    let two = plain(&mut file, 2);
    let first = file.add_parameter(module, "A", &one);
    let second = file.add_parameter(module, "B", &two);
    let scalar = file.make_scalar_type();
    let output = file.add_wire(module, "out", &scalar);
    let equal = file.make_eq(&first.to_expr(), &second.to_expr());
    let conditional = file.add_conditional(module, &equal);
    let consequent = file.conditional_then_block(conditional);
    let enabled = hex(&mut file, "bits[1]:1");
    file.block_add_continuous_assignment(consequent, &output.to_expr(), &enabled);
    let alternate = file.conditional_add_else(conditional);
    let disabled = hex(&mut file, "bits[1]:0");
    file.block_add_continuous_assignment(alternate, &output.to_expr(), &disabled);

    let expected = r#"module parameter_branch;
  parameter A = 1;
  parameter B = 2;
  wire out;
  if (A == B) begin
    assign out = 1'h1;
  end else begin
    assign out = 1'h0;
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn empty_always_blocks_preserve_each_distinct_block_spelling() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("empty_blocks");
    file.add_always_ff(module, &[])
        .expect("empty always_ff sensitivity list");
    file.add_always_at(module, &[])
        .expect("empty always sensitivity list");
    file.add_always_comb(module)
        .expect("empty combinational block");

    let expected = r#"module empty_blocks;
  always_ff @ () begin end
  always @ () begin end
  always_comb begin end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn generate_loops_emit_indexed_assignments_comments_blank_lines_and_inline_text() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("lane_copy");
    let byte = file.make_bit_vector_type(8, false);
    let source = file.add_input(module, "source", &byte);
    let target = file.add_output(module, "target", &byte);
    let zero = plain(&mut file, 0);
    let eight = plain(&mut file, 8);
    let generate = file.add_generate_loop(module, "i", &zero, &eight, Some("lanes"));
    let index = file.generate_genvar(generate).to_expr();
    let lhs = file.make_index_expr(&target.to_indexable_expr(), &index);
    let rhs = file.make_index_expr(&source.to_indexable_expr(), &index);
    file.generate_add_continuous_assignment(generate, &lhs.to_expr(), &rhs.to_expr());
    file.generate_add_blank_line(generate);
    let comment = file.make_comment("lane assignment");
    file.generate_add_comment(generate, &comment);
    let empty_comment = file.make_comment("");
    file.generate_add_comment(generate, &empty_comment);
    let inline = file.make_inline_verilog_statement("/* lane metadata */");
    file.generate_add_inline_statement(generate, &inline);

    let expected = r#"module lane_copy(
  input wire [7:0] source,
  output wire [7:0] target
);
  for (genvar i = 0; i < 8; i = i + 1) begin : lanes
    assign target[i] = source[i];

    // lane assignment
    //
    /* lane metadata */
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn nested_generate_loops_support_unlabeled_inner_blocks_and_multidimensional_selects() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("nested_generate");
    let one = plain(&mut file, 1);
    let vector_element = file.make_bit_vector_type_expr(&one, false);
    let matrix = file.make_packed_array_type(vector_element, &[4, 3]);
    let source = file.add_wire(module, "source", &matrix);
    let target = file.add_wire(module, "target", &matrix);
    let zero = plain(&mut file, 0);
    let four = plain(&mut file, 4);
    let three = plain(&mut file, 3);
    let outer = file.add_generate_loop(module, "i", &zero, &four, Some("rows"));
    let outer_index = file.generate_genvar(outer).to_expr();
    let inner = file.generate_add_generate_loop(outer, "j", &zero, &three, None);
    let inner_index = file.generate_genvar(inner).to_expr();
    let source_row = file.make_index_expr(&source.to_indexable_expr(), &outer_index);
    let target_row = file.make_index_expr(&target.to_indexable_expr(), &outer_index);
    let source_bit = file.make_index_expr(&source_row.to_indexable_expr(), &inner_index);
    let target_bit = file.make_index_expr(&target_row.to_indexable_expr(), &inner_index);
    file.generate_add_continuous_assignment(inner, &target_bit.to_expr(), &source_bit.to_expr());

    let expected = r#"module nested_generate;
  wire [3:0][2:0][0:0] source;
  wire [3:0][2:0][0:0] target;
  for (genvar i = 0; i < 4; i = i + 1) begin : rows
    for (genvar j = 0; j < 3; j = j + 1) begin
      assign target[i][j] = source[i][j];
    end
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn generate_loops_can_contain_typed_localparams_and_all_procedural_block_kinds() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("generated_state");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clock", &scalar);
    let data = file.add_input(module, "data", &scalar);
    let state = file
        .add_reg(module, "state", &scalar)
        .expect("unique state register");
    let zero = plain(&mut file, 0);
    let two = plain(&mut file, 2);
    let generate = file.add_generate_loop(module, "i", &zero, &two, Some("g"));

    let three = plain(&mut file, 3);
    file.generate_add_localparam(generate, "Depth", &three);
    let nibble = file.make_bit_vector_type(4, false);
    let limit_definition = file.make_def("Limit", DataKind::Logic, &nibble);
    let five = hex(&mut file, "bits[4]:5");
    file.generate_add_typed_localparam(generate, &limit_definition, &five);

    file.generate_add_always_comb(generate)
        .expect("empty generated combinational block");
    let posedge = file.make_pos_edge(&clock.to_expr());
    let sequential = file
        .generate_add_always_ff(generate, &[&posedge])
        .expect("generated clocked block");
    let sequential_block = file.statement_block(sequential);
    file.block_add_nonblocking_assignment(sequential_block, &state.to_expr(), &data.to_expr());
    let data_expression = data.to_expr();
    let level = file
        .generate_add_always_at(generate, &[&data_expression])
        .expect("generated level-sensitive block");
    let level_block = file.statement_block(level);
    file.block_add_blocking_assignment(level_block, &state.to_expr(), &data_expression);

    let expected = r#"module generated_state(
  input wire clock,
  input wire data
);
  reg state;
  for (genvar i = 0; i < 2; i = i + 1) begin : g
    localparam Depth = 3;
    localparam logic [3:0] Limit = 4'h5;
    always_comb begin end
    always_ff @ (posedge clock) begin
      state <= data;
    end
    always @ (data) begin
      state = data;
    end
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn generated_instantiations_keep_parameters_open_ports_and_macro_statements() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("generated_instances");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clock", &scalar);
    let output = file.add_output(module, "out", &scalar);
    let zero = plain(&mut file, 0);
    let one = plain(&mut file, 1);
    let generate = file.add_generate_loop(module, "i", &zero, &one, Some("instances"));

    let comment = file.make_comment("instantiate one lane");
    file.generate_add_comment(generate, &comment);
    let four = plain(&mut file, 4);
    let clock_expression = clock.to_expr();
    let output_expression = output.to_expr();
    let instantiation = file.make_instantiation(
        "lane",
        "lane_instance",
        &["Width"],
        &[&four],
        &["clock", "out", "unused"],
        &[Some(&clock_expression), Some(&output_expression), None],
    );
    file.generate_add_instantiation(generate, &instantiation);
    file.generate_add_blank_line(generate);
    let macro_ref = file.make_macro_ref_with_args("TRACE_LANE", &[&one]);
    let macro_statement = file.make_macro_statement(&macro_ref, true);
    file.generate_add_macro_statement(generate, &macro_statement);

    let expected = r#"module generated_instances(
  input wire clock,
  output wire out
);
  for (genvar i = 0; i < 1; i = i + 1) begin : instances
    // instantiate one lane
    lane #(
      .Width(4)
    ) lane_instance (
      .clock(clock),
      .out(out),
      .unused()
    );

    `TRACE_LANE(1);
  end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn instantiations_preserve_macro_parameters_multiple_named_ports_and_open_connections() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("uart_wrapper");
    let scalar = file.make_scalar_type();
    let byte = file.make_bit_vector_type(8, false);
    let clock = file.add_input(module, "my_clk", &scalar);
    let payload = file.add_input(module, "my_tx_byte", &byte);
    let ready = file.add_output(module, "ready", &scalar);

    let default_baud = file.make_macro_ref("DEFAULT_CLOCKS_PER_BAUD");
    let eight = plain(&mut file, 8);
    let baud_expression = default_baud.to_expr();
    let clock_expression = clock.to_expr();
    let payload_expression = payload.to_expr();
    let ready_expression = ready.to_expr();
    let instance = file.make_instantiation(
        "uart_transmitter",
        "tx",
        &["ClocksPerBaud", "Width"],
        &[&baud_expression, &eight],
        &["clk", "tx_byte", "ready", "unused"],
        &[
            Some(&clock_expression),
            Some(&payload_expression),
            Some(&ready_expression),
            None,
        ],
    );
    file.add_member_instantiation(module, instance);

    let expected = r#"module uart_wrapper(
  input wire my_clk,
  input wire [7:0] my_tx_byte,
  output wire ready
);
  uart_transmitter #(
    .ClocksPerBaud(`DEFAULT_CLOCKS_PER_BAUD),
    .Width(8)
  ) tx (
    .clk(my_clk),
    .tx_byte(my_tx_byte),
    .ready(ready),
    .unused()
  );
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn unparameterized_instantiations_preserve_an_explicitly_unconnected_port() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("open_connection");
    let instance = file.make_instantiation("uart_transmitter", "tx", &[], &[], &["clk"], &[None]);
    file.add_member_instantiation(module, instance);

    let expected = r#"module open_connection;
  uart_transmitter tx (
    .clk()
  );
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn macro_statements_distinguish_absent_arguments_empty_arguments_and_semicolons() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("macros");
    let byte = file.make_bit_vector_type(8, false);
    let first = file.add_input(module, "a", &byte);
    let second = file.add_input(module, "b", &byte);

    let no_arguments = file.make_macro_ref("MY_MACRO1");
    let no_arguments_statement = file.make_macro_statement(&no_arguments, true);
    file.add_member_macro_statement(module, no_arguments_statement);

    let empty_arguments = file.make_macro_ref_with_args("MY_MACRO2", &[]);
    let empty_arguments_statement = file.make_macro_statement(&empty_arguments, false);
    file.add_member_macro_statement(module, empty_arguments_statement);

    let first_expression = first.to_expr();
    let one_argument = file.make_macro_ref_with_args("MY_MACRO3", &[&first_expression]);
    let one_argument_statement = file.make_macro_statement(&one_argument, false);
    file.add_member_macro_statement(module, one_argument_statement);

    let second_expression = second.to_expr();
    let two_arguments =
        file.make_macro_ref_with_args("MY_MACRO4", &[&first_expression, &second_expression]);
    let two_arguments_statement = file.make_macro_statement(&two_arguments, true);
    file.add_member_macro_statement(module, two_arguments_statement);

    let expected = r#"module macros(
  input wire [7:0] a,
  input wire [7:0] b
);
  `MY_MACRO1;
  `MY_MACRO2()
  `MY_MACRO3(a)
  `MY_MACRO4(a, b);
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn multiline_module_comments_and_inline_statements_indent_every_emitted_line() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("annotations");
    let comment = file.make_comment("first line\n\nlast line");
    file.add_member_comment(module, comment);
    let separator = file.make_blank_line();
    file.add_member_blank_line(module, separator);
    let inline = file.make_inline_verilog_statement("/* synthesis\n   keep */");
    file.add_member_inline_statement(module, inline);

    let expected = r#"module annotations;
  // first line
  //
  // last line

  /* synthesis
     keep */
endmodule
"#;
    assert_eq!(file.emit(), expected);
}
