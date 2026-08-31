// SPDX-License-Identifier: Apache-2.0

//! Automatic function construction, calls, ordering, and ownership tests.

use xlsynth_vast::{VastFile, VastFileType};

#[test]
fn casez_function_patterns_and_default_have_explicit_assignments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("encoder");
    let ty = file.make_bit_vector_type(2, false);
    let function = file.add_function(module, "leading", &ty).unwrap();
    let input = file
        .function_add_logic_input(function, "value", &ty)
        .unwrap();
    let result = file.function_result(function).to_expr();
    let case = file.block_add_casez(file.function_body(function), &input.to_expr());
    file.case_set_unique(case, true).unwrap();
    for (pattern, count) in [("1?", 0), ("01", 1)] {
        let pattern = file.make_binary_pattern(pattern).unwrap();
        let arm = file.case_add_item(case, &pattern);
        let count = file.make_unsized_decimal_literal(count);
        file.block_add_blocking_assignment(arm, &result, &count);
    }
    let default = file.case_add_default(case);
    let two = file.make_unsized_decimal_literal(2);
    file.block_add_blocking_assignment(default, &result, &two);
    assert_eq!(
        file.emit(),
        r#"module encoder;
  function automatic logic [1:0] leading (input logic [1:0] value);
    unique casez (value)
      2'b1?: leading = 0;
      2'b01: leading = 1;
      default: leading = 2;
    endcase
  endfunction
endmodule
"#
    );
}

#[test]
fn binary_patterns_reject_empty_invalid_and_excessive_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    for pattern in ["", "2", "x", "z", "01_?", "01;", "01\n", "é"] {
        assert!(file.make_binary_pattern(pattern).is_err(), "{pattern:?}");
    }
    assert!(
        file.make_binary_pattern(&"?".repeat((1 << 20) + 1))
            .is_err()
    );
}

#[test]
fn casez_keeps_grouping_for_multi_statement_arms_and_function_bodies() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("grouped");
    let ty = file.make_scalar_type();
    let function = file.add_function(module, "invert", &ty).unwrap();
    let input = file
        .function_add_logic_input(function, "value", &ty)
        .unwrap();
    let temporary = file.function_add_logic(function, "temporary", &ty).unwrap();
    let result = file.function_result(function).to_expr();
    let body = file.function_body(function);
    let case = file.block_add_casez(body, &input.to_expr());
    let one = file.make_binary_pattern("1").unwrap();
    let arm = file.case_add_item(case, &one);
    file.block_add_blocking_assignment(arm, &temporary.to_expr(), &input.to_expr());
    file.block_add_blocking_assignment(arm, &result, &temporary.to_expr());
    let default = file.case_add_default(case);
    let zero = file.make_unsized_decimal_literal(0);
    file.block_add_blocking_assignment(default, &result, &zero);
    let inverted = file.make_bitwise_not(&result);
    file.block_add_blocking_assignment(body, &result, &inverted);
    assert_eq!(
        file.emit(),
        r#"module grouped;
  function automatic logic invert (input logic value);
    logic temporary;
    begin
      casez (value)
        1'b1: begin
          temporary = value;
          invert = temporary;
        end
        default: invert = 0;
      endcase
      invert = ~invert;
    end
  endfunction
endmodule
"#
    );
}

#[test]
fn automatic_function_matches_compact_systemverilog_helper_layout() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("multiply");
    let unsigned = file.make_bit_vector_type(32, false);
    let signed = file.make_bit_vector_type(32, true);
    let left = file.add_input(module, "left", &unsigned);
    let right = file.add_input(module, "right", &unsigned);
    let output = file.add_output(module, "output_value", &unsigned);

    let helper = file
        .add_function(module, "multiply_signed", &unsigned)
        .expect("function declaration should succeed");
    let lhs = file
        .function_add_input(helper, "lhs", &unsigned)
        .expect("first function input should succeed");
    let rhs = file
        .function_add_input(helper, "rhs", &unsigned)
        .expect("second function input should succeed");
    let signed_lhs = file
        .function_add_reg(helper, "signed_lhs", &signed)
        .expect("first local should succeed");
    let signed_rhs = file
        .function_add_reg(helper, "signed_rhs", &signed)
        .expect("second local should succeed");
    let signed_result = file
        .function_add_reg(helper, "signed_result", &signed)
        .expect("result local should succeed");
    let body = file.function_body(helper);
    let lhs_cast = file.make_function_call("$signed", &[&lhs.to_expr()]);
    file.block_add_blocking_assignment(body, &signed_lhs.to_expr(), &lhs_cast);
    let rhs_cast = file.make_function_call("$signed", &[&rhs.to_expr()]);
    file.block_add_blocking_assignment(body, &signed_rhs.to_expr(), &rhs_cast);
    let product = file.make_mul(&signed_lhs.to_expr(), &signed_rhs.to_expr());
    file.block_add_blocking_assignment(body, &signed_result.to_expr(), &product);
    let result = file.make_function_call("$unsigned", &[&signed_result.to_expr()]);
    file.block_add_blocking_assignment(body, &file.function_result(helper).to_expr(), &result);

    let call = file.make_function_call("multiply_signed", &[&left.to_expr(), &right.to_expr()]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &call);
    file.add_member_continuous_assignment(module, assignment);

    assert_eq!(
        file.emit(),
        r#"module multiply(
  input wire [31:0] left,
  input wire [31:0] right,
  output wire [31:0] output_value
);
  function automatic logic [31:0] multiply_signed (input reg [31:0] lhs, input reg [31:0] rhs);
    reg signed [31:0] signed_lhs;
    reg signed [31:0] signed_rhs;
    reg signed [31:0] signed_result;
    begin
      signed_lhs = $signed(lhs);
      signed_rhs = $signed(rhs);
      signed_result = signed_lhs * signed_rhs;
      multiply_signed = $unsigned(signed_result);
    end
  endfunction
  assign output_value = multiply_signed(left, right);
endmodule
"#
    );
}

#[test]
fn scalar_functions_use_systemverilog_logic_return_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("predicates");
    let scalar = file.make_scalar_type();
    let unsigned = file.make_bit_vector_type(8, false);
    let output = file.add_output(module, "enabled", &scalar);

    let helper = file
        .add_function(module, "constant_predicate", &scalar)
        .expect("scalar function should be supported");
    let one = file.make_unsized_one_literal();
    file.block_add_blocking_assignment(
        file.function_body(helper),
        &file.function_result(helper).to_expr(),
        &one,
    );

    let signed = file.make_bit_vector_type(1, true);
    let signed_helper = file
        .add_function(module, "signed_predicate", &signed)
        .expect("signed scalar function should be supported");
    let arg = file
        .function_add_input(signed_helper, "value", &unsigned)
        .expect("function input should be supported");
    let selected = file.make_index(&arg.to_indexable_expr(), 0);
    file.block_add_blocking_assignment(
        file.function_body(signed_helper),
        &file.function_result(signed_helper).to_expr(),
        &selected.to_expr(),
    );

    let call = file.make_function_call("constant_predicate", &[]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &call);
    file.add_member_continuous_assignment(module, assignment);

    assert_eq!(
        file.emit(),
        r#"module predicates(
  output wire enabled
);
  function automatic logic constant_predicate ();
    begin
      constant_predicate = '1;
    end
  endfunction
  function automatic logic signed signed_predicate (input reg [7:0] value);
    begin
      signed_predicate = value[0];
    end
  endfunction
  assign enabled = constant_predicate();
endmodule
"#
    );
}

#[test]
fn function_inputs_and_locals_preserve_types_and_insertion_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("ordered_helpers");
    let scalar = file.make_scalar_type();
    let signed = file.make_bit_vector_type(5, true);
    let vector = file.make_bit_vector_type(12, false);
    let external = file.make_extern_package_type("types", "word_t");

    let helper = file
        .add_function(module, "convert", &signed)
        .expect("function declaration should succeed");
    file.function_add_input(helper, "first", &external)
        .expect("external typed input should succeed");
    let second = file
        .function_add_input(helper, "second", &vector)
        .expect("vector input should succeed");
    file.function_add_logic(helper, "temporary", &scalar)
        .expect("logic local should succeed");
    file.function_add_reg(helper, "accumulator", &signed)
        .expect("register local should succeed");
    let slice = file.make_slice(&second.to_indexable_expr(), 4, 0);
    file.block_add_blocking_assignment(
        file.function_body(helper),
        &file.function_result(helper).to_expr(),
        &slice.to_expr(),
    );

    assert_eq!(
        file.emit(),
        r#"module ordered_helpers;
  function automatic logic signed [4:0] convert (input types::word_t first, input reg [11:0] second);
    logic temporary;
    reg signed [4:0] accumulator;
    begin
      convert = second[4:0];
    end
  endfunction
endmodule
"#
    );
}

#[test]
fn functions_preserve_module_member_order_and_separate_local_namespaces() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("ordered");
    let scalar = file.make_scalar_type();
    let input = file.add_input(module, "value", &scalar);
    let comment = file.make_comment("before helper");
    file.add_member_comment(module, comment);

    for name in ["first", "second"] {
        let helper = file
            .add_function(module, name, &scalar)
            .expect("function names should be independent");
        let argument = file
            .function_add_input(helper, "value", &scalar)
            .expect("function may reuse module and sibling argument names");
        file.block_add_blocking_assignment(
            file.function_body(helper),
            &file.function_result(helper).to_expr(),
            &argument.to_expr(),
        );
    }

    let call = file.make_function_call("first", &[&input.to_expr()]);
    let output = file.add_output(module, "result", &scalar);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &call);
    file.add_member_continuous_assignment(module, assignment);

    assert_eq!(
        file.emit(),
        r#"module ordered(
  input wire value,
  output wire result
);
  // before helper
  function automatic logic first (input reg value);
    begin
      first = value;
    end
  endfunction
  function automatic logic second (input reg value);
    begin
      second = value;
    end
  endfunction
  assign result = first(value);
endmodule
"#
    );
}

#[test]
fn function_return_types_preserve_packed_shapes_and_external_typedefs() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("result_types");
    let byte = file.make_bit_vector_type(8, false);
    let signed_byte = file.make_bit_vector_type(8, true);
    let packed = file.make_packed_array_type(signed_byte, &[2, 3]);
    let external = file.make_extern_package_type("types", "word_t");
    let packed_external = file.make_packed_array_type(external, &[2]);
    for (name, ty) in [
        ("unsigned_result", byte),
        ("signed_result", signed_byte),
        ("packed_result", packed),
        ("typed_result", external),
        ("packed_typed_result", packed_external),
    ] {
        let function = file.add_function(module, name, &ty).unwrap();
        let zero = file.make_unsized_zero_literal();
        file.block_add_blocking_assignment(
            file.function_body(function),
            &file.function_result(function).to_expr(),
            &zero,
        );
    }
    assert_eq!(
        file.emit(),
        r#"module result_types;
  function automatic logic [7:0] unsigned_result ();
    begin
      unsigned_result = '0;
    end
  endfunction
  function automatic logic signed [7:0] signed_result ();
    begin
      signed_result = '0;
    end
  endfunction
  function automatic logic signed [1:0][2:0][7:0] packed_result ();
    begin
      packed_result = '0;
    end
  endfunction
  function automatic types::word_t typed_result ();
    begin
      typed_result = '0;
    end
  endfunction
  function automatic types::word_t [1:0] packed_typed_result ();
    begin
      packed_typed_result = '0;
    end
  endfunction
endmodule
"#
    );
}

#[test]
fn verilog_function_return_types_do_not_introduce_systemverilog_keywords() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("result_types");
    let byte = file.make_bit_vector_type(8, true);
    let function = file.add_function(module, "signed_result", &byte).unwrap();
    let zero = file.make_unsized_decimal_literal(0);
    file.block_add_blocking_assignment(
        file.function_body(function),
        &file.function_result(function).to_expr(),
        &zero,
    );
    assert_eq!(
        file.emit(),
        r#"module result_types;
  function automatic signed [7:0] signed_result ();
    begin
      signed_result = 0;
    end
  endfunction
endmodule
"#
    );
}

#[test]
fn rejected_function_and_local_names_preserve_existing_output() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("duplicates");
    let scalar = file.make_scalar_type();
    file.add_input(module, "existing", &scalar);

    let before = file.emit();
    let error = file
        .add_function(module, "existing", &scalar)
        .expect_err("a function may not reuse a module-port name");
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: Attempted to declare function with name 'existing' multiple times in the same module. Already defined: [existing]"
    );
    assert_eq!(file.emit(), before);

    let helper = file
        .add_function(module, "helper", &scalar)
        .expect("unique function name should succeed");
    file.function_add_input(helper, "value", &scalar)
        .expect("first input should succeed");
    let before = file.emit();
    let error = file
        .function_add_reg(helper, "value", &scalar)
        .expect_err("locals may not reuse function-input names");
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: Attempted to declare reg with name 'value' multiple times in function 'helper'. Already defined: [helper, value]"
    );
    assert_eq!(file.emit(), before);

    let error = file
        .function_add_input(helper, "helper", &scalar)
        .expect_err("inputs may not reuse the implicit function result");
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: Attempted to declare input with name 'helper' multiple times in function 'helper'. Already defined: [helper, value]"
    );
    assert_eq!(file.emit(), before);
}

#[test]
fn anonymous_unpacked_function_results_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("helpers");
    let integer = file.make_integer_type(false);
    let unpacked = file.make_unpacked_array_type(integer, &[2]);
    let before = file.emit();

    let error = file
        .add_function(module, "unsupported", &unpacked)
        .expect_err("anonymous unpacked results require a named typedef");
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: function result type must not be an anonymous unpacked array; use a named typedef"
    );
    assert_eq!(file.emit(), before);
}

#[test]
fn indexing_compound_expressions_inserts_required_parentheses() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("indexing");
    let byte = file.make_bit_vector_type(8, false);
    let left = file.add_input(module, "left", &byte);
    let right = file.add_input(module, "right", &byte);
    let sum = file.make_add(&left.to_expr(), &right.to_expr());
    let indexable = file.make_indexable_expression(&sum);
    let bit = file.make_index(&indexable, 3);
    let slice = file.make_slice(&indexable, 5, 2);

    assert_eq!(file.emit_expression(&bit.to_expr()), "(left + right)[3]");
    assert_eq!(
        file.emit_expression(&slice.to_expr()),
        "(left + right)[5:2]"
    );

    let call = file.make_function_call("transform", &[&left.to_expr(), &right.to_expr()]);
    let indexed_call = file.make_slice(&file.make_indexable_expression(&call), 3, 0);
    assert_eq!(
        file.emit_expression(&indexed_call.to_expr()),
        "transform(left, right)[3:0]"
    );
}

#[test]
fn function_calls_are_atomic_with_nested_and_system_function_arguments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("calls");
    let byte = file.make_bit_vector_type(8, false);
    let left = file.add_input(module, "left", &byte);
    let right = file.add_input(module, "right", &byte);
    let signed_left = file.make_function_call("$signed", &[&left.to_expr()]);
    let signed_right = file.make_function_call("$signed", &[&right.to_expr()]);
    let nested = file.make_function_call("helper", &[&signed_left, &signed_right]);
    let sum = file.make_add(&nested, &left.to_expr());

    assert_eq!(
        file.emit_expression(&sum),
        "helper($signed(left), $signed(right)) + left"
    );
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn function_inputs_reject_foreign_data_types() {
    let mut original = VastFile::new(VastFileType::SystemVerilog);
    let foreign = original.make_scalar_type();
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("ownership");
    let local = file.make_scalar_type();
    let helper = file
        .add_function(module, "helper", &local)
        .expect("function declaration should succeed");

    let _ = file.function_add_input(helper, "value", &foreign);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn function_calls_reject_foreign_arguments() {
    let mut original = VastFile::new(VastFileType::SystemVerilog);
    let foreign = original.make_unsized_one_literal();
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    let _ = file.make_function_call("helper", &[&foreign]);
}

#[test]
#[should_panic(expected = "function call name must not be empty")]
fn function_calls_reject_empty_names() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    let _ = file.make_function_call("", &[]);
}
