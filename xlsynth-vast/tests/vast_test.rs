// SPDX-License-Identifier: Apache-2.0

use xlsynth_vast::{DataKind, Expr, LiteralFormat, VastDataType, VastFile, VastFileType};

type VastBinOp = fn(&mut VastFile, &Expr, &Expr) -> Expr;

#[test]
fn test_vast() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("main");
    let input_type = file.make_bit_vector_type(32, false);
    let output_type = file.make_scalar_type();
    file.add_input(module, "in", &input_type);
    file.add_output(module, "out", &output_type);
    let verilog = file.emit();
    let want = r#"module main(
  input wire [31:0] in,
  output wire out
);

endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_continuous_assignment_of_slice() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let input_type = file.make_bit_vector_type(8, false);
    let output_type = file.make_bit_vector_type(4, false);
    let input = file.add_input(module, "my_input", &input_type);
    let output = file.add_output(module, "my_output", &output_type);
    let slice = file.make_slice(&input.to_indexable_expr(), 3, 0);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &slice.to_expr());
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] my_input,
  output wire [3:0] my_output
);
  assign my_output = my_input[3:0];
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_instantiation() {
    let mut file = VastFile::new(VastFileType::Verilog);

    let data_type = file.make_bit_vector_type(8, false);

    let a_module = file.add_module("A");
    file.add_output(a_module, "bus", &data_type);

    let b_module = file.add_module("B");
    let bus = file.add_wire(b_module, "bus", &data_type);

    let param_value = file
        .make_literal("bits[32]:42", &LiteralFormat::UnsignedDecimal)
        .unwrap();

    let instantiation = file.make_instantiation(
        "A",
        "a_i",
        &["a_param"],
        &[&param_value],
        &["bus", "empty_thing"],
        &[Some(&bus.to_expr()), None],
    );
    file.add_member_instantiation(b_module, instantiation);

    let verilog = file.emit();
    let want = r#"module A(
  output wire [7:0] bus
);

endmodule
module B;
  wire [7:0] bus;
  A #(
    .a_param(32'd42)
  ) a_i (
    .bus(bus),
    .empty_thing()
  );
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_main_module() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("main");
    let input_type = file.make_bit_vector_type(32, false);
    let output_type = file.make_scalar_type();
    file.add_input(module, "in", &input_type);
    file.add_output(module, "out", &output_type);
    let verilog = file.emit();
    let want = r#"module main(
  input wire [31:0] in,
  output wire out
);

endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_literal() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let wire_data_type = file.make_bit_vector_type(128, false);
    let wire = file.add_wire(module, "bus", &wire_data_type);
    let literal = file
        .make_literal(
            "bits[128]:0xFFEEDDCCBBAA99887766554433221100",
            &LiteralFormat::Hex,
        )
        .unwrap();
    let assignment = file.make_continuous_assignment(&wire.to_expr(), &literal);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module;
  wire [127:0] bus;
  assign bus = 128'hffee_ddcc_bbaa_9988_7766_5544_3322_1100;
endmodule
"#;
    assert_eq!(verilog, want);
}

/// Tests that we can make a port with an external-package-defined struct as
/// the type, and we also place it in a packed array.
#[test]
fn test_port_with_external_package_struct() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let my_struct = file.make_extern_package_type("mypack", "mystruct_t");
    let input_type = file.make_packed_array_type(my_struct, &[2, 3, 4]);
    file.add_input(module, "my_input", &input_type);
    let want = r#"module my_module(
  input mypack::mystruct_t [1:0][2:0][3:0] my_input
);

endmodule
"#;
    assert_eq!(file.emit(), want);
}

/// Tests that we can build a module with a simple concatenation.
#[test]
fn test_simple_concat() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let input_type = file.make_bit_vector_type(8, false);
    let output_type = file.make_bit_vector_type(16, false);
    let input = file.add_input(module, "my_input", &input_type);
    let output = file.add_output(module, "my_output", &output_type);
    let concat = file.make_concat(&[&input.to_expr(), &input.to_expr()]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] my_input,
  output wire [15:0] my_output
);
  assign my_output = {my_input, my_input};
endmodule
"#;
    assert_eq!(verilog, want);
}

/// Tests that we can reference a slice of a multidimensional packed array
/// on the LHS or RHS of an assign statement.
#[test]
fn test_slice_on_both_sides_of_assignment() {
    let want = r#"module my_module;
  wire [2:0][4:0][1:0] a;
  wire [1:0] b;
  wire [2:0] c;
  assign a[1][2][3:4] = b[1:0];
  assign a[3:4] = c[2:1];
endmodule
"#;

    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let u2 = file.make_bit_vector_type(2, false);
    let a_type = file.make_packed_array_type(u2, &[3, 5]);
    let b_type = file.make_bit_vector_type(2, false);
    let c_type = file.make_bit_vector_type(3, false);
    let a = file.add_wire(module, "a", &a_type);
    let b = file.add_wire(module, "b", &b_type);
    let c = file.add_wire(module, "c", &c_type);

    // First assignment.
    {
        let a_1 = file.make_index(&a.to_indexable_expr(), 1);
        let a_2 = file.make_index(&a_1.to_indexable_expr(), 2);
        let a_lhs = file.make_slice(&a_2.to_indexable_expr(), 3, 4);
        let b_slice = file.make_slice(&b.to_indexable_expr(), 1, 0);
        let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &b_slice.to_expr());
        file.add_member_continuous_assignment(module, assignment);
    }

    // Second assignment.
    {
        let a_lhs = file.make_slice(&a.to_indexable_expr(), 3, 4);
        let c_slice = file.make_slice(&c.to_indexable_expr(), 2, 1);
        let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &c_slice.to_expr());
        file.add_member_continuous_assignment(module, assignment);
    }

    let verilog = file.emit();
    assert_eq!(verilog, want);
}

#[test]
fn test_index_then_add_constant_uses_indexable_to_expr() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("idx_add");

    // Create a 3-bit wire `x` and a 1-bit wire `y`.
    let x_data_type = file.make_bit_vector_type(3, false);
    let x = file.add_wire(module, "x", &x_data_type);
    let y_data_type = file.make_scalar_type();
    let y = file.add_wire(module, "y", &y_data_type);

    // Build (x[2]) using index -> indexable -> expr to exercise the new API.
    let idx = file.make_index(&x.to_indexable_expr(), 2);
    let idx_expr = idx.to_indexable_expr().to_expr();

    // Add a 1-bit constant to the indexed bit.
    let one = file
        .make_literal("bits[1]:1", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let sum = file.make_add(&idx_expr, &one);

    // Emit as a continuous assignment so it appears in the module body.
    let assign = file.make_continuous_assignment(&y.to_expr(), &sum);
    file.add_member_continuous_assignment(module, assign);

    let verilog = file.emit();
    let want = r#"module idx_add;
  wire [2:0] x;
  wire y;
  assign y = x[2] + 1'd1;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_slice_and_index_with_expressions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let element_type = file.make_bit_vector_type(8, false);
    let arr = file.add_wire(module, "arr", &element_type);
    let hi_data_type = file.make_bit_vector_type(4, false);
    let hi = file.add_wire(module, "hi", &hi_data_type);
    let lo_data_type = file.make_bit_vector_type(4, false);
    let lo = file.add_wire(module, "lo", &lo_data_type);
    let idx_data_type = file.make_bit_vector_type(3, false);
    let idx = file.add_wire(module, "idx", &idx_data_type);
    let slice_out = file.add_wire(module, "slice_out", &element_type);
    let index_out_data_type = file.make_scalar_type();
    let index_out = file.add_wire(module, "index_out", &index_out_data_type);

    let arr_indexable = arr.to_indexable_expr();
    let hi_expr = hi.to_expr();
    let lo_expr = lo.to_expr();
    let idx_expr = idx.to_expr();

    let slice = file.make_slice_expr(&arr_indexable, &hi_expr, &lo_expr);
    let index = file.make_index_expr(&arr_indexable, &idx_expr);

    let slice_assign = file.make_continuous_assignment(&slice_out.to_expr(), &slice.to_expr());
    file.add_member_continuous_assignment(module, slice_assign);
    let index_assign = file.make_continuous_assignment(&index_out.to_expr(), &index.to_expr());
    file.add_member_continuous_assignment(module, index_assign);

    let verilog = file.emit();
    let want = r#"module my_module;
  wire [7:0] arr;
  wire [3:0] hi;
  wire [3:0] lo;
  wire [2:0] idx;
  wire [7:0] slice_out;
  wire index_out;
  assign slice_out = arr[hi:lo];
  assign index_out = arr[idx];
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_concat_various_expressions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let input_data_type = file.make_bit_vector_type(8, false);
    let input = file.add_input(module, "my_input", &input_data_type);
    let output_data_type = file.make_bit_vector_type(9, false);
    let output = file.add_output(module, "my_output", &output_data_type);
    let input_indexable = input.to_indexable_expr();
    let index = file.make_index(&input_indexable, 0);
    let slice = file.make_slice(&input_indexable, 7, 0);
    let concat = file.make_concat(&[&index.to_expr(), &slice.to_expr()]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] my_input,
  output wire [8:0] my_output
);
  assign my_output = {my_input[0], my_input[7:0]};
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_unary_ops() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let input_data_type = file.make_bit_vector_type(8, false);
    let input = file.add_input(module, "my_input", &input_data_type);
    let not_input = file.make_not(&input.to_expr());
    let negate_input = file.make_negate(&input.to_expr());
    let bitwise_not_input = file.make_bitwise_not(&input.to_expr());
    let logical_not_input = file.make_logical_not(&input.to_expr());
    let and_reduce_input = file.make_and_reduce(&input.to_expr());
    let or_reduce_input = file.make_or_reduce(&input.to_expr());
    let xor_reduce_input = file.make_xor_reduce(&input.to_expr());
    let concat = file.make_concat(&[
        &not_input,         // 8 bits
        &negate_input,      // 8 bits
        &bitwise_not_input, // 8 bits
        &logical_not_input, // 1 bit
        &and_reduce_input,  // 1 bit
        &or_reduce_input,   // 1 bit
        &xor_reduce_input,  // 1 bit
    ]);
    let concat_type = file.make_bit_vector_type(8 + 8 + 8 + 1 + 1 + 1 + 1, false);
    let output = file.add_output(module, "my_output", &concat_type);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] my_input,
  output wire [27:0] my_output
);
  assign my_output = {~my_input, -my_input, ~my_input, !my_input, &my_input, |my_input, ^my_input};
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_binary_ops() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let u8 = file.make_bit_vector_type(8, false);
    let u1 = file.make_bit_vector_type(1, false);
    let lhs = file.add_input(module, "lhs", &u8);
    let rhs = file.add_input(module, "rhs", &u8);
    let functions: Vec<(&str, VastBinOp, &VastDataType)> = vec![
        ("add", VastFile::make_add, &u8),
        ("logical_and", VastFile::make_logical_and, &u1),
        ("bitwise_and", VastFile::make_bitwise_and, &u8),
        ("ne", VastFile::make_ne, &u1),
        ("case_ne", VastFile::make_case_ne, &u1),
        ("eq", VastFile::make_eq, &u1),
        ("case_eq", VastFile::make_case_eq, &u1),
        ("ge", VastFile::make_ge, &u1),
        ("gt", VastFile::make_gt, &u1),
        ("le", VastFile::make_le, &u1),
        ("lt", VastFile::make_lt, &u1),
        ("div", VastFile::make_div, &u8),
        ("mod", VastFile::make_mod, &u8),
        ("mul", VastFile::make_mul, &u8),
        ("power", VastFile::make_power, &u8),
        ("bitwise_or", VastFile::make_bitwise_or, &u8),
        ("logical_or", VastFile::make_logical_or, &u1),
        ("bitwise_xor", VastFile::make_bitwise_xor, &u8),
        ("shll", VastFile::make_shll, &u8),
        ("shra", VastFile::make_shra, &u8),
        ("shrl", VastFile::make_shrl, &u8),
        ("sub", VastFile::make_sub, &u8),
        ("ne_x", VastFile::make_ne_x, &u1),
        ("eq_x", VastFile::make_eq_x, &u1),
    ];
    for (name, f, output_type) in functions {
        let wire = file.add_wire(module, name, output_type);
        let rhs_expr = f(&mut file, &lhs.to_expr(), &rhs.to_expr());
        let assignment = file.make_continuous_assignment(&wire.to_expr(), &rhs_expr);
        file.add_member_continuous_assignment(module, assignment);
    }

    // Now emit the VAST as text.
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] lhs,
  input wire [7:0] rhs
);
  wire [7:0] add;
  assign add = lhs + rhs;
  wire logical_and;
  assign logical_and = lhs && rhs;
  wire [7:0] bitwise_and;
  assign bitwise_and = lhs & rhs;
  wire ne;
  assign ne = lhs != rhs;
  wire case_ne;
  assign case_ne = lhs !== rhs;
  wire eq;
  assign eq = lhs == rhs;
  wire case_eq;
  assign case_eq = lhs === rhs;
  wire ge;
  assign ge = lhs >= rhs;
  wire gt;
  assign gt = lhs > rhs;
  wire le;
  assign le = lhs <= rhs;
  wire lt;
  assign lt = lhs < rhs;
  wire [7:0] div;
  assign div = lhs / rhs;
  wire [7:0] mod;
  assign mod = lhs % rhs;
  wire [7:0] mul;
  assign mul = lhs * rhs;
  wire [7:0] power;
  assign power = lhs ** rhs;
  wire [7:0] bitwise_or;
  assign bitwise_or = lhs | rhs;
  wire logical_or;
  assign logical_or = lhs || rhs;
  wire [7:0] bitwise_xor;
  assign bitwise_xor = lhs ^ rhs;
  wire [7:0] shll;
  assign shll = lhs << rhs;
  wire [7:0] shra;
  assign shra = lhs >>> rhs;
  wire [7:0] shrl;
  assign shrl = lhs >> rhs;
  wire [7:0] sub;
  assign sub = lhs - rhs;
  wire ne_x;
  assign ne_x = lhs !== rhs;
  wire eq_x;
  assign eq_x = lhs === rhs;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_nested_generate_loops_with_assignment() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("gen_nested");
    let scalar = file.make_scalar_type();

    let a = file.add_input(module, "a", &scalar);
    let b = file.add_output(module, "b", &scalar);

    // for (genvar i = 0; i < 2; ++i) begin: outer
    let zero = file.make_unsized_decimal_literal(0);
    let two = file.make_unsized_decimal_literal(2);
    let outer = file.add_generate_loop(module, "i", &zero, &two, Some("outer"));

    //   for (genvar j = 1; j < 3; ++j) begin: inner
    let one = file.make_unsized_decimal_literal(1);
    let three = file.make_unsized_decimal_literal(3);
    let inner = file.generate_add_generate_loop(outer, "j", &one, &three, Some("inner"));

    //     assign b = a;
    file.generate_add_continuous_assignment(inner, &b.to_expr(), &a.to_expr());

    let verilog = file.emit();
    let want = r#"module gen_nested(
  input wire a,
  output wire b
);
  for (genvar i = 0; i < 2; i = i + 1) begin : outer
    for (genvar j = 1; j < 3; j = j + 1) begin : inner
      assign b = a;
    end
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_width_cast_basic() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("width_cast");
    let u16 = file.make_bit_vector_type(16, false);
    let u8 = file.make_bit_vector_type(8, false);

    let x = file.add_input(module, "x", &u16);
    let y = file.add_output(module, "y", &u8);

    let width8 = file.make_unsized_decimal_literal(8);
    let cast = file.make_width_cast(&width8, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    file.add_member_continuous_assignment(module, assign);

    let verilog = file.emit();
    let want = r#"module width_cast(
  input wire [15:0] x,
  output wire [7:0] y
);
  assign y = 8'(x);
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_generate_loop_with_localparam_and_empty_always_blocks() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("gen_empty_blocks");

    let zero = file.make_unsized_decimal_literal(0);
    let two = file.make_unsized_decimal_literal(2);
    let gen_loop = file.add_generate_loop(module, "i", &zero, &two, Some("G"));

    let five = file.make_unsized_decimal_literal(5);
    file.generate_add_localparam(gen_loop, "LP", &five);

    file.generate_add_always_comb(gen_loop).unwrap();
    file.generate_add_always_ff(gen_loop, &[]).unwrap();

    let verilog = file.emit();
    let want = r#"module gen_empty_blocks;
  for (genvar i = 0; i < 2; i = i + 1) begin : G
    localparam LP = 5;
    always_comb begin end
    always_ff @ () begin end
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_type_cast_basic() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("type_cast");
    let u8 = file.make_bit_vector_type(8, false);
    let x = file.add_input(module, "x", &u8);
    let y = file.add_output(module, "y", &u8);

    let user_t = file.make_extern_package_type("", "my_type_t");
    let cast = file.make_type_cast(&user_t, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    file.add_member_continuous_assignment(module, assign);

    let verilog = file.emit();
    let want = r#"module type_cast(
  input wire [7:0] x,
  output wire [7:0] y
);
  assign y = ::my_type_t'(x);
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_add_inout_port() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("with_inout");
    let scalar = file.make_scalar_type();
    file.add_inout(module, "io", &scalar);
    let verilog = file.emit();
    let want = r#"module with_inout(
  inout wire io
);

endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_ternary() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let selector_data_type = file.make_bit_vector_type(8, false);
    let selector = file.add_input(module, "selector", &selector_data_type);
    let on_true_data_type = file.make_bit_vector_type(8, false);
    let on_true = file.add_input(module, "on_true", &on_true_data_type);
    let on_false_data_type = file.make_bit_vector_type(8, false);
    let on_false = file.add_input(module, "on_false", &on_false_data_type);
    let ternary = file.make_ternary(&selector.to_expr(), &on_true.to_expr(), &on_false.to_expr());
    let output_data_type = file.make_bit_vector_type(8, false);
    let output = file.add_output(module, "my_output", &output_data_type);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &ternary);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire [7:0] selector,
  input wire [7:0] on_true,
  input wire [7:0] on_false,
  output wire [7:0] my_output
);
  assign my_output = selector ? on_true : on_false;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_replicated_concat_i64() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let a_data_type = file.make_scalar_type();
    let a = file.add_input(module, "a", &a_data_type);
    let b_data_type = file.make_scalar_type();
    let b = file.add_input(module, "b", &b_data_type);
    let w_data_type = file.make_bit_vector_type(6, false);
    let w = file.add_wire(module, "w", &w_data_type);
    let expr = file.make_replicated_concat_i64(3, &[&a.to_expr(), &b.to_expr()]);
    let assignment = file.make_continuous_assignment(&w.to_expr(), &expr);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire a,
  input wire b
);
  wire [5:0] w;
  assign w = {3{a, b}};
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_replicated_concat_expr() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("my_module");
    let a_data_type = file.make_scalar_type();
    let a = file.add_input(module, "a", &a_data_type);
    let b_data_type = file.make_scalar_type();
    let b = file.add_input(module, "b", &b_data_type);
    let w_data_type = file.make_bit_vector_type(6, false);
    let w = file.add_wire(module, "w", &w_data_type);
    let rep = file
        .make_literal("bits[32]:3", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let expr = file.make_replicated_concat(&rep, &[&a.to_expr(), &b.to_expr()]);
    let assignment = file.make_continuous_assignment(&w.to_expr(), &expr);
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module my_module(
  input wire a,
  input wire b
);
  wire [5:0] w;
  assign w = {32'd3{a, b}};
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_integer_type_port() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("m");
    let int_t = file.make_integer_type(true);
    file.add_input(module, "i", &int_t);
    let verilog = file.emit();
    let want = r#"module m(
  input wire i
);

endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_module_parameter_and_use_in_assignment() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("P");
    let lit = file
        .make_literal("bits[32]:4", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let pref = file.add_parameter(module, "N", &lit);
    let out_data_type = file.make_bit_vector_type(32, false);
    let out = file.add_output(module, "o", &out_data_type);
    let assignment = file.make_continuous_assignment(&out.to_expr(), &pref.to_expr());
    file.add_member_continuous_assignment(module, assignment);
    let verilog = file.emit();
    let want = r#"module P(
  output wire [31:0] o
);
  parameter N = 32'd4;
  assign o = N;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_module_parameter_with_def_integer() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("P2");
    let int_t = file.make_integer_type(true);
    let def = file.make_def("N2", DataKind::Integer, &int_t);
    let lit = file
        .make_literal("bits[32]:7", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let _pref = file.add_parameter_with_def(module, &def, &lit);
    let verilog = file.emit();
    let want = r#"module P2;
  parameter integer N2 = 32'd7;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_module_localparams_various_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("LM");

    // localparam int Foo = 42;
    let forty_two = file.make_unsized_decimal_literal(42);
    file.add_int_localparam(module, "Foo", &forty_two);

    // localparam Bar = 100;
    let one_hundred = file.make_unsized_decimal_literal(100);
    file.add_localparam(module, "Bar", &one_hundred);

    // localparam Qux = 'h10000;
    let qux = file
        .make_literal("bits[32]:0x10000", &LiteralFormat::UnsizedHex)
        .unwrap();
    file.add_localparam(module, "Bar", &qux);

    // localparam logic [7:0] Baz = 8'h44;
    let logic8 = file.make_bit_vector_type(8, false);
    let baz_def = file.make_def("Baz", DataKind::Logic, &logic8);
    let hex_44 = file
        .make_literal("bits[8]:0x44", &LiteralFormat::Hex)
        .unwrap();
    file.add_typed_localparam(module, &baz_def, &hex_44);

    // localparam logic [7:0] Zero = '0;
    let zero_def = file.make_def("Zero", DataKind::Logic, &logic8);
    let unsized_zero = file.make_unsized_zero_literal();
    file.add_typed_localparam(module, &zero_def, &unsized_zero);

    // localparam logic [7:0] Ones = '1;
    let ones_def = file.make_def("Ones", DataKind::Logic, &logic8);
    let unsized_one = file.make_unsized_one_literal();
    file.add_typed_localparam(module, &ones_def, &unsized_one);

    let verilog = file.emit();
    let want = r#"module LM;
  localparam int Foo = 42;
  localparam Bar = 100;
  localparam Bar = 'h1_0000;
  localparam logic [7:0] Baz = 8'h44;
  localparam logic [7:0] Zero = '0;
  localparam logic [7:0] Ones = '1;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_inline_blank_comment_module_members() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("M");
    let member = file.make_inline_verilog_statement("/* first */");
    file.add_member_inline_statement(module, member);
    let member = file.make_blank_line();
    file.add_member_blank_line(module, member);
    let member = file.make_comment("Single line comment");
    file.add_member_comment(module, member);
    let member = file.make_inline_verilog_statement("/* second */");
    file.add_member_inline_statement(module, member);
    let verilog = file.emit();
    let want = r#"module M;
  /* first */

  // Single line comment
  /* second */
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_sequential_logic_system_verilog() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("test_module");

    let scalar_type = file.make_scalar_type();

    let clk = file.add_input(module, "clk", &scalar_type);
    let pred = file.add_input(module, "pred", &scalar_type);
    let x = file.add_input(module, "x", &scalar_type);
    file.add_output(module, "out", &scalar_type);

    let p0_pred_reg = file.add_reg(module, "p0_pred", &scalar_type).unwrap();
    let p0_x_reg = file.add_reg(module, "p0_x", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk.to_expr());

    let always_block = file.add_always_ff(module, &[&posedge_clk]).unwrap();

    let stmt_block = file.statement_block(always_block);

    file.block_add_nonblocking_assignment(stmt_block, &p0_pred_reg.to_expr(), &pred.to_expr());
    file.block_add_comment_text(stmt_block, "capture pred");
    file.block_add_blank_line(stmt_block);
    file.block_add_inline_text(stmt_block, "/* combo capture */");
    file.block_add_nonblocking_assignment(stmt_block, &p0_x_reg.to_expr(), &x.to_expr());

    let verilog = file.emit();

    let want = r#"module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always_ff @ (posedge clk) begin
    p0_pred <= pred;
    // capture pred

    /* combo capture */
    p0_x <= x;
  end
endmodule
"#;

    assert_eq!(verilog, want);
}

#[test]
fn blocking_assignment_emits_system_verilog() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("test_module");
    let scalar_type = file.make_scalar_type();
    let _clk = file.add_input(module, "clk", &scalar_type);
    let x = file.add_input(module, "x", &scalar_type);
    let r = file.add_reg(module, "r", &scalar_type).unwrap();

    let always_block = file.add_always_comb(module).unwrap();
    let sb = file.statement_block(always_block);
    file.block_add_blocking_assignment(sb, &r.to_expr(), &x.to_expr());

    let verilog = file.emit();
    let want = r#"module test_module(
  input wire clk,
  input wire x
);
  reg r;
  always_comb begin
    r = x;
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn blocking_assignment_emits_verilog() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("test_module");
    let scalar_type = file.make_scalar_type();
    let clk = file.add_input(module, "clk", &scalar_type);
    let x = file.add_input(module, "x", &scalar_type);
    let r = file.add_reg(module, "r", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always_block = file.add_always_at(module, &[&posedge_clk]).unwrap();
    let sb = file.statement_block(always_block);
    file.block_add_blocking_assignment(sb, &r.to_expr(), &x.to_expr());

    let verilog = file.emit();
    let want = r#"module test_module(
  input wire clk,
  input wire x
);
  reg r;
  always @ (posedge clk) begin
    r = x;
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn conditional_emits_system_verilog() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("M");
    let bit = file.make_scalar_type();
    let clk = file.add_input(module, "clk", &bit);
    let a = file.add_input(module, "a", &bit);
    let b = file.add_input(module, "b", &bit);
    let r = file.add_reg(module, "r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = file.add_always_ff(module, &[&posedge_clk]).unwrap();
    let sb = file.statement_block(always);
    let cond = file.block_add_cond(sb, &a.to_expr());
    let then_block = file.conditional_then_block(cond);
    file.block_add_nonblocking_assignment(then_block, &r.to_expr(), &a.to_expr());
    let else_if_block = file.conditional_add_else_if(cond, &b.to_expr());
    file.block_add_nonblocking_assignment(else_if_block, &r.to_expr(), &b.to_expr());
    let else_block = file.conditional_add_else(cond);
    file.block_add_nonblocking_assignment(else_block, &r.to_expr(), &a.to_expr());
    let verilog = file.emit();
    let want = r#"module M(
  input wire clk,
  input wire a,
  input wire b
);
  reg r;
  always_ff @ (posedge clk) begin
    if (a) begin
      r <= a;
    end else if (b) begin
      r <= b;
    end else begin
      r <= a;
    end
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn conditional_emits_verilog() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("M");
    let bit = file.make_scalar_type();
    let clk = file.add_input(module, "clk", &bit);
    let a = file.add_input(module, "a", &bit);
    let b = file.add_input(module, "b", &bit);
    let r = file.add_reg(module, "r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = file.add_always_at(module, &[&posedge_clk]).unwrap();
    let sb = file.statement_block(always);
    let cond = file.block_add_cond(sb, &a.to_expr());
    let then_block = file.conditional_then_block(cond);
    file.block_add_nonblocking_assignment(then_block, &r.to_expr(), &a.to_expr());
    let else_if_block = file.conditional_add_else_if(cond, &b.to_expr());
    file.block_add_nonblocking_assignment(else_if_block, &r.to_expr(), &b.to_expr());
    let else_block = file.conditional_add_else(cond);
    file.block_add_nonblocking_assignment(else_block, &r.to_expr(), &a.to_expr());
    let verilog = file.emit();
    let want = r#"module M(
  input wire clk,
  input wire a,
  input wire b
);
  reg r;
  always @ (posedge clk) begin
    if (a) begin
      r <= a;
    end else if (b) begin
      r <= b;
    end else begin
      r <= a;
    end
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn case_emits_system_verilog() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("C");
    let bit = file.make_scalar_type();
    let clk = file.add_logic_input(module, "clk", &bit);
    let sel = file.add_logic_input(module, "sel", &bit);
    let a = file.add_logic_input(module, "a", &bit);
    let b = file.add_logic_input(module, "b", &bit);
    let r = file.add_logic(module, "r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = file.add_always_ff(module, &[&posedge_clk]).unwrap();
    let sb = file.statement_block(always);
    let case_stmt = file.block_add_case(sb, &sel.to_expr());
    let item_a = file.case_add_item(case_stmt, &a.to_expr());
    file.block_add_nonblocking_assignment(item_a, &r.to_expr(), &a.to_expr());
    let item_b = file.case_add_item(case_stmt, &b.to_expr());
    file.block_add_nonblocking_assignment(item_b, &r.to_expr(), &b.to_expr());
    let default_block = file.case_add_default(case_stmt);
    file.block_add_nonblocking_assignment(default_block, &r.to_expr(), &a.to_expr());
    let verilog = file.emit();
    let want = r#"module C(
  input logic clk,
  input logic sel,
  input logic a,
  input logic b
);
  logic r;
  always_ff @ (posedge clk) begin
    case (sel)
      a: begin
        r <= a;
      end
      b: begin
        r <= b;
      end
      default: begin
        r <= a;
      end
    endcase
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn case_emits_verilog() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("C");
    let bit = file.make_scalar_type();
    let clk = file.add_input(module, "clk", &bit);
    let sel = file.add_input(module, "sel", &bit);
    let a = file.add_input(module, "a", &bit);
    let b = file.add_input(module, "b", &bit);
    let r = file.add_reg(module, "r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = file.add_always_ff(module, &[&posedge_clk]).unwrap();
    let sb = file.statement_block(always);
    let case_stmt = file.block_add_case(sb, &sel.to_expr());
    let item_a = file.case_add_item(case_stmt, &a.to_expr());
    file.block_add_nonblocking_assignment(item_a, &r.to_expr(), &a.to_expr());
    let item_b = file.case_add_item(case_stmt, &b.to_expr());
    file.block_add_nonblocking_assignment(item_b, &r.to_expr(), &b.to_expr());
    let default_block = file.case_add_default(case_stmt);
    file.block_add_nonblocking_assignment(default_block, &r.to_expr(), &a.to_expr());
    let verilog = file.emit();
    let want = r#"module C(
  input wire clk,
  input wire sel,
  input wire a,
  input wire b
);
  reg r;
  always_ff @ (posedge clk) begin
    case (sel)
      a: begin
        r <= a;
      end
      b: begin
        r <= b;
      end
      default: begin
        r <= a;
      end
    endcase
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn bit_vector_type_expr_with_parameter_port() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("P");
    let lit = file
        .make_literal("bits[32]:4", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let n = file.add_parameter(module, "N", &lit);
    let dt = file.make_bit_vector_type_expr(&n.to_expr(), false);
    file.add_output(module, "o", &dt);
    let verilog = file.emit();
    let want = r#"module P(
  output wire [N - 1:0] o
);
  parameter N = 32'd4;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn bit_vector_type_expr_with_literal() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("M");
    let lit = file
        .make_literal("bits[32]:5", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    let dt = file.make_bit_vector_type_expr(&lit, false);
    file.add_wire(module, "w", &dt);
    let verilog = file.emit();
    let want = r#"module M;
  wire [4:0] w;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn module_with_parameters() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("C");
    let bit = file.make_scalar_type();
    let n_default = file.make_unsized_decimal_literal(42);
    let n = file.add_parameter_port(module, "N", &n_default);
    let foo_type = file.make_bit_vector_type(16, false);
    let foo_default = file
        .make_literal("bits[16]:5", &LiteralFormat::UnsignedDecimal)
        .unwrap();
    file.add_typed_parameter_port(module, "Foo", &foo_type, &foo_default);
    file.add_logic_input(module, "clk", &bit);
    let a_type = file.make_bit_vector_type_expr(&n.to_expr(), false);
    file.add_logic_input(module, "a", &a_type);
    let verilog = file.emit();
    let want = r#"module C #(
  parameter N = 42,
  parameter logic [15:0] Foo = 16'd5
) (
  input logic clk,
  input logic [N - 1:0] a
);

endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_type_cast_to_unqualified_user_type() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("type_cast2");
    let u8 = file.make_bit_vector_type(8, false);
    let x = file.add_input(module, "x", &u8);
    let y = file.add_output(module, "y", &u8);

    // Use an unqualified user type via extern_type.
    let user_t = file.make_extern_type("my_type_t");
    let cast = file.make_type_cast(&user_t, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    file.add_member_continuous_assignment(module, assign);

    let verilog = file.emit();
    let want = r#"module type_cast2(
  input wire [7:0] x,
  output wire [7:0] y
);
  assign y = my_type_t'(x);
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_module_macro_statement_simple() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("macro_mod");

    // Add a simple macro statement: `MY_MACRO;
    let mref1 = file.make_macro_ref("MY_MACRO1");
    let mstmt1 = file.make_macro_statement(&mref1, true);
    let mref2 = file.make_macro_ref("MY_MACRO2");
    let mstmt2 = file.make_macro_statement(&mref2, false);
    file.add_member_macro_statement(module, mstmt1);
    file.add_member_macro_statement(module, mstmt2);

    let verilog = file.emit();
    let want = r#"module macro_mod;
  `MY_MACRO1;
  `MY_MACRO2
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_generate_loop_with_inline_and_macro() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("gen_with_macros");

    // for (genvar i = 0; i < 1; ++i) begin : G
    let zero = file.make_unsized_decimal_literal(0);
    let one = file.make_unsized_decimal_literal(1);
    let gen_loop = file.add_generate_loop(module, "i", &zero, &one, Some("G"));

    // Comment, blank line, and macro statements inside the loop.
    let comment = file.make_comment("inside");
    file.generate_add_comment(gen_loop, &comment);
    file.generate_add_blank_line(gen_loop);
    let mref = file.make_macro_ref("DO_SOMETHING");
    let mstmt = file.make_macro_statement(&mref, true);
    file.generate_add_macro_statement(gen_loop, &mstmt);
    // Macro with arguments.
    let three = file.make_unsized_decimal_literal(3);
    let mref_args = file.make_macro_ref_with_args("DO_THING", &[&three]);
    let mstmt_args = file.make_macro_statement(&mref_args, false);
    file.generate_add_macro_statement(gen_loop, &mstmt_args);

    let verilog = file.emit();
    let want = r#"module gen_with_macros;
  for (genvar i = 0; i < 1; i = i + 1) begin : G
    // inside

    `DO_SOMETHING;
    `DO_THING(3)
  end
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_expression_emit_plain_literal() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let three = file.make_unsized_decimal_literal(3);
    let s = file.emit_expression(&three);
    assert_eq!(s, "3");
}

#[test]
fn test_file_level_comment_and_blank_line() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.add_comment_text("top-level comment");
    let blank = file.make_blank_line();
    file.add_blank_line(blank);
    let module = file.add_module("M");
    let scalar = file.make_scalar_type();
    file.add_wire(module, "w", &scalar);
    let verilog = file.emit();
    let want = r#"// top-level comment

module M;
  wire w;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_array_parameters_with_def_and_assignment_pattern() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");

    // Common scalar type: 1-bit element for several unpacked arrays.
    let scalar = file.make_scalar_type();

    // P0: parameter P0[2] = '{'0, '0};
    let p0_type = file.make_unpacked_array_type(scalar.clone(), &[2]);
    let p0_def = file.make_def("P0", DataKind::User, &p0_type);
    let tick0 = file.make_unsized_zero_literal();
    let p0_rhs = file.make_array_assignment_pattern(&[&tick0, &tick0]);
    file.add_parameter_with_def(module, &p0_def, &p0_rhs);

    // P1: parameter int P1[3] = '{1, 2, 3};
    let p1_type = file.make_unpacked_array_type(scalar.clone(), &[3]);
    let p1_def = file.make_def("P1", DataKind::Int, &p1_type);
    let one = file.make_unsized_decimal_literal(1);
    let two = file.make_unsized_decimal_literal(2);
    let three = file.make_unsized_decimal_literal(3);
    let four = file.make_unsized_decimal_literal(4);
    let five = file.make_unsized_decimal_literal(5);
    let six = file.make_unsized_decimal_literal(6);
    let p1_rhs = file.make_array_assignment_pattern(&[&one, &two, &three]);
    file.add_parameter_with_def(module, &p1_def, &p1_rhs);

    // P2: parameter logic [7:0] P2[2] = '{8'h42, 8'h43};
    let u8 = file.make_bit_vector_type(8, false);
    let p2_type = file.make_unpacked_array_type(u8, &[2]);
    let p2_def = file.make_def("P2", DataKind::Logic, &p2_type);
    let lit_42 = file
        .make_literal("bits[8]:0x42", &LiteralFormat::Hex)
        .unwrap();
    let lit_43 = file
        .make_literal("bits[8]:0x43", &LiteralFormat::Hex)
        .unwrap();
    let p2_rhs = file.make_array_assignment_pattern(&[&lit_42, &lit_43]);
    file.add_parameter_with_def(module, &p2_def, &p2_rhs);

    // P3: parameter integer P3[1][4] = '{'{1, 2, 3, 4}};
    let p3_type = file.make_unpacked_array_type(scalar.clone(), &[1, 4]);
    let p3_def = file.make_def("P3", DataKind::Integer, &p3_type);
    let inner_1234 = file.make_array_assignment_pattern(&[&one, &two, &three, &four]);
    let p3_rhs = file.make_array_assignment_pattern(&[&inner_1234]);
    file.add_parameter_with_def(module, &p3_def, &p3_rhs);

    // P4: parameter int P4[2][3] = '{'{1, 2, 3}, '{4, 5, 6}};
    let p4_type = file.make_unpacked_array_type(scalar, &[2, 3]);
    let p4_def = file.make_def("P4", DataKind::Int, &p4_type);
    let row_123 = file.make_array_assignment_pattern(&[&one, &two, &three]);
    let row_456 = file.make_array_assignment_pattern(&[&four, &five, &six]);
    let p4_rhs = file.make_array_assignment_pattern(&[&row_123, &row_456]);
    file.add_parameter_with_def(module, &p4_def, &p4_rhs);

    let verilog = file.emit();
    let want = r#"module top;
  parameter P0[2] = '{'0, '0};
  parameter int P1[3] = '{1, 2, 3};
  parameter logic [7:0] P2[2] = '{8'h42, 8'h43};
  parameter integer P3[1][4] = '{'{1, 2, 3, 4}};
  parameter int P4[2][3] = '{'{1, 2, 3}, '{4, 5, 6}};
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_module_level_conditional() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("top");

    // parameter A = 1;
    // parameter B = 2;
    let one_param = file.make_unsized_decimal_literal(1);
    let two_param = file.make_unsized_decimal_literal(2);
    let a = file.add_parameter(module, "A", &one_param);
    let b = file.add_parameter(module, "B", &two_param);

    // wire out;
    let scalar = file.make_scalar_type();
    let out = file.add_wire(module, "out", &scalar);

    // if (A == B) begin
    //   assign out = 1'h1;
    // end else begin
    //   assign out = 1'h0;
    // end
    let cond_expr = file.make_eq(&a.to_expr(), &b.to_expr());
    let cond = file.add_conditional(module, &cond_expr);
    let then_block = file.conditional_then_block(cond);
    let one = file
        .make_literal("bits[1]:0x1", &LiteralFormat::Hex)
        .unwrap();
    let zero = file
        .make_literal("bits[1]:0x0", &LiteralFormat::Hex)
        .unwrap();
    file.block_add_continuous_assignment(then_block, &out.to_expr(), &one);
    let else_block = file.conditional_add_else(cond);
    file.block_add_continuous_assignment(else_block, &out.to_expr(), &zero);

    let verilog = file.emit();
    let want = r#"module top;
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
    assert_eq!(verilog, want);
}

#[test]
fn test_generate_loop_conditional_assignments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");

    // wire [2:0] out;
    let out_type = file.make_bit_vector_type(3, false);
    let out = file.add_wire(module, "out", &out_type);

    // for (genvar i = 0; i < 3; i = i + 1) begin : g
    let zero = file.make_unsized_decimal_literal(0);
    let three = file.make_unsized_decimal_literal(3);
    let gen_loop = file.add_generate_loop(module, "i", &zero, &three, Some("g"));

    let i_ref = file.generate_genvar(gen_loop);
    let i_expr = i_ref.to_expr();
    let zero_cond = file.make_unsized_decimal_literal(0);
    let one_cond = file.make_unsized_decimal_literal(1);

    let cond0 = file.make_eq(&i_expr, &zero_cond);
    let cond1 = file.make_eq(&i_expr, &one_cond);
    let cond = file.generate_add_conditional(gen_loop, &cond0);
    let then_block = file.conditional_then_block(cond);

    let out_indexable = out.to_indexable_expr();
    let idx = file.make_index_expr(&out_indexable, &i_expr);
    let lhs = idx.to_expr();

    let zero_val = file
        .make_literal("bits[1]:0x0", &LiteralFormat::Hex)
        .unwrap();
    let one_val = file
        .make_literal("bits[1]:0x1", &LiteralFormat::Hex)
        .unwrap();
    let x_val = file.make_unsized_x_literal();

    file.block_add_continuous_assignment(then_block, &lhs, &zero_val);

    let else_if_block = file.conditional_add_else_if(cond, &cond1);
    file.block_add_continuous_assignment(else_if_block, &lhs, &one_val);

    let else_block = file.conditional_add_else(cond);
    file.block_add_continuous_assignment(else_block, &lhs, &x_val);

    let verilog = file.emit();
    let want = r#"module top;
  wire [2:0] out;
  for (genvar i = 0; i < 3; i = i + 1) begin : g
    if (i == 0) begin
      assign out[i] = 1'h0;
    end else if (i == 1) begin
      assign out[i] = 1'h1;
    end else begin
      assign out[i] = 'X;
    end
  end
endmodule
"#;
    assert_eq!(verilog, want);
}
#[test]
fn test_index_unpacked_array_parameter() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");

    // parameter P0[2] = '{'0, '1};
    let scalar = file.make_scalar_type();
    let p0_type = file.make_unpacked_array_type(scalar.clone(), &[2]);
    let p0_def = file.make_def("P0", DataKind::User, &p0_type);
    let tick0 = file.make_unsized_zero_literal();
    let tick1 = file.make_unsized_one_literal();
    let p0_rhs = file.make_array_assignment_pattern(&[&tick0, &tick1]);
    let p0_param = file.add_parameter_with_def(module, &p0_def, &p0_rhs);

    // wire w;
    let w = file.add_wire(module, "w", &scalar);

    // assign w = P0[0];
    let p0_indexable = p0_param.to_indexable_expr();
    let p0_0 = file.make_index(&p0_indexable, 0);
    let rhs = p0_0.to_expr();
    let assign = file.make_continuous_assignment(&w.to_expr(), &rhs);
    file.add_member_continuous_assignment(module, assign);

    let verilog = file.emit();
    let want = r#"module top;
  parameter P0[2] = '{'0, '1};
  wire w;
  assign w = P0[0];
endmodule
"#;
    assert_eq!(verilog, want);
}
