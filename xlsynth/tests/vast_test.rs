// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
use xlsynth::{
    ir_value::IrFormatPreference,
    vast::{DataKind, Expr, VastDataType, VastFile, VastFileType},
};

type VastBinOp = fn(&mut VastFile, &Expr, &Expr) -> Expr;

#[test]
fn test_vast() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let mut module = file.add_module("main");
    let input_type = file.make_bit_vector_type(32, false);
    let output_type = file.make_scalar_type();
    module.add_input("in", &input_type);
    module.add_output("out", &output_type);
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
    let mut module = file.add_module("my_module");
    let input_type = file.make_bit_vector_type(8, false);
    let output_type = file.make_bit_vector_type(4, false);
    let input = module.add_input("my_input", &input_type);
    let output = module.add_output("my_output", &output_type);
    let slice = file.make_slice(&input.to_indexable_expr(), 3, 0);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &slice.to_expr());
    module.add_member_continuous_assignment(assignment);
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

    let mut a_module = file.add_module("A");
    a_module.add_output("bus", &data_type);

    let mut b_module = file.add_module("B");
    let bus = b_module.add_wire("bus", &data_type);

    let param_value = file
        .make_literal("bits[32]:42", &IrFormatPreference::UnsignedDecimal)
        .unwrap();

    b_module.add_member_instantiation(file.make_instantiation(
        "A",
        "a_i",
        &["a_param"],
        &[&param_value],
        &["bus", "empty_thing"],
        &[Some(&bus.to_expr()), None],
    ));

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
    let mut module = file.add_module("main");
    let input_type = file.make_bit_vector_type(32, false);
    let output_type = file.make_scalar_type();
    module.add_input("in", &input_type);
    module.add_output("out", &output_type);
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
    let mut module = file.add_module("my_module");
    let wire = module.add_wire("bus", &file.make_bit_vector_type(128, false));
    let literal = file
        .make_literal(
            "bits[128]:0xFFEEDDCCBBAA99887766554433221100",
            &IrFormatPreference::Hex,
        )
        .unwrap();
    let assignment = file.make_continuous_assignment(&wire.to_expr(), &literal);
    module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("my_module");
    let my_struct = file.make_extern_package_type("mypack", "mystruct_t");
    let input_type = file.make_packed_array_type(my_struct, &[2, 3, 4]);
    module.add_input("my_input", &input_type);
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
    let mut module = file.add_module("my_module");
    let input_type = file.make_bit_vector_type(8, false);
    let output_type = file.make_bit_vector_type(16, false);
    let input = module.add_input("my_input", &input_type);
    let output = module.add_output("my_output", &output_type);
    let concat = file.make_concat(&[&input.to_expr(), &input.to_expr()]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("my_module");
    let u2 = file.make_bit_vector_type(2, false);
    let a_type = file.make_packed_array_type(u2, &[3, 5]);
    let b_type = file.make_bit_vector_type(2, false);
    let c_type = file.make_bit_vector_type(3, false);
    let a = module.add_wire("a", &a_type);
    let b = module.add_wire("b", &b_type);
    let c = module.add_wire("c", &c_type);

    // First assignment.
    {
        let a_1 = file.make_index(&a.to_indexable_expr(), 1);
        let a_2 = file.make_index(&a_1.to_indexable_expr(), 2);
        let a_lhs = file.make_slice(&a_2.to_indexable_expr(), 3, 4);
        let b_slice = file.make_slice(&b.to_indexable_expr(), 1, 0);
        let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &b_slice.to_expr());
        module.add_member_continuous_assignment(assignment);
    }

    // Second assignment.
    {
        let a_lhs = file.make_slice(&a.to_indexable_expr(), 3, 4);
        let c_slice = file.make_slice(&c.to_indexable_expr(), 2, 1);
        let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &c_slice.to_expr());
        module.add_member_continuous_assignment(assignment);
    }

    let verilog = file.emit();
    assert_eq!(verilog, want);
}

#[test]
fn test_index_then_add_constant_uses_indexable_to_expr() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let mut module = file.add_module("idx_add");

    // Create a 3-bit wire `x` and a 1-bit wire `y`.
    let x = module.add_wire("x", &file.make_bit_vector_type(3, false));
    let y = module.add_wire("y", &file.make_scalar_type());

    // Build (x[2]) using index -> indexable -> expr to exercise the new API.
    let idx = file.make_index(&x.to_indexable_expr(), 2);
    let idx_expr = idx.to_indexable_expr().to_expr();

    // Add a 1-bit constant to the indexed bit.
    let one = file
        .make_literal("bits[1]:1", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let sum = file.make_add(&idx_expr, &one);

    // Emit as a continuous assignment so it appears in the module body.
    let assign = file.make_continuous_assignment(&y.to_expr(), &sum);
    module.add_member_continuous_assignment(assign);

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
    let mut module = file.add_module("my_module");
    let element_type = file.make_bit_vector_type(8, false);
    let arr = module.add_wire("arr", &element_type);
    let hi = module.add_wire("hi", &file.make_bit_vector_type(4, false));
    let lo = module.add_wire("lo", &file.make_bit_vector_type(4, false));
    let idx = module.add_wire("idx", &file.make_bit_vector_type(3, false));
    let slice_out = module.add_wire("slice_out", &element_type);
    let index_out = module.add_wire("index_out", &file.make_scalar_type());

    let arr_indexable = arr.to_indexable_expr();
    let hi_expr = hi.to_expr();
    let lo_expr = lo.to_expr();
    let idx_expr = idx.to_expr();

    let slice = file.make_slice_expr(&arr_indexable, &hi_expr, &lo_expr);
    let index = file.make_index_expr(&arr_indexable, &idx_expr);

    let slice_assign = file.make_continuous_assignment(&slice_out.to_expr(), &slice.to_expr());
    module.add_member_continuous_assignment(slice_assign);
    let index_assign = file.make_continuous_assignment(&index_out.to_expr(), &index.to_expr());
    module.add_member_continuous_assignment(index_assign);

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
    let mut module = file.add_module("my_module");
    let input = module.add_input("my_input", &file.make_bit_vector_type(8, false));
    let output = module.add_output("my_output", &file.make_bit_vector_type(9, false));
    let input_indexable = input.to_indexable_expr();
    let index = file.make_index(&input_indexable, 0);
    let slice = file.make_slice(&input_indexable, 7, 0);
    let concat = file.make_concat(&[&index.to_expr(), &slice.to_expr()]);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("my_module");
    let input = module.add_input("my_input", &file.make_bit_vector_type(8, false));
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
    let output = module.add_output("my_output", &concat_type);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
    module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("my_module");
    let u8 = file.make_bit_vector_type(8, false);
    let u1 = file.make_bit_vector_type(1, false);
    let lhs = module.add_input("lhs", &u8);
    let rhs = module.add_input("rhs", &u8);
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
        let wire = module.add_wire(name, output_type);
        let rhs_expr = f(&mut file, &lhs.to_expr(), &rhs.to_expr());
        let assignment = file.make_continuous_assignment(&wire.to_expr(), &rhs_expr);
        module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("gen_nested");
    let scalar = file.make_scalar_type();

    let a = module.add_input("a", &scalar);
    let b = module.add_output("b", &scalar);

    // for (genvar i = 0; i < 2; ++i) begin: outer
    let zero = file.make_plain_literal(0, &IrFormatPreference::UnsignedDecimal);
    let two = file.make_plain_literal(2, &IrFormatPreference::UnsignedDecimal);
    let mut outer = module.add_generate_loop("i", &zero, &two, Some("outer"));

    //   for (genvar j = 1; j < 3; ++j) begin: inner
    let one = file.make_plain_literal(1, &IrFormatPreference::UnsignedDecimal);
    let three = file.make_plain_literal(3, &IrFormatPreference::UnsignedDecimal);
    let mut inner = outer.add_generate_loop("j", &one, &three, Some("inner"));

    //     assign b = a;
    inner.add_continuous_assignment(&b.to_expr(), &a.to_expr());

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
    let mut module = file.add_module("width_cast");
    let u16 = file.make_bit_vector_type(16, false);
    let u8 = file.make_bit_vector_type(8, false);

    let x = module.add_input("x", &u16);
    let y = module.add_output("y", &u8);

    let width8 = file.make_plain_literal(8, &IrFormatPreference::UnsignedDecimal);
    let cast = file.make_width_cast(&width8, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    module.add_member_continuous_assignment(assign);

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
    let mut module = file.add_module("gen_empty_blocks");

    let zero = file.make_plain_literal(0, &IrFormatPreference::UnsignedDecimal);
    let two = file.make_plain_literal(2, &IrFormatPreference::UnsignedDecimal);
    let mut gen = module.add_generate_loop("i", &zero, &two, Some("G"));

    let five = file.make_plain_literal(5, &IrFormatPreference::UnsignedDecimal);
    gen.add_localparam("LP", &five);

    gen.add_always_comb().unwrap();
    gen.add_always_ff(&[]).unwrap();

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
    let mut module = file.add_module("type_cast");
    let u8 = file.make_bit_vector_type(8, false);
    let x = module.add_input("x", &u8);
    let y = module.add_output("y", &u8);

    let user_t = file.make_extern_package_type("", "my_type_t");
    let cast = file.make_type_cast(&user_t, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    module.add_member_continuous_assignment(assign);

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
    let mut module = file.add_module("with_inout");
    let scalar = file.make_scalar_type();
    module.add_inout("io", &scalar);
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
    let mut module = file.add_module("my_module");
    let selector = module.add_input("selector", &file.make_bit_vector_type(8, false));
    let on_true = module.add_input("on_true", &file.make_bit_vector_type(8, false));
    let on_false = module.add_input("on_false", &file.make_bit_vector_type(8, false));
    let ternary = file.make_ternary(&selector.to_expr(), &on_true.to_expr(), &on_false.to_expr());
    let output = module.add_output("my_output", &file.make_bit_vector_type(8, false));
    let assignment = file.make_continuous_assignment(&output.to_expr(), &ternary);
    module.add_member_continuous_assignment(assignment);
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
    let mut module = file.add_module("my_module");
    let a = module.add_input("a", &file.make_scalar_type());
    let b = module.add_input("b", &file.make_scalar_type());
    let w = module.add_wire("w", &file.make_bit_vector_type(6, false));
    let expr = file.make_replicated_concat_i64(3, &[&a.to_expr(), &b.to_expr()]);
    module.add_member_continuous_assignment(file.make_continuous_assignment(&w.to_expr(), &expr));
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
    let mut module = file.add_module("my_module");
    let a = module.add_input("a", &file.make_scalar_type());
    let b = module.add_input("b", &file.make_scalar_type());
    let w = module.add_wire("w", &file.make_bit_vector_type(6, false));
    let rep = file
        .make_literal("bits[32]:3", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let expr = file.make_replicated_concat(&rep, &[&a.to_expr(), &b.to_expr()]);
    module.add_member_continuous_assignment(file.make_continuous_assignment(&w.to_expr(), &expr));
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
    let mut module = file.add_module("m");
    let int_t = file.make_integer_type(true);
    module.add_input("i", &int_t);
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
    let mut module = file.add_module("P");
    let lit = file
        .make_literal("bits[32]:4", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let pref = module.add_parameter("N", &lit);
    let out = module.add_output("o", &file.make_bit_vector_type(32, false));
    module.add_member_continuous_assignment(
        file.make_continuous_assignment(&out.to_expr(), &pref.to_expr()),
    );
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
    let mut module = file.add_module("P2");
    let int_t = file.make_integer_type(true);
    let def = file.make_def("N2", DataKind::Integer, &int_t);
    let lit = file
        .make_literal("bits[32]:7", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let _pref = module.add_parameter_with_def(&def, &lit);
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
    let mut module = file.add_module("LM");

    // localparam int Foo = 42;
    let forty_two = file.make_plain_literal(42, &IrFormatPreference::UnsignedDecimal);
    module.add_int_localparam("Foo", &forty_two);

    // localparam Bar = 100;
    let one_hundred = file.make_plain_literal(100, &IrFormatPreference::UnsignedDecimal);
    module.add_localparam("Bar", &one_hundred);

    // localparam Qux = 'h10000;
    let qux = file
        .make_literal("bits[32]:0x10000", &IrFormatPreference::PlainHex)
        .unwrap();
    module.add_localparam("Bar", &qux);

    // localparam logic [7:0] Baz = 8'h44;
    let logic8 = file.make_bit_vector_type(8, false);
    let baz_def = file.make_def("Baz", DataKind::Logic, &logic8);
    let hex_44 = file
        .make_literal("bits[8]:0x44", &IrFormatPreference::Hex)
        .unwrap();
    module.add_typed_localparam(&baz_def, &hex_44);

    // localparam logic [7:0] Zero = '0;
    let zero_def = file.make_def("Zero", DataKind::Logic, &logic8);
    let unsized_zero = file.make_unsized_zero_literal();
    module.add_typed_localparam(&zero_def, &unsized_zero);

    // localparam logic [7:0] Ones = '1;
    let ones_def = file.make_def("Ones", DataKind::Logic, &logic8);
    let unsized_one = file.make_unsized_one_literal();
    module.add_typed_localparam(&ones_def, &unsized_one);

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
fn test_inline_and_blank_members() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let mut module = file.add_module("M");
    module.add_member_inline_statement(file.make_inline_verilog_statement("/* first */"));
    module.add_member_blank_line(file.make_blank_line());
    module.add_member_inline_statement(file.make_inline_verilog_statement("/* second */"));
    let verilog = file.emit();
    let want = r#"module M;
  /* first */

  /* second */
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn test_sequential_logic_system_verilog() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut module = file.add_module("test_module");

    let scalar_type = file.make_scalar_type();

    let clk = module.add_input("clk", &scalar_type);
    let pred = module.add_input("pred", &scalar_type);
    let x = module.add_input("x", &scalar_type);
    module.add_output("out", &scalar_type);

    let p0_pred_reg = module.add_reg("p0_pred", &scalar_type).unwrap();
    let p0_x_reg = module.add_reg("p0_x", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk.to_expr());

    let always_block = module.add_always_ff(&[&posedge_clk]).unwrap();

    let mut stmt_block = always_block.get_statement_block();

    stmt_block.add_nonblocking_assignment(&p0_pred_reg.to_expr(), &pred.to_expr());
    stmt_block.add_comment_text("capture pred");
    stmt_block.add_blank_line();
    stmt_block.add_inline_text("/* combo capture */");
    stmt_block.add_nonblocking_assignment(&p0_x_reg.to_expr(), &x.to_expr());

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
    let mut module = file.add_module("test_module");
    let scalar_type = file.make_scalar_type();
    let _clk = module.add_input("clk", &scalar_type);
    let x = module.add_input("x", &scalar_type);
    let r = module.add_reg("r", &scalar_type).unwrap();

    let always_block = module.add_always_comb().unwrap();
    let mut sb = always_block.get_statement_block();
    sb.add_blocking_assignment(&r.to_expr(), &x.to_expr());

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
    let mut module = file.add_module("test_module");
    let scalar_type = file.make_scalar_type();
    let clk = module.add_input("clk", &scalar_type);
    let x = module.add_input("x", &scalar_type);
    let r = module.add_reg("r", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always_block = module.add_always_at(&[&posedge_clk]).unwrap();
    let mut sb = always_block.get_statement_block();
    sb.add_blocking_assignment(&r.to_expr(), &x.to_expr());

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
    let mut file = xlsynth::vast::VastFile::new(xlsynth::vast::VastFileType::SystemVerilog);
    let mut module = file.add_module("M");
    let bit = file.make_scalar_type();
    let clk = module.add_input("clk", &bit);
    let a = module.add_input("a", &bit);
    let b = module.add_input("b", &bit);
    let r = module.add_reg("r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = module.add_always_ff(&[&posedge_clk]).unwrap();
    let mut sb = always.get_statement_block();
    let cond = sb.add_cond(&a.to_expr());
    let mut then_block = cond.then_block();
    then_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
    let mut else_if_block = cond.add_else_if(&b.to_expr());
    else_if_block.add_nonblocking_assignment(&r.to_expr(), &b.to_expr());
    let mut else_block = cond.add_else();
    else_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
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
    let mut file = xlsynth::vast::VastFile::new(xlsynth::vast::VastFileType::Verilog);
    let mut module = file.add_module("M");
    let bit = file.make_scalar_type();
    let clk = module.add_input("clk", &bit);
    let a = module.add_input("a", &bit);
    let b = module.add_input("b", &bit);
    let r = module.add_reg("r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = module.add_always_at(&[&posedge_clk]).unwrap();
    let mut sb = always.get_statement_block();
    let cond = sb.add_cond(&a.to_expr());
    let mut then_block = cond.then_block();
    then_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
    let mut else_if_block = cond.add_else_if(&b.to_expr());
    else_if_block.add_nonblocking_assignment(&r.to_expr(), &b.to_expr());
    let mut else_block = cond.add_else();
    else_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
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
    let mut file = xlsynth::vast::VastFile::new(xlsynth::vast::VastFileType::SystemVerilog);
    let mut module = file.add_module("C");
    let bit = file.make_scalar_type();
    let clk = module.add_logic_input("clk", &bit);
    let sel = module.add_logic_input("sel", &bit);
    let a = module.add_logic_input("a", &bit);
    let b = module.add_logic_input("b", &bit);
    let r = module.add_logic("r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = module.add_always_ff(&[&posedge_clk]).unwrap();
    let mut sb = always.get_statement_block();
    let case_stmt = sb.add_case(&sel.to_expr());
    let mut item_a = case_stmt.add_item(&a.to_expr());
    item_a.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
    let mut item_b = case_stmt.add_item(&b.to_expr());
    item_b.add_nonblocking_assignment(&r.to_expr(), &b.to_expr());
    let mut default_block = case_stmt.add_default();
    default_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
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
    let mut module = file.add_module("C");
    let bit = file.make_scalar_type();
    let clk = module.add_input("clk", &bit);
    let sel = module.add_input("sel", &bit);
    let a = module.add_input("a", &bit);
    let b = module.add_input("b", &bit);
    let r = module.add_reg("r", &bit).unwrap();
    let posedge_clk = file.make_pos_edge(&clk.to_expr());
    let always = module.add_always_ff(&[&posedge_clk]).unwrap();
    let mut sb = always.get_statement_block();
    let case_stmt = sb.add_case(&sel.to_expr());
    let mut item_a = case_stmt.add_item(&a.to_expr());
    item_a.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
    let mut item_b = case_stmt.add_item(&b.to_expr());
    item_b.add_nonblocking_assignment(&r.to_expr(), &b.to_expr());
    let mut default_block = case_stmt.add_default();
    default_block.add_nonblocking_assignment(&r.to_expr(), &a.to_expr());
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
    let mut module = file.add_module("P");
    let lit = file
        .make_literal("bits[32]:4", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let n = module.add_parameter("N", &lit);
    let dt = file.make_bit_vector_type_expr(&n.to_expr(), false);
    module.add_output("o", &dt);
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
    let mut module = file.add_module("M");
    let lit = file
        .make_literal("bits[32]:5", &IrFormatPreference::UnsignedDecimal)
        .unwrap();
    let dt = file.make_bit_vector_type_expr(&lit, false);
    module.add_wire("w", &dt);
    let verilog = file.emit();
    let want = r#"module M;
  wire [4:0] w;
endmodule
"#;
    assert_eq!(verilog, want);
}

#[test]
fn module_with_parameters() {
    let mut file = xlsynth::vast::VastFile::new(xlsynth::vast::VastFileType::SystemVerilog);
    let mut module = file.add_module("C");
    let bit = file.make_scalar_type();
    let n = module.add_parameter_port(
        "N",
        &file.make_plain_literal(42, &IrFormatPreference::UnsignedDecimal),
    );
    module.add_typed_parameter_port(
        "Foo",
        &file.make_bit_vector_type(16, false),
        &file
            .make_literal("bits[16]:5", &IrFormatPreference::UnsignedDecimal)
            .unwrap(),
    );
    module.add_logic_input("clk", &bit);
    module.add_logic_input("a", &file.make_bit_vector_type_expr(&n.to_expr(), false));
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
    let mut module = file.add_module("type_cast2");
    let u8 = file.make_bit_vector_type(8, false);
    let x = module.add_input("x", &u8);
    let y = module.add_output("y", &u8);

    // Use an unqualified user type via extern_type.
    let user_t = file.make_extern_type("my_type_t");
    let cast = file.make_type_cast(&user_t, &x.to_expr());
    let assign = file.make_continuous_assignment(&y.to_expr(), &cast);
    module.add_member_continuous_assignment(assign);

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
    let mut module = file.add_module("macro_mod");

    // Add a simple macro statement: `MY_MACRO;
    let mref1 = file.make_macro_ref("MY_MACRO1");
    let mstmt1 = file.make_macro_statement(&mref1, true);
    let mref2 = file.make_macro_ref("MY_MACRO2");
    let mstmt2 = file.make_macro_statement(&mref2, false);
    module.add_member_macro_statement(mstmt1);
    module.add_member_macro_statement(mstmt2);

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
    let mut module = file.add_module("gen_with_macros");

    // for (genvar i = 0; i < 1; ++i) begin : G
    let zero = file.make_plain_literal(0, &IrFormatPreference::UnsignedDecimal);
    let one = file.make_plain_literal(1, &IrFormatPreference::UnsignedDecimal);
    let mut gen = module.add_generate_loop("i", &zero, &one, Some("G"));

    // Comment, blank line, and macro statements inside the loop.
    let comment = file.make_comment("inside");
    gen.add_comment(&comment);
    gen.add_blank_line();
    let mref = file.make_macro_ref("DO_SOMETHING");
    let mstmt = file.make_macro_statement(&mref, true);
    gen.add_macro_statement(&mstmt);
    // Macro with arguments.
    let three = file.make_plain_literal(3, &IrFormatPreference::UnsignedDecimal);
    let mref_args = file.make_macro_ref_with_args("DO_THING", &[&three]);
    let mstmt_args = file.make_macro_statement(&mref_args, false);
    gen.add_macro_statement(&mstmt_args);

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
    let three = file.make_plain_literal(3, &IrFormatPreference::UnsignedDecimal);
    let s = three.emit();
    assert_eq!(s, "3");
}

#[test]
fn test_file_level_comment_and_blank_line() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.add_comment_text("top-level comment");
    let blank = file.make_blank_line();
    file.add_blank_line(blank);
    let mut module = file.add_module("M");
    let scalar = file.make_scalar_type();
    module.add_wire("w", &scalar);
    let verilog = file.emit();
    let want = r#"// top-level comment

module M;
  wire w;
endmodule
"#;
    assert_eq!(verilog, want);
}
