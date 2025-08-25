// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use xlsynth::ir_value::*;
    use xlsynth::vast::*;

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
        let want = "module main(\n  input wire [31:0] in,\n  output wire out\n);\n\nendmodule\n";
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
        let want = "module my_module(\n  input wire [7:0] my_input,\n  output wire [3:0] my_output\n);\n  assign my_output = my_input[3:0];\nendmodule\n";
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
        let want = "module A(\n  output wire [7:0] bus\n);\n\nendmodule\nmodule B;\n  wire [7:0] bus;\n  A #(\n    .a_param(32'd42)\n  ) a_i (\n    .bus(bus),\n    .empty_thing()\n  );\nendmodule\n";
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
        let want = "module main(\n  input wire [31:0] in,\n  output wire out\n);\n\nendmodule\n";
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
        let want = "module my_module;\n  wire [127:0] bus;\n  assign bus = 128'hffee_ddcc_bbaa_9988_7766_5544_3322_1100;\nendmodule\n";
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
        let want = "module my_module(\n  input mypack::mystruct_t [1:0][2:0][3:0] my_input\n);\n\nendmodule\n";
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
        let want = "module my_module(\n  input wire [7:0] my_input,\n  output wire [15:0] my_output\n);\n  assign my_output = {my_input, my_input};\nendmodule\n";
        assert_eq!(verilog, want);
    }

    /// Tests that we can reference a slice of a multidimensional packed array
    /// on the LHS or RHS of an assign statement.
    #[test]
    fn test_slice_on_both_sides_of_assignment() {
        let want = "module my_module;\n  wire [1:0][2:0][4:0] a;\n  wire [1:0] b;\n  wire [2:0] c;\n  assign a[1][2][3:4] = b[1:0];\n  assign a[3:4] = c[2:1];\nendmodule\n";

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
        let want = "module my_module(\n  input wire [7:0] my_input,\n  output wire [8:0] my_output\n);\n  assign my_output = {my_input[0], my_input[7:0]};\nendmodule\n";
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
        let want = "module my_module(\n  input wire [7:0] my_input,\n  output wire [27:0] my_output\n);\n  assign my_output = {~my_input, -my_input, ~my_input, !my_input, &my_input, |my_input, ^my_input};\nendmodule\n";
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
    fn test_ternary() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let selector = module.add_input("selector", &file.make_bit_vector_type(8, false));
        let on_true = module.add_input("on_true", &file.make_bit_vector_type(8, false));
        let on_false = module.add_input("on_false", &file.make_bit_vector_type(8, false));
        let ternary =
            file.make_ternary(&selector.to_expr(), &on_true.to_expr(), &on_false.to_expr());
        let output = module.add_output("my_output", &file.make_bit_vector_type(8, false));
        let assignment = file.make_continuous_assignment(&output.to_expr(), &ternary);
        module.add_member_continuous_assignment(assignment);
        let verilog = file.emit();
        let want = "module my_module(\n  input wire [7:0] selector,\n  input wire [7:0] on_true,\n  input wire [7:0] on_false,\n  output wire [7:0] my_output\n);\n  assign my_output = selector ? on_true : on_false;\nendmodule\n";
        assert_eq!(verilog, want);
    }
}

fn run_sequential_logic_test(use_system_verilog: bool) {
    use xlsynth::vast::*;
    let mut file = VastFile::new(if use_system_verilog {
        VastFileType::SystemVerilog
    } else {
        VastFileType::Verilog
    });
    let mut module = file.add_module("test_module");

    let scalar_type = file.make_scalar_type();

    let clk = module.add_input("clk", &scalar_type);
    let pred = module.add_input("pred", &scalar_type);
    let x = module.add_input("x", &scalar_type);
    module.add_output("out", &scalar_type);

    let p0_pred_reg = module.add_reg("p0_pred", &scalar_type).unwrap();
    let p0_x_reg = module.add_reg("p0_x", &scalar_type).unwrap();

    let posedge_clk = file.make_pos_edge(&clk.to_expr());

    let always_block = if use_system_verilog {
        module.add_always_ff(&[&posedge_clk]).unwrap()
    } else {
        module.add_always_at(&[&posedge_clk]).unwrap()
    };

    let mut stmt_block = always_block.get_statement_block();

    stmt_block.add_nonblocking_assignment(&p0_pred_reg.to_expr(), &pred.to_expr());
    stmt_block.add_nonblocking_assignment(&p0_x_reg.to_expr(), &x.to_expr());

    let verilog = file.emit();

    let want = if use_system_verilog {
        r#"module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always_ff @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end
endmodule
"#
    } else {
        r#"module test_module(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  reg p0_pred;
  reg p0_x;
  always @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end
endmodule
"#
    };

    assert_eq!(verilog, want);
}

#[test]
fn test_sequential_logic_system_verilog() {
    run_sequential_logic_test(true);
}

#[test]
fn test_sequential_logic_verilog() {
    run_sequential_logic_test(false);
}
