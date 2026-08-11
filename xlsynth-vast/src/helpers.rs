// SPDX-License-Identifier: Apache-2.0

//! Reusable AST-building helpers that do not require code-generation options.

use crate::model::{ExprData, TypeData};
use crate::{Expr, GenerateLoop, IndexableExpr, LiteralFormat, VastError, VastFile, VastModule};

/// Describes the reset signal and polarity used by a group of registers.
#[derive(Clone, Copy, Debug)]
pub struct Reset {
    pub signal: Expr,
    pub active_low: bool,
}

/// Describes one register, its next value, and optional reset/enable signals.
#[derive(Clone, Copy, Debug)]
pub struct RegisterDefinition {
    pub reg: Expr,
    pub next: Expr,
    pub reset_value: Option<Expr>,
    pub enable: Option<Expr>,
}

/// Identifies the module or generate loop containing a register group.
#[derive(Clone, Copy, Debug)]
pub enum RegisterScope {
    Module(VastModule),
    GenerateLoop(GenerateLoop),
}

/// Builds one clocked block for registers with optional resets and enables.
pub fn add_registers(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    scope: RegisterScope,
    file: &mut VastFile,
) -> Result<(), VastError> {
    if registers.is_empty() {
        return Ok(());
    }

    let any_with_reset = registers
        .iter()
        .any(|register| register.reset_value.is_some());
    if any_with_reset && reset.is_none() {
        return Err(VastError(
            "reset signal is required when a register has a reset value".into(),
        ));
    }

    file.check(clk.0);
    match scope {
        RegisterScope::Module(module) => {
            file.check(module.0);
        }
        RegisterScope::GenerateLoop(generate_loop) => {
            file.check(generate_loop.0);
        }
    }
    if let Some(reset) = reset {
        file.check(reset.signal.0);
    }
    for register in registers {
        file.check(register.reg.0);
        file.check(register.next.0);
        if let Some(reset_value) = register.reset_value {
            file.check(reset_value.0);
        }
        if let Some(enable) = register.enable {
            file.check(enable.0);
        }
    }

    let posedge_clk = file.make_pos_edge(clk);
    let always = match scope {
        RegisterScope::Module(module) => file.add_always_ff(module, &[&posedge_clk])?,
        RegisterScope::GenerateLoop(generate_loop) => {
            file.generate_add_always_ff(generate_loop, &[&posedge_clk])?
        }
    };
    let always_ff = file.statement_block(always);

    let mut assigned_values = Vec::with_capacity(registers.len());
    for register in registers {
        let value = match register.enable {
            Some(enable) => file.make_ternary(&enable, &register.next, &register.reg),
            None => register.next,
        };
        assigned_values.push(value);
    }

    for (index, register) in registers.iter().enumerate() {
        if register.reset_value.is_none() {
            file.block_add_nonblocking_assignment(
                always_ff,
                &register.reg,
                &assigned_values[index],
            );
        }
    }

    if any_with_reset {
        let reset = reset.expect("reset must be provided when registers have reset values");
        let condition = if reset.active_low {
            file.make_logical_not(&reset.signal)
        } else {
            reset.signal
        };
        let conditional = file.block_add_cond(always_ff, &condition);
        let reset_block = file.conditional_then_block(conditional);
        let next_value_block = file.conditional_add_else(conditional);

        for (index, register) in registers.iter().enumerate() {
            if let Some(reset_value) = register.reset_value {
                file.block_add_nonblocking_assignment(reset_block, &register.reg, &reset_value);
                file.block_add_nonblocking_assignment(
                    next_value_block,
                    &register.reg,
                    &assigned_values[index],
                );
            }
        }
    }

    Ok(())
}

/// Folds expressions from left to right, returning `identity` for no inputs.
fn reduce_with<F>(inputs: &[Expr], identity: Expr, mut combine: F, file: &mut VastFile) -> Expr
where
    F: FnMut(&mut VastFile, &Expr, &Expr) -> Expr,
{
    inputs
        .iter()
        .copied()
        .reduce(|accumulated, input| combine(file, &accumulated, &input))
        .unwrap_or(identity)
}

/// Builds a logical-OR reduction, optionally negating its complete result.
pub fn logical_or_reduce(
    inputs: &[Expr],
    invert: bool,
    file: &mut VastFile,
) -> Result<Expr, VastError> {
    for input in inputs {
        file.check(input.0);
    }

    if inputs.is_empty() && invert {
        return file.make_literal("bits[1]:1", &LiteralFormat::Hex);
    }

    let identity = file.make_literal("bits[1]:0", &LiteralFormat::Hex)?;
    let reduced = reduce_with(inputs, identity, VastFile::make_logical_or, file);
    if invert {
        Ok(file.make_logical_not(&reduced))
    } else {
        Ok(reduced)
    }
}

/// Builds a logical-AND reduction, optionally negating its complete result.
pub fn logical_and_reduce(
    inputs: &[Expr],
    invert: bool,
    file: &mut VastFile,
) -> Result<Expr, VastError> {
    for input in inputs {
        file.check(input.0);
    }

    if inputs.is_empty() && invert {
        return file.make_literal("bits[1]:0", &LiteralFormat::Hex);
    }

    let identity = file.make_literal("bits[1]:1", &LiteralFormat::Hex)?;
    let reduced = reduce_with(inputs, identity, VastFile::make_logical_and, file);
    if invert {
        Ok(file.make_logical_not(&reduced))
    } else {
        Ok(reduced)
    }
}

/// Builds a bitwise-OR reduction, using an unsized zero for no inputs.
pub fn bitwise_or_reduce(inputs: &[Expr], file: &mut VastFile) -> Result<Expr, VastError> {
    for input in inputs {
        file.check(input.0);
    }

    let identity = file.make_unsized_zero_literal();
    Ok(reduce_with(
        inputs,
        identity,
        VastFile::make_bitwise_or,
        file,
    ))
}

/// Returns known packed dimensions, accounting for already-indexed arrays.
fn packed_array_dimensions(file: &VastFile, expression: Expr) -> Option<Vec<i64>> {
    match &file.ast.expressions[file.check(expression.0)] {
        ExprData::Name {
            data_type: Some(data_type),
            ..
        }
        | ExprData::TypeCast { data_type, .. } => {
            let mut dimensions = Vec::new();
            let mut current = *data_type;
            while let TypeData::PackedArray {
                element,
                dimensions: nested_dimensions,
            } = &file.ast.data_types[file.check(current.0)]
            {
                dimensions.extend_from_slice(nested_dimensions);
                current = *element;
            }
            Some(dimensions)
        }
        ExprData::Index { subject, .. } => {
            let mut dimensions = packed_array_dimensions(file, *subject)?;
            if !dimensions.is_empty() {
                dimensions.remove(0);
            }
            Some(dimensions)
        }
        _ => None,
    }
}

/// Collects an array's indexed expressions in major-dimension-first order.
fn gather_elements(
    expr: &IndexableExpr,
    dimensions: &[i64],
    elements: &mut Vec<Expr>,
    file: &mut VastFile,
) {
    if dimensions.is_empty() {
        elements.push(expr.to_expr());
        return;
    }

    for index in 0..dimensions[0] {
        let indexed = file.make_index(expr, index).to_indexable_expr();
        gather_elements(&indexed, &dimensions[1..], elements, file);
    }
}

/// Builds the bitwise OR of every element in a packed multidimensional array.
pub fn bitwise_or_reduce_array_elements(
    expr: &IndexableExpr,
    dimensions: &[i64],
    file: &mut VastFile,
) -> Result<Expr, VastError> {
    file.check(expr.0);
    if dimensions.is_empty() {
        return Err(VastError(
            "packed-array reduction requires at least one dimension".into(),
        ));
    }
    if dimensions.iter().any(|dimension| *dimension <= 0) {
        return Err(VastError(
            "packed-array dimensions must be greater than zero".into(),
        ));
    }

    if let Some(expected_dimensions) = packed_array_dimensions(file, expr.to_expr())
        && expected_dimensions != dimensions
    {
        return Err(VastError(format!(
            "packed-array dimensions do not match expression type: expected \
             {expected_dimensions:?}, got {dimensions:?}"
        )));
    }

    let element_count = dimensions.iter().try_fold(1usize, |count, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| VastError("packed-array element count exceeds usize".into()))?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| VastError("packed-array element count exceeds usize".into()))
    })?;
    let mut elements = Vec::new();
    elements
        .try_reserve(element_count)
        .map_err(|_| VastError("packed-array elements cannot be allocated".into()))?;
    gather_elements(expr, dimensions, &mut elements, file);
    bitwise_or_reduce(&elements, file)
}

#[cfg(test)]
mod tests {
    use crate::{LiteralFormat, VastFile, VastFileType};

    use super::*;

    #[test]
    fn test_logical_or_reduce_various_arity() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("lor");

        // Inputs: scalar logic signals
        let scalar = file.make_scalar_type();
        let a = file.add_input(module, "a", &scalar);
        let b = file.add_input(module, "b", &scalar);
        let c = file.add_input(module, "c", &scalar);

        // Outputs
        let o0 = file.add_output(module, "o0", &scalar);
        let o1 = file.add_output(module, "o1", &scalar);
        let o2 = file.add_output(module, "o2", &scalar);
        let o3 = file.add_output(module, "o3", &scalar);

        // 0 inputs -> 0
        let e0 = logical_or_reduce(&[], false, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o0.to_expr(), &e0);
        file.add_member_continuous_assignment(module, assignment);

        // 1 input -> a
        let e1 = logical_or_reduce(&[a.to_expr()], false, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o1.to_expr(), &e1);
        file.add_member_continuous_assignment(module, assignment);

        // 2 inputs -> a || b
        let e2 = logical_or_reduce(&[a.to_expr(), b.to_expr()], false, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o2.to_expr(), &e2);
        file.add_member_continuous_assignment(module, assignment);

        // 3 inputs -> a || b || c
        let e3 =
            logical_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], false, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o3.to_expr(), &e3);
        file.add_member_continuous_assignment(module, assignment);

        let verilog = file.emit();
        let want = r#"module lor(
  input wire a,
  input wire b,
  input wire c,
  output wire o0,
  output wire o1,
  output wire o2,
  output wire o3
);
  assign o0 = 1'h0;
  assign o1 = a;
  assign o2 = a || b;
  assign o3 = a || b || c;
endmodule
"#;
        assert_eq!(verilog, want);
    }

    #[test]
    fn test_add_registers_with_reset_and_enable() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("regs_rst_en");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = file.add_input(module, "clk", &bit1);
        let clk_expr = clk.to_expr();
        let rst = file.add_input(module, "rst", &bit1);
        let en = file.add_input(module, "en", &bit1);
        let r = file.add_logic(module, "r", &u8).expect("add_logic r");
        let r_next = file
            .add_logic(module, "r_next", &u8)
            .expect("add_logic r_next");
        let r2 = file.add_logic(module, "r2", &u8).expect("add_logic r2");
        let r2_next = file
            .add_logic(module, "r2_next", &u8)
            .expect("add_logic r2_next");

        let reset_val = file
            .make_literal("bits[8]:0xAA", &LiteralFormat::Hex)
            .expect("literal ok");
        let reset_val_r2 = file
            .make_literal("bits[8]:0x55", &LiteralFormat::Hex)
            .expect("literal ok");

        let regs = [
            RegisterDefinition {
                reg: r.to_expr(),
                next: r_next.to_expr(),
                reset_value: Some(reset_val),
                enable: Some(en.to_expr()),
            },
            RegisterDefinition {
                reg: r2.to_expr(),
                next: r2_next.to_expr(),
                reset_value: Some(reset_val_r2),
                enable: None,
            },
        ];

        add_registers(
            &clk_expr,
            Some(Reset {
                signal: rst.to_expr(),
                active_low: true,
            }),
            &regs,
            RegisterScope::Module(module),
            &mut file,
        )
        .expect("add_registers ok");

        let sv = file.emit();
        let want = r#"module regs_rst_en(
  input wire clk,
  input wire rst,
  input wire en
);
  logic [7:0] r;
  logic [7:0] r_next;
  logic [7:0] r2;
  logic [7:0] r2_next;
  always_ff @ (posedge clk) begin
    if (!rst) begin
      r <= 8'haa;
      r2 <= 8'h55;
    end else begin
      r <= en ? r_next : r;
      r2 <= r2_next;
    end
  end
endmodule
"#;
        assert_eq!(sv, want);
    }

    #[test]
    fn test_add_registers_no_reset_path() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("regs_no_rst");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = file.add_input(module, "clk", &bit1);
        let clk_expr = clk.to_expr();
        let en = file.add_input(module, "en", &bit1);
        let r = file.add_logic(module, "r", &u8).expect("add_logic r");
        let r_next = file
            .add_logic(module, "r_next", &u8)
            .expect("add_logic r_next");
        let r2 = file.add_logic(module, "r2", &u8).expect("add_logic r2");
        let r2_next = file
            .add_logic(module, "r2_next", &u8)
            .expect("add_logic r2_next");

        let regs = [
            RegisterDefinition {
                reg: r.to_expr(),
                next: r_next.to_expr(),
                reset_value: None,
                enable: Some(en.to_expr()),
            },
            RegisterDefinition {
                reg: r2.to_expr(),
                next: r2_next.to_expr(),
                reset_value: None,
                enable: None,
            },
        ];

        add_registers(
            &clk_expr,
            None,
            &regs,
            RegisterScope::Module(module),
            &mut file,
        )
        .expect("add_registers ok");

        let sv = file.emit();
        let want = r#"module regs_no_rst(
  input wire clk,
  input wire en
);
  logic [7:0] r;
  logic [7:0] r_next;
  logic [7:0] r2;
  logic [7:0] r2_next;
  always_ff @ (posedge clk) begin
    r <= en ? r_next : r;
    r2 <= r2_next;
  end
endmodule
"#;
        assert_eq!(sv, want);
    }

    #[test]
    fn test_add_registers_mixed_resets() {
        // Mixed resets are now supported; verify generated output
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("regs_mixed");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = file.add_input(module, "clk", &bit1);
        let clk_expr = clk.to_expr();
        let rst = file.add_input(module, "rst", &bit1);
        let en = file.add_input(module, "en", &bit1);
        let r1 = file.add_logic(module, "r1", &u8).expect("r1");
        let n1 = file.add_logic(module, "n1", &u8).expect("n1");
        let r2 = file.add_logic(module, "r2", &u8).expect("r2");
        let n2 = file.add_logic(module, "n2", &u8).expect("n2");

        let reset_val = file
            .make_literal("bits[8]:0xAA", &LiteralFormat::Hex)
            .expect("literal ok");

        let regs = [
            RegisterDefinition {
                reg: r1.to_expr(),
                next: n1.to_expr(),
                reset_value: Some(reset_val),
                enable: Some(en.to_expr()),
            },
            RegisterDefinition {
                reg: r2.to_expr(),
                next: n2.to_expr(),
                reset_value: None,
                enable: None,
            },
        ];

        add_registers(
            &clk_expr,
            Some(Reset {
                signal: rst.to_expr(),
                active_low: true,
            }),
            &regs,
            RegisterScope::Module(module),
            &mut file,
        )
        .expect("add_registers ok");

        let sv = file.emit();
        let want = r#"module regs_mixed(
  input wire clk,
  input wire rst,
  input wire en
);
  logic [7:0] r1;
  logic [7:0] n1;
  logic [7:0] r2;
  logic [7:0] n2;
  always_ff @ (posedge clk) begin
    r2 <= n2;
    if (!rst) begin
      r1 <= 8'haa;
    end else begin
      r1 <= en ? n1 : r1;
    end
  end
endmodule
"#;
        assert_eq!(sv, want);
    }

    #[test]
    fn test_add_registers_missing_reset_returns_error_without_mutating_module() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("regs_missing_rst");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = file.add_input(module, "clk", &bit1);
        let clk_expr = clk.to_expr();
        let r = file.add_logic(module, "r", &u8).expect("r");
        let n = file.add_logic(module, "n", &u8).expect("n");
        let reset_val = file
            .make_literal("bits[8]:0xAA", &LiteralFormat::Hex)
            .expect("literal ok");

        let regs = [RegisterDefinition {
            reg: r.to_expr(),
            next: n.to_expr(),
            reset_value: Some(reset_val),
            enable: None,
        }];
        let original = file.emit();

        let error = add_registers(
            &clk_expr,
            None,
            &regs,
            RegisterScope::Module(module),
            &mut file,
        )
        .expect_err("registers with reset values require a reset signal");

        assert_eq!(
            error.to_string(),
            "reset signal is required when a register has a reset value"
        );
        assert_eq!(file.emit(), original);
    }
    #[test]
    fn test_logical_or_reduce_inverted() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("lori");

        // Inputs: scalar logic signals
        let scalar = file.make_scalar_type();
        let a = file.add_input(module, "a", &scalar);
        let b = file.add_input(module, "b", &scalar);
        let c = file.add_input(module, "c", &scalar);

        // Outputs
        let o0 = file.add_output(module, "o0", &scalar);
        let o1 = file.add_output(module, "o1", &scalar);
        let o2 = file.add_output(module, "o2", &scalar);
        let o3 = file.add_output(module, "o3", &scalar);

        // 0 inputs -> 1
        let e0 = logical_or_reduce(&[], true, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o0.to_expr(), &e0);
        file.add_member_continuous_assignment(module, assignment);

        // 1 input -> !a
        let e1 = logical_or_reduce(&[a.to_expr()], true, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o1.to_expr(), &e1);
        file.add_member_continuous_assignment(module, assignment);

        // 2 inputs -> !(a || b)
        let e2 = logical_or_reduce(&[a.to_expr(), b.to_expr()], true, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o2.to_expr(), &e2);
        file.add_member_continuous_assignment(module, assignment);

        // 3 inputs -> !(a || b || c)
        let e3 =
            logical_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], true, &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o3.to_expr(), &e3);
        file.add_member_continuous_assignment(module, assignment);

        let verilog = file.emit();
        let want = r#"module lori(
  input wire a,
  input wire b,
  input wire c,
  output wire o0,
  output wire o1,
  output wire o2,
  output wire o3
);
  assign o0 = 1'h1;
  assign o1 = !a;
  assign o2 = !(a || b);
  assign o3 = !(a || b || c);
endmodule
"#;
        assert_eq!(verilog, want);
    }

    #[test]
    fn test_bitwise_or_reduce_various_arity() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("bor");

        // Inputs: 8-bit vectors
        let u8 = file.make_bit_vector_type(8, false);
        let a = file.add_input(module, "a", &u8);
        let b = file.add_input(module, "b", &u8);
        let c = file.add_input(module, "c", &u8);

        // Outputs
        let o0 = file.add_output(module, "o0", &u8);
        let o1 = file.add_output(module, "o1", &u8);
        let o2 = file.add_output(module, "o2", &u8);
        let o3 = file.add_output(module, "o3", &u8);

        // 0 inputs -> '0
        let e0 = bitwise_or_reduce(&[], &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o0.to_expr(), &e0);
        file.add_member_continuous_assignment(module, assignment);

        // 1 input -> a
        let e1 = bitwise_or_reduce(&[a.to_expr()], &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o1.to_expr(), &e1);
        file.add_member_continuous_assignment(module, assignment);

        // 2 inputs -> a | b
        let e2 = bitwise_or_reduce(&[a.to_expr(), b.to_expr()], &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o2.to_expr(), &e2);
        file.add_member_continuous_assignment(module, assignment);

        // 3 inputs -> a | b | c
        let e3 = bitwise_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], &mut file).unwrap();
        let assignment = file.make_continuous_assignment(&o3.to_expr(), &e3);
        file.add_member_continuous_assignment(module, assignment);

        let verilog = file.emit();
        let want = r#"module bor(
  input wire [7:0] a,
  input wire [7:0] b,
  input wire [7:0] c,
  output wire [7:0] o0,
  output wire [7:0] o1,
  output wire [7:0] o2,
  output wire [7:0] o3
);
  assign o0 = '0;
  assign o1 = a;
  assign o2 = a | b;
  assign o3 = a | b | c;
endmodule
"#;
        assert_eq!(verilog, want);
    }

    #[test]
    fn logical_and_reductions_cover_empty_inverted_and_multiple_inputs() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("logical_and_reductions");
        let scalar = file.make_scalar_type();
        let a = file.add_input(module, "a", &scalar).to_expr();
        let b = file.add_input(module, "b", &scalar).to_expr();
        let c = file.add_input(module, "c", &scalar).to_expr();

        let cases = [
            ("empty", vec![], false),
            ("inverted_empty", vec![], true),
            ("single", vec![a], false),
            ("inverted_single", vec![a], true),
            ("multiple", vec![a, b], false),
            ("inverted_multiple", vec![a, b, c], true),
        ];
        for (name, inputs, inverted) in cases {
            let output = file.add_output(module, name, &scalar);
            let reduced = logical_and_reduce(&inputs, inverted, &mut file)
                .expect("logical-and reduction is representable");
            let assignment = file.make_continuous_assignment(&output.to_expr(), &reduced);
            file.add_member_continuous_assignment(module, assignment);
        }

        let expected = r#"module logical_and_reductions(
  input wire a,
  input wire b,
  input wire c,
  output wire empty,
  output wire inverted_empty,
  output wire single,
  output wire inverted_single,
  output wire multiple,
  output wire inverted_multiple
);
  assign empty = 1'h1;
  assign inverted_empty = 1'h0;
  assign single = a;
  assign inverted_single = !a;
  assign multiple = a && b;
  assign inverted_multiple = !(a && b && c);
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn bitwise_or_reduces_multidimensional_packed_array_elements_in_order() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("array_reduction");
        let element = file.make_bit_vector_type(4, false);
        let array = file.make_packed_array_type(element, &[2, 3]);
        let input = file.add_input(module, "elements", &array);
        let output = file.add_output(module, "reduced", &element);

        let reduced =
            bitwise_or_reduce_array_elements(&input.to_indexable_expr(), &[2, 3], &mut file)
                .expect("packed array elements can be reduced");
        let assignment = file.make_continuous_assignment(&output.to_expr(), &reduced);
        file.add_member_continuous_assignment(module, assignment);

        let expected = r#"module array_reduction(
  input wire [1:0][2:0][3:0] elements,
  output wire [3:0] reduced
);
  assign reduced = elements[0][0] | elements[0][1] | elements[0][2] | elements[1][0] | elements[1][1] | elements[1][2];
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn native_register_groups_can_be_added_inside_generate_loops() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("generated_register");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let state = file
            .add_logic(module, "state", &scalar)
            .expect("register name is unique");
        let zero = file.make_plain_literal(0, &LiteralFormat::Default);
        let one = file.make_plain_literal(1, &LiteralFormat::Default);
        let generate = file.add_generate_loop(module, "index", &zero, &one, Some("lanes"));

        add_registers(
            &clock.to_expr(),
            None,
            &[RegisterDefinition {
                reg: state.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            }],
            RegisterScope::GenerateLoop(generate),
            &mut file,
        )
        .expect("generate-scope register group is valid");

        let expected = r#"module generated_register(
  input wire clock,
  input wire data
);
  logic state;
  for (genvar index = 0; index < 1; index = index + 1) begin : lanes
    always_ff @ (posedge clock) begin
      state <= data;
    end
  end
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn logical_or_reduction_rejects_a_foreign_singleton() {
        let mut original = VastFile::new(VastFileType::SystemVerilog);
        let foreign = original.make_plain_literal(1, &LiteralFormat::Default);
        let mut destination = VastFile::new(VastFileType::SystemVerilog);

        let _ = logical_or_reduce(&[foreign], false, &mut destination);
    }

    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn logical_and_reduction_rejects_a_foreign_singleton() {
        let mut original = VastFile::new(VastFileType::SystemVerilog);
        let foreign = original.make_plain_literal(1, &LiteralFormat::Default);
        let mut destination = VastFile::new(VastFileType::SystemVerilog);

        let _ = logical_and_reduce(&[foreign], false, &mut destination);
    }

    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn bitwise_or_reduction_rejects_a_foreign_singleton() {
        let mut original = VastFile::new(VastFileType::SystemVerilog);
        let foreign = original.make_plain_literal(1, &LiteralFormat::Default);
        let mut destination = VastFile::new(VastFileType::SystemVerilog);

        let _ = bitwise_or_reduce(&[foreign], &mut destination);
    }

    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn array_reduction_checks_foreign_subject_before_validating_dimensions() {
        let mut original = VastFile::new(VastFileType::SystemVerilog);
        let module = original.add_module("foreign_array");
        let scalar = original.make_scalar_type();
        let foreign = original.add_input(module, "foreign", &scalar);
        let mut destination = VastFile::new(VastFileType::SystemVerilog);

        let _ =
            bitwise_or_reduce_array_elements(&foreign.to_indexable_expr(), &[], &mut destination);
    }

    #[test]
    fn array_reductions_reject_empty_nonpositive_and_mismatched_dimensions() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("checked_array");
        let element = file.make_bit_vector_type(4, false);
        let array = file.make_packed_array_type(element, &[2, 3]);
        let input = file.add_input(module, "elements", &array);
        let before = file.emit();

        let cases: &[(&[i64], &str)] = &[
            (
                &[],
                "packed-array reduction requires at least one dimension",
            ),
            (&[0, 3], "packed-array dimensions must be greater than zero"),
            (
                &[-1, 3],
                "packed-array dimensions must be greater than zero",
            ),
            (
                &[2, 4],
                "packed-array dimensions do not match expression type: expected \
                 [2, 3], got [2, 4]",
            ),
            (
                &[2],
                "packed-array dimensions do not match expression type: expected \
                 [2, 3], got [2]",
            ),
        ];
        for (dimensions, expected) in cases {
            let error =
                bitwise_or_reduce_array_elements(&input.to_indexable_expr(), dimensions, &mut file)
                    .expect_err("invalid array dimensions are rejected");
            assert_eq!(error.to_string(), *expected);
            assert_eq!(file.emit(), before);
        }
    }

    #[test]
    fn array_reductions_account_for_previously_indexed_dimensions() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("array_row");
        let element = file.make_bit_vector_type(4, false);
        let array = file.make_packed_array_type(element, &[2, 3]);
        let input = file.add_input(module, "elements", &array);
        let row = file.make_index(&input.to_indexable_expr(), 1);

        let reduced = bitwise_or_reduce_array_elements(&row.to_indexable_expr(), &[3], &mut file)
            .expect("the indexed dimension has already been consumed");

        assert_eq!(
            file.emit_expression(&reduced),
            "elements[1][0] | elements[1][1] | elements[1][2]"
        );
    }

    #[test]
    fn array_reductions_reject_unrepresentable_element_counts() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("large_array");
        let element = file.make_scalar_type();
        let array = file.make_packed_array_type(element, &[i64::MAX, 3]);
        let input = file.add_input(module, "elements", &array);

        let error =
            bitwise_or_reduce_array_elements(&input.to_indexable_expr(), &[i64::MAX, 3], &mut file)
                .expect_err("packed arrays with overflowing element counts are rejected");

        assert_eq!(
            error.to_string(),
            "packed-array element count exceeds usize"
        );
    }
}
