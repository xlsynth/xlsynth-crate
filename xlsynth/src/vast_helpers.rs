// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::{
    ir_value::IrFormatPreference,
    vast::{Expr, GenerateLoop, IndexableExpr, VastFile, VastModule},
    XlsynthError,
};

use crate::vast_helpers_options::{CodegenOptions, TemplateVariable};

#[derive(Clone)]
pub struct Reset {
    pub signal: Expr,
    pub active_low: bool,
}

#[derive(Clone)]
pub struct RegisterDefinition {
    pub reg: Expr,
    pub next: Expr,
    pub reset_value: Option<Expr>,
    pub enable: Option<Expr>,
}

fn emit_template(template: &str, keys: &HashMap<TemplateVariable, String>) -> String {
    let mut template = template.to_string();
    for (key, value) in keys {
        let placeholder = format!("{{{{{}}}}}", key);
        template = template.replace(&placeholder, value);
    }
    template
}

fn emit_registers_with_templates(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    file: &mut VastFile,
    scope: &mut RegisterScope,
    opts: &CodegenOptions,
) -> Result<(), XlsynthError> {
    for register in registers {
        let mut keys: HashMap<TemplateVariable, String> = HashMap::new();
        keys.insert(TemplateVariable::Clock, clk.emit());
        if let Some(ref r) = reset {
            keys.insert(TemplateVariable::Reset, r.signal.emit());
        }
        keys.insert(TemplateVariable::Reg, register.reg.emit());
        keys.insert(TemplateVariable::Next, register.next.emit());
        if let Some(reset_value) = &register.reset_value {
            keys.insert(TemplateVariable::ResetValue, reset_value.emit());
        }
        if let Some(enable) = &register.enable {
            keys.insert(TemplateVariable::Enable, enable.emit());
        }
        let inline_string = match (
            keys.get(&TemplateVariable::ResetValue),
            keys.get(&TemplateVariable::Enable),
        ) {
            // TODO(meheff): use a CodegenOptions verify method instead of these unwraps
            (Some(_), Some(_)) => emit_template(
                opts.reg_with_reset_with_en_template.as_ref().unwrap(),
                &keys,
            ),
            (Some(_), None) => emit_template(opts.reg_with_reset_template.as_ref().unwrap(), &keys),
            (None, Some(_)) => emit_template(opts.reg_with_en_template.as_ref().unwrap(), &keys),
            (None, None) => emit_template(opts.reg_template.as_ref().unwrap(), &keys),
        };
        let inline_statement = file.make_inline_verilog_statement(&format!("{};", inline_string));
        match scope {
            RegisterScope::Module(module) => {
                module.add_member_inline_statement(inline_statement);
            }
            RegisterScope::GenerateLoop(generate_loop) => {
                generate_loop.add_inline_statement(&inline_statement);
            }
        }
    }
    Ok(())
}

pub enum RegisterScope<'a> {
    Module(&'a mut VastModule),
    GenerateLoop(&'a mut GenerateLoop),
}

// Defines an always_ff block for the given registers.
pub fn add_registers<'a>(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    scope: &mut RegisterScope<'a>,
    file: &mut VastFile,
    options: Option<&CodegenOptions>,
) -> Result<(), XlsynthError> {
    if registers.is_empty() {
        return Ok(());
    }

    let any_with_reset = registers.iter().any(|r| r.reset_value.is_some());
    if any_with_reset {
        assert!(
            reset.is_some(),
            "Reset must be provided when any register has a reset value"
        );
    }

    if let Some(opts) = options {
        if [
            &opts.reg_template,
            &opts.reg_with_en_template,
            &opts.reg_with_reset_template,
            &opts.reg_with_reset_with_en_template,
        ]
        .iter()
        .any(|t| t.is_some())
        {
            return emit_registers_with_templates(clk, reset, registers, file, scope, opts);
        }
    }

    // Build a single always_ff block for all registers
    let posedge_clk = file.make_pos_edge(clk);
    let mut always_ff = match scope {
        RegisterScope::Module(module) => {
            module.add_always_ff(&[&posedge_clk])?.get_statement_block()
        }
        RegisterScope::GenerateLoop(generate_loop) => generate_loop
            .add_always_ff(&[&posedge_clk])?
            .get_statement_block(),
    };

    // Precompute assigned values (with enable ternary when provided)
    let mut assigned_values: Vec<Expr> = Vec::with_capacity(registers.len());
    for r in registers.iter() {
        let val: Expr = match &r.enable {
            Some(en) => file.make_ternary(en, &r.next, &r.reg),
            None => r.next.clone(),
        };
        assigned_values.push(val);
    }

    // Unconditionally assign registers without reset values
    for (i, r) in registers.iter().enumerate() {
        if r.reset_value.is_none() {
            always_ff.add_nonblocking_assignment(&r.reg, &assigned_values[i]);
        }
    }

    // Conditionally assign registers with reset values
    if any_with_reset {
        let rst = reset.expect("reset must be provided when registers have reset values");
        let cond_expr = if rst.active_low {
            file.make_logical_not(&rst.signal)
        } else {
            rst.signal.clone()
        };
        let cond = always_ff.add_cond(&cond_expr);
        let mut then_blk = cond.then_block();
        let mut else_blk = cond.add_else();
        for (i, r) in registers.iter().enumerate() {
            if let Some(ref rv) = r.reset_value {
                then_blk.add_nonblocking_assignment(&r.reg, rv);
                else_blk.add_nonblocking_assignment(&r.reg, &assigned_values[i]);
            }
        }
    }

    Ok(())
}

// Internal helper to fold a list of expressions using a binary reducer.
// - inputs: slice of expressions to reduce
// - identity: expression used when inputs is empty
// - combine: function that combines two expressions into one (e.g., a || b)
fn reduce_with<F>(
    inputs: &[Expr],
    identity: Expr,
    mut combine: F,
    file: &mut VastFile,
) -> Result<Expr, XlsynthError>
where
    F: FnMut(&mut VastFile, &Expr, &Expr) -> Expr,
{
    if inputs.is_empty() {
        return Ok(identity);
    }
    if inputs.len() == 1 {
        return Ok(inputs[0].clone());
    }
    let mut temps: Vec<Expr> = Vec::new();
    let mut accum: &Expr = &inputs[0];
    for input in inputs[1..].iter() {
        temps.push(combine(file, accum, input));
        accum = temps.last().unwrap();
    }
    Ok(accum.clone())
}

pub fn logical_or_reduce(
    inputs: &[Expr],
    invert: bool,
    file: &mut VastFile,
) -> Result<Expr, XlsynthError> {
    if inputs.is_empty() && invert {
        // Special case to avoid emitting !1'h0.
        return file.make_literal("bits[1]:1", &IrFormatPreference::Hex);
    }
    let id0 = file.make_literal("bits[1]:0", &IrFormatPreference::Hex)?;
    let reduced = reduce_with(inputs, id0, |f, a, b| f.make_logical_or(a, b), file)?;
    if invert {
        Ok(file.make_logical_not(&reduced))
    } else {
        Ok(reduced)
    }
}

pub fn logical_and_reduce(
    inputs: &[Expr],
    invert: bool,
    file: &mut VastFile,
) -> Result<Expr, XlsynthError> {
    if inputs.is_empty() && invert {
        // Special case to avoid emitting !1'h1.
        return file.make_literal("bits[1]:0", &IrFormatPreference::Hex);
    };
    let id1 = file.make_literal("bits[1]:1", &IrFormatPreference::Hex)?;
    let reduced = reduce_with(inputs, id1, |f, a, b| f.make_logical_and(a, b), file)?;
    if invert {
        Ok(file.make_logical_not(&reduced))
    } else {
        Ok(reduced)
    }
}

pub fn bitwise_or_reduce(inputs: &[Expr], file: &mut VastFile) -> Result<Expr, XlsynthError> {
    let id = file.make_unsized_zero_literal();
    reduce_with(inputs, id, |f, a, b| f.make_bitwise_or(a, b), file)
}

// Gathers all of the elements of the packed array given by `expr` into
// `elements`. `expr` should be a packed array with dimensions matching
// `dimensions` (major dimension first). Elements are created by making Verilog
// index expressions.
fn gather_elements(
    expr: &IndexableExpr,
    dimensions: &[i64],
    elements: &mut Vec<Expr>,
    file: &mut VastFile,
) {
    if dimensions.is_empty() {
        elements.push(expr.to_expr());
    } else {
        for i in 0..dimensions[0] {
            let index_literal = file.make_plain_literal(i as i32, &IrFormatPreference::Default);
            let index_expr = file
                .make_index_expr(expr, &index_literal)
                .to_indexable_expr();
            gather_elements(&index_expr, &dimensions[1..], elements, file);
        }
    }
}

/// Returns an expression which is the bitwise OR of all of the elements of the
/// packed array given by `expr`. `expr` should be a packed array with
/// dimensions matching `dimensions` (major dimension first).
pub fn bitwise_or_reduce_array_elements(
    expr: &IndexableExpr,
    dimensions: &[i64],
    file: &mut VastFile,
) -> Result<Expr, XlsynthError> {
    let mut elements: Vec<Expr> = Vec::new();
    gather_elements(expr, dimensions, &mut elements, file);
    bitwise_or_reduce(&elements, file)
}

#[cfg(test)]
mod tests {
    use crate::vast::{VastFile, VastFileType};
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_logical_or_reduce_various_arity() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let mut module = file.add_module("lor");

        // Inputs: scalar logic signals
        let scalar = file.make_scalar_type();
        let a = module.add_input("a", &scalar);
        let b = module.add_input("b", &scalar);
        let c = module.add_input("c", &scalar);

        // Outputs
        let o0 = module.add_output("o0", &scalar);
        let o1 = module.add_output("o1", &scalar);
        let o2 = module.add_output("o2", &scalar);
        let o3 = module.add_output("o3", &scalar);

        // 0 inputs -> 0
        let e0 = logical_or_reduce(&[], false, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o0.to_expr(), &e0));

        // 1 input -> a
        let e1 = logical_or_reduce(&[a.to_expr()], false, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o1.to_expr(), &e1));

        // 2 inputs -> a || b
        let e2 = logical_or_reduce(&[a.to_expr(), b.to_expr()], false, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o2.to_expr(), &e2));

        // 3 inputs -> a || b || c
        let e3 =
            logical_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], false, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o3.to_expr(), &e3));

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
        let mut module = file.add_module("regs_rst_en");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = module.add_input("clk", &bit1);
        let clk_expr = clk.to_expr();
        let rst = module.add_input("rst", &bit1);
        let en = module.add_input("en", &bit1);
        let r = module.add_logic("r", &u8).expect("add_logic r");
        let r_next = module.add_logic("r_next", &u8).expect("add_logic r_next");
        let r2 = module.add_logic("r2", &u8).expect("add_logic r2");
        let r2_next = module.add_logic("r2_next", &u8).expect("add_logic r2_next");

        let reset_val = file
            .make_literal("bits[8]:0xAA", &IrFormatPreference::Hex)
            .expect("literal ok");
        let reset_val_r2 = file
            .make_literal("bits[8]:0x55", &IrFormatPreference::Hex)
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
            &mut RegisterScope::Module(&mut module),
            &mut file,
            None,
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
        let mut module = file.add_module("regs_no_rst");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = module.add_input("clk", &bit1);
        let clk_expr = clk.to_expr();
        let en = module.add_input("en", &bit1);
        let r = module.add_logic("r", &u8).expect("add_logic r");
        let r_next = module.add_logic("r_next", &u8).expect("add_logic r_next");
        let r2 = module.add_logic("r2", &u8).expect("add_logic r2");
        let r2_next = module.add_logic("r2_next", &u8).expect("add_logic r2_next");

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
            &mut RegisterScope::Module(&mut module),
            &mut file,
            None,
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
    fn test_add_registers_mixed_resets_panics() {
        // Mixed resets are now supported; verify generated output
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let mut module = file.add_module("regs_mixed");

        let bit1 = file.make_bit_vector_type(1, false);
        let u8 = file.make_bit_vector_type(8, false);

        let clk = module.add_input("clk", &bit1);
        let clk_expr = clk.to_expr();
        let rst = module.add_input("rst", &bit1);
        let en = module.add_input("en", &bit1);
        let r1 = module.add_logic("r1", &u8).expect("r1");
        let n1 = module.add_logic("n1", &u8).expect("n1");
        let r2 = module.add_logic("r2", &u8).expect("r2");
        let n2 = module.add_logic("n2", &u8).expect("n2");

        let reset_val = file
            .make_literal("bits[8]:0xAA", &IrFormatPreference::Hex)
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
            &mut RegisterScope::Module(&mut module),
            &mut file,
            None,
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
    fn test_add_registers_missing_reset_panics() {
        let res = std::panic::catch_unwind(|| {
            let mut file = VastFile::new(VastFileType::SystemVerilog);
            let mut module = file.add_module("regs_missing_rst");

            let bit1 = file.make_bit_vector_type(1, false);
            let u8 = file.make_bit_vector_type(8, false);

            let clk = module.add_input("clk", &bit1);
            let clk_expr = clk.to_expr();
            let r = module.add_logic("r", &u8).expect("r");
            let n = module.add_logic("n", &u8).expect("n");
            let reset_val = file
                .make_literal("bits[8]:0xAA", &IrFormatPreference::Hex)
                .expect("literal ok");

            let regs = [RegisterDefinition {
                reg: r.to_expr(),
                next: n.to_expr(),
                reset_value: Some(reset_val),
                enable: None,
            }];

            let _ = add_registers(
                &clk_expr,
                None,
                &regs,
                &mut RegisterScope::Module(&mut module),
                &mut file,
                None,
            );
        });
        assert!(
            res.is_err(),
            "expected panic when reset not provided but required"
        );
    }
    #[test]
    fn test_logical_or_reduce_inverted() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let mut module = file.add_module("lori");

        // Inputs: scalar logic signals
        let scalar = file.make_scalar_type();
        let a = module.add_input("a", &scalar);
        let b = module.add_input("b", &scalar);
        let c = module.add_input("c", &scalar);

        // Outputs
        let o0 = module.add_output("o0", &scalar);
        let o1 = module.add_output("o1", &scalar);
        let o2 = module.add_output("o2", &scalar);
        let o3 = module.add_output("o3", &scalar);

        // 0 inputs -> 1
        let e0 = logical_or_reduce(&[], true, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o0.to_expr(), &e0));

        // 1 input -> !a
        let e1 = logical_or_reduce(&[a.to_expr()], true, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o1.to_expr(), &e1));

        // 2 inputs -> !(a || b)
        let e2 = logical_or_reduce(&[a.to_expr(), b.to_expr()], true, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o2.to_expr(), &e2));

        // 3 inputs -> !(a || b || c)
        let e3 =
            logical_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], true, &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o3.to_expr(), &e3));

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
        let mut module = file.add_module("bor");

        // Inputs: 8-bit vectors
        let u8 = file.make_bit_vector_type(8, false);
        let a = module.add_input("a", &u8);
        let b = module.add_input("b", &u8);
        let c = module.add_input("c", &u8);

        // Outputs
        let o0 = module.add_output("o0", &u8);
        let o1 = module.add_output("o1", &u8);
        let o2 = module.add_output("o2", &u8);
        let o3 = module.add_output("o3", &u8);

        // 0 inputs -> '0
        let e0 = bitwise_or_reduce(&[], &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o0.to_expr(), &e0));

        // 1 input -> a
        let e1 = bitwise_or_reduce(&[a.to_expr()], &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o1.to_expr(), &e1));

        // 2 inputs -> a | b
        let e2 = bitwise_or_reduce(&[a.to_expr(), b.to_expr()], &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o2.to_expr(), &e2));

        // 3 inputs -> a | b | c
        let e3 = bitwise_or_reduce(&[a.to_expr(), b.to_expr(), c.to_expr()], &mut file).unwrap();
        module
            .add_member_continuous_assignment(file.make_continuous_assignment(&o3.to_expr(), &e3));

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
}
