// SPDX-License-Identifier: Apache-2.0

//! Generate typed expressions in the common BX and DSLX value language.

use std::fmt;

/// The declarations and body shared by the BX block and DSLX function.
#[derive(Debug)]
pub struct ExpressionSample {
    pub params: String,
    pub result_type: String,
    pub bindings: String,
    pub expression: String,
}

/// The signedness and bit count required of a generated scalar expression.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Bits {
    signed: bool,
    width: usize,
}

impl Bits {
    fn unsigned(width: usize) -> Self {
        Self {
            signed: false,
            width,
        }
    }

    fn from_byte(byte: u8) -> Self {
        const WIDTHS: [usize; 16] = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 96, 128, 0];
        let width = WIDTHS[usize::from(byte & 15)];
        Self {
            signed: byte & 16 != 0 && width != 0,
            width,
        }
    }
}

impl fmt::Display for Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.signed { 's' } else { 'u' };
        if (1..=64).contains(&self.width) {
            write!(f, "{prefix}{}", self.width)
        } else {
            write!(f, "{prefix}N[{}]", self.width)
        }
    }
}

/// Consume one byte per choice, falling back to a variable leaf at end of
/// input.
struct Generator<'a> {
    data: &'a [u8],
    variables: Vec<(String, Bits)>,
    inferred_slice_width: Option<usize>,
    nodes_left: usize,
    domain: EvaluationDomain,
}

impl Generator<'_> {
    fn byte(&mut self) -> u8 {
        match self.data.split_first() {
            Some((&first, rest)) => {
                self.data = rest;
                first
            }
            None => 0,
        }
    }

    /// A hex literal with the high bit set has exactly the given minimum width.
    fn hex_literal(&mut self, width: usize, set_high_bit: bool) -> String {
        let digits = width.div_ceil(4);
        let mut hex = String::from("0x");
        for i in 0..digits {
            let digit_width = if i == 0 { width - 4 * (digits - 1) } else { 4 };
            let mut digit = u32::from(self.byte()) & ((1 << digit_width) - 1);
            if i == 0 && set_high_bit {
                digit |= 1 << (digit_width - 1);
            }
            hex.push(char::from_digit(digit, 16).unwrap());
        }
        hex
    }

    /// Generate a literal that fits its explicit type or its operand context.
    fn literal(&mut self, ty: Bits, explicit_type: bool) -> String {
        let format = self.byte();
        // A typed signed hex literal can spell any bit pattern. A leading
        // implicit hex literal may merge with other bare operands before
        // encountering the signed type, requiring a bit for the sign.
        let hex_width = ty.width - usize::from(ty.signed && !explicit_type);
        if format % 3 == 2 && hex_width != 0 {
            return self.hex_literal(hex_width, false);
        }
        let magnitude_bits = (ty.width - usize::from(ty.signed)).min(8);
        let magnitude = u16::from(self.byte()) & ((1 << magnitude_bits) - 1);
        if ty.signed && format % 3 == 1 {
            format!("-{}", magnitude + 1)
        } else {
            magnitude.to_string()
        }
    }

    fn leaf(&mut self, ty: Bits) -> String {
        let selector = usize::from(self.byte());
        if selector & 0x80 != 0 {
            let value = self.literal(ty, true);
            return format!("({ty}:{value})");
        }
        let (name, variable_ty) = &self.variables[selector % self.variables.len()];
        if ty == *variable_ty {
            name.clone()
        } else {
            format!("({name} as {ty})")
        }
    }

    fn expr(&mut self, ty: Bits, depth: u8) -> String {
        let opcode = self.byte();
        self.with_opcode(ty, depth, opcode)
    }

    /// Generate a fold over a small constant range. Its seed and body use the
    /// surrounding value grammar, including enclosing loop variables.
    fn fold(&mut self, ty: Bits, depth: u8, array: bool) -> String {
        let magnitude = i16::from(self.byte() % 4);
        let count = i16::from(self.byte() % 4);
        let flags = self.byte();
        let index_ty = Bits {
            signed: flags & 0x80 != 0,
            width: [32, 4, 16, 65][usize::from(flags >> 5) % 4],
        };
        let start = if index_ty.signed {
            -magnitude
        } else {
            magnitude
        };
        let limit = if flags & 1 == 0 {
            format!("{index_ty}:{}", start + count)
        } else {
            format!("({index_ty}:{start} + {index_ty}:{count})")
        };
        let acc_type = if array {
            format!("{ty}[2]")
        } else {
            ty.to_string()
        };
        let annotation = if flags & 2 == 0 {
            format!(": ({index_ty}, {acc_type})")
        } else {
            String::new()
        };
        let seed = if flags & 4 != 0 {
            format!("zero!<{acc_type}>()")
        } else if array {
            format!("[{}, {}]", self.expr(ty, depth), self.expr(ty, depth))
        } else {
            self.expr(ty, depth)
        };
        // Depth decreases on recursion, giving nested loops distinct binders.
        let idx = format!("idx_{depth}");
        let acc = format!("acc_{depth}");
        let locals_start = self.variables.len();
        self.variables.push((idx.clone(), index_ty));
        let slot = format!("({idx} as u32) % u32:2");
        let (current, mut bindings) = if array {
            let item = format!("item_{depth}");
            let read = format!("let {item} = {acc}[{slot}]; ");
            (item, read)
        } else {
            (acc.clone(), String::new())
        };
        self.variables.push((current.clone(), ty));
        let mut term = self.expr(ty, depth);
        if flags & 8 != 0 {
            let local = format!("term_{depth}");
            let hint = if flags & 16 == 0 {
                format!(": {ty}")
            } else {
                String::new()
            };
            bindings.push_str(&format!("let {local}{hint} = {term}; "));
            term = local;
        }
        let result = match self.byte() % 4 {
            0 => term,
            1 => format!("({current} + {term})"),
            2 => format!("(({idx} as {ty}) - ({current} ^ {term}))"),
            _ => format!("(({current} * {term}) - ({idx} as {ty}))"),
        };
        self.variables.truncate(locals_start);
        let result = if array {
            format!("update({acc}, {slot}, {result})")
        } else {
            result
        };
        format!(
            "(unroll_for! ({idx}, {acc}){annotation} in {index_ty}:{start}..{limit} {{ {bindings}{result} }}({seed}))"
        )
    }

    /// Produce a fixed-width slice with either a dynamic or a constant offset.
    fn slice(&mut self, result: Bits, depth: u8) -> String {
        let source_width = Bits::from_byte(self.byte()).width.max(result.width).max(1);
        let source = match self.byte() % 4 {
            2 => self.hex_literal(source_width, true),
            3 => {
                let left = self.hex_literal(source_width, true);
                let right = self.literal(Bits::unsigned(source_width), false);
                format!("({left} + {right})")
            }
            _ => self.expr(Bits::unsigned(source_width), depth),
        };
        let start = match self.byte() % 4 {
            0 => String::from("offset"),
            1 => {
                let ty = Bits::unsigned(Bits::from_byte(self.byte()).width);
                let index = self.expr(ty, depth);
                if ty.width > source_width {
                    // A wider index is legal when explicitly typed. Its
                    // arithmetic then infers leading literals independently
                    // of the narrower word being sliced.
                    format!("({index} as {ty})")
                } else {
                    index
                }
            }
            _ => {
                let index = match self.byte() % 4 {
                    0 => source_width - 1,
                    1 => source_width,
                    2 => source_width + 1,
                    _ => usize::from(self.byte()),
                };
                (index & ((1 << source_width.min(8)) - 1)).to_string()
            }
        };
        // The pinned bytecode interpreter returns an unsigned zero for a
        // signed slice whose start is past the end. A later signed operation
        // can then fail on incompatible runtime values. Keep signed starts
        // in bounds for bytecode comparisons and retain the unrestricted
        // generation for the hardware differential target.
        let start = if self.domain == EvaluationDomain::Bytecode && result.signed {
            format!("((({start}) as u32) % u32:{source_width})")
        } else {
            start
        };
        let result_type = match self.byte() % 4 {
            0 => result.to_string(),
            1 => format!("xN[{}][{}]", result.signed, result.width),
            2 if self
                .inferred_slice_width
                .is_none_or(|width| width == result.width) =>
            {
                self.inferred_slice_width = Some(result.width);
                format!("xN[{}][width]", result.signed)
            }
            _ => {
                let prefix = if result.signed { "sN" } else { "uN" };
                let split = usize::from(self.byte()) % (result.width + 1);
                format!("{prefix}[{split} + {}]", result.width - split)
            }
        };
        format!("({source})[{start}+:{result_type}]")
    }

    /// Compose terms of the requested type. Depth and total nodes bound the
    /// work even for inputs that repeatedly select branching expressions.
    fn with_opcode(&mut self, ty: Bits, depth: u8, opcode: u8) -> String {
        if depth == 0 || self.nodes_left == 0 || opcode.is_multiple_of(16) {
            return self.leaf(ty);
        }
        self.nodes_left -= 1;
        let child_depth = depth - 1;
        if ty.width == 0 {
            // A zero-width value can be produced by slicing any unsigned word.
            return self.slice(ty, child_depth);
        }
        match opcode % 16 {
            1..=3 => {
                const OPS: [&str; 8] = ["+", "-", "*", "/", "%", "&", "|", "^"];
                let op = OPS[usize::from(self.byte()) % OPS.len()];
                let operand = self.expr(ty, child_depth);
                let (left, right) = if opcode % 16 == 3 {
                    let literal = self.literal(ty, false);
                    if opcode & 16 == 0 {
                        (operand, literal)
                    } else {
                        (literal, operand)
                    }
                } else {
                    (operand, self.expr(ty, child_depth))
                };
                format!("({left} {op} {right})")
            }
            4 => {
                let op = if self.byte() & 1 == 0 { "!" } else { "-" };
                let value = self.expr(ty, child_depth);
                format!("({op}({value}))")
            }
            5 => {
                let op = if self.byte() & 1 == 0 { "<<" } else { ">>" };
                let left = self.expr(ty, child_depth);
                let right = if self.byte() & 1 == 0 {
                    String::from("offset")
                } else {
                    (usize::from(self.byte()) % (ty.width + 1)).to_string()
                };
                // Parenthesize the shifted term itself. In e.g.
                // `x < (x >> 1)` DSLX can read the angle brackets as
                // parametric arguments, whereas `x < ((x) >> 1)` is clear.
                format!("(({left}) {op} {right})")
            }
            6 | 7 => self.slice(ty, child_depth),
            8 => {
                let from = Bits::from_byte(self.byte());
                let value = self.expr(from, child_depth);
                format!("({value} as {ty})")
            }
            9 => {
                let cond = self.expr(Bits::unsigned(1), child_depth);
                let then_value = self.expr(ty, child_depth);
                let else_value = self.expr(ty, child_depth);
                format!("(if {cond} {{ {then_value} }} else {{ {else_value} }})")
            }
            10 => {
                const OPS: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];
                let op = OPS[usize::from(self.byte()) % OPS.len()];
                let operand = Bits::from_byte(self.byte());
                let left = self.expr(operand, child_depth);
                let right = self.expr(operand, child_depth);
                format!("(({left} {op} {right}) as {ty})")
            }
            11 => {
                let op = if self.byte() & 1 == 0 { "&&" } else { "||" };
                let left = self.expr(Bits::unsigned(1), child_depth);
                let right = self.expr(Bits::unsigned(1), child_depth);
                format!("(({left} {op} {right}) as {ty})")
            }
            12 => {
                let split = usize::from(self.byte()) % (ty.width + 1);
                let left = self.expr(Bits::unsigned(split), child_depth);
                let right = self.expr(Bits::unsigned(ty.width - split), child_depth);
                format!("(({left} ++ {right}) as {ty})")
            }
            13 => {
                let left = self.expr(ty, child_depth);
                let right = self.expr(ty, child_depth);
                let index = self.expr(Bits::unsigned(1), child_depth);
                // A one-bit subscript always selects an existing element.
                format!("([{left}, {right}])[{index}]")
            }
            14 => {
                let field = self.byte() & 1;
                let other = Bits::from_byte(self.byte());
                let left = self.expr(if field == 0 { ty } else { other }, child_depth);
                let right = self.expr(if field == 1 { ty } else { other }, child_depth);
                format!("({left}, {right}).{field}")
            }
            _ => self.fold(ty, child_depth, false),
        }
    }
}

/// The execution models that the generated expressions must exercise.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EvaluationDomain {
    Hardware,
    Bytecode,
}

/// Decode even an empty byte sequence into a small, type-directed program.
pub fn generate(data: &[u8], domain: EvaluationDomain) -> ExpressionSample {
    let mut generator = Generator {
        data,
        variables: Vec::new(),
        inferred_slice_width: None,
        nodes_left: 32,
        domain,
    };
    let root_op = generator.byte();
    let result = Bits::from_byte(generator.byte());
    let x = Bits::from_byte(generator.byte());
    let y = Bits::from_byte(generator.byte());
    let offset = Bits::unsigned(Bits::from_byte(generator.byte()).width);
    generator.variables.extend([
        ("x".into(), x),
        ("y".into(), y),
        ("offset".into(), offset),
        ("flag".into(), Bits::unsigned(1)),
    ]);
    let params = format!("x: {x}, y: {y}, offset: {offset}, flag: bool");

    let shape = generator.byte();
    let mut bindings = String::new();
    if shape & 16 != 0 {
        let binding_ty = Bits::from_byte(generator.byte());
        let initializer = generator.expr(binding_ty, 3);
        bindings = if shape & 32 != 0 {
            format!("let v = {initializer}; ")
        } else {
            format!("let v: {binding_ty} = {initializer}; ")
        };
        generator.variables.push(("v".into(), binding_ty));
    }
    let first = if shape & 7 == 5 {
        generator.fold(result, 3, true)
    } else {
        generator.with_opcode(result, 5, root_op)
    };
    let (result_type, expression) = match shape & 7 {
        5 => (format!("{result}[2]"), first),
        6 => {
            let second_ty = Bits::from_byte(generator.byte());
            let second = generator.expr(second_ty, 4);
            (
                format!("({result}, {second_ty})"),
                format!("({first}, {second})"),
            )
        }
        7 => {
            let second = generator.expr(result, 4);
            (format!("{result}[2]"), format!("[{first}, {second}]"))
        }
        _ => (result.to_string(), first),
    };
    if let Some(width) = generator.inferred_slice_width {
        bindings.insert_str(0, &format!("let width = {width}; "));
    }
    ExpressionSample {
        params,
        result_type,
        bindings,
        expression,
    }
}
