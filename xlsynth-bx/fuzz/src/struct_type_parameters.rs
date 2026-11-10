// SPDX-License-Identifier: Apache-2.0

//! Compare type-parametric BX block results with monomorphic DSLX records.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

/// A payload and its structural inputs, with an optional nominal wrapper.
struct Payload {
    ty: String,
    input: String,
    argument: String,
    projection: &'static str,
    declaration: String,
}

impl Payload {
    fn new(shape: u8, width: usize, name: &str, input: &str) -> Self {
        let scalar = if shape & 1 == 0 {
            format!("uN[{width}]")
        } else {
            format!("sN[{width}]")
        };
        let input_type = match (shape >> 1) & 3 {
            0 => scalar,
            1 => format!("({scalar}, bool)"),
            2 => format!("{scalar}[{}]", (width % 4) + 1),
            _ => format!("(bool, {scalar}[2])"),
        };
        if shape & 8 == 0 {
            Self {
                ty: input_type.clone(),
                input: input_type,
                argument: input.into(),
                projection: "",
                declaration: String::new(),
            }
        } else {
            Self {
                ty: name.into(),
                argument: format!("{name} {{ item: {input} }}"),
                projection: ".item",
                declaration: format!("struct {name} {{ item: {input_type} }}"),
                input: input_type,
            }
        }
    }
}

/// Check the named fields of block results across type specializations and
/// compare nominal assignment checks with the corresponding DSLX declarations.
pub fn check_struct_type_parameters(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let shapes = bytes.next().unwrap();
    let offset = usize::from(bytes.next().unwrap() % 16);
    let left = Payload::new(shapes & 15, width, "LeftPayload", "x");
    let right = if flags & 1 == 0 {
        Payload::new(shapes >> 4, width + offset, "RightPayload", "y")
    } else {
        Payload::new(shapes & 15, width, "LeftPayload", "y")
    };
    let distinct = left.ty != right.ty;
    let mismatch = distinct && flags & 2 != 0;
    let declaration = if flags & 4 == 0 { "T" } else { "T: type" };
    let constructor = match (flags >> 3) % 3 {
        0 => "FlowOut",
        1 => "FlowOut<T>",
        _ => "FlowOut::<T>",
    };
    let left_decl = &left.declaration;
    let right_decl = if distinct { &right.declaration } else { "" };
    let left_arg = &left.argument;
    let right_arg = &right.argument;
    let left_type = &left.ty;
    let right_type = &right.ty;
    let right_annotation = if mismatch { left_type } else { right_type };
    let params = format!("x: {}, y: {}, valid: bool", left.input, right.input);
    let result = format!(
        "(({}, bool), (bool, {}), {})",
        left.input, right.input, left.input
    );
    let left_field = left.projection;
    let right_field = right.projection;
    let output = format!(
        "((left.pop_data{left_field}, left.pop_valid),
          (right.pop_valid, right.pop_data{right_field}), again.pop_data{left_field})"
    );
    let bx = format!(
        r#"
{left_decl}
{right_decl}
struct FlowOut<{declaration}> {{ pop_valid: bool, pop_data: T }}
#[allow(combinational_block)]
block flow_result<T>(clk, valid: bool, data: T) -> FlowOut<T> {{
    {constructor} {{ pop_data: data, pop_valid: valid }}
}}
{}"#,
        bx_source(
            &params,
            &result,
            &format!(
                "let left = flow_result(valid, {left_arg});
                 let right: FlowOut<{right_annotation}> = flow_result(!valid, {right_arg});
                 let again = flow_result(!valid, {left_arg});
                 {output}"
            ),
        ),
    );
    let mut dslx = format!(
        r#"
{left_decl}
{right_decl}
struct FlowLeft {{ pop_valid: bool, pop_data: {left_type} }}
fn flow_left(valid: bool, data: {left_type}) -> FlowLeft {{
    FlowLeft {{ pop_valid: valid, pop_data: data }}
}}
"#
    );
    let (right_helper, right_result) = if distinct {
        dslx.push_str(&format!(
            r#"
struct FlowRight {{ pop_valid: bool, pop_data: {right_type} }}
fn flow_right(valid: bool, data: {right_type}) -> FlowRight {{
    FlowRight {{ pop_valid: valid, pop_data: data }}
}}
"#
        ));
        ("flow_right", "FlowRight")
    } else {
        ("flow_left", "FlowLeft")
    };
    let right_result = if mismatch { "FlowLeft" } else { right_result };
    let body = format!(
        "let left = flow_left(valid, {left_arg});
         let right: {right_result} = {right_helper}(!valid, {right_arg});
         let again = flow_left(!valid, {left_arg});
         {output}"
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: dslx,
            params,
            result,
            body,
            valid: Some(!mismatch),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::check_struct_type_parameters;

    #[test]
    fn compare_specialized_block_results() -> Result<(), String> {
        for flags in [0, 1, 2, 4, 8, 16, 22] {
            for shapes in [0x00, 0x23, 0x64, 0x88, 0xca] {
                check_struct_type_parameters(&[flags, 128, shapes, 0])?;
                check_struct_type_parameters(&[flags, 7, shapes, 1])?;
            }
        }
        Ok(())
    }
}
