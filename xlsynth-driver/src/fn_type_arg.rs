// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

pub fn parse_type_text(text: &str) -> Result<ir::Type, String> {
    let mut parser = ir_parser::Parser::new(text);
    parser
        .parse_type()
        .map_err(|e| format!("failed to parse type `{text}`: {e}"))
}

/// Parses a CLI function-type string of the form
/// `<param_tuple_type> -> <return_type>`.
pub fn parse_function_type_text(text: &str) -> Result<ir::FunctionType, String> {
    let (params_text, ret_text) = text
        .split_once("->")
        .ok_or_else(|| format!("expected `<param_tuple_type> -> <return_type>`, got `{text}`"))?;
    let params_ty = parse_type_text(params_text.trim())?;
    let ret_ty = parse_type_text(ret_text.trim())?;
    let ir::Type::Tuple(param_members) = params_ty else {
        return Err(format!(
            "expected parameter side to be a tuple type, got `{}`",
            params_ty
        ));
    };
    let param_types = param_members
        .into_iter()
        .map(|boxed| *boxed)
        .collect::<Vec<ir::Type>>();
    Ok(ir::FunctionType {
        param_types,
        return_type: ret_ty,
    })
}
