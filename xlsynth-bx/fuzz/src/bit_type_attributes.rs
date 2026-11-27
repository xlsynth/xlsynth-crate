// SPDX-License-Identifier: Apache-2.0

//! Compare bit-type attributes and qualified member lookup with native DSLX.

use std::fs;

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;

/// Check the same exported scalar alias through module and member imports
/// from BX or DSLX. Native DSLX uses its qualified import spelling.
fn compare_import(
    mut program: FunctionProgram,
    concrete: &str,
    attr: &str,
    flags: u8,
    seed: u64,
) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let library = format!("type Field = {concrete};\npub type Lane = Field;");
    fs::write(dir.path().join("native.x"), &library).map_err(|e| e.to_string())?;
    fs::write(dir.path().join("local.bx"), &library).map_err(|e| e.to_string())?;
    let module = if flags & 8 == 0 { "native" } else { "local" };
    let (import, receiver) = if flags & 16 == 0 {
        (format!("use {module};"), format!("{module}::Lane"))
    } else {
        (format!("use {module}::Lane;"), "Lane".to_string())
    };
    let bx = format!(
        "{import}\nconst BOUND = {receiver}::{attr};\n{}",
        bx_source(&program.params, &program.result, &program.body)
    );
    program.functions = format!("import native;\nconst BOUND = native::Lane::{attr};");
    compare_bx_program(
        &bx,
        program,
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[dir.path()],
    )
    .map_err(|e| format!("{e}\nimported module:\n{library}"))
}

/// Vary the receiver's defining scope, underlying type and specialization.
/// Compare inferred constant types and values, native executions and both
/// lowered results.
pub fn check_bit_type_attributes(data: &[u8]) -> Result<(), String> {
    let [family, selector, width, flags] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let family = family % 8;
    let attr = if flags & 128 == 0 {
        ["MIN", "MAX", "ZERO"][usize::from(selector % 3)]
    } else {
        "MISSING"
    };
    let boolean = flags & 4 != 0 && family != 0;
    let signed = !boolean && flags & 1 != 0;
    let width = if family == 0 {
        1 + usize::from(width % 64)
    } else if boolean {
        1
    } else {
        usize::from(width)
    };
    let sign = if signed { 's' } else { 'u' };
    let concrete = if boolean {
        "bool".into()
    } else {
        format!("{sign}N[{width}]")
    };
    let target = if boolean {
        "bool".into()
    } else if flags & 32 != 0 {
        "xN[SIGNED][WIDTH]".into()
    } else if !signed && flags & 2 != 0 {
        "bits[WIDTH]".into()
    } else {
        format!("{sign}N[WIDTH]")
    };
    let aliases = format!(
        "const WIDTH = u32:{width}; const SIGNED = {signed}; type Storage = {target}; type Lane = Storage;"
    );
    let mut program = FunctionProgram {
        functions: String::new(),
        params: format!("x: {concrete}"),
        result: concrete.clone(),
        body: "x ^ BOUND".into(),
        valid: Some(attr != "MISSING"),
        dslx_ir: true,
        dynamic_index: None,
    };
    match family {
        0 => program.functions = format!("const BOUND = {sign}{width}::{attr};"),
        1 => program.functions = format!("{aliases}\nconst BOUND = Lane::{attr};"),
        2 => {
            program.functions = format!(
                "{aliases}\nfn read(x: Lane) -> Lane {{ type Local = Lane; const BOUND = Local::{attr}; x ^ BOUND }}"
            );
            program.body = "read(x)".into();
        }
        3 => {
            // The second instantiation changes both width and signedness.
            let other = format!("{}N[{}]", if signed { 'u' } else { 's' }, width + 1);
            program.functions = format!(
                "fn bound<S: bool, N: u32>(x: xN[S][N]) -> xN[S][N] {{ type Lane = xN[S][N]; x ^ Lane::{attr} }}"
            );
            program.params = format!("x: {concrete}, y: {other}");
            program.result = format!("({concrete}, {other})");
            program.body = "(bound(x), bound(y))".into();
        }
        4 => return compare_import(program, &concrete, attr, flags, input_seed(data)),
        5 => {
            // Endpoints named by aliases are also scalar match patterns. A
            // call with the boundary itself exercises the matching arm at
            // widths where a uniformly random argument will rarely do so.
            program.functions = format!(
                "{aliases}\nconst BOUND = Lane::{attr};\nfn is_bound(x: Lane) -> bool {{ match x {{ Lane::{attr} => true, _ => false }} }}"
            );
            program.result = "(bool, bool)".into();
            program.body = "(is_bound(BOUND), is_bound(x))".into();
        }
        6 => {
            // An enum can define members with the same names and different
            // values. Its alias must preserve the enum's member lookup.
            program.functions =
                "enum Mode: u4 { MIN = 5, MAX = 6, ZERO = 7 } type Lane = Mode;".into();
            program.params = "x: u4".into();
            program.result = "u4".into();
            program.body = format!("x ^ (Lane::{attr} as u4)");
        }
        _ => {
            let body = match selector % 8 {
                0 => format!("type Lane = {concrete}[2]; Lane::{attr}"),
                1 => format!("type Lane = ({concrete}, bool); Lane::{attr}"),
                2 => {
                    program.functions = "struct Lane { field: bool }".into();
                    format!("Lane::{attr}")
                }
                3 => {
                    program.functions = "enum Lane: u4 { Only = 0 }".into();
                    format!("Lane::{attr}")
                }
                4 => format!("type Lane = {concrete}; let Lane = x; Lane::{attr}"),
                5 => format!("bool::{attr}"),
                6 => format!("{sign}N[{width}]::{attr}"),
                _ => format!("xN[{signed}][{width}]::{attr}"),
            };
            program.functions.push_str(&format!(
                "\nfn rejected(x: {concrete}) -> {concrete} {{ {body} }}"
            ));
            program.body = "rejected(x)".into();
            program.valid = Some(false);
        }
    }
    compare_program(
        program,
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_bit_type_attributes;

    #[test]
    fn compare_bit_type_attributes_with_native_dslx() -> Result<(), String> {
        for family in 0..8 {
            for selector in 0..8 {
                for (width, flags) in [(0, 0), (0, 1), (1, 4), (1, 9), (65, 18), (129, 57)] {
                    check_bit_type_attributes(&[family, selector, width, flags])?;
                }
                check_bit_type_attributes(&[family, selector, 17, 128])?;
            }
        }
        Ok(())
    }
}
