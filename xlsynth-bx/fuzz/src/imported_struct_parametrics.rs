// SPDX-License-Identifier: Apache-2.0

//! Compare nominal DSLX struct specializations across the BX import boundary.

use std::error::Error;
use std::fs;
use std::path::Path;

use xlsynth::dslx::{self, ImportData};
use xlsynth_bx::{ImportOptions, check_module_with_options, parse_module};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

use crate::eval::expr_eval::{ExpressionEvaluator, bx_source, compile_bundled_module, dslx_source};
use crate::input_seed;

/// A DSLX declaration and the two concrete spellings exercised by the caller.
struct PacketCase {
    module: String,
    left_args: String,
    right_args: String,
    member_import: bool,
    alias_input: bool,
    pass_to_function: bool,
}

/// Generate distinct actuals with the same field layout and vary how BX names
/// them.
fn packet_case(data: &[u8]) -> PacketCase {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let mut byte = || bytes.next().unwrap();
    let flags = byte();
    let a = 1 + u32::from(byte() % 96);
    let b = 2 + u32::from(byte());
    let delta = 1 + u32::from(byte()) % (b - 1);
    let (width_expression, concrete_width, right_a, right_b) = if flags & 1 == 0 {
        // B is a phantom parameter. The consumer observes it explicitly.
        ("A", a, a, b + delta)
    } else {
        // A different pair yields exactly the same value of A + B.
        ("A + B", a + b, a + delta, b - delta)
    };
    let sign = if flags & 2 == 0 { "u" } else { "s" };
    let typed = if flags & 8 == 0 { "" } else { "u32:" };
    let (default, left_args) = if flags & 16 == 0 {
        (String::new(), format!("{typed}{a}, {typed}{b}"))
    } else {
        (format!(" = {{u32:{b}}}"), format!("{typed}{a}"))
    };
    let right_args = format!("{typed}{right_a}, {typed}{right_b}");
    let module = format!(
        r#"pub struct Packet<A: u32, B: u32{default}> {{ payload: {sign}N[{width_expression}] }}
pub type Left = Packet<u32:{a}, u32:{b}>;
pub fn make_left(x: u32) -> Left {{
    Left {{ payload: x as {sign}N[{concrete_width}] }}
}}
pub fn make_right(x: u32) -> Packet<u32:{right_a}, u32:{right_b}> {{
    Packet<u32:{right_a}, u32:{right_b}> {{ payload: x as {sign}N[{concrete_width}] }}
}}
pub fn observe<A: u32, B: u32>(p: Packet<A, B>) -> u32 {{
    (p.payload as u32) ^ ((A << u32:16) | B)
}}"#
    );
    PacketCase {
        module,
        left_args,
        right_args,
        member_import: flags & 32 != 0,
        alias_input: flags & 64 != 0,
        pass_to_function: flags & 4 != 0,
    }
}

/// Compare direct constructions, alias interchangeability, inferred call
/// results and an invalid mix.
fn compare_packets(root: &Path, seed: u64, case: &PacketCase) -> Result<(), Box<dyn Error>> {
    let paths = [root];
    let options = ImportOptions {
        additional_search_paths: paths.to_vec(),
        ..Default::default()
    };
    let left = format!("packets::Packet<{}>", case.left_args);
    let right = format!("packets::Packet<{}>", case.right_args);
    let input = if case.alias_input {
        "facade::Payload"
    } else {
        &left
    };
    let params = format!("input: {input}, x: u32");
    let result = "(u32, u32, u32)";
    let helper =
        format!("fn copy(p: packets::Left) -> {left} {{ {left} {{ payload: p.payload }} }}");
    let body = format!(
        r#"let left: {left} = if (x & u32:1) == u32:0 {{ input }} else {{ packets::make_left(x) }};
let right: {right} = packets::make_right(x);
let other = {right} {{ payload: right.payload }};
(facade::read(copy(left)), packets::observe(other), packets::observe(input))"#
    );
    // The pinned DSLX parser accepts a use-bound type annotation but requires
    // a qualified name to recognize an imported struct construction.
    let bx_program = |declarations: String| {
        if case.member_import {
            format!(
                "use packets; use packets::Packet; use facade;\n{}",
                declarations.replace("packets::Packet<", "Packet<")
            )
        } else {
            format!("use packets; use facade;\n{declarations}")
        }
    };
    let native_imports = "import packets; import facade;";
    let bx = bx_program(format!("{helper}\n{}", bx_source(&params, result, &body)));
    let native = format!(
        "{native_imports}\n{helper}\n{}",
        dslx_source(&params, result, &body)
    );
    let evaluator =
        ExpressionEvaluator::compile_block(&bx, &options, compile_bundled_module(&native, &paths))?;
    for args in generate_argument_sets_from_seed(&evaluator.bx.function, seed, 3) {
        evaluator.compare(&args)?;
    }

    let mixed = if case.pass_to_function {
        "facade::read(packets::make_right(x))".to_string()
    } else {
        format!("let p: {left} = packets::make_right(x); packets::observe(p)")
    };
    let bx = bx_program(bx_source("x: u32", "u32", &mixed));
    let native = format!("{native_imports}\n{}", dslx_source("x: u32", "u32", &mixed));
    let mut imports = ImportData::new(None, &paths);
    let dslx = dslx::parse_and_typecheck(&native, "oracle.x", "oracle", &mut imports);
    let checked = check_module_with_options(&parse_module(&bx)?, &options);
    if dslx.is_ok() || checked.is_ok() {
        return Err(format!(
            "mixed specialization accepted in {bx}: DSLX accepted {}, BX accepted {}",
            dslx.is_ok(),
            checked.is_ok()
        )
        .into());
    }
    Ok(())
}

/// Compare qualified or use-bound specializations with native DSLX.
pub fn check_imported_struct_parametrics(data: &[u8]) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let case = packet_case(data);
    let facade = r#"import packets;
pub type Payload = packets::Left;
pub fn read(p: Payload) -> u32 { packets::observe(p) }"#;
    fs::write(dir.path().join("packets.x"), &case.module).map_err(|e| e.to_string())?;
    fs::write(dir.path().join("facade.x"), facade).map_err(|e| e.to_string())?;
    compare_packets(dir.path(), input_seed(data), &case)
        .map_err(|e| format!("{e}\n{}\n{facade}", case.module))
}

#[cfg(test)]
mod tests {
    use super::check_imported_struct_parametrics;

    #[test]
    fn replay_same_layout_specializations() -> Result<(), String> {
        check_imported_struct_parametrics(&[])?;
        check_imported_struct_parametrics(&[7, 255, 195, 133])?;
        check_imported_struct_parametrics(&[120, 22, 15, 0])
    }
}
