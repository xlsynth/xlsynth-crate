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

/// Generate two instances of one nominal declaration with the same field
/// layout.
fn packet_module(data: &[u8]) -> String {
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
    format!(
        r#"pub struct Packet<A: u32, B: u32> {{ payload: {sign}N[{width_expression}] }}
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
    )
}

/// Compare alias interchangeability, inferred call results and an invalid mix.
fn compare_packets(root: &Path, seed: u64, pass_to_function: bool) -> Result<(), Box<dyn Error>> {
    let paths = [root];
    let options = ImportOptions {
        additional_search_paths: paths.to_vec(),
        ..Default::default()
    };
    let params = "input: facade::Payload, x: u32";
    let result = "(u32, u32, u32)";
    let body = r#"let left: packets::Left = if (x & u32:1) == u32:0 { input } else { packets::make_left(x) };
let right = packets::make_right(x);
(facade::read(left), packets::observe(right), packets::observe(input))"#;
    let bx = format!(
        "use packets; use facade; {}",
        bx_source(params, result, body)
    );
    let native = format!(
        "import packets; import facade; {}",
        dslx_source(params, result, body)
    );
    let evaluator =
        ExpressionEvaluator::compile_block(&bx, &options, compile_bundled_module(&native, &paths))?;
    for args in generate_argument_sets_from_seed(&evaluator.bx.function, seed, 3) {
        evaluator.compare(&args)?;
    }

    let mixed = if pass_to_function {
        "facade::read(packets::make_right(x))"
    } else {
        "let p: facade::Payload = packets::make_right(x); packets::observe(p)"
    };
    let bx = format!(
        "use packets; use facade; {}",
        bx_source("x: u32", "u32", mixed)
    );
    let native = format!(
        "import packets; import facade; {}",
        dslx_source("x: u32", "u32", mixed)
    );
    let mut imports = ImportData::new(None, &paths);
    let dslx = dslx::parse_and_typecheck(&native, "oracle.x", "oracle", &mut imports);
    let checked = check_module_with_options(&parse_module(&bx)?, &options);
    if dslx.is_ok() || checked.is_ok() {
        return Err(format!(
            "mixed specialization accepted in {mixed}: DSLX accepted {}, BX accepted {}",
            dslx.is_ok(),
            checked.is_ok()
        )
        .into());
    }
    Ok(())
}

/// Evaluate a public alias and an unaliased specialization using both
/// frontends.
pub fn check_imported_struct_parametrics(data: &[u8]) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let packets = packet_module(data);
    let facade = r#"import packets;
pub type Payload = packets::Left;
pub fn read(p: Payload) -> u32 { packets::observe(p) }"#;
    fs::write(dir.path().join("packets.x"), &packets).map_err(|e| e.to_string())?;
    fs::write(dir.path().join("facade.x"), facade).map_err(|e| e.to_string())?;
    let pass_to_function = data.first().is_some_and(|flags| flags & 4 != 0);
    compare_packets(dir.path(), input_seed(data), pass_to_function)
        .map_err(|e| format!("{e}\n{packets}\n{facade}"))
}

#[cfg(test)]
mod tests {
    use super::check_imported_struct_parametrics;

    #[test]
    fn replay_same_layout_specializations() -> Result<(), String> {
        check_imported_struct_parametrics(&[])?;
        check_imported_struct_parametrics(&[7, 255, 195, 133])
    }
}
