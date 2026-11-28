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
    make_right_call: String,
    payload_width: u32,
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
    let right_type = format!("Packet<u32:{right_a}, u32:{right_b}>");
    let (update_helpers, make_right) = if byte() & 1 == 0 {
        (
            String::new(),
            format!(
                "{right_type} {{ payload: x as {sign}N[{concrete_width}], stamp: x ^ u32:{right_b} }}"
            ),
        )
    } else {
        let (binding, base) = if flags & 128 == 0 {
            ("let base = seed<A, B>(x);\n    ", "base")
        } else {
            ("", "seed<A, B>(x)")
        };
        let helpers = format!(
            r#"fn seed<A: u32, B: u32>(x: u32) -> Packet<A, B> {{
    Packet<A, B> {{ payload: (!x) as {sign}N[{width_expression}], stamp: x ^ B }}
}}
fn update<A: u32, B: u32>(x: u32) -> Packet<A, B> {{
    {binding}Packet<A, B> {{ payload: x as {sign}N[{width_expression}], ..{base} }}
}}
"#
        );
        (helpers, format!("update<u32:{right_a}, u32:{right_b}>(x)"))
    };
    let module = format!(
        r#"pub struct Packet<A: u32, B: u32{default}> {{ payload: {sign}N[{width_expression}], stamp: u32 }}
pub type Left = Packet<u32:{a}, u32:{b}>;
pub fn make_left(x: u32) -> Left {{
    Left {{ payload: x as {sign}N[{concrete_width}], stamp: x }}
}}
{update_helpers}pub fn make_right(x: u32) -> {right_type} {{
    {make_right}
}}
pub fn observe<A: u32, B: u32>(p: Packet<A, B>) -> u32 {{
    (p.payload as u32) ^ ((A << u32:16) | B)
}}"#
    );
    PacketCase {
        module,
        left_args,
        right_args,
        make_right_call: "packets::make_right(x)".into(),
        payload_width: concrete_width,
        member_import: flags & 32 != 0,
        alias_input: flags & 64 != 0,
        pass_to_function: flags & 4 != 0,
    }
}

/// Vary a Boolean that selects the field signedness or only the nominal
/// specialization. A native call infers both values from its scalar input.
fn boolean_packet_case(data: &[u8]) -> PacketCase {
    let flags = data[0];
    let width = 1 + u32::from(data[1] % 129);
    let phantom = flags & 1 != 0;
    let left_signed = flags & 2 != 0;
    let right_signed = !left_signed;
    let width_first = flags & 128 != 0;
    let has_default = flags & 16 != 0;
    let args = |signed: &str, width: &str| {
        if width_first {
            format!("{width}, {signed}")
        } else {
            format!("{signed}, {width}")
        }
    };
    let spelling = |signed: bool| match data[2] % 4 {
        0 => signed.to_string(),
        1 => format!("bool:{}", u8::from(signed)),
        2 => format!("u1:{}", u8::from(signed)),
        _ => u8::from(signed).to_string(),
    };
    let width_arg = if flags & 8 == 0 {
        width.to_string()
    } else {
        format!("u32:{width}")
    };
    let mut formals = args("S: bool", "N: u32");
    let left_args = if has_default {
        if width_first {
            formals.push_str(&format!(" = {{{left_signed}}}"));
            width_arg.clone()
        } else {
            formals.push_str(&format!(" = {{u32:{width}}}"));
            spelling(left_signed)
        }
    } else {
        args(&spelling(left_signed), &width_arg)
    };
    let right_args = args(&spelling(right_signed), &width_arg);
    let left_type_args = args(&left_signed.to_string(), &format!("u32:{width}"));
    let symbolic_args = args("S", "N");
    let field_type = if phantom { "uN[N]" } else { "xN[S][N]" };
    let sign = |signed| if signed { "s" } else { "u" };
    let left_type = format!("{}N[{width}]", sign(!phantom && left_signed));
    let right_scalar = format!("{}N[{width}]", sign(right_signed));
    // Shifting before inversion sets the sign bit for every generated width.
    let high_bit_set = format!("!((x as uN[{width}]) >> u32:1)");
    let module = format!(
        r#"pub struct Packet<{formals}> {{ payload: {field_type}, stamp: u32 }}
pub type Left = Packet<{left_type_args}>;
pub fn make_left(x: u32) -> Left {{
    Left {{ payload: ({high_bit_set}) as {left_type}, stamp: x }}
}}
pub fn make_right<S: bool, N: u32>(v: xN[S][N], stamp: u32) -> Packet<{symbolic_args}> {{
    Packet<{symbolic_args}> {{ payload: v as {field_type}, stamp: stamp }}
}}
pub fn observe<S: bool, N: u32>(p: Packet<{symbolic_args}>) -> u32 {{
    (p.payload as u32) ^ ((N << u32:1) | (S as u32))
}}"#
    );
    PacketCase {
        module,
        left_args,
        right_args,
        make_right_call: format!("packets::make_right(({high_bit_set}) as {right_scalar}, x)"),
        payload_width: width,
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
    let widened = format!("uN[{}]", case.payload_width + 1);
    let result = format!("(u32, u32, u32, u32, {widened}, {widened})");
    let make_right = &case.make_right_call;
    let helper = format!(
        "fn copy(p: packets::Left) -> {left} {{ {left} {{ payload: p.payload, stamp: p.stamp }} }}"
    );
    let body = format!(
        r#"let left: {left} = if (x & u32:1) == u32:0 {{ input }} else {{ packets::make_left(x) }};
let inferred = {make_right};
let right: {right} = inferred;
let other = {right} {{ payload: right.payload, stamp: right.stamp }};
(facade::read(copy(left)), packets::observe(other), packets::observe(input), other.stamp,
 left.payload as {widened}, inferred.payload as {widened})"#
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
    let bx = bx_program(format!("{helper}\n{}", bx_source(&params, &result, &body)));
    let native = format!(
        "{native_imports}\n{helper}\n{}",
        dslx_source(&params, &result, &body)
    );
    let evaluator =
        ExpressionEvaluator::compile_block(&bx, &options, compile_bundled_module(&native, &paths))?;
    for args in generate_argument_sets_from_seed(&evaluator.bx.function, seed, 3) {
        evaluator.compare(&args)?;
    }

    let mixed = if case.pass_to_function {
        format!("facade::read({make_right})")
    } else {
        // Right supplies both actuals. A native let annotation with an
        // omitted trailing actual can infer it from the initializer.
        format!("let p: {right} = packets::make_left(x); packets::observe(p)")
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
    let case = if data.get(4).is_some_and(|mode| mode & 2 != 0) {
        boolean_packet_case(data)
    } else {
        packet_case(data)
    };
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
        check_imported_struct_parametrics(&[135, 255, 195, 133, 1])?;
        check_imported_struct_parametrics(&[120, 22, 15, 0, 1])
    }

    #[test]
    fn replay_boolean_specializations() -> Result<(), String> {
        for (flags, width_byte, spelling) in [
            (0, 0, 0),             // xN[false][1] and xN[true][1].
            (0b1111_1110, 128, 1), // Boolean default, alias input and 129-bit fields.
            (0b0111_1001, 63, 2),  // Width default and phantom Boolean.
            (0b1000_0111, 64, 3),  // Inferred phantom in a rejected function call.
        ] {
            check_imported_struct_parametrics(&[flags, width_byte, spelling, 0, 2])?;
        }
        Ok(())
    }
}
