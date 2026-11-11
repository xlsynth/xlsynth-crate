// SPDX-License-Identifier: Apache-2.0

//! Compare native BX struct specializations with the same declarations in DSLX.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;
use xlsynth_pir::{IrBits, IrFormatPreference};

/// Generate two instances of a packet template and a nested frame template.
fn packet_program(data: &[u8], turbofish: bool) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = u32::from(bytes.next().unwrap() % 96) + 1;
    let width_delta = u32::from(bytes.next().unwrap() % 32) + 1;
    let length = (width & 3) + 1;
    let right_width = if flags & 1 == 0 {
        width + width_delta
    } else {
        width
    };
    let right_length = (length % 4) + 1;
    let sign = if flags & 2 == 0 { "u" } else { "s" };
    let separator = if turbofish { "::" } else { "" };
    let (helper, length_default) = if data.get(4).is_some_and(|flags| flags & 2 != 0) {
        (
            "fn label_count<N: u32>() -> u32 { (N & u32:3) + u32:1 }",
            format!("label_count{separator}<N>()"),
        )
    } else {
        ("", "(N & u32:3) + u32:1".to_owned())
    };
    let (signature_arg, constructor_arg) = if flags & 4 == 0 {
        ("", "u32:")
    } else {
        ("u32:", "")
    };
    let left = if flags & 64 == 0 {
        format!("Packet{separator}<{signature_arg}{width}>")
    } else {
        format!("Packet{separator}<{{BASE_WIDTH + u32:{}}}>", width - 1)
    };
    let right =
        format!("Packet{separator}<{signature_arg}{right_width}, {signature_arg}{right_length}>");
    let left_ctor = match (flags >> 3) & 3 {
        0 => format!("Packet{separator}<{constructor_arg}{width}, {constructor_arg}{length}>"),
        1 => format!("Packet{separator}<{constructor_arg}{width}>"),
        _ => "Packet".to_owned(),
    };
    let mutation = bytes.next().unwrap() % 12;
    let right_ctor = if mutation == 9 {
        format!(
            "Packet{separator}<{constructor_arg}{right_width}, {constructor_arg}{right_length}, u32:1>"
        )
    } else {
        format!(
            "Packet{separator}<{constructor_arg}{right_width}, {constructor_arg}{right_length}>"
        )
    };
    let frame_ctor = if flags & 32 == 0 {
        format!("Frame{separator}<N, M>")
    } else {
        "Frame".to_owned()
    };
    let right_source = if mutation == 8 { "x" } else { "y" };
    let functions = format!(
        r#"
const BASE_WIDTH = u32:1;
{helper}
struct Packet<N: u32, M: u32 = {{{length_default}}}> {{
    data: {sign}N[N],
    labels: u8[M],
}}
struct Frame<N: u32, M: u32> {{
    packets: (Packet{separator}<N, M>[2], bool),
}}
struct Other<N: u32, M: u32> {{
    data: {sign}N[N],
    labels: u8[M],
}}
fn make_left(x: {sign}N[{width}], labels: u8[{length}]) -> {left} {{
    let packet = {left_ctor} {{ labels: labels, data: x }};
    packet
}}
fn invert<N: u32, M: u32>(p: Packet{separator}<N, M>) -> Packet{separator}<N, M> {{
    Packet{separator}<N, M> {{ data: !p.data, labels: p.labels }}
}}
fn pack<N: u32, M: u32>(p: Packet{separator}<N, M>, valid: bool) -> Frame{separator}<N, M> {{
    {frame_ctor} {{ packets: ([p, invert{separator}<N, M>(p)], valid) }}
}}
fn unpack<N: u32, M: u32>(f: Frame{separator}<N, M>) -> (Packet{separator}<N, M>[2], bool) {{
    f.packets
}}
fn inspect<N: u32, M: u32>(p: Packet{separator}<N, M>) -> ({sign}N[N], u8[M], u32, u32) {{
    (p.data, p.labels, N, M)
}}
fn read_left(p: {left}) -> ({sign}N[{width}], u8[{length}], u32, u32) {{
    inspect(p)
}}
"#
    );
    let left_expr = match mutation {
        6 => "right".to_owned(),
        7 => format!("Other{separator}<{width}, {length}> {{ data: x, labels: labels }}"),
        _ => "make_left(x, labels)".to_owned(),
    };
    let params = format!(
        "x: {sign}N[{width}], y: {sign}N[{right_width}], labels: u8[{length}], other: u8[{right_length}], valid: bool"
    );
    let left_result = format!("({sign}N[{width}], u8[{length}], u32, u32)");
    let result = format!(
        "({left_result}, {left_result}, ({sign}N[{right_width}], u8[{right_length}], u32, u32), bool)"
    );
    let body = format!(
        r#"
    let right: {right} = {right_ctor} {{ labels: other, data: {right_source} }};
    let left: {left} = {left_expr};
    let (copies, flag) = unpack(pack(left, valid));
    (read_left(copies[u32:0]), inspect(copies[u32:1]), inspect(right), flag)
"#
    );
    FunctionProgram {
        functions,
        params,
        result,
        body,
        valid: Some(match mutation {
            6 | 7 | 9 => false,
            8 => width == right_width,
            _ => true,
        }),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Check how constructor fields bind an unsigned or signed width in declaration
/// order.
fn literal_program(data: &[u8]) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let magnitude = u32::from(bytes.next().unwrap()) + 1;
    let extra_width = u32::from(bytes.next().unwrap() % 8) + 1;
    let mode = bytes.next().unwrap() % 4;
    let (sign, literal, width) = if flags & 2 == 0 {
        ("u", magnitude.to_string(), magnitude.ilog2() + 1)
    } else {
        let width = u32::BITS - (magnitude - 1).leading_zeros() + 1;
        ("s", format!("-{magnitude}"), width)
    };
    let typed_width = if mode >= 2 {
        width + extra_width
    } else {
        width
    };
    let (first, second, inferred_width) = match mode {
        0 => (literal.as_str(), literal.as_str(), width),
        2 => ("x", literal.as_str(), typed_width),
        _ => (literal.as_str(), "x", width),
    };
    let fields = if flags & 1 == 0 {
        format!("first: {first}, second: {second}")
    } else {
        format!("second: {second}, first: {first}")
    };
    let functions = format!(
        r#"
struct Record<N: u32> {{ first: {sign}N[N], second: {sign}N[N] }}
fn inspect<N: u32>(r: Record<N>) -> ({sign}N[N], {sign}N[N], u32) {{
    (r.first, r.second, N)
}}
"#
    );
    let body = format!(
        r#"
    let r = Record {{ {fields} }};
    let (first, second, width) = inspect(r);
    (first, second, width, x)
"#
    );
    FunctionProgram {
        functions,
        params: format!("x: {sign}N[{typed_width}]"),
        result: format!(
            "({sign}N[{inferred_width}], {sign}N[{inferred_width}], u32, {sign}N[{typed_width}])"
        ),
        body,
        valid: Some(mode != 3),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Bind signedness and a non-u32 dimension from scalar and array fields.
fn signedness_program(data: &[u8]) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 64) + 1;
    let formal_width = usize::from(bytes.next().unwrap() % 122) + 8;
    let mode = bytes.next().unwrap() % 7;
    let signed = flags & 1 != 0;
    let prefix = if signed { "s" } else { "u" };
    let formal_prefix = if flags & 2 == 0 { "u" } else { "s" };
    let dim = format!("{formal_prefix}N[{formal_width}]");
    // In v0.62.0 the dimension in sN[6] carries u32 evidence, whereas s6
    // allows the same width to be inferred at N's declared type.
    let ty = format!("{prefix}{width}");
    let other = if mode == 6 {
        format!("{}{width}", if signed { "u" } else { "s" })
    } else {
        ty.clone()
    };
    let actuals = match mode {
        1 => format!("<{signed}>"),
        2 => format!("<{signed},{dim}:{width}>"),
        3 => format!("<{signed},{width}>"),
        4 => format!("<{}>", !signed),
        5 => format!("<{signed},{dim}:{}>", width + 1),
        _ => String::new(),
    };
    // The pinned DSLX infers u32 widths when specializing a generic
    // function from its fields. A concrete consumer checks the inferred
    // struct arguments against the width's declared type.
    let functions = format!(
        r#"struct Pair<S:bool, N:{dim}> {{ first:xN[S][N], lanes:xN[S][N][2] }}
fn read(p:Pair<{signed},{dim}:{width}>) -> ({ty},bool) {{
    (p.first ^ p.lanes[u32:0], p.first < p.lanes[u32:1])
}}"#
    );
    FunctionProgram {
        functions,
        params: format!("x:{ty}, y:{other}[2]"),
        result: format!("({ty},bool)"),
        body: format!("let p = Pair{actuals} {{ lanes:y, first:x }}; read(p)"),
        valid: Some(mode < 4),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Observe distinct phantom actuals in otherwise identical records.
fn phantom_program(data: &[u8]) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let value = bytes.next().unwrap();
    let mode = bytes.next().unwrap() % 6;
    let (declarations, ty, output, projection, wrong) = if flags & 2 != 0 {
        let alternate = value ^ 0x80;
        (
            format!(
                r#"enum Mode:u8 {{ First={value}, Second={alternate} }}
enum Other:u8 {{ Second={alternate} }}
const FIRST = Mode::First;
const SECOND = Mode::Second;"#
            ),
            "Mode".to_owned(),
            "u8".to_owned(),
            "K as u8",
            "Other::Second".to_owned(),
        )
    } else {
        let prefix = if flags & 1 == 0 { "u" } else { "s" };
        let ty = format!("{prefix}N[{width}]");
        let first = IrBits::from_lsb_fn(width, |i| i + 1 < width && i < 8 && value & (1 << i) != 0);
        let second = first.or(&IrBits::signed_min_value(width));
        let hex = |bits: &IrBits| bits.to_string_fmt(IrFormatPreference::Hex, false);
        (
            format!(
                "const FIRST = {ty}:{};\nconst SECOND = {ty}:{};",
                hex(&first),
                hex(&second)
            ),
            ty.clone(),
            ty,
            "K",
            format!("{prefix}N[{}]:0", width + 1),
        )
    };
    let left_ctor = if flags & 4 != 0 {
        "Packet"
    } else {
        "Packet<FIRST>"
    };
    let (right_ty, right_ctor) = match mode {
        1 => ("FIRST", "Packet".to_owned()),
        2 => ("SECOND", "Packet<{SECOND}>".to_owned()),
        3 => ("FIRST", "Packet<SECOND>".to_owned()),
        4 => ("SECOND", format!("Packet<{wrong}>")),
        _ => ("SECOND", "Packet<SECOND>".to_owned()),
    };
    let read_right = if mode == 5 {
        "read<FIRST>(r)"
    } else {
        "read(r)"
    };
    FunctionProgram {
        functions: format!(
            r#"{declarations}
struct Packet<K:{ty}={{FIRST}}> {{ payload:u8 }}
fn read<K:{ty}>(p:Packet<K>) -> (u8,{output}) {{ (p.payload,{projection}) }}"#
        ),
        params: "x:u8".to_owned(),
        result: format!("((u8,{output}),(u8,{output}),(u8,{output}))"),
        body: format!(
            r#"let p:Packet<FIRST> = {left_ctor} {{ payload:x }};
let r:Packet<{right_ty}> = {right_ctor} {{ payload:!x }};
(read(p),{read_right},read(p))"#
        ),
        valid: Some(mode < 3),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Compare aggregate-valued BX struct specializations to concrete DSLX
/// calculations. The pinned DSLX cannot specialize structs on these types.
fn check_aggregate_actuals(data: &[u8], seed: u64) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let value = bytes.next().unwrap();
    let alternate = value.wrapping_add(1);
    let (declaration, ty, first, second, projection) = match flags % 3 {
        0 => (
            "",
            "(s8,bool)",
            format!("(s8:0x{value:x},true)"),
            format!("(s8:0x{alternate:x},false)"),
            "if k.1 { k.0 as u8 } else { !(k.0 as u8) }",
        ),
        1 => (
            "",
            "u8[2]",
            format!("[u8:{value},u8:{alternate}]"),
            format!("[u8:{alternate},u8:{value}]"),
            "k[u32:0]",
        ),
        _ => (
            "struct Config { offsets:(s8,u8[2]) }",
            "Config",
            format!("Config {{ offsets:(s8:-1,[u8:{value},u8:{alternate}]) }}"),
            format!("Config {{ offsets:(s8:-1,[u8:{alternate},u8:{value}]) }}"),
            "(k.offsets.0 as u8) ^ k.offsets.1[u32:0]",
        ),
    };
    let shared = format!(
        r#"{declaration}
const FIRST = {first};
const SECOND = {second};
fn width(k:{ty}) -> u32 {{ ((({projection}) as u32) & u32:7) + u32:1 }}
const LEFT_WIDTH = width(FIRST);
const RIGHT_WIDTH = width(SECOND);"#
    );
    let left = if flags & 4 != 0 {
        "Packet"
    } else {
        "Packet<FIRST>"
    };
    let body = format!(
        r#"let p = {left} {{ payload:x as uN[LEFT_WIDTH] }};
let r = Packet<SECOND> {{ payload:y as uN[RIGHT_WIDTH] }};
(p.payload as u8,r.payload as u8)"#
    );
    let params = "x:u8,y:u8";
    let result = "(u8,u8)";
    let bx = format!(
        r#"{shared}
struct Packet<K:{ty}={{FIRST}}> {{ payload:uN[width(K)] }}
{}"#,
        bx_source(params, result, &body),
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: shared,
            params: params.to_owned(),
            result: result.to_owned(),
            body: "((x as uN[LEFT_WIDTH]) as u8,(y as uN[RIGHT_WIDTH]) as u8)".to_owned(),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Compare untyped literal spellings at a signed formal's range boundaries.
fn radix_program(data: &[u8]) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let low = bytes.next().unwrap();
    let boundary = bytes.next().unwrap() % 3;
    let bit = |i: usize| {
        (i + 1 < width && i < 8 && low & (1u8 << i) != 0)
            || (boundary == 1 && i + 1 == width)
            || (boundary == 2 && i == width)
    };
    let radix = match flags % 3 {
        0 => IrFormatPreference::UnsignedDecimal,
        1 => IrFormatPreference::Hex,
        _ => IrFormatPreference::Binary,
    };
    let literal = IrBits::from_lsb_fn(width + 1, bit).to_string_fmt(radix, false);
    let tag = IrBits::from_lsb_fn(width, bit).to_string_fmt(IrFormatPreference::Hex, false);
    let (annotation, constructor) = match (flags >> 2) % 3 {
        0 => (literal.as_str(), "TAG"),
        1 => ("TAG", literal.as_str()),
        _ => (literal.as_str(), literal.as_str()),
    };
    FunctionProgram {
        functions: format!(
            r#"const TAG = sN[{width}]:{tag};
struct Packet<K:sN[{width}]> {{ payload:u8 }}
fn read<K:sN[{width}]>(p:Packet<K>) -> (u8,sN[{width}]) {{ (p.payload,K) }}"#
        ),
        params: "x:u8".to_owned(),
        result: format!("(u8,sN[{width}])"),
        body: format!(
            "let p:Packet<{annotation}> = Packet<{constructor}> {{ payload:x }}; read(p)"
        ),
        valid: Some(
            boundary == 0 || (boundary == 1 && radix != IrFormatPreference::UnsignedDecimal),
        ),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Forward a struct-bound integer directly to a nested record whose formal
/// can differ in signedness and width.
fn check_forwarded_actuals(data: &[u8], seed: u64) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let source_bool = flags & 4 != 0;
    let target_bool = flags & 8 != 0;
    let source_signed = !source_bool && flags & 1 != 0;
    let target_signed = !target_bool && flags & 2 != 0;
    let dimension = |width: u8, is_bool| {
        if is_bool {
            1
        } else {
            usize::from(width % 129) + 1
        }
    };
    let spelling = |width, signed, is_bool| {
        if is_bool {
            "bool".to_owned()
        } else {
            format!("{}N[{width}]", if signed { "s" } else { "u" })
        }
    };
    let source_width = dimension(bytes.next().unwrap(), source_bool);
    let target_width = dimension(bytes.next().unwrap(), target_bool);
    let source = spelling(source_width, source_signed, source_bool);
    let target = spelling(target_width, target_signed, target_bool);
    let sample = bytes.next().unwrap();
    let pattern = bytes.next().unwrap() % 4;
    let pivot = usize::from(sample) % source_width;
    let source_bit = |i: usize| match pattern {
        0 => i < 8 && sample & (1u8 << i) != 0,
        1 => i == pivot,
        2 => i >= pivot,
        _ => sample & (1u8 << (i % 8)) != 0,
    };
    let value = IrBits::from_lsb_fn(source_width, source_bit);
    let negative = source_signed && source_bit(source_width - 1);
    let significant_bits = target_width - usize::from(target_signed);
    let fits = (!negative || target_signed)
        && (significant_bits..source_width).all(|i| source_bit(i) == negative);
    let literal = value.to_string_fmt(IrFormatPreference::Hex, false);
    let declarations = |destination: &str| {
        format!(
            r#"struct Inner<P:{target}> {{ payload:u8 }}
struct Outer<P:{source}> {{ child:Inner<P> }}
const SOURCE = {source}:{literal};
const DEST = {destination};
fn inspect<P:{target}>(c:Inner<P>) -> (u8,{target}) {{ (c.payload,P) }}"#
        )
    };
    let mut program = FunctionProgram {
        functions: declarations(&format!("SOURCE as {target}")),
        params: "x:u8".to_owned(),
        result: format!("(u8,{target})"),
        body: r#"let child = Inner<DEST> { payload:x };
let outer = Outer<SOURCE> { child:child };
inspect(outer.child)"#
            .to_owned(),
        valid: Some(fits),
        dslx_ir: true,
        dynamic_index: None,
    };
    // v0.62.0 forwards wide nonnegative values as hexadecimal. It accepts a
    // signed target with its high bit set even though that changes the value.
    // Require a constexpr checked_cast in the reference for this boundary.
    if source_width > 64
        && target_signed
        && !negative
        && !fits
        && (target_width..source_width).all(|i| !source_bit(i))
    {
        let bx = format!(
            "{}\n{}",
            program.functions,
            bx_source(&program.params, &program.result, &program.body)
        );
        program.functions = declarations(&format!("checked_cast<{target}>(SOURCE)"));
        return compare_bx_program(
            &bx,
            program,
            seed,
            DslxExecution::ConcreteConstants,
            Vec::new(),
            &[],
        );
    }
    compare_program(program, seed, DslxExecution::ConcreteConstants, Vec::new())
}

/// Compare source acceptance, lowered IR, and concrete calls of each frontend.
pub fn check_struct_parametrics(data: &[u8]) -> Result<(), String> {
    let seed = input_seed(data);
    let program = match data.first().copied().unwrap_or(0) >> 5 {
        4 => literal_program(data),
        5 => signedness_program(data),
        6 if data[0] & 16 != 0 => radix_program(data),
        6 => phantom_program(data),
        7 if data[0] & 16 != 0 => return check_forwarded_actuals(data, seed),
        7 => return check_aggregate_actuals(data, seed),
        _ => {
            let program = packet_program(data, false);
            if data.get(4).is_some_and(|flags| flags & 1 != 0) {
                let bx = packet_program(data, true);
                let source = format!(
                    "{}\n{}",
                    bx.functions,
                    bx_source(&bx.params, &bx.result, &bx.body)
                );
                return compare_bx_program(
                    &source,
                    program,
                    seed,
                    DslxExecution::ConcreteConstants,
                    Vec::new(),
                    &[],
                );
            }
            program
        }
    };
    compare_program(program, seed, DslxExecution::ConcreteConstants, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::check_struct_parametrics;

    #[test]
    fn replay_native_parametric_structs() -> Result<(), String> {
        for flags in [0, 7, 8, 15, 16, 23, 32, 55, 64, 119] {
            for mutation in [0, 6, 7, 8, 9] {
                for mode in [0, 1, 2, 3] {
                    check_struct_parametrics(&[flags, 70, 16, mutation, mode])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_struct_literal_field_inference() -> Result<(), String> {
        for flags in 128..132 {
            for magnitude in [0, 130] {
                for mode in 0..4 {
                    check_struct_parametrics(&[flags, magnitude, 3, mode])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_inferred_signedness_and_width() -> Result<(), String> {
        for flags in 160..164 {
            for width in [0, 8, 63] {
                for mode in 0..7 {
                    check_struct_parametrics(&[flags, width, 0xa7, mode])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_phantom_struct_arguments() -> Result<(), String> {
        for flags in [192, 193, 194, 196, 197, 198] {
            for width in [0, 8, 95, 128] {
                for mode in 0..6 {
                    check_struct_parametrics(&[flags, width, 0xa7, mode])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_signed_struct_argument_radices() -> Result<(), String> {
        for flags in 208..224 {
            for width in [0, 2, 63, 64, 128] {
                for boundary in 0..3 {
                    check_struct_parametrics(&[flags, width, 0xa7, boundary])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_aggregate_struct_arguments() -> Result<(), String> {
        for flags in 224..232 {
            check_struct_parametrics(&[flags, 0x9a])?;
        }
        Ok(())
    }

    #[test]
    fn replay_forwarded_scalar_arguments() -> Result<(), String> {
        for flags in 240..=255 {
            for (source, target, sample) in [
                (7, 2, 3),
                (7, 0, 1),
                (63, 2, 4),
                (64, 2, 4),
                (95, 64, 64),
                (72, 90, 72),
            ] {
                for pattern in 0..4 {
                    check_struct_parametrics(&[flags, source, target, sample, pattern])?;
                }
            }
        }
        Ok(())
    }
}
