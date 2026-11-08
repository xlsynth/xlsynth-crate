// SPDX-License-Identifier: Apache-2.0

//! Compare module constant dependencies, lexical lookup, and computed values.

use std::fmt::Write;

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

/// Generate dependent scalar and aggregate declarations, then read them from
/// a block or a local function on the same inputs as native DSLX.
pub fn check_module_constants(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let consumer = bytes.next().unwrap();
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let mutation = bytes.next().unwrap() % 22;
    let count = usize::from(bytes.next().unwrap() % 5) + 1;
    let ty = format!("{}N[{width}]", if flags & 1 == 0 { 'u' } else { 's' });
    let entropy = data
        .get(5 + count..)
        .filter(|b| !b.is_empty())
        .unwrap_or(data);
    let base = IrBits::from_lsb_fn(width, |i| {
        !entropy.is_empty() && entropy[(i / 8) % entropy.len()] & (1 << (i % 8)) != 0
    });
    let radix = match (flags >> 1) % 3 {
        0 => IrFormatPreference::Hex,
        1 => IrFormatPreference::Binary,
        _ if flags & 1 == 0 => IrFormatPreference::UnsignedDecimal,
        _ => IrFormatPreference::SignedDecimal,
    };
    let digits = base.to_string_fmt(radix, false);
    let public = if flags & 8 != 0 { "pub " } else { "" };
    let mut declarations = if flags & 16 == 0 {
        format!("{public}const C0: {ty} = {digits};\n")
    } else {
        format!("{public}const C0 = {ty}:{digits};\n")
    };
    writeln!(
        declarations,
        "fn combine(a: {ty}, b: {ty}) -> {ty} {{ (a - b) ^ C0 }}"
    )
    .unwrap();
    // The pinned DSLX converter evaluates a counted fold in a concrete call,
    // but lacks constexpr metadata for a for-expression directly on a const
    // RHS.
    writeln!(
        declarations,
        "fn fold(a: {ty}, b: {ty}) -> {ty} {{ for (item, acc) in [a, b] {{ item - acc }}(C0) }}"
    )
    .unwrap();
    for i in 1..=count {
        let choice = bytes.next().unwrap();
        let prev = format!("C{}", i - 1);
        let other = format!("C{}", usize::from(choice >> 4) % i);
        let value = match choice % 8 {
            0 => format!("{prev} + {other}"),
            1 => format!("!{prev}"),
            2 => format!("combine({prev}, {other})"),
            3 => format!("{{ let C0 = {prev} - {other}; C0 ^ {other} }}"),
            4 => format!("if {prev} < {other} {{ {prev} }} else {{ {other} }}"),
            5 => format!("({prev}, {other}).1"),
            6 => format!("map([{prev}, {other}], |item| combine(item, C0))[u32:1]"),
            _ => format!("fold({prev}, {other})"),
        };
        let annotation = if choice & 8 == 0 {
            String::new()
        } else {
            format!(": {ty}")
        };
        writeln!(declarations, "const C{i}{annotation} = {value};").unwrap();
    }
    let last = format!("C{count}");
    let array = format!("{ty}[2]");
    let pair = format!("({ty}, {array})");
    writeln!(
        declarations,
        "const WORDS: {array} = [C0, {last}];\nconst PAIR = ({last}, WORDS);"
    )
    .unwrap();
    let comparison = if flags & 64 == 0 {
        format!("C0 < {last}")
    } else {
        format!("{count} < {}", flags % 7)
    };
    writeln!(declarations, "const LESS = {comparison};").unwrap();
    let (result, body) = match consumer % 8 {
        0 => (
            ty.clone(),
            format!("if LESS {{ x ^ {last} }} else {{ x - {last} }}"),
        ),
        1 => (pair.clone(), "PAIR".into()),
        2 => (
            ty.clone(),
            "let C0 = C0 + x; if flag { C0 } else { PAIR.0 }".into(),
        ),
        3 => {
            writeln!(
                declarations,
                "fn shadow(C0: {ty}) -> {ty} {{ C0 - {last} }}"
            )
            .unwrap();
            (ty.clone(), "shadow(x) + C0".into())
        }
        4 => (array, "map(WORDS, |item| item ^ C0)".into()),
        5 => (
            ty.clone(),
            format!("match x {{ C0 => {last}, _ => !{last} }}"),
        ),
        6 => {
            writeln!(declarations, "const COUNT = u32:{};", flags % 5).unwrap();
            (
                ty.clone(),
                format!("for (i, acc) in u32:0..COUNT {{ acc + ((i as {ty}) ^ {last}) }}(x)"),
            )
        }
        _ => {
            let slice_width = width.div_ceil(2);
            writeln!(declarations, "const WIDTH: u32 = {slice_width};").unwrap();
            (
                ty.clone(),
                format!("((x as uN[{width}])[0+:uN[WIDTH]] as {ty}) ^ {last}"),
            )
        }
    };
    // An erroneous unused declaration must still be rejected. Insert forward
    // references before their definitions so the dependency graph tests order.
    match mutation {
        12 => declarations.insert_str(0, "const BAD = C0;\n"),
        13 => declarations.insert_str(0, &format!("const BAD = combine({ty}:0, {ty}:0);\n")),
        14 => declarations.insert_str(0, &format!("fn early() -> {ty} {{ C0 }}\n")),
        15 => declarations.push_str("const C0 = C1;\n"),
        16 => declarations.push_str("const combine = C1;\n"),
        17 => declarations.push_str("const BAD = BAD;\n"),
        18 => writeln!(declarations, "const BAD: uN[{}] = C0;", width + 1).unwrap(),
        19 => declarations.push_str("const BAD = 1 + 2;\n"),
        20 => {
            let builtins = [
                "map",
                "zero!",
                "bit_count",
                "signex",
                "trace_fmt!",
                "xor_reduce",
                "Source",
                "IOResult",
            ];
            let builtin = builtins[usize::from(flags) % builtins.len()];
            writeln!(declarations, "const {builtin} = C0;").unwrap();
        }
        21 => declarations.push_str("const BAD = 0 || 1;\n"),
        _ => { /* Keep the generated declarations well-typed. */ }
    }
    let params = format!("x: {ty}, flag: bool");
    let body = if flags & 32 == 0 {
        body
    } else {
        writeln!(declarations, "fn read({params}) -> {result} {{ {body} }}").unwrap();
        "read(x, flag)".into()
    };
    // Always take both sides of the named constant pattern, including at
    // widths where the oracle's random arguments are unlikely to equal C0.
    let inputs = if consumer % 8 == 5 {
        vec![
            vec![
                IrValue::Bits(base.clone()),
                IrValue::make_ubits(1, 0).unwrap(),
            ],
            vec![
                IrValue::Bits(base.not()),
                IrValue::make_ubits(1, 1).unwrap(),
            ],
        ]
    } else {
        Vec::new()
    };
    compare_program(
        FunctionProgram {
            functions: declarations,
            params,
            result,
            body,
            valid: Some(mutation < 12),
            dslx_ir: true,
            dynamic_index: None,
        },
        crate::input_seed(data),
        DslxExecution::ConcreteConstants,
        inputs,
    )
}

#[cfg(test)]
mod tests {
    use super::check_module_constants;
    use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
    use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
    use xlsynth_bx::{check_module, parse_module};
    use xlsynth_pir::IrValue;
    use xlsynth_pir::libxls_bridge::value_to_libxls;

    #[test]
    fn compare_constant_declarations_and_uses() -> Result<(), String> {
        for consumer in 0..8 {
            for flags in [0, 3, 10, 21, 35, 60, 64, 111] {
                check_module_constants(&[consumer, flags, 96, 0, 4, 2, 11, 22, 31, 19, 0xf7])?;
            }
        }
        for mutation in 12..22 {
            check_module_constants(&[0, 0, 7, mutation])?;
        }
        for builtin in 1..8 {
            check_module_constants(&[0, builtin, 7, 20])?;
        }
        Ok(())
    }

    #[test]
    fn zero_trip_unroll_resolves_the_callee_name_before_checking_its_kind() -> Result<(), String> {
        for count in [0, 1] {
            let declarations = format!(
                r#"const V = u8:1;
const C = unroll_for! (i, a) in u32:0..u32:{count} {{ V(a) }}(u8:1);"#
            );
            compare_program(
                FunctionProgram {
                    functions: declarations,
                    params: String::new(),
                    result: "u8".into(),
                    body: "C".into(),
                    valid: Some(count == 0),
                    dslx_ir: true,
                    dynamic_index: None,
                },
                0,
                DslxExecution::ConcreteConstants,
                Vec::new(),
            )?;
        }
        Ok(())
    }

    #[test]
    fn helper_bounds_errors_depend_on_the_evaluated_result() -> Result<(), String> {
        let expected = IrValue::make_ubits(8, 2).map_err(|e| e.to_string())?;
        for (body, bx_value, native_value) in [
            ("let a = [u8:1][x]; a", false, false),
            ("if x == u32:1 { u8:2 } else { [u8:1][x] }", true, true),
            // BX evaluates the dependencies of the helper's result. Native
            // DSLX executes the unused let and has no constexpr for this call.
            ("let a = [u8:1][x]; u8:2", true, false),
        ] {
            let source = format!("fn f(x: u32) -> u8 {{ {body} }}\nconst C = f(u32:1);");
            let bx = check_module(&parse_module(&source)?);
            if bx_value {
                assert_eq!(bx?.constants()[0].value, expected, "{source}");
            } else {
                assert_eq!(
                    bx.unwrap_err(),
                    "const 'C': initializer must evaluate to a compile-time constant",
                    "{source}"
                );
            }
            let native = dslx::parse_and_typecheck(
                &source,
                "oracle.x",
                "oracle",
                &mut ImportData::new(None, &[]),
            )
            .map_err(|e| format!("{e}\n{source}"))?;
            let Some(MatchableModuleMember::ConstantDef(c)) =
                native.get_module().get_member(1).to_matchable()
            else {
                return Err(format!("missing DSLX constant C\n{source}"));
            };
            let value = native.get_type_info().get_const_expr(&c.get_value());
            if native_value {
                assert_eq!(
                    value
                        .and_then(|v| v.convert_to_ir())
                        .map_err(|e| e.to_string())?,
                    value_to_libxls(&expected).map_err(|e| e.to_string())?,
                    "{source}"
                );
            } else {
                let err = value
                    .err()
                    .ok_or_else(|| format!("unexpected DSLX constant C\n{source}"))?;
                assert!(
                    err.to_string().contains("No constexpr value found"),
                    "{err}\n{source}"
                );
            }
        }
        Ok(())
    }
}
