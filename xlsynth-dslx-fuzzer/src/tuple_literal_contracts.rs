// SPDX-License-Identifier: Apache-2.0

//! Checks contextual integer widths in tuple bindings and matches.

use crate::SourceModule;
use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// Compares an accepted return or argument with its declared width and literal
/// value. The bytecode control supplies the width explicitly at the same use.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let flags = byte(0);
    let width = [8, 1, 2, 4, 5, 7, 16, 32][usize::from(byte(1) % 8)];
    let value = u64::from(byte(2).wrapping_add(1)) & ((1u64 << width) - 1);
    let ty = format!("u{width}");
    let literal = if flags & 2 != 0 {
        format!("{ty}:{value}")
    } else {
        value.to_string()
    };
    let (pattern, rhs) = match (flags >> 2) & 3 {
        0 => ("(a, _)", format!("({literal}, x)")),
        1 => ("(_, a)", format!("(x, {literal})")),
        2 => ("((a, _), _)", format!("(({literal}, x), x)")),
        _ => ("(_, (_, a))", format!("(x, (x, {literal}))")),
    };
    let cast = format!("a as {ty}");
    let (family, helper, original_use, runtime_use) = if flags & 1 != 0 {
        (
            "tuple_literal_argument_width",
            format!("fn take{width}(n: {ty}) -> {ty} {{ n }}\n"),
            format!("take{width}(a)"),
            format!("take{width}({cast})"),
        )
    } else {
        (
            "tuple_literal_return_width",
            String::new(),
            "a".into(),
            cast,
        )
    };
    let module = |expr: &str| {
        format!("{helper}pub fn main(x: {ty}) -> {ty} {{ let {pattern} = {rhs}; {expr} }}\n")
    };
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> bits[{width}] {{
  ret result: bits[{width}] = literal(value={value}, id=2)
}}
"#
    );
    let case = ContractCase::single(family, module(&original_use), expected_ir);
    let runtime_module = SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{}#[test]
fn literal_runtime() {{
    assert_eq(main({ty}:0), {ty}:{value});
    assert_eq(main({ty}:1), {ty}:{value});
}}
"#,
            module(&runtime_use)
        ),
    };
    AuditCase::TupleLiteral {
        case,
        runtime_module,
    }
}

/// Compares a tuple literal field against the width supplied by a match
/// pattern.
pub(crate) fn match_pattern(data: &[u8]) -> AuditCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let flags = byte(0);
    let width = [8, 1, 2, 4, 5, 7, 16, 32][usize::from(byte(1) % 8)];
    let value = (u64::from(byte(2)) + 1) & ((1u64 << width) - 1);
    let matches = flags & 8 == 0;
    let pattern_value = if matches { value } else { value ^ 1 };
    let ty = format!("u{width}");
    let literal = if flags & 1 == 0 {
        value.to_string()
    } else {
        format!("{ty}:{value}")
    };
    let constant = format!("{ty}:{pattern_value}");
    let constexpr = flags & 16 != 0;
    let boolean = if constexpr { "false" } else { "x" };
    let tuple = |boolean: &str| match (flags >> 1) & 3 {
        0 => (
            format!("({literal}, {boolean})"),
            format!("({constant}, false)"),
        ),
        1 => (
            format!("({boolean}, {literal})"),
            format!("(false, {constant})"),
        ),
        2 => (
            format!("(({literal}, {boolean}), {boolean})"),
            format!("(({constant}, false), _)"),
        ),
        _ => (
            format!("({boolean}, ({boolean}, {literal}))"),
            format!("(_, (false, {constant}))"),
        ),
    };
    let (plain, pattern) = tuple(boolean);
    let selector = if constexpr { "true" } else { "p" };
    let alternate = if constexpr { "true" } else { "!x" };
    let (scrutinee, compound) = match (flags >> 5) & 3 {
        1 => (
            format!(
                "(if {selector} {{ {plain} }} else {{ {} }})",
                tuple(alternate).0
            ),
            true,
        ),
        2 => (
            format!(
                "(match {selector} {{ true => {plain}, false => {}, }})",
                tuple(alternate).0
            ),
            true,
        ),
        _ => (plain, false),
    };
    let source = if constexpr {
        format!(
            r#"const PICK = match {scrutinee} {{
    {pattern} => true,
    _ => false,
}};
pub fn main() -> bool {{ PICK }}
"#
        )
    } else {
        let params = if compound {
            "p: bool, x: bool"
        } else {
            "x: bool"
        };
        format!(
            r#"pub fn main({params}) -> bool {{
    match {scrutinee} {{
        {pattern} => true,
        _ => false,
    }}
}}
"#
        )
    };
    let family = "tuple_match_literal_width";
    if flags & 128 != 0 {
        let assertions = if constexpr {
            format!("    assert_eq(main(), {matches});\n")
        } else if compound {
            format!(
                r#"    assert_eq(main(false, false), false);
    assert_eq(main(false, true), {matches});
    assert_eq(main(true, false), {matches});
    assert_eq(main(true, true), false);
"#
            )
        } else {
            format!(
                r#"    assert_eq(main(false), {matches});
    assert_eq(main(true), false);
"#
            )
        };
        let module = SourceModule {
            name: "sample".into(),
            text: format!("{source}#[test]\nfn literal_runtime() {{\n{assertions}}}\n"),
        };
        return AuditCase::LiteralRuntime { family, module };
    }
    // The literal comparison is known from construction, leaving only the
    // Boolean predicate when the two integer fields match.
    let (params, result_id) = if constexpr {
        ("", 1)
    } else if compound {
        ("p: bits[1] id=1, x: bits[1] id=2", 3)
    } else {
        ("x: bits[1] id=1", 2)
    };
    let result = if constexpr || !matches {
        let value = u8::from(matches);
        format!("literal(value={value}, id={result_id})")
    } else if compound {
        "xor(p, x, id=3)".into()
    } else {
        "not(x, id=2)".into()
    };
    let expected_ir = format!(
        r#"package expected

top fn main({params}) -> bits[1] {{
  ret result: bits[1] = {result}
}}
"#
    );
    AuditCase::Semantic(ContractCase::single(family, source, expected_ir))
}

#[cfg(test)]
mod tests {
    use super::{generate, match_pattern};
    use crate::audit_contracts::AuditCase;
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn bare_return_keeps_the_cast_out_of_the_original_source() {
        let AuditCase::TupleLiteral {
            case,
            runtime_module,
        } = generate(&[])
        else {
            panic!("expected tuple literal contract");
        };
        assert_eq!(case.family, "tuple_literal_return_width");
        assert_eq!(
            case.modules[0].text,
            "pub fn main(x: u8) -> u8 { let (a, _) = (1, x); a }\n"
        );
        assert_eq!(
            runtime_module.text,
            r#"pub fn main(x: u8) -> u8 { let (a, _) = (1, x); a as u8 }
#[test]
fn literal_runtime() {
    assert_eq(main(u8:0), u8:1);
    assert_eq(main(u8:1), u8:1);
}
"#
        );
    }

    #[test]
    fn typed_control_preserves_the_helper_contract_and_reference_value() {
        let AuditCase::TupleLiteral {
            case,
            runtime_module,
        } = generate(&[13, 4, 16])
        else {
            panic!("expected tuple literal contract");
        };
        let AuditCase::TupleLiteral { case: typed, .. } = generate(&[15, 4, 16]) else {
            panic!("expected typed tuple literal contract");
        };
        assert_eq!(case.family, "tuple_literal_argument_width");
        assert_eq!(
            case.modules[0].text,
            r#"fn take5(n: u5) -> u5 { n }
pub fn main(x: u5) -> u5 { let (_, (_, a)) = (x, (x, 17)); take5(a) }
"#
        );
        assert_eq!(
            typed.modules[0].text,
            case.modules[0].text.replace("x, 17", "x, u5:17")
        );
        assert_eq!(case.expected_ir, typed.expected_ir);
        assert_eq!(
            runtime_module.text,
            r#"fn take5(n: u5) -> u5 { n }
pub fn main(x: u5) -> u5 { let (_, (_, a)) = (x, (x, 17)); take5(a as u5) }
#[test]
fn literal_runtime() {
    assert_eq(main(u5:0), u5:17);
    assert_eq(main(u5:1), u5:17);
}
"#
        );
    }

    #[test]
    fn tuple_match_controls_observe_the_integer_and_boolean_fields()
    -> Result<(), Box<dyn std::error::Error>> {
        let (AuditCase::Semantic(bare), AuditCase::Semantic(typed)) =
            (match_pattern(&[]), match_pattern(&[1]))
        else {
            panic!("expected tuple match IR contracts");
        };
        assert_eq!(
            bare.modules[0].text,
            r#"pub fn main(x: bool) -> bool {
    match (1, x) {
        (u8:1, false) => true,
        _ => false,
    }
}
"#
        );
        assert_eq!(bare.expected_ir, typed.expected_ir);
        let AuditCase::LiteralRuntime { family, module } = match_pattern(&[128]) else {
            panic!("expected original bytecode comparison");
        };
        assert_eq!(family, bare.family);
        assert_eq!(
            module.text,
            format!(
                r#"{}#[test]
fn literal_runtime() {{
    assert_eq(main(false), true);
    assert_eq(main(true), false);
}}
"#,
                bare.modules[0].text
            )
        );
        for flags in [1, 3, 5, 7, 9, 11, 13, 15, 33, 41, 65, 73] {
            let AuditCase::Semantic(case) = match_pattern(&[flags]) else {
                panic!("expected tuple match IR contract");
            };
            let oracle = IrPackage::parse_ir(&case.expected_ir, None)?;
            let native = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            let compound = flags & 96 != 0;
            let selectors: &[bool] = if compound { &[false, true] } else { &[true] };
            for &p in selectors {
                for x in [false, true] {
                    let args = if compound {
                        vec![
                            XlsIrValue::make_ubits(1, u64::from(p))?,
                            XlsIrValue::make_ubits(1, u64::from(x))?,
                        ]
                    } else {
                        vec![XlsIrValue::make_ubits(1, u64::from(x))?]
                    };
                    let expected = XlsIrValue::make_ubits(1, u64::from((p ^ x) && flags & 8 == 0))?;
                    assert_eq!(oracle.get_function("main")?.interpret(&args)?, expected);
                    assert_eq!(
                        native.ir.get_function("__sample__main")?.interpret(&args)?,
                        expected
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn constexpr_match_controls_preserve_the_selected_arm() -> Result<(), Box<dyn std::error::Error>>
    {
        for (flags, expected) in [(17, 1), (25, 0), (49, 1), (81, 1)] {
            let AuditCase::Semantic(case) = match_pattern(&[flags]) else {
                panic!("expected tuple match IR contract");
            };
            let oracle = IrPackage::parse_ir(&case.expected_ir, None)?;
            let native = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            let expected = XlsIrValue::make_ubits(1, expected)?;
            assert_eq!(oracle.get_function("main")?.interpret(&[])?, expected);
            assert_eq!(
                native.ir.get_function("__sample__main")?.interpret(&[])?,
                expected
            );
        }
        Ok(())
    }
}
