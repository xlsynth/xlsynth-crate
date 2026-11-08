// SPDX-License-Identifier: Apache-2.0

//! Checks the contextual width of an integer literal bound by a tuple pattern.

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

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::audit_contracts::AuditCase;

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
}
