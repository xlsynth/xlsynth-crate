// SPDX-License-Identifier: Apache-2.0

//! Focused audit cases for formatted trace values and assertion labels.

use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, ir};

/// Both observations return the input. The assertion case also fails on false.
fn reference(assertion_label: Option<&str>) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    if let Some(label) = assertion_label {
        let token = builder.after_all(&[])?;
        builder.assert(token, x, label, label)?;
    }
    Ok(builder.build_package(x, "reference")?.to_string())
}

/// Exercises both sides of the sign bit under default and explicit decimal
/// formatting. The expected text is computed from the source value and type.
pub(crate) fn trace_message(data: &[u8]) -> Result<AuditCase, String> {
    let mode = data.first().copied().unwrap_or_default() % 3;
    let unsigned = data.get(1).copied().unwrap_or_default() & 1 != 0;
    let width = 2 + u32::from(data.get(2).copied().unwrap_or_default() % 7);
    let modulus = 1i32 << width;
    // Both +magnitude and -magnitude must be representable in the signed type.
    let magnitude = 1 + i32::from(data.get(3).copied().unwrap_or_default()) % (modulus / 2 - 1);
    let high_bits = modulus - magnitude;
    let (sign, high_literal) = if unsigned {
        ('u', high_bits)
    } else {
        ('s', -magnitude)
    };
    let (family, specifier, signed_display) = match mode {
        0 if unsigned => ("trace_default_unsigned", "{}", false),
        0 => ("trace_default_signed", "{}", true),
        1 => ("trace_explicit_signed", "{:d}", true),
        _ => ("trace_explicit_unsigned", "{:u}", false),
    };
    let source = format!(
        r#"pub fn main(x: bool) -> bool {{
    let n = if x {{ {sign}{width}:{magnitude} }} else {{ {sign}{width}:{high_literal} }};
    trace_fmt!("VALUE {specifier}", n);
    x
}}
"#
    );
    let high_display = if signed_display {
        -magnitude
    } else {
        high_bits
    };
    Ok(AuditCase::TraceMessages {
        case: ContractCase::single(family, source, reference(None).map_err(|e| e.to_string())?),
        expected: [
            format!("VALUE {high_display}"),
            format!("VALUE {magnitude}"),
        ],
    })
}

/// The same ordinary assertion label can be supplied literally or formed by
/// constexpr formatting. Every spelling must retain the predicate and value.
pub(crate) fn assertion_label(data: &[u8]) -> Result<AuditCase, String> {
    let form = data.first().copied().unwrap_or_default() % 3;
    let separator = if data.get(1).copied().unwrap_or_default() & 1 == 0 {
        '$'
    } else {
        '_'
    };
    let suffix = data.get(2).copied().unwrap_or_default();
    let label = format!("lane{separator}{suffix}");
    let (family, assertion) = match form {
        0 => (
            "assert_label_formatted",
            format!(r#"assert_fmt!(x, "{label}")"#),
        ),
        1 => (
            "assert_label_interpolated",
            format!(r#"assert_fmt!(x, "lane{separator}{{:u}}", u8:{suffix})"#),
        ),
        _ => ("assert_label_plain", format!(r#"assert!(x, "{label}")"#)),
    };
    let source = format!("pub fn main(x: bool) -> bool {{ {assertion}; x }}\n");
    Ok(AuditCase::Semantic(ContractCase::single(
        family,
        source,
        reference(Some(&label)).map_err(|e| e.to_string())?,
    )))
}
