// SPDX-License-Identifier: Apache-2.0

//! Range checks for values forwarded into signed struct parametrics.

use crate::SourceModule;
use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// Forward a value into an s3 parametric, then observe it through a signed
/// widening. The source type's bit count and literal spelling cannot change
/// whether the value fits.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let widths = [65, 64, 63, 66, 128];
    let values = [4, 3, 5, 7, 8, 0, 1];
    let width = widths[usize::from(data.first().copied().unwrap_or_default()) % widths.len()];
    let value = values[usize::from(data.get(1).copied().unwrap_or_default()) % values.len()];
    let flags = data.get(2).copied().unwrap_or_default();
    let outer_type = if width == 64 {
        "u64".to_owned()
    } else {
        format!("uN[{width}]")
    };
    let literal = if flags & 2 == 0 {
        value.to_string()
    } else {
        format!("0x{value:x}")
    };
    let inner = r#"pub struct Inner<N: s3> { payload: bool }
pub fn read_parametric<N: s3>(_x: Inner<N>) -> s8 { N as s8 }
"#;
    let mut modules = Vec::new();
    let (prefix, qualifier) = if flags & 1 == 0 {
        (inner, "")
    } else {
        modules.push(SourceModule {
            name: "inner".into(),
            text: inner.into(),
        });
        ("import inner;\n", "inner::")
    };
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{prefix}struct Outer<M: {outer_type}> {{ inner: {qualifier}Inner<M> }}
pub fn main(x: Outer<{outer_type}:{literal}>) -> s8 {{ {qualifier}read_parametric(x.inner) }}
"#
        ),
    });
    let family = "nested_struct_parametric_range";
    // An unsigned value fits s3 only if it fits in the two magnitude bits.
    if value > 3 {
        return AuditCase::ExpectedTypeRejection { family, modules };
    }
    AuditCase::Semantic(ContractCase {
        family,
        modules,
        expected_ir: format!(
            r#"package expected

top fn main(x: ((bits[1])) id=1) -> bits[8] {{
  ret value: bits[8] = literal(value={value}, id=2)
}}
"#
        ),
        max_and_nodes: None,
    })
}
