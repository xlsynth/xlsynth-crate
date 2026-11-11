// SPDX-License-Identifier: Apache-2.0

//! Array ellipses with equivalent element spellings and length contexts.

use crate::converter_contracts::ContractCase;

/// Compares explicit unit-array syntax against the same type named by an alias
/// or supplied by the return annotation.
pub(crate) fn unit(data: &[u8]) -> ContractCase {
    let spelling = data.first().copied().unwrap_or_default() % 3;
    let length = 2 + usize::from(data.get(1).copied().unwrap_or_default() % 4);
    let (prefix, annotation) = match spelling {
        0 => ("", format!("()[{length}]:")),
        1 => ("", String::new()),
        _ => ("type Unit = ();\n", format!("Unit[{length}]:")),
    };
    let source = format!("{prefix}pub fn main(x: ()) -> ()[{length}] {{ {annotation}[x, ...] }}\n");
    let elements = vec!["x"; length].join(", ");
    let expected_ir = format!(
        r#"package expected

top fn main(x: () id=1) -> ()[{length}] {{
  ret result: ()[{length}] = array({elements}, id=2)
}}
"#
    );
    ContractCase::single("unit_array_ellipsis_spelling", source, expected_ir)
}

/// Checks an ellipsis whose sibling branch has a concrete function return
/// type, with bound-call and explicit-annotation controls.
pub(crate) fn sibling_call(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [8, 1, 5, 16, 32, 64];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let lengths = [3, 2, 4, 5];
    let length = lengths[usize::from(data.get(2).copied().unwrap_or_default()) % lengths.len()];
    let index = usize::from(data.get(3).copied().unwrap_or_default()) % length;
    let array_type = format!("u{width}[{length}]");
    let (binding, call, annotation, literal) = match flags & 3 {
        0 => ("", "make(y)", String::new(), "[x, ...]".into()),
        1 => (
            "    let ys = make(y);\n",
            "ys",
            String::new(),
            "[x, ...]".into(),
        ),
        2 => ("", "make(y)", format!(": {array_type}"), "[x, ...]".into()),
        _ => (
            "",
            "make(y)",
            String::new(),
            format!("{array_type}:[x, ...]"),
        ),
    };
    let (on_true, on_false, true_value, false_value) = if flags & 4 == 0 {
        (call, literal.as_str(), "y", "x")
    } else {
        (literal.as_str(), call, "x", "y")
    };
    let source = format!(
        r#"fn make(x: u{width}) -> {array_type} {{ [x, ...] }}

pub fn main(c: bool, x: u{width}, y: u{width}) -> (u32, u{width}) {{
{binding}    let a{annotation} = if c {{ {on_true} }} else {{ {on_false} }};
    (array_size(a), a[u32:{index}])
}}
"#
    );
    // Both branches fill their entire arrays with one value, so the scalar
    // selection models the indexed result without constructing either array.
    let expected_ir = format!(
        r#"package expected

top fn main(c: bits[1] id=1, x: bits[{width}] id=2, y: bits[{width}] id=3) -> (bits[32], bits[{width}]) {{
  size: bits[32] = literal(value={length}, id=4)
  value: bits[{width}] = sel(c, cases=[{false_value}, {true_value}], id=5)
  ret result: (bits[32], bits[{width}]) = tuple(size, value, id=6)
}}
"#
    );
    ContractCase::single("array_ellipsis_sibling_call", source, expected_ir)
}
