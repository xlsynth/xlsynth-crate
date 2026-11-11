// SPDX-License-Identifier: Apache-2.0

//! Aliased aggregate value parametrics and their specialization identities.

use crate::{SourceModule, converter_contracts::ContractCase};

/// Calls two specializations with the same interface and distinct aggregate
/// values, then uses the first specialization again on a different input.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let byte = |index: usize, default| data.get(index).copied().unwrap_or(default);
    let flags = byte(0, 0);
    let shape = if flags & 64 == 0 { flags & 3 } else { 4 };
    let width = 2 + u32::from(byte(1, 6) % 15);
    let mask = (1 << width) - 1;
    let sign = if flags & 32 == 0 { 'u' } else { 's' };
    let ty = format!("{sign}{width}");
    let (alias, selectors): (String, &[&str]) = match shape {
        0 => (format!("({ty}, {ty})"), &["C.0", "C.1"]),
        1 => (format!("{ty}[2]"), &["C[u32:0]", "C[u32:1]"]),
        2 => (
            format!("(({ty}, {ty}), ({ty}, {ty}))"),
            &["C.0.0", "C.0.1", "C.1.0", "C.1.1"],
        ),
        3 => (
            format!("(({ty}, {ty}), {ty}[2])"),
            &["C.0.0", "C.0.1", "C.1[u32:0]", "C.1[u32:1]"],
        ),
        _ => ("Record".into(), &["C.a", "C.b"]),
    };
    let imported = flags & 4 != 0;
    let forward = flags & 8 != 0;
    let qualifier = if imported { "leaf::" } else { "" };
    let first = std::array::from_fn(|index| {
        let low = u32::from(byte(2 + index, (index + 1) as u8));
        let high = u32::from(byte(6 + index, 0));
        (low | (high << 8)) & mask
    });
    let mut second = first;
    let changed_leaf = usize::from(byte(10, 0)) % selectors.len();
    second[changed_leaf] ^= 1 << (u32::from(byte(11, 0)) % width);
    let literal = |values: [u32; 4]| {
        let [a, b, c, d] = values.map(|value| format!("{ty}:0x{value:x}"));
        match shape {
            0 => format!("({a}, {b})"),
            1 => format!("{ty}[2]:[{a}, {b}]"),
            2 => format!("(({a}, {b}), ({c}, {d}))"),
            3 => format!("(({a}, {b}), {ty}[2]:[{c}, {d}])"),
            _ => format!("{qualifier}Record {{ a: {a}, b: {b} }}"),
        }
    };
    let arity = selectors.len();
    let result_ty = format!("({})", vec![ty.as_str(); arity].join(", "));
    let observations = selectors
        .iter()
        .map(|selector| format!("x ^ {selector}"))
        .collect::<Vec<_>>()
        .join(", ");
    // Type aliases allow aggregate function parametrics in the native
    // typechecker. Direct tuple and array annotations are rejected there.
    let record = if shape == 4 {
        format!("pub struct Record {{ a: {ty}, b: {ty} }}\n")
    } else {
        String::new()
    };
    let observer = format!(
        r#"{record}pub type Aggregate = {alias};
pub fn observe<C: Aggregate>(x: {ty}) -> {result_ty} {{ ({observations}) }}
"#,
    );
    let mut modules = Vec::new();
    let mut preamble = String::new();
    if imported {
        modules.push(SourceModule {
            name: "leaf".into(),
            text: observer,
        });
        if !forward || shape == 4 {
            preamble.push_str("import leaf;\n");
        }
    } else {
        preamble.push_str(&observer);
    }
    let callee = if forward {
        let relay = format!(
            "pub fn relay<C: {qualifier}Aggregate>(x: {ty}) -> {result_ty} {{ {qualifier}observe<C>(x) }}\n"
        );
        if imported {
            modules.push(SourceModule {
                name: "middle".into(),
                text: format!("import leaf;\n{relay}"),
            });
            preamble.push_str("import middle;\n");
            "middle::relay".to_owned()
        } else {
            preamble.push_str(&relay);
            "relay".to_owned()
        }
    } else {
        format!("{qualifier}observe")
    };
    let (first_arg, second_arg) = if flags & 16 == 0 {
        (
            format!("{{{}}}", literal(first)),
            format!("{{{}}}", literal(second)),
        )
    } else {
        preamble.push_str(&format!(
            "const FIRST = {};\nconst SECOND = {};\n",
            literal(first),
            literal(second),
        ));
        ("FIRST".into(), "SECOND".into())
    };
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{preamble}
pub fn main(x: {ty}, y: {ty}) -> ({result_ty}, {result_ty}, {result_ty}) {{
    ({callee}<{first_arg}>(x), {callee}<{second_arg}>(x), {callee}<{first_arg}>(y))
}}
"#,
        ),
    });

    let ir_bits = format!("bits[{width}]");
    let ir_result = format!("({})", vec![ir_bits.as_str(); arity].join(", "));
    let mut nodes = Vec::new();
    for (call, (input, values)) in [("x", first), ("x", second), ("y", first)]
        .into_iter()
        .enumerate()
    {
        let mut observed = Vec::new();
        for (index, value) in values.into_iter().take(arity).enumerate() {
            let constant = format!("constant_{call}_{index}");
            let result = format!("value_{call}_{index}");
            let constant_id = nodes.len() + 3;
            let result_id = constant_id + 1;
            nodes.push(format!(
                "  {constant}: {ir_bits} = literal(value={value}, id={constant_id})"
            ));
            nodes.push(format!(
                "  {result}: {ir_bits} = xor({input}, {constant}, id={result_id})"
            ));
            observed.push(result);
        }
        let tuple_id = nodes.len() + 3;
        nodes.push(format!(
            "  call_{call}: {ir_result} = tuple({}, id={tuple_id})",
            observed.join(", "),
        ));
    }
    let result_id = nodes.len() + 3;
    let nodes = nodes.join("\n");
    let expected_ir = format!(
        r#"package expected

top fn main(x: {ir_bits} id=1, y: {ir_bits} id=2) -> ({ir_result}, {ir_result}, {ir_result}) {{
{nodes}
  ret result: ({ir_result}, {ir_result}, {ir_result}) = tuple(call_0, call_1, call_2, id={result_id})
}}
"#,
    );
    ContractCase {
        family: "aggregate_value_parametric_specializations",
        modules,
        expected_ir,
        max_and_nodes: None,
    }
}
