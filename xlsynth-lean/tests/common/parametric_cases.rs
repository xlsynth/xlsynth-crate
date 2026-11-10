// SPDX-License-Identifier: Apache-2.0

/// A DSLX module and two argument sets for its `g` entry point.
pub struct Case {
    pub source: String,
    pub parameters: Vec<(String, u32)>,
    pub inputs: Vec<Vec<u16>>,
    /// Number of `(u16, u16)` pairs in the returned tuple.
    pub result_count: usize,
}

/// Generates nested value-parametric calls from a fixed-size fuzz input.
///
/// Byte 0 selects the family (bit 0, set for square/subtract), reverses formal
/// declaration order (bit 1), and swaps A/B roles in the body (bit 2). Bits 3-4
/// select `1 + value % 3` ordinary arguments. Bits 5-7 are unused. The
/// remaining bytes encode little-endian u16 values a, b, c, x, and bias, in
/// that order.
pub fn generate(data: [u8; 11]) -> Case {
    let arity = usize::from((data[0] >> 3) & 3) % 3 + 1;
    let [a, b, c, x, bias] =
        std::array::from_fn(|index| u16::from_le_bytes([data[1 + index * 2], data[2 + index * 2]]));
    let formals = if data[0] & 2 == 0 {
        ["A", "B"]
    } else {
        ["B", "A"]
    };
    let [outer_value, inner_value] = if data[0] & 4 == 0 {
        ["A", "B"]
    } else {
        ["B", "A"]
    };
    let parameters = ["x", "y", "z"][..arity]
        .iter()
        .map(|name| ((*name).to_string(), 16))
        .collect::<Vec<_>>();
    let signature = parameters
        .iter()
        .map(|(name, _)| format!("{name}: u16"))
        .collect::<Vec<_>>()
        .join(", ");
    let [first_formal, second_formal] = formals;

    let (source, inputs, result_count) = if data[0] & 1 == 0 {
        let values = [a & 15, b & 15, c & 15];
        // Runtime x, y, z replace c, b, a respectively. Index by position so
        // equal literals still exercise distinct runtime argument bindings.
        let operands = ["z", "y", "x"]
            .iter()
            .enumerate()
            .map(|(index, name)| {
                if 2 - index < arity {
                    format!("({name} as u4)")
                } else {
                    format!("u4:{}", values[index])
                }
            })
            .collect::<Vec<_>>();
        let calls = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (0, 1, 2),
        ]
        .map(|(first, second, third)| {
            format!(
                "outer<u4:{}, u4:{}>({})",
                values[first], values[second], operands[third]
            )
        })
        .join(",\n        ");
        let result_type = ["(u16, u16)"; 7].join(", ");
        let source = format!(
            "fn inner<Z: u4>(x: u4) -> u8 {{ Z ++ x }}
fn outer<{first_formal}: u4, {second_formal}: u4>(x: u4) -> (u16, u16) {{
    let mid = inner<{inner_value}>(x);
    (({outer_value} ++ mid) as u16, mid as u16)
}}
fn g({signature}) -> ({result_type}) {{
    (
        {calls}
    )
}}
"
        );
        let inputs = vec![
            [c & 15, b & 15, a & 15][..arity].to_vec(),
            [x, bias, x.wrapping_add(bias)][..arity].to_vec(),
        ];
        (source, inputs, 7)
    } else {
        // Unsigned arithmetic wraps at each operation and keeps these helpers
        // on the readable emitter's supported source-signature path.
        let bias_expression = match arity {
            1 => format!("u16:{c}"),
            2 => "y".to_string(),
            3 => "y + z".to_string(),
            _ => unreachable!(),
        };
        let source = format!(
            "fn inner<Z: u16>(x: u16) -> u16 {{ Z - x }}
fn outer<{first_formal}: u16, {second_formal}: u16>(x: u16, bias: u16) -> (u16, u16) {{
    let mid = inner<{inner_value}>(x);
    ({outer_value} * {outer_value} + mid + bias, mid)
}}
fn g({signature}) -> ((u16, u16), (u16, u16), (u16, u16)) {{
    let bias = {bias_expression};
    (
        outer<u16:{a}, u16:{b}>(x, bias),
        outer<u16:{b}, u16:{a}>(x, bias),
        outer<u16:{a}, u16:{b}>(x, bias)
    )
}}
"
        );
        let inputs = vec![
            [x, bias, c][..arity].to_vec(),
            [x ^ 0x8000, bias ^ 0xffff, c ^ 0x8000][..arity].to_vec(),
        ];
        (source, inputs, 3)
    };

    Case {
        source,
        parameters,
        inputs,
        result_count,
    }
}
