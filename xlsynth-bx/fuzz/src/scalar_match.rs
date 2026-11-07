// SPDX-License-Identifier: Apache-2.0

//! Compare scalar pattern coverage and first-match priority with native DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

/// Scalar type and endpoints in its source comparison order.
struct Domain {
    ty: String,
    boolean: bool,
    signed: bool,
    min: IrBits,
    max: IrBits,
}

impl Domain {
    fn new(selector: u8) -> Self {
        let boolean = selector & 32 != 0;
        let width = if boolean {
            1
        } else {
            [0, 1, 2, 3, 4, 5, 8, 16, 31, 32, 63, 64, 65, 96, 127, 129][usize::from(selector % 16)]
        };
        let signed = !boolean && width != 0 && selector & 16 != 0;
        let (min, max) = if signed {
            (
                IrBits::signed_min_value(width),
                IrBits::signed_max_value(width),
            )
        } else {
            (IrBits::zero(width), IrBits::all_ones(width))
        };
        let ty = if boolean {
            "bool".to_string()
        } else {
            format!("{}N[{width}]", if signed { "s" } else { "u" })
        };
        Self {
            ty,
            boolean,
            signed,
            min,
            max,
        }
    }

    fn literal(&self, value: &IrBits, spelling: u8) -> String {
        if self.boolean {
            return (!value.is_zero()).to_string();
        }
        if self.signed && spelling & 6 == 4 && !value.is_zero() {
            // A typed negative magnitude can fill all w bits. Negating it
            // modulo 2^w recovers the original endpoint (e.g. s3:-5 is +3).
            let magnitude = value
                .negate()
                .to_string_fmt(IrFormatPreference::UnsignedDecimal, false);
            return format!("{}:-{magnitude}", self.ty);
        }
        let radix = if spelling & 1 != 0 {
            IrFormatPreference::Hex
        } else if self.signed {
            IrFormatPreference::SignedDecimal
        } else {
            IrFormatPreference::UnsignedDecimal
        };
        let text = value.to_string_fmt(radix, false);
        if spelling & 2 == 0 {
            format!("{}:{text}", self.ty)
        } else {
            text
        }
    }
}

/// Vary the order of high-bit unsized literals, other literals, and concrete
/// or call-derived types in the result arms, independently of selector typing.
fn compare_signed_results(
    seed: u64,
    width_selector: u8,
    flags: u8,
    order: u8,
    anchor_selector: u8,
) -> Result<(), String> {
    let width = [3, 4, 7, 8, 16, 33, 64, 65, 96, 129][usize::from(width_selector) % 10];
    let ty = format!("sN[{width}]");
    let high = IrBits::all_ones(width).to_string_fmt(IrFormatPreference::Hex, false);
    let mut functions = format!("fn anchor() -> {ty} {{ {ty}:3 }}");
    let anchor = match anchor_selector % 5 {
        0 => format!("{ty}:3"),
        1 => "anchor()".into(),
        2 => format!("anchor() as {ty}"),
        3 => "anchor() + 0".into(),
        _ => format!("anchor() + {ty}:0"),
    };
    let mut leaves = [
        (high, flags & 8 != 0),
        ((order % 3).to_string(), false),
        (anchor, false),
    ];
    leaves.rotate_left(usize::from(order % 3));
    if order & 4 != 0 {
        leaves.swap(0, 1);
    }
    let shape = (flags >> 1) % 3;
    let result = match shape {
        1 => format!("({ty}, u8)"),
        2 => format!("{ty}[1]"),
        _ => ty,
    };
    let arms = leaves
        .into_iter()
        .zip(["0", "1", "_"])
        .enumerate()
        .map(|(i, ((leaf, scoped), pattern))| {
            let field = if scoped { "raw" } else { &leaf };
            let result = match shape {
                1 => format!("({field}, u8:{i})"),
                2 => format!("[{field}]"),
                _ => field.to_owned(),
            };
            if scoped {
                format!("{pattern} => {{ let raw = {leaf}; {result} }}")
            } else {
                format!("{pattern} => {result}")
            }
        })
        .collect::<Vec<_>>();
    let params = "x: u2".to_owned();
    let mut body = format!("match x {{ {} }}", arms.join(", "));
    if flags & 32 != 0 {
        body = format!("let inferred = {body}; inferred");
    }
    if flags & 1 != 0 {
        functions.push_str(&format!("\nfn route({params}) -> {result} {{ {body} }}"));
        body = "route(x)".into();
    }
    let inputs = (0..4)
        .map(|x| IrValue::make_ubits(2, x).map(|value| vec![value]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    compare_program(
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid: None,
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::BytecodeTests,
        inputs,
    )
}

/// Generate literal and range patterns together with inputs at and on either
/// side of every endpoint. The native source checker decides acceptance for
/// the freely combined arms; partitions and intentional errors also carry an
/// independent acceptance expectation.
pub fn check_scalar_match(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let mode = bytes.next().unwrap() % 9;
    let selector = bytes.next().unwrap();
    let flags = bytes.next().unwrap();
    if mode == 8 {
        return compare_signed_results(
            crate::input_seed(data),
            selector,
            flags,
            bytes.next().unwrap(),
            bytes.next().unwrap(),
        );
    }
    let domain = Domain::new(selector);
    let width = domain.min.get_bit_count();
    let one = IrBits::from_lsb_fn(width, |i| i == 0);
    let zero = IrBits::zero(width);
    // A repeating byte stream can place patterns above bit 64 even for short
    // fuzz inputs, while longer inputs independently choose the high limbs.
    let entropy = data.get(3..).unwrap_or_default();
    let pivot = IrBits::from_lsb_fn(width, |i| {
        !entropy.is_empty() && entropy[(i / 8) % entropy.len()] & (1 << (i % 8)) != 0
    });
    let mut endpoints = vec![
        domain.min.clone(),
        domain.max.clone(),
        zero.clone(),
        pivot.clone(),
        pivot.sub(&one),
        pivot.add(&one),
    ];
    let spelling = flags >> 4;
    // Known-valid partitions use explicit type annotations at both ends.
    let typed = |v: &IrBits| domain.literal(v, spelling & 5);
    let mut valid = Some(true);
    let mut patterns = Vec::new();
    match mode {
        0..=2 => {
            valid = None;
            let arm_count = 1 + bytes.next().unwrap() % 4;
            for _ in 0..arm_count {
                let alternatives = 1 + bytes.next().unwrap() % 3;
                let mut arm = Vec::new();
                for _ in 0..alternatives {
                    let form = bytes.next().unwrap();
                    let start = &endpoints[usize::from(bytes.next().unwrap()) % endpoints.len()];
                    let literal = domain.literal(start, form >> 3);
                    let pattern = match form % 8 {
                        3..=6 if !domain.boolean => {
                            let end =
                                &endpoints[usize::from(bytes.next().unwrap()) % endpoints.len()];
                            let range = if form & 1 == 0 { ".." } else { "..=" };
                            format!("{literal}{range}{}", domain.literal(end, form >> 5))
                        }
                        7 => "_".into(),
                        _ => literal,
                    };
                    arm.push(pattern);
                }
                patterns.push(arm.join(" | "));
            }
            if flags & 8 == 0 {
                patterns.push("_".into());
            }
        }
        5 if width >= 2 => {
            let start = if domain.signed && flags & 8 == 0 {
                zero.sub(&one).sub(&one)
            } else {
                domain.min.clone()
            };
            let second = start.add(&one);
            let third = second.add(&one);
            let fourth = third.add(&one);
            patterns.push(format!("{}..={}", typed(&start), typed(&third)));
            patterns.push(format!("{}..={}", typed(&second), typed(&fourth)));
            if flags & 8 != 0 {
                patterns.reverse();
            }
            patterns.push("_".into());
            endpoints.extend([start, second, third, fourth]);
        }
        6 => {
            // Only the zero arm performs the array read. Inputs beyond the
            // array boundary must succeed in the native bytecode interpreter.
            patterns.extend([typed(&zero), "_".into()]);
        }
        7 => {
            valid = Some(false);
            // Vary missing alternative, range endpoint, and arm arrow.
            patterns.push(match flags % 3 {
                0 => format!("{} |", typed(&pivot)),
                1 => format!("{}..=", typed(&pivot)),
                _ => format!("{} => =>", typed(&pivot)),
            });
            patterns.push("_".into());
        }
        _ => {
            if width == 0 {
                patterns.push(typed(&zero));
            } else if domain.boolean {
                patterns.extend(["false".into(), "true".into()]);
            } else {
                let split = if pivot == domain.min {
                    &domain.max
                } else {
                    &pivot
                };
                patterns.push(format!("{}..{}", typed(&domain.min), typed(split)));
                patterns.push(format!("{}..={}", typed(split), typed(&domain.max)));
            }
            if flags & 8 != 0 {
                patterns.reverse();
            }
            if mode == 4 {
                patterns.pop();
                valid = Some(false);
            }
        }
    }

    let shape = (flags >> 1) % 3;
    let result = ["u8", "u8[2]", "(u8, u8[2])"][usize::from(shape)].to_owned();
    let mut arms = Vec::new();
    for (i, pattern) in patterns.iter().enumerate() {
        let tag = if mode == 6 && i == 0 {
            format!("[u8:{}][x as u32]", i + 1)
        } else if flags & 32 != 0 && i + 1 < patterns.len() {
            (i + 1).to_string()
        } else {
            format!("u8:{}", i + 1)
        };
        let rhs = match shape {
            1 => format!("[{tag}, u8:{i}]"),
            2 => format!("({tag}, [u8:{i}, {tag}])"),
            _ => tag,
        };
        arms.push(format!("{pattern} => {rhs}"));
    }
    let params = format!("x: {}", domain.ty);
    let mut functions = format!("fn key({params}) -> {} {{ x }}", domain.ty);
    let subject = if flags & 128 != 0 && mode != 6 {
        // Patterns can supply the width of an unsized subject literal.
        domain.literal(&pivot, 2)
    } else if flags & 64 != 0 {
        // A call result is a delayed type constraint. An explicit cast or a
        // typed operand anchors the group. These subjects preserve x so the
        // same concrete input vectors still hit each pattern boundary.
        match bytes.next().unwrap() % 8 {
            1 => "!(!key(x))".to_owned(),
            2 if width != 0 && !domain.boolean => "key(x) + 0".to_owned(),
            3 => format!("key(x) as {}", domain.ty),
            4 => format!("key(x) ^ {}", typed(&zero)),
            5 => "(key(x),).0".to_owned(),
            6 => "if true { key(x) } else { key(x) }".to_owned(),
            7 => "if true { x } else { key(x) }".to_owned(),
            _ => "key(x)".to_owned(),
        }
    } else {
        "x".to_owned()
    };
    let mut expression = format!("match {subject} {{ {} }}", arms.join(", "));
    if flags & 32 != 0 {
        // Infer the result across arms before the surrounding return type
        // supplies any context.
        expression = format!("let chosen = {expression}; chosen");
    }
    let body = if flags & 1 != 0 {
        functions.push_str(&format!(
            "\nfn route({params}) -> {result} {{ {expression} }}"
        ));
        "route(x)".to_owned()
    } else {
        expression
    };
    let mut inputs = Vec::new();
    for endpoint in endpoints {
        for neighbor in [endpoint.sub(&one), endpoint.add(&one), endpoint] {
            let args = vec![IrValue::from_bits(&neighbor)];
            if !inputs.contains(&args) {
                inputs.push(args);
            }
        }
    }
    compare_program(
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid,
            dslx_ir: true,
            dynamic_index: None,
        },
        crate::input_seed(data),
        DslxExecution::BytecodeTests,
        inputs,
    )
}
