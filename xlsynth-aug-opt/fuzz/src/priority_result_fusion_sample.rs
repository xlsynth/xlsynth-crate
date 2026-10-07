// SPDX-License-Identifier: Apache-2.0

//! Typed, bounded input graphs for focused priority-result fusion fuzzing.

use std::collections::BTreeSet;

use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::push_node;
use xlsynth_pir::ir_verify::verify_function;
use xlsynth_pir::math::ceil_log2;
use xlsynth_pir::{IrBits, IrValue};

pub const CASE_NAMES: [&str; 16] = [
    "decode",
    "one_shift",
    "expanded_concat",
    "expanded_xor",
    "shared_count",
    "shared_amount",
    "truncated_count",
    "incomplete_guard",
    "reversed_leaf",
    "non_one_shift",
    "shared_hot",
    "shared_input",
    "wide_constant",
    "variable_index",
    "shared_predicate",
    "sign_extended_index",
];
const INPUT_WIDTHS: [usize; 7] = [3, 7, 8, 31, 63, 64, 65];
const STEP_NAMES: [&str; 6] = [
    "add_right",
    "add_left",
    "subtract",
    "reflect",
    "zero_extend",
    "zero_prefix",
];

/// One generated function, its intended rejection controls, and live features.
pub struct Sample {
    pub function: ir::Fn,
    pub case: usize,
    pub pipeline: usize,
    pub must_reject: bool,
    pub may_decline: bool,
    pub features: BTreeSet<&'static str>,
}

/// Reads independently mutable choices; depleted input defaults to zero.
struct Choices<'a> {
    remaining: &'a [u8],
}

impl Choices<'_> {
    fn next(&mut self) -> u8 {
        let Some((&value, remaining)) = self.remaining.split_first() else {
            return 0;
        };
        self.remaining = remaining;
        value
    }
}

/// Constructs a literal modulo its width, without a fixed-width IR conversion.
fn small_literal(f: &mut ir::Fn, width: usize, value: usize) -> NodeRef {
    let bits: Vec<_> = (0..width)
        .map(|bit| bit < usize::BITS as usize && value & (1usize << bit) != 0)
        .collect();
    literal(f, IrBits::from_lsb_is_0(&bits))
}

fn literal(f: &mut ir::Fn, bits: IrBits) -> NodeRef {
    push_node(
        f,
        Type::Bits(bits.get_bit_count()),
        NodePayload::Literal(IrValue::from_bits(&bits)),
    )
}

fn parameter(f: &mut ir::Fn, width: usize, name: &str) -> NodeRef {
    let node = push_node(f, Type::Bits(width), NodePayload::Param);
    f.get_node_mut(node).name = Some(name.to_string());
    f.params.push(node);
    node
}

fn slice(f: &mut ir::Fn, arg: NodeRef, start: usize, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::BitSlice { arg, start, width },
    )
}

/// Chooses boundary constants or arbitrary bits, including bits above bit 63.
fn constant(
    f: &mut ir::Fn,
    width: usize,
    input_width: usize,
    output_width: usize,
    bytes: &mut Choices,
) -> NodeRef {
    match bytes.next() % 7 {
        0 => small_literal(f, width, 0),
        1 => small_literal(f, width, 1),
        2 => small_literal(f, width, input_width),
        3 => small_literal(f, width, output_width),
        4 => literal(f, IrBits::from_lsb_is_0(&vec![true; width])),
        5 => {
            let mut bits = vec![false; width];
            bits[width - 1] = true;
            literal(f, IrBits::from_lsb_is_0(&bits))
        }
        _ => {
            let mut bits = Vec::with_capacity(width);
            while bits.len() < width {
                let byte = bytes.next();
                for bit in 0..8 {
                    if bits.len() < width {
                        bits.push(byte & (1 << bit) != 0);
                    }
                }
            }
            literal(f, IrBits::from_lsb_is_0(&bits))
        }
    }
}

/// Extends a live index through one operation of the recognized grammar.
fn index_step(
    f: &mut ir::Fn,
    amount: NodeRef,
    width: &mut usize,
    input_width: usize,
    output_width: usize,
    bytes: &mut Choices,
    features: &mut BTreeSet<&'static str>,
) -> NodeRef {
    let step = usize::from(bytes.next() % 6);
    features.insert(STEP_NAMES[step]);
    if step >= 4 {
        let extension = 1 + usize::from(bytes.next() % 5);
        let new_width = (*width + extension).min(96);
        let payload = if step == 4 {
            NodePayload::ZeroExt {
                arg: amount,
                new_bit_count: new_width,
            }
        } else {
            let zero = small_literal(f, new_width - *width, 0);
            NodePayload::Nary(NaryOp::Concat, vec![zero, amount])
        };
        *width = new_width;
        return push_node(f, Type::Bits(*width), payload);
    }
    let value = constant(f, *width, input_width, output_width, bytes);
    if step <= 1
        && matches!(f.get_node(amount).payload, NodePayload::Encode { .. })
        && input_width + 1 == 1usize << *width
        && matches!(&f.get_node(value).payload, NodePayload::Literal(value) if value.bits_equals_u64_value(1))
    {
        features.insert("wrapping");
    }
    let payload = match step {
        0 => NodePayload::Binop(Binop::Add, amount, value),
        1 => NodePayload::Binop(Binop::Add, value, amount),
        2 => NodePayload::Binop(Binop::Sub, amount, value),
        _ => NodePayload::Binop(Binop::Sub, value, amount),
    };
    push_node(f, Type::Bits(*width), payload)
}

/// Builds the select/concat decoder independently from the fusion matcher.
fn expanded_decoder(
    f: &mut ir::Fn,
    amount: NodeRef,
    count: usize,
    xor_leaf: bool,
    reversed: bool,
    flags: u8,
    features: &mut BTreeSet<&'static str>,
) -> NodeRef {
    let bit = slice(f, amount, 0, 1);
    let mut previous = if xor_leaf {
        let mut mask = push_node(
            f,
            Type::Bits(2),
            NodePayload::SignExt {
                arg: bit,
                new_bit_count: 2,
            },
        );
        if flags & 8 != 0 {
            let all_ones = small_literal(f, 2, 3);
            mask = push_node(
                f,
                Type::Bits(2),
                NodePayload::Nary(NaryOp::And, vec![all_ones, mask]),
            );
            features.insert("redundant_leaf_mask");
        }
        let one = small_literal(f, 2, 1);
        let parts = if flags & 16 == 0 {
            vec![one, mask]
        } else {
            vec![mask, one]
        };
        vec![push_node(
            f,
            Type::Bits(2),
            NodePayload::Nary(NaryOp::Xor, parts),
        )]
    } else {
        let complement = push_node(f, Type::Bits(1), NodePayload::Unop(Unop::Not, bit));
        let leaf_parts = if reversed {
            vec![complement, bit]
        } else {
            vec![bit, complement]
        };
        if flags & 4 != 0 {
            features.insert("flattened_concat");
            leaf_parts
        } else {
            vec![push_node(
                f,
                Type::Bits(2),
                NodePayload::Nary(NaryOp::Concat, leaf_parts),
            )]
        }
    };
    for level in 1..count {
        let half = 1usize << level;
        let padding = if flags & 8 != 0 {
            features.insert("split_padding");
            vec![small_literal(f, 1, 0), small_literal(f, half - 1, 0)]
        } else {
            vec![small_literal(f, half, 0)]
        };
        let low = push_node(
            f,
            Type::Bits(half * 2),
            NodePayload::Nary(NaryOp::Concat, [padding.clone(), previous.clone()].concat()),
        );
        let high = push_node(
            f,
            Type::Bits(half * 2),
            NodePayload::Nary(NaryOp::Concat, [previous, padding].concat()),
        );
        let selector = slice(f, amount, level, 1);
        let tree = push_node(
            f,
            Type::Bits(half * 2),
            NodePayload::Sel {
                selector,
                cases: vec![low, high],
                default: None,
            },
        );
        previous = vec![tree];
    }
    let tree = if previous.len() == 1 {
        previous[0]
    } else {
        push_node(
            f,
            Type::Bits(2),
            NodePayload::Nary(NaryOp::Concat, previous),
        )
    };
    let overflow = slice(f, amount, count, 1);
    let valid = push_node(f, Type::Bits(1), NodePayload::Unop(Unop::Not, overflow));
    let width = 1 << count;
    let guard = push_node(
        f,
        Type::Bits(width),
        NodePayload::SignExt {
            arg: valid,
            new_bit_count: width,
        },
    );
    let parts = if flags & 16 == 0 {
        vec![tree, guard]
    } else {
        vec![guard, tree]
    };
    push_node(f, Type::Bits(width), NodePayload::Nary(NaryOp::And, parts))
}

/// Decodes byte-controlled operation sequences and deliberate valid near
/// misses.
pub fn generate_sample(data: &[u8]) -> Result<Sample, String> {
    let mut bytes = Choices { remaining: data };
    let case = usize::from(bytes.next()) % CASE_NAMES.len();
    let input_width = INPUT_WIDTHS[usize::from(bytes.next()) % INPUT_WIDTHS.len()];
    let flags = bytes.next();
    let steps = usize::from(bytes.next() % 7);
    let pipeline = usize::from(bytes.next() % 3);
    let output_choice = usize::from(bytes.next() % 4);
    let prefix_steps = usize::from(bytes.next() % 4);
    let count_width = ceil_log2(input_width + 1);
    let expanded = matches!(case, 2 | 3 | 7 | 8);
    let mut output_width = if expanded {
        1 << count_width
    } else {
        [
            input_width,
            input_width + 1,
            input_width + 3,
            1 << count_width,
        ][output_choice]
    };
    let mut f = ir::Fn {
        graph: ir::NodeGraph::new("main"),
        params: Vec::new(),
        ret_ty: Type::Bits(output_width),
        ret_node_ref: None,
    };
    let x = parameter(&mut f, input_width, "x");
    let p = parameter(&mut f, 1, "p");
    let mut input = x;
    for _ in 0..prefix_steps {
        let op = if bytes.next() & 1 == 0 {
            Unop::Reverse
        } else {
            Unop::Not
        };
        input = push_node(
            &mut f,
            Type::Bits(input_width),
            NodePayload::Unop(op, input),
        );
    }
    let mut features = BTreeSet::from([
        if flags & 1 == 0 {
            "msb_priority"
        } else {
            "lsb_priority"
        },
        if flags & 2 == 0 { "unmasked" } else { "masked" },
        if input_width > 64 {
            "wide_input"
        } else {
            "narrow_input"
        },
        ["pir_only", "sandwich_1", "sandwich_3"][pipeline],
    ]);
    if steps > 1 && case != 7 {
        features.insert("operation_sequence");
    }
    let hot = push_node(
        &mut f,
        Type::Bits(input_width + 1),
        NodePayload::OneHot {
            arg: input,
            lsb_prio: flags & 1 != 0,
        },
    );
    let count = push_node(
        &mut f,
        Type::Bits(count_width),
        NodePayload::Encode { arg: hot },
    );
    let mut amount = count;
    let mut amount_width = count_width;
    if case == 6 {
        amount_width -= 1;
        amount = slice(&mut f, amount, 0, amount_width);
    }
    if case == 12 {
        amount_width = 80;
        amount = push_node(
            &mut f,
            Type::Bits(80),
            NodePayload::ZeroExt {
                arg: amount,
                new_bit_count: 80,
            },
        );
        let all_ones = literal(&mut f, IrBits::from_lsb_is_0(&vec![true; 80]));
        amount = push_node(
            &mut f,
            Type::Bits(80),
            NodePayload::Binop(Binop::Add, amount, all_ones),
        );
        features.insert("wide_constant");
        features.insert("wrapping");
        features.insert("overshift");
    }
    if case == 7 {
        // A reachable bit above the guard makes this valid IR an incomplete
        // decoder; the semantic fusion must decline it.
        amount_width = count_width + 2;
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::ZeroExt {
                arg: amount,
                new_bit_count: amount_width,
            },
        );
        let high = small_literal(&mut f, amount_width, 1 << (count_width + 1));
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::Binop(Binop::Add, amount, high),
        );
    } else {
        for _ in 0..steps {
            amount = index_step(
                &mut f,
                amount,
                &mut amount_width,
                input_width,
                output_width,
                &mut bytes,
                &mut features,
            );
        }
    }
    if expanded && amount_width <= count_width {
        amount_width = count_width + 1;
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::ZeroExt {
                arg: amount,
                new_bit_count: amount_width,
            },
        );
    }
    if !expanded && steps == 0 && output_width <= input_width && !matches!(case, 6 | 12 | 13 | 15) {
        features.insert("overshift");
    }
    if case == 13 {
        let variable = parameter(&mut f, amount_width, "variable");
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::Binop(Binop::Add, amount, variable),
        );
    }
    if case == 15 {
        amount_width += 1;
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::SignExt {
                arg: amount,
                new_bit_count: amount_width,
            },
        );
    }
    let mut predicate = p;
    if flags & 32 != 0 {
        let bit = slice(&mut f, x, 0, 1);
        predicate = push_node(
            &mut f,
            Type::Bits(1),
            NodePayload::Nary(NaryOp::Xor, vec![p, bit]),
        );
        if flags & 2 != 0 || case == 14 {
            features.insert("data_dependent_predicate");
        }
    }
    if flags & 2 != 0 {
        let mask = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::SignExt {
                arg: predicate,
                new_bit_count: amount_width,
            },
        );
        let parts = if flags & 16 == 0 {
            vec![amount, mask]
        } else {
            vec![mask, amount]
        };
        amount = push_node(
            &mut f,
            Type::Bits(amount_width),
            NodePayload::Nary(NaryOp::And, parts),
        );
    }
    if !expanded && !matches!(case, 1 | 9) && amount_width < usize::BITS as usize {
        output_width = output_width.min(1usize << amount_width);
    }
    let output = if expanded {
        features.insert(if case == 3 {
            "expanded_xor"
        } else {
            "expanded_concat"
        });
        expanded_decoder(
            &mut f,
            amount,
            count_width,
            case == 3,
            case == 8,
            flags,
            &mut features,
        )
    } else if matches!(case, 1 | 9) {
        features.insert("one_shift");
        let one = small_literal(&mut f, output_width, if case == 9 { 2 } else { 1 });
        push_node(
            &mut f,
            Type::Bits(output_width),
            NodePayload::Binop(Binop::Shll, one, amount),
        )
    } else {
        features.insert("decode");
        push_node(
            &mut f,
            Type::Bits(output_width),
            NodePayload::Decode {
                arg: amount,
                width: output_width,
            },
        )
    };
    let retained = match case {
        4 => Some(count),
        5 => Some(amount),
        10 => Some(hot),
        11 => Some(x),
        14 => Some(predicate),
        _ => None,
    };
    let result = if let Some(retained) = retained {
        let ty = Type::Tuple(vec![
            Box::new(f.get_node(output).ty.clone()),
            Box::new(f.get_node(retained).ty.clone()),
        ]);
        push_node(&mut f, ty, NodePayload::Tuple(vec![output, retained]))
    } else {
        output
    };
    f.ret_ty = f.get_node(result).ty.clone();
    f.ret_node_ref = Some(result);
    verify_function(&f).map_err(|error| format!("invalid generated priority IR: {error}"))?;
    Ok(Sample {
        function: f,
        case,
        pipeline,
        must_reject: matches!(case, 4..=9 | 13 | 15),
        may_decline: matches!(case, 2 | 3),
        features,
    })
}

/// Supplies a small reproducible corpus for semantic coverage auditing.
pub fn validation_inputs() -> Vec<Vec<u8>> {
    let mut inputs = Vec::new();
    for case in 0..CASE_NAMES.len() {
        for flags in [0, 1, 2, 3, 15, 51] {
            inputs.push(vec![
                case as u8,
                (case % INPUT_WIDTHS.len()) as u8,
                flags,
                0,
                (case % 3) as u8,
                1,
                0,
            ]);
        }
    }
    for step in 0..6 {
        inputs.push(vec![0, 0, 3, 1, step % 3, 1, 0, step, 1]);
    }
    inputs.push(vec![0, 6, 3, 0, 2, 1, 0]);
    inputs.push(vec![0, 0, 3, 0, 1, 0, 0]);
    inputs.push(vec![
        0, 0, 3, 6, 2, 1, 0, 0, 1, 4, 2, 3, 4, 5, 1, 1, 2, 2, 5,
    ]);
    inputs.push(vec![0, 0, 3, 2, 1, 1, 0, 4, 0, 0, 1]);
    inputs
}
