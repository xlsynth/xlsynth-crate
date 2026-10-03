// SPDX-License-Identifier: Apache-2.0

//! Structured inputs for the dedicated constant-shift-choice equivalence
//! fuzzer.

use xlsynth_aug_opt::constant_shift_choices::ConstantShiftChoiceLimits;
use xlsynth_pir::ir::{self, Type};
use xlsynth_pir::{BValue, FnBuilder, IrBits, IrValue};

/// Whether the generated graph must reach, or must avoid, candidate costing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expectation {
    GuaranteedCandidate,
    GuaranteedRejected,
    Unspecified,
}

/// A typed graph and the recognition limits and oracle appropriate to it.
pub struct Sample {
    pub function: ir::Fn,
    pub limits: ConstantShiftChoiceLimits,
    pub expectation: Expectation,
    /// All parameters are bits; these samples have at most seven input bits.
    /// Their small widths also bound independent local cuts to sixteen bits.
    pub exhaustive_inputs: bool,
}

/// Consumes explicit choices, defaulting missing bytes to zero for
/// minimization.
struct Bytes<'a> {
    remaining: &'a [u8],
}

impl Bytes<'_> {
    fn next(&mut self) -> u8 {
        match self.remaining.split_first() {
            Some((&byte, rest)) => {
                self.remaining = rest;
                byte
            }
            None => 0,
        }
    }

    fn index(&mut self, len: usize) -> usize {
        usize::from(self.next()) % len
    }

    fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.index(values.len())]
    }
}

/// Constructs the low `width` bits without assuming the IR width fits in u64.
fn literal(builder: &mut FnBuilder, width: usize, value: usize) -> BValue {
    let bits: Vec<_> = (0..width)
        .map(|bit| bit < usize::BITS as usize && value & (1usize << bit) != 0)
        .collect();
    builder
        .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&bits)))
        .unwrap()
}

/// Places a set bit above bit 63 whenever the amount type permits it.
fn high_literal(builder: &mut FnBuilder, width: usize, bytes: &mut Bytes<'_>) -> BValue {
    let mut bits = vec![false; width];
    let start = 64.min(width - 1);
    bits[start + bytes.index(width - start)] = true;
    bits[0] |= bytes.next() & 1 != 0;
    builder
        .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&bits)))
        .unwrap()
}

/// Supplies independent controls, shared inversions, and data-dependent
/// controls.
fn controls(
    builder: &mut FnBuilder,
    input: BValue,
    data: BValue,
    width: usize,
    bytes: &mut Bytes<'_>,
) -> Vec<BValue> {
    let mut values: Vec<_> = (0..3)
        .map(|start| builder.bit_slice(input, start, 1).unwrap())
        .collect();
    values.push(builder.not(values[0]).unwrap());
    values.push(builder.xor(values[0], values[1]).unwrap());
    values.push(builder.and(values[1], values[2]).unwrap());
    if width != 0 {
        let bit = builder.bit_slice(data, bytes.index(width), 1).unwrap();
        values.push(bit);
        values.push(builder.xor(values[2], bit).unwrap());
    }
    values
}

/// Builds literal-prefix concatenation with independently mutable prefix
/// pieces.
fn concat_amount(
    builder: &mut FnBuilder,
    width: usize,
    data_width: usize,
    predicates: &[BValue],
    bytes: &mut Bytes<'_>,
) -> BValue {
    let split = bytes.index(width);
    let high = literal(builder, split, bytes.choose(&[0, 1, data_width]));
    let low = literal(
        builder,
        width - 1 - split,
        bytes.choose(&[0, 1, data_width / 2]),
    );
    builder
        .concat(&[high, low, bytes.choose(predicates)])
        .unwrap()
}

/// Extends an earlier amount through the exact recognized expression grammar.
fn amount_dag(
    builder: &mut FnBuilder,
    width: usize,
    data_width: usize,
    predicates: &[BValue],
    steps: usize,
    bytes: &mut Bytes<'_>,
) -> Vec<BValue> {
    let mut values: Vec<_> = [
        0,
        1,
        data_width.saturating_sub(1),
        data_width,
        data_width + 1,
    ]
    .into_iter()
    .map(|value| literal(builder, width, value))
    .collect();
    values.push(
        builder
            .literal(IrValue::from_bits(&IrBits::all_ones(width)))
            .unwrap(),
    );
    values.push(high_literal(builder, width, bytes));
    for _ in 0..steps {
        // Each operand indexes the already-built pool. Repeating an index
        // creates sharing rather than a second copy of its expression.
        let op = bytes.index(8);
        let a = bytes.choose(&values);
        let b = bytes.choose(&values);
        let predicate = bytes.choose(predicates);
        let value = match op {
            0 => builder.identity(a).unwrap(),
            1 => builder.select(predicate, &[a, b], None).unwrap(),
            2 => {
                let count = 1 + bytes.index(3);
                let selectors: Vec<_> = (0..count).map(|_| bytes.choose(predicates)).collect();
                let selector = builder.concat(&selectors).unwrap();
                let cases: Vec<_> = (0..count).map(|_| bytes.choose(&values)).collect();
                builder.priority_select(selector, &cases, a).unwrap()
            }
            3 => {
                let selector = builder
                    .concat(&[predicate, bytes.choose(predicates)])
                    .unwrap();
                let third = bytes.choose(&values);
                builder.select(selector, &[a, b, third], Some(a)).unwrap()
            }
            4 => concat_amount(builder, width, data_width, predicates, bytes),
            5 => builder.sign_extend(predicate, width).unwrap(),
            6 => {
                let first = builder.sign_extend(predicate, width).unwrap();
                let second = builder
                    .sign_extend(bytes.choose(predicates), width)
                    .unwrap();
                match bytes.index(3) {
                    0 => {
                        let inner = builder.and(a, first).unwrap();
                        builder.and(inner, second).unwrap()
                    }
                    1 => {
                        let inner = builder.and(second, a).unwrap();
                        builder.and(first, inner).unwrap()
                    }
                    _ => builder.and(first, second).unwrap(),
                }
            }
            _ => builder.select(predicate, &[a, a], None).unwrap(),
        };
        values.push(value);
    }
    // Even an identity of a literal becomes a real recognition opportunity.
    // Equal effective amounts are intentional, including saturated overshifts.
    let last = *values.last().unwrap();
    let root = builder
        .select(
            bytes.choose(predicates),
            &[last, bytes.choose(&values)],
            None,
        )
        .unwrap();
    values.push(root);
    values
}

/// Makes valid neighbors of the grammar which the recognizer must reject.
fn rejected_amount(
    builder: &mut FnBuilder,
    root: BValue,
    variable: BValue,
    width: usize,
    predicate: BValue,
    variant: usize,
) -> BValue {
    match variant {
        0 => variable,
        1 => builder.select(predicate, &[root, variable], None).unwrap(),
        2 => {
            let non_mask = builder.zero_extend(predicate, width).unwrap();
            builder.and(non_mask, root).unwrap()
        }
        3 => {
            let zero = literal(builder, width, 0);
            let one = literal(builder, width, 1);
            builder.and(zero, one).unwrap()
        }
        4 => {
            let prefix = builder.bit_slice(variable, 1, width - 1).unwrap();
            builder.concat(&[prefix, predicate]).unwrap()
        }
        5 => root, // This variant uses an arithmetic shift below.
        6 => literal(builder, width, 1),
        _ => builder.not(root).unwrap(),
    }
}

/// Chooses boundary-aligned slices, including the empty slice at the end.
fn slice(builder: &mut FnBuilder, value: BValue, width: usize, bytes: &mut Bytes<'_>) -> BValue {
    let start = bytes.choose(&[0, width / 2, width.saturating_sub(1), width]);
    let available = width - start;
    let size = bytes.choose(&[available, available, available.div_ceil(2), 0]);
    builder.bit_slice(value, start, size).unwrap()
}

/// Decodes bounded graphs; the first five bytes select lane, widths, sites, and
/// amount-DAG size. Later bytes select operations, backward edges, and outputs.
///
/// Lanes 0..=3 promise candidates (lane 1 has 130/131-bit amounts; lane 2 has
/// multiple sites), 4 promises rejection, 5 varies recognition limits, and 6/7
/// use exhaustive interpretation for small graphs and observable effects.
pub fn generate_sample(data: &[u8]) -> Sample {
    let mut bytes = Bytes { remaining: data };
    let lane = bytes.index(8);
    let exhaustive_inputs = lane >= 6;
    let width = if exhaustive_inputs {
        bytes.index(5)
    } else {
        bytes.choose(&[1, 2, 3, 4, 7, 8, 9, 16, 31, 63, 64, 65, 129])
    };
    let amount_width = if exhaustive_inputs {
        1 + bytes.index(4)
    } else if lane == 1 {
        bytes.choose(&[130, 131])
    } else {
        bytes.choose(&[1, 2, 3, 4, 7, 8, 16, 63, 64, 65, 130, 131])
    };
    let sites = if lane == 2 {
        2 + bytes.index(2)
    } else {
        1 + bytes.index(3)
    };
    let steps = 1 + bytes.index(8);
    let variant = bytes.index(8);
    let mut expectation = if lane == 4 {
        Expectation::GuaranteedRejected
    } else if lane == 5 {
        Expectation::Unspecified
    } else {
        Expectation::GuaranteedCandidate
    };

    let mut builder = FnBuilder::new("constant_shift_choices_sample");
    let input = builder.param("x", Type::Bits(width)).unwrap();
    let predicate_input = builder.param("controls", Type::Bits(3)).unwrap();
    let variable = (lane == 4).then(|| {
        builder
            .param("variable_amount", Type::Bits(amount_width))
            .unwrap()
    });
    let predicates = controls(&mut builder, predicate_input, input, width, &mut bytes);
    let amounts = amount_dag(
        &mut builder,
        amount_width,
        width,
        &predicates,
        steps,
        &mut bytes,
    );
    let mut amount = *amounts.last().unwrap();
    if let Some(variable) = variable {
        amount = rejected_amount(
            &mut builder,
            amount,
            variable,
            amount_width,
            predicates[0],
            variant,
        );
    } else if lane == 6 {
        // These legal select shapes are unsupported by the SMT translator.
        // Keep them in the small exhaustive lane, including their local cuts.
        if variant == 0 {
            amount = builder
                .select(predicates[0], &[], Some(amounts[0]))
                .unwrap();
            expectation = Expectation::GuaranteedRejected;
        } else if variant == 1 {
            let selector = literal(&mut builder, 0, 0);
            amount = builder.select(selector, &[amounts[1]], None).unwrap();
        }
    }

    let mut limits = ConstantShiftChoiceLimits {
        max_distinct_shifts: 64,
        max_visited_nodes: 512,
        max_emitted_nodes: 4096,
        ..ConstantShiftChoiceLimits::default()
    };
    if lane == 5 {
        limits = ConstantShiftChoiceLimits::default();
        match variant % 4 {
            0 => limits.max_distinct_shifts = bytes.choose(&[0, 1, 2, 3, 4, 5]),
            1 => {
                for _ in 0..bytes.choose(&[57, 60, 61, 62, 63, 64, 65]) {
                    amount = builder.identity(amount).unwrap();
                }
            }
            2 => limits.max_emitted_nodes = bytes.choose(&[0, 1, 3, 6, 7, 8, 12, 127, 128, 129]),
            _ => {
                let maximum = width.max(amount_width);
                limits.max_bit_width = bytes.choose(&[maximum - 1, maximum, maximum + 1]);
            }
        }
        // Every site has these data and amount widths. Empty allowances or
        // an excluded operand width rule out even the first candidate.
        if limits.max_distinct_shifts == 0
            || limits.max_visited_nodes == 0
            || limits.max_emitted_nodes == 0
            || limits.max_bit_width < width
            || limits.max_bit_width < amount_width
        {
            expectation = Expectation::GuaranteedRejected;
        }
    }

    let mut data_values = vec![input, builder.not(input).unwrap()];
    if width != 0 {
        let mask = builder.sign_extend(predicates[0], width).unwrap();
        data_values.push(builder.or(input, mask).unwrap());
        data_values.push(builder.xor(input, mask).unwrap());
    }
    let mut outputs = Vec::new();
    for site in 0..sites {
        let data = if lane == 3 && site == 0 && width != 0 {
            data_values[2]
        } else {
            bytes.choose(&data_values)
        };
        let direction = bytes.index(3);
        let shifted = if lane == 4 && variant == 5 {
            builder.shra(data, amount).unwrap()
        } else if direction == 0 {
            builder.shll(data, amount).unwrap()
        } else {
            builder.shrl(data, amount).unwrap()
        };
        let retain = bytes.next();
        if direction == 2 {
            outputs.push(slice(&mut builder, shifted, width, &mut bytes));
            if retain & 1 != 0 {
                outputs.push(shifted);
            }
            if retain & 2 != 0 {
                outputs.push(slice(&mut builder, shifted, width, &mut bytes));
            }
        } else {
            outputs.push(shifted);
        }
        if retain & 4 != 0 {
            outputs.push(amount);
        }
        if retain & 8 != 0 {
            outputs.push(bytes.choose(&amounts));
        }
        data_values.push(shifted);
    }
    let result = builder.tuple(&outputs).unwrap();
    let result = if lane == 7 {
        // Every shift is observable only through this trace. Cover/assert are
        // independent effect roots whose identity and order must also survive.
        let token = builder.after_all(&[]).unwrap();
        let trace = builder
            .trace(token, predicates[0], "shifted={}", &[result], 0)
            .unwrap();
        builder.cover(predicates[1], "covered").unwrap();
        builder
            .assert(trace, predicates[2], "assertion", "checked")
            .unwrap();
        literal(&mut builder, 1, 0)
    } else {
        result
    };
    Sample {
        function: builder
            .build(result)
            .expect("the structured sample is valid IR"),
        limits,
        expectation,
        exhaustive_inputs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_aug_opt::constant_shift_choices::constant_shift_choice_candidate;
    use xlsynth_pir::ir_utils::operands;

    #[test]
    fn deterministic_samples_are_valid_and_only_reference_earlier_nodes() {
        for lane in 0..8 {
            for salt in 0..32u8 {
                let mut data: Vec<_> = (0..128u8)
                    .map(|index| index.wrapping_mul(17).wrapping_add(salt))
                    .collect();
                data[0] = lane;
                let sample = generate_sample(&data);
                assert_eq!(
                    sample.function.to_string(),
                    generate_sample(&data).function.to_string()
                );
                for (index, node) in sample.function.nodes.iter().enumerate() {
                    assert!(operands(&node.payload).iter().all(|arg| arg.index < index));
                }
                if sample.exhaustive_inputs {
                    assert!(
                        sample
                            .function
                            .param_nodes()
                            .map(|node| node.ty.bit_count())
                            .sum::<usize>()
                            <= 7
                    );
                }
                let candidate = constant_shift_choice_candidate(&sample.function, sample.limits);
                match sample.expectation {
                    Expectation::GuaranteedCandidate => assert!(candidate.is_some()),
                    Expectation::GuaranteedRejected => assert!(candidate.is_none()),
                    Expectation::Unspecified => {
                        // Recognition may legitimately stop at the selected
                        // limit.
                    }
                }
            }
        }
    }
}
