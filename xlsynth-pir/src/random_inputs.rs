// SPDX-License-Identifier: Apache-2.0

//! Reusable runtime input generation for simulation, fuzzing, and semantic
//! checks.

use crate::{IrBits, IrValue};
use rand::{Rng, RngCore, SeedableRng, seq::SliceRandom};
use rand_pcg::Pcg64Mcg;

use crate::ir::{Fn, Type};
use crate::ir_random::{EntropySource, RngEntropy};

/// Generates a uniformly distributed typed value.
pub fn generate_uniform_value<S: EntropySource>(source: &mut S, ty: &Type) -> IrValue {
    match ty {
        Type::Token => IrValue::make_token(),
        Type::Bits(width) => IrValue::from_bits(&generate_uniform_irbits(source, *width)),
        Type::Tuple(elements) => {
            let values: Vec<IrValue> = elements
                .iter()
                .map(|element| generate_uniform_value(source, element))
                .collect();
            IrValue::make_tuple(&values)
        }
        Type::Array(array) => {
            let values: Vec<IrValue> = (0..array.element_count)
                .map(|_| generate_uniform_value(source, &array.element_type))
                .collect();
            IrValue::make_array_typed((*array.element_type).clone(), &values)
                .expect("generated array values have identical types")
        }
    }
}

/// Generates a uniformly distributed typed value from an RNG.
pub fn generate_uniform_value_with_rng<R: RngCore + ?Sized>(rng: &mut R, ty: &Type) -> IrValue {
    let mut source = RngEntropy::new(rng);
    generate_uniform_value(&mut source, ty)
}

/// Generates uniformly distributed inputs matching a function's parameters.
pub fn generate_uniform_arguments<S: EntropySource>(source: &mut S, function: &Fn) -> Vec<IrValue> {
    function
        .param_nodes()
        .map(|param| generate_uniform_value(source, &param.ty))
        .collect()
}

/// Generates uniformly distributed inputs matching a function's parameters
/// from an RNG.
pub fn generate_uniform_arguments_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    function: &Fn,
) -> Vec<IrValue> {
    let mut source = RngEntropy::new(rng);
    generate_uniform_arguments(&mut source, function)
}

/// Generates a typed value with a bias toward useful bitvector corner cases.
pub fn generate_biased_value<S: EntropySource>(source: &mut S, ty: &Type) -> IrValue {
    match ty {
        Type::Token => IrValue::make_token(),
        Type::Bits(width) => IrValue::from_bits(&generate_biased_irbits(source, *width)),
        Type::Tuple(elements) => {
            let values: Vec<IrValue> = elements
                .iter()
                .map(|element| generate_biased_value(source, element))
                .collect();
            IrValue::make_tuple(&values)
        }
        Type::Array(array) => {
            let values: Vec<IrValue> = (0..array.element_count)
                .map(|_| generate_biased_value(source, &array.element_type))
                .collect();
            IrValue::make_array_typed((*array.element_type).clone(), &values)
                .expect("generated array values have identical types")
        }
    }
}

/// Generates biased inputs matching a function's parameters.
pub fn generate_biased_arguments<S: EntropySource>(source: &mut S, function: &Fn) -> Vec<IrValue> {
    function
        .param_nodes()
        .map(|param| generate_biased_value(source, &param.ty))
        .collect()
}

/// Generates a biased typed value from an RNG.
pub fn generate_biased_value_with_rng<R: RngCore + ?Sized>(rng: &mut R, ty: &Type) -> IrValue {
    let mut source = RngEntropy::new(rng);
    generate_biased_value(&mut source, ty)
}

/// Generates biased inputs matching a function's parameters from an RNG.
pub fn generate_biased_arguments_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    function: &Fn,
) -> Vec<IrValue> {
    let mut source = RngEntropy::new(rng);
    generate_biased_arguments(&mut source, function)
}

/// Generates biased inputs matching a function's parameters from a stable
/// seed.
pub fn generate_biased_arguments_from_seed(function: &Fn, seed: u64) -> Vec<IrValue> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    generate_biased_arguments_with_rng(&mut rng, function)
}

/// Generates reproducible argument sets, starting with whole-input corner
/// cases.
pub fn generate_argument_sets_from_seed(
    function: &Fn,
    seed: u64,
    count: usize,
) -> Vec<Vec<IrValue>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    generate_argument_sets_with_rng(function, &mut rng, count)
}

/// Generates argument sets from an RNG, starting with whole-input corner
/// cases.
pub fn generate_argument_sets_with_rng<R: RngCore + ?Sized>(
    function: &Fn,
    rng: &mut R,
    count: usize,
) -> Vec<Vec<IrValue>> {
    let mut sets = Vec::with_capacity(count);
    if count > 0 {
        sets.push(generate_pattern_arguments(function, BitValuePattern::Zero));
    }
    if count > 1 {
        sets.push(generate_pattern_arguments(
            function,
            BitValuePattern::AllOnes,
        ));
    }
    let mut source = RngEntropy::new(rng);
    while sets.len() < count {
        sets.push(generate_biased_arguments(&mut source, function));
    }
    sets
}

/// Generates a shuffled, randomly proportioned mix of uniform and corner-biased
/// argument sets. Budgets of at least two use both sampling strategies; a
/// one-set budget randomly chooses one. Each biased leaf independently chooses
/// a corner pattern or uniform value, allowing mixed cases within one vector.
pub fn generate_mixed_argument_sets_with_rng<R: RngCore + ?Sized>(
    function: &Fn,
    rng: &mut R,
    count: usize,
) -> Vec<Vec<IrValue>> {
    let biased_count = mixed_biased_count(rng, count);
    let mut sets = Vec::with_capacity(count);
    for _ in 0..biased_count {
        sets.push(generate_biased_arguments_with_rng(rng, function));
    }
    for _ in biased_count..count {
        sets.push(generate_uniform_arguments_with_rng(rng, function));
    }
    sets.shuffle(rng);
    sets
}

/// Allocates a bounded sample budget between the two generation strategies.
fn mixed_biased_count<R: RngCore + ?Sized>(rng: &mut R, count: usize) -> usize {
    match count {
        0 => 0,
        1 => usize::from(rng.gen_bool(0.5)),
        _ => rng.gen_range(1..count),
    }
}

/// Generates flat bitvector argument sets from an RNG, starting with
/// whole-input corner cases.
pub fn generate_flat_bitvector_argument_sets_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    input_widths: &[usize],
    count: usize,
) -> Vec<Vec<IrBits>> {
    let mut sets = Vec::with_capacity(count);
    if count > 0 {
        sets.push(
            input_widths
                .iter()
                .map(|width| generate_pattern_irbits(*width, BitValuePattern::Zero))
                .collect(),
        );
    }
    if count > 1 {
        sets.push(
            input_widths
                .iter()
                .map(|width| generate_pattern_irbits(*width, BitValuePattern::AllOnes))
                .collect(),
        );
    }
    let mut source = RngEntropy::new(rng);
    while sets.len() < count {
        sets.push(
            input_widths
                .iter()
                .map(|width| generate_biased_irbits(&mut source, *width))
                .collect(),
        );
    }
    sets
}

/// Generates reproducible flat bitvector argument sets.
pub fn generate_flat_bitvector_argument_sets_from_seed(
    input_widths: &[usize],
    seed: u64,
    count: usize,
) -> Vec<Vec<IrBits>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    generate_flat_bitvector_argument_sets_with_rng(&mut rng, input_widths, count)
}

/// Generates uniformly distributed bits from an entropy source.
pub fn generate_uniform_irbits<S: EntropySource>(source: &mut S, width: usize) -> IrBits {
    if width == 0 {
        return IrBits::make_ubits(0, 0).expect("bits[0] zero literal must construct");
    }
    let mut bytes = vec![0_u8; width.div_ceil(8)];
    for chunk in bytes.chunks_mut(8) {
        let word = source.take_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    if !width.is_multiple_of(8) {
        let mask = (1_u8 << (width % 8)) - 1;
        *bytes.last_mut().expect("nonzero bit width has storage") &= mask;
    }
    IrBits::from_le_bytes(width, &bytes).expect("valid generated bit representation")
}

/// Generates uniformly distributed bits from an RNG.
pub fn generate_uniform_irbits_with_rng<R: RngCore + ?Sized>(rng: &mut R, width: usize) -> IrBits {
    let mut source = RngEntropy::new(rng);
    generate_uniform_irbits(&mut source, width)
}

/// A useful bitvector corner pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitValuePattern {
    Zero,
    AllOnes,
    SignedMin,
    SignedMax,
    OneHot(usize),
    LowOnes(usize),
    HighOnes(usize),
    Alternating { lsb_is_one: bool },
}

/// Generates bits matching a requested corner pattern.
pub fn generate_pattern_irbits(width: usize, pattern: BitValuePattern) -> IrBits {
    match pattern {
        BitValuePattern::Zero => IrBits::zero(width),
        BitValuePattern::AllOnes => IrBits::all_ones(width),
        BitValuePattern::SignedMin => IrBits::signed_min_value(width),
        BitValuePattern::SignedMax => IrBits::signed_max_value(width),
        BitValuePattern::OneHot(bit_index) => {
            let mut bits = vec![false; width];
            if bit_index < width {
                bits[bit_index] = true;
            }
            IrBits::from_lsb_is_0(&bits)
        }
        BitValuePattern::LowOnes(one_count) => {
            let mut bits = vec![false; width];
            bits[..one_count.min(width)].fill(true);
            IrBits::from_lsb_is_0(&bits)
        }
        BitValuePattern::HighOnes(one_count) => {
            let mut bits = vec![false; width];
            bits[width.saturating_sub(one_count)..].fill(true);
            IrBits::from_lsb_is_0(&bits)
        }
        BitValuePattern::Alternating { lsb_is_one } => {
            let bits: Vec<bool> = (0..width)
                .map(|bit_index| (bit_index % 2 == 0) == lsb_is_one)
                .collect();
            IrBits::from_lsb_is_0(&bits)
        }
    }
}

/// Generates a typed value whose leaves use the requested corner pattern.
pub fn generate_pattern_value(ty: &Type, pattern: BitValuePattern) -> IrValue {
    match ty {
        Type::Token => IrValue::make_token(),
        Type::Bits(width) => IrValue::from_bits(&generate_pattern_irbits(*width, pattern)),
        Type::Tuple(elements) => {
            let values: Vec<IrValue> = elements
                .iter()
                .map(|element| generate_pattern_value(element, pattern))
                .collect();
            IrValue::make_tuple(&values)
        }
        Type::Array(array) => {
            let values: Vec<IrValue> = (0..array.element_count)
                .map(|_| generate_pattern_value(&array.element_type, pattern))
                .collect();
            IrValue::make_array_typed((*array.element_type).clone(), &values)
                .expect("generated array values have identical types")
        }
    }
}

/// Generates inputs whose leaves use the requested corner pattern.
pub fn generate_pattern_arguments(function: &Fn, pattern: BitValuePattern) -> Vec<IrValue> {
    function
        .param_nodes()
        .map(|param| generate_pattern_value(&param.ty, pattern))
        .collect()
}

/// Generates bits biased toward useful corner patterns.
pub fn generate_biased_irbits<S: EntropySource>(source: &mut S, width: usize) -> IrBits {
    match source.take_u64() % 14 {
        0 => generate_pattern_irbits(width, BitValuePattern::Zero),
        1 => generate_pattern_irbits(width, BitValuePattern::AllOnes),
        2 => generate_pattern_irbits(width, BitValuePattern::SignedMin),
        3 => generate_pattern_irbits(width, BitValuePattern::SignedMax),
        4 => generate_pattern_irbits(width, BitValuePattern::OneHot(choose_count(source, width))),
        5 => generate_pattern_irbits(
            width,
            BitValuePattern::LowOnes(choose_between(source, 0, width)),
        ),
        6 => generate_pattern_irbits(
            width,
            BitValuePattern::HighOnes(choose_between(source, 0, width)),
        ),
        _ => generate_uniform_irbits(source, width),
    }
}

/// Generates bits biased toward useful corner patterns from an RNG.
pub fn generate_biased_irbits_with_rng<R: RngCore + ?Sized>(rng: &mut R, width: usize) -> IrBits {
    let mut source = RngEntropy::new(rng);
    generate_biased_irbits(&mut source, width)
}

/// Generates a reusable corpus of bitvector corner cases.
pub fn generate_corner_irbits(width: usize) -> Vec<IrBits> {
    let mut values = Vec::new();
    let mut push_unique = |pattern| {
        let bits = generate_pattern_irbits(width, pattern);
        if !values.contains(&bits) {
            values.push(bits);
        }
    };
    for pattern in [
        BitValuePattern::Zero,
        BitValuePattern::AllOnes,
        BitValuePattern::SignedMin,
        BitValuePattern::SignedMax,
        BitValuePattern::Alternating { lsb_is_one: true },
        BitValuePattern::Alternating { lsb_is_one: false },
    ] {
        push_unique(pattern);
    }
    for bit_index in 0..width {
        push_unique(BitValuePattern::OneHot(bit_index));
    }
    for one_count in 0..=width {
        push_unique(BitValuePattern::LowOnes(one_count));
        push_unique(BitValuePattern::HighOnes(one_count));
    }
    values
}

fn choose_count<S: EntropySource>(source: &mut S, exclusive_limit: usize) -> usize {
    if exclusive_limit <= 1 {
        0
    } else {
        (source.take_u64() as usize) % exclusive_limit
    }
}

fn choose_between<S: EntropySource>(source: &mut S, minimum: usize, maximum: usize) -> usize {
    debug_assert!(minimum <= maximum);
    minimum + choose_count(source, maximum - minimum + 1)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::{RngCore, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::{
        generate_corner_irbits, generate_mixed_argument_sets_with_rng, mixed_biased_count,
    };
    use crate::FnBuilder;
    use crate::ir::Type;

    #[test]
    fn mixed_sample_budget_uses_both_strategies_and_varies_the_ratio() {
        let mut singleton_counts = BTreeSet::new();
        let mut mixed_counts = BTreeSet::new();
        for seed in 0..128 {
            let mut rng = Pcg64Mcg::seed_from_u64(seed);
            assert_eq!(mixed_biased_count(&mut rng, 0), 0);
            assert_eq!(mixed_biased_count(&mut rng, 2), 1);
            singleton_counts.insert(mixed_biased_count(&mut rng, 1));
            mixed_counts.insert(mixed_biased_count(&mut rng, 8));
        }
        assert_eq!(singleton_counts, BTreeSet::from([0, 1]));
        assert_eq!(mixed_counts, (1..8).collect());
    }

    #[test]
    fn mixed_arguments_replay_and_preserve_wide_aggregate_types() {
        let types = [
            Type::Bits(129),
            Type::Tuple(vec![
                Box::new(Type::Bits(0)),
                Box::new(Type::new_array(Type::Bits(65), 2)),
                Box::new(Type::Token),
            ]),
            Type::new_array(Type::Bits(257), 0),
        ];
        let mut builder = FnBuilder::new("mixed_input_shapes");
        let params = types
            .iter()
            .enumerate()
            .map(|(i, ty)| builder.param(&format!("p{i}"), ty.clone()).unwrap())
            .collect::<Vec<_>>();
        let result = builder.tuple(&params).unwrap();
        let function = builder.build(result).unwrap();
        for seed in 0..16 {
            for count in [0, 1, 2, 8] {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                let mut replay = rng.clone();
                let sets = generate_mixed_argument_sets_with_rng(&function, &mut rng, count);
                assert_eq!(
                    sets,
                    generate_mixed_argument_sets_with_rng(&function, &mut replay, count),
                );
                assert_eq!(sets.len(), count);
                for args in sets {
                    assert_eq!(args.len(), types.len());
                    for (arg, ty) in args.iter().zip(&types) {
                        assert_eq!(&arg.type_(), ty);
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_arguments_handle_empty_budgets_and_nullary_functions() {
        let mut builder = FnBuilder::new("no_inputs");
        let result = builder.tuple(&[]).unwrap();
        let function = builder.build(result).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let mut untouched = rng.clone();
        assert!(generate_mixed_argument_sets_with_rng(&function, &mut rng, 0).is_empty());
        assert_eq!(rng.next_u64(), untouched.next_u64());
        for count in [1, 2, 8] {
            assert_eq!(
                generate_mixed_argument_sets_with_rng(&function, &mut rng, count),
                vec![vec![]; count],
            );
        }
    }

    #[test]
    fn mixed_arguments_exercise_corners_and_mixed_parameter_values() {
        let mut builder = FnBuilder::new("mixed_input_values");
        let lhs = builder.param("lhs", Type::Bits(129)).unwrap();
        let rhs = builder.param("rhs", Type::Bits(129)).unwrap();
        let result = builder.tuple(&[lhs, rhs]).unwrap();
        let function = builder.build(result).unwrap();
        let corners = generate_corner_irbits(129);
        let mut seen_zero = false;
        let mut seen_all_ones = false;
        let mut seen_non_corner = false;
        let mut seen_mixed_parameters = false;
        for seed in 0..64 {
            let mut rng = Pcg64Mcg::seed_from_u64(seed);
            for args in generate_mixed_argument_sets_with_rng(&function, &mut rng, 8) {
                let lhs = args[0].as_bits().unwrap();
                let rhs = args[1].as_bits().unwrap();
                for bits in [lhs, rhs] {
                    seen_zero |= bits.is_zero();
                    seen_all_ones |= bits.not().is_zero();
                    seen_non_corner |= !corners.contains(bits);
                }
                seen_mixed_parameters |= (lhs.is_zero() && !corners.contains(rhs))
                    || (rhs.is_zero() && !corners.contains(lhs));
            }
        }
        assert!(seen_zero && seen_all_ones && seen_non_corner && seen_mixed_parameters);
    }
}
