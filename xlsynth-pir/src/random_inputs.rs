// SPDX-License-Identifier: Apache-2.0

//! Reusable runtime input generation for simulation, fuzzing, and semantic
//! checks.

use rand::{RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;
use xlsynth::{IrBits, IrValue};

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
            IrValue::make_array(&values).expect("generated array values have identical types")
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
        .params
        .iter()
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
            IrValue::make_array(&values).expect("generated array values have identical types")
        }
    }
}

/// Generates biased inputs matching a function's parameters.
pub fn generate_biased_arguments<S: EntropySource>(source: &mut S, function: &Fn) -> Vec<IrValue> {
    function
        .params
        .iter()
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
            IrValue::make_array(&values).expect("generated array values have identical types")
        }
    }
}

/// Generates inputs whose leaves use the requested corner pattern.
pub fn generate_pattern_arguments(function: &Fn, pattern: BitValuePattern) -> Vec<IrValue> {
    function
        .params
        .iter()
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
