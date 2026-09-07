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

/// Generates reproducible argument sets using per-vector mixed sampling.
pub fn generate_argument_sets_from_seed(
    function: &Fn,
    seed: u64,
    count: usize,
) -> Vec<Vec<IrValue>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    generate_argument_sets_with_rng(function, &mut rng, count)
}

/// Generates argument sets using per-vector mixed sampling.
pub fn generate_argument_sets_with_rng<R: RngCore + ?Sized>(
    function: &Fn,
    rng: &mut R,
    count: usize,
) -> Vec<Vec<IrValue>> {
    generate_mixed_argument_sets_with_rng(function, rng, count)
}

/// Generates independent mixed vectors, choosing a fresh special-value subset
/// for each evaluation. Arguments remain in signature order.
pub fn generate_mixed_argument_sets_with_rng<R: RngCore + ?Sized>(
    function: &Fn,
    rng: &mut R,
    count: usize,
) -> Vec<Vec<IrValue>> {
    (0..count)
        .map(|_| generate_mixed_arguments_with_rng(rng, function))
        .collect()
}

/// Generates one mixed input vector matching a function's parameter types.
pub fn generate_mixed_arguments_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    function: &Fn,
) -> Vec<IrValue> {
    generate_mixed_values_with_rng(rng, function.param_nodes().map(|param| &param.ty))
}

/// Generates one mixed input vector from a stable seed.
pub fn generate_mixed_arguments_from_seed(function: &Fn, seed: u64) -> Vec<IrValue> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    generate_mixed_arguments_with_rng(&mut rng, function)
}

/// Generates one vector in type order: 10% use a whole-vector pattern,
/// otherwise K uniformly chosen positions (K in 0..=N) use special values and
/// the rest are uniform. A vector-wide 50% gate enables independent
/// 1/64-probability bit flips in special-derived values only. Aggregates count
/// as single input positions.
pub fn generate_mixed_values_with_rng<'a, R, I>(rng: &mut R, types: I) -> Vec<IrValue>
where
    R: RngCore + ?Sized,
    I: IntoIterator<Item = &'a Type>,
{
    let types = types.into_iter().collect::<Vec<_>>();
    let kinds = input_kinds(rng, types.len());
    let mut values = {
        let mut source = RngEntropy::new(&mut *rng);
        types
            .into_iter()
            .zip(&kinds)
            .map(|(ty, kind)| match kind {
                InputKind::Uniform => generate_uniform_value(&mut source, ty),
                InputKind::Special => generate_special_value(&mut source, ty),
                InputKind::Pattern(pattern) => generate_pattern_value(ty, *pattern),
            })
            .collect::<Vec<_>>()
    };
    perturb_specials(rng, &kinds, &mut values, perturb_value);
    values
}

/// Generates one flat vector with the same per-position policy as typed values.
pub fn generate_mixed_irbits_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    widths: &[usize],
) -> Vec<IrBits> {
    let kinds = input_kinds(rng, widths.len());
    let mut values = {
        let mut source = RngEntropy::new(&mut *rng);
        widths
            .iter()
            .zip(&kinds)
            .map(|(&width, kind)| match kind {
                InputKind::Uniform => generate_uniform_irbits(&mut source, width),
                InputKind::Special => generate_special_irbits(&mut source, width),
                InputKind::Pattern(pattern) => generate_pattern_irbits(width, *pattern),
            })
            .collect::<Vec<_>>()
    };
    perturb_specials(rng, &kinds, &mut values, perturb_irbits);
    values
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputKind {
    Uniform,
    Special,
    Pattern(BitValuePattern),
}

/// Chooses base-value strategies without changing argument/port order.
fn input_kinds<R: RngCore + ?Sized>(rng: &mut R, count: usize) -> Vec<InputKind> {
    if count == 0 {
        return Vec::new();
    }
    if rng.gen_ratio(1, 10) {
        let pattern = rng.gen_range(0..4);
        return (0..count)
            .map(|index| {
                let one = match pattern {
                    0 => false,
                    1 => true,
                    2 => index % 2 == 0,
                    _ => index % 2 != 0,
                };
                InputKind::Pattern(if one {
                    BitValuePattern::AllOnes
                } else {
                    BitValuePattern::Zero
                })
            })
            .collect();
    }
    special_positions(rng, count)
        .into_iter()
        .map(|special| {
            if special {
                InputKind::Special
            } else {
                InputKind::Uniform
            }
        })
        .collect()
}

/// Applies one vector-wide gate, never perturbing uniform-derived positions.
fn perturb_specials<R: RngCore + ?Sized, T>(
    rng: &mut R,
    kinds: &[InputKind],
    values: &mut [T],
    mut perturb: impl FnMut(&mut R, &T) -> T,
) {
    assert_eq!(kinds.len(), values.len());
    if values.is_empty() || !rng.gen_bool(0.5) {
        return;
    }
    for (kind, value) in kinds.iter().zip(values) {
        if *kind != InputKind::Uniform {
            *value = perturb(rng, value);
        }
    }
}

/// Toggles each valid bit independently; enabling perturbation may flip none.
fn perturb_irbits<R: RngCore + ?Sized>(rng: &mut R, bits: &IrBits) -> IrBits {
    let mut bytes = bits.to_le_bytes();
    for index in 0..bits.get_bit_count() {
        if rng.gen_ratio(1, 64) {
            bytes[index / 8] ^= 1 << (index % 8);
        }
    }
    IrBits::from_le_bytes(bits.get_bit_count(), &bytes).expect("valid bits remain in width")
}

/// Perturbs every leaf while preserving token and empty-aggregate shapes.
fn perturb_value<R: RngCore + ?Sized>(rng: &mut R, value: &IrValue) -> IrValue {
    match value {
        IrValue::Bits(bits) => IrValue::from_bits(&perturb_irbits(rng, bits)),
        IrValue::Token => IrValue::make_token(),
        IrValue::Tuple(fields) => IrValue::make_tuple(
            &fields
                .iter()
                .map(|value| perturb_value(rng, value))
                .collect::<Vec<_>>(),
        ),
        IrValue::Array(array) => IrValue::make_array_typed(
            array.element_type().clone(),
            &array
                .elements()
                .iter()
                .map(|value| perturb_value(rng, value))
                .collect::<Vec<_>>(),
        )
        .expect("perturbation preserves array element types"),
    }
}

/// Chooses a uniform subset conditional on a uniformly selected cardinality.
fn special_positions<R: RngCore + ?Sized>(rng: &mut R, count: usize) -> Vec<bool> {
    if count == 0 {
        return Vec::new();
    }
    let special_count = rng.gen_range(0..=count);
    let mut positions = vec![false; count];
    positions[..special_count].fill(true);
    positions.shuffle(rng);
    positions
}

/// Generates a type-shaped special value without a uniform-sampling fallback.
fn generate_special_value<S: EntropySource>(source: &mut S, ty: &Type) -> IrValue {
    match ty {
        Type::Bits(width) => IrValue::from_bits(&generate_special_irbits(source, *width)),
        Type::Token => IrValue::make_token(),
        Type::Tuple(fields) => IrValue::make_tuple(
            &fields
                .iter()
                .map(|ty| generate_special_value(source, ty))
                .collect::<Vec<_>>(),
        ),
        Type::Array(array) => IrValue::make_array_typed(
            (*array.element_type).clone(),
            &(0..array.element_count)
                .map(|_| generate_special_value(source, &array.element_type))
                .collect::<Vec<_>>(),
        )
        .expect("special array values preserve their element type"),
    }
}

/// Chooses generic boundary patterns, including arbitrary-width extrema.
fn generate_special_irbits<S: EntropySource>(source: &mut S, width: usize) -> IrBits {
    let choice = source.take_u64() % 11;
    if choice == 9 {
        return generate_pattern_irbits(width, BitValuePattern::OneHot(0));
    }
    if choice == 10 {
        return generate_pattern_irbits(
            width,
            BitValuePattern::OneHot(choose_count(source, width)),
        )
        .not();
    }
    let pattern = match choice {
        0 => BitValuePattern::Zero,
        1 => BitValuePattern::AllOnes,
        2 => BitValuePattern::SignedMin,
        3 => BitValuePattern::SignedMax,
        4 => BitValuePattern::OneHot(choose_count(source, width)),
        5 => BitValuePattern::LowOnes(choose_between(source, 0, width)),
        6 => BitValuePattern::HighOnes(choose_between(source, 0, width)),
        7 => BitValuePattern::Alternating { lsb_is_one: true },
        _ => BitValuePattern::Alternating { lsb_is_one: false },
    };
    generate_pattern_irbits(width, pattern)
}

/// Generates flat bitvector argument sets using per-vector mixed sampling.
pub fn generate_flat_bitvector_argument_sets_with_rng<R: RngCore + ?Sized>(
    rng: &mut R,
    input_widths: &[usize],
    count: usize,
) -> Vec<Vec<IrBits>> {
    (0..count)
        .map(|_| generate_mixed_irbits_with_rng(rng, input_widths))
        .collect()
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

    use rand::{RngCore, SeedableRng, rngs::mock::StepRng};
    use rand_pcg::Pcg64Mcg;

    use super::{
        BitValuePattern, InputKind, generate_corner_irbits, generate_mixed_argument_sets_with_rng,
        generate_mixed_arguments_with_rng, generate_mixed_irbits_with_rng,
        generate_mixed_values_with_rng, input_kinds, perturb_irbits, perturb_specials,
        perturb_value, special_positions,
    };
    use crate::ir::Type;
    use crate::{FnBuilder, IrBits, IrValue};

    #[test]
    fn special_positions_cover_zero_through_all_without_reordering_arguments() {
        for count in 0..=8 {
            let mut cardinalities = BTreeSet::new();
            let mut frequencies = vec![0; count + 1];
            let mut subsets = BTreeSet::new();
            for seed in 0..1024 {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                let positions = special_positions(&mut rng, count);
                assert_eq!(positions.len(), count);
                let special_count = positions.iter().filter(|&&special| special).count();
                cardinalities.insert(special_count);
                frequencies[special_count] += 1;
                subsets.insert(positions);
            }
            assert_eq!(cardinalities, (0..=count).collect());
            // A deterministic broad check distinguishes uniform K from
            // independently marking each argument special (binomial K).
            let expected = 1024 / (count + 1);
            for frequency in frequencies {
                assert!((expected / 2..=expected * 2).contains(&frequency));
            }
            if count <= 3 {
                assert_eq!(subsets.len(), 1 << count);
            }
        }
    }

    #[test]
    fn typed_and_flat_vectors_share_policy_and_rng_draws() {
        let widths = [129, 65, 257, 0];
        let types = widths.map(Type::Bits);
        for seed in 0..128 {
            let mut rng = Pcg64Mcg::seed_from_u64(seed);
            let mut flat_rng = rng.clone();
            let values = generate_mixed_values_with_rng(&mut rng, &types);
            let flat = generate_mixed_irbits_with_rng(&mut flat_rng, &widths);
            assert_eq!(rng.next_u64(), flat_rng.next_u64());
            for (index, (value, bits)) in values.iter().zip(&flat).enumerate() {
                assert_eq!(value.as_bits().unwrap(), bits);
                assert_eq!(bits.get_bit_count(), widths[index]);
            }
        }
    }

    #[test]
    fn structured_vectors_alternate_by_argument_not_by_bit() {
        let mut observed = BTreeSet::new();
        let mut mixed_seen = false;
        let mut structured_count = 0;
        for seed in 0..4096 {
            let kinds = input_kinds(&mut Pcg64Mcg::seed_from_u64(seed), 3);
            if kinds
                .iter()
                .all(|kind| matches!(kind, InputKind::Pattern(_)))
            {
                structured_count += 1;
                observed.insert(
                    kinds
                        .iter()
                        .map(|kind| match kind {
                            InputKind::Pattern(BitValuePattern::Zero) => false,
                            InputKind::Pattern(BitValuePattern::AllOnes) => true,
                            _ => panic!("whole-vector patterns must fill entire arguments"),
                        })
                        .collect::<Vec<_>>(),
                );
            } else {
                mixed_seen = true;
            }
        }
        assert_eq!(
            observed,
            BTreeSet::from([
                vec![false, false, false],
                vec![true, true, true],
                vec![true, false, true],
                vec![false, true, false],
            ])
        );
        assert!(mixed_seen);
        // Fixed seeds keep this non-flaky; the broad interval catches a
        // policy regression without depending on an exact RNG histogram.
        assert!((328..=492).contains(&structured_count));
    }

    #[test]
    fn one_gate_controls_all_special_positions_and_never_uniform_ones() {
        let kinds = [
            InputKind::Special,
            InputKind::Uniform,
            InputKind::Pattern(BitValuePattern::Zero),
        ];
        let original = vec![IrBits::zero(129); 3];
        let mut unchanged = original.clone();
        perturb_specials(
            &mut StepRng::new(u64::MAX, 0),
            &kinds,
            &mut unchanged,
            perturb_irbits,
        );
        assert_eq!(unchanged, original);
        let mut changed = original.clone();
        perturb_specials(
            &mut StepRng::new(0, 0),
            &kinds,
            &mut changed,
            perturb_irbits,
        );
        assert_eq!(
            changed,
            vec![
                IrBits::all_ones(129),
                IrBits::zero(129),
                IrBits::all_ones(129)
            ]
        );
        // The gate succeeds, but every later draw is above the bit threshold.
        // Enabling perturbation must not force a flip.
        perturb_specials(
            &mut StepRng::new(0, u64::MAX),
            &kinds,
            &mut unchanged,
            perturb_irbits,
        );
        assert_eq!(unchanged, original);
    }

    #[test]
    fn per_bit_probability_and_aggregate_perturbation_preserve_widths() {
        let changed = perturb_irbits(&mut StepRng::new(0, 1u64 << 58), &IrBits::zero(129));
        for bit in 0..129 {
            assert_eq!(changed.get_bit(bit).unwrap(), bit % 64 == 0);
        }
        let aggregate = IrValue::make_tuple(&[
            IrValue::make_token(),
            IrValue::from_bits(&IrBits::zero(0)),
            IrValue::make_array_typed(Type::Bits(257), &[]).unwrap(),
            IrValue::make_array(&[IrValue::from_bits(&IrBits::zero(65))]).unwrap(),
        ]);
        let result = perturb_value(&mut StepRng::new(0, 0), &aggregate);
        assert_eq!(result.type_(), aggregate.type_());
        assert_eq!(
            result.as_elements().unwrap()[3].as_elements().unwrap()[0]
                .as_bits()
                .unwrap(),
            &IrBits::all_ones(65),
        );
    }

    #[test]
    fn batches_choose_a_fresh_subset_per_evaluation() {
        let mut builder = FnBuilder::new("per_vector_selection");
        let lhs = builder.param("lhs", Type::Bits(129)).unwrap();
        let rhs = builder.param("rhs", Type::Bits(65)).unwrap();
        let result = builder.tuple(&[lhs, rhs]).unwrap();
        let function = builder.build(result).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let mut replay = rng.clone();
        assert_eq!(
            generate_mixed_argument_sets_with_rng(&function, &mut rng, 16),
            (0..16)
                .map(|_| generate_mixed_arguments_with_rng(&mut replay, &function))
                .collect::<Vec<_>>(),
        );
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
