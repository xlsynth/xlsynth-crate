// SPDX-License-Identifier: Apache-2.0

//! Width-independent patterned and correlated block stimuli.

use std::collections::BTreeSet;

use rand::Rng;
use xlsynth_pir::ir::{Fn, NodePayload, Type};
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;
use xlsynth_pir::{IrBits, IrValue};

/// Collects small discontinuities from the generated graph, including array
/// bounds, select case counts, and shift/data widths. No circuit is injected.
pub fn relevant_bounds(block: &Fn) -> Vec<usize> {
    let mut bounds = BTreeSet::from([0, 1]);
    for node in &block.nodes {
        let bound = match &node.payload {
            NodePayload::Sel { cases, .. }
            | NodePayload::PrioritySel { cases, .. }
            | NodePayload::OneHotSel { cases, .. } => cases.len(),
            _ => node.ty.bit_count(),
        };
        for value in [bound.saturating_sub(1), bound, bound.saturating_add(1)] {
            bounds.insert(value);
        }
        let mut types = vec![&node.ty];
        while let Some(ty) = types.pop() {
            match ty {
                Type::Array(array) => {
                    let bound = array.element_count;
                    bounds.extend([bound.saturating_sub(1), bound, bound.saturating_add(1)]);
                    types.push(&array.element_type);
                }
                Type::Tuple(fields) => types.extend(fields.iter().map(Box::as_ref)),
                _ => { /* Scalar widths are already represented above. */ }
            }
        }
    }
    bounds.into_iter().collect()
}

/// Generates a complete vector with equal same-type operands in selected
/// samples and otherwise independent values. Four of sixteen slots are uniform.
pub fn inputs<R: Rng>(block: &Fn, rng: &mut R, sample: usize, bounds: &[usize]) -> Vec<IrValue> {
    let mut values: Vec<IrValue> = Vec::with_capacity(block.params.len());
    for (index, param) in block.params.iter().enumerate() {
        let equal_to = (sample % 16 == 11)
            .then(|| {
                block.params[..index]
                    .iter()
                    .position(|other| other.ty == param.ty)
            })
            .flatten();
        let value = if let Some(previous) = equal_to {
            values[previous].clone()
        } else {
            // Alternate extrema and -1, and ones and 1, across operands. These
            // generic correlations exercise overflow and carry propagation.
            let pattern = match sample % 16 {
                9 if index % 2 == 0 => 4,
                9 => 1,
                10 if index % 2 == 0 => 1,
                10 => 6,
                slot => slot,
            };
            value(&param.ty, rng, pattern, bounds)
        };
        values.push(value);
    }
    values
}

/// Recursively generates generic bit patterns at arbitrary widths and shapes.
pub fn value<R: Rng>(ty: &Type, rng: &mut R, pattern: usize, bounds: &[usize]) -> IrValue {
    if pattern >= 11 {
        return generate_uniform_value_with_rng(rng, ty);
    }
    match ty {
        Type::Bits(width) => {
            let position = if *width == 0 {
                0
            } else {
                rng.gen_range(0..*width)
            };
            let bound = bounds[rng.gen_range(0..bounds.len())];
            let bits = (0..*width)
                .map(|bit| match pattern {
                    0 => false,
                    1 => true,
                    2 => bit == position,
                    3 => bit % 2 == 0,
                    4 => bit + 1 == *width,
                    5 => bit + 1 != *width,
                    6 => bit == 0,
                    7 => bit != position,
                    _ => bit < usize::BITS as usize && (bound >> bit) & 1 != 0,
                })
                .collect::<Vec<_>>();
            IrValue::from_bits(&IrBits::from_lsb_is_0(&bits))
        }
        Type::Array(array) => IrValue::make_array(
            &(0..array.element_count)
                .map(|_| value(&array.element_type, rng, pattern, bounds))
                .collect::<Vec<_>>(),
        )
        .expect("generated arrays are nonempty and type-compatible"),
        Type::Tuple(fields) => IrValue::make_tuple(
            &fields
                .iter()
                .map(|field| value(field, rng, pattern, bounds))
                .collect::<Vec<_>>(),
        ),
        Type::Token => IrValue::make_token(),
    }
}

#[cfg(test)]
mod tests {
    use super::value;
    use rand::{SeedableRng, rngs::StdRng};
    use xlsynth_pir::ir::Type;

    #[test]
    fn patterns_include_arbitrary_width_extrema_and_zero_width() {
        let mut rng = StdRng::seed_from_u64(42);
        for width in [0, 1, 8, 65, 129, 256] {
            let ty = Type::Bits(width);
            assert!(
                value(&ty, &mut rng, 0, &[0, 1])
                    .to_bits()
                    .unwrap()
                    .is_zero()
            );
            let ones = value(&ty, &mut rng, 1, &[0, 1]).to_bits().unwrap();
            let min = value(&ty, &mut rng, 4, &[0, 1]).to_bits().unwrap();
            for bit in 0..width {
                assert!(ones.get_bit(bit).unwrap());
                assert_eq!(min.get_bit(bit).unwrap(), bit + 1 == width);
            }
        }
    }
}
