// SPDX-License-Identifier: Apache-2.0

//! Small bounded truth-table helpers used by the final technology mapper.

use crate::aig::AigRef;

/// Largest cut size supported by the compact u64 truth-table encoding.
pub(super) const MAX_TRUTH_TABLE_INPUTS: usize = 6;

/// Returns the mask covering every assignment bit for input_count inputs.
pub(super) fn truth_mask(input_count: usize) -> u64 {
    debug_assert!(input_count <= MAX_TRUTH_TABLE_INPUTS);
    let assignment_count = 1usize << input_count;
    if assignment_count == u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << assignment_count) - 1
    }
}

/// Returns the truth table of variable variable_index.
pub(super) fn variable_truth(input_count: usize, variable_index: usize) -> u64 {
    debug_assert!(input_count <= MAX_TRUTH_TABLE_INPUTS);
    debug_assert!(variable_index < input_count);
    let assignment_count = 1usize << input_count;
    let mut truth = 0u64;
    for assignment in 0..assignment_count {
        if ((assignment >> variable_index) & 1) != 0 {
            truth |= 1u64 << assignment;
        }
    }
    truth
}

/// Complements a truth table while keeping unused high bits clear.
pub(super) fn complement_truth(truth: u64, input_count: usize) -> u64 {
    (!truth) & truth_mask(input_count)
}

/// Re-expresses a truth table over a sorted superset of leaves.
pub(super) fn remap_truth(truth: u64, old_leaves: &[AigRef], new_leaves: &[AigRef]) -> u64 {
    debug_assert!(old_leaves.len() <= new_leaves.len());
    debug_assert!(new_leaves.len() <= MAX_TRUTH_TABLE_INPUTS);
    let old_positions: Vec<usize> = old_leaves
        .iter()
        .map(|leaf| {
            new_leaves
                .binary_search(leaf)
                .expect("old cut leaf should be present in merged cut")
        })
        .collect();
    let mut remapped = 0u64;
    for new_assignment in 0..(1usize << new_leaves.len()) {
        let mut old_assignment = 0usize;
        for (old_index, new_index) in old_positions.iter().enumerate() {
            if ((new_assignment >> new_index) & 1) != 0 {
                old_assignment |= 1usize << old_index;
            }
        }
        if ((truth >> old_assignment) & 1) != 0 {
            remapped |= 1u64 << new_assignment;
        }
    }
    remapped
}

/// Removes leaves that the truth table does not actually depend on.
///
/// ABC NF minimizes the support after composing each cut truth table. Doing
/// the same here is important both for cut priority and for matching a
/// smaller-arity Liberty root when a merged structural cut collapses.
pub(super) fn minimize_support(truth: u64, leaves: &[AigRef]) -> (u64, Vec<AigRef>) {
    debug_assert!(leaves.len() <= MAX_TRUTH_TABLE_INPUTS);
    let input_count = leaves.len();
    let mut kept_indices = Vec::new();
    for input_index in 0..input_count {
        let input_bit = 1usize << input_index;
        let depends_on_input = (0..(1usize << input_count))
            .filter(|assignment| assignment & input_bit == 0)
            .any(|assignment| {
                ((truth >> assignment) & 1) != ((truth >> (assignment | input_bit)) & 1)
            });
        if depends_on_input {
            kept_indices.push(input_index);
        }
    }
    if kept_indices.len() == input_count {
        return (truth & truth_mask(input_count), leaves.to_vec());
    }

    let minimized_leaves = kept_indices
        .iter()
        .map(|input_index| leaves[*input_index])
        .collect::<Vec<_>>();
    let mut minimized_truth = 0u64;
    for new_assignment in 0..(1usize << kept_indices.len()) {
        let mut old_assignment = 0usize;
        for (new_index, old_index) in kept_indices.iter().copied().enumerate() {
            if ((new_assignment >> new_index) & 1) != 0 {
                old_assignment |= 1usize << old_index;
            }
        }
        if ((truth >> old_assignment) & 1) != 0 {
            minimized_truth |= 1u64 << new_assignment;
        }
    }
    (minimized_truth, minimized_leaves)
}

/// Applies pin permutation and per-input polarity to a cell truth table.
///
/// input_negated[input_index] records whether that cell pin sees the
/// complement of its selected cut leaf.
pub(super) fn transform_truth(truth: u64, input_to_leaf: &[usize], input_negated: &[bool]) -> u64 {
    debug_assert!(input_to_leaf.len() <= MAX_TRUTH_TABLE_INPUTS);
    debug_assert_eq!(input_to_leaf.len(), input_negated.len());
    let input_count = input_to_leaf.len();
    let mut permuted = 0u64;
    for leaf_assignment in 0..(1usize << input_count) {
        let mut input_assignment = 0usize;
        for (input_index, leaf_index) in input_to_leaf.iter().enumerate() {
            let leaf_value = ((leaf_assignment >> leaf_index) & 1) != 0;
            if leaf_value ^ input_negated[input_index] {
                input_assignment |= 1usize << input_index;
            }
        }
        if ((truth >> input_assignment) & 1) != 0 {
            permuted |= 1u64 << leaf_assignment;
        }
    }
    permuted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_truth_uses_lsb_first_assignments() {
        assert_eq!(variable_truth(1, 0), 0b10);
        assert_eq!(variable_truth(2, 0), 0b1010);
        assert_eq!(variable_truth(2, 1), 0b1100);
    }

    #[test]
    fn permutation_swaps_variable_roles() {
        let a_and_not_b = variable_truth(2, 0) & complement_truth(variable_truth(2, 1), 2);
        let swapped = transform_truth(a_and_not_b, &[1, 0], &[false, false]);
        let b_and_not_a = variable_truth(2, 1) & complement_truth(variable_truth(2, 0), 2);
        assert_eq!(swapped, b_and_not_a);
    }

    #[test]
    fn transform_applies_input_polarity() {
        let and = variable_truth(2, 0) & variable_truth(2, 1);
        let a_and_not_b = transform_truth(and, &[0, 1], &[false, true]);
        assert_eq!(a_and_not_b, 0b0010);
    }

    #[test]
    fn minimize_support_removes_unused_inputs() {
        let leaves = vec![AigRef { id: 3 }, AigRef { id: 7 }];
        let truth = variable_truth(2, 1);

        let (minimized_truth, minimized_leaves) = minimize_support(truth, leaves.as_slice());

        assert_eq!(minimized_leaves, vec![AigRef { id: 7 }]);
        assert_eq!(minimized_truth, variable_truth(1, 0));
    }
}
