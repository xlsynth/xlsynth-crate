// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigBitVector, AigOperand, GateFn};
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive: Inverts a single bit in one of the primary output ports.
/// Also inverts the `negated` attribute on the corresponding `AigOperand`.
fn do_toggle_output_bit(g: &mut GateFn, output_idx: usize, bit_idx: usize) -> Result<()> {
    if output_idx >= g.outputs.len() {
        return Err(anyhow!(
            "Output index {} out of bounds ({} outputs)",
            output_idx,
            g.outputs.len()
        ));
    }
    let output_spec = &mut g.outputs[output_idx];
    if bit_idx >= output_spec.bit_vector.get_bit_count() {
        return Err(anyhow!(
            "Bit index {} out of bounds for output '{}' ({} bits)",
            bit_idx,
            output_spec.name,
            output_spec.bit_vector.get_bit_count()
        ));
    }

    let mut current_ops: Vec<AigOperand> =
        output_spec.bit_vector.iter_lsb_to_msb().copied().collect();
    if bit_idx < current_ops.len() {
        current_ops[bit_idx].negated = !current_ops[bit_idx].negated;
        output_spec.bit_vector = AigBitVector::from_lsb_is_index_0(&current_ops);
        Ok(())
    } else {
        Err(anyhow!(
            "Bit index {} out of bounds for collected ops ({} ops) for output '{}'",
            bit_idx,
            current_ops.len(),
            output_spec.name
        ))
    }
}

#[derive(Debug)]
pub struct ToggleOutputBitTransform;

impl ToggleOutputBitTransform {
    pub fn new() -> Self {
        ToggleOutputBitTransform
    }
}

impl Transform for ToggleOutputBitTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::ToggleOutputBit
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        log::trace!(
            "Finding candidates for ToggleOutputBitTransform; direction: {:?}",
            direction
        );
        let mut candidates = Vec::new();
        for (output_idx, output_spec) in g.outputs.iter().enumerate() {
            for bit_idx in 0..output_spec.bit_vector.get_bit_count() {
                candidates.push(TransformLocation::OutputPortBit {
                    output_idx,
                    bit_idx,
                });
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection, // This transform is its own inverse
    ) -> Result<()> {
        log::trace!(
            "Applying ToggleOutputBitTransform to {:?}",
            candidate_location
        );
        match candidate_location {
            TransformLocation::OutputPortBit {
                output_idx,
                bit_idx,
            } => do_toggle_output_bit(g, *output_idx, *bit_idx),
            _ => Err(anyhow!(
                "Invalid location type for ToggleOutputBitTransform: {:?}",
                candidate_location
            )),
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gate::AigRef,
        gate_builder::{GateBuilder, GateBuilderOptions},
    };

    #[test]
    fn test_toggle_output_bit_transform_applies_and_reverses() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o0".to_string(), i0.into());
        let g_original = gb.build();
        let mut g_transformed = g_original.clone();

        let mut transform = ToggleOutputBitTransform::new();
        let candidates = transform.find_candidates(&g_transformed, TransformDirection::Forward);
        assert!(!candidates.is_empty(), "Should find candidates");

        let candidate_loc = &candidates[0];

        let original_output_negation = g_transformed.outputs[0].bit_vector.get_lsb(0).negated;
        transform
            .apply(
                &mut g_transformed,
                candidate_loc,
                TransformDirection::Forward,
            )
            .unwrap();
        let transformed_output_negation = g_transformed.outputs[0].bit_vector.get_lsb(0).negated;
        assert_ne!(
            original_output_negation, transformed_output_negation,
            "Forward transform should change negation"
        );

        transform
            .apply(
                &mut g_transformed,
                candidate_loc,
                TransformDirection::Backward,
            )
            .unwrap();
        let reverted_output_negation = g_transformed.outputs[0].bit_vector.get_lsb(0).negated;
        assert_eq!(
            original_output_negation, reverted_output_negation,
            "Backward transform should revert negation"
        );
        assert_eq!(
            g_original.to_string(),
            g_transformed.to_string(),
            "Graph should be identical after forward and backward transform"
        );
    }

    #[test]
    fn test_toggle_output_bit_transform_invalid_location() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o0".to_string(), i0.into());
        let mut g = gb.build();
        let transform = ToggleOutputBitTransform::new();
        let invalid_loc = TransformLocation::Node(AigRef { id: 0 });
        assert!(transform
            .apply(&mut g, &invalid_loc, TransformDirection::Forward)
            .is_err());
    }

    #[test]
    fn test_do_toggle_output_bit_out_of_bounds() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o0".to_string(), i0.into());
        let mut g = gb.build();

        assert!(do_toggle_output_bit(&mut g, 1, 0).is_err());
        assert!(do_toggle_output_bit(&mut g, 0, 1).is_err());
    }
}
