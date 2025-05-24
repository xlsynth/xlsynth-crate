// SPDX-License-Identifier: Apache-2.0

use crate::gate::GateFn;
use crate::transforms::transform_trait::{
    Transform, TransformDirection, TransformKind, TransformLocation,
};
use anyhow::{anyhow, Result};

/// Primitive: swaps two output bits in a `GateFn`.
pub fn swap_output_bits_primitive(
    g: &mut GateFn,
    out_a_idx: usize,
    bit_a_idx: usize,
    out_b_idx: usize,
    bit_b_idx: usize,
) -> Result<(), &'static str> {
    if out_a_idx >= g.outputs.len() || out_b_idx >= g.outputs.len() {
        return Err("swap_output_bits_primitive: output index out of bounds");
    }
    if bit_a_idx >= g.outputs[out_a_idx].bit_vector.get_bit_count()
        || bit_b_idx >= g.outputs[out_b_idx].bit_vector.get_bit_count()
    {
        return Err("swap_output_bits_primitive: bit index out of bounds");
    }

    let op_a = *g.outputs[out_a_idx].bit_vector.get_lsb(bit_a_idx);
    let op_b = *g.outputs[out_b_idx].bit_vector.get_lsb(bit_b_idx);
    g.outputs[out_a_idx].bit_vector.set_lsb(bit_a_idx, op_b);
    g.outputs[out_b_idx].bit_vector.set_lsb(bit_b_idx, op_a);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SwapOutputBitsLocation {
    pub out_a_idx: usize,
    pub bit_a_idx: usize,
    pub out_b_idx: usize,
    pub bit_b_idx: usize,
}

#[derive(Debug)]
pub struct SwapOutputBitsTransform;

impl SwapOutputBitsTransform {
    pub fn new() -> Self {
        SwapOutputBitsTransform
    }
}

impl Transform for SwapOutputBitsTransform {
    fn kind(&self) -> TransformKind {
        TransformKind::SwapOutputBits
    }

    fn find_candidates(
        &mut self,
        g: &GateFn,
        _direction: TransformDirection,
    ) -> Vec<TransformLocation> {
        let mut candidates = Vec::new();
        for out_a in 0..g.outputs.len() {
            for bit_a in 0..g.outputs[out_a].bit_vector.get_bit_count() {
                for out_b in out_a..g.outputs.len() {
                    let start_b = if out_b == out_a { bit_a + 1 } else { 0 };
                    for bit_b in start_b..g.outputs[out_b].bit_vector.get_bit_count() {
                        candidates.push(TransformLocation::Custom(Box::new(
                            SwapOutputBitsLocation {
                                out_a_idx: out_a,
                                bit_a_idx: bit_a,
                                out_b_idx: out_b,
                                bit_b_idx: bit_b,
                            },
                        )));
                    }
                }
            }
        }
        candidates
    }

    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        _direction: TransformDirection,
    ) -> Result<()> {
        match candidate_location {
            TransformLocation::Custom(b) => {
                let loc = b
                    .downcast_ref::<SwapOutputBitsLocation>()
                    .ok_or_else(|| anyhow!("Invalid location type for SwapOutputBitsTransform"))?;
                swap_output_bits_primitive(
                    g,
                    loc.out_a_idx,
                    loc.bit_a_idx,
                    loc.out_b_idx,
                    loc.bit_b_idx,
                )
                .map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "Invalid location type for SwapOutputBitsTransform: {:?}",
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
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::setup_simple_graph;

    #[test]
    fn test_swap_output_bits_primitive_round_trip() {
        let test = setup_simple_graph();
        let mut g = test.g.clone();
        swap_output_bits_primitive(&mut g, 0, 0, 1, 0).unwrap();
        assert_eq!(g.outputs[0].name, "o");
        assert_eq!(g.outputs[1].name, "c");
        assert_eq!(
            g.outputs[0].bit_vector.get_lsb(0),
            test.g.outputs[1].bit_vector.get_lsb(0)
        );
        swap_output_bits_primitive(&mut g, 0, 0, 1, 0).unwrap();
        assert_eq!(g.to_string(), test.g.to_string());
    }

    #[test]
    fn test_swap_output_bits_transform_apply() {
        let test = setup_simple_graph();
        let mut g = test.g.clone();
        let mut t = SwapOutputBitsTransform::new();
        let cands = t.find_candidates(&g, TransformDirection::Forward);
        assert!(!cands.is_empty());
        let first = &cands[0];
        t.apply(&mut g, first, TransformDirection::Forward).unwrap();
    }

    #[test]
    fn test_swap_output_bits_primitive_oob() {
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o0".to_string(), i0.into());
        let mut g = gb.build();
        assert!(swap_output_bits_primitive(&mut g, 0, 0, 0, 1).is_err());
        assert!(swap_output_bits_primitive(&mut g, 0, 0, 1, 0).is_err());
    }
}
