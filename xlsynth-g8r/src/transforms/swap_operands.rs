// SPDX-License-Identifier: Apache-2.0

use rand::seq::SliceRandom;
use rand::Rng;

use crate::gate::{AigNode, AigRef, GateFn};

/// Swaps the left and right operands of an `And2` gate.
///
/// Returns `Ok(())` if the given node is an `And2` gate, otherwise
/// returns an error.
pub fn swap_operands(g: &mut GateFn, node: AigRef) -> Result<(), &'static str> {
    match &mut g.gates[node.id] {
        AigNode::And2 { a, b, .. } => {
            core::mem::swap(a, b);
            Ok(())
        }
        _ => Err("swap_operands: node is not And2"),
    }
}

/// Picks a random `And2` node in the graph and swaps its operands.
///
/// Returns the `AigRef` of the swapped node on success.
pub fn swap_operands_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<AigRef, &'static str> {
    let candidates: Vec<AigRef> = g
        .gates
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| match node {
            AigNode::And2 { .. } => Some(AigRef { id: idx }),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return Err("swap_operands_rand: no And2 nodes");
    }
    let chosen = *candidates.choose(rng).unwrap();
    swap_operands(g, chosen)?;
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_swap_operands_self_inverse() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let g1 = gb.build();
        let mut g2 = g1.clone();
        swap_operands(&mut g2, a.node).unwrap();
        swap_operands(&mut g2, a.node).unwrap();
        assert_eq!(g1.to_string(), g2.to_string());
    }

    #[test]
    fn test_swap_operands_rand_round_trip() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let b = gb.add_and_binary(i1, i2);
        let o = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), o.into());
        let mut g = gb.build();
        let pre = g.to_string();
        let mut rng = StdRng::seed_from_u64(123);
        let loc = swap_operands_rand(&mut g, &mut rng).unwrap();
        swap_operands(&mut g, loc).unwrap();
        let post = g.to_string();
        assert_eq!(pre, post);
    }

    #[test]
    fn test_swap_operands_invalid_node() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let res = swap_operands(&mut g, i0.node);
        assert!(res.is_err());
    }

    #[test]
    fn test_swap_operands_rand_no_and() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let mut rng = StdRng::seed_from_u64(42);
        let res = swap_operands_rand(&mut g, &mut rng);
        assert!(res.is_err());
    }
}
