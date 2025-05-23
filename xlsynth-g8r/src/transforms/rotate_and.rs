// SPDX-License-Identifier: Apache-2.0

use rand::seq::SliceRandom;
use rand::Rng;

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::use_count::get_id_to_use_count;

/// Rotates an AND tree to the right: `((a & b) & c) -> (a & (b & c))`.
///
/// Requires that the left operand of `outer` is a non-negated `And2` node
/// used only by `outer` itself. Returns `Ok(())` on success.
pub fn rotate_and_right(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left, right) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_right: outer is not And2"),
    };

    if left.negated {
        return Err("rotate_and_right: left operand negated");
    }

    let inner_ref = left.node;
    let (a, b) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_right: inner is not And2"),
    };

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("rotate_and_right: inner fanout > 1");
    }

    if let AigNode::And2 { a: ia, b: ib, .. } = &mut g.gates[inner_ref.id] {
        *ia = b;
        *ib = right;
    }
    if let AigNode::And2 { a: oa, b: ob, .. } = &mut g.gates[outer.id] {
        *oa = a;
        *ob = AigOperand {
            node: inner_ref,
            negated: false,
        };
    }
    Ok(())
}

/// Inverse of [`rotate_and_right`]: transforms `(a & (b & c)) -> ((a & b) &
/// c)`. The right operand of `outer` must be a non-negated `And2` node with
/// fanout 1.
pub fn rotate_and_left(g: &mut GateFn, outer: AigRef) -> Result<(), &'static str> {
    let (left, right) = match g.gates[outer.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_left: outer is not And2"),
    };

    if right.negated {
        return Err("rotate_and_left: right operand negated");
    }

    let inner_ref = right.node;
    let (b, c) = match g.gates[inner_ref.id] {
        AigNode::And2 { a, b, .. } => (a, b),
        _ => return Err("rotate_and_left: inner is not And2"),
    };

    let use_counts = get_id_to_use_count(g);
    if *use_counts.get(&inner_ref).unwrap_or(&0) != 1 {
        return Err("rotate_and_left: inner fanout > 1");
    }

    if let AigNode::And2 { a: ia, b: ib, .. } = &mut g.gates[inner_ref.id] {
        *ia = left;
        *ib = b;
    }
    if let AigNode::And2 { a: oa, b: ob, .. } = &mut g.gates[outer.id] {
        *oa = AigOperand {
            node: inner_ref,
            negated: false,
        };
        *ob = c;
    }
    Ok(())
}

/// Picks a random location eligible for [`rotate_and_right`] and applies it.
/// Returns the `AigRef` of the rotated node on success.
pub fn rotate_and_right_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<AigRef, &'static str> {
    let use_counts = get_id_to_use_count(g);
    let mut candidates = Vec::new();
    for (idx, node) in g.gates.iter().enumerate() {
        if let AigNode::And2 { a, b: _, .. } = node {
            if a.negated {
                continue;
            }
            if let AigNode::And2 { .. } = g.gates[a.node.id] {
                if *use_counts.get(&a.node).unwrap_or(&0) == 1 {
                    candidates.push(AigRef { id: idx });
                }
            }
        }
    }
    if candidates.is_empty() {
        return Err("rotate_and_right_rand: no candidates");
    }
    let chosen = *candidates.choose(rng).unwrap();
    rotate_and_right(g, chosen)?;
    Ok(chosen)
}

/// Picks a random location eligible for [`rotate_and_left`] and applies it.
/// Returns the `AigRef` of the rotated node on success.
pub fn rotate_and_left_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<AigRef, &'static str> {
    let use_counts = get_id_to_use_count(g);
    let mut candidates = Vec::new();
    for (idx, node) in g.gates.iter().enumerate() {
        if let AigNode::And2 { a: _, b, .. } = node {
            if b.negated {
                continue;
            }
            if let AigNode::And2 { .. } = g.gates[b.node.id] {
                if *use_counts.get(&b.node).unwrap_or(&0) == 1 {
                    candidates.push(AigRef { id: idx });
                }
            }
        }
    }
    if candidates.is_empty() {
        return Err("rotate_and_left_rand: no candidates");
    }
    let chosen = *candidates.choose(rng).unwrap();
    rotate_and_left(g, chosen)?;
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_rotate_and_round_trip() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let o = gb.add_and_binary(a, i2);
        gb.add_output("o".to_string(), o.into());
        let g1 = gb.build();

        let mut g2 = g1.clone();
        rotate_and_right(&mut g2, o.node).unwrap();
        rotate_and_left(&mut g2, o.node).unwrap();
        assert_eq!(g1.to_string(), g2.to_string());
    }

    #[test]
    fn test_rotate_and_right_rand_round_trip() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let o = gb.add_and_binary(a, i2);
        gb.add_output("o".to_string(), o.into());
        let mut g = gb.build();
        let pre = g.to_string();
        let mut rng = StdRng::seed_from_u64(123);
        let loc = rotate_and_right_rand(&mut g, &mut rng).unwrap();
        rotate_and_left(&mut g, loc).unwrap();
        let post = g.to_string();
        assert_eq!(pre, post);
    }

    #[test]
    fn test_rotate_and_right_fanout_fail() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let i2 = gb.add_input("i2".to_string(), 1).get_lsb(0).clone();
        let i3 = gb.add_input("i3".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        let b = gb.add_and_binary(a, i2);
        let c = gb.add_and_binary(a, i3);
        gb.add_output("o".to_string(), b.into());
        gb.add_output("c".to_string(), c.into());
        let mut g = gb.build();
        let res = rotate_and_right(&mut g, b.node);
        assert!(res.is_err());
    }

    #[test]
    fn test_rotate_and_right_rand_none() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        let mut rng = StdRng::seed_from_u64(42);
        let res = rotate_and_right_rand(&mut g, &mut rng);
        assert!(res.is_err());
    }
}
