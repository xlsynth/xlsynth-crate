// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};

#[test]
fn aliased_complements_recover_across_renumbering_and_reassociation() {
    let (old, wire, _) = grouped(false, false, 3, false);
    let (new, rewired, outside) = grouped(true, true, 3, false);
    let mut budget = RailBudget::default();
    let witness = SignedCone::capture(&old, wire.negate(), &mut budget).unwrap();
    let mut resolver = RailResolver::new(&new);
    assert_eq!(resolver.find(&witness, &mut budget), Some(rewired.negate()));
    let bits: Vec<_> = (0..3)
        .map(|bit| CutBit {
            port: "data".into(),
            width: 3,
            bit,
        })
        .collect();
    let src = CutIndex::build(&old, &bits, &mut budget).unwrap();
    let dst = CutIndex::build(&new, &bits, &mut budget).unwrap();
    let signed = src.signature(wire.negate()).unwrap();
    assert_eq!(dst.resolve(&signed), Some(rewired.negate()));
    assert_eq!(
        dst.best_equivalent(rewired.negate()),
        Some(rewired.negate())
    );
    assert_eq!(signed.rows(), 0xbf);
    assert_eq!(
        dst.resolve(&CutSignature::from_rows("test", &bits, signed.rows()).unwrap()),
        Some(rewired.negate())
    );
    assert!(dst.signature(outside).is_none());
    let mut wrong_order = bits.clone();
    wrong_order.reverse();
    assert!(
        CutIndex::build(&new, &wrong_order, &mut budget)
            .unwrap()
            .resolve(&signed)
            .is_none()
    );
    let mut repeated = bits.clone();
    repeated.push(bits[0].clone());
    assert!(CutIndex::build(&old, &repeated, &mut budget).is_none());
    let mut changed = bits.clone();
    changed[1].width = 4;
    assert!(CutIndex::build(&old, &changed, &mut budget).is_none());
    let mut wrong = new.clone();
    wrong.name = "different".into();
    assert!(
        RailResolver::new(&wrong)
            .find(&witness, &mut budget)
            .is_none()
    );
}

#[test]
fn constant_and_high_support_structural_fallback() {
    let mut budget = RailBudget::default();
    let (old, wire, constant) = grouped(false, false, 7, true);
    let (new, rewired, _) = grouped(false, true, 7, true);
    let mut resolver = RailResolver::new(&new);
    let cone = SignedCone::capture(&old, wire, &mut budget).unwrap();
    assert!(cone.cut_rows().is_none());
    assert_eq!(resolver.find(&cone, &mut budget), Some(rewired));
    let true_cone = SignedCone::capture(&old, constant.negate(), &mut budget).unwrap();
    assert!(true_cone.cut_rows().unwrap().1 == 1);
    assert_eq!(
        resolver.find(&true_cone, &mut budget),
        Some(constant.negate())
    );
    let (regrouped, regrouped_wire, _) = grouped(true, true, 7, true);
    assert_eq!(
        RailResolver::new(&regrouped).find(&cone, &mut budget),
        Some(regrouped_wire)
    );
    assert!(resolver.find(&cone, &mut RailBudget(0)).is_none());
    assert!(SignedCone::capture(&old, wire, &mut RailBudget(2)).is_none());
    let bits: Vec<_> = (0..7)
        .map(|bit| CutBit {
            port: "data".into(),
            width: 7,
            bit,
        })
        .collect();
    assert!(CutIndex::build(&new, &bits, &mut budget).is_none());
}

#[test]
fn wide_signed_conjunction_keeps_coordinates_and_reconvergent_uses() {
    let width = 70;
    let (old, wire, _) = grouped(false, true, width, false);
    let mut budget = RailBudget::default();
    let witness = SignedCone::capture(&old, wire.negate(), &mut budget).unwrap();
    let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    let data = b.add_input("data".into(), width);
    let other = *b.add_input("other".into(), 1).get_lsb(0);
    let bits: Vec<_> = data.iter_lsb_to_msb().copied().collect();
    let mut partial = bits[1];
    for bit in &bits[2..] {
        partial = b.add_and_binary(partial, *bit);
    }
    let mut repeated = partial;
    for _ in 0..16 {
        repeated = b.add_and_binary(repeated, repeated);
    }
    let wrong_port = b.add_and_binary(repeated, other.negate());
    let wrong_sign = b.add_and_binary(repeated, bits[0]);
    let wrong_index = b.add_and_binary(repeated, bits[1].negate());
    let wrong_phase = b.add_and_binary(repeated.negate(), bits[0].negate());
    for (name, value) in [
        ("port", wrong_port),
        ("sign", wrong_sign),
        ("index", wrong_index),
        ("phase", wrong_phase),
    ] {
        b.add_output(name.into(), value.into());
    }
    let negatives = b.clone().build();
    assert!(
        RailResolver::new(&negatives)
            .find(&witness, &mut budget)
            .is_none()
    );
    let wire = b.add_and_binary(repeated, bits[0].negate());
    b.add_output("wire".into(), wire.into());
    let new = b.build();
    let mut resolver = RailResolver::new(&new);
    assert_eq!(resolver.find(&witness, &mut budget), Some(wire.negate()));
    // Each computed output was already live; resolving a witness is read-only.
    assert_eq!(new.outputs.len(), 5);
    let mut cut_off = RailResolver::new(&new);
    cut_off.conjunction_budget = RailBudget(0);
    assert!(cut_off.find(&witness, &mut RailBudget::default()).is_none());
}

/// Capture a complemented AND chain grouped from the most significant bit.
fn reverse_chain_witness(width: usize, budget: &mut RailBudget) -> Option<SignedCone> {
    let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    let data = b.add_input("data".into(), width);
    let reversed: Vec<_> = data.iter_msb_to_lsb().copied().collect();
    let root = b.add_and_nary(&reversed, ReductionKind::Linear).negate();
    b.add_output("wire".into(), root.into());
    SignedCone::capture(&b.build(), root, budget)
}

#[test]
fn regrouped_conjunction_prefers_depth_then_node_id() {
    let width = 16;
    let mut budget = RailBudget::default();
    let witness = reverse_chain_witness(width, &mut budget).unwrap();
    let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    let data = b.add_input("data".into(), width);
    let chain = b.add_and_reduce(&data, ReductionKind::Linear);
    let tree = b.add_and_reduce(&data, ReductionKind::Tree);
    let duplicate_tree = b.add_and_reduce(&data, ReductionKind::Tree);
    // The lowest-ID root is 15 gates deep. Both trees are only 4 gates deep.
    // Arrange the outputs so the lower-ID tree is visited last in postorder.
    b.add_output("tree".into(), tree.into());
    b.add_output("duplicate".into(), duplicate_tree.into());
    b.add_output("chain".into(), chain.into());
    let graph = b.build();
    assert_eq!(
        RailResolver::new(&graph).find(&witness, &mut budget),
        Some(tree.negate())
    );
}

#[test]
fn regrouped_conjunction_keeps_proven_match_at_budget_limit() {
    let width = 8;
    let mut budget = RailBudget::default();
    let witness = reverse_chain_witness(width, &mut budget).unwrap();
    let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    let data = b.add_input("data".into(), width);
    let tree = b.add_and_reduce(&data, ReductionKind::Tree);
    let repeated = b.add_and_binary(tree, *data.get_lsb(0));
    b.add_output("wire".into(), repeated.into());
    let graph = b.build();

    // Each merge costs one word. The last two merges prove the tree and
    // then the repeated root. Stop immediately before and after the proof.
    let mut resolver = RailResolver::new(&graph);
    let initial = resolver.conjunction_budget.0;
    assert_eq!(resolver.find(&witness, &mut budget), Some(tree.negate()));
    let full_scan = initial - resolver.conjunction_budget.0;
    for (work, expected) in [(full_scan - 2, None), (full_scan - 1, Some(tree.negate()))] {
        resolver.conjunction_budget = RailBudget(work);
        assert_eq!(resolver.find(&witness, &mut budget), expected);
    }
}

fn grouped(
    regroup: bool,
    shuffle: bool,
    width: usize,
    with_constant: bool,
) -> (GateFn, AigOperand, AigOperand) {
    let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    if shuffle {
        b.add_input("extra".into(), 1);
    }
    let data = b.add_input("data".into(), width);
    let extra = if shuffle {
        *b.inputs[0].bit_vector.get_lsb(0)
    } else {
        *b.add_input("extra".into(), 1).get_lsb(0)
    };
    let mut nodes: Vec<_> = data.iter_lsb_to_msb().copied().collect();
    nodes[0] = nodes[0].negate();
    if regroup {
        nodes.rotate_left(1);
    }
    let mut wire = nodes[0];
    for n in &nodes[1..] {
        wire = b.add_and_binary(wire, *n);
    }
    let mut alias = nodes[0];
    for n in &nodes[1..] {
        alias = b.add_and_binary(*n, alias);
    }
    let mixed = b.add_or_binary(wire, extra);
    b.add_output("mixed".into(), mixed.into());
    b.add_output("alias".into(), alias.negate().into());
    let c = b.get_false();
    if with_constant {
        b.add_output("constant".into(), c.into());
    }
    if with_constant {
        (b.build(), wire, c)
    } else {
        (b.build(), wire, mixed)
    }
}
