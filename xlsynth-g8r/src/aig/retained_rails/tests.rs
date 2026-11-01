// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

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
    let (unrelated, _, _) = grouped(true, true, 7, true);
    assert!(
        RailResolver::new(&unrelated)
            .find(&cone, &mut budget)
            .is_none()
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
