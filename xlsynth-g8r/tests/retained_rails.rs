// SPDX-License-Identifier: Apache-2.0

use test_case::test_matrix;
use xlsynth_g8r::aig::get_summary_stats::get_aig_cone_stats;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, PreparedGateSim};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

const OR_MASK: &str = "mask: bits[1] = or(g,h,i, id=17)";

/// Builds an affine priority bit gated by a computed mask.
fn masked_priority_ir(mask: &str, op: &str, result_type: &str, result: &str) -> String {
    format!(
        r#"package p
top fn main(x: bits[8] id=1, g: bits[1] id=2, h: bits[1] id=3, i: bits[1] id=4) -> {result_type} {{
  rev: bits[8] = reverse(x, id=10)
  hot: bits[9] = one_hot(rev, lsb_prio=true, id=11)
  ix: bits[4] = encode(hot, id=12)
  wide: bits[5] = zero_ext(ix, new_bit_count=5, id=13)
  c: bits[5] = literal(value=8, id=15)
  diff: bits[5] = sub(c,wide, id=16)
  {mask}
  full: bits[5] = sign_ext(mask, new_bit_count=5, id=18)
  word: bits[5] = {op}(diff,full, id=19)
  bit: bits[1] = bit_slice(word, start=3, width=1, id=20)
  inv: bits[1] = not(bit, id=21)
  ret out: {result_type} = {result}
}}"#
    )
}

fn lowering_options(rewrite: bool) -> CanonicalG8rOptions {
    CanonicalG8rOptions {
        enable_formal_array_alias_analysis: false,
        fraig: false,
        enable_rewrite_prio_encode: rewrite,
        enable_rewrite_nary_add: rewrite,
        reassociation: rewrite,
        cut_db_rewrite: rewrite,
        ..Default::default()
    }
}

#[test_matrix([false, true])]
fn mask_and_complement_share_the_typed_returned_bit(rewrite: bool) {
    let ir = masked_priority_ir(
        OR_MASK,
        "and",
        "(bits[1], bits[1])",
        "tuple(bit, inv, id=22)",
    );
    let gates =
        canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &lowering_options(rewrite))
            .unwrap()
            .gate_fn;
    let stats = get_aig_cone_stats(
        &gates.gates,
        &gates.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .copied()
            .collect::<Vec<_>>(),
    );
    assert_eq!(stats.and_nodes, 3);
    assert_eq!(
        gates.outputs[0].bit_vector.get_lsb(0).negate(),
        *gates.outputs[0].bit_vector.get_lsb(1)
    );
    let mut sim = PreparedGateSim::new(&gates);
    for x in 0..256 {
        for mask in 0..8 {
            let expected = if x >= 128 && mask != 0 { 2 } else { 1 };
            let inputs = [
                IrBits::make_ubits(8, x).unwrap(),
                IrBits::bool(mask & 1 != 0),
                IrBits::bool(mask & 2 != 0),
                IrBits::bool(mask & 4 != 0),
            ];
            assert_eq!(
                sim.eval_outputs(&inputs),
                vec![IrBits::make_ubits(2, expected).unwrap()]
            );
        }
    }
}

#[test]
fn reconstructed_ordinal_preserves_concat_output_tag() {
    let ir = masked_priority_ir(OR_MASK, "and", "bits[2]", "concat(bit, g, id=22)");
    let gates =
        canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &lowering_options(false))
            .unwrap()
            .gate_fn;
    let root = *gates.outputs[0].bit_vector.get_lsb(1);
    assert_eq!(get_aig_cone_stats(&gates.gates, &[root]).and_nodes, 3);
    let mut sim = PreparedGateSim::new(&gates);
    for x in [0, 127, 128, 255] {
        for mask in 0..8 {
            let inputs = [
                IrBits::make_ubits(8, x).unwrap(),
                IrBits::bool(mask & 1 != 0),
                IrBits::bool(mask & 2 != 0),
                IrBits::bool(mask & 4 != 0),
            ];
            let result = sim.eval(&inputs, Collect::Tagged);
            let bit = x >= 128 && mask != 0;
            assert_eq!(
                result.tagged_values.unwrap().get("concat_22_output_bit_1"),
                Some(&bit)
            );
            assert_eq!(
                result.outputs,
                [IrBits::make_ubits(2, u64::from(bit) * 2 + (mask & 1)).unwrap()]
            );
        }
    }
}

#[test]
fn equivalent_mask_preserves_the_sign_extension_tag() {
    let masks = r#"j: bits[1] = bit_slice(x, start=0, width=1, id=5)
  gh: bits[1] = or(g,h, id=6)
  ghi: bits[1] = or(gh,i, id=7)
  ij: bits[1] = or(i,j, id=8)
  balanced: bits[1] = or(gh,ij, id=9)
  mask: bits[1] = or(ghi,j, id=17)"#;
    let ir = masked_priority_ir(
        masks,
        "and",
        "(bits[1], bits[1])",
        "tuple(bit, balanced, id=22)",
    );
    let gates =
        canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &lowering_options(false))
            .unwrap()
            .gate_fn;
    let mut sim = PreparedGateSim::new(&gates);
    for mask in 0..16 {
        for winner in [false, true] {
            let x = (u64::from(winner) << 7) | (mask >> 3);
            let result = sim.eval(
                &[
                    IrBits::make_ubits(8, x).unwrap(),
                    IrBits::bool(mask & 1 != 0),
                    IrBits::bool(mask & 2 != 0),
                    IrBits::bool(mask & 4 != 0),
                ],
                Collect::Tagged,
            );
            // OR is a complemented AND. Tags observe its positive node.
            assert_eq!(
                result.tagged_values.unwrap().get("sign_ext_18_msb"),
                Some(&(mask == 0))
            );
            assert_eq!(
                result.outputs,
                [IrBits::from_lsb_is_0(&[mask != 0, winner && mask != 0])]
            );
        }
    }
    let root = *gates.outputs[0].bit_vector.get_lsb(1);
    assert_eq!(get_aig_cone_stats(&gates.gates, &[root]).root_depths, [3]);
}

#[test]
fn reconstructed_xor_attributes_all_three_ands() {
    let ir = masked_priority_ir(
        OR_MASK,
        "xor",
        "(bits[1], bits[1])",
        "tuple(bit, inv, id=22)",
    );
    let options = CanonicalG8rOptions {
        track_pir_node_ids: true,
        ..lowering_options(false)
    };
    let gates = canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &options)
        .unwrap()
        .gate_fn;
    let root = gates.outputs[0].bit_vector.get_lsb(1).node;
    let outer = &gates.gates[root.id];
    assert!(outer.get_pir_node_ids().contains(&19));
    for inner in outer.args() {
        assert_eq!(gates.gates[inner.id].get_pir_node_ids(), &[19]);
    }
    assert_eq!(
        get_aig_cone_stats(&gates.gates, &[root.into()]).and_nodes,
        5
    );
}

#[test]
fn computed_mask_is_shared_across_many_return_positions() {
    // Either recapturing or re-resolving the mask per position would
    // exhaust the shared rail budget before all aliases are replaced.
    let mask = r#"j: bits[1] = bit_slice(x, start=0, width=1, id=5)
  mask: bits[1] = or(g,h,i,j, id=17)"#;
    let args = ["bit", "inv"].repeat(2048).join(", ");
    let ir = masked_priority_ir(mask, "and", "bits[4096]", &format!("concat({args}, id=22)"));
    let gates =
        canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &lowering_options(false))
            .unwrap()
            .gate_fn;
    let bits = &gates.outputs[0].bit_vector;
    let bit = *bits.get_lsb(1);
    assert_eq!(get_aig_cone_stats(&gates.gates, &[bit]).and_nodes, 4);
    for (i, &operand) in bits.iter_lsb_to_msb().enumerate() {
        assert_eq!(operand, if i % 2 == 0 { bit.negate() } else { bit });
    }
}
