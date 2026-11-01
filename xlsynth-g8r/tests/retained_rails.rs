// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_cone_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

#[test]
fn reassociated_mask_and_complement_share_the_typed_returned_bit() {
    let ir = r#"package p
top fn main(x: bits[8] id=1, g: bits[1] id=2, h: bits[1] id=3, i: bits[1] id=4) -> (bits[1],bits[1]) {
  rev: bits[8] = reverse(x, id=10)
  hot: bits[9] = one_hot(rev, lsb_prio=true, id=11)
  ix: bits[4] = encode(hot, id=12)
  wide: bits[5] = zero_ext(ix, new_bit_count=5, id=13)
  c: bits[5] = literal(value=8, id=15)
  diff: bits[5] = sub(c,wide, id=16)
  mask: bits[1] = or(g,h,i, id=17)
  full: bits[5] = sign_ext(mask, new_bit_count=5, id=18)
  word: bits[5] = and(diff,full, id=19)
  bit: bits[1] = bit_slice(word, start=3, width=1, id=20)
  inv: bits[1] = not(bit, id=21)
  ret out: (bits[1],bits[1]) = tuple(bit, inv, id=22)
}"#;
    let gates = canonical_ir_text_to_g8r_lowering_artifacts(
        ir,
        Some("main"),
        &CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            fold: true,
            hash: true,
            fraig: false,
            ..Default::default()
        },
    )
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
