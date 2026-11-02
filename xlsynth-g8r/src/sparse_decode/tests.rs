// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig_sim::gate_sim::{Collect, eval};
use crate::process_ir_path::{CanonicalG8rOptions, canonical_ir_text_to_g8r_artifacts};
use xlsynth_pir::IrBits;

#[test]
fn codebook_with_mode_covers_unassigned_keys_and_preserves_tracked_topology() {
    let text = r#"package mode_lookup

top fn main(mode: bits[1] id=2, key: bits[5] id=1) -> bits[3] {
  v3: bits[5] = literal(value=24, id=3)
  v4: bits[1] = eq(v3, key, id=4)
  v5: bits[5] = literal(value=5, id=5)
  v6: bits[1] = eq(v5, key, id=6)
  v7: bits[5] = literal(value=14, id=7)
  v8: bits[1] = eq(v7, key, id=8)
  v9: bits[5] = literal(value=7, id=9)
  v10: bits[1] = eq(v9, key, id=10)
  v11: bits[5] = literal(value=11, id=11)
  v12: bits[1] = eq(v11, key, id=12)
  v13: bits[5] = literal(value=15, id=13)
  v14: bits[1] = eq(v13, key, id=14)
  v15: bits[5] = literal(value=31, id=15)
  v16: bits[1] = eq(v15, key, id=16)
  v17: bits[5] = literal(value=26, id=17)
  v18: bits[1] = eq(v17, key, id=18)
  v19: bits[5] = literal(value=23, id=19)
  v20: bits[1] = eq(v19, key, id=20)
  v21: bits[1] = or(v12, v10, v6, v8, id=21)
  v22: bits[1] = sel(mode, cases=[v21, v8], id=22)
  v23: bits[1] = or(v4, v16, id=23)
  v24: bits[1] = or(v20, v6, v18, v14, v10, v4, v8, id=24)
  v25: bits[1] = sel(mode, cases=[v23, v24], id=25)
  v26: bits[1] = or(v6, v4, v8, v18, v10, v20, v12, v14, id=26)
  v27: bits[1] = or(v6, v12, v18, v10, v16, v14, v4, v20, id=27)
  v28: bits[1] = sel(mode, cases=[v26, v27], id=28)
  v29: bits[1] = not(v28, id=29)
  ret out: bits[3] = concat(v22, v25, v29, id=30)
}
"#;
    let opts = CanonicalG8rOptions {
        fold: true,
        hash: true,
        fraig: false,
        enable_formal_array_alias_analysis: false,
        ..Default::default()
    };
    let ordinary = canonical_ir_text_to_g8r_artifacts(text, Some("main"), &opts).unwrap();
    let tracked = canonical_ir_text_to_g8r_artifacts(
        text,
        Some("main"),
        &CanonicalG8rOptions {
            track_pir_node_ids: true,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(ordinary.binary_aiger, tracked.binary_aiger);
    let graph = &tracked.lowering.gate_fn;
    let stats = get_output_cone_stats(graph);
    assert!(stats.and_nodes <= 44 && stats.root_depths.iter().all(|&d| d <= 6));
    for port in &graph.outputs {
        for bit in port.bit_vector.iter_lsb_to_msb() {
            assert!(!graph.gates[bit.node.id].get_pir_node_ids().is_empty());
        }
    }
    for mode in 0..=1 {
        for key in 0..32 {
            let code = if mode == 0 {
                [
                    matches!(key, 11 | 7 | 5 | 14),
                    matches!(key, 24 | 31),
                    !matches!(key, 5 | 24 | 14 | 26 | 7 | 23 | 11 | 15),
                ]
            } else {
                [
                    key == 14,
                    matches!(key, 23 | 5 | 26 | 15 | 7 | 24 | 14),
                    !matches!(key, 5 | 11 | 26 | 7 | 31 | 15 | 24 | 23),
                ]
            };
            let expected = code.into_iter().fold(0u64, |n, b| (n << 1) | u64::from(b));
            let inputs = [
                IrBits::make_ubits(1, mode).unwrap(),
                IrBits::make_ubits(5, key).unwrap(),
            ];
            assert_eq!(
                eval(graph, &inputs, Collect::None).outputs,
                vec![IrBits::make_ubits(3, expected).unwrap()]
            );
        }
    }
}

#[test]
fn local_covers_are_total_on_all_three_input_functions() {
    for n in 0..=3 {
        let rows = 1 << n;
        for table in 0..1u16 << rows {
            let terms = minimum_cube_cover(n, table as u8);
            for key in 0..rows {
                let actual = terms.iter().any(|&(mask, pat)| key & mask == pat);
                assert_eq!(actual, (table & (1 << key)) != 0);
            }
        }
    }
}

#[test]
fn independent_decode_keys_do_not_authorize_a_partition() {
    let text = r#"package quart

top fn main(key0: bits[3] id=1, key1: bits[3] id=2) -> bits[4] {
  c3: bits[3] = literal(value=0, id=3)
  e4: bits[1] = eq(key0, c3, id=4)
  c5: bits[3] = literal(value=3, id=5)
  e6: bits[1] = eq(key0, c5, id=6)
  c7: bits[3] = literal(value=2, id=7)
  e8: bits[1] = eq(key1, c7, id=8)
  c9: bits[3] = literal(value=6, id=9)
  e10: bits[1] = eq(key1, c9, id=10)
  ret out: bits[4] = concat(e4, e6, e8, e10, id=11)
}
"#;
    let pkg = xlsynth_pir::ir_parser::Parser::new(text)
        .parse_package()
        .unwrap();
    let f = pkg.get_fn("main").unwrap();
    let lowered = canonical_ir_text_to_g8r_artifacts(
        text,
        Some("main"),
        &CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(decoded_key(f, &lowered.lowering.gate_fn), None);
}
