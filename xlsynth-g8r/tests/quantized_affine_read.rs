// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{
    ArrayIndexLoweringStrategy, GatifyOptions, gatify, gatify_node_as_fn, gatify_prepared_fn,
};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, canonical_ir_text_to_prepared_gatify_ir};
use xlsynth_pir::{
    IrBits, IrValue,
    ir::NodePayload,
    ir_eval::{FnEvalResult, eval_fn},
    ir_parser::Parser,
};

#[test]
fn fractional_tables_preserve_the_enclosing_function_and_integral_reads() {
    for (w, k, scale, direction, phase) in [(6, 2, 0, 1, 3), (70, 2, 3, -1, 2), (3, 0, 0, 1, 0)] {
        let mut words: Vec<_> = (0..32i128)
            .map(|x| {
                let bias = ((1i128 << (w - scale - 1)) << k) + phase;
                ((((bias + direction * x) >> k) << scale) | ((1 << scale) - 1)) as u128
                    & ((1 << w) - 1)
            })
            .collect();
        words[0] = 7;
        words[31] = 7;
        let source = table_ir(w, 5, &words, true);
        let before = Parser::new(&source).parse_and_validate_package().unwrap();
        let prepared = canonical_ir_text_to_prepared_gatify_ir(
            &source,
            Some("lookup"),
            &CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                ..CanonicalG8rOptions::default()
            },
        )
        .unwrap();
        let after = Parser::new(&prepared).parse_and_validate_package().unwrap();
        let f = after.get_top_fn().unwrap();
        assert_eq!(
            f.nodes
                .iter()
                .any(|n| matches!(n.payload, NodePayload::ArrayIndex { .. })),
            k == 0
        );
        for x in 0..32 {
            for strobe in [false, true] {
                let args = [
                    IrValue::make_ubits(5, x).unwrap(),
                    IrValue::make_ubits(1, strobe.into()).unwrap(),
                ];
                let FnEvalResult::Success(expected) = eval_fn(before.get_top_fn().unwrap(), &args)
                else {
                    panic!()
                };
                let FnEvalResult::Success(actual) = eval_fn(f, &args) else {
                    panic!()
                };
                assert_eq!(actual.value, expected.value);
            }
        }
    }
}

#[test]
fn preserves_an_irregular_low_page_and_clamped_addressing() {
    let irregular: Vec<_> = (0..32).map(|a| a / 4 + u128::from(a % 4 == 2)).collect();
    let ramp: Vec<_> = (0..32u128).map(|a| a / 4).collect();
    for source in [
        table_ir(5, 5, &irregular, false),
        table_ir(5, 6, &ramp, false),
        table_ir(5, 5, &ramp[..31], false),
        "package p
 top fn lookup(row: bits[2] id=1, col: bits[2] id=2) -> bits[3] {
            matrix: bits[3][4][4] = literal(value=[[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]], id=3)
            ret read: bits[3] = array_index(matrix, indices=[row,col], id=4)
        }"
        .into(),
    ] {
        let prepared = canonical_ir_text_to_prepared_gatify_ir(
            &source,
            Some("lookup"),
            &CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                ..CanonicalG8rOptions::default()
            },
        )
        .unwrap();
        let p = Parser::new(&prepared).parse_and_validate_package().unwrap();
        assert!(
            p.get_top_fn()
                .unwrap()
                .nodes
                .iter()
                .any(|n| matches!(n.payload, NodePayload::ArrayIndex { .. }))
        );
    }
}

#[test]
fn preserves_forced_lowerings_and_independent_node_operands() {
    let words: Vec<_> = (0..16u128).map(|i| (i + 3) / 2).collect();
    let source = table_ir(4, 4, &words, false);
    let p = Parser::new(&source).parse_and_validate_package().unwrap();
    let f = p.get_top_fn().unwrap();
    let ordinary = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_enabled());
    for (strategy, unsafe_gate) in [
        (ArrayIndexLoweringStrategy::ForceOobOneHot, false),
        (ArrayIndexLoweringStrategy::ForceNearPow2MuxTree, false),
        (ArrayIndexLoweringStrategy::Auto, true),
    ] {
        let opts = GatifyOptions {
            array_index_lowering_strategy: strategy,
            unsafe_gatify_gate_operation: unsafe_gate,
            ..GatifyOptions::all_opts_enabled()
        };
        let actual = gatify(f, opts.clone()).unwrap().gate_fn;
        let expected = gatify_prepared_fn(&ordinary, opts).unwrap().gate_fn;
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
    let projected = gatify_node_as_fn(
        f,
        f.ret_node_ref.unwrap(),
        &GatifyOptions::all_opts_enabled(),
    )
    .unwrap();
    assert_eq!(
        projected
            .inputs
            .iter()
            .map(|p| p.get_bit_count())
            .collect::<Vec<_>>(),
        vec![64, 4]
    );
    let mut sim = PreparedGateSim::new(&projected);
    // A replacement table independent of the table in the original program.
    for table in [0, u64::MAX] {
        for i in 0..16 {
            assert_eq!(
                sim.eval_outputs(&[
                    IrBits::make_ubits(64, table).unwrap(),
                    IrBits::make_ubits(4, i).unwrap()
                ]),
                vec![IrBits::make_ubits(4, table & 15).unwrap()]
            );
        }
    }
}

#[test]
fn low_address_bits_must_affect_the_regular_quotient() {
    for k in 1..=3u32 {
        let period = 1i32 << k;
        for direction in [-1, 1] {
            for phase in 0..period {
                let mut words: Vec<_> = (0..128)
                    .map(|a| (((32 * period + phase + direction * a) >> k) & 63) as u128)
                    .collect();
                // One low-address exception cannot make a coarse translation
                // fractional.
                words[0] = 57;
                let source = table_ir(6, 7, &words, false);
                let prepared = canonical_ir_text_to_prepared_gatify_ir(
                    &source,
                    Some("lookup"),
                    &CanonicalG8rOptions {
                        enable_formal_array_alias_analysis: false,
                        ..CanonicalG8rOptions::default()
                    },
                )
                .unwrap();
                let p = Parser::new(&prepared).parse_and_validate_package().unwrap();
                let f = p.get_top_fn().unwrap();
                let coarse_only = if direction > 0 {
                    phase == 0
                } else {
                    phase == period - 1
                };
                assert_eq!(
                    f.nodes
                        .iter()
                        .any(|node| matches!(node.payload, NodePayload::ArrayIndex { .. })),
                    coarse_only,
                    "k={k} direction={direction} phase={phase}"
                );
                for (a, &word) in words.iter().enumerate() {
                    let inputs = [
                        IrValue::make_ubits(7, a as u64).unwrap(),
                        IrValue::make_ubits(1, 0).unwrap(),
                    ];
                    let FnEvalResult::Success(actual) = eval_fn(f, &inputs) else {
                        panic!("evaluation failed")
                    };
                    assert_eq!(actual.value, IrValue::make_ubits(6, word as u64).unwrap());
                }
            }
        }
    }
}

#[test]
fn quotient_and_borrow_can_be_observed_together() {
    let words: Vec<u128> = (0..64u32)
        .map(|a| ((1u32.wrapping_sub(a) >> 2) & 7) as u128)
        .collect();
    let source = table_ir(3, 6, &words, false)
        .replace("-> bits[3]", "-> (bits[3], bits[1], bits[2])")
        .replace("ret read:", "read:")
        .replace(
            "\n}",
            "\n  low: bits[2] = bit_slice(addr, start=0, width=2, id=5)
  phase: bits[2] = literal(value=1, id=6)
  borrow: bits[1] = ugt(low, phase, id=7)
  ret result: (bits[3], bits[1], bits[2]) = tuple(read, borrow, low, id=8)\n}",
        );
    let prepared = canonical_ir_text_to_prepared_gatify_ir(
        &source,
        Some("lookup"),
        &CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            ..CanonicalG8rOptions::default()
        },
    )
    .unwrap();
    let parsed = Parser::new(&prepared).parse_and_validate_package().unwrap();
    let f = parsed.get_top_fn().unwrap();
    assert!(
        !f.nodes
            .iter()
            .any(|node| matches!(node.payload, NodePayload::ArrayIndex { .. }))
    );
    for (a, &word) in words.iter().enumerate() {
        let inputs = [
            IrValue::make_ubits(6, a as u64).unwrap(),
            IrValue::make_ubits(1, 0).unwrap(),
        ];
        let expected = IrValue::make_tuple(&[
            IrValue::make_ubits(3, word as u64).unwrap(),
            IrValue::make_ubits(1, u64::from(a % 4 > 1)).unwrap(),
            IrValue::make_ubits(2, (a % 4) as u64).unwrap(),
        ]);
        let FnEvalResult::Success(actual) = eval_fn(f, &inputs) else {
            panic!("evaluation failed")
        };
        assert_eq!(actual.value, expected);
    }
}

fn table_ir(w: usize, n: usize, words: &[u128], wrapper: bool) -> String {
    let words_text = words
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let word = format!("bits[{w}]");
    let table = format!("{word}[{}]", words.len());
    let (ret, mark, tail) = if wrapper {
        (
            format!("({word}, {word})"),
            "",
            format!(
                "\n  zero: {word} = literal(value=0, id=5)\n  enabled: {word} = sel(en, cases=[zero, read], id=6)\n  ret pair: ({word}, {word}) = tuple(read, enabled, id=7)"
            ),
        )
    } else {
        (word.clone(), "ret ", String::new())
    };
    format!(
        "package p\n\ntop fn lookup(addr: bits[{n}] id=1, en: bits[1] id=2) -> {ret} {{\n  mem: {table} = literal(value=[{words_text}], id=3)\n  {mark}read: {word} = array_index(mem, indices=[addr], id=4){tail}\n}}"
    )
}
