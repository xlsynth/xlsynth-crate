// SPDX-License-Identifier: Apache-2.0

use test_case::test_case;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig::topo::extract_cone;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{
    ArrayIndexLoweringStrategy, GatifyOptions, gatify, gatify_node_as_fn, gatify_prepared_fn,
};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, canonical_ir_text_to_prepared_gatify_ir};
use xlsynth_pir::{
    IrBits, IrValue,
    ir::NodePayload,
    ir_eval::{FnEvalResult, eval_fn},
    ir_parser::Parser,
    ir_utils::find_node_by_name,
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

fn rounded_subtract_table() -> Vec<u128> {
    (0..64u32)
        .map(|a| ((1u32.wrapping_sub(a) >> 2) & 7) as u128)
        .collect()
}

#[test]
fn quotient_and_borrow_can_be_observed_together() {
    let words = rounded_subtract_table();
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

#[test_case(1, "ret bit: bits[1] = bit_slice(read, start=3, width=1, id=4)"; "single bit")]
#[test_case(5, "ret bits: bits[5] = bit_slice(read, start=3, width=5, id=4)"; "constant upper columns")]
#[test_case(1, r#"mask: bits[16] = literal(value=8, id=4)
  masked: bits[16] = and(read, mask, id=5)
  ret bit: bits[1] = or_reduce(masked, id=6)"#; "masked read")]
fn sparse_observed_bit_keeps_the_single_row_decode(width: usize, consumer: &str) {
    let words = (0..64)
        .map(|a| if a < 16 { (a + 1) >> 1 } else { 0 })
        .map(|word| word.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        r#"fn lookup(addr: bits[6] id=1) -> bits[{width}] {{
  table: bits[16][64] = literal(value=[{words}], id=2)
  read: bits[16] = array_index(table, indices=[addr], assumed_in_bounds=true, id=3)
  {consumer}
}}"#
    );
    let f = Parser::new(&source).parse_fn().unwrap();
    let gates = gatify(&f, GatifyOptions::all_opts_enabled())
        .unwrap()
        .gate_fn;
    let mut sim = PreparedGateSim::new(&gates);
    for a in 0..64 {
        assert_eq!(
            sim.eval_outputs(&[IrBits::make_ubits(6, a).unwrap()]),
            vec![IrBits::make_ubits(width, u64::from(a == 15)).unwrap()]
        );
    }
    // Equality with a six-bit literal needs only five two-input ANDs.
    assert_eq!(get_aig_stats(&gates).and_nodes, 5);
}

#[test]
fn lookup_arithmetic_area_belongs_to_the_read() {
    let source = table_ir(3, 6, &rounded_subtract_table(), false);
    let p = Parser::new(&source).parse_and_validate_package().unwrap();
    let f = p.get_top_fn().unwrap();
    let options = GatifyOptions {
        track_pir_node_ids: true,
        ..GatifyOptions::all_opts_enabled()
    };
    let gates = gatify(f, options).unwrap().gate_fn;
    let area = build_area_table_report(&gates, f).unwrap();
    assert_eq!(
        area.rows.iter().map(|r| r.pir_node_id).collect::<Vec<_>>(),
        [4]
    );
    assert_eq!(area.unattributed, None);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn tracking_an_unused_lookup_preserves_the_returned_parameter() {
    let source = table_ir(3, 6, &rounded_subtract_table(), false);
    let p = Parser::new(&source).parse_and_validate_package().unwrap();
    let mut f = p.get_top_fn().unwrap().clone();
    let read = f.ret_node_ref.unwrap();
    f.get_node_mut(read).text_id = u32::MAX as usize + 1;
    f.ret_node_ref = Some(f.params[1]);
    f.ret_ty = f.get_node_ty(f.params[1]).clone();
    let lowered = gatify(
        &f,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    assert!(!lowered.lowering_map.contains_key(&read));
    let mut sim = PreparedGateSim::new(&lowered.gate_fn);
    for en in [false, true] {
        assert_eq!(
            sim.eval_outputs(&[IrBits::zero(6), IrBits::bool(en)]),
            vec![IrBits::bool(en)]
        );
    }
}

fn wide_lookup_word(addr: u64) -> u64 {
    if addr < 16 { addr } else { (addr + 1) >> 3 }
}

/// Pair a rounded lookup with 15 exceptional rows and an independent predicate.
fn wide_lookup_and_predicate(predicate: &str) -> String {
    let words = (0..64)
        .map(wide_lookup_word)
        .map(|word| word.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"fn lookup(addr: bits[6] id=1, x: bits[8] id=2) -> (bits[64], bits[1]) {{
  table: bits[64][64] = literal(value=[{words}], id=3)
  read: bits[64] = array_index(table, indices=[addr], id=4)
  {predicate}
  ret result: (bits[64], bits[1]) = tuple(read, matched, id=13)
}}"#
    )
}

#[test]
fn wide_lookup_preserves_the_conjunctions_gate_provenance() {
    let source = wide_lookup_and_predicate(
        r#"one: bits[8] = literal(value=1, id=5)
  word: bits[8] = add(x, one, id=6)
  zero: bits[4] = literal(value=0, id=7)
  lo: bits[4] = bit_slice(word, start=0, width=4, id=8)
  p: bits[1] = eq(lo, zero, id=9)
  hi: bits[4] = bit_slice(word, start=4, width=4, id=10)
  q: bits[1] = eq(hi, zero, id=11)
  matched: bits[1] = and(p, q, id=12)"#,
    );
    let output = ir2gates_from_ir_text(
        &format!("package p\n\ntop {source}"),
        None,
        Ir2GatesOptions {
            track_pir_node_ids: true,
            enable_formal_array_alias_analysis: false,
            ..Ir2GatesOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let join = find_node_by_name(output.pir_top_fn(), "matched").unwrap();
    let lowered = output.gatify_output;
    let join_gate = lowered.lowering_map[&join].get_lsb(0).node;
    let gates = &lowered.gate_fn.gates;
    let (cone, _) = extract_cone(&[join_gate], gates);
    // Testing x + 1 == 0 reduces to a single eight-input AND.
    let origins: Vec<_> = cone
        .iter()
        .map(|g| gates[g.id].get_pir_node_ids())
        .collect();
    assert_eq!(origins, [&[12][..]; 7]);
}

#[test]
fn lookup_reserves_ids_after_the_clz_predicate() {
    let source = wide_lookup_and_predicate(
        r#"count: bits[4] = ext_clz(x, offset=0, new_bit_count=4, id=5)
  eight: bits[4] = literal(value=8, id=6)
  matched: bits[1] = eq(count, eight, id=7)"#,
    );
    let mut f = Parser::new(&source).parse_fn().unwrap();
    let options = GatifyOptions::all_opts_enabled();
    let automatic = gatify(&f, options.clone()).unwrap();
    let ordinary = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
    let decoder = gatify_prepared_fn(&ordinary, options.clone()).unwrap();
    let automatic_stats = get_aig_stats(&automatic.gate_fn);
    let decoder_stats = get_aig_stats(&decoder.gate_fn);
    assert_ne!(automatic_stats, decoder_stats);
    // Programmatic PIR can retain the explicit opcode name on the read.
    let read = find_node_by_name(&f, "read").unwrap();
    f.get_node_mut(read).name = Some("array_index.4".to_string());
    let result = f.ret_node_ref.unwrap();
    // The CLZ predicate emits one helper and the lookup needs another 70.
    for (remaining, expected) in [(70, decoder_stats), (71, automatic_stats)] {
        f.get_node_mut(result).text_id = usize::MAX - remaining;
        let lowered = gatify(&f, options.clone()).unwrap();
        assert_eq!(get_aig_stats(&lowered.gate_fn), expected);
        let mut sim = PreparedGateSim::new(&lowered.gate_fn);
        for addr in 0..64 {
            for x in [0, 1] {
                let pair = (wide_lookup_word(addr) << 1) | u64::from(x == 0);
                assert_eq!(
                    sim.eval_outputs(&[
                        IrBits::make_ubits(6, addr).unwrap(),
                        IrBits::make_ubits(8, x).unwrap()
                    ]),
                    vec![IrBits::make_ubits(65, pair).unwrap()]
                );
            }
        }
    }
}

#[test_case(1, false; "one ID left")]
#[test_case(5, false; "one ID short")]
#[test_case(6, true; "exact fit")]
fn lookup_requires_ids_for_all_helpers(remaining: usize, fits: bool) {
    let source = table_ir(3, 6, &rounded_subtract_table(), false);
    let p = Parser::new(&source).parse_and_validate_package().unwrap();
    let mut f = p.get_top_fn().unwrap().clone();
    let options = GatifyOptions::all_opts_enabled();
    let expected = if fits {
        gatify(&f, options.clone())
    } else {
        gatify_prepared_fn(&f, options.clone())
    }
    .unwrap();
    let read = f.ret_node_ref.unwrap();
    f.get_node_mut(read).text_id = usize::MAX - remaining;
    let actual = gatify(&f, options).unwrap();
    assert_eq!(
        get_aig_stats(&actual.gate_fn),
        get_aig_stats(&expected.gate_fn)
    );
}

#[test]
fn verification_budget_retains_the_verified_fractional_model() {
    let words = (0..1024u128)
        .map(|a| if a < 12 { a } else { ((a + 1) >> 1) & 255 })
        .collect::<Vec<_>>();
    let source = table_ir(8, 10, &words, false);
    let prepared = canonical_ir_text_to_prepared_gatify_ir(
        &source,
        Some("lookup"),
        &CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            ..CanonicalG8rOptions::default()
        },
    )
    .unwrap();
    let package = Parser::new(&prepared).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    assert!(
        !f.nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::ArrayIndex { .. }))
    );
    let gates = gatify_prepared_fn(f, GatifyOptions::all_opts_enabled())
        .unwrap()
        .gate_fn;
    let mut sim = PreparedGateSim::new(&gates);
    for (a, &word) in words.iter().enumerate() {
        assert_eq!(
            sim.eval_outputs(&[IrBits::make_ubits(10, a as u64).unwrap(), IrBits::zero(1)]),
            vec![IrBits::make_ubits(8, word as u64).unwrap()]
        );
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
