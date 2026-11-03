// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, PreparedGateSim};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir::{Binop, ExtNaryAddArchitecture, Fn, NodePayload, Type, Unop};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils::find_node_by_name;
use xlsynth_pir::{FnBuilder, IrBits, IrValue};
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn merged_comparisons_lower_with_their_own_range_facts_and_provenance() {
    let text = r#"package sample
top fn main(x: bits[1] id=1, guard: bits[1] id=2) -> bits[1] {
  zero: bits[1] = literal(value=0, id=3)
  input: bits[2] = concat(x, zero, id=4)
  one: bits[2] = literal(value=1, id=5)
  word: bits[2] = add(input, one, id=6)
  lo: bits[1] = bit_slice(word, start=0, width=1, id=7)
  hi: bits[1] = bit_slice(word, start=1, width=1, id=8)
  on: bits[1] = literal(value=1, id=9)
  p: bits[1] = eq(lo, on, id=10)
  q: bits[1] = eq(hi, on, id=11)
  nested: bits[1] = and(p, on, id=12)
  joined: bits[1] = and(nested, q, id=13)
  ret out: bits[1] = and(joined, guard, id=14)
}"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let f = package.get_fn("main").unwrap();
    let package = xlsynth::IrPackage::parse_ir(text, None).unwrap();
    let analysis = package.create_ir_analysis().unwrap();
    let ranges = IrRangeInfo::build_from_analysis(&analysis, f).unwrap();
    for name in ["p", "nested"] {
        let id = f.get_node(find_node_by_name(f, name).unwrap()).text_id;
        let known = ranges.get(id).unwrap().known_bits.as_ref().unwrap();
        assert_eq!(known.mask(), &IrBits::bool(true));
        assert_eq!(known.value(), &IrBits::bool(true));
    }
    let lowered = gatify(
        f,
        GatifyOptions {
            range_info: Some(ranges),
            track_pir_node_ids: true,
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let mut sim = PreparedGateSim::new(&lowered.gate_fn);
    for x in [false, true] {
        for guard in [true, false] {
            let sample = sim.eval(
                &[IrBits::bool(x), IrBits::bool(guard)],
                Collect::AllWithInputs,
            );
            assert_eq!(
                sample.outputs,
                vec![IrBits::bool(x && guard)],
                "x={x} guard={guard}"
            );
            let values = sample.all_values.unwrap();
            for (name, expected) in [("joined", x), ("out", x && guard), ("guard", guard)] {
                let original = find_node_by_name(f, name).unwrap();
                let gate = lowered.lowering_map[&original].get_lsb(0);
                assert_eq!(values[gate.node.id] ^ gate.negated, expected, "{name}");
            }
        }
    }
    for name in ["p", "q", "nested", "word"] {
        let original = find_node_by_name(f, name).unwrap();
        assert!(!lowered.lowering_map.contains_key(&original), "{name}");
        let id = u32::try_from(f.get_node(original).text_id).unwrap();
        assert!(
            lowered
                .gate_fn
                .gates
                .iter()
                .all(|g| !g.get_pir_node_ids().contains(&id)),
            "{name}"
        );
    }
}

#[test]
fn merged_comparator_gates_belong_to_the_original_conjunction() {
    let mut b = FnBuilder::new("nibbles");
    let x = b.param("x", Type::Bits(8)).unwrap();
    let one = b.literal(IrValue::make_ubits(8, 1).unwrap()).unwrap();
    let word = b.add(x, one).unwrap();
    let zero = b.literal(IrValue::make_ubits(4, 0).unwrap()).unwrap();
    let [lo, hi] = [0, 4].map(|start| {
        let nibble = b.bit_slice(word, start, 4).unwrap();
        b.eq(nibble, zero).unwrap()
    });
    let both = b.and(lo, hi).unwrap();
    let f = b.build(both).unwrap();
    let id = u32::try_from(f.get_node(f.ret_node_ref.unwrap()).text_id).unwrap();
    let lowered = gatify(
        &f,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let ands: Vec<_> = lowered
        .gate_fn
        .gates
        .iter()
        .filter(|g| matches!(g, AigNode::And2 { .. }))
        .collect();
    assert!(!ands.is_empty());
    for gate in ands {
        assert_eq!(gate.get_pir_node_ids(), &[id]);
    }
}

#[test]
fn merged_comparator_inherits_the_origin_of_a_mask_pruned_join() {
    let f = Parser::new(
        r#"fn nibbles(x: bits[8] id=1, y: bits[2] id=2) -> bits[1] {
  one: bits[8] = literal(value=1, id=3)
  word: bits[8] = add(x, one, id=4)
  zero: bits[4] = literal(value=0, id=5)
  lo: bits[4] = bit_slice(word, start=0, width=4, id=6)
  hi: bits[4] = bit_slice(word, start=4, width=4, id=7)
  p: bits[1] = eq(lo, zero, id=8)
  q: bits[1] = eq(hi, zero, id=9)
  mask: bits[1] = or_reduce(y, id=10)
  inner: bits[1] = and(mask, p, q, id=11)
  parity: bits[1] = xor_reduce(y, id=12)
  ret out: bits[1] = and(parity, inner, id=13)
}"#,
    )
    .parse_fn()
    .unwrap();
    let lowered = gatify(
        &f,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let mut sim = PreparedGateSim::new(&lowered.gate_fn);
    for x in 0..256 {
        for y in 0..4 {
            let inputs = [
                IrBits::make_ubits(8, x).unwrap(),
                IrBits::make_ubits(2, y).unwrap(),
            ];
            assert_eq!(
                sim.eval_outputs(&inputs),
                [IrBits::bool(x == 255 && y.count_ones() == 1)]
            );
        }
    }
    // At x = 255 and y = 0 the merged comparison is true but inner was false.
    for name in ["p", "q", "inner"] {
        let original = find_node_by_name(&f, name).unwrap();
        assert!(!lowered.lowering_map.contains_key(&original), "{name}");
    }
    // The seven ANDs for x == 255 all belong to the original inner AND.
    let area = build_area_table_report(&lowered.gate_fn, &f).unwrap();
    let comparison_area: Vec<_> = area
        .rows
        .iter()
        .filter(|row| row.pir_node_id != 12 && row.pir_node_id != 13)
        .map(|row| (row.pir_node_id, row.weighted_aig_node_count))
        .collect();
    assert_eq!(comparison_area, [(11, 7.0)]);
    assert_eq!(area.unattributed, None);
}

#[test]
fn interleaved_groups_preserve_ids_only_after_both_groups_rejoin() {
    let mut b = FnBuilder::new("interleaved");
    let x = b.param("x", Type::Bits(2)).unwrap();
    let y = b.param("y", Type::Bits(2)).unwrap();
    let one = b.literal(IrValue::make_ubits(2, 1).unwrap()).unwrap();
    let on = b.literal(IrValue::bool(true)).unwrap();
    let [[x_lo, x_hi], [y_lo, y_hi]] = [x, y].map(|input| {
        let word = b.add(input, one).unwrap();
        [0, 1].map(|bit| {
            let slice = b.bit_slice(word, bit, 1).unwrap();
            b.eq(slice, on).unwrap()
        })
    });
    let partial = b.and(x_lo, y_lo).unwrap();
    b.set_name(partial, "partial").unwrap();
    let all_x = b.and(partial, x_hi).unwrap();
    b.set_name(all_x, "all_x").unwrap();
    let out = b.and(all_x, y_hi).unwrap();
    let f = b.build(out).unwrap();
    let lowered = gatify(&f, GatifyOptions::all_opts_disabled()).unwrap();
    // all_x still contains only part of the y group.
    for name in ["partial", "all_x"] {
        assert!(
            !lowered
                .lowering_map
                .contains_key(&find_node_by_name(&f, name).unwrap())
        );
    }
    assert!(lowered.lowering_map.contains_key(&f.ret_node_ref.unwrap()));
    let mut sim = PreparedGateSim::new(&lowered.gate_fn);
    for x in 0..4 {
        for y in 0..4 {
            let inputs = [x, y].map(|n| IrBits::make_ubits(2, n).unwrap());
            assert_eq!(
                sim.eval_outputs(&inputs),
                vec![IrBits::bool(x == 2 && y == 2)]
            );
        }
    }
}

#[test]
fn merges_adjacent_fields_across_a_guard_and_inverts_the_operand_order() {
    for (operation, input_residue) in [
        ("add(x, c, id=4)", 18),
        ("add(c, x, id=4)", 18),
        ("sub(x, c, id=4)", 28),
        ("sub(c, x, id=4)", 14),
    ] {
        let source = function(6, 5, operation, "bits[1]", NESTED_FIELDS);
        let after = prove_preparation(&source);
        assert!(!has_sum(&after));
        for x in 0..64 {
            for guard in [false, true] {
                assert_eq!(
                    evaluate(&after, 6, x, guard),
                    guard && x % 32 == input_residue
                );
            }
        }
    }
}

#[test]
fn a_free_gap_can_either_absorb_or_propagate_a_borrow() {
    // Constrain difference[5:4] = 3 and difference[1:0] = 0.
    let sparse = NESTED_FIELDS
        .replace("bits[3]", "bits[2]")
        .replace("start=2, width=3", "start=4, width=2")
        .replace("value=3, id=7", "value=0, id=7")
        .replace("value=5, id=8", "value=3, id=8");
    let stable = prove_preparation(&function(6, 13, "sub(c, x, id=4)", "bits[1]", &sparse));
    assert!(!has_sum(&stable));
    let varying = prove_preparation(&function(6, 1, "sub(c, x, id=4)", "bits[1]", &sparse));
    for x in 0..64 {
        assert_eq!(evaluate(&stable, 6, x, true), x & 0x33 == 0x11);
        let y = 1_u64.wrapping_sub(x) & 63;
        assert_eq!(evaluate(&varying, 6, x, true), y & 0x33 == 0x30);
    }
    // These words agree on every constrained input bit. Only the gap differs.
    assert_ne!(
        evaluate(&varying, 6, 17, true),
        evaluate(&varying, 6, 21, true)
    );
}

#[test]
fn overlapping_fields_in_a_single_view_must_agree() {
    // The low bit of hi and the high bit of lo both constrain word[2].
    for (target, consistent) in [(0b01_100, true), (0b01_000, false)] {
        let observer = format!(
            r#"
          lo: bits[3] = bit_slice(word, start=0, width=3, id=5)
          hi: bits[2] = bit_slice(word, start=2, width=2, id=6)
          packed: bits[5] = concat(hi, lo, id=7)
          target: bits[5] = literal(value={target}, id=8)
          ret out: bits[1] = eq(packed, target, id=9)"#
        );
        let after = prove_preparation(&function(6, 5, "add(x, c, id=4)", "bits[1]", &observer));
        assert!(!has_sum(&after));
        for x in 0..64 {
            assert_eq!(evaluate(&after, 6, x, true), consistent && x % 16 == 15);
        }
    }
}

#[test]
fn disjoined_inequalities_and_concatenated_views_preserve_polarity() {
    let dual = NESTED_FIELDS
        .replace(" = eq(", " = ne(")
        .replace(" = and(", " = or(");
    let cat = r#"
  lo: bits[2] = bit_slice(word, start=0, width=2, id=5)
  hi: bits[3] = bit_slice(word, start=2, width=3, id=6)
  packed: bits[5] = concat(lo, hi, id=7)
  target: bits[5] = literal(value=29, id=8)
  cmp: bits[1] = eq(target, packed, id=9)
  ret out: bits[1] = and(guard, cmp, id=10)"#;
    for (observer, is_dual) in [(dual.as_str(), true), (cat, false)] {
        let after = prove_preparation(&function(6, 5, "sub(c, x, id=4)", "bits[1]", observer));
        assert!(!has_sum(&after));
        for x in 0..64 {
            for guard in [false, true] {
                let expected = if is_dual {
                    guard || x % 32 != 14
                } else {
                    guard && x % 32 == 14
                };
                assert_eq!(evaluate(&after, 6, x, guard), expected);
            }
        }
    }
}

#[test]
fn a_limb_crossing_and_one_observed_bit_are_proved_at_their_actual_widths() {
    // A carry out of bit 62 changes the bit observed by the next field.
    let across = r#"
  lo: bits[63] = bit_slice(word, start=0, width=63, id=5)
  hi: bits[5] = bit_slice(word, start=63, width=5, id=6)
  a: bits[63] = literal(value=0x7ffffffffffffffe, id=7)
  b: bits[5] = literal(value=19, id=8)
  p: bits[1] = eq(lo, a, id=9)
  q: bits[1] = eq(b, hi, id=10)
  ret out: bits[1] = and(guard, q, p, id=11)"#;
    let after = prove_preparation(&function(
        73,
        (1_u128 << 68) + 5,
        "sub(c, x, id=4)",
        "bits[1]",
        across,
    ));
    assert!(!has_sum(&after));
    let one = r#"
  low: bits[1] = bit_slice(word, start=0, width=1, id=5)
  bit: bits[1] = literal(value=1, id=6)
  pred: bits[1] = eq(bit, low, id=7)
  ret out: bits[1] = and(pred, guard, id=8)"#;
    for width in [1, 7] {
        let f = prove_preparation(&function(width, 1, "add(x, c, id=4)", "bits[1]", one));
        for x in 0..1 << width {
            assert_eq!(evaluate(&f, width, x, true), x & 1 == 0);
        }
    }
}

#[test]
fn preserve_scheduled_adders_and_independently_observable_values() {
    let scheduled =
        "ext_nary_add(c, x, signed=[false, false], negated=[false, true], arch=kogge_stone, id=4)";
    let after = prove_preparation(&function(6, 5, scheduled, "bits[1]", NESTED_FIELDS));
    assert!(after.nodes.iter().any(|n| matches!(
        n.payload,
        NodePayload::ExtNaryAdd {
            arch: Some(ExtNaryAddArchitecture::KoggeStone),
            ..
        }
    )));
    let exposed = NESTED_FIELDS.replace("ret out:", "observed:")
        + "\n  ret out: (bits[1], bits[6]) = tuple(observed, word, id=13)";
    let after = prove_preparation(&function(
        6,
        5,
        "sub(c, x, id=4)",
        "(bits[1], bits[6])",
        &exposed,
    ));
    assert!(has_sum(&after));
    // The guarded low equality is optional here, so it cannot constrain the
    // high test.
    let optional = NESTED_FIELDS.replace("and(p, guard", "or(p, guard");
    let after = prove_preparation(&function(6, 5, "sub(c, x, id=4)", "bits[1]", &optional));
    assert!(evaluate(&after, 6, 17, true));
    assert!(!evaluate(&after, 6, 17, false));
}

#[test]
fn whole_word_zero_tests_keep_the_negation_and_independent_sum() {
    let adjusted = r#"
      high: bits[3] = bit_slice(x, start=3, width=3, id=4)
      step: bits[3] = literal(value=3, id=5)
      shifted: bits[3] = add(step, high, id=6)
      low: bits[3] = bit_slice(x, start=0, width=3, id=7)
      packed: bits[6] = concat(shifted, low, id=8)"#;
    for (prefix, operand, relation, combine, matching_x) in [
        ("", "x", "eq", "and", 0),
        (adjusted, "packed", "ne", "nor", 40),
    ] {
        let text = format!(
            r#"fn guarded(x: bits[6] id=1, counter: bits[8] id=2, enable: bits[1] id=3) -> bits[1] {{
              {prefix}
              negated: bits[6] = neg({operand}, id=9)
              zero: bits[6] = literal(value=0, id=10)
              observes_word: bits[1] = {relation}(zero, negated, id=11)
              increment: bits[8] = zero_ext(enable, new_bit_count=8, id=12)
              sum: bits[8] = add(counter, increment, id=13)
              count_zero: bits[8] = literal(value=0, id=14)
              observes_sum: bits[1] = {relation}(sum, count_zero, id=15)
              ret result: bits[1] = {combine}(observes_word, observes_sum, id=16)
            }}"#
        );
        let after = prove_preparation(&text);
        let mut compares_negation = false;
        let mut compares_sum = false;
        for node in &after.nodes {
            if let NodePayload::Binop(Binop::Eq | Binop::Ne, lhs, rhs) = node.payload {
                for arg in [lhs, rhs] {
                    let arg = after.get_node(arg);
                    compares_negation |= matches!(arg.payload, NodePayload::Unop(Unop::Neg, _));
                    compares_sum |= arg.ty.bit_count() == 8
                        && matches!(
                            arg.payload,
                            NodePayload::ExtNaryAdd { .. } | NodePayload::Binop(Binop::Add, _, _)
                        );
                }
            }
        }
        assert!(compares_negation && compares_sum, "{after}");
        for (x, count, enable, expected) in [
            (matching_x, 255, 1, true),
            (matching_x, 255, 0, false),
            (matching_x ^ 1, 0, 0, false),
        ] {
            let args = [
                IrValue::make_ubits(6, x).unwrap(),
                IrValue::make_ubits(8, count).unwrap(),
                IrValue::make_ubits(1, enable).unwrap(),
            ];
            let FnEvalResult::Success(result) = eval_fn(&after, &args) else {
                panic!("IR evaluation failed")
            };
            assert_eq!(
                result.value,
                IrValue::make_ubits(1, u64::from(expected)).unwrap()
            );
        }
    }
}

#[test]
fn preserve_shared_source_bits_but_allow_independent_fields() {
    // x[4:0] is the part of c - x needed by the two comparisons.
    let observed = NESTED_FIELDS.replace("ret out:", "observed:");
    for (start, source, retained) in [(0, "x", true), (5, "x", false), (1, "alias", true)] {
        let context = format!(
            r#"
          {observed}
          high: bits[4] = bit_slice(x, start=4, width=4, id=13)
          low: bits[4] = bit_slice(x, start=0, width=4, id=14)
          alias: bits[8] = concat(high, low, id=15)
          piece: bits[3] = bit_slice({source}, start={start}, width=3, id=16)
          target: bits[3] = literal(value=2, id=17)
          independent: bits[1] = eq(piece, target, id=18)
          ret out: bits[2] = concat(observed, independent, id=19)"#
        );
        let after = prove_preparation(&function(8, 5, "sub(c, x, id=4)", "bits[2]", &context));
        assert_eq!(
            has_sum(&after),
            retained,
            "{source} from bit {start}: {after}"
        );
        for x in 0..256 {
            for guard in [false, true] {
                let args = [
                    IrValue::make_ubits(8, x).unwrap(),
                    IrValue::make_ubits(1, u64::from(guard)).unwrap(),
                ];
                let FnEvalResult::Success(result) = eval_fn(&after, &args) else {
                    panic!()
                };
                let expected =
                    ((guard && x % 32 == 14) as u64) * 2 + (((x >> start) & 7) == 2) as u64;
                assert_eq!(result.value, IrValue::make_ubits(2, expected).unwrap());
            }
        }
    }
}

#[test]
fn distinguish_fields_and_bits_of_an_external_tuple_input() {
    let observed = NESTED_FIELDS.replace("ret out:", "observed:");
    for (field, start, retain) in [(0, 0, true), (0, 5, false), (1, 0, false)] {
        let text = format!(
            r#"fn main(pair: (bits[8], bits[8]) id=1, guard: bits[1] id=2) -> (bits[1], bits[1]) {{
          x: bits[8] = tuple_index(pair, index=0, id=21)
          c: bits[8] = literal(value=5, id=3)
          word: bits[8] = sub(c, x, id=4)
          {observed}
          sibling: bits[8] = tuple_index(pair, index={field}, id=13)
          piece: bits[3] = bit_slice(sibling, start={start}, width=3, id=14)
          target: bits[3] = literal(value=2, id=15)
          external: bits[1] = eq(piece, target, id=16)
          ret out: (bits[1], bits[1]) = tuple(observed, external, id=17)
        }}"#
        );
        let after = prove_preparation(&text);
        assert_eq!(
            has_sum(&after),
            retain,
            "field {field} from bit {start}: {after}"
        );
        for (x, y, guard) in [(14, 2, true), (78, 9, false), (2, 14, true)] {
            let pair = IrValue::make_tuple(&[
                IrValue::make_ubits(8, x).unwrap(),
                IrValue::make_ubits(8, y).unwrap(),
            ]);
            let args = [pair, IrValue::make_ubits(1, u64::from(guard)).unwrap()];
            let FnEvalResult::Success(result) = eval_fn(&after, &args) else {
                panic!()
            };
            let external = if field == 0 { x } else { y };
            let expect = IrValue::make_tuple(&[
                IrValue::make_ubits(1, u64::from(guard && x % 32 == 14)).unwrap(),
                IrValue::make_ubits(1, u64::from(((external >> start) & 7) == 2)).unwrap(),
            ]);
            assert_eq!(result.value, expect);
        }
    }
}

const NESTED_FIELDS: &str = r#"
  lo: bits[2] = bit_slice(word, start=0, width=2, id=5)
  hi: bits[3] = bit_slice(word, start=2, width=3, id=6)
  l: bits[2] = literal(value=3, id=7)
  h: bits[3] = literal(value=5, id=8)
  p: bits[1] = eq(lo, l, id=9)
  q: bits[1] = eq(h, hi, id=10)
  nested: bits[1] = and(p, guard, id=11)
  ret out: bits[1] = and(q, nested, id=12)"#;

fn function(width: usize, constant: u128, op: &str, output_type: &str, observer: &str) -> String {
    format!(
        "fn main(x: bits[{width}] id=1, guard: bits[1] id=2) -> {output_type} {{\n  c: bits[{width}] = literal(value={constant:#x}, id=3)\n  word: bits[{width}] = {op}\n{observer}\n}}"
    )
}

fn prove_preparation(text: &str) -> Fn {
    let original = Parser::new(text).parse_fn().unwrap();
    let prepared = prep_for_gatify(
        &original,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    let proof = prove_prepared_ir_equiv(&original, &prepared);
    assert!(
        matches!(proof, EquivResult::Proved),
        "{proof:?}\n{text}\n{prepared}"
    );
    prepared
}

fn has_sum(f: &Fn) -> bool {
    f.nodes.iter().any(|n| {
        matches!(
            n.payload,
            NodePayload::ExtNaryAdd { .. } | NodePayload::Binop(Binop::Add | Binop::Sub, _, _)
        )
    })
}

fn evaluate(f: &Fn, width: usize, x: u64, guard: bool) -> bool {
    let args = [
        IrValue::make_ubits(width, x).unwrap(),
        IrValue::make_ubits(1, u64::from(guard)).unwrap(),
    ];
    let FnEvalResult::Success(result) = eval_fn(f, &args) else {
        panic!("IR evaluation failed")
    };
    result.value.to_bits().unwrap().to_u64().unwrap() != 0
}
