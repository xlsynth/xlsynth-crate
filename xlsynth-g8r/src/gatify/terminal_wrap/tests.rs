// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::gatify::ir2gate::{
    GatifyOptions, gatify, gatify_prepared_fn, gatify_with_terminal_observations,
};
use crate::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser::Parser;

const SCALAR: &str = r#"package cap
top fn f(x: bits[5] id=1, c: bits[1] id=2) -> bits[1] {
  inc: bits[5] = zero_ext(c, new_bit_count=5, id=3)
  sum: bits[5] = add(x, inc, id=4)
  low: bits[5] = bit_slice(sum, start=0, width=5, id=5)
  zero: bits[5] = literal(value=0, id=6)
  ret p: bits[1] = eq(zero, low, id=7)
}"#;

fn parse(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone()
}

fn ordinary(text: &str) -> Option<TerminalWrapCapture> {
    let f = parse(text);
    let opt = GatifyOptions::all_opts_enabled();
    let reference = gatify(&f, opt.clone()).unwrap();
    let (observed, observations) = gatify_with_terminal_observations(&f, opt, true).unwrap();
    assert_eq!(
        reference.gate_fn.to_string_including_dead_nodes(),
        observed.gate_fn.to_string_including_dead_nodes()
    );
    observations.wrap
}

#[test]
fn recognizes_commuted_zero_observations_and_both_add_orders() {
    for (op, equal) in [
        ("eq(low, zero", true),
        ("eq(zero, low", true),
        ("ne(low, zero", false),
        ("ne(zero, low", false),
        ("or_reduce(low", false),
    ] {
        for add in ["add(x, inc", "add(inc, x"] {
            let ir = SCALAR
                .replace("eq(zero, low", op)
                .replace("add(x, inc", add);
            let captured = ordinary(&ir).unwrap();
            assert_eq!(captured.equal, equal);
            assert_eq!(captured.word_lsb.len(), 5);
            assert_eq!(
                captured.outputs,
                vec![vec![ReturnBit::Predicate { inverted: false }]]
            );
        }
    }
}

#[test]
fn retains_signed_duplicate_constant_five_bit_continuation_in_output_order() {
    let ir = SCALAR.replace("-> bits[1]", "-> (bits[1], bits[1], bits[3])").replace("ret p:", "p:")
        .replace("\n}", "\n  inv: bits[1] = not(p, id=8)\n  one: bits[1] = literal(value=1, id=9)\n  bus: bits[3] = concat(p, one, inv, id=10)\n  ret out: (bits[1], bits[1], bits[3]) = tuple(p, inv, bus, id=11)\n}");
    let p = ReturnBit::Predicate { inverted: false };
    let n = ReturnBit::Predicate { inverted: true };
    assert_eq!(
        ordinary(&ir).unwrap().outputs,
        vec![vec![n, ReturnBit::Constant(true), p, n, p]]
    );
}

#[test]
fn abstains_on_retained_sum_explicit_extended_arithmetic_and_word_related_carry() {
    let retained = SCALAR
        .replace("-> bits[1]", "-> (bits[1], bits[5])")
        .replace("ret p:", "p:")
        .replace(
            "\n}",
            "\n  ret out: (bits[1], bits[5]) = tuple(p, sum, id=9)\n}",
        );
    let forced = SCALAR.replace("  inc:", "  x0: bits[1] = bit_slice(x, start=0, width=1, id=8)\n  flag: bits[1] = and(x0, c, id=9)\n  inc:")
        .replace("zero_ext(c,", "zero_ext(flag,");
    for text in [
        retained,
        forced,
        SCALAR.replace("zero_ext(c", "sign_ext(c"),
        SCALAR.replace(
            "add(x, inc, id=4)",
            "ext_nary_add(x, c, signed=[false, false], negated=[false, false], id=4)",
        ),
    ] {
        assert!(ordinary(&text).is_none(), "{text}");
    }
}

#[test]
fn needs_fully_resolved_lowerings_and_only_real_predicate_or_literal_outputs() {
    let f = parse(SCALAR);
    let out = gatify_prepared_fn(&f, GatifyOptions::all_opts_enabled()).unwrap();
    let permit = || TerminalWrapPermit::at_source(&f).unwrap();
    let real = permit()
        .capture(&f, &out.gate_fn, |n| out.lowering_map.get(&n))
        .unwrap();
    assert_eq!(real.word_lsb.len(), 5);
    for text_id in [1, 2, 7] {
        assert!(
            permit()
                .capture(&f, &out.gate_fn, |n| (f.get_node(n).text_id != text_id)
                    .then(|| out.lowering_map.get(&n))
                    .flatten())
                .is_none()
        );
    }
    let mut bad = out.gate_fn;
    let literal = bad
        .gates
        .iter()
        .position(|n| matches!(n, AigNode::Literal { .. }))
        .unwrap();
    bad.outputs[0].bit_vector =
        AigBitVector::from_bit(crate::aig::gate::AigRef { id: literal }.into());
    assert!(
        permit()
            .capture(&f, &bad, |n| out.lowering_map.get(&n))
            .is_none()
    );
    bad.outputs[0].bit_vector = AigBitVector::from_bit(real.carry);
    assert!(
        permit()
            .capture(&f, &bad, |n| out.lowering_map.get(&n))
            .is_none()
    );
}

#[test]
fn declines_changed_roots_sums_carry_definitions_and_explicit_architecture() {
    let f = parse(SCALAR);
    for text in [
        SCALAR.replace("low, id=7", "low, id=17"),
        SCALAR.replace("inc, id=4", "inc, id=14"),
        SCALAR.replace("eq(zero, low", "ne(zero, low"),
        SCALAR.replace(
            "add(x, inc, id=4)",
            "ext_nary_add(x, c, signed=[false, false], negated=[false, true], id=4)",
        ),
        SCALAR.replace(
            "add(x, inc, id=4)",
            "ext_nary_add(x, c, signed=[false, true], negated=[false, false], id=4)",
        ),
    ] {
        let changed = parse(&text);
        let out = gatify_prepared_fn(&changed, GatifyOptions::all_opts_enabled()).unwrap();
        assert!(
            TerminalWrapPermit::at_source(&f)
                .unwrap()
                .capture(&changed, &out.gate_fn, |n| out.lowering_map.get(&n))
                .is_none()
        );
    }
    // The same ID and type no longer designate the captured carry operand.
    let original = SCALAR
        .replace("  inc:", "  flag: bits[1] = not(c, id=8)\n  inc:")
        .replace("zero_ext(c,", "zero_ext(flag,");
    let changed = parse(&original.replace("not(c,", "identity(c,"));
    let out = gatify_prepared_fn(&changed, GatifyOptions::all_opts_enabled()).unwrap();
    assert!(
        TerminalWrapPermit::at_source(&parse(&original))
            .unwrap()
            .capture(&changed, &out.gate_fn, |n| out.lowering_map.get(&n))
            .is_none()
    );
    let mut options = GatifyOptions::all_opts_enabled();
    options.adder_mapping = AdderMapping::RippleCarry;
    assert!(
        gatify_with_terminal_observations(&f, options, true)
            .unwrap()
            .1
            .wrap
            .is_none()
    );
}

#[test]
fn rejects_physically_partial_word_and_nonzero_offset_observation() {
    let partial = SCALAR
        .replace("bits[5]", "bits[6]")
        .replace("new_bit_count=5", "new_bit_count=6")
        .replace("low: bits[6]", "low: bits[5]")
        .replace("zero: bits[6]", "zero: bits[5]");
    // Opaque adder lowering resolves the whole operand even for this low
    // observation.
    assert_eq!(ordinary(&partial).unwrap().word_lsb.len(), 5);
    let slice = parse(
        "package partial\ntop fn f(x: bits[6] id=1) -> bits[5] {\n  ret low: bits[5] = bit_slice(x, start=0, width=5, id=2)\n}",
    );
    let out = gatify_prepared_fn(&slice, GatifyOptions::all_opts_enabled()).unwrap();
    let x = slice.nodes.iter().position(|n| n.text_id == 1).unwrap();
    assert!(!out.lowering_map.contains_key(&NodeRef { index: x }));
    let upper = partial.replace("start=0, width=5", "start=1, width=5");
    assert!(ordinary(&upper).is_none());
}

#[test]
fn captures_only_the_original_tracked_predicate_id() {
    let f = parse(SCALAR);
    let root = f.get_node(f.ret_node_ref.unwrap()).text_id as u32;
    for track in [false, true] {
        let mut options = GatifyOptions::all_opts_enabled();
        options.track_pir_node_ids = track;
        let (_, observations) = gatify_with_terminal_observations(&f, options, true).unwrap();
        assert_eq!(
            observations.wrap.unwrap().predicate_pir_id,
            track.then_some(root)
        );
    }
}
