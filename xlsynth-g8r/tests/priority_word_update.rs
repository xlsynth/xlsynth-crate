// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::ir_parser::Parser;

#[test]
fn overlapping_requests_preserve_priority_polarity_and_external_uses() {
    // The middle request explicitly holds the default and suppresses lower
    // priority updates. One local request takes the updated word on false.
    for update in ["xor(x, mask, id=6)", "add(x, mask, id=6)"] {
        for (hold, change) in [("x", "updated"), ("updated", "x")] {
            for output in ["result", "tuple(result, first, updated, id=11)"] {
                let (ty, result) = if output == "result" {
                    ("bits[6]", String::new())
                } else {
                    (
                        "(bits[6], bits[6], bits[6])",
                        format!("ret all: (bits[6], bits[6], bits[6]) = {output}"),
                    )
                };
                let ret = if result.is_empty() { "ret " } else { "" };
                let ir = format!("fn f(x: bits[6] id=1, mask: bits[6] id=2, p: bits[1] id=3, q: bits[1] id=4, s: bits[3] id=5) -> {ty} {{
                    updated: bits[6] = {update}
                    first: bits[6] = sel(p, cases=[{hold}, {change}], id=7)
                    last: bits[6] = sel(q, cases=[{change}, {hold}], id=8)
                    {ret}result: bits[6] = priority_sel(s, cases=[first, {hold}, last], default={hold}, id=9)
                    {result}
                }}");
                let f = Parser::new(&ir).parse_fn().unwrap();
                let mapped = gatify_prepared_fn(
                    &f,
                    GatifyOptions {
                        fold: true,
                        hash: true,
                        ..GatifyOptions::all_opts_disabled()
                    },
                )
                .unwrap();
                validate_same_fn_via_toolchain(&f, &dce(&mapped.gate_fn)).unwrap();
            }
        }
    }
}

#[test]
fn isolated_toggle_has_the_cost_of_a_single_explicit_grant() {
    use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
    let nested = r#"fn apply(data: bits[9] id=1, mask: bits[9] id=2, allow: bits[1] id=3, retry: bits[1] id=4, select: bits[3] id=5) -> bits[9] {
  toggled: bits[9] = xor(data, mask, id=6)
  low: bits[9] = sel(allow, cases=[data, toggled], id=7)
  high: bits[9] = sel(retry, cases=[toggled, data], id=8)
  ret result: bits[9] = priority_sel(select, cases=[low, data, high], default=data, id=9)
}"#;
    let shared = r#"fn apply(data: bits[9] id=1, mask: bits[9] id=2, allow: bits[1] id=3, retry: bits[1] id=4, select: bits[3] id=5) -> bits[9] {
  never: bits[1] = literal(value=0, id=6)
  on: bits[1] = not(retry, id=7)
  request: bits[1] = priority_sel(select, cases=[allow, never, on], default=never, id=8)
  zero: bits[9] = literal(value=0, id=9)
  update: bits[9] = sel(request, cases=[zero, mask], id=10)
  ret result: bits[9] = xor(data, update, id=11)
}"#;
    let mapped = |ir: &str| {
        let f = Parser::new(ir).parse_fn().unwrap();
        let result = gatify_prepared_fn(
            &f,
            GatifyOptions {
                fold: true,
                hash: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let gates = dce(&result.gate_fn);
        validate_same_fn_via_toolchain(&f, &gates).unwrap();
        get_aig_stats(&gates)
    };
    let actual = mapped(nested);
    let reference = mapped(shared);
    assert_eq!(actual.and_nodes, reference.and_nodes);
    assert!(actual.max_depth <= reference.max_depth);
}

#[test]
fn different_width_recipients_share_one_signed_priority_grant() {
    use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;

    let header = "fn choose(a: bits[9] id=1, t: bits[9] id=2, b: bits[5] id=3, c: bits[5] id=4, p: bits[1] id=5, q: bits[1] id=6, s: bits[3] id=7)";
    let nested = "at: bits[9] = xor(a, t, id=8)
        lo_a: bits[9] = sel(p, cases=[a, at], id=9)
        hi_a: bits[9] = sel(q, cases=[at, a], id=10)
        left: bits[9] = priority_sel(s, cases=[lo_a, a, hi_a], default=a, id=11)
        np: bits[1] = not(p, id=12)
        lo_b: bits[5] = sel(np, cases=[c, b], id=13)
        hi_b: bits[5] = sel(q, cases=[c, b], id=14)
        right: bits[5] = priority_sel(s, cases=[lo_b, b, hi_b], default=b, id=15)";
    let explicit = "off: bits[1] = literal(value=0, id=8)
        nq: bits[1] = not(q, id=9)
        g: bits[1] = priority_sel(s, cases=[p, off, nq], default=off, id=10)
        zero: bits[9] = literal(value=0, id=11)
        active: bits[9] = sel(g, cases=[zero, t], id=12)
        left: bits[9] = xor(a, active, id=13)
        right: bits[5] = sel(g, cases=[b, c], id=14)";
    let retained = "off: bits[1] = literal(value=0, id=8)
        nq: bits[1] = not(q, id=9)
        g: bits[1] = priority_sel(s, cases=[p, off, nq], default=off, id=10)
        at: bits[9] = xor(a, t, id=11)
        lo_a: bits[9] = sel(p, cases=[a, at], id=12)
        left: bits[9] = sel(g, cases=[a, at], id=13)
        right: bits[5] = sel(g, cases=[b, c], id=14)";
    for (ty, pins, shared) in [
        ("(bits[9], bits[5], bits[1])", "left, right, q", explicit),
        (
            "(bits[9], bits[5], bits[1], bits[9], bits[9])",
            "left, right, q, at, lo_a",
            retained,
        ),
    ] {
        let run = |body| {
            let ir =
                format!("{header} -> {ty} {{\n{body}\nret out: {ty} = tuple({pins}, id=16)\n}}");
            let f = Parser::new(&ir).parse_fn().unwrap();
            let result = gatify_prepared_fn(
                &f,
                GatifyOptions {
                    hash: true,
                    fold: true,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            let graph = dce(&result.gate_fn);
            validate_same_fn_via_toolchain(&f, &graph).unwrap();
            get_aig_stats(&graph)
        };
        let actual = run(nested);
        let wanted = run(shared);
        assert_eq!(
            (actual.and_nodes, actual.max_depth),
            (wanted.and_nodes, wanted.max_depth)
        );
    }
}

#[test]
fn a_later_recipient_can_observe_an_earlier_word() {
    let f = Parser::new(r#"fn dependent(a: bits[3] id=1, t: bits[3] id=2, b: bits[2] id=3, p: bits[1] id=4, q: bits[1] id=5, s: bits[3] id=6) -> (bits[3], bits[2], bits[3], bits[1]) {
        toggled: bits[3] = xor(a, t, id=7)
        lo: bits[3] = sel(p, cases=[a, toggled], id=8)
        hi: bits[3] = sel(q, cases=[toggled, a], id=9)
        word: bits[3] = priority_sel(s, cases=[lo, a, hi], default=a, id=10)
        sum: bits[3] = add(word, t, id=11)
        data: bits[2] = bit_slice(sum, start=0, width=2, id=12)
        left: bits[2] = sel(p, cases=[b, data], id=13)
        right: bits[2] = sel(q, cases=[data, b], id=14)
        peer: bits[2] = priority_sel(s, cases=[left, b, right], default=b, id=15)
        different: bits[1] = xor(p, q, id=16)
        ret out: (bits[3], bits[2], bits[3], bits[1]) = tuple(word, peer, sum, different, id=17)
    }"#).parse_fn().unwrap();
    let result = gatify_prepared_fn(
        &f,
        GatifyOptions {
            hash: true,
            fold: true,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    assert!(
        result.lowering_map[&f.ret_node_ref.unwrap()]
            .iter_lsb_to_msb()
            .eq(result.gate_fn.outputs[0].bit_vector.iter_lsb_to_msb())
    );
    validate_same_fn_via_toolchain(&f, &dce(&result.gate_fn)).unwrap();
}
