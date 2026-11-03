// SPDX-License-Identifier: Apache-2.0

use test_case::test_case;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{NodePayload, NodeRef};
use xlsynth_pir::ir_parser::Parser;

#[test_case("x, y, x, z, x, y, z, y", "z", 4; "two payloads with intervening zero arms")]
#[test_case("w, x, y, w, x, z, y, w", "z", 5; "three interleaved payloads")]
#[test_case("m, x, z, m, x, m, x, z", "z", 4; "word mask becomes a scalar enable")]
#[test_case("x, y, w, x, y, x, w, y", "w", 0; "nonzero default")]
#[test_case("w, x, n, y, x, n, z, y", "z", 7; "overlapping data leaves")]
#[test_case("z, y, t, x, t, z, x, x", "z", 7; "nested mux can return zero")]
fn repeated_words_preserve_priority(arms: &str, default: &str, gates_per_bit: usize) {
    let widths = [8, 32];
    let costs = widths.map(|width| {
    let text = format!(
        r#"package priority_words

top fn choose(s: bits[8] id=1, x: bits[{width}] id=2, y: bits[{width}] id=3, w: bits[{width}] id=4, e: bits[1] id=5) -> bits[{width}] {{
  z: bits[{width}] = literal(value=0, id=6)
  mask: bits[{width}] = sign_ext(e, new_bit_count={width}, id=7)
  m: bits[{width}] = and(mask, y, id=8)
  n: bits[{width}] = sel(e, cases=[w, x], id=9)
  t: bits[{width}] = sel(e, cases=[z, x], id=10)
  ret selected: bits[{width}] = priority_sel(s, cases=[{arms}], default={default}, id=11)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &gates).unwrap();
    get_aig_stats(&gates).and_nodes
    });
    if gates_per_bit > 0 {
        // Word width changes only the data path. Measuring the incremental
        // cost removes the scalar decoder and works across different tag
        // shapes.
        let extra_bits = widths[1] - widths[0];
        assert!(
            costs[1] <= costs[0] + extra_bits * gates_per_bit,
            "{costs:?}"
        );
    }
}

#[test]
fn grouped_priority_preserves_aggregate_layout() {
    let text = r#"package aggregate_priority

top fn choose(s: bits[5] id=1, a: (bits[2], bits[7]) id=2, b: (bits[2], bits[7]) id=3) -> (bits[2], bits[7]) {
  z: (bits[2], bits[7]) = literal(value=(0, 0), id=4)
  ret value: (bits[2], bits[7]) = priority_sel(s, cases=[b, z, a, b, a], default=z, id=5)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gate_fn = gatify(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &gate_fn).unwrap();
}

#[test_case(&["a", "a"]; "single data word")]
#[test_case(&["a", "p", "q", "a"]; "three data words")]
fn priority_accepts_token_in_aggregate(cases: &[&str]) {
    let width = cases.len();
    let arms = cases.join(", ");
    let text = format!(
        r#"package token_priority

top fn choose(s: bits[{width}] id=1, a: (token, bits[4]) id=2) -> (token, bits[4]) {{
  z: (token, bits[4]) = literal(value=(token, 0), id=3)
  p: (token, bits[4]) = literal(value=(token, 5), id=4)
  q: (token, bits[4]) = literal(value=(token, 6), id=5)
  ret value: (token, bits[4]) = priority_sel(s, cases=[{arms}], default=z, id=6)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify(f, GatifyOptions::all_opts_disabled())
        .unwrap()
        .gate_fn;
    for selector in 0u64..1 << width {
        let inputs = [
            IrBits::make_ubits(width, selector).unwrap(),
            IrBits::make_ubits(4, 13).unwrap(),
        ];
        let value = match cases.get(selector.trailing_zeros() as usize) {
            None => 0,
            Some(&"a") => 13,
            Some(&"p") => 5,
            Some(&"q") => 6,
            other => panic!("unexpected arm {other:?}"),
        };
        let expected = IrBits::make_ubits(4, value).unwrap();
        assert_eq!(
            gate_sim::eval(&gates, &inputs, Collect::None).outputs,
            vec![expected]
        );
    }
}

#[test]
fn sparse_literal_groups_share_priority_grants() {
    let mut literals = String::new();
    let mut arms = vec!["x".to_string()];
    for bit in 0..63 {
        let value = 1u64 << bit;
        let id = bit + 4;
        literals.push_str(&format!(
            "  k{bit}: bits[64] = literal(value={value}, id={id})\n"
        ));
        arms.push(format!("k{bit}"));
    }
    arms.push("x".to_string());
    let arms = arms.join(", ");
    let text = format!(
        r#"package sparse_priority

top fn choose(s: bits[65] id=1, x: bits[64] id=2) -> bits[64] {{
  z: bits[64] = literal(value=0, id=3)
{literals}  ret value: bits[64] = priority_sel(s, cases=[{arms}], default=z, id=67)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let options = GatifyOptions {
        fold: true,
        hash: true,
        ..GatifyOptions::all_opts_disabled()
    };
    let masking = gatify_prepared_fn(f, options.clone()).unwrap().gate_fn;
    let grouped = gatify(f, options).unwrap().gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &grouped).unwrap();
    let masking_cost = get_aig_stats(&masking).and_nodes;
    let grouped_cost = get_aig_stats(&grouped).and_nodes;
    assert!(
        grouped_cost < masking_cost,
        "grouped {grouped_cost}, per-case masking {masking_cost}"
    );
}

#[test]
fn tag_decisions_on_a_wide_selector() {
    let width = 4096;
    let slices: String = (0..width)
        .map(|i| {
            format!(
                "  b{i}: bits[1] = bit_slice(input, start={i}, width=1, id={})\n",
                i + 7
            )
        })
        .collect();
    let bits = (0..width)
        .rev()
        .map(|i| format!("b{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arms = (0..width)
        .map(|i| if i % 2 == 0 { "x" } else { "y" })
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        r#"package wide_priority

top fn choose(input: bits[{width}] id=1, x: bits[4] id=2, y: bits[4] id=3) -> bits[4] {{
{slices}  s: bits[{width}] = concat({bits}, id=6)
  z: bits[4] = literal(value=0, id=4)
  ret value: bits[4] = priority_sel(s, cases=[{arms}], default=z, id=5)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let mut sim = gate_sim::PreparedGateSim::new(&gates);
    for (slots, value) in [
        (&[][..], 0),
        (&[4095], 9),
        (&[4094, 4095], 6),
        (&[1, 4094], 9),
        (&[0, 4095], 6),
    ] {
        let mut selector = vec![false; width];
        for &slot in slots {
            selector[slot] = true;
        }
        let inputs = [
            IrBits::from_lsb_is_0(&selector),
            IrBits::make_ubits(4, 6).unwrap(),
            IrBits::make_ubits(4, 9).unwrap(),
        ];
        assert_eq!(
            sim.eval(&inputs, Collect::None).outputs,
            vec![IrBits::make_ubits(4, value).unwrap()]
        );
    }
}

#[test]
fn grouping_leaves_ids_for_mask_low() {
    let selector_id = usize::MAX - 3;
    let text = format!(
        r#"package sparse_ids

top fn choose(s: bits[2] id={selector_id}, x: bits[4] id=2) -> (bits[4], bits[4]) {{
  z: bits[4] = literal(value=0, id=3)
  value: bits[4] = priority_sel(s, cases=[x, x], default=z, id=4)
  ones: bits[4] = literal(value=15, id=5)
  high: bits[4] = shll(ones, s, id=6)
  ret pair: (bits[4], bits[4]) = tuple(value, high, id=7)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let prepared = prep_for_gatify(
        package.get_top_fn().unwrap(),
        None,
        PrepForGatifyOptions {
            enable_rewrite_mask_low: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    assert!(prepared.nodes.iter().any(
        |n| n.name.as_deref() == Some("value") && matches!(n.payload, NodePayload::Sel { .. })
    ));
    assert!(
        prepared
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::ExtMaskLow { .. }))
    );
    let gates = gatify_prepared_fn(&prepared, GatifyOptions::all_opts_disabled())
        .unwrap()
        .gate_fn;
    for (s, expected) in [(0, 0x0f), (1, 0xae), (2, 0xac), (3, 0xa8)] {
        let inputs = [
            IrBits::make_ubits(2, s).unwrap(),
            IrBits::make_ubits(4, 10).unwrap(),
        ];
        assert_eq!(
            gate_sim::eval(&gates, &inputs, Collect::None).outputs,
            vec![IrBits::make_ubits(8, expected).unwrap()]
        );
    }
}

#[test]
fn priority_with_explicit_opcode_name() {
    let text = r#"package named_priority

top fn choose(s: bits[2] id=1, x: bits[4] id=2) -> bits[4] {
  z: bits[4] = literal(value=0, id=3)
  ret priority_sel.4: bits[4] = priority_sel(s, cases=[x, x], default=z, id=4)
}
"#;
    let mut package = Parser::new(text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn_mut().unwrap();
    let ret = f.ret_node_ref.unwrap();
    // The parser normalizes default names, but PIR callers may supply one.
    f.get_node_mut(ret).name = Some("priority_sel.4".to_string());
    gatify(f, GatifyOptions::all_opts_disabled()).unwrap();
}

#[test]
fn dead_priority_with_unrepresentable_aggregate_width() {
    let width = usize::MAX;
    let text = format!(
        r#"package dead_priority

top fn choose(s: bits[1] id=1) -> bits[1] {{
  wide: bits[{width}] = zero_ext(s, new_bit_count={width}, id=2)
  t: (bits[{width}], bits[1]) = tuple(wide, s, id=3)
  dead: (bits[{width}], bits[1]) = priority_sel(s, cases=[t], default=t, id=4)
  both: bits[2] = concat(s, s, id=5)
  dead_pair: (bits[{width}], bits[1]) = priority_sel(both, cases=[t, t], default=t, id=6)
  ret s: bits[1] = param(name=s, id=1)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap();
}

#[test_case(0; "maximum selector id")]
#[test_case(1; "one id above selector")]
#[test_case(3; "three ids above selector")]
fn priority_with_sparse_text_ids(remaining_ids: usize) {
    let selector_id = usize::MAX - remaining_ids;
    let text = format!(
        r#"package sparse_ids

top fn choose(s: bits[2] id={selector_id}, x: bits[4] id=2) -> bits[4] {{
  z: bits[4] = literal(value=0, id=3)
  ret value: bits[4] = priority_sel(s, cases=[x, x], default=z, id=4)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify(f, GatifyOptions::all_opts_disabled())
        .unwrap()
        .gate_fn;
    for selector in 0..4 {
        let inputs = [
            IrBits::make_ubits(2, selector).unwrap(),
            IrBits::make_ubits(4, 15).unwrap(),
        ];
        let expected = IrBits::make_ubits(4, if selector == 0 { 0 } else { 15 }).unwrap();
        assert_eq!(
            gate_sim::eval(&gates, &inputs, Collect::None).outputs,
            vec![expected]
        );
    }
}

#[test_case("sel(en, cases=[zero, word], id=9)", false; "word enabled on true")]
#[test_case("sel(en, cases=[word, zero], id=9)", false; "word enabled on false")]
#[test_case("and(mask, word, id=9)", false; "replicated enable")]
#[test_case("and(word, mask, id=9)", true; "another consumer needs the unmasked word")]
fn scalar_grants_preserve_word_observers(mask_op: &str, shared: bool) {
    let (result_type, result) = if shared {
        ("(bits[12], bits[12])", "tuple(word, guarded, id=10)")
    } else {
        ("bits[12]", "identity(guarded, id=10)")
    };
    let text = format!(
        r#"package masked_priority
top fn f(s: bits[5] id=1, en: bits[1] id=2, a: bits[12] id=3, b: bits[12] id=4, c: bits[12] id=5) -> {result_type} {{
  zero: bits[12] = literal(value=0, id=6)
  mask: bits[12] = sign_ext(en, new_bit_count=12, id=7)
  word: bits[12] = priority_sel(s, cases=[b, a, c, b, a], default=zero, id=8)
  guarded: bits[12] = {mask_op}
  ret result: {result_type} = {result}
}}"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_disabled());
    assert_eq!(
        prepared
            .nodes
            .iter()
            .any(|n| n.text_id == 9 && matches!(n.payload, NodePayload::OneHotSel { .. })),
        !shared
    );
    let word = NodeRef {
        index: f.nodes.iter().position(|n| n.text_id == 8).unwrap(),
    };
    for tracking in [false, true] {
        let mapped = gatify(
            f,
            GatifyOptions {
                fold: true,
                hash: true,
                track_pir_node_ids: tracking,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        check_equivalence::validate_same_fn_via_toolchain(f, &mapped.gate_fn).unwrap();
        assert_eq!(mapped.lowering_map.contains_key(&word), shared || tracking);
        if let Some(bits) = mapped.lowering_map.get(&word) {
            // Even when the output mask is closed, the source word must
            // still report the winning payload through the lowering map.
            let inputs = [(5, 7), (1, 0), (12, 1), (12, 2), (12, 4)]
                .map(|(width, value)| IrBits::make_ubits(width, value).unwrap());
            let values = gate_sim::eval(&mapped.gate_fn, &inputs, Collect::AllWithInputs)
                .all_values
                .unwrap();
            for (i, bit) in bits.iter_lsb_to_msb().enumerate() {
                assert_eq!(values[bit.node.id] ^ bit.negated, i == 1);
            }
        }
    }
}

#[test]
fn interleaved_word_grants_share_prefixes() {
    // Repeated independent groups spread throughout a full selector word.
    // Even moving a wide output mask cannot remove the serial alternations.
    let arms = (0..64)
        .map(|i| ["a", "b", "c"][i % 3])
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        r#"package interleaved_priority
top fn f(s: bits[64] id=1, en: bits[1] id=2, a: bits[256] id=3, b: bits[256] id=4, c: bits[256] id=5) -> bits[256] {{
  z: bits[256] = literal(value=0, id=6)
  word: bits[256] = priority_sel(s, cases=[{arms}], default=z, id=7)
  ret masked: bits[256] = sel(en, cases=[z, word], id=8)
}}"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_disabled());
    assert_eq!(
        prepared
            .nodes
            .iter()
            .filter(|n| matches!(n.payload, NodePayload::OneHot { .. }))
            .count(),
        1
    );
}

#[test]
fn padding_does_not_pay_for_scalar_grants() {
    let params = (0..8)
        .map(|i| format!("p{i}: bits[2] id={}", i + 3))
        .collect::<Vec<_>>()
        .join(", ");
    let words = (0..8)
        .map(|i| {
            format!(
                "  w{i}: bits[80] = zero_ext(p{i}, new_bit_count=80, id={})\n",
                i + 11
            )
        })
        .collect::<String>();
    let arms = (0..16)
        .map(|i| format!("w{}", i / 2))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        r#"package padded_priority
top fn f(s: bits[16] id=1, en: bits[1] id=2, {params}) -> bits[80] {{
{words}  z: bits[80] = literal(value=0, id=19)
  word: bits[80] = priority_sel(s, cases=[{arms}], default=z, id=20)
  ret guarded: bits[80] = sel(en, cases=[z, word], id=21)
}}"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_disabled());
    assert!(
        prepared
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::OneHot { .. }))
    );
}

#[test]
fn repeated_fields_share_a_data_mask() {
    let words = ["a", "b", "c", "d"];
    let arms = words.repeat(3).join(", ");
    let text = format!(
        r#"package field_grants
top fn f(s: bits[12] id=1, x: bits[20] id=2, y: bits[20] id=3) -> bits[20][3] {{
  z: bits[20] = literal(value=0, id=4)
  a: bits[20][3] = array(x, y, z, id=5)
  b: bits[20][3] = array(x, x, z, id=6)
  c: bits[20][3] = array(x, z, z, id=7)
  d: bits[20][3] = array(y, x, z, id=8)
  zero: bits[20][3] = literal(value=[0,0,0], id=9)
  ret word: bits[20][3] = priority_sel(s, cases=[{arms}], default=zero, id=10)
}}"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_disabled());
    let NodePayload::Array(fields) = &prepared.get_node(prepared.ret_node_ref.unwrap()).payload
    else {
        panic!("expected field routing")
    };
    assert!(
        matches!(&prepared.get_node(fields[0]).payload, NodePayload::OneHotSel{cases,..} if cases.len()==2)
    );
    assert!(matches!(
        &prepared.get_node(fields[2]).payload,
        NodePayload::Literal(_)
    ));
    let gates = gatify_prepared_fn(&prepared, GatifyOptions::all_opts_disabled())
        .unwrap()
        .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &gates).unwrap();
}

/// Compare direct mapping with the shared decoder retained for tracked words.
fn assert_shared_grant_cost(text: &str) {
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let costs = [false, true].map(|track_pir_node_ids| {
        let mapped = gatify(
            f,
            GatifyOptions {
                fold: true,
                hash: true,
                track_pir_node_ids,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        check_equivalence::validate_same_fn_via_toolchain(f, &mapped.gate_fn).unwrap();
        let stats = get_aig_stats(&mapped.gate_fn);
        (stats.and_nodes, stats.max_depth)
    });
    assert_eq!(costs[0], costs[1]);
}

#[test]
fn direct_mapping_charges_serial_grant_depth() {
    let cases = ["z"; 60]
        .into_iter()
        .chain(["a", "b", "c", "a"])
        .collect::<Vec<_>>()
        .join(", ");
    assert_shared_grant_cost(&format!(
        r#"package p

top fn f(s: bits[64] id=1, en: bits[1] id=2, a: bits[123] id=3, b: bits[123] id=4, c: bits[123] id=5) -> bits[1] {{
  z: bits[123] = literal(value=0, id=6)
  word: bits[123] = priority_sel(s, cases=[{cases}], default=z, id=7)
  guard: bits[123] = sel(en, cases=[z, word], id=8)
  ret low: bits[1] = bit_slice(guard, start=0, width=1, id=9)
}}"#
    ));
}

#[test]
fn grant_enable_cost_survives_padding_discount() {
    assert_shared_grant_cost(
        r#"package p

top fn f(s: bits[5] id=1, en: bits[1] id=2, a: bits[1] id=3, b: bits[1] id=4, c: bits[1] id=5, d: bits[1] id=6) -> bits[64] {
  z: bits[64] = literal(value=0, id=7)
  wa: bits[64] = zero_ext(a, new_bit_count=64, id=8)
  wb: bits[64] = zero_ext(b, new_bit_count=64, id=9)
  wc: bits[64] = zero_ext(c, new_bit_count=64, id=10)
  wd: bits[64] = zero_ext(d, new_bit_count=64, id=11)
  word: bits[64] = priority_sel(s, cases=[wa, wb, wc, wa, wd], default=z, id=12)
  ret guarded: bits[64] = sel(en, cases=[z, word], id=13)
}"#,
    );
}

#[test_case(0)]
#[test_case(54)]
fn trailing_defaults_do_not_pay_for_grants(padding: usize) {
    let cases = ["a", "b", "c", "d", "e", "d", "a", "f", "d", "c"]
        .into_iter()
        .chain(std::iter::repeat_n("z", padding))
        .collect::<Vec<_>>()
        .join(", ");
    let slots = 10 + padding;
    assert_shared_grant_cost(&format!(
        r#"package p

top fn word(s: bits[{slots}] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4, d: bits[8] id=5, e: bits[8] id=6, f: bits[8] id=7) -> bits[8] {{
  z: bits[8] = literal(value=0, id=8)
  ret word: bits[8] = priority_sel(s, cases=[{cases}], default=z, id=9)
}}"#
    ));
}

#[test]
fn singleton_fields_keep_the_masks_used_by_pair_fields() {
    assert_shared_grant_cost(
        r#"package p

top fn f(s: bits[16] id=1, x: bits[8] id=2) -> bits[8][6] {
  z: bits[8] = literal(value=0, id=3)
  zero: bits[8][6] = literal(value=[0,0,0,0,0,0], id=4)
  a: bits[8][6] = array(x, z, z, x, x, z, id=5)
  b: bits[8][6] = array(z, x, z, x, z, x, id=6)
  c: bits[8][6] = array(z, z, x, z, x, x, id=7)
  ret word: bits[8][6] = priority_sel(s, cases=[a,b,c,a,b,c,a,b,c,a,b,c,a,b,c,a], default=zero, id=8)
}"#,
    );
}
