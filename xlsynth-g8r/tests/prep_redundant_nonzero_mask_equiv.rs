// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::AigBitVector;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{
    GatifyOptions, IrRegion, gatify, gatify_node_as_fn, gatify_prepared_fn, gatify_region,
};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, process_ir_text_with_gatefn};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, Binop, NodePayload, Unop};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::known_bits::analyze_fn;
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn word_and_keeps_both_inputs_in_standalone_and_region_lowering() {
    let f = parse(
        r#"package independent
top fn f(x: bits[8] id=1) -> bits[8] {
 nonzero: bits[1] = or_reduce(x, id=2)
 mask: bits[8] = sign_ext(nonzero, new_bit_count=8, id=3)
 ret word: bits[8] = and(x, mask, id=4)
}"#,
    );
    let word = f.ret_node_ref.unwrap();
    let inputs = xlsynth_pir::ir_utils::operands(&f.get_node(word).payload);
    for gates in [
        gatify_node_as_fn(&f, word, &GatifyOptions::all_opts_enabled()).unwrap(),
        gatify_region(
            IrRegion {
                function: &f,
                inputs: &inputs,
                nodes: &[word],
                outputs: &[word],
            },
            GateBuilderOptions::opt(),
        )
        .unwrap(),
    ] {
        assert_eq!(gates.inputs.len(), 2);
        assert_eq!(get_aig_stats(&gates).and_nodes, 8);
        let result = gate_sim::eval(
            &gates,
            &[
                IrBits::make_ubits(8, 0x3c).unwrap(),
                IrBits::make_ubits(8, 0x5a).unwrap(),
            ],
            gate_sim::Collect::None,
        );
        assert_eq!(result.outputs, vec![IrBits::make_ubits(8, 0x18).unwrap()]);
    }
}

#[test]
fn standalone_unused_product_accepts_any_value_for_a_literal_operand() {
    let f = parse(
        r#"package independent
top fn f(x: bits[8] id=1) -> bits[8] {
 zero: bits[8] = literal(value=0, id=2)
 product: bits[8] = umul(x, zero, id=3)
 ret result: bits[8] = identity(x, id=4)
}"#,
    );
    let product = ir::NodeRef {
        index: f.nodes.iter().position(|node| node.text_id == 3).unwrap(),
    };
    let ranges = IrRangeInfo::from_known_bits(&analyze_fn(&f).unwrap());
    let gates = gatify_node_as_fn(
        &f,
        product,
        &GatifyOptions {
            range_info: Some(ranges.into()),
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let result = gate_sim::eval(
        &gates,
        &[
            IrBits::make_ubits(8, 5).unwrap(),
            IrBits::make_ubits(8, 7).unwrap(),
        ],
        gate_sim::Collect::None,
    );
    assert_eq!(result.outputs, vec![IrBits::make_ubits(8, 35).unwrap()]);
}

#[test]
fn pruning_an_inner_mask_preserves_the_outer_parity_path() {
    let names: Vec<_> = (0..31).map(|i| format!("a{i}")).collect();
    let params = names
        .iter()
        .map(|name| format!("{name}: bits[1]"))
        .collect::<Vec<_>>()
        .join(", ");
    let args = names.join(", ");
    let source = parse(&format!(
        r#"package parity_path
top fn f(x: bits[256], {params}) -> (bits[1], bits[1]) {{
 parity: bits[1] = xor_reduce(x, id=33)
 nonzero: bits[1] = or_reduce(x, id=34)
 inner: bits[1] = and(nonzero, {args}, id=35)
 word: bits[1] = and(parity, inner, id=36)
 ret pair: (bits[1], bits[1]) = tuple(nonzero, word, id=37)
}}"#
    ));
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    let stats = |f| {
        let gates = gatify_prepared_fn(f, GatifyOptions::all_opts_disabled()).unwrap();
        get_aig_stats(&gates.gate_fn)
    };
    let before = stats(&source);
    let after = stats(&prepared);
    assert_eq!(after.and_nodes + 1, before.and_nodes);
    assert_eq!(after.max_depth, before.max_depth);
    assert_equivalent(&source, &prepared);
}

#[test]
fn combined_reduction_keeps_the_concat_out_of_original_facts() {
    for (xw, yw) in [(0, 1), (8, 8)] {
        let f = parse(&format!(
            r#"package combined
top fn f(x: bits[{xw}] id=1, y: bits[{yw}] id=2) -> (bits[1], bits[1]) {{
 r: bits[1] = or_reduce(x, id=3)
 t: bits[1] = or_reduce(y, id=4)
 zero: bits[1] = literal(value=0, id=5)
 masked: bits[1] = and(r, zero, id=6)
 either: bits[1] = or(r, t, id=7)
 ret pair: (bits[1], bits[1]) = tuple(masked, either, id=8)
}}"#
        ));
        let ranges = IrRangeInfo::from_known_bits(&analyze_fn(&f).unwrap());
        let lowered = gatify(
            &f,
            GatifyOptions {
                range_info: Some(ranges.into()),
                ..GatifyOptions::all_opts_enabled()
            },
        )
        .unwrap();
        let result = gate_sim::eval(
            &lowered.gate_fn,
            &[
                IrBits::make_ubits(xw, 0).unwrap(),
                IrBits::make_ubits(yw, 1).unwrap(),
            ],
            gate_sim::Collect::None,
        );
        assert_eq!(result.outputs, [IrBits::make_ubits(2, 1).unwrap()]);
        let r = node_by_id(&f, 3).unwrap();
        assert!(!lowered.lowering_map.contains_key(&r));
    }
}

#[test]
fn projected_sign_mask_omits_all_three_changed_predicate_values() {
    let f = parse(
        r#"package projected
top fn f(x: bits[4] id=1) -> bits[1] {
 zero: bits[4] = literal(value=0, id=2)
 le: bits[1] = sle(x, zero, id=3)
 positive: bits[1] = not(le, id=4)
 mask: bits[3] = sign_ext(positive, new_bit_count=3, id=5)
 low: bits[3] = bit_slice(x, start=0, width=3, id=6)
 nz: bits[1] = or_reduce(x, id=7)
 nz_mask: bits[3] = sign_ext(nz, new_bit_count=3, id=8)
 gated: bits[3] = and(low, mask, nz_mask, id=9)
 ret result: bits[1] = bit_slice(gated, start=2, width=1, id=10)
}"#,
    );
    let lowered = gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
    for id in 3..=5 {
        let n = node_by_id(&f, id).unwrap();
        assert!(!lowered.lowering_map.contains_key(&n), "stale ID {id}");
    }
    for x in 0..16 {
        let result = gate_sim::eval(
            &lowered.gate_fn,
            &[IrBits::make_ubits(4, x).unwrap()],
            gate_sim::Collect::None,
        );
        assert_eq!(result.outputs, [IrBits::bool((4..8).contains(&x))]);
    }
}

#[test]
fn pruning_inside_a_nary_and_preserves_native_splits_and_sharing() {
    let f = parse(
        r#"package shared_split
top fn f(x: bits[256] id=1, a: bits[1] id=2, b: bits[1] id=3, c: bits[1] id=4, d: bits[1] id=5) -> (bits[1], bits[1], bits[1]) {
 p: bits[1] = xor_reduce(x, id=6)
 m: bits[1] = or_reduce(x, id=7)
 abc: bits[1] = and(a, b, c, id=8)
 masked: bits[1] = and(a, b, c, p, m, d, id=9)
 ret out: (bits[1], bits[1], bits[1]) = tuple(m, abc, masked, id=10)
}"#,
    );
    let before = get_aig_stats(
        &gatify_prepared_fn(&f, GatifyOptions::all_opts_disabled())
            .unwrap()
            .gate_fn,
    );
    let after = get_aig_stats(
        &gatify(&f, GatifyOptions::all_opts_disabled())
            .unwrap()
            .gate_fn,
    );
    assert_eq!(after.and_nodes + 1, before.and_nodes);
    assert_eq!(after.max_depth, before.max_depth);
}

#[test]
fn pruning_preserves_a_priority_threshold_and_selected_bit_fold() {
    let text = r#"package threshold
top fn f(a: bits[8], b: bits[8], control: bits[1], x: bits[8]) -> bits[1] {
 sum: bits[8] = add(a, b, id=5)
 negative: bits[8] = neg(sum, id=6)
 selected: bits[8] = sel(control, cases=[sum, negative], id=7)
 high: bits[1] = bit_slice(selected, start=4, width=1, id=8)
 reversed: bits[8] = reverse(selected, id=9)
 hot: bits[9] = one_hot(reversed, lsb_prio=true, id=10)
 clz: bits[4] = encode(hot, id=11)
 count_high: bits[2] = bit_slice(clz, start=2, width=2, id=12)
 threshold: bits[1] = or_reduce(count_high, id=13)
 parity: bits[1] = xor_reduce(x, id=14)
 mask: bits[1] = or_reduce(x, id=15)
 ret result: bits[1] = and(OPERANDS, id=16)
}"#;
    // The four-operand shape can swap the two surviving halves. Removing the
    // mask in the five-operand shape would leave an uneven one/three split.
    for operands in [
        "parity, high, mask, threshold",
        "mask, high, parity, control, threshold",
    ] {
        let f = parse(&text.replace("OPERANDS", operands));
        for options in [
            GatifyOptions::all_opts_enabled(),
            GatifyOptions::all_opts_disabled(),
        ] {
            for gates in [
                gatify_prepared_fn(&f, options.clone()).unwrap().gate_fn,
                gatify(&f, options).unwrap().gate_fn,
            ] {
                let stats = get_aig_stats(&gates);
                assert_eq!((stats.and_nodes, stats.max_depth), (0, 0));
                assert_eq!(
                    gates.outputs[0].bit_vector.get_lsb(0),
                    AigBitVector::zeros(1).get_lsb(0)
                );
            }
        }
    }
}

#[test]
fn pruning_and_mask_low_respect_the_last_available_text_id() {
    let text = r#"package last_id
top fn f(x: bits[8], a: bits[1], b: bits[1], k: bits[4]) -> (bits[1], bits[8]) {
 p: bits[1] = xor_reduce(x, id=5)
 m: bits[1] = or_reduce(x, id=6)
 MASKED
 ones: bits[8] = literal(value=255, id=8)
 SHIFTED
 ret pair: (bits[1], bits[8]) = tuple(masked, shifted, id=10)
}"#;
    // The flat AND leaves the final ID for mask-low. A changed interior
    // consumes it first, in which case the ordinary shift must still lower.
    let flat = "masked: bits[1] = and(p, a, m, b, id=7)";
    let nested = "inner: bits[1] = and(m, a, b, id=11)\n masked: bits[1] = and(p, inner, id=7)";
    let shift = "shifted: bits[8] = shll(ones, k, id=9)";
    let two_shifts =
        "right: bits[8] = shrl(ones, k, id=12)\n shifted: bits[8] = shll(right, k, id=9)";
    for (masked, shifted, expect_mask_low) in [
        (flat, shift, true),
        (nested, shift, false),
        (nested, two_shifts, false),
    ] {
        let mut f = parse(&text.replace("MASKED", masked).replace("SHIFTED", shifted));
        let ret = f.ret_node_ref.unwrap();
        f.get_node_mut(ret).text_id = usize::MAX - 1;
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        let last = node_by_id(&prepared, usize::MAX).unwrap();
        assert_eq!(
            matches!(
                prepared.get_node(last).payload,
                NodePayload::ExtMaskLow { .. }
            ),
            expect_mask_low
        );
        let gates = gatify(&f, GatifyOptions::all_opts_enabled())
            .unwrap()
            .gate_fn;
        for k in 0..16 {
            let value = gate_sim::eval(
                &gates,
                &[
                    IrBits::make_ubits(8, 1).unwrap(),
                    IrBits::bool(true),
                    IrBits::bool(true),
                    IrBits::make_ubits(4, k).unwrap(),
                ],
                gate_sim::Collect::None,
            );
            assert_eq!(
                value.outputs,
                [IrBits::make_ubits(9, 256 | ((255 << k) & 255)).unwrap()]
            );
        }
    }
}

#[test]
fn tracked_prepared_reductions_accept_the_largest_provenance_id() {
    let mut f = parse(
        r#"package reductions
top fn f(x: bits[8], y: bits[8]) -> bits[1] {
 lhs: bits[1] = or_reduce(x, id=3)
 rhs: bits[1] = or_reduce(y, id=4)
 ret either: bits[1] = or(lhs, rhs, id=5)
}"#,
    );
    let ret = f.ret_node_ref.unwrap();
    let options = GatifyOptions {
        track_pir_node_ids: true,
        ..GatifyOptions::all_opts_enabled()
    };
    for last in [u32::MAX - 1, u32::MAX] {
        f.get_node_mut(ret).text_id = last as usize;
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        let combined = matches!(
            prepared.get_node(prepared.ret_node_ref.unwrap()).payload,
            NodePayload::Unop(Unop::OrReduce, _)
        );
        assert_eq!(combined, last < u32::MAX);
        for gates in [
            gatify(&f, options.clone()).unwrap().gate_fn,
            gatify_prepared_fn(&prepared, options.clone())
                .unwrap()
                .gate_fn,
        ] {
            let root = gates.outputs[0].bit_vector.get_lsb(0).node;
            assert_eq!(gates.gates[root.id].get_pir_node_ids(), &[last]);
        }
    }
}

const INTERIOR_AND: &str = r#"package interior
top fn f(x: bits[8] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {
 p: bits[1] = xor_reduce(x, id=4)
 m: bits[1] = or_reduce(x, id=5)
 inner: bits[1] = and(m, a, b, id=6)
 ret result: bits[1] = and(p, inner, id=7)
}"#;

#[test]
fn rebuilt_interior_and_has_its_source_operation_as_provenance() {
    let f = parse(INTERIOR_AND);
    let lowered = gatify(
        &f,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let gates = &lowered.gate_fn;
    let a = gates.inputs[1].bit_vector.get_lsb(0).node;
    let b = gates.inputs[2].bit_vector.get_lsb(0).node;
    let output = gates.outputs[0].bit_vector.get_lsb(0).node;
    let inner = gates.gates[output.id]
        .args()
        .map(|r| &gates.gates[r.id])
        .find(|n| n.args().any(|r| r == a) && n.args().any(|r| r == b))
        .unwrap();
    assert_eq!(inner.get_pir_node_ids(), &[6]);
    let original_inner = node_by_id(&f, 6).unwrap();
    assert!(!lowered.lowering_map.contains_key(&original_inner));
}

#[test]
fn interior_pruning_handles_exhausted_ids() {
    for last in [usize::MAX, usize::MAX - 1] {
        let mut f = parse(INTERIOR_AND);
        let ret = f.ret_node_ref.unwrap();
        f.get_node_mut(ret).text_id = last;
        xlsynth_pir::ir_verify::verify_function(&f).unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        xlsynth_pir::ir_verify::verify_function(&prepared).unwrap();
        assert_eq!(
            prepared.get_node(prepared.ret_node_ref.unwrap()).text_id,
            last
        );
        // With one spare ID, the mask and its original AND can be removed.
        assert_eq!(node_by_id(&prepared, 6).is_some(), last == usize::MAX);
    }
}

#[test]
fn mask_root_identity_preserves_user_names_and_drops_opcode_names() {
    let mut f = parse(
        r#"package names
top fn f(x: bits[8] id=1) -> bits[8] {
 nz: bits[1] = or_reduce(x, id=2)
 mask: bits[8] = sign_ext(nz, new_bit_count=8, id=3)
 ret result: bits[8] = and(x, mask, id=4)
}"#,
    );
    let ret = f.ret_node_ref.unwrap();
    for (name, expected) in [("and.4", None), ("result", Some("result"))] {
        f.get_node_mut(ret).name = Some(name.into());
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        let root = prepared.get_node(prepared.ret_node_ref.unwrap());
        assert_eq!(
            root.payload,
            NodePayload::Unop(Unop::Identity, prepared.params[0])
        );
        assert_eq!(root.name.as_deref(), expected);
        gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
    }
}

#[test]
fn standalone_unused_empty_reductions_return_their_identities() {
    let text = r#"package independent
top fn f(e: bits[0] id=1, live: bits[1] id=2) -> bits[1] {
 unused_or: bits[1] = or_reduce(e, id=3)
 unused_and: bits[1] = and_reduce(e, id=4)
 unused_xor: bits[1] = xor_reduce(e, id=5)
 ret result: bits[1] = identity(live, id=6)
}"#;
    let options = CanonicalG8rOptions {
        enable_formal_array_alias_analysis: false,
        fraig: false,
        cut_db_rewrite: false,
        ..Default::default()
    }
    .to_process_ir_path_options(
        None, /* quiet= */ true, /* emit_netlist= */ false,
        /* emit_independent_op_stats= */ true, None,
    );
    let (_, stats) = process_ir_text_with_gatefn(text, &options).unwrap();
    let stats = stats.independent_op_stats.unwrap();
    let f = parse(text);
    for (id, identity) in [(3, false), (4, true), (5, false)] {
        let entry = stats
            .per_node
            .iter()
            .find(|entry| entry.text_id == id)
            .unwrap();
        assert_eq!(entry.deepest_path, 0);
        let unused = node_by_id(&f, id).unwrap();
        let gates = gatify_node_as_fn(
            &f,
            unused,
            &GatifyOptions {
                fold: false,
                ..GatifyOptions::all_opts_enabled()
            },
        )
        .unwrap();
        let result = gate_sim::eval(
            &gates,
            &[IrBits::make_ubits(0, 0).unwrap()],
            gate_sim::Collect::None,
        );
        assert_eq!(result.outputs, [IrBits::bool(identity)], "ID {id}");
    }
}

#[test]
fn zero_preserving_selections_discard_redundant_masks() {
    for width in [3, 9, 65, 129] {
        for predicate in 0..3 {
            for association in 0..4 {
                let source = parse(&selection_case(width, predicate, association));
                let prepared =
                    prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
                assert!(
                    prepared.nodes.iter().all(|node| !matches!(
                        node.payload,
                        NodePayload::Unop(Unop::OrReduce, _) | NodePayload::Binop(Binop::Ne, _, _)
                    )),
                    "redundant predicate survived: width={width}, predicate={predicate}, association={association}"
                );
                assert_equivalent(&source, &prepared);
            }
        }
    }
}

#[test]
fn removing_redundant_masks_reduces_native_and_gates() {
    let source = parse(&selection_case(9, 0, 0));
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    let original = gatify_prepared_fn(&source, GatifyOptions::all_opts_enabled())
        .expect("original gates")
        .gate_fn;
    let improved = gatify_prepared_fn(&prepared, GatifyOptions::all_opts_enabled())
        .expect("prepared gates")
        .gate_fn;
    let before = get_aig_stats(&original);
    let after = get_aig_stats(&improved);
    assert!(
        after.and_nodes < before.and_nodes,
        "{before:?} -> {after:?}"
    );
    assert_equivalent(&source, &prepared);
}

#[test]
fn values_that_need_the_mask_retain_it() {
    for value in [
        "not(x, id=8)",
        "or(x, one, id=8)",
        "add(x, one, id=8)",
        "priority_sel(s, cases=[x, x], default=one, id=8)",
        "one_hot_sel(s, cases=[x, one], id=8)",
        "identity(y, id=8)",
    ] {
        let text = format!(
            r#"package needed
top fn f(x: bits[8] id=1, y: bits[8] id=2, s: bits[2] id=3) -> bits[8] {{
 one: bits[8] = literal(value=1, id=4)
 nonzero: bits[1] = or_reduce(x, id=5)
 mask: bits[8] = sign_ext(nonzero, new_bit_count=8, id=6)
 value: bits[8] = {value}
 ret result: bits[8] = and(mask, value, id=9)
}}
"#
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(
            prepared
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Unop(Unop::OrReduce, _))),
            "necessary predicate removed for {value}"
        );
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn one_hot_sentinel_decode_and_zero_extensions_are_not_zero_masks() {
    for (source_width, width, value, extension) in [
        (3, 4, "one_hot(x, lsb_prio=true, id=5)", "sign_ext"),
        (3, 8, "decode(x, width=8, id=5)", "sign_ext"),
        (8, 8, "identity(x, id=5)", "zero_ext"),
    ] {
        let text = format!(
            r#"package sentinel
top fn f(x: bits[{source_width}] id=1) -> bits[{width}] {{
 nonzero: bits[1] = or_reduce(x, id=2)
 mask: bits[{width}] = {extension}(nonzero, new_bit_count={width}, id=3)
 value: bits[{width}] = {value}
 ret result: bits[{width}] = and(mask, value, id=6)
}}
"#
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(
            prepared
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Unop(Unop::OrReduce, _)))
        );
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn shared_masked_values_preserve_every_use() {
    let text = r#"package shared
top fn f(x: bits[9] id=1, y: bits[9] id=2) -> (bits[9], bits[9]) {
 nonzero: bits[1] = or_reduce(x, id=3)
 mask: bits[9] = sign_ext(nonzero, new_bit_count=9, id=4)
 shared: bits[9] = and(mask, y, id=5)
 result: bits[9] = and(shared, x, id=6)
 ret pair: (bits[9], bits[9]) = tuple(result, shared, id=7)
}
"#;
    let source = parse(text);
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    assert_equivalent(&source, &prepared);
}

#[test]
fn duplicate_masks_retain_a_witness_unless_the_data_provides_one() {
    let source = parse(
        r#"package duplicate_masks
top fn f(x: bits[8] id=1) -> (bits[8], bits[8]) {
 nonzero: bits[1] = or_reduce(x, id=2)
 mask: bits[8] = sign_ext(nonzero, new_bit_count=8, id=3)
 retained: bits[8] = and(mask, mask, id=4)
 inner: bits[8] = and(mask, mask, id=5)
 data: bits[8] = and(inner, x, id=6)
 ret pair: (bits[8], bits[8]) = tuple(retained, data, id=7)
}"#,
    );
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    assert!(
        prepared
            .nodes
            .iter()
            .all(|node| !matches!(node.payload, NodePayload::Nary(ir::NaryOp::And, _)))
    );
    assert_equivalent(&source, &prepared);
}

/// Exercises unconstrained shift/select controls and commuted/reassociated
/// masks.
fn selection_case(width: usize, predicate: usize, association: usize) -> String {
    let narrow = width - 1;
    let predicate = match predicate {
        0 => "or_reduce(x, id=11)",
        1 => "ne(x, zero, id=11)",
        _ => "ne(zero, x, id=11)",
    };
    let output = match association {
        0 => format!("ret result: bits[{width}] = and(value, mask, id=14)"),
        1 => format!("ret result: bits[{width}] = and(mask, value, id=14)"),
        2 => format!("ret result: bits[{width}] = and(y, mask, value, id=14)"),
        _ => format!(
            "inner: bits[{width}] = and(mask, y, id=13)\n ret result: bits[{width}] = and(value, inner, id=14)"
        ),
    };
    format!(
        r#"package zero_preserving
top fn f(x: bits[{width}] id=1, y: bits[{width}] id=2, s: bits[2] id=3, amount: bits[2] id=4) -> bits[{width}] {{
 zero: bits[{width}] = literal(value=0, id=5)
 pad: bits[1] = literal(value=0, id=6)
 slice: bits[{narrow}] = bit_slice(x, start=1, width={narrow}, id=7)
 projected: bits[{width}] = concat(slice, pad, id=8)
 shifted: bits[{width}] = shra(x, amount, id=9)
 choices: bits[{width}] = one_hot_sel(s, cases=[projected, shifted], id=10)
 nonzero: bits[1] = {predicate}
 mask: bits[{width}] = sign_ext(nonzero, new_bit_count={width}, id=12)
 value: bits[{width}] = priority_sel(s, cases=[choices, projected], default=zero, id=15)
 {output}
}}
"#
    )
}

fn node_by_id(f: &ir::Fn, id: usize) -> Option<ir::NodeRef> {
    f.nodes
        .iter()
        .position(|n| n.text_id == id)
        .map(|index| ir::NodeRef { index })
}

fn parse(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .expect("valid synthetic IR")
        .get_top_fn()
        .expect("top function")
        .clone()
}

fn assert_equivalent(source: &ir::Fn, prepared: &ir::Fn) {
    let result = prove_prepared_ir_equiv(source, prepared);
    assert!(matches!(result, EquivResult::Proved), "{result:?}");
}
