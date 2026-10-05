// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::cost::GateBuilderCostEvaluator;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_aug_opt::run_aug_opt_over_ir_text_with_stats;
use xlsynth_aug_opt::{
    AugOptMode, AugOptOptions, ir2gates_from_ir_text, run_aug_opt_over_ir_text_with_evaluator,
};
use xlsynth_g8r::aig_sim::gate_sim;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::ir2gates::{self, Ir2GatesOptions};
use xlsynth_pir::IrBits;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_pir::ir::{NaryOp, NodePayload};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_pir::ir_parser::Parser;

const PIR_ONLY: AugOptOptions = AugOptOptions {
    enable: true,
    rounds: 1,
    mode: AugOptMode::PirOnly,
    recover_split_adders: true,
    fuse_priority_results: false,
};

#[test]
fn package_top_and_explicit_override_select_the_mapped_function() {
    let text = r#"package top_selection

fn overridden(x: bits[4] id=1) -> bits[4] {
  ret result: bits[4] = not(x, id=2)
}

top fn package_top(x: bits[4] id=3) -> bits[4] {
  ret result: bits[4] = identity(x, id=4)
}
"#;
    for (enable, mode) in [
        (false, AugOptMode::Sandwich),
        (true, AugOptMode::PirOnly),
        (true, AugOptMode::Sandwich),
    ] {
        for (top, expected_name, expected_value) in [
            (None, "package_top", 6),
            (Some("overridden"), "overridden", 9),
        ] {
            let output = ir2gates_from_ir_text(
                text,
                top,
                AugOptOptions {
                    enable,
                    mode,
                    rounds: 1,
                    ..Default::default()
                },
                Ir2GatesOptions::all_opts_disabled(),
            )
            .unwrap();
            assert_eq!(output.top_fn_name, expected_name);
            assert_eq!(output.pir_top_fn().name, expected_name);
            let result = gate_sim::eval(
                &output.gatify_output.gate_fn,
                &[IrBits::make_ubits(4, 6).unwrap()],
                gate_sim::Collect::None,
            );
            assert_eq!(
                result.outputs,
                vec![IrBits::make_ubits(4, expected_value).unwrap()],
                "enable={enable}, mode={mode:?}, top={top:?}"
            );
        }
    }
}

#[test]
fn convenience_matches_explicit_costing_and_mapping_for_each_gate_setting() {
    let text = r#"package gate_settings

top fn main(x: bits[4] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> bits[4] {
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  ret result: bits[4] = shll(x, amount, id=11)
}
"#;
    for fold in [false, true] {
        for hash in [false, true] {
            let mut evaluator = GateBuilderCostEvaluator::default()
                .with_gate_builder_options(GateBuilderOptions { fold, hash });
            let optimized = run_aug_opt_over_ir_text_with_evaluator(
                text,
                Some("main"),
                PIR_ONLY,
                &mut evaluator,
            )
            .unwrap();
            if fold && hash {
                assert_eq!(optimized.rewrite_stats.constant_shift_choices, 1);
            }
            let explicit = ir2gates::ir2gates_from_ir_text(
                &optimized.output_text,
                Some("main"),
                Ir2GatesOptions {
                    fold,
                    hash,
                    ..Ir2GatesOptions::all_opts_disabled()
                },
            )
            .unwrap();
            let combined = ir2gates_from_ir_text(
                text,
                None,
                PIR_ONLY,
                Ir2GatesOptions {
                    fold,
                    hash,
                    ..Ir2GatesOptions::all_opts_disabled()
                },
            )
            .unwrap();
            assert_eq!(
                combined.pir_package.to_string(),
                explicit.pir_package.to_string(),
                "optimizer output differs for fold={fold}, hash={hash}"
            );
            assert_eq!(
                serde_json::to_value(&combined.gatify_output.gate_fn).unwrap(),
                serde_json::to_value(&explicit.gatify_output.gate_fn).unwrap(),
                "mapped gates differ for fold={fold}, hash={hash}"
            );
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn optimizer_output_is_prepared_for_alias_and_range_analysis() {
    let text = r#"package preparation_order

top fn main(value: bits[8] id=1, read_index: bits[2] id=2, shift: bits[3] id=3) -> bits[1] {
  base: bits[8][4] = literal(value=[1, 3, 5, 7], id=4)
  one: bits[2] = literal(value=1, id=5)
  write_index: bits[2] = xor(read_index, one, id=6)
  updated: bits[8][4] = array_update(base, value, indices=[write_index], id=7)
  read: bits[8] = array_index(updated, indices=[read_index], id=8)
  shifted: bits[8] = shll(read, shift, id=9)
  ret low_bit: bits[1] = bit_slice(shifted, start=0, width=1, id=10)
}
"#;
    let optimized = run_aug_opt_over_ir_text_with_stats(text, Some("main"), PIR_ONLY).unwrap();
    assert_eq!(optimized.rewrite_stats.lsb_of_shll, 1);

    // The optimizer creates the new low-bit expression while the read still
    // refers to the update; formal alias preparation must consume this IR.
    let optimized_package = Parser::new(&optimized.output_text)
        .parse_and_validate_package()
        .unwrap();
    let optimized_fn = optimized_package.get_top_fn().unwrap();
    let optimized_read = optimized_fn
        .nodes
        .iter()
        .find(|node| node.name.as_deref() == Some("read"))
        .unwrap();
    let NodePayload::ArrayIndex { array, .. } = optimized_read.payload else {
        panic!("the optimized read should remain an array index");
    };
    assert!(matches!(
        optimized_fn.get_node(array).payload,
        NodePayload::ArrayUpdate { .. }
    ));

    let explicit = ir2gates::ir2gates_from_ir_text(
        &optimized.output_text,
        Some("main"),
        Ir2GatesOptions {
            enable_formal_array_alias_analysis: true,
            ..Ir2GatesOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let combined = ir2gates_from_ir_text(
        text,
        None,
        PIR_ONLY,
        Ir2GatesOptions {
            enable_formal_array_alias_analysis: true,
            ..Ir2GatesOptions::all_opts_disabled()
        },
    )
    .unwrap();
    assert_eq!(
        combined.pir_package.to_string(),
        explicit.pir_package.to_string()
    );
    assert_eq!(
        serde_json::to_value(&combined.gatify_output.gate_fn).unwrap(),
        serde_json::to_value(&explicit.gatify_output.gate_fn).unwrap()
    );

    let prepared = combined.pir_top_fn();
    let prepared_read = prepared
        .nodes
        .iter()
        .find(|node| node.name.as_deref() == Some("read"))
        .unwrap();
    let NodePayload::ArrayIndex { array, .. } = prepared_read.payload else {
        panic!("the prepared read should remain an array index");
    };
    assert_eq!(prepared.get_node(array).name.as_deref(), Some("base"));

    let result = prepared.get_node(prepared.ret_node_ref.unwrap());
    let NodePayload::Nary(NaryOp::And, operands) = &result.payload else {
        panic!("the mapped function must contain the optimizer's low-bit rewrite");
    };
    let low_bit = operands
        .iter()
        .map(|&operand| prepared.get_node(operand))
        .find(|node| matches!(node.payload, NodePayload::BitSlice { .. }))
        .unwrap();
    assert!(low_bit.text_id > 10, "the slice must be created by aug-opt");

    // Every base element is odd, and the update cannot affect this read.
    // Facts for this newly introduced slice must come from the prepared graph.
    let facts = combined
        .range_info
        .get(low_bit.text_id)
        .unwrap()
        .known_bits
        .as_ref()
        .unwrap();
    let one = IrBits::make_ubits(1, 1).unwrap();
    assert_eq!(facts.mask(), &one);
    assert_eq!(facts.value(), &one);
    assert_eq!(
        Some(facts),
        explicit
            .range_info
            .get(low_bit.text_id)
            .unwrap()
            .known_bits
            .as_ref()
    );
}
