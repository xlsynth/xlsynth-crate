// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig::gate::AigBitVector;
use crate::aig::get_summary_stats::get_aig_cone_stats;
use crate::aig_sim::gate_sim::PreparedGateSim;
use crate::gate_builder::GateBuilderOptions;
use crate::gate_fn_optimize::{GateFnOptimizeOptions, optimize_gate_fn};
use crate::gatify::ir2gate::{
    GatifyOptions, TerminalObservations, gatify_lower_prepared_fn,
    gatify_with_terminal_observations,
};
use crate::ir2gates::ir2gates_with_terminal_observations;
use crate::process_ir_path::{CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts};
use std::fmt::Write;
use test_case::test_matrix;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn zero_width_concat_dag_in_return_wiring() {
    let mut ir =
        String::from("package empty\ntop fn f(x: bits[1], e0: bits[0]) -> (bits[1], bits[0]) {\n");
    for i in 1..=34 {
        let prev = i - 1;
        writeln!(
            ir,
            "  e{i}: bits[0] = concat(e{prev}, e{prev}, id={})",
            i + 2
        )
        .unwrap();
    }
    ir.push_str("  ret out: (bits[1], bits[0]) = tuple(x, e34, id=37)\n}");
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let mut observations = TerminalObservations::default();
    // Lower the DAG as written so prep cannot simplify away the empty parts.
    let gates = gatify_lower_prepared_fn(
        f,
        &GatifyOptions::all_opts_enabled(),
        None,
        f,
        Some(&mut observations),
    )
    .unwrap()
    .gate_fn;
    assert!(observations.ordinals.is_empty());
    assert_eq!(gates.outputs[0].get_bit_count(), 1);
    assert_eq!(
        gates.outputs[0].bit_vector.get_lsb(0),
        gates.inputs[0].bit_vector.get_lsb(0)
    );
}

#[test]
fn shared_nonempty_concat_with_many_empty_operands() {
    const COPIES: usize = 16_384;
    let ir = format!(
        r#"fn f(x: bits[1] id=1, e: bits[0] id=2) -> bits[{COPIES}] {{
  p: bits[1] = concat(x, {}id=3)
  ret flags: bits[{COPIES}] = concat({}id=4)
}}"#,
        "e, ".repeat(COPIES),
        "p, ".repeat(COPIES),
    );
    let f = Parser::new(&ir).parse_fn().unwrap();
    let mut observations = TerminalObservations::default();
    let mapped = gatify_lower_prepared_fn(
        &f,
        &GatifyOptions::all_opts_enabled(),
        None,
        &f,
        Some(&mut observations),
    )
    .unwrap()
    .gate_fn;
    assert!(observations.ordinals.is_empty());
    let x = mapped.inputs[0].bit_vector.get_lsb(0);
    assert!(
        mapped.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .all(|b| b == x)
    );
}

#[test]
fn distinct_ordinal_masks_fold_to_one_false_gate() {
    const FOLDED: usize = 64;
    let width = FOLDED + 1;
    let words: Vec<_> = (0..width).map(|i| format!("w{i}: bits[13]")).collect();
    let mut ir = format!(
        r#"package masks
top fn main({}, mask: bits[5]) -> bits[{width}] {{
  low: bits[5] = literal(value=15, id={})
  k: bits[5] = literal(value=16, id={})"#,
        words.join(", "),
        width + 2,
        width + 3
    );
    let mut mask_ids = Vec::new();
    let mut outputs = Vec::new();
    for i in 0..width {
        let id = width + 4 + 6 * i;
        let mask = if i == FOLDED { "mask" } else { "low" };
        write!(
            ir,
            r#"
  hot{i}: bits[14] = one_hot(w{i}, lsb_prio=true, id={id})
  index{i}: bits[4] = encode(hot{i}, id={})
  wide{i}: bits[5] = zero_ext(index{i}, new_bit_count=5, id={})
  diff{i}: bits[5] = sub(k, wide{i}, id={})
  masked{i}: bits[5] = and(diff{i}, {mask}, id={})
  bit{i}: bits[1] = bit_slice(masked{i}, start=4, width=1, id={})"#,
            id + 1,
            id + 2,
            id + 3,
            id + 4,
            id + 5
        )
        .unwrap();
        mask_ids.push((id + 4) as u32);
        outputs.push(format!("bit{i}"));
    }
    write!(
        ir,
        "\n  ret flags: bits[{width}] = concat({}, id={})\n}}",
        outputs.join(", "),
        width + 4 + 6 * width
    )
    .unwrap();

    let options = CanonicalG8rOptions {
        track_pir_node_ids: true,
        enable_formal_array_alias_analysis: false,
        fraig: false,
        reassociation: false,
        cut_db_rewrite: false,
        ..Default::default()
    };
    let opts = options.to_process_ir_path_options(Some("main"), true, false, false, None);
    let (ordinary, mut capture) =
        ir2gates_with_terminal_observations(&ir, Some("main"), (&opts).into(), true).unwrap();
    let stock = optimize_gate_fn(
        ordinary.gatify_output.gate_fn,
        &GateFnOptimizeOptions::from(&opts),
    )
    .unwrap()
    .gate_fn;
    let stock_bits = &stock.outputs[0].bit_vector;
    assert!(get_aig_cone_stats(&stock.gates, &[*stock_bits.get_lsb(0)]).and_nodes > 1);
    let trial = build_trial(&stock, &capture.ordinals, &mut capture.rail_budget, || {
        GateBuilder::append_to(stock.clone())
    })
    .unwrap();
    let bits = &trial.graph.outputs[0].bit_vector;
    let literal = bits.get_msb(0).node;
    let variable = bits.get_lsb(0).node;
    // All distinct constant masks contribute to one batch. The other batch
    // belongs to the variable AND which makes the trial profitable.
    assert_eq!(trial.provenance.len(), 2);
    let mut folded_ids = trial.provenance[&literal].clone();
    folded_ids.sort_unstable();
    assert_eq!(folded_ids.as_slice(), &mask_ids[..FOLDED]);
    assert!(trial.provenance[&variable].contains(&mask_ids[FOLDED]));

    let gates = canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("main"), &options)
        .unwrap()
        .gate_fn;
    let bits = &gates.outputs[0].bit_vector;
    let literal = *bits.get_msb(0);
    assert!(!literal.negated);
    assert!(matches!(
        gates.get(literal.node),
        AigNode::Literal { value: false, .. }
    ));
    assert!(bits.iter_msb_to_lsb().take(FOLDED).all(|&b| b == literal));
    let stock_root = stock_bits.get_msb(0);
    let stock_ids = stock.get(stock_root.node).get_pir_node_ids();
    assert!(
        folded_ids
            .iter()
            .all(|id| stock_ids.binary_search(id).is_ok())
    );
    assert_eq!(gates.get(literal.node).get_pir_node_ids(), stock_ids);
    let variable = *bits.get_lsb(0);
    assert_eq!(
        gates.get(variable.node).get_pir_node_ids(),
        &mask_ids[FOLDED..]
    );
    let AigNode::And2 { a, b, .. } = gates.get(variable.node) else {
        panic!("expected the variable mask to lower to a single AND")
    };
    assert!(!variable.negated);
    let word = gates.inputs[FOLDED].bit_vector.get_lsb(0);
    let mask = gates.inputs[width].bit_vector.get_lsb(4);
    assert!((a == word && b == mask) || (a == mask && b == word));
}

#[test]
fn apply_resolves_many_pins_by_name_and_bit_after_port_reordering() {
    const OUTPUT_COUNT: usize = 4096;
    let mut builder = GateBuilder::new("pins".into(), GateBuilderOptions::no_opt());
    let mut outputs = Vec::new();
    let mut plans = Vec::new();
    for bit in 0..OUTPUT_COUNT {
        let name = format!("word{bit}");
        let word = builder.add_input(name.clone(), 2);
        let rail = word.get_lsb(1).negate();
        // A redundant AND makes forwarding each inverted input profitable.
        outputs.push(builder.add_and_binary(rail, rail));
        plans.push(ReturnBit {
            positions: vec![ReturnPosition {
                bit,
                inverted: false,
            }],
            width: OUTPUT_COUNT,
            lhs: Pin::Input {
                name,
                bit: 1,
                inverted: true,
            },
            mask: None,
        });
    }
    builder.add_output(
        "output_value".into(),
        AigBitVector::from_lsb_is_index_0(&outputs),
    );
    let mut stock = builder.build();
    stock.inputs.reverse();
    let forwarded = apply(stock.clone(), &plans, &mut RailBudget::default());
    for (pin, word) in forwarded.outputs[0]
        .bit_vector
        .iter_lsb_to_msb()
        .zip(forwarded.inputs.iter().rev())
    {
        assert_eq!(*pin, word.bit_vector.get_lsb(1).negate());
    }

    // A coordinate missing from the rewritten port invalidates the entire
    // trial.
    stock.inputs[0].bit_vector = stock.inputs[0].bit_vector.get_lsb_slice(0, 1);
    let retained = apply(stock, &plans, &mut RailBudget::default());
    assert!(
        retained.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .eq(&outputs)
    );
}

#[test_matrix([("x", 8), ("hot", 9), ("index", 4), ("extended", 5), ("difference", 5)])]
fn shared_affine_ordinal_intermediates_preserve_the_word(exposure: (&str, usize)) {
    let (other, width) = exposure;
    let ir = format!(
        r#"package shared
top fn main(x: bits[8], mask: bits[5]) -> (bits[1], bits[{width}]) {{
  hot: bits[9] = one_hot(x, lsb_prio=true, id=3)
  index: bits[4] = encode(hot, id=4)
  extended: bits[5] = zero_ext(index, new_bit_count=5, id=5)
  k: bits[5] = literal(value=16, id=6)
  difference: bits[5] = sub(k, extended, id=7)
  masked: bits[5] = and(mask, difference, id=8)
  bit: bits[1] = bit_slice(masked, start=4, width=1, id=9)
  ret result: (bits[1], bits[{width}]) = tuple(bit, {other}, id=10)
}}"#
    );
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let (mapped, observations) = gatify_with_terminal_observations(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
        true,
    )
    .unwrap();
    // Returning the input word leaves every ordinal intermediate exclusive.
    assert_eq!(observations.ordinals.len(), usize::from(other == "x"));
    let gates = mapped.gate_fn;
    let root = *gates.outputs[0].bit_vector.get_lsb(width);
    assert!(
        get_aig_cone_stats(&gates.gates, &[root]).and_nodes > 1,
        "the observed word keeps the priority arithmetic for {other}"
    );
    let mut sim = PreparedGateSim::new(&gates);
    for x in 0u64..256 {
        let ordinal = u64::from(x.trailing_zeros().min(8));
        let retained = match other {
            "x" => x,
            "hot" => 1 << ordinal,
            "difference" => 16 - ordinal,
            _ => ordinal,
        };
        for mask in [0, 16] {
            let expected = (u64::from(mask != 0 && x & 1 != 0) << width) | retained;
            let inputs = [
                IrBits::make_ubits(8, x).unwrap(),
                IrBits::make_ubits(5, mask).unwrap(),
            ];
            assert_eq!(
                sim.eval_outputs(&inputs),
                vec![IrBits::make_ubits(width + 1, expected).unwrap()]
            );
        }
    }
}

#[test_matrix([false, true])]
fn computed_mask_capture_requires_an_unshared_ordinal_with_an_input_rail(compute_mask: bool) {
    let (source, mask) = if compute_mask {
        ("x", "lt")
    } else {
        ("sum", "m")
    };
    let mut bare =
        String::from("package computed\n\ntop fn main(x: bits[4], y: bits[4], m: bits[1])");
    let body = format!(
        r#"
  sum: bits[4] = xor(x, y, id=4)
  lt: bits[1] = ult(x, y, id=5)
  hot: bits[5] = one_hot({source}, lsb_prio=true, id=6)
  index: bits[3] = encode(hot, id=7)
  wide: bits[4] = zero_ext(index, new_bit_count=4, id=8)
  k: bits[4] = literal(value=8, id=9)
  difference: bits[4] = sub(k, wide, id=10)
  mask: bits[4] = sign_ext({mask}, new_bit_count=4, id=11)
  masked: bits[4] = xor(difference, mask, id=12)
  flag: bits[1] = bit_slice(masked, start=3, width=1, id=13)"#
    );
    let retained = format!(
        "{bare} -> (bits[1], bits[4]) {{{body}\n  ret out: (bits[1], bits[4]) = tuple(flag, difference, id=14)\n}}"
    );
    bare.push_str(&format!(
        " -> bits[1] {{{}\n}}",
        body.replace("  flag: ", "  ret flag: ")
    ));

    let mut sizes = Vec::new();
    for (ir, retained_width) in [(&bare, 0), (&retained, 4)] {
        let package = Parser::new(ir).parse_and_validate_package().unwrap();
        let (mapped, observations) = gatify_with_terminal_observations(
            package.get_top_fn().unwrap(),
            GatifyOptions {
                fold: true,
                hash: true,
                ..GatifyOptions::all_opts_disabled()
            },
            true,
        )
        .unwrap();
        if compute_mask && retained_width == 0 {
            assert!(matches!(
                observations.ordinals.as_slice(),
                [ReturnBit {
                    mask: Some(MaskOperation {
                        op: NaryOp::Xor,
                        rhs: MaskBit::Cone { .. },
                        ..
                    }),
                    ..
                }]
            ));
        } else {
            assert!(observations.ordinals.is_empty());
        }
        let gates = mapped.gate_fn;
        let flag = *gates.outputs[0].bit_vector.get_lsb(retained_width);
        sizes.push(get_aig_cone_stats(&gates.gates, &[flag]).and_nodes);
        let mut sim = PreparedGateSim::new(&gates);
        for x in 0u64..16 {
            for y in 0u64..16 {
                for m in [false, true] {
                    let source = if compute_mask { x } else { x ^ y };
                    let mask = if compute_mask { x < y } else { m };
                    let difference = 8 - u64::from(source.trailing_zeros().min(4));
                    let value = u64::from((source & 1 != 0) ^ mask);
                    let expected = if retained_width == 0 {
                        value
                    } else {
                        (value << 4) | difference
                    };
                    let inputs = [
                        IrBits::make_ubits(4, x).unwrap(),
                        IrBits::make_ubits(4, y).unwrap(),
                        IrBits::bool(m),
                    ];
                    assert_eq!(
                        sim.eval_outputs(&inputs),
                        vec![IrBits::make_ubits(retained_width + 1, expected).unwrap()]
                    );
                }
            }
        }
    }
    assert_eq!(sizes[0], sizes[1]);
}
