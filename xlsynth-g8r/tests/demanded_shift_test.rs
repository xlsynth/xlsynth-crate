// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig_sim::gate_sim;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{
    GatifyOptions, IrRegion, gatify_node_as_fn, gatify_prepared_fn, gatify_region,
};
use xlsynth_g8r::ir_aig_sharing::{
    CandidateProofResult, IrAigCandidateRhs, IrAigEquivalenceCandidate,
    prove_equivalence_candidates_with_backend_streaming,
};
use xlsynth_g8r::prove_gate_fn_equiv_common::GateFormalBackend;
use xlsynth_pir::{IrBits, ir_parser::Parser, ir_utils};
use xlsynth_prover::prover::SolverChoice;

#[test]
#[cfg(feature = "has-bitwuzla")]
fn demanded_shift_observable_function_equivalence() {
    // The amount includes high overshift bits. Formal comparison covers every
    // input, including negative arithmetic values and non-power-of-two widths.
    for op in ["shll", "shrl", "shra"] {
        for width in [1usize, 3, 8, 9, 29, 65] {
            for (start, out_width) in [
                (0, 1),
                (width / 2, 1),
                (1.min(width), width.saturating_sub(2)),
            ] {
                let text = format!(
                    r#"fn f(x: bits[{width}] id=1, a: bits[9] id=2) -> bits[{out_width}] {{
  shifted: bits[{width}] = {op}(x, a, id=3)
  ret result: bits[{out_width}] = bit_slice(shifted, start={start}, width={out_width}, id=4)
}}"#
                );
                let f = Parser::new(&text).parse_fn().unwrap();
                for fold in [false, true] {
                    let options = GatifyOptions {
                        fold,
                        hash: fold,
                        check_equivalence: true,
                        equivalence_solver: SolverChoice::Bitwuzla,
                        ..GatifyOptions::all_opts_disabled()
                    };
                    gatify_prepared_fn(&f, options).unwrap_or_else(|e| panic!("{text}\n{e}"));
                }
            }
        }
    }
}

#[test]
#[cfg(feature = "has-bitwuzla")]
fn demanded_shift_shared_consumers_and_aggregate_wiring() {
    for extra in [false, true] {
        let (ret_ty, ret_expr) = if extra {
            ("(bits[8], bits[29])", "tuple(joined, shifted, id=10)")
        } else {
            ("(bits[8])", "tuple(joined, id=10)")
        };
        let text = format!(
            r#"fn f(x: bits[28] id=1, a: bits[5] id=2) -> {ret_ty} {{
  wide: bits[29] = zero_ext(x, new_bit_count=29, id=3)
  shifted: bits[29] = shll(wide, a, id=4)
  window: bits[27] = bit_slice(shifted, start=1, width=27, id=5)
  low: bits[3] = bit_slice(window, start=0, width=3, id=6)
  high: bits[2] = bit_slice(shifted, start=25, width=2, id=7)
  signed: bits[5] = sign_ext(high, new_bit_count=5, id=8)
  joined: bits[8] = concat(signed, low, id=9)
  ret result: {ret_ty} = {ret_expr}
}}"#
        );
        let f = Parser::new(&text).parse_fn().unwrap();
        let result = gatify_prepared_fn(
            &f,
            GatifyOptions {
                check_equivalence: true,
                equivalence_solver: SolverChoice::Bitwuzla,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let shift = ir_utils::get_topological(&f)
            .into_iter()
            .find(|r| f.get_node(*r).text_id == 4)
            .unwrap();
        assert_eq!(result.lowering_map.contains_key(&shift), extra);
    }
}

#[test]
fn demanded_shift_standalone_and_region_boundaries() {
    let f = Parser::new(
        r#"fn f(x: bits[8] id=1, a: bits[4] id=2) -> bits[1] {
  shifted: bits[8] = shrl(x, a, id=3)
  ret low: bits[1] = bit_slice(shifted, start=0, width=1, id=4)
}"#,
    )
    .parse_fn()
    .unwrap();
    let topo = ir_utils::get_topological(&f);
    let shift = *topo.iter().find(|r| f.get_node(**r).text_id == 3).unwrap();
    let target = gatify_node_as_fn(&f, shift, &GatifyOptions::all_opts_disabled()).unwrap();
    let region = gatify_region(
        IrRegion {
            function: &f,
            nodes: &[shift],
            inputs: &f.params,
            outputs: &[shift, shift],
        },
        GateBuilderOptions::opt(),
    )
    .unwrap();
    for a in 0..16 {
        let inputs = [
            IrBits::make_ubits(8, 0xe8).unwrap(),
            IrBits::make_ubits(4, a).unwrap(),
        ];
        let expected = 0xe8u64 >> a;
        assert_eq!(
            gate_sim::eval(&target, &inputs, gate_sim::Collect::None).outputs,
            vec![IrBits::make_ubits(8, expected).unwrap()]
        );
        assert_eq!(
            gate_sim::eval(&region, &inputs, gate_sim::Collect::None).outputs,
            vec![IrBits::make_ubits(16, expected | (expected << 8)).unwrap()]
        );
    }
}

#[test]
fn demanded_shift_unused_intermediate_bits_cannot_be_proved_constant() {
    let f = Parser::new(
        r#"fn f(x: bits[8] id=1, a: bits[4] id=2) -> bits[1] {
  shifted: bits[8] = shll(x, a, id=3)
  wired: bits[8] = identity(shifted, id=4)
  ret low: bits[1] = bit_slice(wired, start=0, width=1, id=5)
}"#,
    )
    .parse_fn()
    .unwrap();
    let opts = GatifyOptions {
        equivalence_solver: SolverChoice::Bitwuzla,
        ..GatifyOptions::all_opts_disabled()
    };
    let lowered = gatify_prepared_fn(&f, opts.clone()).unwrap();
    let candidates = ir_utils::get_topological(&f)
        .into_iter()
        .filter(|r| [3, 4].contains(&f.get_node(*r).text_id))
        .map(|r| IrAigEquivalenceCandidate {
            pir_node_ref: r,
            pir_node_text_id: f.get_node(r).text_id,
            bit_index: 7,
            rhs: IrAigCandidateRhs::Const(false),
        })
        .collect::<Vec<_>>();
    for candidate in &candidates {
        assert!(!lowered.lowering_map.contains_key(&candidate.pir_node_ref));
    }
    // Missing entries must skip before invoking a solver: bit 7 is not really
    // constant, even though the emitter may use a zero placeholder for it.
    let proofs = prove_equivalence_candidates_with_backend_streaming(
        &f,
        &lowered.gate_fn,
        &candidates,
        &opts,
        GateFormalBackend::Ir,
        |_| {},
    )
    .unwrap();
    assert_eq!(proofs.len(), 2);
    assert!(
        proofs
            .iter()
            .all(|p| matches!(p.result, CandidateProofResult::Skipped { .. }))
    );
}

#[test]
fn demanded_shift_late_low_control_qor() {
    // Minimized from frozen BVC five- and seven-operation cones: a late
    // low control and early high controls select one bit of a left shift.
    for priority in [false, true] {
        let width = if priority { 32 } else { 33 };
        let prefix = if priority {
            "reversed: bits[32] = reverse(control, id=4)\n  encoded_input: bits[33] = one_hot(reversed, lsb_prio=true, id=5)"
        } else {
            "encoded_input: bits[33] = identity(control, id=5)"
        };
        let text = format!(
            r#"fn f(control: bits[{width}] id=1, high: bits[6] id=2, data: bits[33] id=3) -> bits[1] {{
  {prefix}
  encoded: bits[6] = encode(encoded_input, id=6)
  low: bits[1] = bit_slice(encoded, start=0, width=1, id=7)
  amount: bits[7] = concat(high, low, id=8)
  shifted: bits[33] = shll(data, amount, id=9)
  ret result: bits[1] = bit_slice(shifted, start=7, width=1, id=10)
}}"#
        );
        let f = Parser::new(&text).parse_fn().unwrap();
        let result = gatify_prepared_fn(
            &f,
            GatifyOptions {
                check_equivalence: cfg!(feature = "has-bitwuzla"),
                equivalence_solver: SolverChoice::Bitwuzla,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let stats = xlsynth_g8r::aig::get_summary_stats::get_aig_stats(&result.gate_fn);
        let (max_nodes, max_depth) = if priority { (108, 16) } else { (40, 7) };
        assert!(stats.and_nodes <= max_nodes, "{stats:?}");
        assert!(stats.max_depth <= max_depth, "{stats:?}");
    }
}
