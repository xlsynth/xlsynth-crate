// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, IrRegion, gatify_prepared_fn, gatify_region};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, NodeRef};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;

/// Parses a self-contained test function with a selected top.
fn parse(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone()
}

/// Exercises independent products, signs, truncation and literal corrections.
fn cluster(
    x: usize,
    y: usize,
    product: usize,
    result: usize,
    signed: bool,
    negative: bool,
    arch: &str,
) -> ir::Fn {
    let op = if signed { "smul" } else { "umul" };
    parse(&format!(
        r#"package cluster
 top fn f(a: bits[{x}] id=1, b: bits[{y}] id=2, c: bits[{result}] id=3, d: bits[{x}] id=4) -> bits[{result}] {{
  p: bits[{product}] = {op}(a, b, id=5)
  q: bits[{product}] = {op}(d, b, id=6)
  k: bits[{result}] = literal(value=1, id=7)
  ret r: bits[{result}] = ext_nary_add(p, q, c, k, signed=[true, false, false, false], negated=[{negative}, false, true, false], arch={arch}, id=8)
 }}"#
    ))
}

/// Keeps source interpretation independent of the gate construction.
fn check_samples(f: &ir::Fn, count: usize, rng: &mut StdRng, hash: bool) {
    let mapped = gatify_prepared_fn(
        f,
        GatifyOptions {
            fold: hash,
            hash,
            enable_rewrite_nary_add: true,
            adder_mapping: AdderMapping::BrentKung,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    for args in generate_mixed_argument_sets_with_rng(f, rng, count) {
        let expected = match eval_fn(f, &args) {
            FnEvalResult::Success(r) => r.value.to_bits().unwrap(),
            failure => panic!("source evaluation failed: {failure:?}"),
        };
        let inputs = args
            .iter()
            .map(|a| a.to_bits().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            gate_sim::eval(&mapped.gate_fn, &inputs, Collect::None).outputs,
            vec![expected],
            "{f}\nargs={args:?}"
        );
    }
}

#[test]
fn arithmetic_cluster_directed_property_coverage() {
    let mut rng = StdRng::seed_from_u64(0xdadda_2026);
    for (x, y, product, result) in [
        (1, 1, 1, 1),
        (2, 3, 5, 3),
        (3, 3, 6, 6),
        (4, 7, 11, 8),
        (8, 8, 16, 16),
        (17, 5, 22, 17),
        (65, 3, 68, 65),
    ] {
        for signed in [false, true] {
            for negative in [false, true] {
                for arch in ["ripple_carry", "brent_kung", "kogge_stone"] {
                    let f = cluster(x, y, product, result, signed, negative, arch);
                    check_samples(&f, 64, &mut rng, true);
                    if x <= 3 {
                        check_samples(&f, 16, &mut rng, false);
                    }
                }
            }
        }
    }
    let wide = parse(
        r#"package wide
 top fn f(a: bits[129] id=1, c: bits[129] id=2) -> bits[129] {
  k: bits[129] = literal(value=0x1ffffffffffffffffffffffffffffffff, id=3)
  p: bits[129] = umul(a, k, id=4)
  ret r: bits[129] = ext_nary_add(p, c, signed=[false, false], negated=[true, false], id=5)
 }"#,
    );
    check_samples(&wide, 256, &mut rng, true);
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn arithmetic_cluster_small_formal_equivalence() {
    for (x, y, p, r) in [(1, 1, 1, 1), (2, 3, 5, 3), (3, 3, 6, 6)] {
        for signed in [false, true] {
            for negative in [false, true] {
                let f = cluster(x, y, p, r, signed, negative, "brent_kung");
                let mapped = gatify_prepared_fn(
                    &f,
                    GatifyOptions {
                        enable_rewrite_nary_add: true,
                        ..GatifyOptions::all_opts_disabled()
                    },
                )
                .unwrap();
                xlsynth_g8r::check_equivalence::validate_same_fn_with_solver(
                    &f,
                    &mapped.gate_fn,
                    xlsynth_prover::prover::SolverChoice::Bitwuzla,
                    None,
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn arithmetic_cluster_multiplicity_boundaries_and_provenance() {
    let f = cluster(3, 3, 6, 6, false, false, "brent_kung");
    let mapped = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let product = NodeRef {
        index: f.nodes.iter().position(|n| n.text_id == 5).unwrap(),
    };
    assert!(!mapped.lowering_map.contains_key(&product));
    assert!(mapped.lowering_map.contains_key(&f.ret_node_ref.unwrap()));
    for gate in &mapped.gate_fn.gates {
        if matches!(gate, AigNode::And2 { .. }) {
            assert_eq!(gate.get_pir_node_ids(), &[8]);
        }
    }
    let resolved = gatify_prepared_fn(&f, GatifyOptions::all_opts_disabled()).unwrap();
    assert!(resolved.lowering_map.contains_key(&product));
    let mut rng = StdRng::seed_from_u64(0x1234);
    for expression in [
        "ext_nary_add(p, p, signed=[false, false], negated=[false, false], id=6)",
        "ext_nary_add(p, c, signed=[false, false], negated=[false, false], id=6)",
    ] {
        let f = parse(&format!(
            r#"package multiplicity
 top fn f(a: bits[3] id=1, b: bits[3] id=2, c: bits[6] id=3) -> bits[6] {{
  p: bits[6] = umul(a,b,id=5)
  ret r: bits[6] = {expression}
 }}"#
        ));
        check_samples(&f, 128, &mut rng, true);
    }
    for extension in ["zero_ext", "sign_ext"] {
        let f = parse(&format!(
            r#"package narrow
 top fn f(a: bits[4] id=1, b: bits[4] id=2, c: bits[8] id=3) -> bits[8] {{
  p: bits[4] = smul(a,b,id=4)
  e: bits[8] = {extension}(p,new_bit_count=8,id=5)
  ret r: bits[8] = ext_nary_add(e,c,signed=[false,false],negated=[false,false],id=6)
 }}"#
        ));
        let mapped = gatify_prepared_fn(
            &f,
            GatifyOptions {
                enable_rewrite_nary_add: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let p = NodeRef {
            index: f.nodes.iter().position(|n| n.text_id == 4).unwrap(),
        };
        assert!(mapped.lowering_map.contains_key(&p));
        check_samples(&f, 128, &mut rng, true);
    }
    // Arithmetic values remain independent region inputs; interior multiply
    // remains unsupported by the existing region API.
    let output = gatify_region(
        IrRegion {
            function: &f,
            nodes: &[],
            inputs: &[product],
            outputs: &[product],
        },
        GateBuilderOptions::opt(),
    )
    .unwrap();
    assert_eq!(output.inputs.len(), 1);
    assert!(
        gatify_region(
            IrRegion {
                function: &f,
                nodes: &[product],
                inputs: &f.params,
                outputs: &[product]
            },
            GateBuilderOptions::opt()
        )
        .is_err()
    );
}

#[test]
fn arithmetic_cluster_published_dot_product() {
    let f = parse(include_str!("data/arithmetic_cluster_dot_s8_n2.ir"));
    let resolved = xlsynth_g8r::gatify::ir2gate::gatify(
        &f,
        GatifyOptions {
            adder_mapping: AdderMapping::BrentKung,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let fused = xlsynth_g8r::gatify::ir2gate::gatify(
        &f,
        GatifyOptions {
            adder_mapping: AdderMapping::BrentKung,
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    assert!(
        get_aig_stats(&dce(&fused.gate_fn)).and_nodes
            < get_aig_stats(&dce(&resolved.gate_fn)).and_nodes
    );
    let mut rng = StdRng::seed_from_u64(0x2026_10_12);
    for args in generate_mixed_argument_sets_with_rng(&f, &mut rng, 256) {
        let expected = match eval_fn(&f, &args) {
            FnEvalResult::Success(r) => r.value.to_bits().unwrap(),
            failure => panic!("published function evaluation failed: {failure:?}"),
        };
        let inputs = args
            .iter()
            .zip(f.param_nodes())
            .map(|(arg, param)| {
                let mut bits = Vec::new();
                flatten_ir_value_to_lsb0_bits_for_type(arg, &param.ty, &mut bits).unwrap();
                IrBits::from_lsb_is_0(&bits)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            gate_sim::eval(&fused.gate_fn, &inputs, Collect::None).outputs,
            vec![expected]
        );
    }
}

#[test]
fn arithmetic_cluster_selected_negate_consumer() {
    let f = parse(
        r#"package selected
 top fn f(a: bits[3] id=1, b: bits[3] id=2, c: bits[6] id=3, s: bits[1] id=4) -> bits[6] {
  p: bits[6] = smul(a,b,id=5)
  n: bits[6] = neg(c,id=6)
  v: bits[6] = sel(s, cases=[c, n], id=7)
  ret r: bits[6] = ext_nary_add(p,v,signed=[false,false],negated=[false,false],id=8)
 }"#,
    );
    let mut rng = StdRng::seed_from_u64(0x5678);
    check_samples(&f, 256, &mut rng, true);
}

#[test]
fn arithmetic_cluster_zero_width_boundary() {
    let f = parse(
        r#"package empty
 top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[0] {
  p: bits[1] = umul(a, b, id=3)
  ret r: bits[0] = ext_nary_add(p, signed=[false], negated=[false], id=4)
 }"#,
    );
    let mapped = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    assert_eq!(mapped.gate_fn.outputs[0].bit_vector.get_bit_count(), 0);
}

/// Keeps the late-accumulator boundary, truncation, and late-product control
/// reproducible through actual gate QoR and complete result-vector proofs.
#[test]
fn arithmetic_cluster_arrival_boundary_regressions() {
    for (source, expected_ands, expected_depth) in [
        (
            include_str!("data/arithmetic_cluster_late_accumulator.ir"),
            625,
            103,
        ),
        (
            include_str!("data/arithmetic_cluster_late_accumulator_truncated.ir"),
            1195,
            142,
        ),
        (
            include_str!("data/arithmetic_cluster_late_product.ir"),
            500,
            98,
        ),
    ] {
        let f = parse(source);
        let mapped = gatify_prepared_fn(
            &f,
            GatifyOptions {
                fold: true,
                hash: true,
                enable_rewrite_nary_add: true,
                adder_mapping: AdderMapping::BrentKung,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let stats = get_aig_stats(&dce(&mapped.gate_fn));
        assert_eq!(
            (stats.and_nodes, stats.max_depth),
            (expected_ands, expected_depth)
        );
        let mut rng = StdRng::seed_from_u64(0x2026_1013);
        check_samples(&f, 128, &mut rng, true);
        #[cfg(feature = "has-bitwuzla")]
        xlsynth_g8r::check_equivalence::validate_same_fn_with_solver(
            &f,
            &mapped.gate_fn,
            xlsynth_prover::prover::SolverChoice::Bitwuzla,
            None,
        )
        .unwrap();
    }
}

/// Keeps canonical add-only trees, input-arrival skew, and externally visible
/// partial sums in the same semantic checks as product rows.
#[test]
fn arithmetic_rows_add_only_and_shared_boundaries() {
    let mut rng = StdRng::seed_from_u64(0xa441_2026);
    for width in [1, 4, 17, 65] {
        for delay in [0, 8] {
            for shared in [false, true] {
                let mut body = Vec::new();
                let mut late = "d".to_string();
                let mut id = 5;
                for _ in 0..delay {
                    body.push(format!("v{id}: bits[{width}] = reverse({late}, id={id})"));
                    let previous = id;
                    id += 1;
                    body.push(format!(
                        "v{id}: bits[{width}] = add(v{previous}, c, id={id})"
                    ));
                    late = format!("v{id}");
                    id += 1;
                }
                body.push(format!("ab: bits[{width}] = add(a,b,id={id})"));
                id += 1;
                body.push(format!("early: bits[{width}] = sub(ab,c,id={id})"));
                id += 1;
                body.push(format!("sum: bits[{width}] = add(early,{late},id={id})"));
                id += 1;
                let result_type = if shared {
                    format!("(bits[{width}], bits[{width}])")
                } else {
                    format!("bits[{width}]")
                };
                body.push(if shared {
                    format!("ret result: {result_type} = tuple(early,sum,id={id})")
                } else {
                    format!("ret result: {result_type} = identity(sum,id={id})")
                });
                let f = parse(&format!(
                    "package add_rows\ntop fn f(a: bits[{width}] id=1,b: bits[{width}] id=2,c: bits[{width}] id=3,d: bits[{width}] id=4) -> {result_type} {{\n{}\n}}",
                    body.join("\n")
                ));
                for mapping in [
                    AdderMapping::BrentKung,
                    AdderMapping::RippleCarry,
                    AdderMapping::KoggeStone,
                ] {
                    let mapped = xlsynth_g8r::gatify::ir2gate::gatify(
                        &f,
                        GatifyOptions {
                            fold: true,
                            hash: true,
                            enable_rewrite_nary_add: true,
                            adder_mapping: mapping,
                            ..GatifyOptions::all_opts_disabled()
                        },
                    )
                    .unwrap();
                    for args in generate_mixed_argument_sets_with_rng(&f, &mut rng, 64) {
                        let expected = match eval_fn(&f, &args) {
                            FnEvalResult::Success(r) => r.value,
                            failure => panic!("{failure:?}"),
                        };
                        let mut bits = Vec::new();
                        flatten_ir_value_to_lsb0_bits_for_type(&expected, &f.ret_ty, &mut bits)
                            .unwrap();
                        let inputs: Vec<_> = args.iter().map(|a| a.to_bits().unwrap()).collect();
                        assert_eq!(
                            gate_sim::eval(&mapped.gate_fn, &inputs, Collect::None).outputs,
                            vec![IrBits::from_lsb_is_0(&bits)],
                            "{f}\n{args:?}"
                        );
                    }
                    #[cfg(feature = "has-bitwuzla")]
                    if width <= 4 {
                        xlsynth_g8r::check_equivalence::validate_same_fn_with_solver(
                            &f,
                            &mapped.gate_fn,
                            xlsynth_prover::prover::SolverChoice::Bitwuzla,
                            None,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

/// Real MFFC from acm_random::rng_next64. Its shared, signed narrow sum
/// prevents IR flattening; repeated high bits still benefit from row lowering.
#[test]
fn arithmetic_rows_rng_corpus_regression() {
    let f = parse(include_str!("data/arithmetic_rows_rng.ir"));
    let mapped = gatify_prepared_fn(
        &f,
        GatifyOptions {
            fold: true,
            hash: true,
            enable_rewrite_nary_add: true,
            adder_mapping: AdderMapping::BrentKung,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let stats = get_aig_stats(&dce(&mapped.gate_fn));
    // Pinned main: 1,207 live ANDs and depth 53; graph-LE is unchanged.
    assert_eq!((stats.and_nodes, stats.max_depth), (1087, 53));
    let mut rng = StdRng::seed_from_u64(0x0589_8897);
    check_samples(&f, 256, &mut rng, true);
    #[cfg(feature = "has-bitwuzla")]
    xlsynth_g8r::check_equivalence::validate_same_fn_with_solver(
        &f,
        &mapped.gate_fn,
        xlsynth_prover::prover::SolverChoice::Bitwuzla,
        None,
    )
    .unwrap();
}
