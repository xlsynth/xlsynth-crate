// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};
use xlsynth_pir::IrBits;

fn scalar_options() -> Ir2GatesOptions {
    Ir2GatesOptions {
        enable_formal_array_alias_analysis: false,
        ..Ir2GatesOptions::default()
    }
}

#[test]
fn sparse_products_only_replace_a_carry_that_can_reach_an_output() {
    for (op, factor, data, width, commuting, last, sparse) in [
        ("umul", "sparse", "data", 8, false, "product", true),
        ("umul", "sparse", "data", 8, true, "product", true),
        ("umul", "sparse", "data", 3, false, "product", true),
        ("umul", "sparse", "data", 2, false, "product", false),
        ("umul", "sparse", "separated", 8, false, "product", false),
        ("umul", "sparse", "data", 8, false, "low", false),
        ("umul", "multi", "data", 8, false, "product", false),
        ("smul", "sparse", "data", 8, false, "product", false),
    ] {
        let (a, b) = if commuting {
            (factor, data)
        } else {
            (data, factor)
        };
        let out = if last == "low" { 2 } else { width };
        let ir = format!(
            "package p\n\
             top fn f(data: bits[6] id=1, index: bits[2] id=2, en: bits[1] id=3, multi: bits[4] id=4) -> bits[{out}] {{\n\
               dec: bits[4] = decode(index, width=4, id=5)\n\
               z: bits[4] = literal(value=0, id=6)\n\
               sparse: bits[4] = sel(en, cases=[z, dec], id=7)\n\
               mask: bits[6] = literal(value=17, id=8)\n\
               separated: bits[6] = and(data, mask, id=9)\n\
               product: bits[{width}] = {op}({a}, {b}, id=10)\n\
               low: bits[2] = bit_slice(product, start=0, width=2, id=11)\n\
               ret result: bits[{out}] = identity({last}, id=12)\n\
             }}\n"
        );
        let mut graphs = Vec::new();
        for mapping in [None, Some(AdderMapping::default())] {
            let options = Ir2GatesOptions {
                mul_adder_mapping: mapping,
                ..scalar_options()
            };
            let lowered = ir2gates_from_ir_text(&ir, Some("f"), options).unwrap();
            let gates = &lowered.gatify_output.gate_fn;
            check_equivalence::validate_same_fn_via_toolchain(lowered.pir_top_fn(), gates).unwrap();
            graphs.push(emit_aiger(gates, false).unwrap());
        }
        assert_eq!(
            graphs[0] != graphs[1],
            sparse,
            "{op} {a} {b}, {width} -> {last}"
        );
    }
}

#[test]
fn independent_comparisons_can_prove_a_sparse_multiplicand() {
    let ir = "package pred
top fn product(x: bits[5] id=1, c: bits[3] id=2) -> bits[8] {
  one: bits[3] = literal(value=1, id=3)
  two: bits[3] = literal(value=2, id=4)
  four: bits[3] = literal(value=4, id=5)
  a: bits[1] = eq(c, one, id=6)
  b: bits[1] = eq(c, two, id=7)
  d: bits[1] = eq(c, four, id=8)
  scale: bits[3] = concat(a, b, d, id=9)
  ret y: bits[8] = umul(scale, x, id=10)
}";
    let lowered = ir2gates_from_ir_text(ir, Some("product"), scalar_options()).unwrap();
    assert!(lowered.range_info.is_sparse_product_factor(9, 3));
    assert!(!lowered.range_info.is_sparse_product_factor(9, 4));
    // This relation belongs to the product. Other range consumers still see
    // three independently unknown positions and no population promise.
    let known = lowered
        .range_info
        .get(9)
        .unwrap()
        .known_bits
        .as_ref()
        .unwrap();
    assert_eq!((known.min_ones(), known.max_ones()), (0, 3));
    let fixed = ir2gates_from_ir_text(
        ir,
        Some("product"),
        Ir2GatesOptions {
            mul_adder_mapping: Some(AdderMapping::default()),
            ..scalar_options()
        },
    )
    .unwrap();
    assert!(!fixed.range_info.is_sparse_product_factor(9, 3));
    check_equivalence::validate_same_fn_via_toolchain(
        lowered.pir_top_fn(),
        &lowered.gatify_output.gate_fn,
    )
    .unwrap();
}

#[test]
fn a_sparse_product_feeds_unsigned_or_negated_accumulators() {
    for (negate, arch) in [(false, ""), (true, ""), (false, ", arch=ripple_carry")] {
        let ir = format!("package mac
top fn f(a: bits[9] id=1, index: bits[2] id=2, bias: bits[10] id=3) -> bits[10] {{
  factor: bits[4] = decode(index, width=4, id=4)
  prod: bits[10] = umul(a, factor, id=5)
  ret y: bits[10] = ext_nary_add(bias, prod, signed=[false, false], negated=[false, {negate}]{arch}, id=6)
}}");
        let reference = ir.replace(
            &format!("ext_nary_add(bias, prod, signed=[false, false], negated=[false, {negate}]{arch}, id=6)"),
            if negate { "sub(bias, prod, id=6)" } else { "add(bias, prod, id=6)" },
        );
        let mut parser = xlsynth_pir::ir_parser::Parser::new(&reference);
        let reference = parser.parse_and_validate_package().unwrap();
        let mut graphs = Vec::new();
        for mul_adder_mapping in [None, Some(AdderMapping::default())] {
            let lowered = ir2gates_from_ir_text(
                &ir,
                Some("f"),
                Ir2GatesOptions {
                    mul_adder_mapping,
                    ..scalar_options()
                },
            )
            .unwrap();
            check_equivalence::validate_same_fn_via_toolchain(
                reference.get_top_fn().unwrap(),
                &lowered.gatify_output.gate_fn,
            )
            .unwrap();
            graphs.push(emit_aiger(&lowered.gatify_output.gate_fn, false).unwrap());
        }
        assert_ne!(graphs[0], graphs[1], "negated={negate} arch={arch}");
    }
}

#[test]
fn truncated_sparse_product_skips_zero_padding_in_the_accumulator() {
    let options = || Ir2GatesOptions {
        fold: false,
        hash: false,
        ..scalar_options()
    };
    let ir = r#"package mac
top fn f(data: bits[256] id=1, en: bits[1] id=2, bias: bits[3] id=3) -> bits[3] {
  factor: bits[256] = zero_ext(en, new_bit_count=256, id=4)
  prod: bits[3] = umul(data, factor, id=5)
  ret y: bits[3] = ext_nary_add(bias, prod, signed=[false, false], negated=[false, false], id=6)
}"#;
    let lowered = ir2gates_from_ir_text(ir, Some("f"), options()).unwrap();
    let gates = &lowered.gatify_output.gate_fn;
    let mut sim = PreparedGateSim::new(gates);
    for data in 0..8 {
        for en in 0..2 {
            for bias in 0..8 {
                let inputs = [
                    IrBits::make_ubits(256, data).unwrap(),
                    IrBits::make_ubits(1, en).unwrap(),
                    IrBits::make_ubits(3, bias).unwrap(),
                ];
                let expected = IrBits::make_ubits(3, (data * en + bias) % 8).unwrap();
                assert_eq!(
                    sim.eval_outputs(&inputs),
                    vec![expected],
                    "data={data} en={en} bias={bias}"
                );
            }
        }
    }

    let add_ir = r#"package acc
top fn f(bias: bits[3] id=1, prod: bits[3] id=2) -> bits[3] {
  ret y: bits[3] = ext_nary_add(bias, prod, signed=[false, false], negated=[false, false], id=3)
}"#;
    let add = ir2gates_from_ir_text(add_ir, Some("f"), options()).unwrap();
    let adder_ands = get_aig_stats(&add.gatify_output.gate_fn).and_nodes;
    let mac_ands = get_aig_stats(gates).and_nodes;
    // The retained columns have 1, 2 and 3 partial products: six ANDs and
    // at most three binary ORs in addition to the accumulator.
    assert!(
        mac_ands <= adder_ands + 6 + 3,
        "mac={mac_ands} adder={adder_ands}"
    );
}

#[test]
fn missing_facts_and_standalone_operands_keep_the_general_multiplier() {
    use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_node_as_fn, gatify_prepared_fn};
    let ir = "package boundary
top fn f(a: bits[5] id=1, index: bits[2] id=2) -> bits[7] {
  factor: bits[4] = decode(index, width=4, id=3)
  ret prod: bits[7] = umul(a, factor, id=4)
}";
    let prepared = ir2gates_from_ir_text(ir, Some("f"), scalar_options()).unwrap();
    let f = prepared.pir_top_fn();
    let product = f.ret_node_ref.unwrap();
    let mut missing = Vec::new();
    let mut standalone = Vec::new();
    for mul_adder_mapping in [None, Some(AdderMapping::default())] {
        let mut options = GatifyOptions {
            mul_adder_mapping,
            ..GatifyOptions::all_opts_enabled()
        };
        let lowered = gatify_prepared_fn(f, options.clone()).unwrap();
        missing.push(emit_aiger(&lowered.gate_fn, false).unwrap());
        options.range_info = Some(prepared.range_info.clone());
        // The standalone function exposes the decoded word as a free formal.
        let gates = gatify_node_as_fn(f, product, &options).unwrap();
        standalone.push(emit_aiger(&gates, false).unwrap());
    }
    assert_eq!(missing[0], missing[1]);
    assert_eq!(standalone[0], standalone[1]);
}

#[test]
fn a_multirow_carry_can_cross_an_empty_column() {
    let ir = "package sparse_holes
top fn f(data: bits[7] id=1, idx: bits[3] id=2) -> bits[1] {
  mask: bits[7] = literal(value=85, id=3)
  decoded: bits[7] = decode(idx, width=7, id=4)
  factor: bits[7] = and(decoded, mask, id=5)
  other: bits[7] = and(data, mask, id=6)
  prod: bits[9] = umul(factor, other, id=7)
  ret high: bits[1] = bit_slice(prod, start=8, width=1, id=8)
}";
    let mut graphs = Vec::new();
    for mul_adder_mapping in [None, Some(AdderMapping::default())] {
        let result = ir2gates_from_ir_text(
            ir,
            Some("f"),
            Ir2GatesOptions {
                mul_adder_mapping,
                ..scalar_options()
            },
        )
        .unwrap();
        check_equivalence::validate_same_fn_via_toolchain(
            result.pir_top_fn(),
            &result.gatify_output.gate_fn,
        )
        .unwrap();
        graphs.push(emit_aiger(&result.gatify_output.gate_fn, false).unwrap());
    }
    // Column six has four potentially live terms. Its carry can reach bit
    // eight even though every direct term in column seven is zero.
    assert_ne!(graphs[0], graphs[1]);
}
