// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use test_case::test_matrix;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::find_node_by_name;
use xlsynth_pir::libxls_bridge::{value_from_libxls, value_to_libxls};
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;
use xlsynth_pir::{IrBits, IrValue};

#[test_matrix([false, true], [false, true])]
fn nonnegative_signed_multiply_preserves_semantics(fused: bool, both_nonnegative: bool) {
    for (signed_width, positive_width, output_width) in [
        (1, 1, 1),
        (2, 1, 5),
        (3, 2, 3),
        (7, 2, 11),
        (17, 3, 22),
        (65, 2, 69),
    ] {
        for commuted in [false, true] {
            for fold in [false, true] {
                let ir = multiply_ir(
                    signed_width,
                    positive_width,
                    output_width,
                    both_nonnegative,
                    commuted,
                    false,
                );
                let signed = lower_and_check(&ir, fused, fold).unwrap();
                if signed_width > 7 {
                    let reference_ir = multiply_ir(
                        signed_width,
                        positive_width,
                        output_width,
                        both_nonnegative,
                        false,
                        true,
                    );
                    let reference = lower_and_check(&reference_ir, fused, fold).unwrap();
                    // Identical AIGs prove equivalence to this modular unsigned
                    // reference for every input without a wide multiplier SAT
                    // query.
                    assert_eq!(
                        emit_aiger_binary(&signed, false).unwrap(),
                        emit_aiger_binary(&reference, false).unwrap(),
                        "wide signed fallback should match its explicit modular unsigned reference"
                    );
                }
            }
        }
    }
}

#[test_matrix([false, true])]
fn nonnegative_signed_multiply_uses_narrow_rows_in_either_order(fused: bool) {
    let unsigned_ir = multiply_ir(17, 2, 21, false, false, true);
    let unsigned = lower_and_check(&unsigned_ir, fused, true).unwrap();
    let expected = get_aig_stats(&unsigned);
    for commuted in [false, true] {
        let signed_ir = multiply_ir(17, 2, 21, false, commuted, false);
        let signed = lower_and_check(&signed_ir, fused, true).unwrap();
        let actual = get_aig_stats(&signed);
        assert_eq!(
            (actual.and_nodes, actual.max_depth),
            (expected.and_nodes, expected.max_depth),
            "known-nonnegative multiplier should cost the same as explicit modular unsigned lowering"
        );
    }
}

#[test_matrix([false, true], [false, true])]
fn dense_positive_literal_uses_constant_multiply_rows(fused: bool, commuted: bool) {
    let literal = "k: bits[17] = literal(value=65535, id=4)";
    let zero_extended = r#"ones: bits[16] = literal(value=65535, id=3)
  k: bits[17] = zero_ext(ones, new_bit_count=17, id=4)"#;
    // Also exercise sign extension of an 8-bit multiplicand to the result
    // width.
    for input_width in [18, 8] {
        let lower = |op: &str, constant: &str| {
            let x = if op == "umul" { "extended" } else { "x" };
            let operands = if commuted {
                format!("{x}, k")
            } else {
                format!("k, {x}")
            };
            let product = format!("p: bits[18] = {op}({operands}, id=6)");
            let result = if fused {
                format!("{product}\n  ret r: bits[18] = add(p, c, id=7)")
            } else {
                format!("ret {product}")
            };
            let ir = format!(
                r#"package sample
top fn main(x: bits[{input_width}] id=1, c: bits[18] id=2) -> bits[18] {{
  {constant}
  extended: bits[18] = sign_ext(x, new_bit_count=18, id=5)
  {result}
}}"#
            );
            let gates = lower_product(&ir, fused).unwrap();
            let stats = get_aig_stats(&gates);
            // Check signed bounds and overflow independently of row selection.
            let mut sim = PreparedGateSim::new(&gates);
            let sign_bit = 1i64 << (input_width - 1);
            for x in [-sign_bit, -7, -1, 0, 1, 7, sign_bit - 1] {
                let c = 11;
                let expected = (65535 * x + if fused { c } else { 0 }) & 0x3ffff;
                assert_eq!(
                    sim.eval_outputs(&[
                        IrBits::make_sbits(input_width, x).unwrap(),
                        IrBits::make_ubits(18, c as u64).unwrap(),
                    ]),
                    vec![IrBits::make_ubits(18, expected as u64).unwrap()]
                );
            }
            (stats.and_nodes, stats.max_depth)
        };
        let expected = lower("umul", literal);
        for constant in [literal, zero_extended] {
            assert_eq!(
                lower("smul", constant),
                expected,
                "input width {input_width}, {constant}"
            );
        }
    }
}

#[test_matrix(["x, k", "k, x"])]
fn sparse_multiplicand_times_dense_constant_is_wiring(operands: &str) {
    for constant in [
        "k: bits[17] = zero_ext(ones, new_bit_count=17, id=7)",
        "k: bits[17] = literal(value=65535, id=7)",
    ] {
        let ir = format!(
            r#"package sample
top fn main(a: bits[1] id=1, b: bits[1] id=2) -> bits[18] {{
  z: bits[1] = literal(value=0, id=3)
  pad: bits[15] = literal(value=0, id=4)
  x: bits[18] = concat(z, b, pad, a, id=5)
  ones: bits[16] = literal(value=65535, id=6)
  {constant}
  ret p: bits[18] = smul({operands}, id=8)
}}"#
        );
        let gates = ir2gates_from_ir_text(
            &ir,
            None,
            Ir2GatesOptions {
                enable_formal_array_alias_analysis: cfg!(feature = "has-bitwuzla"),
                ..Ir2GatesOptions::default()
            },
        )
        .unwrap()
        .gatify_output
        .gate_fn;
        // Multiplying by 65535 copies a to bits 0..16. Only the two low
        // bits of b's shifted copy fit in the 18-bit product.
        let result = &gates.outputs[0].bit_vector;
        for i in 0..18 {
            let input = if i < 16 {
                &gates.inputs[0]
            } else {
                &gates.inputs[1]
            };
            assert_eq!(
                result.get_lsb(i),
                input.bit_vector.get_lsb(0),
                "bit {i}: {ir}"
            );
        }
    }
}

#[test]
fn fused_truncating_signed_multiply_matches_explicit_low_bits() {
    let lower = |factor: &str| {
        let ir = format!(
            r#"package sample
top fn main(x: bits[4096] id=1, y: bits[8] id=2, c: bits[8] id=3) -> bits[8] {{
  low: bits[8] = bit_slice(x, start=0, width=8, id=4)
  p: bits[8] = smul({factor}, y, id=5)
  ret r: bits[8] = add(p, c, id=6)
}}"#
        );
        let gates = lower_product(&ir, true).unwrap();
        emit_aiger_binary(&gates, false).unwrap()
    };
    assert_eq!(lower("x"), lower("low"));
}

#[test_matrix([false, true])]
fn widening_product_preserves_negative_literal_sign(fused: bool) {
    let ir = r#"package sample
top fn main(x: bits[3] id=1, c: bits[7] id=2) -> bits[7] {
  k: bits[4] = literal(value=15, id=3)
  p: bits[7] = smul(k, x, id=4)
  ret r: bits[7] = add(p, c, id=5)
}"#;
    lower_and_check(ir, fused, true).unwrap();
}

/// Lowers the function and checks whether its product was consumed as rows.
fn lower_product(ir: &str, fused: bool) -> Result<GateFn, String> {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    let function = package.get_top_fn().ok_or("missing top function")?;
    let product = find_node_by_name(function, "p").ok_or("missing product")?;
    let mapped = gatify(
        function,
        GatifyOptions {
            enable_rewrite_nary_add: fused,
            ..GatifyOptions::all_opts_disabled()
        },
    )?;
    assert_eq!(mapped.lowering_map.contains_key(&product), !fused);
    Ok(mapped.gate_fn)
}

/// Checks the gate result against native XLS signed arithmetic, including
/// overflow.
fn lower_and_check(ir: &str, fused: bool, fold: bool) -> Result<GateFn, String> {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    let function = package.get_top_fn().ok_or("test has no top function")?;
    let lowered = gatify(
        function,
        GatifyOptions {
            fold,
            hash: true,
            enable_rewrite_nary_add: fused,
            ..GatifyOptions::all_opts_disabled()
        },
    )?;
    if function
        .param_nodes()
        .take(2)
        .all(|p| p.ty.bit_count() <= 7)
    {
        check_equivalence::validate_same_fn_via_toolchain(function, &lowered.gate_fn)?;
    } else {
        // Wide mixed-sign multiplier-vs-bit-circuit SAT queries can be
        // expensive. Exercise signed bounds and overflow independently through
        // native XLS.
        let native_package = xlsynth::IrPackage::parse_ir(ir, None).map_err(|e| e.to_string())?;
        let native_function = native_package
            .get_function(&function.name)
            .map_err(|e| e.to_string())?;
        let mut rng = StdRng::seed_from_u64(0x6d75_6c70_6f73);
        let mut samples = generate_mixed_argument_sets_with_rng(function, &mut rng, 64);
        let boundaries: Vec<Vec<IrValue>> = function
            .param_nodes()
            .map(|p| {
                let width = p.ty.bit_count();
                (0..7)
                    .map(|pattern| {
                        let bits: Vec<bool> = (0..width)
                            .map(|bit| match pattern {
                                0 => false,
                                1 => true,
                                2 => bit + 1 == width,
                                3 => bit + 1 != width,
                                4 => bit == 0,
                                5 => bit % 2 == 0,
                                _ => bit % 2 != 0,
                            })
                            .collect();
                        IrValue::from_bits(&IrBits::from_lsb_is_0(&bits))
                    })
                    .collect()
            })
            .collect();
        for a in &boundaries[0] {
            for b in &boundaries[1] {
                for c in &boundaries[2] {
                    samples.push(vec![a.clone(), b.clone(), c.clone()]);
                }
            }
        }
        let mut simulation = PreparedGateSim::new(&lowered.gate_fn);
        for args in samples {
            let native_args = args
                .iter()
                .map(value_to_libxls)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            let native_result = native_function
                .interpret(&native_args)
                .map_err(|e| e.to_string())?;
            let expected = value_from_libxls(&native_result, &function.ret_ty)
                .map_err(|e| e.to_string())?
                .to_bits()
                .map_err(|e| e.to_string())?;
            let gate_args = args
                .iter()
                .map(|arg| arg.to_bits())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            assert_eq!(
                simulation.eval_outputs(&gate_args),
                vec![expected],
                "native interpreter mismatch for inputs {args:?}\n{ir}"
            );
        }
    }
    Ok(lowered.gate_fn)
}

/// Generates modular multiply-adds with a dynamic signed or nonnegative lhs.
fn multiply_ir(
    signed_width: usize,
    positive_width: usize,
    output_width: usize,
    both_nonnegative: bool,
    commuted: bool,
    explicit_unsigned: bool,
) -> String {
    let rhs_width = positive_width + 1;
    let lhs_width = signed_width + usize::from(both_nonnegative);
    let lhs_definition = if both_nonnegative {
        format!(
            "  positive_lhs: bits[{lhs_width}] = zero_ext(a, new_bit_count={lhs_width}, id=5)\n"
        )
    } else {
        String::new()
    };
    let lhs = if both_nonnegative {
        "positive_lhs"
    } else {
        "a"
    };
    let extend_lhs = explicit_unsigned && !both_nonnegative && lhs_width < output_width;
    let extension = if extend_lhs {
        format!(
            "  extended: bits[{output_width}] = sign_ext({lhs}, new_bit_count={output_width}, id=6)\n"
        )
    } else {
        String::new()
    };
    let lhs = if extend_lhs { "extended" } else { lhs };
    let operands = if commuted {
        format!("positive_rhs, {lhs}")
    } else {
        format!("{lhs}, positive_rhs")
    };
    let operation = if explicit_unsigned { "umul" } else { "smul" };
    format!(
        r#"package sample
 top fn main(a: bits[{signed_width}] id=1, b: bits[{positive_width}] id=2, c: bits[{output_width}] id=3) -> bits[{output_width}] {{
  positive_rhs: bits[{rhs_width}] = zero_ext(b, new_bit_count={rhs_width}, id=4)
{lhs_definition}{extension}  product: bits[{output_width}] = {operation}({operands}, id=7)
  ret result: bits[{output_width}] = add(product, c, id=8)
 }}
"#
    )
}
