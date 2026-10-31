// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use test_case::test_matrix;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir_parser::Parser;
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
