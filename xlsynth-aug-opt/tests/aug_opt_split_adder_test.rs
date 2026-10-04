// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{
    AugOptMode, AugOptOptions, AugOptRunResult, run_aug_opt_over_ir_text_with_stats,
};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_g8r::check_equivalence::check_equivalence;
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{IrBits, IrValue};

/// Varies the representation of one split addition without changing its value.
#[derive(Clone, Copy, Debug)]
struct Spelling {
    upper_order: [usize; 3],
    right_associated: bool,
    swap_carry: bool,
    swap_xor: bool,
    zero_concat: bool,
    permuted_ids: bool,
    same_operand: bool,
}

impl Default for Spelling {
    fn default() -> Self {
        Self {
            upper_order: [0, 1, 2],
            right_associated: false,
            swap_carry: false,
            swap_xor: false,
            zero_concat: false,
            permuted_ids: false,
            same_operand: false,
        }
    }
}

/// Chooses a use of the sum and any intermediate that remains observable.
#[derive(Clone, Copy, Debug)]
enum Consumer {
    Sum,
    Masked,
    SharedCarry,
    SharedXor,
    SharedUpper,
    SharedPartialUpper,
}

/// Builds `(x[1:N] + y[1:N] + zext(x[0+:u1] & y[0+:u1])) ++
/// (x[0+:u1] ^ y[0+:u1])`, with every upper addition at width `N - 1`.
fn split_adder_ir(width: usize, spelling: Spelling, consumer: Consumer) -> String {
    assert!(width >= 2);
    let upper = width - 1;
    let id = |index: usize| {
        if spelling.permuted_ids {
            1000 - 37 * index
        } else {
            index
        }
    };
    let second = if spelling.same_operand { "x" } else { "y" };
    let mut slices = vec![
        format!(
            "xlo: bits[1] = bit_slice(x, start=0, width=1, id={})",
            id(4)
        ),
        format!(
            "ylo: bits[1] = bit_slice({second}, start=0, width=1, id={})",
            id(5)
        ),
        format!(
            "xhi: bits[{upper}] = bit_slice(x, start=1, width={upper}, id={})",
            id(6)
        ),
        format!(
            "yhi: bits[{upper}] = bit_slice({second}, start=1, width={upper}, id={})",
            id(7)
        ),
    ];
    if spelling.permuted_ids {
        slices.reverse();
    }
    let carry_args = if spelling.swap_carry {
        "ylo, xlo"
    } else {
        "xlo, ylo"
    };
    let xor_args = if spelling.swap_xor {
        "ylo, xlo"
    } else {
        "xlo, ylo"
    };
    let (carry_text, carry_term) = if width == 2 {
        (String::new(), "carry")
    } else if spelling.zero_concat {
        (
            format!(
                "padding: bits[{}] = literal(value=0, id={})\n  carry_wide: bits[{upper}] = concat(padding, carry, id={})",
                width - 2,
                id(9),
                id(10)
            ),
            "carry_wide",
        )
    } else {
        (
            format!(
                "carry_wide: bits[{upper}] = zero_ext(carry, new_bit_count={upper}, id={})",
                id(10)
            ),
            "carry_wide",
        )
    };
    let terms = ["xhi", "yhi", carry_term];
    let [a, b, c] = spelling.upper_order.map(|index| terms[index]);
    let (partial_args, upper_args) = if spelling.right_associated {
        (format!("{b}, {c}"), format!("{a}, partial"))
    } else {
        (format!("{a}, {b}"), format!("partial, {c}"))
    };
    let (return_type, consumer_text) = match consumer {
        Consumer::Sum => (
            format!("bits[{width}]"),
            format!("ret result: bits[{width}] = identity(sum, id={})", id(16)),
        ),
        Consumer::Masked => (
            format!("bits[{width}]"),
            format!(
                "mask: bits[{width}] = sign_ext(valid, new_bit_count={width}, id={})\n  ret result: bits[{width}] = and(sum, mask, id={})",
                id(15),
                id(16)
            ),
        ),
        Consumer::SharedCarry
        | Consumer::SharedXor
        | Consumer::SharedUpper
        | Consumer::SharedPartialUpper => {
            let (shared, shared_width) = match consumer {
                Consumer::SharedCarry => ("carry", 1),
                Consumer::SharedXor => ("low", 1),
                Consumer::SharedUpper => ("high", upper),
                Consumer::SharedPartialUpper => ("partial", upper),
                _ => unreachable!("the outer match selects a shared intermediate"),
            };
            let ty = format!("(bits[{width}], bits[{shared_width}])");
            (
                ty.clone(),
                format!("ret result: {ty} = tuple(sum, {shared}, id={})", id(16)),
            )
        }
    };
    format!(
        r#"package split_adder

top fn main(x: bits[{width}] id={x_id}, y: bits[{width}] id={y_id}, valid: bits[1] id={valid_id}) -> {return_type} {{
  {slices}
  carry: bits[1] = and({carry_args}, id={carry_id})
  {carry_text}
  partial: bits[{upper}] = add({partial_args}, id={partial_id})
  high: bits[{upper}] = add({upper_args}, id={high_id})
  low: bits[1] = xor({xor_args}, id={low_id})
  sum: bits[{width}] = concat(high, low, id={sum_id})
  {consumer_text}
}}
"#,
        x_id = id(1),
        y_id = id(2),
        valid_id = id(3),
        slices = slices.join("\n  "),
        carry_id = id(8),
        partial_id = id(11),
        high_id = id(12),
        low_id = id(13),
        sum_id = id(14),
    )
}

/// Keeps the two same-build runs available for independent value checks.
struct OptimizationPair {
    disabled: AugOptRunResult,
    enabled: AugOptRunResult,
}

/// Runs both flag settings and proves them when Bitwuzla support is enabled.
fn optimize_and_prove(text: &str, mode: AugOptMode) -> Result<OptimizationPair, String> {
    let options = AugOptOptions {
        enable: true,
        rounds: 1,
        mode,
        recover_split_adders: false,
    };
    let disabled = run_aug_opt_over_ir_text_with_stats(text, Some("main"), options)?;
    let enabled = run_aug_opt_over_ir_text_with_stats(
        text,
        Some("main"),
        AugOptOptions {
            recover_split_adders: true,
            ..options
        },
    )?;
    // Default builds retain recognition and interpreter coverage. Solver builds
    // additionally prove each rewrite for every input, including wide values.
    #[cfg(feature = "has-bitwuzla")]
    {
        check_equivalence(text, &disabled.output_text)?;
        check_equivalence(text, &enabled.output_text)?;
    }
    assert_eq!(disabled.rewrite_stats.split_adders_recovered, 0);
    Ok(OptimizationPair { disabled, enabled })
}

/// Parses and verifies the emitted package before evaluating its top function.
fn parse_main(text: &str) -> Result<ir::Fn, String> {
    let package = Parser::new(text)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    package
        .get_top_fn()
        .cloned()
        .ok_or_else(|| "missing top function".to_string())
}

/// Preserves interpreter failures while extracting a successful return value.
fn evaluate(function: &ir::Fn, args: &[IrValue]) -> Result<IrValue, String> {
    match eval_fn(function, args) {
        FnEvalResult::Success(result) => Ok(result.value),
        result => Err(format!("unexpected evaluation failure: {result:?}")),
    }
}

#[test]
fn width_sweep_preserves_wrapping_addition_in_both_modes() {
    assert!(AugOptOptions::default().recover_split_adders);
    for width in [2, 3, 5, 8, 16, 32, 65, 129] {
        let source = split_adder_ir(width, Spelling::default(), Consumer::Sum);
        for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
            let pair = optimize_and_prove(&source, mode).unwrap();
            // XLS can fold the one-bit upper additions before our rewrite runs.
            if mode == AugOptMode::PirOnly || width > 2 {
                assert_eq!(
                    pair.enabled.rewrite_stats.split_adders_recovered, 1,
                    "width={width}, mode={mode:?}"
                );
            }
            let functions = [
                parse_main(&source).unwrap(),
                parse_main(&pair.disabled.output_text).unwrap(),
                parse_main(&pair.enabled.output_text).unwrap(),
            ];
            if width <= 5 {
                let limit = 1u64 << width;
                for x in 0..limit {
                    for y in 0..limit {
                        let args = [
                            IrValue::make_ubits(width, x).unwrap(),
                            IrValue::make_ubits(width, y).unwrap(),
                            IrValue::bool(true),
                        ];
                        let expected = IrValue::make_ubits(width, (x + y) & (limit - 1)).unwrap();
                        for function in &functions {
                            assert_eq!(
                                evaluate(function, &args).unwrap(),
                                expected,
                                "width={width}, mode={mode:?}, x={x}, y={y}"
                            );
                        }
                    }
                }
            }
            let args = [
                IrValue::from_bits(&IrBits::all_ones(width)),
                IrValue::make_ubits(width, 1).unwrap(),
                IrValue::bool(true),
            ];
            let zero = IrValue::from_bits(&IrBits::zero(width));
            for function in &functions {
                assert_eq!(
                    evaluate(function, &args).unwrap(),
                    zero,
                    "carry must wrap at width {width}"
                );
            }
        }
    }
}

#[test]
fn recovered_adders_are_stable_across_optimizer_rounds() {
    for width in [2, 3, 5, 8, 16, 32, 65, 129] {
        for consumer in [Consumer::Sum, Consumer::Masked, Consumer::SharedCarry] {
            let source = split_adder_ir(width, Spelling::default(), consumer);
            for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
                let options = AugOptOptions {
                    enable: true,
                    mode,
                    ..Default::default()
                };
                let first =
                    run_aug_opt_over_ir_text_with_stats(&source, Some("main"), options).unwrap();
                // At width two, XLS may simplify the one-bit upper arithmetic
                // before recovery. Wider unconstrained sums should stay adds.
                if mode == AugOptMode::PirOnly || width > 2 {
                    assert_eq!(first.rewrite_stats.split_adders_recovered, 1);
                    let function = parse_main(&first.output_text).unwrap();
                    assert!(function.nodes.iter().any(|node| {
                        node.ty == ir::Type::Bits(width)
                            && matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Add, ..))
                    }));
                }
                // Check the whole pipeline, including the trailing libxls pass.
                // Equal final text alone could hide splitting and recovering
                // the same addition again on every round.
                let multiple = run_aug_opt_over_ir_text_with_stats(
                    &source,
                    Some("main"),
                    AugOptOptions {
                        rounds: 3,
                        ..options
                    },
                )
                .unwrap();
                assert_eq!(
                    multiple.rewrite_stats.split_adders_recovered,
                    first.rewrite_stats.split_adders_recovered,
                    "width={width}, consumer={consumer:?}, mode={mode:?}"
                );
                assert_eq!(
                    multiple.output_text, first.output_text,
                    "width={width}, consumer={consumer:?}, mode={mode:?}"
                );
                // Re-entering also reruns the sandwich's initial libxls pass.
                let again =
                    run_aug_opt_over_ir_text_with_stats(&first.output_text, Some("main"), options)
                        .unwrap();
                assert_eq!(
                    again.rewrite_stats.split_adders_recovered, 0,
                    "width={width}, consumer={consumer:?}, mode={mode:?}"
                );
                assert_eq!(
                    again.output_text, first.output_text,
                    "width={width}, consumer={consumer:?}, mode={mode:?}"
                );
            }
        }
    }
}

#[test]
fn commutation_reassociation_and_id_order_preserve_recognition() {
    for upper_order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for right_associated in [false, true] {
            for swaps in 0..4 {
                let spelling = Spelling {
                    upper_order,
                    right_associated,
                    swap_carry: swaps & 1 != 0,
                    swap_xor: swaps & 2 != 0,
                    zero_concat: right_associated,
                    permuted_ids: swaps == 3,
                    ..Spelling::default()
                };
                let source = split_adder_ir(5, spelling, Consumer::Sum);
                let pair = optimize_and_prove(&source, AugOptMode::PirOnly).unwrap();
                assert_eq!(
                    pair.enabled.rewrite_stats.split_adders_recovered, 1,
                    "{spelling:?}"
                );
            }
        }
    }
}

#[test]
fn validity_mask_and_identical_operands_preserve_their_semantics() {
    for same_operand in [false, true] {
        let spelling = Spelling {
            same_operand,
            ..Spelling::default()
        };
        let source = split_adder_ir(5, spelling, Consumer::Masked);
        for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
            let pair = optimize_and_prove(&source, mode).unwrap();
            // XLS folds x + x before recovery in the sandwich's initial pass.
            if mode == AugOptMode::PirOnly || !same_operand {
                assert_eq!(pair.enabled.rewrite_stats.split_adders_recovered, 1);
            }
            let functions = [
                parse_main(&pair.disabled.output_text).unwrap(),
                parse_main(&pair.enabled.output_text).unwrap(),
            ];
            for x in 0..32 {
                for y in 0..32 {
                    for valid in [false, true] {
                        let args = [
                            IrValue::make_ubits(5, x).unwrap(),
                            IrValue::make_ubits(5, y).unwrap(),
                            IrValue::bool(valid),
                        ];
                        let expected = if valid {
                            (x + if same_operand { x } else { y }) & 31
                        } else {
                            0
                        };
                        let expected = IrValue::make_ubits(5, expected).unwrap();
                        for function in &functions {
                            assert_eq!(
                                evaluate(function, &args).unwrap(),
                                expected,
                                "same_operand={same_operand}, mode={mode:?}, x={x}, y={y}, valid={valid}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shared_intermediates_keep_their_own_values() {
    for consumer in [
        Consumer::SharedCarry,
        Consumer::SharedXor,
        Consumer::SharedUpper,
        Consumer::SharedPartialUpper,
    ] {
        let source = split_adder_ir(5, Spelling::default(), consumer);
        let pair = optimize_and_prove(&source, AugOptMode::PirOnly).unwrap();
        // Carry sharing is harmless. Shared upper arithmetic or the low XOR
        // is deliberately rejected by the initial conservative recognizer.
        let expected = usize::from(matches!(consumer, Consumer::SharedCarry));
        assert_eq!(
            pair.enabled.rewrite_stats.split_adders_recovered, expected,
            "{consumer:?}"
        );
    }
}
