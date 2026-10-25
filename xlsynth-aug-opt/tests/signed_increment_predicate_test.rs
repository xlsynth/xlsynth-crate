// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_g8r::check_equivalence::check_equivalence;
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{IrBits, IrValue};

/// Builds the predicate with optional independent numeric use of the sum.
fn sample(width: usize, swapped: bool, shared: bool) -> String {
    let args = if swapped { "one, x" } else { "x, one" };
    let (ty, ret) = if shared {
        (
            format!("(bits[1], bits[{width}])"),
            format!(
                "pred: bits[1] = sgt(sum, zero, id=5)\n  ret pair: (bits[1], bits[{width}]) = tuple(pred, sum, id=6)"
            ),
        )
    } else {
        (
            "bits[1]".into(),
            "ret pred: bits[1] = sgt(sum, zero, id=5)".into(),
        )
    };
    format!(
        r#"package increment

top fn main(x: bits[{width}] id=1) -> {ty} {{
  one: bits[{width}] = literal(value=1, id=2)
  zero: bits[{width}] = literal(value=0, id=3)
  sum: bits[{width}] = add({args}, id=4)
  {ret}
}}
"#
    )
}

/// Runs only PIR rewrites so upstream optimization cannot conceal a missed
/// match.
fn optimize(text: &str, mode: AugOptMode) -> xlsynth_aug_opt::AugOptRunResult {
    run_aug_opt_over_ir_text_with_stats(
        text,
        Some("main"),
        AugOptOptions {
            enable: true,
            rounds: 1,
            mode,
            ..Default::default()
        },
    )
    .expect("aug-opt")
}

/// Parses a complete output for independent interpreter checks.
fn parse(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .expect("valid IR")
        .get_top_fn()
        .expect("top")
        .clone()
}

/// Extracts a successful interpreter result without hiding evaluation failures.
fn evaluate(f: &ir::Fn, x: &IrValue) -> IrValue {
    match eval_fn(f, std::slice::from_ref(x)) {
        FnEvalResult::Success(result) => result.value,
        result => panic!("evaluation failed: {result:?}"),
    }
}

#[test]
fn signed_increment_widths_spellings_and_shared_consumers() {
    for width in [1, 2, 3, 8, 33, 65, 129] {
        for swapped in [false, true] {
            for shared in [false, true] {
                let text = sample(width, swapped, shared);
                for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
                    let out = optimize(&text, mode);
                    if mode == AugOptMode::PirOnly {
                        assert_eq!(out.rewrite_stats.signed_increment_positive, 1);
                    }
                    // The sandwich's initial XLS pass may already fold tiny
                    // widths.
                    #[cfg(feature = "has-bitwuzla")]
                    check_equivalence(&text, &out.output_text).expect("all-input equivalence");
                    let before = parse(&text);
                    let after = parse(&out.output_text);
                    let mut values = vec![
                        IrBits::zero(width),
                        IrBits::all_ones(width),
                        IrBits::from_lsb_fn(width, |i| i == width - 1),
                        IrBits::from_lsb_fn(width, |i| i < width - 1),
                        IrBits::from_lsb_fn(width, |i| i > 0 && i < width - 1),
                    ];
                    if width <= 3 {
                        values.extend(
                            (0..1u64 << width).map(|x| IrBits::make_ubits(width, x).unwrap()),
                        );
                    }
                    for bits in values {
                        let x = IrValue::from_bits(&bits);
                        assert_eq!(
                            evaluate(&before, &x),
                            evaluate(&after, &x),
                            "width={width} swapped={swapped} shared={shared} mode={mode:?} x={x}"
                        );
                        if !shared {
                            let sign = bits.get_bit(width - 1).unwrap();
                            let max_positive = (0..width - 1).all(|i| bits.get_bit(i).unwrap());
                            assert_eq!(evaluate(&after, &x), IrValue::bool(!sign && !max_positive));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn signed_increment_does_not_match_other_arithmetic_or_predicates() {
    let base = sample(65, false, false);
    for text in [
        base.replace("value=1,", "value=2,"),
        base.replace("value=0,", "value=1,"),
        base.replace("sgt(sum, zero", "ugt(sum, zero"),
        base.replace("sgt(sum, zero", "sge(sum, zero"),
        base.replace("sgt(sum, zero", "sgt(zero, sum"),
        base.replace("add(x, one", "sub(x, one"),
        base.replace("add(x, one", "add(x, x"),
        base.replace("value=1,", "value=0x10000000000000001,"),
    ] {
        let out = optimize(&text, AugOptMode::PirOnly);
        assert_eq!(out.rewrite_stats.signed_increment_positive, 0, "{text}");
        #[cfg(feature = "has-bitwuzla")]
        check_equivalence(&text, &out.output_text).expect("negative-case equivalence");
    }
}

/// Collects reachable emission statistics without invoking any AIG optimizer.
fn emission(text: &str) -> (usize, usize) {
    let out = ir2gates_from_ir_text(text, Some("main"), Ir2GatesOptions::all_opts_disabled())
        .expect("emit");
    let stats = get_aig_stats(&out.gatify_output.gate_fn);
    (stats.and_nodes, stats.max_depth)
}

#[test]
fn signed_increment_improves_immediate_emission() {
    for width in [1, 2, 3, 8, 33, 65, 129] {
        for shared in [false, true] {
            let text = sample(width, false, shared);
            let out = optimize(&text, AugOptMode::PirOnly);
            let before = emission(&text);
            let after = emission(&out.output_text);
            eprintln!("width={width} shared={shared} emission {before:?} -> {after:?}");
            assert!(
                after.0 <= before.0,
                "area regression width={width} shared={shared}"
            );
            assert!(
                after.1 <= before.1,
                "depth regression width={width} shared={shared}"
            );
            if width == 33 && !shared {
                assert_eq!(before, (186, 17));
                assert_eq!(after, (32, 6));
            }
        }
    }
}
