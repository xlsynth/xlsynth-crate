// SPDX-License-Identifier: Apache-2.0

//! Targeted survival-predicate proofs with explicit firing and decline checks.
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir::{Binop, NodePayload};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult, ProverFn};
use xlsynth_prover::solver::bitwuzla::Bitwuzla;

pub const KINDS: &[&str] = &[
    "reduce",
    "eq",
    "ne",
    "window",
    "select",
    "priority",
    "shared_shift",
    "shared_window",
    "shared_select",
    "numeric",
    "arithmetic_right",
    "wide_amount",
    "wide_data",
    "nonzero_constant",
    "nested_predicate",
    "independent",
    "wrapping_amount",
    "shared_predicate",
    "dead_user",
    "traversal_budget",
    "zero_width_amount",
    "constructed_amount",
    "term_budget",
    "wide_onehot",
    "admission_budget",
];

/// Builds valid symbolic IR. Expected decisions follow deliberately separated
/// positive and negative shapes, rather than reimplementing the pass's cost
/// model.
pub fn check_case(data: [u8; 4]) -> bool {
    let kind = KINDS[usize::from(data[0]) % KINDS.len()];
    let mut n = [1, 2, 3, 7, 8, 23, 32, 65, 129, 256][usize::from(data[1]) % 10];
    if kind == "window" {
        n = n.max(32);
    }
    if kind == "wide_data" {
        n = 257;
    }
    if kind == "term_budget" {
        n = 32;
    }
    let wide = [
        "wide_amount",
        "term_budget",
        "wide_onehot",
        "admission_budget",
    ]
    .contains(&kind)
        || data[2] & 2 != 0;
    let mut aw = ceil_log2(n).clamp(1, 5);
    if kind == "window" {
        aw = 3;
    }
    if wide {
        aw = [6, 9, 32, 65, 129][usize::from(data[3]) % 5];
    }
    if kind == "zero_width_amount" {
        aw = 0;
    }
    let op = if kind == "arithmetic_right" {
        "shra"
    } else if data[2] & 1 == 0 {
        "shll"
    } else {
        "shrl"
    };
    let (start, width) = if kind == "window" {
        if wide {
            (usize::from(data[3]) % 2, n - 1)
        } else {
            (usize::from(data[3]) % (n - 22), 23)
        }
    } else {
        (0, n)
    };
    let mut body = String::new();
    let amount = if kind == "wrapping_amount" {
        body.push_str(&format!("ones: bits[{aw}] = literal(value=0b{}, id=5)\namount: bits[{aw}] = add(k, ones, id=6)\n", "1".repeat(aw)));
        "amount"
    } else if kind == "constructed_amount" && aw > 1 {
        let kept = aw - 1;
        body.push_str(&format!("kept: bits[{kept}] = bit_slice(k, start=0, width={kept}, id=5)\nlow_zero: bits[1] = literal(value=0, id=6)\namount: bits[{aw}] = concat(kept, low_zero, id=21)\n"));
        "amount"
    } else {
        "k"
    };
    let budget_amount = if kind == "admission_budget" {
        let mut previous = amount.to_owned();
        for i in 0..33 {
            let next = format!("amount_{i}");
            body.push_str(&format!(
                "{next}: bits[{aw}] = identity({previous}, id={})\n",
                300 + i
            ));
            previous = next;
        }
        Some(previous)
    } else {
        None
    };
    let amount = budget_amount.as_deref().unwrap_or(amount);
    let shift_data = if kind == "wide_onehot" {
        let input_width = n - 1;
        body.push_str(&format!("onehot_input: bits[{input_width}] = bit_slice(x, start=0, width={input_width}, id=24)\nonehot: bits[{n}] = one_hot(onehot_input, lsb_prio=true, id=25)\n"));
        match data[3] % 5 {
            0 => "onehot",
            1 => {
                body.push_str(&format!(
                    "wrapped: bits[{n}] = bit_slice(onehot, start=0, width={n}, id=26)\n"
                ));
                "wrapped"
            }
            2 => {
                body.push_str(&format!("wrapped: bits[{n}] = identity(onehot, id=26)\n"));
                "wrapped"
            }
            3 => {
                body.push_str(&format!("inverted: bits[{n}] = not(onehot, id=26)\nwrapped: bits[{n}] = not(inverted, id=27)\n"));
                "wrapped"
            }
            _ => {
                body.push_str(&format!("empty: bits[0] = literal(value=0, id=26)\nwrapped: bits[{n}] = concat(empty, onehot, id=27)\n"));
                "wrapped"
            }
        }
    } else {
        "x"
    };
    body.push_str(&format!(
        "shifted: bits[{n}] = {op}({shift_data}, {amount}, id=7)\n"
    ));
    let mut chosen = "shifted".to_string();
    if ["select", "priority", "shared_select"].contains(&kind) {
        let expr = if kind == "priority" {
            "priority_sel(s, cases=[shifted], default=y, id=8)"
        } else {
            "sel(s, cases=[y, shifted], id=8)"
        };
        body.push_str(&format!("chosen: bits[{n}] = {expr}\n"));
        chosen = "chosen".into();
    }
    if kind == "term_budget" {
        // Three 32-group shifts exceed the 64-group budget: the first two
        // scalarize and the third must remain a numeric shift.
        body.push_str(&format!("second: bits[{n}] = {op}(y, k, id=12)\nthird: bits[{n}] = {op}(z, k, id=13)\ninner: bits[{n}] = sel(t, cases=[second, third], id=14)\nchosen: bits[{n}] = sel(s, cases=[shifted, inner], id=15)\n"));
        chosen = "chosen".into();
    }
    if kind == "traversal_budget" {
        for i in 0..17 {
            let next = format!("choice_{i}");
            body.push_str(&format!(
                "{next}: bits[{n}] = sel(s, cases=[{chosen}, y], id={})\n",
                100 + i
            ));
            chosen = next;
        }
    }
    body.push_str(&format!(
        "window: bits[{width}] = bit_slice({chosen}, start={start}, width={width}, id=9)\n"
    ));
    let constant = usize::from(kind == "nonzero_constant");
    body.push_str(&format!(
        "zero: bits[{width}] = literal(value={constant}, id=10)\n"
    ));
    let predicate = match kind {
        "eq" => "eq(zero, window, id=11)",
        "ne" | "nonzero_constant" => "ne(window, zero, id=11)",
        _ => "or_reduce(window, id=11)",
    };
    body.push_str(&format!("pred: bits[1] = {predicate}\n"));
    let mut expected = n > 1
        && ![
            "shared_shift",
            "shared_window",
            "shared_select",
            "numeric",
            "arithmetic_right",
            "wide_onehot",
            "admission_budget",
            "wide_data",
            "nonzero_constant",
            "traversal_budget",
            "zero_width_amount",
        ]
        .contains(&kind);
    let (ret_type, ret) = match kind {
        "shared_shift" => (
            format!("(bits[1], bits[{n}])"),
            "tuple(pred, shifted, id=20)".to_string(),
        ),
        "shared_window" => (
            format!("(bits[1], bits[{width}])"),
            "tuple(pred, window, id=20)".to_string(),
        ),
        "shared_select" => (
            format!("(bits[1], bits[{n}])"),
            "tuple(pred, chosen, id=20)".to_string(),
        ),
        "shared_predicate" => (
            "(bits[1], bits[1])".into(),
            "tuple(pred, pred, id=20)".into(),
        ),
        "numeric" => (format!("bits[{n}]"), "identity(shifted, id=20)".into()),
        "nested_predicate" => {
            body.push_str("false_value: bits[1] = literal(value=0, id=12)\n");
            ("bits[1]".into(), "eq(pred, false_value, id=20)".into())
        }
        "independent" => {
            body.push_str(&format!("other: bits[{n}] = {op}(y, k, id=12)\nother_pred: bits[1] = or_reduce(other, id=13)\n"));
            (
                "(bits[1], bits[1])".into(),
                "tuple(pred, other_pred, id=20)".into(),
            )
        }
        "dead_user" => {
            body.push_str(&format!("dead: bits[{n}] = not(shifted, id=12)\n"));
            ("bits[1]".into(), "identity(pred, id=20)".into())
        }
        _ => ("bits[1]".into(), "identity(pred, id=20)".into()),
    };
    // Width one does not justify scalar expansion; selectors cannot change
    // this.
    expected &= n > 1 && (!wide || (n <= 32 && !["window", "constructed_amount"].contains(&kind)));
    let extra_params = if kind == "term_budget" {
        format!(", z: bits[{n}] id=22, t: bits[1] id=23")
    } else {
        String::new()
    };
    let text = format!(
        "package survival\ntop fn main(x: bits[{n}] id=1, y: bits[{n}] id=2, k: bits[{aw}] id=3, s: bits[1] id=4{extra_params}) -> {ret_type} {{\n{body}ret out: {ret_type} = {ret}\n}}\n"
    );
    let pkg = Parser::new(&text)
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("{text}\n{e}"));
    let original = pkg.get_top_fn().unwrap();
    for options in [
        PrepForGatifyOptions::all_opts_disabled(),
        PrepForGatifyOptions::all_opts_enabled(),
    ] {
        let prepared = prep_for_gatify(original, None, options);
        let remaining = prepared
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.payload,
                    NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, _, _)
                )
            })
            .count();
        let original_shifts = if kind == "independent" { 2 } else { 1 };
        let expected_remaining = if kind == "term_budget" {
            1
        } else if expected {
            0
        } else {
            original_shifts
        };
        assert_eq!(
            remaining, expected_remaining,
            "decision {kind} {data:?}\n{text}\n{prepared}"
        );
        let result = prove_ir_fn_equiv::<Bitwuzla>(
            &crate::fuzz_bitwuzla_options(),
            &ProverFn::new(original, None),
            &ProverFn::new(&prepared, None),
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(
            matches!(result, EquivResult::Proved),
            "{kind} {data:?}: {result:?}\n{text}"
        );
    }
    expected
}

/// Covers every category and direction before coverage-guided mutations begin.
pub fn coverage_audit() -> (usize, usize) {
    let mut accepted = 0;
    let mut rejected = 0;
    for kind in 0..KINDS.len() {
        for width in 0..10 {
            for direction in 0..4 {
                if check_case([kind as u8, width, direction, 17]) {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }
        }
    }
    let onehot = KINDS
        .iter()
        .position(|kind| *kind == "wide_onehot")
        .unwrap();
    for wrapper in 0..5 {
        for direction in 0..2 {
            assert!(!check_case([onehot as u8, 5, direction, wrapper]));
            rejected += 1;
        }
    }
    (accepted, rejected)
}

#[cfg(test)]
mod tests {
    #[test]
    fn shift_survival_target_covers_firing_rejection_and_equivalence() {
        let (accepted, rejected) = super::coverage_audit();
        assert!(accepted > 100 && rejected > 100, "{accepted} {rejected}");
        assert_eq!(accepted + rejected, super::KINDS.len() * 10 * 4 + 10);
    }
}
