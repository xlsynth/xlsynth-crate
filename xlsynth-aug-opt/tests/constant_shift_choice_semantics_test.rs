// SPDX-License-Identifier: Apache-2.0

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_aug_opt::constant_shift_choices::{
    ConstantShiftChoiceLimits, constant_shift_choice_candidate,
};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::ir::{self, Binop, NodePayload, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_utils::{find_node_by_name, operands};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

#[derive(Clone, Copy, Debug)]
enum Direction {
    Left,
    Right,
}

impl Direction {
    /// Emits the selected logical shift, preserving builder validation errors.
    fn emit(
        self,
        builder: &mut FnBuilder,
        data: BValue,
        amount: BValue,
    ) -> Result<BValue, BuilderError> {
        match self {
            Self::Left => builder.shll(data, amount),
            Self::Right => builder.shrl(data, amount),
        }
    }
}

fn evaluate(function: &ir::Fn, args: &[IrValue]) -> IrValue {
    match eval_fn(function, args) {
        FnEvalResult::Success(result) => result.value,
        result => panic!("unexpected evaluation failure: {result:?}"),
    }
}

/// Computes a logical shift by projecting bits, independently of PIR
/// evaluation.
fn shift_oracle(data: &IrValue, amount: usize, direction: Direction) -> IrValue {
    let data = data.to_bits().unwrap();
    let width = data.get_bit_count();
    let projected: Vec<bool> = (0..width)
        .map(|output_bit| {
            let input_bit = match direction {
                Direction::Left => output_bit.checked_sub(amount),
                Direction::Right => output_bit.checked_add(amount),
            };
            input_bit.is_some_and(|index| index < width && data.get_bit(index).unwrap())
        })
        .collect();
    IrValue::from_bits(&IrBits::from_lsb_is_0(&projected))
}

fn logical_shift_count(function: &ir::Fn) -> usize {
    function
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.payload,
                NodePayload::Binop(Binop::Shll | Binop::Shrl, _, _)
            )
        })
        .count()
}

/// Proves both the rewrite and the projection lowering against the source IR.
fn prove_candidate(source: &ir::Fn, candidate: &ir::Fn) {
    check_equivalence::check_equivalence_via_toolchain(
        &format!("package source\n\ntop {source}"),
        &format!("package candidate\n\ntop {candidate}"),
    )
    .expect("choice fusion must preserve the source function");
    let mapped = gatify_prepared_fn(candidate, GatifyOptions::all_opts_disabled()).unwrap();
    check_equivalence::validate_same_fn_via_toolchain(source, &mapped.gate_fn)
        .expect("mapped projections must preserve the source function");
}

/// Builds the generic masked-choice regression fixture from the plan.
fn masked_choice_fixture(direction: Direction, swapped_mask: bool) -> ir::Fn {
    let mut builder = FnBuilder::new("masked_choice");
    let data = builder.param("x", Type::Bits(4)).unwrap();
    let en = builder.param("en", Type::Bits(1)).unwrap();
    let p = builder.param("p", Type::Bits(1)).unwrap();
    let q = builder.param("q", Type::Bits(1)).unwrap();
    let prefix = builder.literal(IrValue::make_ubits(1, 1).unwrap()).unwrap();
    let pair = builder.concat(&[prefix, q]).unwrap();
    let one = builder.literal(IrValue::make_ubits(2, 1).unwrap()).unwrap();
    let chosen = builder.select(p, &[pair, one], None).unwrap();
    let mask = builder.sign_extend(en, 2).unwrap();
    let amount = if swapped_mask {
        builder.and(mask, chosen).unwrap()
    } else {
        builder.and(chosen, mask).unwrap()
    };
    let result = direction.emit(&mut builder, data, amount).unwrap();
    builder.build(result).unwrap()
}

#[test]
fn four_bit_masked_choices_match_exhaustive_independent_oracle() {
    for direction in [Direction::Left, Direction::Right] {
        for swapped_mask in [false, true] {
            let source = masked_choice_fixture(direction, swapped_mask);
            let candidate =
                constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
                    .expect("the mask and literal-prefix choices must be recognized");
            assert_eq!(logical_shift_count(&candidate), 0);
            prove_candidate(&source, &candidate);

            for x in 0..16 {
                for controls in 0..8 {
                    let en = controls & 1 != 0;
                    let p = controls & 2 != 0;
                    let q = controls & 4 != 0;
                    let amount = if !en {
                        0
                    } else if p {
                        1
                    } else if q {
                        3
                    } else {
                        2
                    };
                    let args = [
                        IrValue::make_ubits(4, x).unwrap(),
                        IrValue::bool(en),
                        IrValue::bool(p),
                        IrValue::bool(q),
                    ];
                    let expected = shift_oracle(&args[0], amount, direction);
                    assert_eq!(evaluate(&source, &args), expected);
                    assert_eq!(evaluate(&candidate, &args), expected);
                }
            }
        }
    }
}

#[test]
fn wide_amounts_and_non_power_of_two_data_preserve_saturation() {
    for width in [1, 3, 5, 65, 129] {
        for direction in [Direction::Left, Direction::Right] {
            let mut builder = FnBuilder::new("wide_amount");
            let data = builder.param("x", Type::Bits(width)).unwrap();
            let selector = builder.param("selector", Type::Bits(3)).unwrap();
            let amounts = [0, width - 1, width, width + 3];
            let cases: Vec<_> = amounts
                .iter()
                .map(|&amount| {
                    builder
                        .literal(IrValue::make_ubits(130, amount as u64).unwrap())
                        .unwrap()
                })
                .collect();
            let mut huge_bits = vec![false; 130];
            huge_bits[129] = true;
            huge_bits[0] = true;
            let huge = builder
                .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&huge_bits)))
                .unwrap();
            let amount = builder.select(selector, &cases, Some(huge)).unwrap();
            let result = direction.emit(&mut builder, data, amount).unwrap();
            let source = builder.build(result).unwrap();
            let candidate =
                constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
                    .expect("oversized amounts are one effective zero projection");
            assert_eq!(logical_shift_count(&candidate), 0);
            prove_candidate(&source, &candidate);

            let patterns = [
                IrValue::from_bits(&IrBits::all_ones(width)),
                IrValue::make_ubits(width, 1).unwrap(),
                IrValue::from_bits(&IrBits::from_lsb_is_0(
                    &(0..width).map(|index| index % 3 == 0).collect::<Vec<_>>(),
                )),
            ];
            for data in patterns {
                for selector in 0..8 {
                    let expected_amount = amounts.get(selector).copied().unwrap_or(width);
                    let expected = shift_oracle(&data, expected_amount, direction);
                    let args = [
                        data.clone(),
                        IrValue::make_ubits(3, selector as u64).unwrap(),
                    ];
                    assert_eq!(evaluate(&source, &args), expected);
                    assert_eq!(evaluate(&candidate, &args), expected);
                }
            }
        }
    }
}

#[test]
fn right_shift_slice_preserves_padding_and_a_live_full_shift() {
    let mut builder = FnBuilder::new("right_shift_slice");
    let data = builder.param("x", Type::Bits(5)).unwrap();
    let selector = builder.param("selector", Type::Bits(2)).unwrap();
    let amounts = [0, 2, 4, 7];
    let cases: Vec<_> = amounts
        .iter()
        .map(|&amount| {
            builder
                .literal(IrValue::make_ubits(3, amount as u64).unwrap())
                .unwrap()
        })
        .collect();
    let amount = builder.select(selector, &cases, None).unwrap();
    let shifted = builder.shrl(data, amount).unwrap();
    let sliced = builder.bit_slice(shifted, 1, 3).unwrap();
    let result = builder.tuple(&[sliced, shifted]).unwrap();
    let source = builder.build(result).unwrap();
    let candidate = constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
        .expect("the slice and its independently live shift must both be fused");
    assert_eq!(logical_shift_count(&candidate), 0);
    let NodePayload::Tuple(elements) = &candidate.get_node(candidate.ret_node_ref.unwrap()).payload
    else {
        panic!("both results must remain live");
    };
    for &element in elements {
        assert!(matches!(
            candidate.get_node(element).payload,
            NodePayload::Sel { .. }
        ));
    }
    prove_candidate(&source, &candidate);

    for x in 0u64..32 {
        for (selector, &amount) in amounts.iter().enumerate() {
            // Shift 2 partially pads the slice, shift 4 zeros only the slice,
            // and shift 7 zeros both independently observable results.
            let full = x >> amount;
            let expected = IrValue::make_tuple(&[
                IrValue::make_ubits(3, (full >> 1) & 7).unwrap(),
                IrValue::make_ubits(5, full).unwrap(),
            ]);
            let args = [
                IrValue::make_ubits(5, x).unwrap(),
                IrValue::make_ubits(2, selector as u64).unwrap(),
            ];
            assert_eq!(evaluate(&source, &args), expected);
            assert_eq!(evaluate(&candidate, &args), expected);
        }
    }
}

#[test]
fn nested_priority_choices_keep_low_bit_precedence_and_default() {
    for direction in [Direction::Left, Direction::Right] {
        let mut builder = FnBuilder::new("nested_priority");
        let data = builder.param("x", Type::Bits(5)).unwrap();
        let p = builder.param("p", Type::Bits(1)).unwrap();
        let selector = builder.param("selector", Type::Bits(2)).unwrap();
        let zero = builder.literal(IrValue::make_ubits(3, 0).unwrap()).unwrap();
        let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
        let three = builder.literal(IrValue::make_ubits(3, 3).unwrap()).unwrap();
        let seven = builder.literal(IrValue::make_ubits(3, 7).unwrap()).unwrap();
        let nested = builder.select(p, &[zero, three], None).unwrap();
        let amount = builder
            .priority_select(selector, &[nested, one], seven)
            .unwrap();
        let result = direction.emit(&mut builder, data, amount).unwrap();
        let source = builder.build(result).unwrap();
        let candidate =
            constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
                .expect("nested priority choices must be recognized");
        prove_candidate(&source, &candidate);
        for x in 0..32 {
            for p in 0..2 {
                for selector in 0..4 {
                    let amount = if selector & 1 != 0 {
                        if p != 0 { 3 } else { 0 }
                    } else if selector & 2 != 0 {
                        1
                    } else {
                        7
                    };
                    let args = [
                        IrValue::make_ubits(5, x).unwrap(),
                        IrValue::make_ubits(1, p).unwrap(),
                        IrValue::make_ubits(2, selector).unwrap(),
                    ];
                    assert_eq!(
                        evaluate(&candidate, &args),
                        shift_oracle(&args[0], amount, direction)
                    );
                }
            }
        }
    }
}

/// Requires all-or-nothing recognition, including preservation of the source.
fn assert_rejected(source: &ir::Fn, limits: ConstantShiftChoiceLimits) {
    let before = source.to_string();
    let node_count = source.nodes.len();
    assert!(constant_shift_choice_candidate(source, limits).is_none());
    assert_eq!(source.to_string(), before);
    assert_eq!(source.nodes.len(), node_count);
    assert_eq!(logical_shift_count(source), 1);
}

#[test]
fn variable_leaves_are_rejected_even_when_nested_or_masked() {
    for direction in [Direction::Left, Direction::Right] {
        for shape in 0..4 {
            let mut builder = FnBuilder::new("variable_leaf");
            let data = builder.param("x", Type::Bits(8)).unwrap();
            let p = builder.param("p", Type::Bits(1)).unwrap();
            let q = builder.param("q", Type::Bits(1)).unwrap();
            let variable = builder.param("variable_amount", Type::Bits(3)).unwrap();
            let two = builder.literal(IrValue::make_ubits(3, 2).unwrap()).unwrap();
            let three = builder.literal(IrValue::make_ubits(3, 3).unwrap()).unwrap();
            let amount = match shape {
                0 => builder.select(p, &[two, variable], None).unwrap(),
                1 => {
                    let nested = builder.select(q, &[three, variable], None).unwrap();
                    builder.select(p, &[two, nested], None).unwrap()
                }
                2 => {
                    let selector = builder.concat(&[q, p]).unwrap();
                    builder
                        .priority_select(selector, &[two, three], variable)
                        .unwrap()
                }
                3 => {
                    let chosen = builder.select(p, &[two, variable], None).unwrap();
                    let mask = builder.sign_extend(q, 3).unwrap();
                    builder.and(chosen, mask).unwrap()
                }
                _ => unreachable!(),
            };
            let result = direction.emit(&mut builder, data, amount).unwrap();
            assert_rejected(
                &builder.build(result).unwrap(),
                ConstantShiftChoiceLimits::default(),
            );
        }

        let mut builder = FnBuilder::new("narrow_variable_is_not_a_choice");
        let data = builder.param("x", Type::Bits(4)).unwrap();
        let variable = builder.param("variable_amount", Type::Bits(1)).unwrap();
        let result = direction.emit(&mut builder, data, variable).unwrap();
        assert_rejected(
            &builder.build(result).unwrap(),
            ConstantShiftChoiceLimits::default(),
        );
    }
}

#[test]
fn generic_and_zero_extended_masks_are_not_replicated_boolean_masks() {
    for direction in [Direction::Left, Direction::Right] {
        for mask_kind in 0..3 {
            let mut builder = FnBuilder::new("non_boolean_mask");
            let data = builder.param("x", Type::Bits(8)).unwrap();
            let p = builder.param("p", Type::Bits(1)).unwrap();
            let generic = builder.param("generic", Type::Bits(3)).unwrap();
            let two_bits = builder.param("two_bits", Type::Bits(2)).unwrap();
            let mask = match mask_kind {
                0 => generic,
                1 => builder.zero_extend(p, 3).unwrap(),
                2 => builder.sign_extend(two_bits, 3).unwrap(),
                _ => unreachable!(),
            };
            let three = builder.literal(IrValue::make_ubits(3, 3).unwrap()).unwrap();
            let amount = builder.and(three, mask).unwrap();
            let result = direction.emit(&mut builder, data, amount).unwrap();
            assert_rejected(
                &builder.build(result).unwrap(),
                ConstantShiftChoiceLimits::default(),
            );
        }
    }
}

#[test]
fn reassociated_boolean_masks_preserve_selector_correlations() {
    for direction in [Direction::Left, Direction::Right] {
        for nested in [false, true] {
            let mut builder = FnBuilder::new("reassociated_masks");
            let data = builder.param("x", Type::Bits(5)).unwrap();
            let en = builder.param("en", Type::Bits(1)).unwrap();
            let p = builder.param("p", Type::Bits(1)).unwrap();
            let q = builder.param("q", Type::Bits(1)).unwrap();
            let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
            let three = builder.literal(IrValue::make_ubits(3, 3).unwrap()).unwrap();
            let chosen = builder.select(q, &[one, three], None).unwrap();
            let predicate = builder.or(p, q).unwrap();
            let first_mask = builder.sign_extend(predicate, 3).unwrap();
            let second_mask = builder.sign_extend(en, 3).unwrap();
            let amount = if nested {
                let first = builder.and(first_mask, chosen).unwrap();
                builder.and(second_mask, first).unwrap()
            } else {
                builder.and_all(&[second_mask, chosen, first_mask]).unwrap()
            };
            let result = direction.emit(&mut builder, data, amount).unwrap();
            let source = builder.build(result).unwrap();
            let candidate =
                constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
                    .expect("operand order and association must not hide Boolean masks");
            prove_candidate(&source, &candidate);
            for x in 0..32 {
                for controls in 0..8 {
                    let en = controls & 1 != 0;
                    let p = controls & 2 != 0;
                    let q = controls & 4 != 0;
                    let amount = if en && (p || q) {
                        if q { 3 } else { 1 }
                    } else {
                        0
                    };
                    let args = [
                        IrValue::make_ubits(5, x).unwrap(),
                        IrValue::bool(en),
                        IrValue::bool(p),
                        IrValue::bool(q),
                    ];
                    assert_eq!(
                        evaluate(&candidate, &args),
                        shift_oracle(&args[0], amount, direction)
                    );
                }
            }
        }
    }
}

#[test]
fn arithmetic_right_shifts_are_outside_the_rewrite() {
    let mut builder = FnBuilder::new("arithmetic_shift");
    let data = builder.param("x", Type::Bits(5)).unwrap();
    let p = builder.param("p", Type::Bits(1)).unwrap();
    let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
    let five = builder.literal(IrValue::make_ubits(3, 5).unwrap()).unwrap();
    let amount = builder.select(p, &[one, five], None).unwrap();
    let result = builder.shra(data, amount).unwrap();
    let source = builder.build(result).unwrap();
    assert!(
        constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default()).is_none()
    );
}

#[test]
fn internal_shifts_retain_amount_users_and_share_the_data_producer() {
    let mut builder = FnBuilder::new("shared_data_and_amount");
    let lhs = builder.param("lhs", Type::Bits(8)).unwrap();
    let rhs = builder.param("rhs", Type::Bits(8)).unwrap();
    let p = builder.param("p", Type::Bits(1)).unwrap();
    let data = builder.add(lhs, rhs).unwrap();
    builder.set_name(data, "shared_data").unwrap();
    let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
    let three = builder.literal(IrValue::make_ubits(3, 3).unwrap()).unwrap();
    let amount = builder.select(p, &[one, three], None).unwrap();
    builder.set_name(amount, "retained_amount").unwrap();
    let left = builder.shll(data, amount).unwrap();
    let right = builder.shrl(data, amount).unwrap();
    let combined = builder.xor(left, data).unwrap();
    let result = builder.tuple(&[combined, amount, data, right]).unwrap();
    let source = builder.build(result).unwrap();
    let candidate = constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
        .expect("shifts with internal and tuple users must be rewritten");
    assert_eq!(logical_shift_count(&candidate), 0);
    assert_eq!(
        candidate
            .nodes
            .iter()
            .filter(|node| matches!(node.payload, NodePayload::Binop(Binop::Add, _, _)))
            .count(),
        1,
        "the shared producer must never be duplicated"
    );
    let shared_data = find_node_by_name(&candidate, "shared_data").unwrap();
    let retained_amount = find_node_by_name(&candidate, "retained_amount").unwrap();
    let NodePayload::Tuple(elements) = &candidate.get_node(candidate.ret_node_ref.unwrap()).payload
    else {
        panic!("tuple result must survive the rewrite");
    };
    assert_eq!(elements[1], retained_amount);
    assert_eq!(elements[2], shared_data);
    assert!(
        candidate
            .nodes
            .iter()
            .filter(|node| operands(&node.payload).contains(&shared_data))
            .count()
            >= 4,
        "all constant projections must reuse the original data node"
    );
    prove_candidate(&source, &candidate);
    for lhs in [0, 1, 127, 128, 255] {
        for rhs in [0, 1, 127, 255] {
            for p in 0..2 {
                let args = [
                    IrValue::make_ubits(8, lhs).unwrap(),
                    IrValue::make_ubits(8, rhs).unwrap(),
                    IrValue::make_ubits(1, p).unwrap(),
                ];
                assert_eq!(evaluate(&source, &args), evaluate(&candidate, &args));
            }
        }
    }
}

#[test]
fn traversal_emission_and_distinct_shift_budgets_leave_the_source_untouched() {
    let source = masked_choice_fixture(Direction::Left, false);
    for limits in [
        ConstantShiftChoiceLimits {
            max_distinct_shifts: 3,
            ..ConstantShiftChoiceLimits::default()
        },
        ConstantShiftChoiceLimits {
            max_visited_nodes: 1,
            ..ConstantShiftChoiceLimits::default()
        },
        ConstantShiftChoiceLimits {
            max_emitted_nodes: 1,
            ..ConstantShiftChoiceLimits::default()
        },
    ] {
        assert_rejected(&source, limits);
    }

    let mut builder = FnBuilder::new("five_effective_shifts");
    let data = builder.param("x", Type::Bits(8)).unwrap();
    let selector = builder.param("selector", Type::Bits(3)).unwrap();
    let cases: Vec<_> = (0..5)
        .map(|value| {
            builder
                .literal(IrValue::make_ubits(3, value).unwrap())
                .unwrap()
        })
        .collect();
    let amount = builder.select(selector, &cases, Some(cases[0])).unwrap();
    let result = builder.shll(data, amount).unwrap();
    assert_rejected(
        &builder.build(result).unwrap(),
        ConstantShiftChoiceLimits::default(),
    );
}

/// Builds a DAG with exponentially many unfolded paths but few distinct nodes.
fn repeated_choice_dag(levels: usize) -> ir::Fn {
    let mut builder = FnBuilder::new("shared_choice_dag");
    let data = builder.param("x", Type::Bits(8)).unwrap();
    let p = builder.param("p", Type::Bits(1)).unwrap();
    let q = builder.param("q", Type::Bits(1)).unwrap();
    let one = builder.literal(IrValue::make_ubits(3, 1).unwrap()).unwrap();
    let two = builder.literal(IrValue::make_ubits(3, 2).unwrap()).unwrap();
    let mut choice = builder.select(p, &[one, two], None).unwrap();
    for _ in 0..levels {
        choice = builder.select(q, &[choice, choice], None).unwrap();
    }
    let result = builder.shrl(data, choice).unwrap();
    builder.build(result).unwrap()
}

#[test]
fn shared_choice_dags_are_memoized_and_fusion_is_idempotent() {
    let source = repeated_choice_dag(24);
    let candidate = constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
        .expect("shared choice nodes must be visited only once");
    assert!(candidate.nodes.len() <= source.nodes.len() + 16);
    assert_eq!(logical_shift_count(&candidate), 0);
    prove_candidate(&source, &candidate);
    assert!(
        constant_shift_choice_candidate(&candidate, ConstantShiftChoiceLimits::default()).is_none(),
        "a second rewrite must not grow or oscillate"
    );
    let repeated =
        constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default()).unwrap();
    assert_eq!(candidate.to_string(), repeated.to_string());
    assert_rejected(
        &repeated_choice_dag(80),
        ConstantShiftChoiceLimits::default(),
    );
}

/// Exercises generic shared trees independently of the directed regression.
fn generated_choice_dag(width: usize, direction: Direction, seed: u64) -> ir::Fn {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut builder = FnBuilder::new("generated_choices");
    let data = builder.param("x", Type::Bits(width)).unwrap();
    let p = builder.param("p", Type::Bits(1)).unwrap();
    let q = builder.param("q", Type::Bits(1)).unwrap();
    let r = builder.param("r", Type::Bits(1)).unwrap();
    let controls = [p, q, r];
    let priority_selector = builder.concat(&[q, p]).unwrap();
    let mut choices: Vec<_> = [0, 1, width - 1, width + 2]
        .into_iter()
        .map(|value| {
            builder
                .literal(IrValue::make_ubits(6, value as u64).unwrap())
                .unwrap()
        })
        .collect();
    for _ in 0..10 {
        let a = choices[rng.gen_range(0..choices.len())];
        let b = choices[rng.gen_range(0..choices.len())];
        let c = choices[rng.gen_range(0..choices.len())];
        let selector = controls[rng.gen_range(0..controls.len())];
        let choice = match rng.gen_range(0..3) {
            0 => builder.select(selector, &[a, b], None).unwrap(),
            1 => builder
                .priority_select(priority_selector, &[a, b], c)
                .unwrap(),
            2 => {
                let mask = builder.sign_extend(selector, 6).unwrap();
                builder.and(a, mask).unwrap()
            }
            _ => unreachable!(),
        };
        choices.push(choice);
    }
    let amount = builder
        .select(r, &[*choices.last().unwrap(), choices[1]], None)
        .unwrap();
    let result = direction.emit(&mut builder, data, amount).unwrap();
    builder.build(result).unwrap()
}

#[test]
fn deterministic_choice_dag_property_sweep_preserves_all_controls() {
    for width in [3, 5, 9] {
        for direction in [Direction::Left, Direction::Right] {
            for seed in 0..4 {
                let source = generated_choice_dag(width, direction, 0x5e1ec7 + seed);
                let candidate =
                    constant_shift_choice_candidate(&source, ConstantShiftChoiceLimits::default())
                        .expect("generated constant-only DAGs are within the recognition budgets");
                assert_eq!(logical_shift_count(&candidate), 0);
                // The small width is exhaustive; larger widths mix corners
                // and deterministic data patterns while exhausting controls.
                let data_values: Vec<_> = if width == 3 {
                    (0..8).collect()
                } else {
                    vec![0, 1, 2, 3, 1 << (width - 1), (1 << width) - 1, 0x15, 0x0a]
                };
                for x in data_values {
                    for controls in 0..8 {
                        let args = [
                            IrValue::make_ubits(width, x).unwrap(),
                            IrValue::bool(controls & 1 != 0),
                            IrValue::bool(controls & 2 != 0),
                            IrValue::bool(controls & 4 != 0),
                        ];
                        assert_eq!(
                            evaluate(&source, &args),
                            evaluate(&candidate, &args),
                            "width={width}, direction={direction:?}, seed={seed}, controls={controls}"
                        );
                    }
                }
            }
        }
    }
}
