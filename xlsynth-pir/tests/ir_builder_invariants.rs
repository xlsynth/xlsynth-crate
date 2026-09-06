// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{Fn, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::{BuilderError, FnBuilder, IrBits, IrValue};

/// Evaluates a constructed function that has no runtime contract events.
fn evaluate(function: &Fn, args: &[IrValue]) -> IrValue {
    match eval_fn(function, args) {
        FnEvalResult::Success(result) => result.value,
        FnEvalResult::Failure(failure) => panic!("unexpected evaluation failure: {failure:?}"),
    }
}

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

/// Checks the public state immediately before and after a rejected request.
fn assert_rejected_preserves_state<T>(
    builder: &mut FnBuilder,
    request: impl FnOnce(&mut FnBuilder) -> Result<T, BuilderError>,
) {
    let last = builder.last_value();
    let last_type = last.map(|value| builder.get_type(value).unwrap().clone());
    assert!(request(builder).is_err());
    assert_eq!(builder.last_value(), last);
    assert_eq!(
        builder
            .last_value()
            .map(|value| builder.get_type(value).unwrap().clone()),
        last_type
    );
}

/// Builds the same function with or without rejected requests interleaved.
fn build_with_rejected_requests(inject_failures: bool) -> Fn {
    let mut builder = FnBuilder::new("recoverable");
    let a = builder.param("a", Type::Bits(4)).unwrap();
    if inject_failures {
        assert_rejected_preserves_state(&mut builder, |builder| {
            builder.param("invalid name", Type::Bits(4))
        });
        assert_rejected_preserves_state(&mut builder, |builder| builder.param("a", Type::Bits(4)));
        assert_rejected_preserves_state(&mut builder, |builder| {
            builder.param("overflow", Type::new_array(Type::Bits(usize::MAX), 2))
        });
    }
    let b = builder.param("b", Type::Bits(4)).unwrap();
    let aggregate = builder.tuple(&[a]).unwrap();
    builder.set_name(aggregate, "aggregate").unwrap();
    if inject_failures {
        // Type deduction rejects this before a tentative node is appended.
        assert_rejected_preserves_state(&mut builder, |builder| builder.tuple_index(aggregate, 1));
        assert_rejected_preserves_state(&mut builder, |builder| builder.array(Type::Bits(3), &[a]));
        // These deduce a result type, then fail local semantic checks.
        assert_rejected_preserves_state(&mut builder, |builder| builder.bit_slice(a, 3, 2));
        assert_rejected_preserves_state(&mut builder, |builder| builder.not(aggregate));
        assert_rejected_preserves_state(&mut builder, |builder| builder.select(b, &[a], None));
        assert_rejected_preserves_state(&mut builder, |builder| builder.set_name(a, "aggregate"));
        assert_rejected_preserves_state(&mut builder, |builder| {
            builder.set_name(aggregate, "invalid name")
        });
    }
    // Renaming releases the previous name, including a parameter's name.
    builder.set_name(a, "input").unwrap();
    builder.set_name(aggregate, "a").unwrap();
    let sum = builder.add(a, b).unwrap();
    if inject_failures {
        assert_rejected_preserves_state(&mut builder, |builder| {
            builder.param("late", Type::Bits(4))
        });
    }
    // A rejected late parameter must not reserve its requested name.
    builder.set_name(sum, "late").unwrap();
    let result = builder.tuple(&[sum, aggregate]).unwrap();
    builder.build(result).unwrap()
}

#[test]
fn rejected_requests_leave_the_complete_function_unchanged() {
    let baseline = build_with_rejected_requests(false);
    let recovered = build_with_rejected_requests(true);
    assert_eq!(recovered.nodes.len(), baseline.nodes.len());
    for (actual, expected) in recovered.nodes.iter().zip(&baseline.nodes) {
        assert_eq!(actual.text_id, expected.text_id);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.ty, expected.ty);
        assert_eq!(actual.payload, expected.payload);
    }
    assert_eq!(recovered.params.len(), baseline.params.len());
    assert_eq!(recovered.params, baseline.params);
    for (actual, expected) in recovered.param_nodes().zip(baseline.param_nodes()) {
        assert_eq!(actual.text_id, expected.text_id);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.ty, expected.ty);
    }
    assert_eq!(recovered.ret_node_ref, baseline.ret_node_ref);
    assert_eq!(recovered.ret_ty, baseline.ret_ty);
    assert_eq!(
        evaluate(&recovered, &[bits(4, 3), bits(4, 5)]),
        IrValue::make_tuple(&[bits(4, 8), IrValue::make_tuple(&[bits(4, 3)])])
    );
}

#[test]
fn reductions_distinguish_parity_from_all_and_any_bits() {
    let mut builder = FnBuilder::new("reductions");
    let x = builder.param("x", Type::Bits(4)).unwrap();
    let and = builder.and_reduce(x).unwrap();
    let or = builder.or_reduce(x).unwrap();
    let xor = builder.xor_reduce(x).unwrap();
    let result = builder.tuple(&[and, or, xor]).unwrap();
    let function = builder.build(result).unwrap();
    for (input, expected) in [
        (0, [false, false, false]),
        (1, [false, true, true]),
        (3, [false, true, false]),
        (7, [false, true, true]),
        (15, [true, true, false]),
    ] {
        assert_eq!(
            evaluate(&function, &[bits(4, input)]),
            IrValue::make_tuple(&expected.map(IrValue::bool))
        );
    }
}

#[test]
fn comparisons_distinguish_equality_and_strict_ordering() {
    let mut builder = FnBuilder::new("comparisons");
    let a = builder.param("a", Type::Bits(4)).unwrap();
    let b = builder.param("b", Type::Bits(4)).unwrap();
    let comparisons = [
        builder.eq(a, b).unwrap(),
        builder.ne(a, b).unwrap(),
        builder.ult(a, b).unwrap(),
        builder.ule(a, b).unwrap(),
        builder.ugt(a, b).unwrap(),
        builder.uge(a, b).unwrap(),
        builder.slt(a, b).unwrap(),
        builder.sle(a, b).unwrap(),
        builder.sgt(a, b).unwrap(),
        builder.sge(a, b).unwrap(),
    ];
    let result = builder.tuple(&comparisons).unwrap();
    let function = builder.build(result).unwrap();
    let expected = [
        true, false, false, true, false, true, false, true, false, true,
    ];
    for value in [0, 7, 8, 15] {
        assert_eq!(
            evaluate(&function, &[bits(4, value), bits(4, value)]),
            IrValue::make_tuple(&expected.map(IrValue::bool))
        );
    }
}

#[test]
fn array_concat_accepts_distinct_lengths_and_preserves_order() {
    let mut builder = FnBuilder::new("unequal_arrays");
    let lhs = builder
        .param("lhs", Type::new_array(Type::Bits(4), 2))
        .unwrap();
    let rhs = builder
        .param("rhs", Type::new_array(Type::Bits(4), 3))
        .unwrap();
    let concatenated = builder.array_concat(&[lhs, rhs]).unwrap();
    assert_eq!(
        builder.get_type(concatenated).unwrap(),
        &Type::new_array(Type::Bits(4), 5)
    );
    let start = builder.literal(bits(3, 1)).unwrap();
    let slice = builder.array_slice(concatenated, start, 3).unwrap();
    let result = builder.tuple(&[concatenated, slice]).unwrap();
    let function = builder.build(result).unwrap();
    let lhs = IrValue::make_array(&[bits(4, 1), bits(4, 2)]).unwrap();
    let rhs = IrValue::make_array(&[bits(4, 3), bits(4, 4), bits(4, 5)]).unwrap();
    assert_eq!(
        evaluate(&function, &[lhs, rhs]),
        IrValue::make_tuple(&[
            IrValue::make_array(&[bits(4, 1), bits(4, 2), bits(4, 3), bits(4, 4), bits(4, 5)])
                .unwrap(),
            IrValue::make_array(&[bits(4, 2), bits(4, 3), bits(4, 4)]).unwrap(),
        ])
    );
}

#[test]
fn zero_counts_cover_each_one_hot_position_including_wide_values() {
    for width in [1usize, 2, 4, 65, 129] {
        let mut builder = FnBuilder::new("zero_counts");
        let x = builder.param("x", Type::Bits(width)).unwrap();
        let leading = builder.clz(x).unwrap();
        let trailing = builder.ctz(x).unwrap();
        let result = builder.tuple(&[leading, trailing]).unwrap();
        let function = builder.build(result).unwrap();
        for set_bit in 0..width {
            let mut input = vec![false; width];
            input[set_bit] = true;
            let input = IrValue::Bits(IrBits::from_lsb_is_0(&input));
            assert_eq!(
                evaluate(&function, &[input]),
                IrValue::make_tuple(&[
                    bits(width, (width - set_bit - 1) as u64),
                    bits(width, set_bit as u64),
                ]),
                "width={width}, set_bit={set_bit}"
            );
        }
    }
}
