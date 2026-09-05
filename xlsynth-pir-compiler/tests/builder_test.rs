// SPDX-License-Identifier: Apache-2.0

//! Function construction feeds Cranelift directly, without a textual/C API
//! bridge.

use xlsynth_pir::ir::{self, ExtNaryAddArchitecture, MemberType, Type};
use xlsynth_pir::ir_eval::{self, FnEvalResult, eval_fn, eval_fn_in_package};
use xlsynth_pir::{FnBuilder, IrValue, NaryAddOptions, NaryAddTerm, NormalizeLeftOptions};
use xlsynth_pir_compiler::{ExecutionOptions, IrExecutionResult, PirFunctionCompiler};

/// Cross-backend event representation, preserving the metadata shared by both.
#[derive(Debug, PartialEq)]
struct ObservedExecution {
    value: IrValue,
    assertion_failures: Vec<ir_eval::AssertionFailure>,
    trace_messages: Vec<ir_eval::TraceMessage>,
    cover_counts: Vec<ir_eval::CoverCount>,
}

impl ObservedExecution {
    fn from_pir(result: FnEvalResult) -> Self {
        match result {
            FnEvalResult::Success(result) => Self {
                value: result.value,
                assertion_failures: Vec::new(),
                trace_messages: result.trace_messages,
                cover_counts: result.cover_counts,
            },
            FnEvalResult::Failure(result) => {
                assert!(result.assumption_failures.is_empty());
                Self {
                    value: result.value,
                    assertion_failures: result.assertion_failures,
                    trace_messages: result.trace_messages,
                    cover_counts: result.cover_counts,
                }
            }
        }
    }

    fn from_compiled(result: IrExecutionResult) -> Self {
        assert!(result.events.assumption_failures.is_empty());
        Self {
            value: result.value,
            assertion_failures: result
                .events
                .assertion_failures
                .into_iter()
                .map(|failure| ir_eval::AssertionFailure {
                    message: failure.message,
                    label: failure.label,
                })
                .collect(),
            trace_messages: result
                .events
                .trace_messages
                .into_iter()
                .map(|trace| ir_eval::TraceMessage {
                    message: trace.message,
                    verbosity: trace.verbosity,
                })
                .collect(),
            cover_counts: result
                .events
                .cover_counts
                .into_iter()
                .map(|cover| ir_eval::CoverCount {
                    node_text_id: cover.node_text_id,
                    label: cover.label,
                    count: cover.count,
                })
                .collect(),
        }
    }
}

/// Exercises the full width, including bits above the first machine word.
fn bit_corners(width: usize) -> [IrValue; 4] {
    [
        IrValue::make_ubits(width, 0).unwrap(),
        IrValue::make_ubits(width, 1).unwrap(),
        IrValue::all_ones_bits(width),
        IrValue::signed_min_bits(width),
    ]
}

/// Checks compiled package execution against the interpreter and its events.
fn assert_package_execution(
    package: &ir::Package,
    compiled: &PirFunctionCompiler,
    args: &[IrValue],
) {
    let expected = eval_fn_in_package(package, package.get_top_fn().unwrap(), args);
    let actual = compiled
        .run_ir_values_with_events(args, ExecutionOptions::collect_all())
        .unwrap();
    assert_eq!(
        ObservedExecution::from_compiled(actual),
        ObservedExecution::from_pir(expected),
        "arguments: {args:?}"
    );
}

#[test]
fn jit_compiles_builder_output_with_wide_aggregates() {
    for width in [1, 4, 64, 65, 129] {
        let mut b = FnBuilder::new("aggregate_arithmetic");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let y = b.param("y", Type::Bits(width)).unwrap();
        let sum = b.add(x, y).unwrap();
        let difference = b.sub(x, y).unwrap();
        let array = b.array(Type::Bits(width), &[sum, difference]).unwrap();
        let count = b.clz(x).unwrap();
        let result = b.tuple(&[array, count]).unwrap();
        let function = b.build(result).unwrap();
        let compiled = PirFunctionCompiler::compile(&function).unwrap();
        let corners = [
            IrValue::make_ubits(width, 0).unwrap(),
            IrValue::make_ubits(width, 1).unwrap(),
            IrValue::all_ones_bits(width),
            IrValue::signed_min_bits(width),
        ];
        for x in &corners {
            for y in &corners {
                let args = [x.clone(), y.clone()];
                let FnEvalResult::Success(expected) = eval_fn(&function, &args) else {
                    panic!("unexpected PIR interpreter failure");
                };
                assert_eq!(compiled.run_ir_values(&args).unwrap(), expected.value);
            }
        }
    }
}

#[test]
fn jit_compiles_builder_scalar_function() {
    let mut b = FnBuilder::new("increment");
    let x = b.param("x", Type::Bits(32)).unwrap();
    let one = b.literal(IrValue::u32(1)).unwrap();
    let incremented = b.add(x, one).unwrap();
    b.set_name(incremented, "incremented").unwrap();
    let function = b.build(incremented).unwrap();
    let compiled = PirFunctionCompiler::compile(&function).unwrap();
    assert_eq!(compiled.run_u64(&[41]).unwrap(), 42);
    assert_eq!(compiled.run_u64(&[u32::MAX as u64]).unwrap(), 0);
}

#[test]
fn jit_compiles_zero_width_one_hot_from_builder() {
    for lsb_priority in [true, false] {
        let mut b = FnBuilder::new("empty_one_hot");
        let x = b.param("x", Type::Bits(0)).unwrap();
        let result = b.one_hot(x, lsb_priority).unwrap();
        let function = b.build(result).unwrap();
        let compiled = PirFunctionCompiler::compile(&function).unwrap();
        assert_eq!(
            compiled
                .run_ir_values(&[IrValue::make_ubits(0, 0).unwrap()])
                .unwrap(),
            IrValue::make_ubits(1, 1).unwrap()
        );
    }
}

#[test]
fn jit_compiles_builder_invokes_with_wide_aggregate_signatures() {
    let width = 129;
    let pair_type = Type::Tuple(vec![
        Box::new(Type::Bits(width)),
        Box::new(Type::Bits(width)),
    ]);
    let mut helper = FnBuilder::new("pair_arithmetic");
    let pair = helper.param("pair", pair_type.clone()).unwrap();
    let x = helper.tuple_index(pair, 0).unwrap();
    let y = helper.tuple_index(pair, 1).unwrap();
    let sum = helper.add(x, y).unwrap();
    let difference = helper.sub(x, y).unwrap();
    let result = helper.array(Type::Bits(width), &[sum, difference]).unwrap();
    let mut package = helper.build_package(result, "aggregate_calls").unwrap();

    let mut forward = FnBuilder::new("forward");
    let pair = forward.param("pair", pair_type.clone()).unwrap();
    let called = forward
        .invoke(package.get_fn("pair_arithmetic").unwrap(), &[pair])
        .unwrap();
    forward.build_into_package(called, &mut package).unwrap();

    let mut entry = FnBuilder::new("entry");
    let pair = entry.param("pair", pair_type).unwrap();
    let called = entry
        .invoke(package.get_fn("forward").unwrap(), &[pair])
        .unwrap();
    entry.build_into_package(called, &mut package).unwrap();
    package.top = Some(("entry".to_string(), MemberType::Function));
    let compiled = PirFunctionCompiler::compile_package(&package).unwrap();
    let corners = bit_corners(width);
    for x in &corners {
        for y in &corners {
            assert_package_execution(
                &package,
                &compiled,
                &[IrValue::make_tuple(&[x.clone(), y.clone()])],
            );
        }
    }
}

#[test]
fn jit_compiles_builder_counted_for_with_wide_array_carry() {
    for width in [65, 129] {
        let carry_type = Type::new_array(Type::Bits(width), 2);
        let mut body = FnBuilder::new("loop_body");
        let index = body.param("index", Type::Bits(4)).unwrap();
        let carry = body.param("carry", carry_type.clone()).unwrap();
        let increment = body.param("increment", Type::Bits(width)).unwrap();
        let token = body.after_all(&[]).unwrap();
        let enabled = body.literal(IrValue::bool(true)).unwrap();
        body.cover(enabled, "loop_iteration").unwrap();
        body.trace(token, enabled, "index={}", &[index], 0).unwrap();
        let slot = body.bit_slice(index, 0, 1).unwrap();
        let current = body.array_index(carry, slot).unwrap();
        let updated = body.add(current, increment).unwrap();
        let next = body.array_update(carry, updated, &[slot]).unwrap();
        let mut package = body.build_package(next, "wide_loop").unwrap();

        let mut entry = FnBuilder::new("entry");
        let init = entry.param("init", carry_type).unwrap();
        let increment = entry.param("increment", Type::Bits(width)).unwrap();
        let result = entry
            .counted_for(
                init,
                5,
                3,
                package.get_fn("loop_body").unwrap(),
                &[increment],
            )
            .unwrap();
        entry.build_into_package(result, &mut package).unwrap();
        package.top = Some(("entry".to_string(), MemberType::Function));
        let compiled = PirFunctionCompiler::compile_package(&package).unwrap();
        let corners = bit_corners(width);
        for (index, first) in corners.iter().enumerate() {
            let second = &corners[(index + 1) % corners.len()];
            let carry = IrValue::make_array(&[first.clone(), second.clone()]).unwrap();
            for increment in &corners {
                assert_package_execution(&package, &compiled, &[carry.clone(), increment.clone()]);
            }
        }
    }
}

#[test]
fn jit_compiles_builder_contracts_with_matching_structured_events() {
    let mut b = FnBuilder::new("contracts");
    let x = b.param("x", Type::Bits(129)).unwrap();
    let okay = b.param("okay", Type::Bits(1)).unwrap();
    let emit = b.param("emit", Type::Bits(1)).unwrap();
    let token = b.after_all(&[]).unwrap();
    b.cover(emit, "trace_enabled").unwrap();
    let checked = b
        .assert(token, okay, "condition failed\n", "valid_input")
        .unwrap();
    let traced = b.trace(checked, emit, "{{value}}={:x}", &[x], 2).unwrap();
    let aggregate = b.tuple(&[x, emit]).unwrap();
    let traced_again = b
        .trace(traced, emit, "aggregate={}", &[aggregate], 3)
        .unwrap();
    let completed = b.after_all(&[checked, traced_again]).unwrap();
    let result = b.tuple(&[x, completed]).unwrap();
    let function = b.build(result).unwrap();
    let compiled = PirFunctionCompiler::compile(&function).unwrap();
    for x in bit_corners(129) {
        for okay in [false, true] {
            for emit in [false, true] {
                let args = [x.clone(), IrValue::bool(okay), IrValue::bool(emit)];
                let expected = ObservedExecution::from_pir(eval_fn(&function, &args));
                assert_eq!(expected.assertion_failures.len(), usize::from(!okay));
                assert_eq!(expected.trace_messages.len(), if emit { 2 } else { 0 });
                assert_eq!(expected.cover_counts.len(), 1);
                assert_eq!(expected.cover_counts[0].count, u64::from(emit));
                let actual = compiled
                    .run_ir_values_with_events(&args, ExecutionOptions::collect_all())
                    .unwrap();
                assert_eq!(ObservedExecution::from_compiled(actual), expected);
            }
        }
    }
}

#[test]
fn jit_compiles_all_builder_extension_operations() {
    for width in [4, 65, 129] {
        let mut b = FnBuilder::new("extensions");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let y = b.param("y", Type::Bits(width)).unwrap();
        let carry = b.param("carry", Type::Bits(1)).unwrap();
        let count = b.param("count", Type::Bits(8)).unwrap();
        let values = [
            b.ext_carry_out(x, y, carry).unwrap(),
            b.ext_prio_encode(x, true).unwrap(),
            b.ext_prio_encode(x, false).unwrap(),
            b.ext_clz(x, 3, 8).unwrap(),
            b.ext_normalize_left(
                x,
                NormalizeLeftOptions {
                    normalized_bit_count: width + 2,
                    shift_offset: 1,
                    clz_bit_count: Some(8),
                },
            )
            .unwrap(),
            b.ext_mask_low(count, width).unwrap(),
            b.ext_nary_add(
                &[
                    NaryAddTerm::unsigned(x),
                    NaryAddTerm::signed(y).negate(),
                    NaryAddTerm::unsigned(carry),
                ],
                NaryAddOptions {
                    bit_count: width + 1,
                    architecture: Some(ExtNaryAddArchitecture::KoggeStone),
                },
            )
            .unwrap(),
            b.ext_nary_add(&[], NaryAddOptions::new(width)).unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let function = b.build(result).unwrap();
        let compiled = PirFunctionCompiler::compile(&function).unwrap();
        let corners = bit_corners(width);
        for (index, x) in corners.iter().enumerate() {
            let y = &corners[(index + 1) % corners.len()];
            for carry in [false, true] {
                for count in [0, width, width + 1] {
                    let args = [
                        x.clone(),
                        y.clone(),
                        IrValue::bool(carry),
                        IrValue::make_ubits(8, count as u64).unwrap(),
                    ];
                    let expected = ObservedExecution::from_pir(eval_fn(&function, &args));
                    let actual = compiled
                        .run_ir_values_with_events(&args, ExecutionOptions::collect_all())
                        .unwrap();
                    assert_eq!(ObservedExecution::from_compiled(actual), expected);
                }
            }
        }
    }
}
