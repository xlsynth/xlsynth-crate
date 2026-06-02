// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{
    AssumptionFailureKind, CompilerError, ExecutionContext, NativeTupleFieldLayout,
    NativeValueLayout, PirFunctionCompiler, ScalarLayout, WideBitsLayout,
};

fn compile(ir: &str) -> PirFunctionCompiler {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    PirFunctionCompiler::compile(function).expect("function should compile")
}

fn compile_package(ir: &str) -> PirFunctionCompiler {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR package should parse and validate");
    PirFunctionCompiler::compile_package(&package).expect("package should compile")
}

fn assert_matches_evaluator(ir: &str, argument_sets: &[Vec<IrValue>]) {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    let compiler = PirFunctionCompiler::compile(function).expect("function should compile");
    for args in argument_sets {
        let expected = match eval_fn(function, args) {
            FnEvalResult::Success(success) => success.value,
            FnEvalResult::Failure(_) => panic!("PIR evaluation failed for arguments {args:?}"),
        };
        let actual = compiler
            .run_ir_values(args)
            .expect("compiled execution should succeed");
        assert_eq!(actual, expected, "compiler mismatch for arguments {args:?}");
    }
}

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

fn wide_bits(text: &str) -> IrValue {
    IrValue::parse_typed(text).unwrap()
}

fn array(width: usize, values: &[u64]) -> IrValue {
    IrValue::make_array(
        &values
            .iter()
            .map(|value| bits(width, *value))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn tuple(elements: &[IrValue]) -> IrValue {
    IrValue::make_tuple(elements)
}

fn unit_array(element_count: usize) -> IrValue {
    IrValue::make_array(&vec![tuple(&[]); element_count]).unwrap()
}

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

#[test]
fn compile_rejects_unvalidated_invalid_xls_node_semantics() {
    let package = Parser::new(
        r#"package test

fn f(x: bits[8] id=1, y: bits[7] id=2) -> bits[8] {
  ret add.3: bits[8] = add(x, y, id=3)
}
"#,
    )
    .parse_package()
    .expect("invalid semantics should still parse structurally");
    let error = match PirFunctionCompiler::compile(package.get_fn("f").unwrap()) {
        Err(error) => error,
        Ok(_) => panic!("invalid XLS node semantics should not compile"),
    };
    assert!(matches!(
        error,
        CompilerError::InvalidFunction(message) if message.contains("right operand")
    ));
}

#[test]
fn package_compiler_lowers_nested_scalar_invokes() {
    let compiler = compile_package(
        r#"package test

fn add_one(x: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  ret result: bits[8] = add(x, one, id=3)
}

fn through(x: bits[8] id=4) -> bits[8] {
  ret result: bits[8] = invoke(x, to_apply=add_one, id=5)
}

top fn f(x: bits[8] id=6) -> bits[8] {
  ret result: bits[8] = invoke(x, to_apply=through, id=7)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[41]).expect("execute"), 42);
    assert!(compiler.scratch_byte_count() > 0);
}

#[test]
fn package_compiler_reuses_static_callee_scratch_across_invoke_sites() {
    let compiler = compile_package(
        r#"package test

fn select_first(x: bits[8] id=1) -> bits[8] {
  values: bits[8][32] = array(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, id=2)
  zero: bits[5] = literal(value=0, id=3)
  ret selected: bits[8] = array_index(values, indices=[zero], id=4)
}

top fn f(x: bits[8] id=5) -> bits[8] {
  first: bits[8] = invoke(x, to_apply=select_first, id=6)
  ret second: bits[8] = invoke(first, to_apply=select_first, id=7)
}
"#,
    );
    // The callee uses one 32-byte aggregate slot. The caller uses 33 bytes for
    // its two invoke sites. Both calls share the callee's package-global slot.
    assert_eq!(compiler.scratch_byte_count(), 65);
    assert_eq!(compiler.run_u64(&[42]).expect("execute"), 42);
}

#[test]
fn package_compiler_passes_aggregate_invokes_by_native_address() {
    let ir = r#"package test

fn preserve(values: bits[8][2] id=1) -> bits[8][2] {
  ret result: bits[8][2] = identity(values, id=2)
}

top fn f(values: bits[8][2] id=3) -> bits[8][2] {
  ret result: bits[8][2] = invoke(values, to_apply=preserve, id=4)
}
"#;
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR package should parse and validate");
    let compiler = PirFunctionCompiler::compile_package(&package).expect("package should compile");
    let values = array(8, &[0x12, 0xa5]);
    let expected =
        match eval_fn_in_package(&package, package.get_top_fn().unwrap(), &[values.clone()]) {
            FnEvalResult::Success(success) => success.value,
            other => panic!("PIR evaluation failed: {other:?}"),
        };
    assert_eq!(
        compiler
            .run_ir_values(&[values.clone()])
            .expect("execute aggregate invoke"),
        expected
    );
}

#[test]
fn package_compiler_accumulates_repeated_callee_events() {
    let compiler = compile_package(
        r#"package test

fn observed(x: bits[1] id=1) -> bits[1] {
  c: () = cover(x, label="callee_cover", id=2)
  t: token = after_all(id=3)
  tr: token = trace(t, x, format="x={}", data_operands=[x], id=4)
  ret result: bits[1] = identity(x, id=5)
}

top fn f(x: bits[1] id=6) -> bits[1] {
  first: bits[1] = invoke(x, to_apply=observed, id=7)
  ret second: bits[1] = invoke(first, to_apply=observed, id=8)
}
"#,
    );
    let result = compiler
        .run_ir_values_with_events(&[bits(1, 1)])
        .expect("execute repeated invokes");
    assert_eq!(result.value, bits(1, 1));
    assert_eq!(result.events.cover_counts.len(), 1);
    assert_eq!(result.events.cover_counts[0].node_text_id, 2);
    assert_eq!(result.events.cover_counts[0].count, 2);
    assert_eq!(
        result
            .events
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["x=1", "x=1"]
    );
}

#[test]
fn native_carrier_storage_is_accessed_without_argument_copying() {
    let compiler = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret sum: bits[8] = add(x, y, id=3)
}
"#,
    );
    let x: u8 = 7;
    let y: u8 = 9;
    let mut output: u8 = 0;
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
    ];

    // SAFETY: the compiled signature describes two bits[8] (`u8`) inputs and one
    // bits[8] (`u8`) output, all alive and properly aligned for this call.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native execution should succeed");
    }
    assert_eq!(output, 16);
}

#[test]
fn native_aggregate_copy_allows_exact_input_output_aliasing() {
    let compiler = compile(
        r#"package test

fn f(values: bits[16][4] id=1) -> bits[16][4] {
  ret result: bits[16][4] = identity(values, id=2)
}
"#,
    );
    let mut values = [11u16, 22, 33, 44];
    let inputs = [std::ptr::from_ref(&values).cast::<u8>()];

    // SAFETY: `values` carries the published native array layout. Exact
    // input/output aliasing is supported by the aggregate memmove lowering.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut values).cast())
            .expect("native aggregate identity should execute in place");
    }
    assert_eq!(values, [11, 22, 33, 44]);
}

#[test]
fn compiles_zero_sized_token_and_cover_unit_results() {
    let token_compiler = compile(
        r#"package test

fn f() -> token {
  ret t: token = after_all(id=1)
}
"#,
    );
    assert_eq!(token_compiler.result_layout(), &NativeValueLayout::Token);
    assert_eq!(
        token_compiler.run_ir_values(&[]).expect("token execution"),
        IrValue::make_token()
    );

    let cover_compiler = compile(
        r#"package test

fn f(x: bits[1] id=1) -> () {
  ret cv: () = cover(x, label="hit", id=2)
}
"#,
    );
    assert_eq!(
        cover_compiler.result_layout(),
        &NativeValueLayout::Tuple {
            fields: vec![],
            byte_count: 0,
            alignment: 1,
        }
    );
    let execution = cover_compiler
        .run_ir_values_with_events(&[bits(1, 1)])
        .expect("unit-valued cover execution");
    assert_eq!(execution.value, IrValue::make_tuple(&[]));
    assert_eq!(execution.events.cover_counts.len(), 1);
    assert_eq!(execution.events.cover_counts[0].label, "hit");
    assert_eq!(execution.events.cover_counts[0].count, 1);
}

#[test]
fn compiles_zero_width_bits_without_native_storage() {
    let compiler = compile(
        r#"package test

fn f(x: bits[0] id=1) -> bits[0] {
  ret value: bits[0] = identity(x, id=2)
}
"#,
    );
    assert_eq!(
        compiler.result_layout(),
        &NativeValueLayout::Scalar(ScalarLayout {
            bit_count: 0,
            byte_count: 0,
        })
    );
    assert_eq!(compiler.scratch_byte_count(), 0);
    assert_eq!(compiler.run_u64(&[0]).expect("execute"), 0);
    assert_eq!(
        compiler
            .run_ir_values(&[bits(0, 0)])
            .expect("execute dynamic value adapter"),
        bits(0, 0)
    );

    let inputs = [std::ptr::null::<u8>()];
    // SAFETY: `bits[0]` has no native bytes, so null input and output pointers
    // satisfy the published zero-sized storage contract.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::null_mut())
            .expect("execute against zero-sized native storage");
    }
}

#[test]
fn zero_width_bits_feed_nonzero_results() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[0] id=1, c_in: bits[1] id=2, y: bits[8] id=3) -> (bits[8], bits[8], bits[1], bits[1], bits[1], bits[1], bits[0], bits[0]) {
  widened: bits[8] = zero_ext(x, new_bit_count=8, id=4)
  joined: bits[8] = concat(x, y, id=5)
  equal: bits[1] = eq(x, x, id=6)
  reduced: bits[1] = and_reduce(x, id=7)
  carry: bits[1] = ext_carry_out(x, x, c_in, id=8)
  decoded: bits[1] = decode(x, width=1, id=9)
  empty_concat: bits[0] = concat(id=10)
  empty_slice: bits[0] = bit_slice(y, start=4, width=0, id=11)
  ret result: (bits[8], bits[8], bits[1], bits[1], bits[1], bits[1], bits[0], bits[0]) = tuple(widened, joined, equal, reduced, carry, decoded, empty_concat, empty_slice, id=12)
}
"#,
        &[vec![bits(0, 0), bits(1, 1), bits(8, 0xa5)]],
    );
}

#[test]
fn zero_width_array_elements_flow_through_native_layout() {
    let zero_array = IrValue::make_array(&[bits(0, 0), bits(0, 0)]).unwrap();
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[0][2] id=1, index: bits[2] id=2) -> bits[8] {
  selected: bits[0] = array_index(values, indices=[index], id=3)
  ret widened: bits[8] = zero_ext(selected, new_bit_count=8, id=4)
}
"#,
        &[
            vec![zero_array.clone(), bits(2, 0)],
            vec![zero_array, bits(2, 3)],
        ],
    );
}

#[test]
fn compiled_trace_formats_zero_width_bits() {
    let compiler = compile(
        r#"package test

fn f(x: bits[0] id=1, emit: bits[1] id=2) -> token {
  t: token = after_all(id=3)
  ret tr: token = trace(t, emit, format="x={}", data_operands=[x], id=4)
}
"#,
    );
    let execution = compiler
        .run_ir_values_with_events(&[bits(0, 0), bits(1, 1)])
        .expect("execute");
    assert_eq!(execution.events.trace_messages.len(), 1);
    assert_eq!(execution.events.trace_messages[0].message, "x=0");
}

#[test]
fn zero_width_bits_match_evaluator_across_scalar_operation_forms() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[0] id=1, count: bits[0] id=2, update: bits[8] id=3) -> (bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[8], bits[0], bits[0], (bits[0], bits[0])) {
  sum: bits[0] = add(x, x, id=4)
  signed_product: bits[0] = smul(x, x, id=5)
  arithmetic_shift: bits[0] = shra(x, count, id=6)
  signed_quotient: bits[0] = sdiv(x, x, id=7)
  encoded: bits[0] = encode(x, id=8)
  sliced: bits[0] = dynamic_bit_slice(x, count, width=0, id=9)
  updated: bits[0] = bit_slice_update(x, count, update, id=10)
  priority: bits[0] = ext_prio_encode(x, lsb_prio=true, id=11)
  zeroes: bits[8] = ext_clz(x, offset=2, new_bit_count=8, id=12)
  mask: bits[0] = ext_mask_low(count, id=13)
  nary_sum: bits[0] = ext_nary_add(x, signed=[true], negated=[false], arch=ripple_carry, id=14)
  parts: (bits[0], bits[0]) = smulp(x, x, id=15)
  ret result: (bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[0], bits[8], bits[0], bits[0], (bits[0], bits[0])) = tuple(sum, signed_product, arithmetic_shift, signed_quotient, encoded, sliced, updated, priority, zeroes, mask, nary_sum, parts, id=16)
}
"#,
        &[vec![bits(0, 0), bits(0, 0), bits(8, 0xa5)]],
    );
}

#[test]
fn caller_owned_context_accumulates_cover_counts() {
    let compiler = compile(
        r#"package test

fn f(x: bits[1] id=1) -> bits[1] {
  cv: () = cover(x, label="observed", id=2)
  ret identity.3: bits[1] = identity(x, id=3)
}
"#,
    );
    let mut context = ExecutionContext::new(compiler.metadata());
    let mut output: u8 = 0;
    for input in [1u8, 1u8, 0u8] {
        let inputs = [std::ptr::from_ref(&input).cast::<u8>()];
        // SAFETY: input and output carry `bits[1]` in the published native
        // scalar layout, and `context` was created for this compiled function.
        unsafe {
            compiler
                .run_native_with_context(
                    &inputs,
                    std::ptr::from_mut(&mut output).cast(),
                    &mut context,
                )
                .expect("native execution with context");
        }
    }
    assert_eq!(context.result().cover_counts[0].count, 2);
    context.clear();
    assert_eq!(context.result().cover_counts[0].count, 0);
}

#[test]
fn compiled_events_match_pir_evaluator_for_cover_assert_and_trace() {
    let ir = r#"package test

fn f(x: bits[8] id=1, ok: bits[1] id=2, emit: bits[1] id=3) -> bits[8] {
  t: token = after_all(id=4)
  cv: () = cover(emit, label="covered", id=5)
  a: token = assert(t, ok, message="bad condition", label="A", id=6)
  tr: token = trace(a, emit, format="x={}", data_operands=[x], id=7)
  ret identity.8: bits[8] = identity(x, id=8)
}
"#;
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    let compiler = PirFunctionCompiler::compile(function).expect("function should compile");

    for args in [
        vec![bits(8, 0xa5), bits(1, 1), bits(1, 1)],
        vec![bits(8, 0x3c), bits(1, 0), bits(1, 1)],
        vec![bits(8, 0x12), bits(1, 1), bits(1, 0)],
    ] {
        let expected = eval_fn(function, &args);
        let actual = compiler
            .run_ir_values_with_events(&args)
            .expect("compiled execution should succeed");
        match expected {
            FnEvalResult::Success(expected) => {
                assert_eq!(actual.value, expected.value);
                assert!(actual.events.assertion_failures.is_empty());
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .trace_messages
                            .iter()
                            .map(|message| (message.message.clone(), message.verbosity))
                            .collect()
                    ),
                    sorted(
                        expected
                            .trace_messages
                            .iter()
                            .map(|message| (message.message.clone(), message.verbosity))
                            .collect()
                    )
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .cover_counts
                            .iter()
                            .map(|cover| (cover.node_text_id, cover.label.clone(), cover.count))
                            .collect()
                    ),
                    sorted(
                        expected
                            .cover_counts
                            .iter()
                            .map(|cover| (cover.node_text_id, cover.label.clone(), cover.count))
                            .collect()
                    )
                );
            }
            FnEvalResult::Failure(expected) => {
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .assertion_failures
                            .iter()
                            .map(|failure| (failure.message.clone(), failure.label.clone()))
                            .collect()
                    ),
                    sorted(
                        expected
                            .assertion_failures
                            .iter()
                            .map(|failure| (failure.message.clone(), failure.label.clone()))
                            .collect()
                    )
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .trace_messages
                            .iter()
                            .map(|message| (message.message.clone(), message.verbosity))
                            .collect()
                    ),
                    sorted(
                        expected
                            .trace_messages
                            .iter()
                            .map(|message| (message.message.clone(), message.verbosity))
                            .collect()
                    )
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .cover_counts
                            .iter()
                            .map(|cover| (cover.node_text_id, cover.label.clone(), cover.count))
                            .collect()
                    ),
                    sorted(
                        expected
                            .cover_counts
                            .iter()
                            .map(|cover| (cover.node_text_id, cover.label.clone(), cover.count))
                            .collect()
                    )
                );
            }
        }
    }
}

#[test]
fn compiled_trace_matches_evaluator_for_zero_through_three_operands() {
    let ir = r#"package test

fn f(emit: bits[1] id=1) -> token {
  unit: () = tuple(id=2)
  units: ()[1] = array(unit, id=3)
  t: token = after_all(id=4)
  tr0: token = trace(t, emit, format="zero", data_operands=[], id=5)
  tr1: token = trace(tr0, emit, format="one={}", data_operands=[unit], id=6)
  tr2: token = trace(tr1, emit, format="two={},{}", data_operands=[unit, units], id=7)
  ret tr3: token = trace(tr2, emit, format="three={},{},{}", data_operands=[unit, units, emit], id=8)
}
"#;
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    let compiler = PirFunctionCompiler::compile(function).expect("function should compile");
    let args = vec![bits(1, 1)];
    let expected = match eval_fn(function, &args) {
        FnEvalResult::Success(success) => success,
        FnEvalResult::Failure(_) => panic!("traces should not cause evaluation failure"),
    };
    let actual = compiler
        .run_ir_values_with_events(&args)
        .expect("compiled execution should succeed");
    assert_eq!(
        actual
            .events
            .trace_messages
            .iter()
            .map(|message| message.message.clone())
            .collect::<Vec<_>>(),
        expected
            .trace_messages
            .iter()
            .map(|message| message.message.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn compiled_trace_formats_match_evaluator_and_xls_syntax() {
    let ir = r#"package test

fn f(x: bits[12] id=1, neg: bits[8] id=2) -> token {
  t: token = after_all(id=3)
  one: bits[1] = literal(value=1, id=4)
  ret tr: token = trace(t, one, format="literal={{ default={} u={:u} d={:d} x={:x} 0x={:0x} #x={:#x} b={:b} 0b={:0b} #b={:#b}", data_operands=[x, x, neg, x, x, x, x, x, x], id=5)
}
"#;
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    let compiler = PirFunctionCompiler::compile(function).expect("function should compile");
    let args = vec![bits(12, 43), bits(8, 251)];
    let actual = compiler
        .run_ir_values_with_events(&args)
        .expect("compiled execution should succeed");
    let expected = match eval_fn(function, &args) {
        FnEvalResult::Success(success) => success,
        FnEvalResult::Failure(_) => panic!("trace should not cause evaluation failure"),
    };
    let actual_messages = actual
        .events
        .trace_messages
        .iter()
        .map(|message| message.message.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        actual_messages,
        expected
            .trace_messages
            .iter()
            .map(|message| message.message.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual_messages,
        vec![
            "literal={{ default=43 u=43 d=-5 x=2b 0x=02b #x=0x2b b=101011 0b=0000_0010_1011 #b=0b10_1011"
                .to_string()
        ]
    );
}

#[test]
fn odd_width_arithmetic_masks_to_pir_width() {
    let compiler = compile(
        r#"package test

fn f(x: bits[3] id=1, y: bits[3] id=2) -> bits[3] {
  ret sum: bits[3] = add(x, y, id=3)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[7, 3]).expect("execute"), 2);
}

#[test]
fn lowers_bitwise_comparison_extension_and_slice_nodes() {
    let compiler = compile(
        r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2) -> bits[5] {
  both: bits[5] = and(x, y, id=3)
  low: bits[3] = bit_slice(both, start=1, width=3, id=4)
  ret widened: bits[5] = zero_ext(low, new_bit_count=5, id=5)
}
"#,
    );
    assert_eq!(
        compiler.run_u64(&[0b1_1110, 0b0_1111]).expect("execute"),
        0b0_0111
    );
    assert_eq!(
        compiler.run_u64(&[0b1_0100, 0b0_1111]).expect("execute"),
        0b0_0010
    );
}

#[test]
fn rejects_unvalidated_out_of_bounds_static_bit_slice() {
    let package = Parser::new(
        r#"package test

fn f(x: bits[8] id=1) -> bits[2] {
  ret bit_slice.2: bits[2] = bit_slice(x, start=7, width=2, id=2)
}
"#,
    )
    .parse_package()
    .expect("unvalidated function should parse as PIR");
    let function = package.get_fn("f").expect("function f should exist");
    let error = match PirFunctionCompiler::compile(function) {
        Ok(_) => panic!("out-of-bounds bit_slice should not compile"),
        Err(error) => error,
    };
    match error {
        CompilerError::InvalidFunction(message) => assert!(
            message.contains("bit_slice start 7") && message.contains("exceeds operand width 8"),
            "unexpected validation diagnostic: {message}"
        ),
        error => panic!("expected InvalidFunction, got {error}"),
    }
}

#[test]
fn signed_comparisons_use_logical_bit_width() {
    let compiler = compile(
        r#"package test

fn f(x: bits[3] id=1, y: bits[3] id=2) -> bits[1] {
  ret result: bits[1] = slt(x, y, id=3)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[0b111, 0b001]).expect("execute"), 1);
    assert_eq!(compiler.run_u64(&[0b010, 0b111]).expect("execute"), 0);
}

#[test]
fn logical_left_shift_returns_zero_on_pir_overshift() {
    let compiler = compile(
        r#"package test

fn f(x: bits[8] id=1, amount: bits[4] id=2) -> bits[8] {
  ret shifted: bits[8] = shll(x, amount, id=3)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[3, 2]).expect("execute"), 12);
    assert_eq!(compiler.run_u64(&[3, 8]).expect("execute"), 0);
    assert_eq!(compiler.run_u64(&[3, 15]).expect("execute"), 0);
}

#[test]
fn concat_combines_scalar_values_of_different_widths() {
    let compiler = compile(
        r#"package test

fn f(high: bits[5] id=1, low: bits[3] id=2) -> bits[8] {
  ret joined: bits[8] = concat(high, low, id=3)
}
"#,
    );
    assert_eq!(
        compiler.run_u64(&[0b10101, 0b011]).expect("execute"),
        0b1010_1011
    );
}

#[test]
fn lowers_reverse_and_reduction_operations() {
    let reverse = compile(
        r#"package test

fn f(x: bits[5] id=1) -> bits[5] {
  ret reversed: bits[5] = reverse(x, id=2)
}
"#,
    );
    assert_eq!(reverse.run_u64(&[0b1_0010]).expect("execute"), 0b0_1001);

    let reductions = compile(
        r#"package test

fn f(x: bits[5] id=1) -> bits[3] {
  any: bits[1] = or_reduce(x, id=2)
  all: bits[1] = and_reduce(x, id=3)
  parity: bits[1] = xor_reduce(x, id=4)
  ret combined: bits[3] = concat(any, all, parity, id=5)
}
"#,
    );
    assert_eq!(reductions.run_u64(&[0]).expect("execute"), 0b000);
    assert_eq!(reductions.run_u64(&[0b1_1111]).expect("execute"), 0b111);
    assert_eq!(reductions.run_u64(&[0b1_0011]).expect("execute"), 0b101);
}

#[test]
fn right_shifts_and_divmod_match_pir_evaluator_on_edge_values() {
    for op in ["shrl", "shra", "udiv", "sdiv", "umod", "smod"] {
        let ir = format!(
            r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2) -> bits[5] {{
  ret result: bits[5] = {op}(x, y, id=3)
}}
"#
        );
        let args = (0..32)
            .flat_map(|x| (0..32).map(move |y| vec![bits(5, x), bits(5, y)]))
            .collect::<Vec<_>>();
        assert_matches_evaluator(&ir, &args);
    }
    for width in [8, 16, 32, 64] {
        let minimum = 1u64 << (width - 1);
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let ir = format!(
            r#"package test

fn f(x: bits[{width}] id=1, y: bits[{width}] id=2) -> bits[{width}] {{
  ret result: bits[{width}] = sdiv(x, y, id=3)
}}
"#
        );
        assert_matches_evaluator(
            &ir,
            &[
                vec![bits(width, minimum), bits(width, mask)],
                vec![bits(width, minimum), bits(width, 0)],
                vec![bits(width, mask >> 1), bits(width, 0)],
            ],
        );
    }
}

#[test]
fn arbitrary_width_multiply_matches_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[24] id=1, rhs: bits[24] id=2) -> bits[48] {
  ret result: bits[48] = umul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(24, 0), bits(24, 0x12_3456)],
            vec![bits(24, 0xff_ffff), bits(24, 0xab_cdef)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[7] id=1, rhs: bits[32] id=2) -> bits[32] {
  ret result: bits[32] = umul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(7, 0x7f), bits(32, 0x1234_5678)],
            vec![bits(7, 3), bits(32, 17)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[5] id=1, rhs: bits[9] id=2) -> bits[20] {
  ret result: bits[20] = smul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(5, 0x1f), bits(9, 3)],
            vec![bits(5, 0x11), bits(9, 0x1ff)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[10] id=1, rhs: bits[9] id=2) -> bits[6] {
  ret result: bits[6] = smul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(10, 0x3ff), bits(9, 3)],
            vec![bits(10, 0x201), bits(9, 0x1ff)],
        ],
    );
}

#[test]
fn gate_select_and_encoding_operations_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(pred: bits[1] id=1, value: bits[8] id=2) -> bits[8] {
  ret result: bits[8] = gate(pred, value, id=3)
}
"#,
        &[
            vec![bits(1, 0), bits(8, 0xab)],
            vec![bits(1, 1), bits(8, 0xab)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[3] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  selected: bits[8] = sel(selector, cases=[a, b], default=d, id=5)
  priority: bits[8] = priority_sel(selector, cases=[a, b, selected], default=d, id=6)
  ret result: bits[8] = one_hot_sel(selector, cases=[priority, b, a], id=7)
}
"#,
        &(0..8)
            .map(|selector| {
                vec![
                    bits(3, selector),
                    bits(8, 0x12),
                    bits(8, 0x41),
                    bits(8, 0x80),
                ]
            })
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[5] id=1) -> bits[3] {
  hot: bits[6] = one_hot(x, lsb_prio=true, id=2)
  ret encoded: bits[3] = encode(hot, id=3)
}
"#,
        &(0..32).map(|x| vec![bits(5, x)]).collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[3] id=1) -> bits[6] {
  ret decoded: bits[6] = decode(x, width=6, id=2)
}
"#,
        &(0..8).map(|x| vec![bits(3, x)]).collect::<Vec<_>>(),
    );
}

#[test]
fn literal_array_index_uses_xls_out_of_bounds_clamping() {
    let compiler = compile(
        r#"package test

fn f(index: bits[4] id=1) -> bits[3] {
  values: bits[3][4] = literal(value=[0, 2, 4, 6], id=2)
  ret selected: bits[3] = array_index(values, indices=[index], id=3)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[0]).expect("execute"), 0);
    assert_eq!(compiler.run_u64(&[2]).expect("execute"), 4);
    assert_eq!(compiler.run_u64(&[15]).expect("execute"), 6);
}

#[test]
fn bounded_literal_array_index_accepts_assumed_in_bounds() {
    let compiler = compile(
        r#"package test

fn f(index: bits[2] id=1) -> bits[3] {
  values: bits[3][4] = literal(value=[0, 2, 4, 6], id=2)
  ret selected: bits[3] = array_index(values, indices=[index], assumed_in_bounds=true, id=3)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[3]).expect("execute"), 6);
}

#[test]
fn statically_safe_assumed_in_bounds_nodes_do_not_create_event_sites() {
    let compiler = compile(
        r#"package test

fn f(a: bits[8][4] id=1, v: bits[8] id=2, i: bits[2] id=3) -> bits[8] {
  dead_index: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=4)
  dead_update: bits[8][4] = array_update(a, v, indices=[i], assumed_in_bounds=true, id=5)
  ret out: bits[8] = identity(v, id=6)
}
"#,
    );
    assert!(compiler.metadata().event_sites.is_empty());
    assert_eq!(compiler.scratch_byte_count(), 0);
}

#[test]
fn assumed_in_bounds_array_violations_accumulate_for_retained_graph_nodes() {
    let compiler = compile(
        r#"package test

fn f(a: bits[8][2] id=1, v: bits[8] id=2, i: bits[2] id=3) -> bits[8] {
  dead_index: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=4)
  dead_update: bits[8][2] = array_update(a, v, indices=[i], assumed_in_bounds=true, id=5)
  ret out: bits[8] = identity(v, id=6)
}
"#,
    );
    let values = array(8, &[10, 11]);
    let in_bounds = compiler
        .run_ir_values_with_events(&[values.clone(), bits(8, 99), bits(2, 1)])
        .expect("in-bounds execution");
    assert!(in_bounds.events.assumption_failures.is_empty());

    let out_of_bounds = compiler
        .run_ir_values_with_events(&[values, bits(8, 99), bits(2, 3)])
        .expect("out-of-bounds execution remains safe");
    assert_eq!(out_of_bounds.value, bits(8, 99));
    assert_eq!(
        sorted(
            out_of_bounds
                .events
                .assumption_failures
                .iter()
                .map(|failure| (failure.node_text_id, failure.kind))
                .collect::<Vec<_>>()
        ),
        vec![
            (4, AssumptionFailureKind::ArrayIndexOutOfBounds),
            (5, AssumptionFailureKind::ArrayUpdateOutOfBounds),
        ]
    );
}

#[test]
fn native_scalar_array_input_uses_c_array_layout() {
    let compiler = compile(
        r#"package test

fn f(values: bits[9][4] id=1, index: bits[3] id=2) -> bits[9] {
  ret selected: bits[9] = array_index(values, indices=[index], id=3)
}
"#,
    );
    assert_eq!(
        compiler.param_layouts()[0],
        NativeValueLayout::Array {
            element: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                bit_count: 9,
                byte_count: 2,
            })),
            element_count: 4,
        }
    );
    assert_eq!(
        compiler.param_layouts()[0].byte_count(),
        std::mem::size_of::<[u16; 4]>()
    );
    assert_eq!(
        compiler.param_layouts()[0].alignment(),
        std::mem::align_of::<[u16; 4]>()
    );
    assert_eq!(
        compiler.param_layouts()[0].element_stride(),
        Some(std::mem::size_of::<u16>())
    );

    let values: [u16; 4] = [3, 101, 255, 509];
    let index: u8 = 2;
    let mut output: u16 = 0;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: `[u16; 4]` is exactly the native layout for `bits[9][4]`;
    // `index` and `output` use the scalar layouts selected by the compiler.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native array indexing should execute");
    }
    assert_eq!(output, 255);

    let index: u8 = 7;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: same native storage contract as the call above.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("out-of-bounds index should clamp");
    }
    assert_eq!(output, 509);
}

#[test]
fn native_array_result_is_written_to_c_array_storage() {
    let compiler = compile(
        r#"package test

fn f(x: bits[9] id=1, y: bits[9] id=2) -> bits[9][3] {
  ret made: bits[9][3] = array(x, y, x, id=3)
}
"#,
    );
    let x: u16 = 12;
    let y: u16 = 300;
    let mut output: [u16; 3] = [0; 3];
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
    ];
    // SAFETY: the inputs and output use the published native layouts; the
    // output is directly writable as a C-compatible array of `u16`.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native array construction should execute");
    }
    assert_eq!(output, [12, 300, 12]);
}

#[test]
fn nested_native_arrays_use_recursive_c_array_strides() {
    let compiler = compile(
        r#"package test

fn f(values: bits[8][2][2] id=1, i: bits[2] id=2, j: bits[2] id=3) -> bits[8] {
  ret selected: bits[8] = array_index(values, indices=[i, j], id=4)
}
"#,
    );
    let values: [[u8; 2]; 2] = [[1, 2], [3, 4]];
    let i: u8 = 1;
    let j: u8 = 0;
    let mut output: u8 = 0;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&i).cast::<u8>(),
        std::ptr::from_ref(&j).cast::<u8>(),
    ];
    // SAFETY: nested Rust arrays are contiguous recursive native array
    // storage corresponding to `bits[8][2][2]`.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("nested native array indexing should execute");
    }
    assert_eq!(output, 3);
}

#[test]
fn native_subarray_result_copies_from_nested_array_storage() {
    let compiler = compile(
        r#"package test

fn f(values: bits[9][2][2] id=1, index: bits[1] id=2) -> bits[9][2] {
  ret selected: bits[9][2] = array_index(values, indices=[index], id=3)
}
"#,
    );
    let values: [[u16; 2]; 2] = [[10, 11], [400, 401]];
    let index: u8 = 1;
    let mut output: [u16; 2] = [0; 2];
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: nested input and separate output arrays obey the recursive
    // native layout contract and do not overlap.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("subarray result should execute");
    }
    assert_eq!(output, [400, 401]);
}

#[test]
fn intermediate_array_construction_uses_scratch_storage() {
    let compiler = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2, index: bits[2] id=3) -> bits[8] {
  made: bits[8][2] = array(x, y, id=4)
  ret selected: bits[8] = array_index(made, indices=[index], id=5)
}
"#,
    );
    assert_eq!(compiler.run_u64(&[7, 13, 0]).expect("execute"), 7);
    assert_eq!(compiler.run_u64(&[7, 13, 1]).expect("execute"), 13);
    assert_eq!(compiler.run_u64(&[7, 13, 3]).expect("execute"), 13);

    let x: u8 = 7;
    let y: u8 = 13;
    let index: u8 = 1;
    let mut output = 0u8;
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    let mut scratch = vec![
        0u64;
        compiler
            .scratch_byte_count()
            .div_ceil(std::mem::size_of::<u64>())
    ];
    assert!(compiler.scratch_byte_count() > 0);
    assert!(compiler.scratch_alignment() <= std::mem::align_of::<u64>());
    // SAFETY: all values use their native scalar layouts, and `scratch` is
    // sufficiently sized and aligned for the published scratch requirement.
    unsafe {
        compiler
            .run_native_with_scratch(
                &inputs,
                std::ptr::from_mut(&mut output).cast(),
                scratch.as_mut_ptr().cast(),
                scratch.len() * std::mem::size_of::<u64>(),
            )
            .expect("execution with caller-owned scratch should succeed");
    }
    assert_eq!(output, 13);
}

#[test]
fn zero_index_array_index_preserves_native_array_value() {
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1) -> bits[8][2] {
  ret result: bits[8][2] = array_index(values, indices=[], id=2)
}
"#,
        &[vec![array(8, &[9, 27])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1, replacement: bits[8][2] id=2) -> bits[8][2] {
  ret result: bits[8][2] = array_update(values, replacement, indices=[], id=3)
}
"#,
        &[vec![array(8, &[9, 27]), array(8, &[41, 63])]],
    );
}

#[test]
fn native_array_concat_slice_and_update_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[8][1] id=1, middle: bits[8][2] id=2, rhs: bits[8][1] id=3) -> bits[8][4] {
  ret joined: bits[8][4] = array_concat(lhs, middle, rhs, id=4)
}
"#,
        &[vec![array(8, &[1]), array(8, &[2, 3]), array(8, &[4])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1) -> bits[8][2] {
  ret joined: bits[8][2] = array_concat(values, id=2)
}
"#,
        &[vec![array(8, &[7, 9])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][3] id=1, start: bits[3] id=2) -> bits[8][4] {
  ret sliced: bits[8][4] = array_slice(values, start, width=4, id=3)
}
"#,
        &(0..8)
            .map(|start| vec![array(8, &[10, 20, 30]), bits(3, start)])
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2][2] id=1, replacement: bits[8] id=2, i: bits[2] id=3, j: bits[2] id=4) -> bits[8][2][2] {
  ret updated: bits[8][2][2] = array_update(values, replacement, indices=[i, j], id=5)
}
"#,
        &(0..4)
            .flat_map(|i| {
                (0..4).map(move |j| {
                    vec![
                        IrValue::make_array(&[array(8, &[1, 2]), array(8, &[3, 4])]).unwrap(),
                        bits(8, 99),
                        bits(2, i),
                        bits(2, j),
                    ]
                })
            })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn zero_sized_aggregates_flow_through_array_operations() {
    assert_matches_evaluator(
        r#"package test

fn f(values: ()[1] id=1) -> ()[1] {
  ret joined: ()[1] = array_concat(values, id=2)
}
"#,
        &[vec![unit_array(1)]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: ()[2] id=1, index: bits[1] id=2) -> () {
  ret selected: () = array_index(values, indices=[index], id=3)
}
"#,
        &[vec![unit_array(2), bits(1, 1)]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: ()[1] id=1, start: bits[1] id=2) -> ()[2] {
  ret sliced: ()[2] = array_slice(values, start, width=2, id=3)
}
"#,
        &[vec![unit_array(1), bits(1, 0)]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: ()[2] id=1, replacement: () id=2, index: bits[1] id=3) -> ()[2] {
  ret updated: ()[2] = array_update(values, replacement, indices=[index], id=4)
}
"#,
        &[vec![unit_array(2), tuple(&[]), bits(1, 1)]],
    );
}

#[test]
fn selection_and_gate_operate_on_native_arrays() {
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[2] id=1, pred: bits[1] id=2, a: bits[8][2] id=3, b: bits[8][2] id=4, d: bits[8][2] id=5) -> bits[8][2] {
  chosen: bits[8][2] = sel(selector, cases=[a, b], default=d, id=6)
  gated: bits[8][2] = gate(pred, chosen, id=7)
  ret result: bits[8][2] = one_hot_sel(selector, cases=[gated, b], id=8)
}
"#,
        &[
            vec![
                bits(2, 0),
                bits(1, 0),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
            vec![
                bits(2, 1),
                bits(1, 1),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
            vec![
                bits(2, 3),
                bits(1, 1),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
        ],
    );
}

#[test]
fn aggregate_sel_and_priority_sel_alias_input_storage_without_scratch() {
    let sel = compile(
        r#"package test

fn f(selector: bits[1] id=1, a: bits[8][2] id=2, b: bits[8][2] id=3, index: bits[1] id=4) -> bits[8] {
  selected: bits[8][2] = sel(selector, cases=[a], default=b, id=5)
  ret result: bits[8] = array_index(selected, indices=[index], id=6)
}
"#,
    );
    assert_eq!(sel.scratch_byte_count(), 0);
    assert_eq!(
        sel.run_ir_values(&[
            bits(1, 0),
            array(8, &[11, 13]),
            array(8, &[17, 19]),
            bits(1, 0),
        ])
        .expect("aggregate sel execution"),
        bits(8, 11)
    );

    let priority_sel = compile(
        r#"package test

fn f(selector: bits[2] id=1, a: bits[8][2] id=2, b: bits[8][2] id=3, d: bits[8][2] id=4, index: bits[1] id=5) -> bits[8] {
  selected: bits[8][2] = priority_sel(selector, cases=[a, b], default=d, id=6)
  ret result: bits[8] = array_index(selected, indices=[index], id=7)
}
"#,
    );
    assert_eq!(priority_sel.scratch_byte_count(), 0);
    assert_eq!(
        priority_sel
            .run_ir_values(&[
                bits(2, 2),
                array(8, &[11, 13]),
                array(8, &[17, 19]),
                array(8, &[23, 29]),
                bits(1, 1),
            ])
            .expect("aggregate priority_sel execution"),
        bits(8, 19)
    );
}

#[test]
fn aggregate_gate_aliases_true_value_and_uses_shared_zero_storage() {
    let compiler = compile(
        r#"package test

fn f(pred: bits[1] id=1, values: bits[8][4] id=2, index: bits[2] id=3) -> bits[8] {
  gated: bits[8][4] = gate(pred, values, id=4)
  ret result: bits[8] = array_index(gated, indices=[index], id=5)
}
"#,
    );
    assert_eq!(compiler.scratch_byte_count(), 4);
    assert_eq!(
        compiler
            .run_ir_values(&[bits(1, 1), array(8, &[3, 5, 7, 11]), bits(2, 2)])
            .expect("enabled aggregate gate execution"),
        bits(8, 7)
    );
    assert_eq!(
        compiler
            .run_ir_values(&[bits(1, 0), array(8, &[3, 5, 7, 11]), bits(2, 2)])
            .expect("disabled aggregate gate execution"),
        bits(8, 0)
    );
}

#[test]
fn scratch_slots_are_reused_after_last_use_and_preserved_through_aliases() {
    let reusable = compile(
        r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2, index: bits[1] id=3) -> bits[8] {
  first: bits[8][2] = array(a, b, id=4)
  selected: bits[8] = array_index(first, indices=[index], id=5)
  second: bits[8][2] = array(selected, b, id=6)
  ret result: bits[8] = array_index(second, indices=[index], id=7)
}
"#,
    );
    assert_eq!(reusable.scratch_byte_count(), 2);
    assert_eq!(
        reusable
            .run_ir_values(&[bits(8, 7), bits(8, 13), bits(1, 0)])
            .expect("reused scratch execution"),
        bits(8, 7)
    );

    let preserved = compile(
        r#"package test

fn f(selector: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, index: bits[1] id=4) -> bits[8] {
  first: bits[8][2] = array(a, b, id=5)
  selected: bits[8][2] = sel(selector, cases=[first], default=first, id=6)
  second: bits[8][2] = array(b, a, id=7)
  rhs: bits[8] = array_index(second, indices=[index], id=8)
  lhs: bits[8] = array_index(selected, indices=[index], id=9)
  ret result: bits[8] = add(lhs, rhs, id=10)
}
"#,
    );
    assert_eq!(preserved.scratch_byte_count(), 4);
    assert_eq!(
        preserved
            .run_ir_values(&[bits(1, 1), bits(8, 7), bits(8, 13), bits(1, 0)])
            .expect("transitively preserved scratch execution"),
        bits(8, 20)
    );
}

#[test]
fn array_update_reuses_dead_scratch_storage_but_preserves_live_sources() {
    let reusable = compile(
        r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2, replacement: bits[8] id=3, index: bits[1] id=4) -> bits[8] {
  original: bits[8][2] = array(a, b, id=5)
  updated: bits[8][2] = array_update(original, replacement, indices=[index], id=6)
  ret result: bits[8] = array_index(updated, indices=[index], id=7)
}
"#,
    );
    assert_eq!(reusable.scratch_byte_count(), 2);
    assert_eq!(
        reusable
            .run_ir_values(&[bits(8, 7), bits(8, 13), bits(8, 41), bits(1, 0)])
            .expect("in-place array update execution"),
        bits(8, 41)
    );

    let preserved = compile(
        r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2, replacement: bits[8] id=3, index: bits[1] id=4) -> bits[8] {
  original: bits[8][2] = array(a, b, id=5)
  updated: bits[8][2] = array_update(original, replacement, indices=[index], id=6)
  before: bits[8] = array_index(original, indices=[index], id=7)
  after: bits[8] = array_index(updated, indices=[index], id=8)
  ret result: bits[8] = add(before, after, id=9)
}
"#,
    );
    assert_eq!(preserved.scratch_byte_count(), 4);
    assert_eq!(
        preserved
            .run_ir_values(&[bits(8, 7), bits(8, 13), bits(8, 41), bits(1, 0)])
            .expect("preserved array update source execution"),
        bits(8, 48)
    );
}

#[test]
fn aggregate_comparisons_and_default_only_sel_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: (bits[8], bits[4][2]) id=1, rhs: (bits[8], bits[4][2]) id=2) -> (bits[1], bits[1]) {
  same: bits[1] = eq(lhs, rhs, id=3)
  different: bits[1] = ne(lhs, rhs, id=4)
  ret result: (bits[1], bits[1]) = tuple(same, different, id=5)
}
"#,
        &[
            vec![
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
            ],
            vec![
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
                tuple(&[bits(8, 3), array(4, &[1, 4])]),
            ],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[2] id=1, fallback: bits[8][2] id=2) -> bits[8][2] {
  ret result: bits[8][2] = sel(selector, cases=[], default=fallback, id=3)
}
"#,
        &[vec![bits(2, 3), array(8, &[11, 27])]],
    );
}

#[test]
fn native_tuple_layout_matches_repr_c_storage() {
    #[repr(C)]
    struct Pair {
        low: u8,
        high: u32,
    }

    let compiler = compile(
        r#"package test

fn f(low: bits[8] id=1, high: bits[17] id=2) -> (bits[8], bits[17]) {
  ret pair: (bits[8], bits[17]) = tuple(low, high, id=3)
}
"#,
    );
    assert_eq!(
        compiler.result_layout(),
        &NativeValueLayout::Tuple {
            fields: vec![
                NativeTupleFieldLayout {
                    layout: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                        bit_count: 8,
                        byte_count: 1,
                    })),
                    offset: std::mem::offset_of!(Pair, low),
                },
                NativeTupleFieldLayout {
                    layout: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                        bit_count: 17,
                        byte_count: 4,
                    })),
                    offset: std::mem::offset_of!(Pair, high),
                },
            ],
            byte_count: std::mem::size_of::<Pair>(),
            alignment: std::mem::align_of::<Pair>(),
        }
    );

    let low: u8 = 0x91;
    let high: u32 = 0x1_2345;
    let mut output = Pair { low: 0, high: 0 };
    let inputs = [
        std::ptr::from_ref(&low).cast::<u8>(),
        std::ptr::from_ref(&high).cast::<u8>(),
    ];
    // SAFETY: `Pair` has the `#[repr(C)]` tuple layout published by the compiler,
    // and the scalar inputs use their respective native carriers.
    unsafe {
        compiler
            .run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native tuple construction should execute");
    }
    assert_eq!(output.low, low);
    assert_eq!(output.high, high);
}

#[test]
fn nested_tuples_arrays_and_tuple_selection_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[1] id=1, x: bits[8] id=2, y: bits[8] id=3) -> ((bits[8], bits[8][2]), bits[8]) {
  values: bits[8][2] = array(x, y, id=4)
  inner: (bits[8], bits[8][2]) = tuple(x, values, id=5)
  selected: (bits[8], bits[8][2]) = sel(selector, cases=[inner], default=inner, id=6)
  gated: (bits[8], bits[8][2]) = gate(selector, selected, id=7)
  ret result: ((bits[8], bits[8][2]), bits[8]) = tuple(gated, y, id=8)
}
"#,
        &[
            vec![bits(1, 0), bits(8, 3), bits(8, 9)],
            vec![bits(1, 1), bits(8, 3), bits(8, 9)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(value: ((bits[8], bits[8][2]), bits[8]) id=1) -> bits[8] {
  inner: (bits[8], bits[8][2]) = tuple_index(value, index=0, id=2)
  values: bits[8][2] = tuple_index(inner, index=1, id=3)
  index: bits[1] = literal(value=1, id=4)
  ret result: bits[8] = array_index(values, indices=[index], id=5)
}
"#,
        &[vec![tuple(&[
            tuple(&[bits(8, 4), array(8, &[7, 11])]),
            bits(8, 17),
        ])]],
    );
}

#[test]
fn dynamic_bit_slice_and_update_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[8] id=1, start: bits[4] id=2, replacement: bits[5] id=3) -> (bits[5], bits[8]) {
  sliced: bits[5] = dynamic_bit_slice(x, start, width=5, id=4)
  updated: bits[8] = bit_slice_update(x, start, replacement, id=5)
  ret result: (bits[5], bits[8]) = tuple(sliced, updated, id=6)
}
"#,
        &(0..16)
            .map(|start| vec![bits(8, 0b1010_0110), bits(4, start), bits(5, 0b1_1101)])
            .collect::<Vec<_>>(),
    );
}

#[test]
fn extension_operations_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2, c_in: bits[1] id=3, count: bits[4] id=4) -> (bits[1], bits[3], bits[3], bits[4], (bits[8], bits[4]), bits[8], bits[8]) {
  carry: bits[1] = ext_carry_out(x, y, c_in, id=5)
  prio_lsb: bits[3] = ext_prio_encode(x,lsb_prio=true,id=6)
  prio_msb: bits[3] = ext_prio_encode(x,lsb_prio=false,id=7)
  zeroes: bits[4] = ext_clz(x, offset=2, new_bit_count=4, id=8)
  normalized: (bits[8], bits[4]) = ext_normalize_left(x, shift_offset=1, normalized_bit_count=8, clz_bit_count=4, id=9)
  mask: bits[8] = ext_mask_low(count, id=10)
  sum: bits[8] = ext_nary_add(x, y, count, signed=[true, false, false], negated=[false, true, false], arch=brent_kung, id=11)
  ret result: (bits[1], bits[3], bits[3], bits[4], (bits[8], bits[4]), bits[8], bits[8]) = tuple(carry, prio_lsb, prio_msb, zeroes, normalized, mask, sum, id=12)
}
"#,
        &(0..32)
            .flat_map(|x| {
                [
                    vec![bits(5, x), bits(5, 0), bits(1, 0), bits(4, 0)],
                    vec![bits(5, x), bits(5, 31), bits(1, 1), bits(4, 9)],
                ]
            })
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[64] id=1, rhs: bits[64] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret carry: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)
}
"#,
        &[
            vec![bits(64, u64::MAX), bits(64, 0), bits(1, 1)],
            vec![bits(64, u64::MAX), bits(64, 1), bits(1, 0)],
            vec![bits(64, 1), bits(64, 1), bits(1, 0)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f() -> bits[7] {
  ret sum: bits[7] = ext_nary_add(signed=[], negated=[], id=1)
}
"#,
        &[vec![]],
    );
}

#[test]
fn partial_product_multiply_tuples_match_the_llvm_jit_convention() {
    let unsigned = compile(
        r#"package test

fn f(lhs: bits[4] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = umulp(lhs, rhs, id=3)
}
"#,
    );
    assert_eq!(
        unsigned
            .run_ir_values(&[bits(4, 3), bits(4, 5)])
            .expect("unsigned partial multiply should execute"),
        tuple(&[bits(8, 0x47), bits(8, 0xc8)])
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[4] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = umulp(lhs, rhs, id=3)
}
"#,
        &(0..16)
            .flat_map(|lhs| (0..16).map(move |rhs| vec![bits(4, lhs), bits(4, rhs)]))
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[5] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = smulp(lhs, rhs, id=3)
}
"#,
        &(0..32)
            .flat_map(|lhs| (0..16).map(move |rhs| vec![bits(5, lhs), bits(4, rhs)]))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn ir_value_adapter_matches_native_result() {
    let compiler = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret difference: bits[8] = sub(x, y, id=3)
}
"#,
    );
    let result = compiler
        .run_ir_values(&[
            IrValue::make_ubits(8, 3).unwrap(),
            IrValue::make_ubits(8, 5).unwrap(),
        ])
        .expect("execute");
    assert_eq!(result, IrValue::make_ubits(8, 254).unwrap());
}

#[test]
fn wide_native_limb_storage_and_direct_lowerings_match_evaluator() {
    let identity = compile(
        r#"package test

fn f(x: bits[129] id=1) -> bits[129] {
  ret value: bits[129] = identity(x, id=2)
}
"#,
    );
    assert_eq!(
        identity.result_layout(),
        &NativeValueLayout::WideBits(WideBitsLayout {
            bit_count: 129,
            limb_count: 3,
        })
    );
    let input = [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210, 1];
    let mut output = [0u64; 3];
    let inputs = [input.as_ptr().cast::<u8>()];
    // SAFETY: `input` and `output` are three aligned, least-significant-first
    // u64 limbs, matching the published `bits[129]` native layout.
    unsafe {
        identity
            .run_native(&inputs, output.as_mut_ptr().cast::<u8>())
            .expect("wide native identity execution");
    }
    assert_eq!(output, input);

    assert_matches_evaluator(
        r#"package test

fn f(x: bits[129] id=1, y: bits[129] id=2, c: bits[1] id=3) -> (bits[129], bits[129], bits[1], bits[128], bits[130], bits[1], bits[130], bits[13]) {
  inverted: bits[129] = not(x, id=4)
  sum: bits[129] = add(x, y, id=5)
  less: bits[1] = ult(x, y, id=6)
  sliced: bits[128] = bit_slice(sum, start=1, width=128, id=7)
  joined: bits[130] = concat(c, x, id=8)
  carry: bits[1] = ext_carry_out(x, y, c, id=9)
  wide_sum: bits[130] = ext_nary_add(x, y, c, signed=[true, false, false], negated=[false, true, false], arch=ripple_carry, id=10)
  low_sum: bits[13] = ext_nary_add(x, y, signed=[true, false], negated=[false, true], arch=kogge_stone, id=11)
  ret result: (bits[129], bits[129], bits[1], bits[128], bits[130], bits[1], bits[130], bits[13]) = tuple(inverted, sum, less, sliced, joined, carry, wide_sum, low_sum, id=12)
}
"#,
        &[
            vec![
                wide_bits("bits[129]:0x1_0123_4567_89ab_cdef_fedc_ba98_7654_3210"),
                wide_bits("bits[129]:0x0_ffff_ffff_ffff_ffff_0000_0000_0000_0001"),
                bits(1, 1),
            ],
            vec![
                wide_bits("bits[129]:0x1_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
                wide_bits("bits[129]:0x0_0000_0000_0000_0000_0000_0000_0000_0001"),
                bits(1, 0),
            ],
        ],
    );
}

#[test]
fn wide_runtime_backed_operations_match_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[129] id=1, y: bits[129] id=2, shift: bits[8] id=3, replacement: bits[73] id=4) -> (bits[129], bits[129], bits[129], bits[129], bits[129], bits[96], bits[32], bits[129]) {
  product: bits[129] = umul(x, y, id=5)
  signed_product: bits[129] = smul(x, y, id=6)
  quotient: bits[129] = udiv(x, y, id=7)
  left: bits[129] = shll(x, shift, id=8)
  right: bits[129] = shra(x, shift, id=9)
  slice: bits[96] = dynamic_bit_slice(x, shift, width=96, id=10)
  low_slice: bits[32] = dynamic_bit_slice(x, shift, width=32, id=11)
  updated: bits[129] = bit_slice_update(x, shift, replacement, id=12)
  ret result: (bits[129], bits[129], bits[129], bits[129], bits[129], bits[96], bits[32], bits[129]) = tuple(product, signed_product, quotient, left, right, slice, low_slice, updated, id=13)
}
"#,
        &[
            vec![
                wide_bits("bits[129]:0x1_0000_0000_0000_0000_0123_4567_89ab_cdef"),
                wide_bits("bits[129]:0x0_0000_0000_0000_0000_0000_0000_0000_0003"),
                bits(8, 63),
                wide_bits("bits[73]:0x1ab_cdef0_1234_5678"),
            ],
            vec![
                wide_bits("bits[129]:0x1_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
                wide_bits("bits[129]:0x0"),
                bits(8, 130),
                wide_bits("bits[73]:0x100_0000_0000_0001"),
            ],
        ],
    );
}

#[test]
fn direct_wide_dynamic_bit_slice_does_not_require_runtime_scratch() {
    let ir = r#"package test

fn f(x: bits[257] id=1, shift: bits[129] id=2) -> bits[96] {
  ret sliced: bits[96] = dynamic_bit_slice(x, shift, width=96, id=3)
}
"#;
    let compiler = compile(ir);
    assert_eq!(compiler.scratch_byte_count(), 0);
    assert_matches_evaluator(
        ir,
        &[
            vec![
                wide_bits(
                    "bits[257]:0x1_0123_4567_89ab_cdef_fedc_ba98_7654_3210_55aa_aa55_1234_5678",
                ),
                wide_bits("bits[129]:0x3f"),
            ],
            vec![
                wide_bits(
                    "bits[257]:0x1_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff",
                ),
                wide_bits("bits[129]:0x100_0000_0000_0000_0000_0000_0000_0000"),
            ],
        ],
    );
}

#[test]
fn direct_wide_decode_does_not_require_runtime_scratch() {
    let ir = r#"package test

fn f(index: bits[129] id=1) -> bits[257] {
  ret decoded: bits[257] = decode(index, width=257, id=2)
}
"#;
    let compiler = compile(ir);
    assert_eq!(compiler.scratch_byte_count(), 0);
    assert_matches_evaluator(
        ir,
        &[
            vec![wide_bits("bits[129]:0x0")],
            vec![wide_bits("bits[129]:0x100")],
            vec![wide_bits("bits[129]:0x101")],
            vec![wide_bits(
                "bits[129]:0x1_0000_0000_0000_0000_0000_0000_0000_0000",
            )],
        ],
    );
}

#[test]
fn wide_input_bit_slice_to_i64_scalar_carrier_matches_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[576] id=1) -> bits[44] {
  ret bit_slice.2: bits[44] = bit_slice(x, start=319, width=44, id=2)
}
"#,
        &[vec![wide_bits("bits[576]:0x0")]],
    );
}

#[test]
fn aggregates_can_contain_wide_native_values() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[129] id=1, y: bits[129] id=2, choose: bits[1] id=3, index: bits[1] id=4) -> bits[129] {
  values: bits[129][2] = array(x, y, id=5)
  pair: (bits[129][2], bits[129]) = tuple(values, x, id=6)
  restored: bits[129][2] = tuple_index(pair, index=0, id=7)
  selected: bits[129] = array_index(restored, indices=[index], id=8)
  ret result: bits[129] = sel(choose, cases=[selected], default=y, id=9)
}
"#,
        &[
            vec![
                wide_bits("bits[129]:0x1_0000_0000_0000_0001"),
                wide_bits("bits[129]:0x0_ffff_ffff_ffff_ffff"),
                bits(1, 0),
                bits(1, 1),
            ],
            vec![
                wide_bits("bits[129]:0x1_0000_0000_0000_0001"),
                wide_bits("bits[129]:0x0_ffff_ffff_ffff_ffff"),
                bits(1, 1),
                bits(1, 0),
            ],
        ],
    );
}

#[test]
fn wide_mulp_and_encoding_operations_match_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[129] id=1, y: bits[73] id=2, index: bits[129] id=3) -> ((bits[130], bits[130]), (bits[130], bits[130]), bits[130], bits[8], bits[130]) {
  unsigned_parts: (bits[130], bits[130]) = umulp(x, y, id=4)
  signed_parts: (bits[130], bits[130]) = smulp(x, y, id=5)
  hot: bits[130] = one_hot(x, lsb_prio=true, id=6)
  encoded: bits[8] = encode(hot, id=7)
  decoded: bits[130] = decode(index, width=130, id=8)
  ret result: ((bits[130], bits[130]), (bits[130], bits[130]), bits[130], bits[8], bits[130]) = tuple(unsigned_parts, signed_parts, hot, encoded, decoded, id=9)
}
"#,
        &[
            vec![
                wide_bits("bits[129]:0x1_0000_0000_0000_0000_0000_0000_0000_0001"),
                wide_bits("bits[73]:0x1_0000_0000_0000_0001"),
                wide_bits("bits[129]:0x40"),
            ],
            vec![
                wide_bits("bits[129]:0x0"),
                wide_bits("bits[73]:0x1ff_ffff_ffff_ffff_ffff"),
                wide_bits("bits[129]:0x100"),
            ],
        ],
    );
}

#[test]
fn wide_extension_operations_match_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[129] id=1, count: bits[129] id=2) -> (bits[8], bits[130], (bits[160], bits[130]), bits[160]) {
  priority: bits[8] = ext_prio_encode(x, lsb_prio=false, id=3)
  zeroes: bits[130] = ext_clz(x, offset=7, new_bit_count=130, id=4)
  normalized: (bits[160], bits[130]) = ext_normalize_left(x, shift_offset=3, normalized_bit_count=160, clz_bit_count=130, id=5)
  mask: bits[160] = ext_mask_low(count, id=6)
  ret result: (bits[8], bits[130], (bits[160], bits[130]), bits[160]) = tuple(priority, zeroes, normalized, mask, id=7)
}
"#,
        &[
            vec![
                wide_bits("bits[129]:0x0_0000_0000_0000_0000_0000_0000_0000_0001"),
                wide_bits("bits[129]:0x50"),
            ],
            vec![wide_bits("bits[129]:0x0"), wide_bits("bits[129]:0x200")],
        ],
    );
}

#[test]
fn wide_array_indices_and_slices_match_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(a: bits[129][3] id=1, index: bits[129] id=2, replacement: bits[129] id=3) -> (bits[129], bits[129][3], bits[129][2]) {
  selected: bits[129] = array_index(a, indices=[index], id=4)
  updated: bits[129][3] = array_update(a, replacement, indices=[index], id=5)
  sliced: bits[129][2] = array_slice(a, index, width=2, id=6)
  ret result: (bits[129], bits[129][3], bits[129][2]) = tuple(selected, updated, sliced, id=7)
}
"#,
        &[
            vec![
                IrValue::make_array(&[
                    wide_bits("bits[129]:0x1"),
                    wide_bits("bits[129]:0x2"),
                    wide_bits("bits[129]:0x3"),
                ])
                .unwrap(),
                wide_bits("bits[129]:0x1"),
                wide_bits("bits[129]:0xff"),
            ],
            vec![
                IrValue::make_array(&[
                    wide_bits("bits[129]:0x1"),
                    wide_bits("bits[129]:0x2"),
                    wide_bits("bits[129]:0x3"),
                ])
                .unwrap(),
                wide_bits("bits[129]:0x1_0000_0000_0000_0000"),
                wide_bits("bits[129]:0xff"),
            ],
        ],
    );
}
