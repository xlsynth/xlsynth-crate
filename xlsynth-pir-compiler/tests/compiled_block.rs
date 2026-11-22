// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvalError, BlockEvaluator, RegisterInit, test_fixtures};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::ExecutionOptions;
use xlsynth_pir_compiler::block::{CompiledBlock, CompiledBlockError};

/// Compare structured errors, including all observations made on a failed edge.
fn check_error(expected: &BlockEvalError, actual: &CompiledBlockError) {
    let CompiledBlockError::Evaluation(actual) = actual else {
        panic!("expected an evaluation failure: {actual:?}");
    };
    assert_eq!(
        std::mem::discriminant(expected),
        std::mem::discriminant(actual)
    );
    assert_eq!(expected.to_string(), actual.to_string());
    if let (BlockEvalError::Runtime(expected), BlockEvalError::Runtime(actual)) = (expected, actual)
    {
        assert_eq!(actual.assertion_failures, expected.assertion_failures);
        assert_eq!(actual.assumption_failures, expected.assumption_failures);
        assert_eq!(actual.trace_messages, expected.trace_messages);
        assert_eq!(actual.cover_counts, expected.cover_counts);
    }
}

/// Advance independent states along the same stimulus, retrying failures at the
/// same edge.
fn compare_clocks(source: &str, top: &str, init: RegisterInit, clocks: &[Vec<IrValue>]) {
    let package = Parser::new(source).parse_package().unwrap();
    let interpreter = BlockEvaluator::prepare(&package, top).unwrap();
    let compiled = CompiledBlock::compile(&package, top).unwrap();
    assert_eq!(compiled.inputs(), interpreter.inputs());
    assert_eq!(compiled.outputs(), interpreter.outputs());
    assert_eq!(compiled.registers(), interpreter.registers());
    let mut oracle_state = interpreter.initial_state(init.clone()).unwrap();
    let mut compiled_state = compiled.initial_state(init.clone()).unwrap();
    let mut instance = compiled.prepare_instance(init).unwrap();
    let mut last_outputs = Vec::new();
    assert_eq!(compiled_state.values(), oracle_state.values());
    assert_eq!(instance.snapshot().unwrap().values(), oracle_state.values());
    for inputs in clocks {
        let typed = compiled.step(&compiled_state, inputs);
        let prepared = instance.tick(inputs);
        match interpreter.step(&oracle_state, inputs) {
            Ok(expected) => {
                let typed = typed.unwrap();
                let events = prepared.unwrap();
                assert_eq!(typed.outputs, expected.outputs);
                assert_eq!(typed.trace_messages, expected.trace_messages);
                assert_eq!(typed.cover_counts, expected.cover_counts);
                assert_eq!(events.trace_messages, expected.trace_messages);
                assert_eq!(events.cover_counts, expected.cover_counts);
                last_outputs = expected.outputs;
                compiled_state = typed.next_state;
                oracle_state = expected.next_state;
            }
            Err(error) => {
                check_error(&error, &typed.unwrap_err());
                check_error(&error, &prepared.unwrap_err());
            }
        }
        // A failed tick keeps every output from the last successful edge.
        for (index, output) in last_outputs.iter().enumerate() {
            assert_eq!(&instance.output(index).unwrap(), output);
        }
        assert_eq!(compiled_state.values(), oracle_state.values());
        assert_eq!(instance.snapshot().unwrap().values(), oracle_state.values());
    }
}

#[test]
fn edge_sampling_reset_enable_write_conflict_and_retry() {
    let supplied = RegisterInit::Supplied(vec![
        IrValue::make_ubits(8, 18).unwrap(),
        IrValue::make_ubits(8, 42).unwrap(),
    ]);
    let clocks: Vec<_> = [
        [true, true, true],
        [true, false, false],
        [false, false, true],
        [false, true, false],
    ]
    .map(|bits| bits.map(IrValue::bool).to_vec())
    .into();
    compare_clocks(test_fixtures::MIXED_RESET, "pair", supplied, &clocks);

    let clocks: Vec<_> = [
        [true, true, true],
        [false, true, true],
        [false, false, false],
        [false, false, true],
        [false, true, false],
    ]
    .map(|bits| bits.map(IrValue::bool).to_vec())
    .into();
    compare_clocks(
        test_fixtures::EXCLUSIVE_WRITES,
        "dual",
        RegisterInit::AllZeros,
        &clocks,
    );
}

#[test]
fn aggregate_registers_and_invalid_input_retry() {
    let value = IrValue::parse_typed(
        "(bits[129]:0x100000000000000000000000000000001, [bits[3]:5, bits[3]:2], bits[0]:0)",
    )
    .unwrap();
    compare_clocks(
        test_fixtures::AGGREGATE_STATE,
        "store",
        RegisterInit::Random { seed: 42 },
        &[
            vec![value.clone()],
            vec![],
            vec![IrValue::bool(true)],
            vec![value],
        ],
    );
}

#[test]
fn token_ports_and_register() {
    let source = r#"package token_state
top block pipeline(clk: clock, d: token, out: token) {
  reg saved(token)
  d: token = input_port(name=d, id=1)
  q: token = register_read(register=saved, id=2)
  w: () = register_write(d, register=saved, id=3)
  out: () = output_port(q, name=out, id=4)
}"#;
    compare_clocks(
        source,
        "pipeline",
        RegisterInit::AllZeros,
        &[
            vec![IrValue::make_token()],
            vec![IrValue::bool(false)],
            vec![IrValue::make_token()],
        ],
    );
}

#[test]
fn native_layout_at_width_boundaries_and_empty_tuple() {
    let ty = "(bits[0], bits[1], bits[63], bits[64], bits[65], ())";
    let source = format!(
        r#"package widths
top block pipeline(clk: clock, d: {ty}, out: {ty}) {{
  reg a({ty})
  reg b({ty})
  d: {ty} = input_port(name=d, id=1)
  qa: {ty} = register_read(register=a, id=2)
  qb: {ty} = register_read(register=b, id=3)
  wa: () = register_write(d, register=a, id=4)
  wb: () = register_write(qa, register=b, id=5)
  out: () = output_port(qb, name=out, id=6)
}}"#
    );
    let value = IrValue::parse_typed("(bits[0]:0, bits[1]:1, bits[63]:0x421234567890abcd, bits[64]:0x8123456789abcdef, bits[65]:0x18023456789abcdef, ())").unwrap();
    compare_clocks(
        &source,
        "pipeline",
        RegisterInit::Random { seed: 19 },
        &vec![vec![value]; 3],
    );
}

#[test]
fn shared_children_and_behavioral_externals() {
    compare_clocks(
        test_fixtures::SIBLING_STATE,
        "pair",
        RegisterInit::Random { seed: 7 },
        &[vec![], vec![], vec![]],
    );
    let data = IrValue::make_ubits(8, 19).unwrap();
    compare_clocks(
        test_fixtures::MODELED_EXTERNAL,
        "top",
        RegisterInit::AllZeros,
        &[
            vec![IrValue::bool(false), data.clone()],
            vec![IrValue::bool(true), data],
            vec![],
        ],
    );
}

#[test]
fn stateless_function_call_uses_header_port_order() {
    let a = IrValue::make_ubits(8, 3).unwrap();
    let b = IrValue::make_ubits(8, 9).unwrap();
    let clocks = [false, true].map(|s| vec![b.clone(), IrValue::bool(s), a.clone()]);
    compare_clocks(
        test_fixtures::CALLED_FUNCTION,
        "observed",
        RegisterInit::AllZeros,
        &clocks,
    );
}

#[test]
fn child_bounds_failure_and_valid_retry() {
    let data = IrValue::parse_typed("[bits[8]:11, bits[8]:22]").unwrap();
    let clocks =
        [1, 2, 0, 3, 1].map(|index| vec![data.clone(), IrValue::make_ubits(2, index).unwrap()]);
    compare_clocks(
        test_fixtures::CHILD_BOUNDS,
        "parent",
        RegisterInit::AllZeros,
        &clocks,
    );
}

// Node IDs interleave between the leaf's two cover sites and the ordinary
// callee's cover site. Both siblings execute the same ordinary function.
const CHILD_EVENTS: &str = r#"package events
fn helper(x: bits[1] id=21) -> bits[1] {
  inner: () = cover(x, label="call", id=22)
  ret result: bits[1] = identity(x, id=23)
}
block leaf(x: bits[1]) {
  x: bits[1] = input_port(name=x, id=31)
  first: () = cover(x, label="before", id=37)
  t: token = after_all(id=33)
  called: bits[1] = invoke(x, to_apply=helper, id=34)
  checked: token = assert(t, called, message="missing pulse", label="pulse", id=35)
  tr: token = trace(checked, called, format="pulse={}", data_operands=[x], verbosity=1, id=36)
  second: () = cover(called, label="after", id=32)
}
top block pair(clk: clock, a: bits[1], b: bits[1], q: bits[1]) {
  reg saved(bits[1])
  instantiation left(block=leaf, kind=block)
  instantiation right(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=40)
  b: bits[1] = input_port(name=b, id=41)
  l: () = instantiation_input(a, instantiation=left, port_name=x, id=42)
  r: () = instantiation_input(b, instantiation=right, port_name=x, id=43)
  old: bits[1] = register_read(register=saved, id=44)
  write: () = register_write(a, register=saved, id=45)
  q: () = output_port(old, name=q, id=46)
}"#;

#[test]
fn diagnostics_from_children_and_calls_clear_each_cycle() {
    let clocks = [
        [true, true],
        [false, true],
        [false, false],
        [true, false],
        [true, true],
    ]
    .map(|bits| bits.map(IrValue::bool).to_vec());
    compare_clocks(CHILD_EVENTS, "pair", RegisterInit::AllZeros, &clocks);
    compare_clocks(
        test_fixtures::CALLED_COVER,
        "pair",
        RegisterInit::AllZeros,
        &clocks,
    );
}

#[test]
fn disabling_observations_keeps_assertions_and_bounds_checks() {
    let package = Parser::new(CHILD_EVENTS).parse_package().unwrap();
    let compiled = CompiledBlock::compile(&package, "pair").unwrap();
    let state = compiled.initial_state(RegisterInit::AllZeros).unwrap();
    let error = compiled
        .step_with_options(
            &state,
            &[IrValue::bool(false), IrValue::bool(true)],
            ExecutionOptions::NO_EVENTS,
        )
        .unwrap_err();
    let CompiledBlockError::Evaluation(BlockEvalError::Runtime(failed)) = error else {
        panic!("expected an assertion failure");
    };
    assert_eq!(failed.assertion_failures.len(), 1);
    assert!(failed.trace_messages.is_empty());
    assert!(failed.cover_counts.is_empty());
    let retry = compiled
        .step_with_options(
            &state,
            &[IrValue::bool(true), IrValue::bool(true)],
            ExecutionOptions::NO_EVENTS,
        )
        .unwrap();
    assert_eq!(retry.outputs, [IrValue::bool(false)]);
    assert!(retry.cover_counts.is_empty());
    let package = Parser::new(test_fixtures::CHILD_BOUNDS)
        .parse_package()
        .unwrap();
    let compiled = CompiledBlock::compile(&package, "parent").unwrap();
    let state = compiled.initial_state(RegisterInit::AllZeros).unwrap();
    let inputs = [
        IrValue::parse_typed("[bits[8]:1, bits[8]:2]").unwrap(),
        IrValue::make_ubits(2, 3).unwrap(),
    ];
    let error = compiled
        .step_with_options(&state, &inputs, ExecutionOptions::NO_EVENTS)
        .unwrap_err();
    let CompiledBlockError::Evaluation(BlockEvalError::Runtime(failed)) = error else {
        panic!("expected a bounds failure");
    };
    assert_eq!(failed.assumption_failures.len(), 1);
}

#[test]
fn instances_restore_typed_snapshots_and_reject_foreign_states() {
    let package = Parser::new(test_fixtures::MIXED_RESET)
        .parse_package()
        .unwrap();
    let compiled = CompiledBlock::compile(&package, "pair").unwrap();
    let other = CompiledBlock::compile(&package, "pair").unwrap();
    let mut first = compiled
        .prepare_instance(RegisterInit::Random { seed: 4 })
        .unwrap();
    let mut second = compiled.prepare_instance(RegisterInit::AllZeros).unwrap();
    let initial = first.snapshot().unwrap();
    let sibling_initial = second.snapshot().unwrap();
    let clocks = [
        IrValue::bool(true),
        IrValue::bool(true),
        IrValue::bool(true),
    ];
    first.tick(&clocks).unwrap();
    let after = first.snapshot().unwrap();
    assert_eq!(
        second.snapshot().unwrap().values(),
        sibling_initial.values()
    );
    assert!(matches!(
        other.step(&initial, &clocks),
        Err(CompiledBlockError::Evaluation(BlockEvalError::ForeignState))
    ));
    let foreign = other.initial_state(RegisterInit::AllZeros).unwrap();
    assert!(matches!(
        first.restore(&foreign),
        Err(CompiledBlockError::Evaluation(BlockEvalError::ForeignState))
    ));
    assert_eq!(first.snapshot().unwrap().values(), after.values());
    second.restore(&initial).unwrap();
    second.tick(&clocks).unwrap();
    assert_eq!(second.snapshot().unwrap().values(), after.values());
    first.restore(&initial).unwrap();
    assert_eq!(first.snapshot().unwrap().values(), initial.values());
    assert!(
        compiled
            .initial_state(RegisterInit::Supplied(vec![]))
            .is_err()
    );
    assert!(
        compiled
            .prepare_instance(RegisterInit::Supplied(vec![IrValue::bool(true); 2]))
            .is_err()
    );
}

#[test]
fn preparation_preserves_reachable_unsupported_block_errors() {
    let asynchronous =
        test_fixtures::MIXED_RESET.replace("asynchronous=false", "asynchronous=true");
    let inout = "package pads\ntop block io(pad: bits[1]) {\n ext_inout_net p(bits[1])\n pad: () = ext_inout_port(name=pad, net=p, id=1)\n}";
    let clock = "package clocks\nblock child(clk: clock) {\n}\ntop block parent() {\n instantiation c(block=child, kind=block)\n}";
    for (source, top) in [
        (asynchronous.as_str(), "pair"),
        (test_fixtures::OPAQUE_EXTERNAL, "wrapper"),
        (test_fixtures::MIXED_RESET, "missing"),
        (inout, "io"),
        (clock, "parent"),
    ] {
        let package = Parser::new(source).parse_package().unwrap();
        let expected = BlockEvaluator::prepare(&package, top).err().unwrap();
        let actual = CompiledBlock::compile(&package, top).err().unwrap();
        check_error(&expected, &actual);
    }
    compare_clocks(
        test_fixtures::OPAQUE_EXTERNAL,
        "independent",
        RegisterInit::AllZeros,
        &[vec![]],
    );
}

#[test]
fn unused_callee_result_still_reports_diagnostics_and_can_fail_an_edge() {
    let ir = r#"package dead_call
fn check(x: bits[1] id=1) -> bits[1] {
  one: bits[1] = literal(value=1, id=2)
  t: token = after_all(id=3)
  tr: token = trace(t, one, format="x={}", data_operands=[x], verbosity=0, id=4)
  checked: token = assert(tr, x, message="bit must be set", label="set", id=5)
  ret copy: bits[1] = identity(x, id=6)
}

fn wrapper(x: bits[1] id=12) -> bits[1] {
  unused_check: bits[1] = invoke(x, to_apply=check, id=13)
  ret copy: bits[1] = identity(x, id=14)
}

top block delay(clk: clock, x: bits[1], out: bits[1]) {
  reg q(bits[1])
  x: bits[1] = input_port(name=x, id=7)
  unused: bits[1] = invoke(x, to_apply=wrapper, id=8)
  old: bits[1] = register_read(register=q, id=9)
  write: () = register_write(x, register=q, id=10)
  out: () = output_port(old, name=out, id=11)
}"#;
    let clocks = [true, false, true].map(|x| vec![IrValue::bool(x)]);
    // All three runners share the projection. Pin the unused call's effects
    // independently so dropping it from the cycle cannot pass by agreement.
    let package = Parser::new(ir).parse_package().unwrap();
    let evaluator = BlockEvaluator::prepare(&package, "delay").unwrap();
    let initial = evaluator.initial_state(RegisterInit::AllZeros).unwrap();
    let first = evaluator.step(&initial, &clocks[0]).unwrap();
    assert_eq!(first.outputs, [IrValue::bool(false)]);
    assert!(matches!(&first.trace_messages[..], [t] if t.message == "x=1"));
    let Err(BlockEvalError::Runtime(failure)) = evaluator.step(&first.next_state, &clocks[1])
    else {
        panic!("the unused call must fail the edge when x is clear");
    };
    assert!(matches!(&failure.trace_messages[..], [t] if t.message == "x=0"));
    assert!(matches!(&failure.assertion_failures[..], [a] if a.label == "set"));
    let retry = evaluator.step(&first.next_state, &clocks[2]).unwrap();
    assert_eq!(retry.outputs, [IrValue::bool(true)]);
    compare_clocks(ir, "delay", RegisterInit::AllZeros, &clocks);
}

#[test]
fn compiler_reports_an_unsupported_zero_length_array() {
    let ir = r#"package empty_array
top block passthrough(x: bits[8][0], y: bits[8][0]) {
  x: bits[8][0] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}"#;
    let package = Parser::new(ir).parse_package().unwrap();
    let evaluator = BlockEvaluator::prepare(&package, "passthrough").unwrap();
    let state = evaluator.initial_state(RegisterInit::AllZeros).unwrap();
    let empty = IrValue::make_array_typed(xlsynth_pir::ir::Type::Bits(8), &[]).unwrap();
    let cycle = evaluator
        .step(&state, std::slice::from_ref(&empty))
        .unwrap();
    assert_eq!(cycle.outputs, [empty]);
    assert!(matches!(
        CompiledBlock::compile(&package, "passthrough"),
        Err(CompiledBlockError::Compiler(
            xlsynth_pir_compiler::CompilerError::UnsupportedType(_)
        ))
    ));
}

#[test]
fn cycle_function_keeps_a_same_named_package_callee_reachable() {
    let ir = r#"package coincident_names
fn flip(x: bits[1] id=1) -> bits[1] {
  ret negated: bits[1] = not(x, id=2)
}
fn flip__cycle_0(x: bits[1] id=3) -> bits[1] {
  ret itself: bits[1] = identity(x, id=4)
}
top block flip(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=5)
  inverted: bits[1] = invoke(x, to_apply=flip, id=6)
  same: bits[1] = invoke(inverted, to_apply=flip__cycle_0, id=7)
  y: () = output_port(same, name=y, id=8)
}"#;
    let clocks = [true, false].map(|x| vec![IrValue::bool(x)]);
    compare_clocks(ir, "flip", RegisterInit::AllZeros, &clocks);
}
