// SPDX-License-Identifier: Apache-2.0

#![no_main]

mod common;

use common::random_argument_sets;
use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, StopPolicy, generate_fn,
};
use xlsynth_pir_compiler::PirFunctionCompiler;

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn options() -> RandomFnOptions {
    RandomFnOptions {
        max_params: 5,
        max_nodes: 64,
        max_bit_width: 1024,
        max_type_depth: 3,
        max_aggregate_leaves: 24,
        max_array_length: 4,
        max_tuple_length: 4,
        allow_arrays: true,
        allow_tuples: true,
        allow_gate: true,
        allow_extension_ops: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_empty_case_sel: true,
        allow_events: true,
        allow_assumed_in_bounds: true,
        enabled_operations: OperationSet::all_supported(),
        ..RandomFnOptions::default()
    }
}

fuzz_target!(|data: &[u8]| {
    let mut graph_entropy = DepletableBytes::new(data);
    let generated = generate_fn(
        &mut graph_entropy,
        &options(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("wide fuzz generator options should always construct a PIR function");
    let function = &generated.function;
    let ir_text = function.to_string();
    let compiler = PirFunctionCompiler::compile(function).unwrap_or_else(|error| {
        panic!("native compilation failed for generated wide IR:\n{ir_text}\n{error}")
    });
    for args in random_argument_sets(data, function) {
        let expected = eval_fn(function, &args);
        let actual = compiler
            .run_ir_values_with_events(&args)
            .unwrap_or_else(|error| panic!("native execution failed:\n{ir_text}\n{error}"));
        match expected {
            FnEvalResult::Success(expected) => {
                assert_eq!(
                    actual.value, expected.value,
                    "PIR compiler/evaluator wide value mismatch\nIR:\n{ir_text}\nargs={args:?}"
                );
                assert!(
                    actual.events.assertion_failures.is_empty(),
                    "compiler unexpectedly reported assertions\nIR:\n{ir_text}\nargs={args:?}"
                );
                assert!(
                    actual.events.assumption_failures.is_empty(),
                    "compiler unexpectedly reported assumption failures\nIR:\n{ir_text}\nargs={args:?}"
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .trace_messages
                            .iter()
                            .map(|trace| (trace.message.clone(), trace.verbosity))
                            .collect()
                    ),
                    sorted(
                        expected
                            .trace_messages
                            .iter()
                            .map(|trace| (trace.message.clone(), trace.verbosity))
                            .collect()
                    ),
                    "PIR compiler/evaluator wide trace mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                    ),
                    "PIR compiler/evaluator wide cover mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                    ),
                    "PIR compiler/evaluator wide assertion mismatch\nIR:\n{ir_text}\nargs={args:?}"
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .assumption_failures
                            .iter()
                            .map(|failure| (failure.node_text_id, format!("{:?}", failure.kind)))
                            .collect()
                    ),
                    sorted(
                        expected
                            .assumption_failures
                            .iter()
                            .map(|failure| (failure.node_text_id, format!("{:?}", failure.kind)))
                            .collect()
                    ),
                    "PIR compiler/evaluator wide assumption mismatch\nIR:\n{ir_text}\nargs={args:?}"
                );
                assert_eq!(
                    sorted(
                        actual
                            .events
                            .trace_messages
                            .iter()
                            .map(|trace| (trace.message.clone(), trace.verbosity))
                            .collect()
                    ),
                    sorted(
                        expected
                            .trace_messages
                            .iter()
                            .map(|trace| (trace.message.clone(), trace.verbosity))
                            .collect()
                    ),
                    "PIR compiler/evaluator wide failing-trace mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                    ),
                    "PIR compiler/evaluator wide failing-cover mismatch\nIR:\n{ir_text}\nargs={args:?}"
                );
            }
        }
    }
});
