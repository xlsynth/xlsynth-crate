// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy,
    generate_arguments, generate_fn,
};
use xlsynth_pir_compiler::PirFunctionJit;

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn options() -> RandomFnOptions {
    RandomFnOptions {
        max_params: 5,
        max_nodes: 64,
        max_bit_width: 64,
        allow_arrays: true,
        allow_tuples: true,
        allow_gate: true,
        allow_extension_ops: true,
        allow_arbitrary_width_multiply: true,
        allow_empty_case_sel: true,
        allow_events: true,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::Identity,
            RandomOperation::Not,
            RandomOperation::Neg,
            RandomOperation::Reverse,
            RandomOperation::OrReduce,
            RandomOperation::AndReduce,
            RandomOperation::XorReduce,
            RandomOperation::And,
            RandomOperation::Nand,
            RandomOperation::Nor,
            RandomOperation::Or,
            RandomOperation::Xor,
            RandomOperation::Add,
            RandomOperation::Sub,
            RandomOperation::Umul,
            RandomOperation::Smul,
            RandomOperation::Udiv,
            RandomOperation::Sdiv,
            RandomOperation::Umod,
            RandomOperation::Smod,
            RandomOperation::Umulp,
            RandomOperation::Smulp,
            RandomOperation::Eq,
            RandomOperation::Ne,
            RandomOperation::Ugt,
            RandomOperation::Uge,
            RandomOperation::Ult,
            RandomOperation::Ule,
            RandomOperation::Sgt,
            RandomOperation::Sge,
            RandomOperation::Slt,
            RandomOperation::Sle,
            RandomOperation::Shll,
            RandomOperation::Shrl,
            RandomOperation::Shra,
            RandomOperation::Gate,
            RandomOperation::ZeroExt,
            RandomOperation::SignExt,
            RandomOperation::BitSlice,
            RandomOperation::DynamicBitSlice,
            RandomOperation::BitSliceUpdate,
            RandomOperation::Concat,
            RandomOperation::Array,
            RandomOperation::ArrayIndex,
            RandomOperation::ArrayConcat,
            RandomOperation::ArraySlice,
            RandomOperation::ArrayUpdate,
            RandomOperation::Tuple,
            RandomOperation::TupleIndex,
            RandomOperation::Sel,
            RandomOperation::PrioritySel,
            RandomOperation::OneHotSel,
            RandomOperation::OneHot,
            RandomOperation::Encode,
            RandomOperation::Decode,
            RandomOperation::ExtCarryOut,
            RandomOperation::ExtPrioEncode,
            RandomOperation::ExtClz,
            RandomOperation::ExtNormalizeLeft,
            RandomOperation::ExtMaskLow,
            RandomOperation::ExtNaryAdd,
            RandomOperation::AfterAll,
            RandomOperation::Cover,
            RandomOperation::Assert,
            RandomOperation::Trace,
        ]),
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
    .expect("aggregate fuzz generator options should always construct a PIR function");
    let function = &generated.function;
    let ir_text = function.to_string();
    let jit = PirFunctionJit::compile(function)
        .unwrap_or_else(|error| panic!("JIT compilation failed for generated IR:\n{ir_text}\n{error}"));
    let mut argument_entropy = DepletableBytes::new(data);
    let args = generate_arguments(&mut argument_entropy, function);
    let expected = eval_fn(function, &args);
    let actual = jit
        .run_ir_values_with_events(&args)
        .unwrap_or_else(|error| panic!("JIT execution failed:\n{ir_text}\n{error}"));
    match expected {
        FnEvalResult::Success(expected) => {
            assert_eq!(
                actual.value, expected.value,
                "PIR compiler/evaluator value mismatch\nIR:\n{ir_text}\nargs={args:?}"
            );
            assert!(
                actual.events.assertion_failures.is_empty(),
                "compiler unexpectedly reported assertions\nIR:\n{ir_text}\nargs={args:?}"
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
                "PIR compiler/evaluator trace mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                "PIR compiler/evaluator cover mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                "PIR compiler/evaluator assertion mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                "PIR compiler/evaluator failing-trace mismatch\nIR:\n{ir_text}\nargs={args:?}"
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
                "PIR compiler/evaluator failing-cover mismatch\nIR:\n{ir_text}\nargs={args:?}"
            );
        }
    }
});
