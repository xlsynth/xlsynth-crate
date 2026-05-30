// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy,
    generate_arguments, generate_fn,
};
use xlsynth_pir_compiler::PirFunctionJit;

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
        allow_arbitrary_width_multiply: true,
        allow_empty_case_sel: true,
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
            RandomOperation::ArrayConcat,
            RandomOperation::Tuple,
            RandomOperation::TupleIndex,
            RandomOperation::Sel,
            RandomOperation::PrioritySel,
            RandomOperation::OneHotSel,
            RandomOperation::ExtCarryOut,
            RandomOperation::ExtNaryAdd,
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
    .expect("wide fuzz generator options should always construct a PIR function");
    let function = &generated.function;
    let ir_text = function.to_string();
    let compiler = PirFunctionJit::compile(function).unwrap_or_else(|error| {
        panic!("native compilation failed for generated wide IR:\n{ir_text}\n{error}")
    });
    let mut argument_entropy = DepletableBytes::new(data);
    let args = generate_arguments(&mut argument_entropy, function);
    let expected = match eval_fn(function, &args) {
        FnEvalResult::Success(success) => success.value,
        other => panic!("PIR evaluator failed for generated wide IR:\n{ir_text}\n{other:?}"),
    };
    let actual = compiler
        .run_ir_values(&args)
        .unwrap_or_else(|error| panic!("native execution failed:\n{ir_text}\n{error}"));
    assert_eq!(
        actual, expected,
        "PIR compiler/evaluator wide mismatch\nIR:\n{ir_text}\nargs={args:?}"
    );
});
