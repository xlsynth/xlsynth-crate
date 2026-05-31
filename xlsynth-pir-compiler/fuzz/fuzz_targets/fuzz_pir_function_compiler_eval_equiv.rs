// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy,
    generate_arguments, generate_fn,
};
use xlsynth_pir_compiler::PirFunctionCompiler;

fn options() -> RandomFnOptions {
    RandomFnOptions {
        max_params: 5,
        max_nodes: 64,
        max_bit_width: 64,
        allow_arrays: false,
        allow_tuples: false,
        allow_gate: true,
        allow_extension_ops: true,
        allow_arbitrary_width_multiply: true,
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
    .expect("the scalar fuzz generator options should always construct a PIR function");
    let function = &generated.function;
    let ir_text = function.to_string();
    let compiler = PirFunctionCompiler::compile(function)
        .unwrap_or_else(|error| panic!("compiled-function compilation failed for generated IR:\n{ir_text}\n{error}"));
    let mut argument_entropy = DepletableBytes::new(data);
    let args = generate_arguments(&mut argument_entropy, function);
    let expected = match eval_fn(function, &args) {
        FnEvalResult::Success(success) => success.value,
        other => panic!("PIR evaluator failed for generated IR:\n{ir_text}\n{other:?}"),
    };
    let actual = compiler
        .run_ir_values(&args)
        .unwrap_or_else(|error| panic!("compiled-function execution failed:\n{ir_text}\n{error}"));
    assert_eq!(
        actual, expected,
        "PIR compiler/evaluator mismatch\nIR:\n{ir_text}\nargs={args:?}"
    );
});
