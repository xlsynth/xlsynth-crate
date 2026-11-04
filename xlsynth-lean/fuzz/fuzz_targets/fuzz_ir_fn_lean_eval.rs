// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_lean::{LeanEvaluator, LeanModuleName, export_ir_function};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, DepletableBytes, OperationSet, RandomFnOptions, RandomOperation,
    RngEntropy, StopPolicy, generate_package,
};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

const INPUTS_PER_FUNCTION: usize = 32;
const MAX_FUZZ_BYTES: usize = 3072;

static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();

/// Keep generation entirely within the pure value contract of the word-level
/// exporter.
fn options() -> RandomFnOptions {
    let operations = OperationSet::all_supported();
    RandomFnOptions {
        max_nodes: 32,
        max_functions: 3,
        max_bit_width: 129,
        max_aggregate_leaves: 12,
        allow_gate: true,
        allow_map: true,
        allow_empty_case_sel: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_extension_ops: true,
        allow_events: false,
        max_counted_fors_per_function: 0,
        array_assumption_mode: ArrayAssumptionMode::ProvenSafe,
        enabled_operations: OperationSet::new(operations.iter().filter(|op| {
            !matches!(
                op,
                RandomOperation::Smulp
                    | RandomOperation::Umulp
                    | RandomOperation::CountedFor
                    | RandomOperation::AfterAll
                    | RandomOperation::Assert
                    | RandomOperation::Trace
                    | RandomOperation::Cover
            )
        })),
        ..RandomFnOptions::default()
    }
}

/// Checks one exported graph and 32 input vectors per reproducible fuzz case.
fn check_function(data: &[u8]) {
    let data = &data[..data.len().min(MAX_FUZZ_BYTES)];
    let (seed_bytes, graph) = data.split_at(data.len().min(8));
    let mut seed = [0; 8];
    seed[..seed_bytes.len()].copy_from_slice(seed_bytes);
    let seed = u64::from_le_bytes(seed);
    let opts = options();
    let generated = if graph.is_empty() {
        let mut rng = StdRng::seed_from_u64(seed);
        let body_nodes = rng.gen_range(1..=opts.max_nodes);
        generate_package(
            &mut RngEntropy::new(rng),
            &opts,
            StopPolicy::ExactBodyNodes(body_nodes),
        )
    } else {
        generate_package(
            &mut DepletableBytes::new(graph),
            &opts,
            StopPolicy::WhenEntropyDepleted,
        )
    };
    let package = generated
        .unwrap_or_else(|error| panic!("PIR generation failed: {error}\nseed={seed}"))
        .package;
    let top = package.get_top_fn().expect("generated top");
    let namespaces = ["Generated.Function", "BitVec", "Vector", "Nat.Generated"];
    let namespace = namespaces[usize::from(blake3::hash(data).as_bytes()[0]) % namespaces.len()];
    let module = LeanModuleName::new(namespace).unwrap();
    let function = export_ir_function(&package, None, &module).unwrap_or_else(|error| {
        panic!("export failed: {error}\nmodule={namespace} seed={seed}\n{package}")
    });
    let inputs = generate_argument_sets_from_seed(top, seed, INPUTS_PER_FUNCTION);
    let evaluator = EVALUATOR.get().expect("initialized Lean");
    let outputs = evaluator
        .eval_ir_function_batch(&function, &inputs)
        .unwrap_or_else(|error| {
            panic!(
                "Lean failed: {error}\nmodule={namespace} seed={seed}\n{package}\n{}",
                function.lean_source()
            )
        });
    assert_eq!(outputs.len(), inputs.len());
    for (args, actual) in inputs.iter().zip(outputs) {
        let expected = match eval_fn_in_package(&package, top, args) {
            FnEvalResult::Success(result) => result.value,
            FnEvalResult::Failure(failure) => {
                panic!("PIR failed: {failure:?}\nargs={args:?}\nseed={seed}\n{package}");
            }
        };
        assert_eq!(
            expected,
            actual,
            "module={namespace} seed={seed}\nargs={args:?}\n{package}\n{}",
            function.lean_source()
        );
    }
}

fuzz_target!(init: {
    let Some(executable) = std::env::var_os("XLSYNTH_LEAN") else {
        eprintln!("fuzz setup failed: set XLSYNTH_LEAN to the pinned Lean executable");
        std::process::exit(2);
    };
    let evaluator = LeanEvaluator::new(PathBuf::from(executable)).unwrap_or_else(|error| {
        eprintln!("fuzz setup failed: could not initialize Lean evaluator: {error}");
        std::process::exit(2);
    });
    assert!(EVALUATOR.set(evaluator).is_ok(), "Lean evaluator initialized twice");
}, |data: &[u8]| {
    check_function(data);
});
