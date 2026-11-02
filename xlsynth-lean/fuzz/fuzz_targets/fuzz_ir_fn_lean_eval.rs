// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use xlsynth_lean::{
    LeanEvaluator, LeanFunction, LeanFunctionInputs, LeanModuleName, export_ir_function,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, DepletableBytes, EntropySource, OperationSet, RandomFnOptions,
    RandomOperation, RngEntropy, StopPolicy, generate_package,
};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

const FUNCTIONS_PER_BATCH: usize = 8;
const INPUTS_PER_FUNCTION: usize = 32;
const MAX_GUIDED_BYTES: usize = 3072;

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

/// One generated top function with its original PIR package and typed stimulus.
struct Candidate {
    package: Package,
    function: LeanFunction,
    inputs: Vec<Vec<IrValue>>,
    /// Replays the vectors and, for a companion, its graph and body budget.
    seed: u64,
}

/// Exports one generated package and samples its typed input vectors.
fn generate_candidate(
    entropy: &mut impl EntropySource,
    opts: &RandomFnOptions,
    stop: StopPolicy,
    seed: u64,
    module: &LeanModuleName,
) -> Candidate {
    let package = generate_package(entropy, opts, stop)
        .unwrap_or_else(|error| panic!("PIR generation failed: {error}\nseed={seed}"))
        .package;
    let top = package.get_top_fn().expect("generated top");
    let function = export_ir_function(&package, None, module)
        .unwrap_or_else(|e| panic!("export failed: {e}\nseed={seed}\n{package}"));
    let inputs = generate_argument_sets_from_seed(top, seed, INPUTS_PER_FUNCTION);
    Candidate {
        package,
        function,
        inputs,
        seed,
    }
}

/// Checks a self-contained batch so libFuzzer can replay and minimize its
/// crashes.
fn check_batch(data: &[u8]) {
    let opts = options();
    let module = LeanModuleName::new("Generated.Function").unwrap();
    let mut candidates = Vec::with_capacity(FUNCTIONS_PER_BATCH);

    // Preserve the first record's encoding so libFuzzer can mutate and shrink
    // its graph.
    let guided = &data[..data.len().min(MAX_GUIDED_BYTES)];
    let (seed_bytes, graph) = guided.split_at(guided.len().min(8));
    let mut seed = [0; 8];
    seed[..seed_bytes.len()].copy_from_slice(seed_bytes);
    candidates.push(generate_candidate(
        &mut DepletableBytes::new(graph),
        &opts,
        StopPolicy::WhenEntropyDepleted,
        u64::from_le_bytes(seed),
        &module,
    ));

    // Hash the whole input so mutations past the guided input seed also vary
    // the companions.
    let mut seeds = StdRng::from_seed(*blake3::hash(data).as_bytes());
    for _ in 1..FUNCTIONS_PER_BATCH {
        let seed = seeds.next_u64();
        let mut rng = StdRng::seed_from_u64(seed);
        let body_nodes = rng.gen_range(1..=opts.max_nodes);
        candidates.push(generate_candidate(
            &mut RngEntropy::new(rng),
            &opts,
            StopPolicy::ExactBodyNodes(body_nodes),
            seed,
            &module,
        ));
    }
    let evaluator = EVALUATOR.get().expect("initialized Lean");
    let batches = candidates
        .iter()
        .map(|c| LeanFunctionInputs {
            function: &c.function,
            input_cases: &c.inputs,
        })
        .collect::<Vec<_>>();
    let outputs = evaluator
        .eval_ir_functions(&batches)
        .unwrap_or_else(|error| {
            // Preserve the batch error even if diagnosing a bad function hits a
            // timeout.
            eprintln!("Lean batch failed: {error}");
            for (slot, c) in candidates.iter().enumerate() {
                if let Err(e) = evaluator.eval_ir_function_batch(&c.function, &c.inputs) {
                    panic!(
                        "Lean failed in slot {slot}: {e}\nseed={}\n{}\n{}",
                        c.seed,
                        c.package,
                        c.function.lean_source()
                    );
                }
            }
            panic!("Lean failed only when the functions were evaluated together: {error}");
        });
    for (slot, (c, results)) in candidates.iter().zip(outputs).enumerate() {
        let top = c.package.get_top_fn().expect("generated top");
        for (args, actual) in c.inputs.iter().zip(results) {
            let expected = match eval_fn_in_package(&c.package, top, args) {
                FnEvalResult::Success(result) => result.value,
                FnEvalResult::Failure(failure) => {
                    panic!(
                        "PIR failed in slot {slot}: {failure:?}\nargs={args:?}\nseed={}\n{}",
                        c.seed, c.package
                    )
                }
            };
            assert_eq!(
                expected,
                actual,
                "slot={slot} seed={}\nargs={args:?}\n{}\n{}",
                c.seed,
                c.package,
                c.function.lean_source()
            );
        }
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
    check_batch(data);
});
