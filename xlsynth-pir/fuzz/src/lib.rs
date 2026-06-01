// SPDX-License-Identifier: Apache-2.0

//! Helper library for fuzzing xlsynth-pir. Exposes equivalence utilities used
//! by fuzz targets and unit tests.

use xlsynth_pir::ir::{Fn as IrFn, Package};
use xlsynth_pir::ir_random::{
    generate_fn, generate_package, generate_same_signature_pair, DepletableBytes, OperationSet,
    RandomFnOptions, RandomOperation, StopPolicy,
};
use xlsynth_prover::prover::SolverLimits;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::solver::bitwuzla::BitwuzlaOptions;

pub mod equiv;

pub const FUZZ_SOLVER_TIME_LIMIT_PER_MS: u64 = 10_000;
pub const FUZZ_SOLVER_MEMORY_LIMIT_MB: u64 = 512;

/// Returns solver limits that keep individual fuzz samples responsive.
pub fn fuzz_solver_limits() -> SolverLimits {
    let mut limits = SolverLimits::with_time_limit_per_ms(FUZZ_SOLVER_TIME_LIMIT_PER_MS);
    limits.memory_limit_mb = Some(FUZZ_SOLVER_MEMORY_LIMIT_MB);
    limits
}

/// Returns Bitwuzla options with the fuzzing per-query limits applied.
#[cfg(feature = "has-bitwuzla")]
pub fn fuzz_bitwuzla_options() -> BitwuzlaOptions {
    let mut options = BitwuzlaOptions::new();
    options.set_time_limit_per(FUZZ_SOLVER_TIME_LIMIT_PER_MS);
    options.set_memory_limit(FUZZ_SOLVER_MEMORY_LIMIT_MB);
    options
}

fn operations_without_product_pairs_or_extensions() -> OperationSet {
    OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
        !matches!(
            operation,
            RandomOperation::Umulp
                | RandomOperation::Smulp
                | RandomOperation::ExtCarryOut
                | RandomOperation::ExtPrioEncode
                | RandomOperation::ExtClz
                | RandomOperation::ExtNormalizeLeft
                | RandomOperation::ExtMaskLow
                | RandomOperation::ExtNaryAdd
        )
    }))
}

fn upstream_formal_random_pir_options(max_nodes: usize) -> RandomFnOptions {
    RandomFnOptions {
        max_nodes,
        max_bit_width: 8,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: operations_without_product_pairs_or_extensions(),
        ..RandomFnOptions::default()
    }
}

/// Generates a standard random PIR package directly from coverage-guided bytes.
pub fn generate_standard_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let mut entropy = DepletableBytes::new(data);
    generate_fn(
        &mut entropy,
        &RandomFnOptions::default(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed standard PIR fuzz options should construct a valid function")
    .into_top_package(package_name)
}

/// Generates standard upstream-compatible PIR for libxls and formal checks.
pub fn generate_upstream_formal_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let options = upstream_formal_random_pir_options(32);
    let mut entropy = DepletableBytes::new(data);
    generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed standard PIR fuzz options should construct a valid function")
        .into_top_package(package_name)
}

/// Generates upstream-compatible PIR for differential interpreter checks.
pub fn generate_upstream_eval_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let options = upstream_formal_random_pir_options(32);
    let mut entropy = DepletableBytes::new(data);
    let mut generated = generate_package(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed standard PIR fuzz options should construct a valid package");
    generated.package.name = package_name.to_string();
    generated.package
}

/// Generates two upstream-compatible functions with matching signatures for
/// toolchain-backed equivalence properties.
pub fn generate_upstream_formal_random_pir_pair(data: &[u8]) -> (IrFn, IrFn) {
    let options = upstream_formal_random_pir_options(64);
    let (mut first_entropy, mut second_entropy) = DepletableBytes::split(data);
    let (first, second) = generate_same_signature_pair(
        &mut first_entropy,
        &mut second_entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed paired PIR fuzz options should construct matching functions");
    (first.function, second.function)
}

/// Generates two directly constructed PIR functions with one shared signature
/// across the full random-function PIR surface.
pub fn generate_full_random_pir_pair(data: &[u8]) -> (IrFn, IrFn) {
    let options = RandomFnOptions {
        max_nodes: 64,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        allow_extension_ops: true,
        allow_events: true,
        allow_assumed_in_bounds: true,
        ..RandomFnOptions::default()
    };
    let (mut first_entropy, mut second_entropy) = DepletableBytes::split(data);
    let (first, second) = generate_same_signature_pair(
        &mut first_entropy,
        &mut second_entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed paired PIR fuzz options should construct matching functions");
    (first.function, second.function)
}
