// SPDX-License-Identifier: Apache-2.0

pub mod abc_mapping;
pub mod direct_g8r;
pub mod external_yosys;
pub mod netlist_resize;
pub mod random_block;
pub mod sequential_cleanup;
pub mod sequential_legalization;
pub mod serdes_roundtrip;

#[cfg(feature = "has-bitwuzla")]
pub mod sentinel_lowering;

use std::time::Duration;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_g8r::prove_gate_fn_equiv_sat::{DEFAULT_CADICAL_TERMINATE_LIMIT, GateFormalOptions};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy, generate_fn,
};
use xlsynth_prover::prover::SolverLimits;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::solver::bitwuzla::BitwuzlaOptions;

pub const FUZZ_SOLVER_TIME_LIMIT_PER_MS: u64 = 10_000;
pub const FUZZ_SOLVER_MEMORY_LIMIT_MB: u64 = 512;
pub const G8R_FUZZ_MAX_NODES: usize = 64;

/// Returns solver limits that keep individual fuzz samples responsive.
pub fn fuzz_solver_limits() -> SolverLimits {
    let mut limits = SolverLimits::with_time_limit_per_ms(FUZZ_SOLVER_TIME_LIMIT_PER_MS);
    limits.memory_limit_mb = Some(FUZZ_SOLVER_MEMORY_LIMIT_MB);
    limits
}

/// Returns gate-formal options that keep individual fuzz samples responsive.
pub fn fuzz_gate_formal_options() -> GateFormalOptions {
    GateFormalOptions::default()
        .with_cadical_timeout(Duration::from_millis(FUZZ_SOLVER_TIME_LIMIT_PER_MS))
        .with_cadical_terminate_limit(DEFAULT_CADICAL_TERMINATE_LIMIT)
}

/// Returns Bitwuzla options with the fuzzing per-query time limit applied.
#[cfg(feature = "has-bitwuzla")]
pub fn fuzz_bitwuzla_options() -> BitwuzlaOptions {
    let mut options = BitwuzlaOptions::new();
    options.set_time_limit_per(FUZZ_SOLVER_TIME_LIMIT_PER_MS);
    options.set_memory_limit(FUZZ_SOLVER_MEMORY_LIMIT_MB);
    options
}

fn gatify_random_pir_options(max_nodes: usize) -> RandomFnOptions {
    let operations =
        OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
            !matches!(operation, RandomOperation::Umulp | RandomOperation::Smulp)
        }));
    RandomFnOptions {
        max_nodes,
        max_bit_width: 8,
        allow_arbitrary_width_multiply: true,
        allow_extension_ops: true,
        enabled_operations: operations,
        ..RandomFnOptions::default()
    }
}

/// Generates PIR supported by gatify, including PIR extension operations.
pub fn generate_gatify_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let options = gatify_random_pir_options(G8R_FUZZ_MAX_NODES);
    let mut entropy = DepletableBytes::new(data);
    generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed gatify PIR fuzz options should construct a valid function")
        .into_top_package(package_name)
}

/// A random source package and its canonical default g8r lowering.
pub struct FullG8rFuzzCase {
    pub source_package: Package,
    pub source_top: String,
    pub source_ir: String,
    pub gate_fn: GateFn,
}

/// Generates random PIR and runs it through the canonical default g8r pipeline.
pub fn generate_full_g8r_fuzz_case(
    data: &[u8],
    package_name: &str,
) -> Result<FullG8rFuzzCase, String> {
    let source_package = generate_gatify_random_pir_package(data, package_name);
    let source_top = source_package
        .get_top_fn()
        .expect("generated package should have a top function")
        .name
        .clone();
    let source_ir = source_package.to_string();
    let artifacts = canonical_ir_text_to_g8r_lowering_artifacts(
        &source_ir,
        Some(&source_top),
        &CanonicalG8rOptions::default(),
    )
    .map_err(|error| {
        format!("full g8r flow failed for generated IR:\n{source_ir}\nerror={error}")
    })?;
    Ok(FullG8rFuzzCase {
        source_package,
        source_top,
        source_ir,
        gate_fn: artifacts.gate_fn,
    })
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_CADICAL_TERMINATE_LIMIT, fuzz_gate_formal_options};

    #[test]
    fn gate_formal_fuzzing_has_deterministic_and_wall_clock_limits() {
        let options = fuzz_gate_formal_options();
        assert!(options.cadical_timeout.is_some());
        assert_eq!(
            options.cadical_terminate_limit,
            Some(DEFAULT_CADICAL_TERMINATE_LIMIT)
        );
    }
}

#[cfg(feature = "has-bitwuzla")]
pub mod projected_positive_mask;

#[cfg(feature = "has-bitwuzla")]
pub mod shift_nonzero;
