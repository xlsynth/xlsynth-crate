// SPDX-License-Identifier: Apache-2.0

pub mod external_yosys;
pub mod random_block;

use std::time::Duration;

use arbitrary::Arbitrary;
use xlsynth_g8r::aig::{AigBitVector, GateBuilder, GateBuilderOptions, GateFn};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_g8r::prove_gate_fn_equiv_sat::GateFormalOptions;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_random::{
    generate_fn, DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy,
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
}

/// Returns Bitwuzla options with the fuzzing per-query time limit applied.
#[cfg(feature = "has-bitwuzla")]
pub fn fuzz_bitwuzla_options() -> BitwuzlaOptions {
    let mut options = BitwuzlaOptions::new();
    options.set_time_limit_per(FUZZ_SOLVER_TIME_LIMIT_PER_MS);
    options.set_memory_limit(FUZZ_SOLVER_MEMORY_LIMIT_MB);
    options
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzOp {
    pub lhs: u16,
    pub rhs: u16,
    pub lhs_neg: bool,
    pub rhs_neg: bool,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzGraph {
    pub num_inputs: u8,
    pub input_width: u8,
    pub num_ops: u8,
    pub num_outputs: u8,
    pub ops: Vec<FuzzOp>,
    pub use_opt: bool,
}

pub fn build_graph(sample: &FuzzGraph) -> Option<GateFn> {
    let num_inputs = sample.num_inputs.min(4);
    let width = sample.input_width.min(4);
    let num_ops = sample.num_ops.min(32);
    let opts = if sample.use_opt {
        GateBuilderOptions::opt()
    } else {
        GateBuilderOptions::no_opt()
    };
    let mut builder = GateBuilder::new("fuzz_rt".to_string(), opts);
    let mut nodes = Vec::new();
    for i in 0..num_inputs {
        let bv = builder.add_input(format!("in{}", i), width as usize);
        for j in 0..width {
            nodes.push(*bv.get_lsb(j as usize));
        }
    }
    if nodes.is_empty() {
        nodes.push(builder.get_false());
    }
    for op in sample.ops.iter().take(num_ops as usize) {
        let a = nodes[(op.lhs as usize) % nodes.len()];
        let b = nodes[(op.rhs as usize) % nodes.len()];
        let a = if op.lhs_neg { builder.add_not(a) } else { a };
        let b = if op.rhs_neg { builder.add_not(b) } else { b };
        let new_node = builder.add_and_binary(a, b);
        nodes.push(new_node);
        if nodes.len() > 256 {
            break;
        }
    }
    let outputs = nodes.len().min(sample.num_outputs as usize).max(1);
    for i in 0..outputs {
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(nodes[i]));
    }
    Some(builder.build())
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
