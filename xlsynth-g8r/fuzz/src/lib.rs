// SPDX-License-Identifier: Apache-2.0

pub mod external_yosys;
pub mod random_block;

use std::time::Duration;

use arbitrary::Arbitrary;
use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::{AigBitVector, GateBuilder, GateBuilderOptions, GateFn};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_g8r::prove_gate_fn_equiv_sat::GateFormalOptions;
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
    /// Mutates output wiring independently of the generated operations.
    pub output_seed: u64,
    pub ops: Vec<FuzzOp>,
    pub use_opt: bool,
}

/// Builds a graph whose first output observes a generated operation when
/// present.
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
    let operations_start = nodes.len();
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
    let mut output_rng = StdRng::seed_from_u64(sample.output_seed);
    for i in 0..outputs {
        // Keep at least one generated operation observable instead of exposing
        // only inputs. Optimizations may still simplify that operation away.
        let start = if i == 0 && operations_start < nodes.len() {
            operations_start
        } else {
            0
        };
        let node = nodes[output_rng.gen_range(start..nodes.len())];
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(node));
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use xlsynth::IrBits;
    use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};

    use super::{FuzzGraph, FuzzOp, GateFn, build_graph};

    /// Evaluates each output for every assignment of two one-bit inputs.
    fn truth_tables(graph: &GateFn) -> Vec<Vec<bool>> {
        let mut tables = vec![Vec::new(); graph.outputs.len()];
        for a in [false, true] {
            for b in [false, true] {
                let inputs = [IrBits::from_lsb_is_0(&[a]), IrBits::from_lsb_is_0(&[b])];
                let result = gate_sim::eval(graph, &inputs, Collect::None);
                for (table, output) in tables.iter_mut().zip(result.outputs) {
                    table.push(output.get_bit(0).unwrap());
                }
            }
        }
        tables
    }

    #[test]
    fn output_wiring_exercises_operations_and_varies_independently() {
        let and = vec![false, false, false, true];
        let and_not = vec![false, false, true, false];
        let a = vec![false, false, true, true];
        let b = vec![false, true, false, true];
        for use_opt in [false, true] {
            let mut first_tables = BTreeSet::new();
            let mut second_tables = BTreeSet::new();
            for output_seed in 0..64 {
                let sample = FuzzGraph {
                    num_inputs: 2,
                    input_width: 1,
                    num_ops: 2,
                    num_outputs: 2,
                    output_seed,
                    ops: vec![
                        FuzzOp {
                            lhs: 0,
                            rhs: 1,
                            lhs_neg: false,
                            rhs_neg: false,
                        },
                        FuzzOp {
                            lhs: 0,
                            rhs: 1,
                            lhs_neg: false,
                            rhs_neg: true,
                        },
                    ],
                    use_opt,
                };
                let tables = truth_tables(&build_graph(&sample).unwrap());
                assert_eq!(tables, truth_tables(&build_graph(&sample).unwrap()));
                first_tables.insert(tables[0].clone());
                second_tables.insert(tables[1].clone());
            }
            assert_eq!(first_tables, BTreeSet::from([and.clone(), and_not.clone()]));
            assert_eq!(
                second_tables,
                BTreeSet::from([and.clone(), and_not.clone(), a.clone(), b.clone()])
            );
        }
    }

    #[test]
    fn output_wiring_handles_graphs_without_operations_or_inputs() {
        for num_inputs in [0, 2] {
            for use_opt in [false, true] {
                let sample = FuzzGraph {
                    num_inputs,
                    input_width: 1,
                    num_ops: 0,
                    num_outputs: 0,
                    output_seed: 42,
                    ops: Vec::new(),
                    use_opt,
                };
                let graph = build_graph(&sample).unwrap();
                let inputs = vec![IrBits::from_lsb_is_0(&[true]); num_inputs as usize];
                let result = gate_sim::eval(&graph, &inputs, Collect::None);
                assert_eq!(
                    result.outputs,
                    vec![IrBits::from_lsb_is_0(&[num_inputs != 0])]
                );
            }
        }
    }
}
