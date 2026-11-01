// SPDX-License-Identifier: Apache-2.0

//! Formal validation of the data bits registered between pipeline stages.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use xlsynth::{IrPackage, XlsIrValue, XlsynthError, optimize_ir};
use xlsynth_pir::ir;
use xlsynth_pir::ir_fn_rm_asserts::remove_asserts_from_fn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
    zero_ir_value_for_type,
};
use xlsynth_pir::{IrBits, IrValue};

use super::layout::verify_stage_port_widths;
use super::verify_stage_signatures;
use crate::aig::import::import_gate_fn;
use crate::aig::topo::extract_cone;
use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::aig_sim::gate_sim::PreparedGateSim;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::gatify::ir2gate::{GatifyOptions, gatify};
use crate::prove_gate_fn_equiv_sat::{
    CadicalSat, IncrementalSat, SatSolveResult, ValidationError, build_sat_clauses,
};

/// Minimum queries sharing each solver when enough bits exist to split work.
const MIN_QUERIES_PER_BATCH: usize = 16;
/// Bound solver replication even when the caller's Rayon pool is much larger.
const MAX_PARALLEL_SOLVERS: usize = 8;
/// Limit worst-case concurrent cone replication to this many AIG nodes.
///
/// This is a planning heuristic, not an RSS limit: SAT learned clauses and
/// per-node allocations vary. One solver is always allowed, even for a graph
/// exceeding this budget. Actual batches encode only their own fan-in cones.
const MAX_PARALLEL_CONE_NODES: usize = 2_000_000;

/// A registered stage-output bit that has the same value for every input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantFlop {
    pub stage_name: String,
    /// Zero-based index in the supplied stage list.
    pub stage_index: usize,
    /// LSB-based index in the stage's packed output port.
    pub output_bit_index: usize,
    pub value: bool,
}

/// The interpreted output of one stage for an all-zero pipeline input.
struct StageBaseline {
    stage_name: String,
    stage_index: usize,
    bits: Vec<bool>,
}

/// A composed pipeline prefix and independently interpreted stage outputs.
struct PipelineProof {
    graph: GateFn,
    baselines: Vec<StageBaseline>,
}

/// One independent existential query over the composed pipeline graph.
struct BitQuery {
    baseline_index: usize,
    output_bit_index: usize,
    value: bool,
    operand: AigOperand,
}

/// Finds constant registered data bits, ordered by stage then packed bit index.
///
/// Each checked stage is interpreted for an all-zero pipeline input. SAT
/// queries then check whether each bit can differ from that baseline for any
/// original pipeline input. Stages are composed, so correlations established by
/// earlier stages remain constraints on later stages. Intermediate outputs are
/// always checked; the final output is checked only when `flop_outputs` is set.
///
/// This checks DSLX data semantics on valid transactions; a custom Verilog
/// template must preserve those semantics. Input registers have freely
/// varying inputs, and validity/control registers and power-up values are not
/// part of this analysis. Solver interruption is an error, never a successful
/// validation. The input package is not modified.
/// Assertions are removed from the temporary proof model: they neither
/// constrain the input domain nor prevent evaluating the zero-input baseline.
///
/// Queries across stages and bits run in bounded batches on the current Rayon
/// pool, respecting its thread count (or `RAYON_NUM_THREADS` for the global
/// pool). Each batch reuses one solver for its combined fan-in cone. At most
/// eight solvers run concurrently, reduced further for large graphs and small
/// query counts. Results and errors are returned in stage/bit order regardless
/// of scheduling.
pub fn find_constant_flops(
    ir: &IrPackage,
    stages: &[(String, String)],
    flop_outputs: bool,
) -> Result<Vec<ConstantFlop>, XlsynthError> {
    if stages.is_empty() {
        return Err(XlsynthError("no pipeline stages found".into()));
    }
    verify_stage_signatures(ir, stages)?;
    verify_stage_port_widths(ir, stages)?;
    let checked_stage_count = stages.len() - usize::from(!flop_outputs);
    if checked_stage_count == 0 {
        return Ok(Vec::new());
    }

    let PipelineProof { graph, baselines } =
        build_prefix_graph(ir, &stages[..checked_stage_count])?;

    // Confirm that the independently interpreted baseline is realizable in
    // the graph used for proofs. A layout/lowering mismatch must not let an
    // incorrect candidate value hide a constant bit.
    let zeros: Vec<IrBits> = graph
        .inputs
        .iter()
        .map(|input| IrBits::zero(input.get_bit_count()))
        .collect();
    let simulated = PreparedGateSim::new(&graph).eval_outputs(&zeros);
    for (baseline, actual) in baselines.iter().zip(&simulated) {
        if IrBits::from_lsb_is_0(&baseline.bits) != *actual {
            return Err(XlsynthError(format!(
                "constant-flop validation: interpreted and lowered outputs disagree for stage '{}'",
                baseline.stage_name
            )));
        }
    }

    prove_graph_outputs(&graph, &baselines, CadicalSat::new)
}

/// Distributes ordered query batches across a bounded set of solver workers.
fn prove_graph_outputs<S, F>(
    graph: &GateFn,
    baselines: &[StageBaseline],
    make_solver: F,
) -> Result<Vec<ConstantFlop>, XlsynthError>
where
    S: IncrementalSat,
    F: Fn() -> Result<S, ValidationError> + Sync,
{
    let queries: Vec<BitQuery> = baselines
        .iter()
        .zip(&graph.outputs)
        .enumerate()
        .flat_map(|(baseline_index, (baseline, output))| {
            baseline
                .bits
                .iter()
                .copied()
                .zip(output.bit_vector.iter_lsb_to_msb().copied())
                .enumerate()
                .map(move |(output_bit_index, (value, operand))| BitQuery {
                    baseline_index,
                    output_bit_index,
                    value,
                    operand,
                })
        })
        .collect();
    if queries.is_empty() {
        return Ok(Vec::new());
    }

    // Avoid initializing the global pool when work or graph size already
    // limits this validation to one solver.
    let max_workers = solver_worker_count(graph.gates.len(), queries.len(), MAX_PARALLEL_SOLVERS);
    let workers = if max_workers > 1 {
        max_workers.min(rayon::current_num_threads())
    } else {
        1
    };
    if workers == 1 {
        return prove_query_batch(graph, baselines, &queries, &make_solver);
    }
    let batch_size = queries
        .len()
        .div_ceil(workers * 4)
        .max(MIN_QUERIES_PER_BATCH);
    let batch_count = queries.len().div_ceil(batch_size);
    let next_batch = AtomicUsize::new(0);
    let worker_results: Vec<_> = (0..workers)
        .into_par_iter()
        .map(|_| {
            let mut results = Vec::new();
            loop {
                // Dynamic assignment lets workers that finish easy earlier
                // stages take more batches from expensive later stages.
                let batch_index = next_batch.fetch_add(1, Ordering::Relaxed);
                if batch_index >= batch_count {
                    break;
                }
                let start = batch_index * batch_size;
                let end = (start + batch_size).min(queries.len());
                results.push((
                    batch_index,
                    prove_query_batch(graph, baselines, &queries[start..end], &make_solver),
                ));
            }
            results
        })
        .collect();
    let mut batch_results: Vec<_> = worker_results.into_iter().flatten().collect();
    batch_results.sort_unstable_by_key(|(batch_index, _)| *batch_index);

    // Resolve errors after ordering results. Parallel short-circuiting would
    // otherwise report whichever failed query happened to finish first.
    let mut constants = Vec::new();
    for (_, result) in batch_results {
        constants.extend(result?);
    }
    Ok(constants)
}

/// Bounds simultaneous solvers by CPU availability, query count, and graph
/// size.
fn solver_worker_count(graph_nodes: usize, queries: usize, pool_threads: usize) -> usize {
    pool_threads
        .min(MAX_PARALLEL_SOLVERS)
        .min(queries.div_ceil(MIN_QUERIES_PER_BATCH))
        .min((MAX_PARALLEL_CONE_NODES / graph_nodes.max(1)).max(1))
        .max(1)
}

/// Encodes one shared cone and checks a batch using temporary SAT assumptions.
fn prove_query_batch<S, F>(
    graph: &GateFn,
    baselines: &[StageBaseline],
    queries: &[BitQuery],
    make_solver: &F,
) -> Result<Vec<ConstantFlop>, XlsynthError>
where
    S: IncrementalSat,
    F: Fn() -> Result<S, ValidationError>,
{
    let mut solver = make_solver().map_err(|error| XlsynthError(error.to_string()))?;
    let output_nodes: Vec<_> = queries.iter().map(|query| query.operand.node).collect();
    let (reachable, inputs) = extract_cone(&output_nodes, &graph.gates);
    let literals = build_sat_clauses(&mut solver, &reachable, &inputs, &graph.gates);
    let mut constants = Vec::new();
    for query in queries {
        let baseline = &baselines[query.baseline_index];
        let literal = literals[&query.operand.node];
        let output_literal = if query.operand.negated {
            !literal
        } else {
            literal
        };
        let differs = if query.value {
            !output_literal
        } else {
            output_literal
        };
        let result = solver.sat_solve_assuming(&[differs]).map_err(|error| {
            XlsynthError(format!(
                "constant-flop validation failed to prove stage '{}' output bit {}: {error}",
                baseline.stage_name, query.output_bit_index
            ))
        })?;
        if result == SatSolveResult::Unsat {
            constants.push(ConstantFlop {
                stage_name: baseline.stage_name.clone(),
                stage_index: baseline.stage_index,
                output_bit_index: query.output_bit_index,
                value: query.value,
            });
        }
    }
    Ok(constants)
}

/// Interprets and lowers each stage once, preserving all prefix correlations.
fn build_prefix_graph(
    ir: &IrPackage,
    stages: &[(String, String)],
) -> Result<PipelineProof, XlsynthError> {
    let mut builder = GateBuilder::new(
        "constant_flop_validation".to_string(),
        GateBuilderOptions::opt(),
    );
    let mut baselines: Vec<StageBaseline> = Vec::with_capacity(stages.len());
    let mut previous_output: Option<AigBitVector> = None;
    for (stage_index, (stage_name, ir_name)) in stages.iter().enumerate() {
        // Optimization inlines helper calls and loops before bit blasting.
        let optimized = optimize_ir(ir, ir_name)?;
        let optimized_text = optimized.to_string();
        let mut package = Parser::new(&optimized_text)
            .parse_and_validate_package()
            .map_err(|error| XlsynthError(error.to_string()))?;
        let function = package.get_fn_mut(ir_name).ok_or_else(|| {
            XlsynthError(format!("optimized pipeline stage '{ir_name}' is missing"))
        })?;
        // Assertions do not drive data outputs or constrain this all-inputs
        // proof. Remove them after inlining, including assertions in helpers,
        // so the arbitrary zero baseline cannot fail a runtime assertion.
        // Keep this package separate from the original used for codegen.
        *function = remove_asserts_from_fn(function).rewritten_fn;
        let data_ir = IrPackage::parse_ir(&package.to_string(), None)?;
        let function = package.get_fn(ir_name).expect("stage was found above");
        let arguments = stage_arguments(function, baselines.last().map(|value| &value.bits[..]))?;
        let interpreted = data_ir.get_function(ir_name)?.interpret(&arguments)?;
        let interpreted = IrValue::parse_typed(&interpreted.to_string())
            .map_err(|error| XlsynthError(error.to_string()))?;
        let mut bits = Vec::with_capacity(function.ret_ty.bit_count());
        flatten_ir_value_to_lsb0_bits_for_type(&interpreted, &function.ret_ty, &mut bits)
            .map_err(XlsynthError)?;
        let lowering_options = GatifyOptions {
            // This graph is only a proof model: gate(pred, value) has the
            // ordinary IR semantics pred ? value : 0. Modeling it as ANDs
            // does not replace the user's physical gate cell or template.
            unsafe_gatify_gate_operation: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let lowered = gatify(function, lowering_options).map_err(|error| {
            XlsynthError(format!(
                "constant-flop validation could not lower stage '{stage_name}': {error}"
            ))
        })?;

        let mut bindings = Vec::with_capacity(lowered.gate_fn.inputs.len());
        let mut offset = 0;
        for input in &lowered.gate_fn.inputs {
            let width = input.get_bit_count();
            let binding = if let Some(previous) = &previous_output {
                // Match build_pipeline's packed-wire slicing: input ports
                // consume low bits first, in declaration order.
                previous.get_lsb_slice(offset, width)
            } else {
                builder.add_input(input.name.clone(), width)
            };
            offset += width;
            bindings.push(binding);
        }
        let mut outputs =
            import_gate_fn(&lowered.gate_fn, &bindings, &mut builder).map_err(XlsynthError)?;
        if outputs.len() != 1 {
            return Err(XlsynthError(format!(
                "constant-flop validation expected one packed output from stage '{stage_name}'"
            )));
        }
        let output = outputs.pop().expect("one output checked above");
        if output.get_bit_count() != bits.len() {
            return Err(XlsynthError(format!(
                "constant-flop validation found different interpreted and lowered output widths for stage '{stage_name}'"
            )));
        }
        builder.add_output(format!("stage_{stage_index}"), output.clone());
        previous_output = Some(output);
        baselines.push(StageBaseline {
            stage_name: stage_name.clone(),
            stage_index,
            bits,
        });
    }
    Ok(PipelineProof {
        graph: builder.build(),
        baselines,
    })
}

/// Reconstructs stage arguments from the preceding packed output wire.
fn stage_arguments(
    function: &ir::Fn,
    previous: Option<&[bool]>,
) -> Result<Vec<XlsIrValue>, XlsynthError> {
    let mut offset = 0;
    let mut arguments = Vec::with_capacity(function.params.len());
    for parameter in function.param_nodes() {
        let width = parameter.ty.bit_count();
        let value = if let Some(previous) = previous {
            let slice = previous.get(offset..offset + width).ok_or_else(|| {
                XlsynthError(format!(
                    "constant-flop validation: stage '{}' input exceeds preceding output width",
                    function.name
                ))
            })?;
            ir_value_from_lsb0_bits_with_layout(&parameter.ty, slice).map_err(XlsynthError)?
        } else {
            zero_ir_value_for_type(&parameter.ty)
        };
        arguments.push(XlsIrValue::parse_typed(&value.to_string())?);
        offset += width;
    }
    if let Some(previous) = previous {
        if offset != previous.len() {
            return Err(XlsynthError(format!(
                "constant-flop validation: stage '{}' input width does not match preceding output",
                function.name
            )));
        }
    }
    Ok(arguments)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rayon::ThreadPoolBuilder;
    use xlsynth::IrPackage;

    use super::{
        ConstantFlop, PipelineProof, StageBaseline, find_constant_flops, prove_graph_outputs,
        solver_worker_count,
    };
    use crate::aig::AigBitVector;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::prove_gate_fn_equiv_sat::{
        CadicalSat, IncrementalSat, SatSolveResult, ValidationError,
    };
    use crate::test_utils::interesting_ir_roundtrip_cases;

    /// Supplies enough mixed stage/bit queries to exercise multiple batches.
    fn parallel_proof_fixture() -> PipelineProof {
        let mut builder = GateBuilder::new("parallel_proof".into(), GateBuilderOptions::no_opt());
        let inputs = builder.add_input("x".into(), 32);
        let mut baselines = Vec::new();
        for stage_index in 0..4 {
            let mut outputs = Vec::new();
            for bit in 0..32 {
                let input = *inputs.get_lsb(bit);
                outputs.push(match bit % 4 {
                    0 => builder.add_and_binary(input, input.negate()),
                    1 => builder.add_or_binary(input, input.negate()),
                    2 => input,
                    _ => input.negate(),
                });
            }
            let stage_name = format!("stage_{stage_index}");
            builder.add_output(
                stage_name.clone(),
                AigBitVector::from_lsb_is_index_0(&outputs),
            );
            baselines.push(StageBaseline {
                stage_name,
                stage_index,
                bits: (0..32).map(|bit| bit % 2 == 1).collect(),
            });
        }
        PipelineProof {
            graph: builder.build(),
            baselines,
        }
    }

    /// Preserves real clause construction but makes proof queries inconclusive.
    struct InterruptedSat {
        inner: CadicalSat,
    }

    impl IncrementalSat for InterruptedSat {
        type Lit = <CadicalSat as IncrementalSat>::Lit;
        type Model = <CadicalSat as IncrementalSat>::Model;

        fn sat_new_lit(&mut self) -> Self::Lit {
            self.inner.sat_new_lit()
        }

        fn sat_add_clause(&mut self, clause: &[Self::Lit]) {
            self.inner.sat_add_clause(clause);
        }

        fn sat_solve_assuming(
            &mut self,
            _assumptions: &[Self::Lit],
        ) -> Result<SatSolveResult, ValidationError> {
            Err(ValidationError::CadicalSolveInterrupted)
        }

        fn sat_model(&self) -> Result<Self::Model, ValidationError> {
            self.inner.sat_model()
        }
    }

    #[test]
    fn parallel_batches_preserve_results_and_reuse_solvers() {
        let proof = parallel_proof_fixture();
        let serial_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let parallel_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let serial_solver_count = AtomicUsize::new(0);
        let parallel_solver_count = AtomicUsize::new(0);
        let serial = serial_pool
            .install(|| {
                prove_graph_outputs(&proof.graph, &proof.baselines, || {
                    serial_solver_count.fetch_add(1, Ordering::Relaxed);
                    CadicalSat::new()
                })
            })
            .unwrap();
        let parallel = parallel_pool
            .install(|| {
                prove_graph_outputs(&proof.graph, &proof.baselines, || {
                    parallel_solver_count.fetch_add(1, Ordering::Relaxed);
                    CadicalSat::new()
                })
            })
            .unwrap();
        let expected: Vec<_> = (0..4)
            .flat_map(|stage_index| {
                (0..32)
                    .filter(|bit| bit % 4 < 2)
                    .map(move |output_bit_index| ConstantFlop {
                        stage_name: format!("stage_{stage_index}"),
                        stage_index,
                        output_bit_index,
                        value: output_bit_index % 4 == 1,
                    })
            })
            .collect();
        assert_eq!(serial, expected);
        assert_eq!(parallel, expected);
        assert_eq!(serial_solver_count.load(Ordering::Relaxed), 1);
        assert_eq!(parallel_solver_count.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn parallel_interruption_reports_the_first_query_deterministically() {
        let proof = parallel_proof_fixture();
        for threads in [1, 4] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let error = pool
                .install(|| {
                    prove_graph_outputs(&proof.graph, &proof.baselines, || {
                        CadicalSat::new().map(|inner| InterruptedSat { inner })
                    })
                })
                .unwrap_err();
            assert_eq!(
                error.0,
                "constant-flop validation failed to prove stage 'stage_0' output bit 0: CadicalSolveInterrupted"
            );
        }
    }

    #[test]
    fn bounds_parallelism_by_pool_work_and_graph_replication() {
        assert_eq!(solver_worker_count(100, 1_000, 4), 4);
        assert_eq!(solver_worker_count(100, 1_000, 64), 8);
        assert_eq!(solver_worker_count(100, 16, 64), 1);
        assert_eq!(solver_worker_count(1_000_000, 1_000, 64), 2);
        assert_eq!(solver_worker_count(3_000_000, 1_000, 64), 1);
        assert_eq!(solver_worker_count(0, 1_000, 4), 4);
    }

    #[test]
    fn identifies_both_constant_values_without_inputs() {
        let ir = IrPackage::parse_ir(
            r#"package sample
top fn main() -> bits[4] {
  ret constant: bits[4] = literal(value=10, id=1)
}
"#,
            None,
        )
        .unwrap();
        let result = find_constant_flops(&ir, &[("main".into(), "main".into())], true).unwrap();
        let expected: Vec<_> = [false, true, false, true]
            .into_iter()
            .enumerate()
            .map(|(output_bit_index, value)| ConstantFlop {
                stage_name: "main".into(),
                stage_index: 0,
                output_bit_index,
                value,
            })
            .collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn accepts_variable_identity_signatures() {
        for case in interesting_ir_roundtrip_cases()
            .iter()
            .filter(|case| case.name.starts_with("identity_"))
        {
            let ir = IrPackage::parse_ir(case.ir_text, None).unwrap();
            assert_eq!(
                find_constant_flops(&ir, &[("main".into(), "main".into())], true).unwrap(),
                Vec::new(),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn models_live_gate_operations_without_changing_codegen() {
        let ir = IrPackage::parse_ir(
            r#"package sample
top fn main(enable: bits[1] id=1, x: bits[2] id=2) -> bits[3] {
  extended: bits[3] = zero_ext(x, new_bit_count=3, id=3)
  ret result: bits[3] = gate(enable, extended, id=4)
}
"#,
            None,
        )
        .unwrap();
        assert_eq!(
            find_constant_flops(&ir, &[("main".into(), "main".into())], true).unwrap(),
            vec![ConstantFlop {
                stage_name: "main".into(),
                stage_index: 0,
                output_bit_index: 2,
                value: false,
            }]
        );
    }

    #[test]
    fn preserves_nested_aggregate_layout_and_wide_values() {
        let ir = IrPackage::parse_ir(
            r#"package sample
top fn main(x: bits[65] id=1) -> ((bits[2], bits[1])[2], bits[65]) {
  low: bits[1] = bit_slice(x, start=0, width=1, id=2)
  inverse: bits[1] = not(low, id=3)
  one: bits[2] = literal(value=1, id=4)
  two: bits[2] = literal(value=2, id=5)
  first: (bits[2], bits[1]) = tuple(one, low, id=6)
  second: (bits[2], bits[1]) = tuple(two, inverse, id=7)
  array_value: (bits[2], bits[1])[2] = array(first, second, id=8)
  ret result: ((bits[2], bits[1])[2], bits[65]) = tuple(array_value, x, id=9)
}
"#,
            None,
        )
        .unwrap();
        let expected: Vec<_> = [(66, true), (67, false), (69, false), (70, true)]
            .into_iter()
            .map(|(output_bit_index, value)| ConstantFlop {
                stage_name: "main".into(),
                stage_index: 0,
                output_bit_index,
                value,
            })
            .collect();
        assert_eq!(
            find_constant_flops(&ir, &[("main".into(), "main".into())], true).unwrap(),
            expected
        );
    }

    #[test]
    fn preserves_cross_stage_correlations_in_parallel() {
        let ir = IrPackage::parse_ir(
            r#"package sample
fn start(x: bits[16] id=1, y: bits[16] id=2) -> (bits[16], bits[16], bits[16]) {
  difference: bits[16] = sub(x, y, id=3)
  ret result: (bits[16], bits[16], bits[16]) = tuple(difference, y, x, id=4)
}
fn finish(x: bits[16] id=5, y: bits[16] id=6, difference: bits[16] id=7) -> bits[16] {
  restored: bits[16] = add(difference, y, id=8)
  ret result: bits[16] = xor(restored, x, id=9)
}
"#,
            None,
        )
        .unwrap();
        let stages = vec![
            ("start".into(), "start".into()),
            ("finish".into(), "finish".into()),
        ];
        let expected: Vec<_> = (0..16)
            .map(|output_bit_index| ConstantFlop {
                stage_name: "finish".into(),
                stage_index: 1,
                output_bit_index,
                value: false,
            })
            .collect();
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        assert_eq!(
            pool.install(|| find_constant_flops(&ir, &stages, true))
                .unwrap(),
            expected
        );
        assert_eq!(
            pool.install(|| find_constant_flops(&ir, &stages, false))
                .unwrap(),
            vec![]
        );
    }

    #[test]
    fn matches_packed_wiring_for_unequal_parameter_widths() {
        let ir = IrPackage::parse_ir(
            r#"package sample
fn start(x: bits[1] id=1) -> (bits[2], bits[1]) {
  inverse: bits[1] = not(x, id=2)
  wide: bits[2] = concat(x, inverse, id=3)
  ret result: (bits[2], bits[1]) = tuple(wide, x, id=4)
}
fn finish(a: bits[2] id=5, b: bits[1] id=6) -> bits[1] {
  low: bits[1] = bit_slice(a, start=0, width=1, id=7)
  ret result: bits[1] = xor(low, b, id=8)
}
"#,
            None,
        )
        .unwrap();
        let stages = vec![
            ("start".into(), "start".into()),
            ("finish".into(), "finish".into()),
        ];
        // The emitted wrapper feeds a with the low two packed bits [x, !x]
        // and b with the high bit x. Semantic tuple unpacking would instead
        // feed a with [!x, x] and would incorrectly report constant one.
        assert_eq!(
            find_constant_flops(&ir, &stages, true).unwrap(),
            vec![ConstantFlop {
                stage_name: "finish".into(),
                stage_index: 1,
                output_bit_index: 0,
                value: false,
            }]
        );
    }

    #[test]
    fn checks_intermediate_registers_without_output_registers() {
        let ir = IrPackage::parse_ir(
            r#"package sample
fn start(x: bits[1] id=1) -> bits[1] {
  ret result: bits[1] = literal(value=1, id=2)
}
fn finish(x: bits[1] id=3) -> bits[1] {
  ret result: bits[1] = identity(x, id=4)
}
"#,
            None,
        )
        .unwrap();
        let stages = vec![
            ("start".into(), "start".into()),
            ("finish".into(), "finish".into()),
        ];
        assert_eq!(
            find_constant_flops(&ir, &stages, false).unwrap(),
            vec![ConstantFlop {
                stage_name: "start".into(),
                stage_index: 0,
                output_bit_index: 0,
                value: true,
            }]
        );
        assert_eq!(
            find_constant_flops(&ir, &stages[..1], false).unwrap(),
            vec![]
        );
    }
}
