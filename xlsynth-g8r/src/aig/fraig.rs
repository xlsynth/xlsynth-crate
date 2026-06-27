// SPDX-License-Identifier: Apache-2.0

//! We implement some form of "fraiging" here -- the basic idea is to concretely
//! evaluate a gate function over some set of random vectors and notice which
//! gates have equivalent outputs regardless of sample input. Those are proposed
//! candidates for verifying equivalence. For gates where we verify equivalence,
//! we replace gates that have higher depth with gates that have lower depth in
//! order to try to reduce critical path and, in likelihood, reduce the number
//! of gates required to implement the function.
//!
//! Note that we could make this replacement heuristic more sophisticated, e.g.
//! by understanding how many gates are in the fan-in cone (perhaps with some
//! exclusivity notion mixed in?) to select some tradeoff between depth and
//! fan-in cone. But anyway, replace deeper with less deep is an easy heuristic
//! for now.

use rand::Rng;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::time::Instant;
use xlsynth::IrBits;

use crate::aig::bulk_replace::{SubstitutionMap, bulk_replace};
use crate::aig::get_summary_stats::{GateDepthStats, get_gate_depth};
use crate::aig::{AigOperand, AigRef, GateFn};
use crate::{
    gate_builder::GateBuilderOptions,
    propose_equiv::{EquivNode, SimulationSignature, propose_equivalence_classes},
    prove_gate_fn_equiv_common::GateFormalBackend,
    prove_gate_fn_equiv_sat::{
        GateFormalOptions, validate_equivalence_classes_presorted_with_backend_and_options,
    },
};

/// Number of proposed equivalence classes validated before applying
/// replacements and re-proposing.
const FRAIG_CLASS_BATCH_SIZE: usize = 256;

pub enum IterationBounds {
    MaxIterations(usize),
    ToConvergence,
}

#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize)]
pub enum DidConverge {
    // Payload is the number of iterations taken to converge.
    Yes(usize),
    No,
}

// NB Direct impl of Into to avoid import issues with From not being in the same
// file as the struct definition.
impl Into<crate::result_proto::DidConverge> for DidConverge {
    fn into(self) -> crate::result_proto::DidConverge {
        match self {
            Self::Yes(val) => crate::result_proto::DidConverge {
                result: Some(crate::result_proto::did_converge::Result::Yes(
                    crate::result_proto::did_converge::Yes { count: val as u64 },
                )),
            },
            Self::No => crate::result_proto::DidConverge {
                result: Some(crate::result_proto::did_converge::Result::No(
                    crate::result_proto::did_converge::No {},
                )),
            },
        }
    }
}

// Add struct to record stats for each fraig iteration
#[derive(Debug, serde::Serialize)]
pub struct FraigIterationStat {
    pub gate_count: usize,
    pub counterexample_count: usize,
    pub proposed_equiv_classes: usize,
    pub replacements_count: usize,
}

/// Statistics for the one-shot simulation-constant sweep before FRAIG.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimulationConstantSweepStat {
    pub gate_count_before: usize,
    pub gate_count_after: usize,
    pub candidate_count: usize,
    pub initially_zero_candidate_count: usize,
    pub initially_one_candidate_count: usize,
    pub proof_query_count: usize,
    pub interrupted_proof_count: usize,
    pub counterexample_count: usize,
    pub proven_zero_count: usize,
    pub proven_one_count: usize,
    pub replacement_count: usize,
    pub simulation_seconds: f64,
    pub proof_seconds: f64,
}

/// Result of proving and replacing simulation-constant AIG nodes.
pub struct SimulationConstantSweepResult {
    pub gate_fn: GateFn,
    pub stat: SimulationConstantSweepStat,
    pub counterexamples: Vec<Vec<IrBits>>,
}

#[derive(Debug, Clone, Copy)]
struct SimulationConstantTarget {
    node_ref: AigRef,
    observed_value: bool,
}

impl Into<crate::result_proto::FraigIterationStat> for FraigIterationStat {
    fn into(self) -> crate::result_proto::FraigIterationStat {
        crate::result_proto::FraigIterationStat {
            gate_count: self.gate_count as u64,
            counterexample_count: self.counterexample_count as u64,
            proposed_equiv_classes: self.proposed_equiv_classes as u64,
            replacements_count: self.replacements_count as u64,
        }
    }
}

/// Orders equivalence classes by the shallowest candidate node in each class.
fn sort_equiv_classes_by_depth(
    equiv_classes: &std::collections::HashMap<
        crate::propose_equiv::SimulationSignature,
        Vec<EquivNode>,
    >,
    stats: &GateDepthStats,
) -> Vec<Vec<EquivNode>> {
    let mut sorted_classes: Vec<Vec<EquivNode>> = equiv_classes
        .values()
        .map(|nodes| {
            let mut sorted_nodes = nodes.clone();
            sorted_nodes.sort_unstable_by_key(|equiv_node| {
                let node_ref = equiv_node.aig_ref();
                (
                    stats.ref_to_depth[&node_ref],
                    equiv_node.is_inverted(),
                    node_ref.id,
                )
            });
            sorted_nodes
        })
        .collect();

    sorted_classes.sort_unstable_by_key(|equiv_class| {
        let representative = equiv_class[0];
        let node_ref = representative.aig_ref();
        (
            stats.ref_to_depth[&node_ref],
            equiv_class.len(),
            representative.is_inverted(),
            node_ref.id,
        )
    });
    sorted_classes
}

/// Builds substitutions from SAT-proven equivalence sets.
fn build_replacements_from_proven_sets(
    stats: &GateDepthStats,
    proven_equiv_sets: Vec<Vec<EquivNode>>,
) -> SubstitutionMap {
    // Mapping from nodes to be replaced to the (potentially negated) operand
    // that should replace them.
    let mut replacements = SubstitutionMap::new();
    for proven_equiv_set in proven_equiv_sets {
        log::debug!("fraig_optimize: proven_equiv_set: {:?}", proven_equiv_set);
        // Determine the minimum-depth node in the proven_equiv_set, using node ID as
        // tiebreaker
        let min_depth_node = proven_equiv_set
            .iter()
            .filter(|equiv_node| {
                // Any node that is being substituted cannot win as the target in an equivalence
                // class.
                //
                // The main reason this is necessary is because we consider both values and
                // their negations, so when two nodes have the same depth we see both:
                // - (normal_a, negated_b)
                // - (negated_a, normal_b)
                //
                // And in this case we need to make sure we don't make a chain, so we'll replace
                // negated_b with normal_a for the first pair and see that b is already being
                // replaced for the second set.
                let node_ref = equiv_node.aig_ref();
                let is_replaced = replacements.contains_key(&node_ref);
                !is_replaced
            })
            .min_by_key(|equiv_node| {
                let node_ref = equiv_node.aig_ref();
                let is_inverted = equiv_node.is_inverted();
                (stats.ref_to_depth[&node_ref], is_inverted, node_ref.id)
            });
        let Some(min_depth_node) = min_depth_node else {
            continue;
        };
        for equiv_node in proven_equiv_set.iter() {
            // if this is the "minimum depth" one in the equivalence class, we don't replace
            // it with anything
            if equiv_node == min_depth_node {
                continue;
            }
            // Determine if the substitution (the min depth node) needs negation
            let dst_operand = match (equiv_node, min_depth_node) {
                // Same polarity: substitute directly
                (EquivNode::Normal(_), EquivNode::Normal(rep_ref))
                | (EquivNode::Inverted(_), EquivNode::Inverted(rep_ref)) => {
                    AigOperand::from(rep_ref)
                }
                // Different polarity: substitute with negation
                (EquivNode::Normal(_), EquivNode::Inverted(rep_ref))
                | (EquivNode::Inverted(_), EquivNode::Normal(rep_ref)) => {
                    AigOperand::from(rep_ref).negate()
                }
            };

            log::debug!(
                "fraig_optimize: adding substitution: {:?} -> {:?}",
                equiv_node.aig_ref(),
                dst_operand
            );
            replacements.add(equiv_node.aig_ref(), dst_operand);
        }
    }
    replacements
}

/// Uses one full-graph SAT instance to prove simulation-constant nodes one at
/// a time, then replaces all proven constants in one graph rebuild.
pub fn sweep_simulation_constants_with_backend_and_options(
    gate_fn: &GateFn,
    input_sample_count: usize,
    validation_backend: GateFormalBackend,
    gate_formal_options: GateFormalOptions,
    rng: &mut impl Rng,
) -> Result<SimulationConstantSweepResult, Box<dyn Error>> {
    let simulation_start = Instant::now();
    let equiv_classes = propose_equivalence_classes(gate_fn, input_sample_count, rng, &[]);
    let simulation_seconds = simulation_start.elapsed().as_secs_f64();

    let mut targets_by_ref: BTreeMap<AigRef, bool> = BTreeMap::new();
    if input_sample_count != 0 {
        let zero_signature = SimulationSignature::constant_value(false, input_sample_count);
        if let Some(zero_class) = equiv_classes.get(&zero_signature) {
            for equiv_node in zero_class {
                let node_ref = equiv_node.aig_ref();
                if node_ref.id != 0 {
                    targets_by_ref
                        .entry(node_ref)
                        .or_insert(equiv_node.is_inverted());
                }
            }
        }
    }
    let targets: Vec<SimulationConstantTarget> = targets_by_ref
        .into_iter()
        .map(|(node_ref, observed_value)| SimulationConstantTarget {
            node_ref,
            observed_value,
        })
        .collect();
    let initially_zero_candidate_count = targets
        .iter()
        .filter(|target| !target.observed_value)
        .count();
    let initially_one_candidate_count = targets.len() - initially_zero_candidate_count;

    if targets.is_empty() {
        return Ok(SimulationConstantSweepResult {
            gate_fn: gate_fn.clone(),
            stat: SimulationConstantSweepStat {
                gate_count_before: gate_fn.gates.len(),
                gate_count_after: gate_fn.gates.len(),
                candidate_count: 0,
                initially_zero_candidate_count: 0,
                initially_one_candidate_count: 0,
                proof_query_count: 0,
                interrupted_proof_count: 0,
                counterexample_count: 0,
                proven_zero_count: 0,
                proven_one_count: 0,
                replacement_count: 0,
                simulation_seconds,
                proof_seconds: 0.0,
            },
            counterexamples: Vec::new(),
        });
    }

    let false_literal = EquivNode::Normal(AigRef { id: 0 });
    // The output singletons force the combined cone to contain the complete
    // live graph. They issue no proof queries themselves.
    let mut classes: Vec<Vec<EquivNode>> = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .map(|operand| {
            vec![if operand.negated {
                EquivNode::Inverted(operand.node)
            } else {
                EquivNode::Normal(operand.node)
            }]
        })
        .collect();
    classes.extend(targets.iter().map(|target| {
        let normalized_target = if target.observed_value {
            EquivNode::Inverted(target.node_ref)
        } else {
            EquivNode::Normal(target.node_ref)
        };
        vec![false_literal, normalized_target]
    }));
    let class_refs: Vec<&[EquivNode]> = classes.iter().map(Vec::as_slice).collect();
    let proof_start = Instant::now();
    let validation = validate_equivalence_classes_presorted_with_backend_and_options(
        gate_fn,
        &class_refs,
        validation_backend,
        gate_formal_options,
    )?;
    let proof_seconds = proof_start.elapsed().as_secs_f64();

    let mut proven_constant_refs = HashSet::new();
    for proven_set in validation.proven_equiv_sets {
        for equiv_node in proven_set {
            if equiv_node.aig_ref().id != 0 {
                proven_constant_refs.insert(equiv_node.aig_ref());
            }
        }
    }
    let mut counterexample_seen = HashSet::new();
    let counterexamples: Vec<Vec<IrBits>> = validation
        .cex_inputs
        .into_iter()
        .filter(|counterexample| counterexample_seen.insert(counterexample.clone()))
        .collect();

    let constant_false = AigOperand::from(AigRef { id: 0 });
    let mut replacements = SubstitutionMap::new();
    let mut proven_zero_count = 0usize;
    let mut proven_one_count = 0usize;
    for target in &targets {
        if !proven_constant_refs.contains(&target.node_ref) {
            continue;
        }
        if target.observed_value {
            proven_one_count += 1;
            replacements.add(target.node_ref, constant_false.negate());
        } else {
            proven_zero_count += 1;
            replacements.add(target.node_ref, constant_false);
        }
    }
    let replacement_count = replacements.len();
    let swept_fn = if replacement_count == 0 {
        gate_fn.clone()
    } else {
        bulk_replace(gate_fn, &replacements, GateBuilderOptions::opt())
    };
    let stat = SimulationConstantSweepStat {
        gate_count_before: gate_fn.gates.len(),
        gate_count_after: swept_fn.gates.len(),
        candidate_count: targets.len(),
        initially_zero_candidate_count,
        initially_one_candidate_count,
        proof_query_count: validation.proof_query_count,
        interrupted_proof_count: validation.interrupted_proof_count,
        counterexample_count: counterexamples.len(),
        proven_zero_count,
        proven_one_count,
        replacement_count,
        simulation_seconds,
        proof_seconds,
    };
    log::info!("pre-fraig simulation-constant sweep: {stat:?}");
    Ok(SimulationConstantSweepResult {
        gate_fn: swept_fn,
        stat,
        counterexamples,
    })
}

/// Runs FRAIG using the default validation backend.
pub fn fraig_optimize(
    f: &GateFn,
    input_sample_count: usize,
    iteration_bounds: IterationBounds,
    rng: &mut impl Rng,
) -> Result<(GateFn, DidConverge, Vec<FraigIterationStat>), Box<dyn Error>> {
    fraig_optimize_with_backend(
        f,
        input_sample_count,
        iteration_bounds,
        GateFormalBackend::Cadical,
        rng,
    )
}

/// Runs FRAIG using an explicit formal backend for equivalence validation.
pub fn fraig_optimize_with_backend(
    f: &GateFn,
    input_sample_count: usize,
    iteration_bounds: IterationBounds,
    validation_backend: GateFormalBackend,
    rng: &mut impl Rng,
) -> Result<(GateFn, DidConverge, Vec<FraigIterationStat>), Box<dyn Error>> {
    fraig_optimize_with_backend_and_options(
        f,
        input_sample_count,
        iteration_bounds,
        validation_backend,
        GateFormalOptions::default(),
        rng,
    )
}

/// Runs FRAIG with backend-specific formal proof limits.
pub fn fraig_optimize_with_backend_and_options(
    f: &GateFn,
    input_sample_count: usize,
    iteration_bounds: IterationBounds,
    validation_backend: GateFormalBackend,
    gate_formal_options: GateFormalOptions,
    rng: &mut impl Rng,
) -> Result<(GateFn, DidConverge, Vec<FraigIterationStat>), Box<dyn Error>> {
    let mut iteration_count = 0;
    let mut current_fn = f.clone();
    let mut counterexample_seen: HashSet<Vec<IrBits>> = HashSet::new();
    let mut counterexamples: Vec<Vec<IrBits>> = Vec::new();
    let mut ran_simulation_constant_sweep = false;
    // Initialize iteration stats collection
    let mut iteration_stats: Vec<FraigIterationStat> = Vec::new();
    loop {
        log::info!(
            "fraig_optimize; iteration: {} counterexamples: {}",
            iteration_count,
            counterexamples.len()
        );
        match iteration_bounds {
            IterationBounds::MaxIterations(max_iterations) => {
                if iteration_count >= max_iterations {
                    break;
                }
            }
            IterationBounds::ToConvergence => {
                // Keep going!
            }
        }
        if !ran_simulation_constant_sweep {
            ran_simulation_constant_sweep = true;
            if matches!(
                validation_backend,
                GateFormalBackend::Cadical | GateFormalBackend::Varisat
            ) {
                let sweep = sweep_simulation_constants_with_backend_and_options(
                    &current_fn,
                    input_sample_count,
                    validation_backend,
                    gate_formal_options,
                    rng,
                )?;
                for counterexample in sweep.counterexamples {
                    if counterexample_seen.insert(counterexample.clone()) {
                        counterexamples.push(counterexample);
                    }
                }
                current_fn = sweep.gate_fn;
            }
        }
        let equiv_classes =
            propose_equivalence_classes(&current_fn, input_sample_count, rng, &counterexamples);

        log::info!(
            "fraig_optimize: propose_equiv proposed {} classes",
            equiv_classes.len()
        );

        let live_nodes: Vec<AigRef> = current_fn
            .gates
            .iter()
            .enumerate()
            .map(|(i, _)| AigRef { id: i })
            .collect();
        let stats = get_gate_depth(&current_fn, &live_nodes);
        let sorted_equiv_classes = sort_equiv_classes_by_depth(&equiv_classes, &stats);
        let batch_size = FRAIG_CLASS_BATCH_SIZE.min(sorted_equiv_classes.len().max(1));
        let is_batched = batch_size < sorted_equiv_classes.len();
        let mut applied_replacements = false;
        let mut restarted_after_counterexample = false;

        for (batch_index, batch) in sorted_equiv_classes.chunks(batch_size).enumerate() {
            if is_batched {
                log::info!(
                    "fraig_optimize: validating class batch {} classes [{}..{}) of {}",
                    batch_index,
                    batch_index * batch_size,
                    batch_index * batch_size + batch.len(),
                    sorted_equiv_classes.len()
                );
            }
            let equiv_classes_vec: Vec<&[EquivNode]> = batch.iter().map(|v| v.as_slice()).collect();
            let validation_result =
                validate_equivalence_classes_presorted_with_backend_and_options(
                    &current_fn,
                    &equiv_classes_vec,
                    validation_backend,
                    gate_formal_options,
                )?;

            let mut new_counterexample_count = 0usize;
            for cex in validation_result.cex_inputs {
                if counterexample_seen.insert(cex.clone()) {
                    counterexamples.push(cex);
                    new_counterexample_count += 1;
                }
            }

            let replacements =
                build_replacements_from_proven_sets(&stats, validation_result.proven_equiv_sets);

            if replacements.len() == 0 {
                if is_batched && new_counterexample_count > 0 {
                    log::info!(
                        "fraig_optimize: class batch {} found {} new counterexamples and no replacements; reproposing",
                        batch_index,
                        new_counterexample_count
                    );
                    iteration_stats.push(FraigIterationStat {
                        gate_count: current_fn.gates.len(),
                        counterexample_count: counterexamples.len(),
                        proposed_equiv_classes: equiv_classes.len(),
                        replacements_count: 0,
                    });
                    restarted_after_counterexample = true;
                    break;
                }
                continue;
            }

            // Record stats for this iteration before replacements
            iteration_stats.push(FraigIterationStat {
                gate_count: current_fn.gates.len(),
                counterexample_count: counterexamples.len(),
                proposed_equiv_classes: equiv_classes.len(),
                replacements_count: replacements.len(),
            });

            // We get the updated function by bulk replacing nodes with their lower-depth
            // equivalents here.
            let new_fn = bulk_replace(&current_fn, &replacements, GateBuilderOptions::opt());
            current_fn = new_fn;
            applied_replacements = true;
            break;
        }

        if !applied_replacements && !restarted_after_counterexample {
            // Converged -- no proven equivalences found.
            // Record stats for this iteration with zero replacements
            iteration_stats.push(FraigIterationStat {
                gate_count: current_fn.gates.len(),
                counterexample_count: counterexamples.len(),
                proposed_equiv_classes: equiv_classes.len(),
                replacements_count: 0,
            });
            return Ok((
                current_fn,
                DidConverge::Yes(iteration_count),
                iteration_stats,
            ));
        }

        iteration_count += 1;
    }
    Ok((current_fn, DidConverge::No, iteration_stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::assert_within;
    use crate::{
        aig::{AigBitVector, get_summary_stats::get_summary_stats},
        aig_sim::gate_sim,
        check_equivalence,
        gate_builder::GateBuilder,
        test_utils::setup_padded_graph_with_equal_depth_opposite_polarity,
    };
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn simulation_constant_sweep_replaces_only_proven_constants() {
        let mut builder =
            GateBuilder::new("constant_sweep".to_string(), GateBuilderOptions::no_opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let constant_zero = builder.add_and_binary(a, a.negate());
        let constant_one = builder.add_and_binary(constant_zero.negate(), constant_zero.negate());
        let input_dependent = builder.add_and_binary(a, b);
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[constant_zero, constant_one, input_dependent]),
        );
        let gate_fn = builder.build();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let result = sweep_simulation_constants_with_backend_and_options(
            &gate_fn,
            1,
            GateFormalBackend::Cadical,
            GateFormalOptions::default().with_cadical_terminate_limit(100),
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.stat.candidate_count, 3);
        assert_eq!(result.stat.proven_zero_count, 1);
        assert_eq!(result.stat.proven_one_count, 1);
        assert_eq!(result.stat.replacement_count, 2);
        assert_eq!(result.stat.counterexample_count, 1);
        assert!(result.stat.gate_count_after < result.stat.gate_count_before);
        for a in [false, true] {
            for b in [false, true] {
                let inputs = [IrBits::bool(a), IrBits::bool(b)];
                let original = gate_sim::eval(&gate_fn, &inputs, gate_sim::Collect::None);
                let swept = gate_sim::eval(&result.gate_fn, &inputs, gate_sim::Collect::None);
                assert_eq!(original.outputs, swept.outputs);
            }
        }
    }

    #[test]
    fn test_equiv_class_with_equal_depth_and_opposite_polarity_canonicalizes() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_graph = setup_padded_graph_with_equal_depth_opposite_polarity();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let (optimized_fn, _did_converge, _iteration_stats) =
            fraig_optimize(&test_graph.g, 8, IterationBounds::ToConvergence, &mut rng)
                .expect("fraig_optimize should not panic");

        // The original graph has 2 inputs, 1 constant literal node (e.g., 'true', which
        // is node 0), and 2 AND nodes (one for the AND, one for the
        // AND-with-true used to force a real node for the inverted output).
        let orig_stats = get_summary_stats(&test_graph.g);
        assert_eq!(orig_stats.live_nodes, 5);

        // After optimization, DCE removes both the extra AND node and the now-unused
        // constant literal node. Only the two inputs and the single AND node
        // remain live, so live_nodes should decrease by 2: from 5 (2 inputs, 1
        // constant, 2 ANDs) to 3 (2 inputs, 1 AND node).
        let opt_stats = get_summary_stats(&optimized_fn);
        assert_eq!(
            opt_stats.live_nodes,
            orig_stats.live_nodes - 2,
            "Should eliminate two nodes"
        );

        // Outputs should be equivalent
        check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&test_graph.g, &optimized_fn)
            .expect("fraig optimization should preserve equivalence");
    }

    #[test]
    fn test_fraig_with_dead_equiv_node_no_panic() {
        let _ = env_logger::builder().is_test(true).try_init();

        // Create a simple 1-bit AND gate that is live plus an identical dead gate.
        let mut gb = GateBuilder::new("dead_redundant".to_string(), GateBuilderOptions::no_opt());
        let in0 = gb.add_input("a".to_string(), 1);
        let in1 = gb.add_input("b".to_string(), 1);
        let a0 = *in0.get_lsb(0);
        let b0 = *in1.get_lsb(0);

        // Live gate used in the output.
        let and_live = gb.add_and_binary(a0, b0);
        // Identical gate that is *not* connected to any output.
        let _and_dead = gb.add_and_binary(a0, b0);

        gb.add_output("out".to_string(), and_live.into());

        let gate_fn = gb.build();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let (optimized_fn, _did_conv, _stats) =
            fraig_optimize(&gate_fn, 64, IterationBounds::ToConvergence, &mut rng)
                .expect("fraig_optimize should not panic on dead redundant nodes");

        // The optimized function must remain equivalent to the original.
        crate::check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&gate_fn, &optimized_fn)
            .expect("optimized function should be equivalent to original");

        // Ensure the optimized function still respects basic invariants.
        optimized_fn.check_invariants_with_debug_assert();
    }
}
