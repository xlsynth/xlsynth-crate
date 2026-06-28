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
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::time::Instant;

use crate::aig::bulk_replace::{SubstitutionMap, bulk_replace};
use crate::aig::get_summary_stats::{GateDepthStats, get_gate_depth};
use crate::aig::{AigOperand, AigRef, GateFn};
use crate::{
    gate_builder::GateBuilderOptions,
    propose_equiv::{
        EquivNode, SimulationPatternBank, SimulationSignature,
        propose_equivalence_classes_from_patterns,
    },
    prove_gate_fn_equiv_common::GateFormalBackend,
    prove_gate_fn_equiv_sat::{
        GateFormalOptions, validate_equivalence_classes_presorted_with_virtual_rewrite_and_options,
        validate_full_graph_fraig_with_backend_and_options,
    },
};

/// Statistics for the single FRAIG proposal-and-validation pass.
#[derive(Debug, serde::Serialize)]
pub struct FraigPassStat {
    pub gate_count: usize,
    pub counterexample_count: usize,
    pub proposed_equiv_classes: usize,
    pub replacements_count: usize,
}

/// Result of the single FRAIG proposal-and-validation pass.
pub struct FraigOptimizationResult {
    pub optimized_fn: GateFn,
    pub stat: FraigPassStat,
}

impl Into<crate::result_proto::FraigPassStat> for FraigPassStat {
    fn into(self) -> crate::result_proto::FraigPassStat {
        crate::result_proto::FraigPassStat {
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
            representative.is_inverted(),
            node_ref.id,
            equiv_class.len(),
        )
    });
    sorted_classes
}

fn constant_one_signature(pattern_bank: &SimulationPatternBank) -> Option<SimulationSignature> {
    (pattern_bank.sample_count() != 0)
        .then(|| SimulationSignature::constant_value(true, pattern_bank.sample_count()))
}

/// Selects proposal classes for FRAIG. The constant-one class is the polarity
/// mirror of the constant-zero class, so only the latter is retained.
fn select_and_sort_equivalence_classes(
    equiv_classes: &HashMap<SimulationSignature, Vec<EquivNode>>,
    pattern_bank: &SimulationPatternBank,
    depth_stats: &GateDepthStats,
) -> Vec<Vec<EquivNode>> {
    let one_signature = constant_one_signature(pattern_bank);
    let mut selected: HashMap<SimulationSignature, Vec<EquivNode>> = HashMap::new();
    for (&signature, equiv_class) in equiv_classes {
        let is_one = one_signature
            .map(|one_signature| signature == one_signature)
            .unwrap_or(false);
        if is_one {
            continue;
        }
        let filtered: Vec<EquivNode> = equiv_class
            .iter()
            .copied()
            .filter(|candidate| candidate.aig_ref().id != 0)
            .collect();
        if filtered.len() > 1 {
            selected.insert(signature, filtered);
        }
    }
    sort_equiv_classes_by_depth(&selected, depth_stats)
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

/// Runs FRAIG using the default validation backend.
pub fn fraig_optimize(
    f: &GateFn,
    input_sample_count: usize,
    rng: &mut impl Rng,
) -> Result<FraigOptimizationResult, Box<dyn Error>> {
    fraig_optimize_with_backend(f, input_sample_count, GateFormalBackend::Cadical, rng)
}

/// Runs FRAIG using an explicit formal backend for equivalence validation.
pub fn fraig_optimize_with_backend(
    f: &GateFn,
    input_sample_count: usize,
    validation_backend: GateFormalBackend,
    rng: &mut impl Rng,
) -> Result<FraigOptimizationResult, Box<dyn Error>> {
    fraig_optimize_with_backend_and_options(
        f,
        input_sample_count,
        validation_backend,
        GateFormalOptions::default(),
        rng,
    )
}

/// Runs FRAIG with backend-specific formal proof limits.
pub fn fraig_optimize_with_backend_and_options(
    f: &GateFn,
    input_sample_count: usize,
    validation_backend: GateFormalBackend,
    gate_formal_options: GateFormalOptions,
    rng: &mut impl Rng,
) -> Result<FraigOptimizationResult, Box<dyn Error>> {
    let fraig_start = Instant::now();
    let simulation_start = Instant::now();
    let simulation_patterns =
        SimulationPatternBank::with_random_samples(f, input_sample_count, rng);
    let live_nodes: Vec<AigRef> = f
        .gates
        .iter()
        .enumerate()
        .map(|(id, _)| AigRef { id })
        .collect();
    let depth_stats = get_gate_depth(f, &live_nodes);
    let initial_equiv_classes = propose_equivalence_classes_from_patterns(f, &simulation_patterns);
    let simulation_seconds = simulation_start.elapsed().as_secs_f64();
    log::info!(
        "fraig simulation and class proposal: seconds={:.6}, samples={}, classes={}",
        simulation_seconds,
        simulation_patterns.sample_count(),
        initial_equiv_classes.len(),
    );
    let proposed_equiv_class_count = initial_equiv_classes.len();
    let sorted_classes = select_and_sort_equivalence_classes(
        &initial_equiv_classes,
        &simulation_patterns,
        &depth_stats,
    );

    let final_validation = if matches!(
        validation_backend,
        GateFormalBackend::Cadical | GateFormalBackend::Varisat
    ) {
        let full_result = validate_full_graph_fraig_with_backend_and_options(
            f,
            &sorted_classes,
            validation_backend,
            gate_formal_options,
        )?;
        log::info!(
            "fraig single-session full-graph validation: {}",
            full_result.stat
        );
        full_result.validation
    } else {
        let class_refs: Vec<&[EquivNode]> = sorted_classes.iter().map(Vec::as_slice).collect();
        validate_equivalence_classes_presorted_with_virtual_rewrite_and_options(
            f,
            &class_refs,
            validation_backend,
            gate_formal_options,
        )?
    };
    let counterexample_count = final_validation
        .cex_inputs
        .into_iter()
        .collect::<HashSet<_>>()
        .len();

    let replacements =
        build_replacements_from_proven_sets(&depth_stats, final_validation.proven_equiv_sets);
    let stat = FraigPassStat {
        gate_count: f.gates.len(),
        counterexample_count,
        proposed_equiv_classes: proposed_equiv_class_count,
        replacements_count: replacements.len(),
    };
    log::info!("fraig_optimize: single-session result: {stat:?}");
    let rewrite_start = Instant::now();
    let optimized_fn = if replacements.len() == 0 {
        f.clone()
    } else {
        bulk_replace(f, &replacements, GateBuilderOptions::opt())
    };
    log::info!(
        "fraig timing summary: total_seconds={:.6}, simulation_seconds={:.6}, rewrite_seconds={:.6}",
        fraig_start.elapsed().as_secs_f64(),
        simulation_seconds,
        rewrite_start.elapsed().as_secs_f64(),
    );
    Ok(FraigOptimizationResult { optimized_fn, stat })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::assert_within;
    use crate::{
        aig::get_summary_stats::get_summary_stats,
        check_equivalence,
        gate_builder::GateBuilder,
        test_utils::{
            setup_graph_with_redundancies, setup_padded_graph_with_equal_depth_opposite_polarity,
        },
    };
    use rand::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_equiv_class_with_equal_depth_and_opposite_polarity_canonicalizes() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_graph = setup_padded_graph_with_equal_depth_opposite_polarity();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let optimized_fn = fraig_optimize(&test_graph.g, 8, &mut rng)
            .expect("fraig_optimize should not panic")
            .optimized_fn;

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
        let optimized_fn = fraig_optimize(&gate_fn, 64, &mut rng)
            .expect("fraig_optimize should not panic on dead redundant nodes")
            .optimized_fn;

        // The optimized function must remain equivalent to the original.
        crate::check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&gate_fn, &optimized_fn)
            .expect("optimized function should be equivalent to original");

        // Ensure the optimized function still respects basic invariants.
        optimized_fn.check_invariants_with_debug_assert();
    }

    #[test]
    fn fraig_generates_random_patterns_once() {
        let graph = setup_graph_with_redundancies();
        let sample_count = 64;

        let mut expected_rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let _expected_patterns =
            SimulationPatternBank::with_random_samples(&graph.g, sample_count, &mut expected_rng);
        let expected_next = expected_rng.next_u64();

        let mut actual_rng = Xoshiro256PlusPlus::seed_from_u64(0);
        fraig_optimize(&graph.g, sample_count, &mut actual_rng).unwrap();
        let actual_next = actual_rng.next_u64();

        assert_eq!(actual_next, expected_next);
    }
}
