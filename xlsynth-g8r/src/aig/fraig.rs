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
use std::{collections::HashSet, error::Error};
use xlsynth::IrBits;

use crate::aig::bulk_replace::{SubstitutionMap, bulk_replace};
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::{AigOperand, AigRef, GateFn};
use crate::{
    gate_builder::GateBuilderOptions, propose_equiv::EquivNode,
    propose_equiv::propose_equivalence_classes,
    prove_gate_fn_equiv_varisat::validate_equivalence_classes,
};

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

// Add struct to record stats for each fraig iteration
#[derive(Debug, serde::Serialize)]
pub struct FraigIterationStat {
    pub gate_count: usize,
    pub counterexample_count: usize,
    pub proposed_equiv_classes: usize,
    pub replacements_count: usize,
}

pub fn fraig_optimize(
    f: &GateFn,
    input_sample_count: usize,
    iteration_bounds: IterationBounds,
    rng: &mut impl Rng,
) -> Result<(GateFn, DidConverge, Vec<FraigIterationStat>), Box<dyn Error>> {
    let mut iteration_count = 0;
    let mut current_fn = f.clone();
    let mut counterexamples: HashSet<Vec<IrBits>> = HashSet::new();
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
        let equiv_classes =
            propose_equivalence_classes(&current_fn, input_sample_count, rng, &counterexamples);

        log::info!(
            "fraig_optimize: propose_equiv proposed {} classes",
            equiv_classes.len()
        );

        let mut equiv_classes_vec: Vec<&[EquivNode]> =
            equiv_classes.values().map(|v| v.as_slice()).collect();

        // Sort the slices deterministically to ensure validate_equiv gets them in a
        // stable order. We sort by the canonical representative node within each class
        // slice.
        equiv_classes_vec.sort_unstable_by_key(|slice| {
            // Find the representative node (first after sorting the slice)
            // Cloning is necessary as sort needs mutable access
            let mut sorted_slice = slice.to_vec();
            sorted_slice.sort_unstable();
            // The key is (is_inverted, node_id) of the representative
            let rep_node = sorted_slice[0];
            (rep_node.is_inverted(), rep_node.aig_ref().id)
        });

        let validation_result = validate_equivalence_classes(&current_fn, &equiv_classes_vec)?;

        if validation_result.proven_equiv_sets.is_empty() {
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
        // Accumulate counterexamples for next iteration
        for cex in validation_result.cex_inputs {
            counterexamples.insert(cex);
        }
        let live_nodes: Vec<AigRef> = current_fn
            .gates
            .iter()
            .enumerate()
            .map(|(i, _)| AigRef { id: i })
            .collect();
        let stats = get_gate_depth(&current_fn, &live_nodes);

        // Mapping from nodes to be replaced to the (potentially negated) operand
        // that should replace them.
        let mut replacements = SubstitutionMap::new();
        for proven_equiv_set in validation_result.proven_equiv_sets {
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
        aig::get_summary_stats::get_summary_stats, check_equivalence, gate_builder::GateBuilder,
        test_utils::setup_padded_graph_with_equal_depth_opposite_polarity,
    };
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::time::Instant;

    const FRAIG_SEED: u64 = 0;

    #[derive(Debug)]
    pub struct OptimizationResults {
        pub did_converge: DidConverge,
        pub original_nodes: usize,
        pub optimized_nodes: usize,
        pub original_depth: usize,
        pub optimized_depth: usize,
    }

    fn do_fraig_and_report(
        gate_fn: &GateFn,
        input_sample_count: usize,
        name: &str,
    ) -> OptimizationResults {
        let _ = env_logger::builder().is_test(true).try_init();
        log::info!("do_fraig_and_report: {}", name);
        let start = Instant::now();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(FRAIG_SEED);
        let (optimized_fn, did_converge, _iteration_stats) = fraig_optimize(
            gate_fn,
            input_sample_count,
            IterationBounds::ToConvergence,
            &mut rng,
        )
        .unwrap();
        let elapsed = start.elapsed();

        let orig = get_summary_stats(gate_fn);
        let opt = get_summary_stats(&optimized_fn);

        let live_node_improvement_percent =
            (orig.live_nodes - opt.live_nodes) as f64 / orig.live_nodes as f64;
        eprintln!(
            "{}: live nodes: {} -> {} ({:.2}% improvement)",
            name,
            orig.live_nodes,
            opt.live_nodes,
            live_node_improvement_percent * 100.0
        );
        let depth_improvement_percent =
            (orig.deepest_path - opt.deepest_path) as f64 / orig.deepest_path as f64;
        eprintln!(
            "{}: depth: {} -> {} ({:.2}% improvement)",
            name,
            orig.deepest_path,
            opt.deepest_path,
            depth_improvement_percent * 100.0
        );
        eprintln!("{}: took {:?}", name, elapsed);

        let check_equiv_start = Instant::now();
        check_equivalence::prove_same_gate_fn_via_ir(&gate_fn, &optimized_fn)
            .expect("fraig optimization should preserve equivalence");
        let check_equiv_elapsed = check_equiv_start.elapsed();
        eprintln!("{}: check_equiv took {:?}", name, check_equiv_elapsed);

        // Ensure the optimized function still respects basic invariants.
        optimized_fn.check_invariants_with_debug_assert();

        OptimizationResults {
            did_converge,
            original_nodes: orig.live_nodes,
            optimized_nodes: opt.live_nodes,
            original_depth: orig.deepest_path,
            optimized_depth: opt.deepest_path,
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
        check_equivalence::prove_same_gate_fn_via_ir(&test_graph.g, &optimized_fn)
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
        crate::check_equivalence::prove_same_gate_fn_via_ir(&gate_fn, &optimized_fn)
            .expect("optimized function should be equivalent to original");

        // Ensure the optimized function still respects basic invariants.
        optimized_fn.check_invariants_with_debug_assert();
    }
}
