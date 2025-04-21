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
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};
use xlsynth::IrBits;

use crate::{
    bulk_replace::bulk_replace, gate::AigOperand, gate::AigRef, gate::GateFn,
    gate_builder::GateBuilderOptions, get_summary_stats::get_gate_depth,
    propose_equiv::propose_equiv, propose_equiv::EquivNode, validate_equiv::validate_equiv,
};

pub enum IterationBounds {
    MaxIterations(usize),
    ToConvergence,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DidConverge {
    // Payload is the number of iterations taken to converge.
    Yes(usize),
    No,
}

// TODO(cdleary): 2025-04-18 Enable the solver context for verisat to be reused
// across fraig iterations -- we can probably use a single graph as an arena
// instead of rebuilding it each time to help preserve all clause info.
pub fn fraig_optimize(
    f: &GateFn,
    input_sample_count: usize,
    iteration_bounds: IterationBounds,
    rng: &mut impl Rng,
) -> Result<(GateFn, DidConverge), Box<dyn Error>> {
    let mut iteration_count = 0;
    let mut current_fn = f.clone();
    let mut counterexamples: HashSet<Vec<IrBits>> = HashSet::new();
    loop {
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
        let equiv_classes = propose_equiv(&current_fn, input_sample_count, rng, &counterexamples);

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

        let validation_result = validate_equiv(&current_fn, &equiv_classes_vec)?;

        if validation_result.proven_equiv_sets.is_empty() {
            // Converged -- no proven equivalences found.
            return Ok((current_fn, DidConverge::Yes(iteration_count)));
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
        let mut replacements: HashMap<AigRef, AigOperand> = HashMap::new();
        for proven_equiv_set in validation_result.proven_equiv_sets {
            // Determine the minimum-depth node in the proven_equiv_set, using node ID as
            // tiebreaker
            let min_depth_node = proven_equiv_set
                .iter()
                .min_by_key(|equiv_node| {
                    let node_ref = equiv_node.aig_ref();
                    let is_inverted = equiv_node.is_inverted();
                    (stats.ref_to_depth[&node_ref], is_inverted, node_ref.id)
                })
                .unwrap();
            for equiv_node in proven_equiv_set.iter() {
                if equiv_node == min_depth_node {
                    continue;
                }
                // Determine if the substitution needs negation
                let representative_op = match (equiv_node, min_depth_node) {
                    // Same polarity: substitute directly
                    (EquivNode::Normal(_), EquivNode::Normal(rep_ref))
                    | (EquivNode::Inverted(_), EquivNode::Inverted(rep_ref)) => {
                        AigOperand::from(*rep_ref)
                    }
                    // Different polarity: substitute with negation
                    (EquivNode::Normal(_), EquivNode::Inverted(rep_ref))
                    | (EquivNode::Inverted(_), EquivNode::Normal(rep_ref)) => {
                        AigOperand::from(*rep_ref).negate()
                    }
                };

                replacements.insert(equiv_node.aig_ref(), representative_op);
            }
        }

        // We get the updated function by bulk replacing nodes with their lower-depth
        // equivalents here.
        let (new_fn, _) =
            bulk_replace(&current_fn, &replacements, GateBuilderOptions::opt(), false);
        current_fn = new_fn;

        iteration_count += 1;
    }
    Ok((current_fn, DidConverge::No))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        check_equivalence,
        get_summary_stats::get_summary_stats,
        test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt},
    };

    use super::*;

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
        let mut rng = StdRng::seed_from_u64(0);
        let (optimized_fn, did_converge) = fraig_optimize(
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
        check_equivalence::validate_same_gate_fn(&gate_fn, &optimized_fn)
            .expect("fraig optimization should preserve equivalence");
        let check_equiv_elapsed = check_equiv_start.elapsed();
        eprintln!("{}: check_equiv took {:?}", name, check_equiv_elapsed);

        OptimizationResults {
            did_converge,
            original_nodes: orig.live_nodes,
            optimized_nodes: opt.live_nodes,
            original_depth: orig.deepest_path,
            optimized_depth: opt.deepest_path,
        }
    }

    #[test]
    fn test_fraig_optimize_bf16_mul() {
        let loaded = load_bf16_mul_sample(Opt::Yes);
        let results = do_fraig_and_report(&loaded.gate_fn, 512, "bf16_mul");
        assert_eq!(results.did_converge, DidConverge::Yes(1));
        assert_eq!(results.original_nodes, 1172);
        assert_eq!(results.optimized_nodes, 1147);
        assert_eq!(results.original_depth, 109);
        assert_eq!(results.optimized_depth, 105);
    }

    #[test]
    fn test_fraig_optimize_bf16_add() {
        let loaded = load_bf16_add_sample(Opt::Yes);
        let results = do_fraig_and_report(&loaded.gate_fn, 512, "bf16_add");
        assert_eq!(results.did_converge, DidConverge::Yes(3));
        assert_eq!(results.original_nodes, 1292);
        assert_eq!(results.optimized_nodes, 1125);
        assert_eq!(results.original_depth, 130);
        assert_eq!(results.optimized_depth, 124);
    }
}
