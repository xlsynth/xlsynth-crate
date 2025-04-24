// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;

use crate::dce::dce;
use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

/// Strongly-typed wrapper for substitutions to prevent chaining.
#[derive(Debug, Clone)]
pub struct SubstitutionMap {
    map: HashMap<AigRef, AigOperand>,
}

impl SubstitutionMap {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Adds a substitution, asserting that the value is not also a key (no
    /// chaining).
    pub fn add(&mut self, key: AigRef, value: AigOperand) {
        assert!(
            !self.map.contains_key(&value.node),
            "Substitution chain detected: value {:?} is also a key.",
            value.node
        );
        assert!(
            value.node != key,
            "Substitution with self detected: {:?} -> {:?}",
            key,
            value
        );
        self.map.insert(key, value);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = &AigRef> {
        self.map.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &AigOperand> {
        self.map.values()
    }

    pub fn get(&self, key: &AigRef) -> Option<&AigOperand> {
        self.map.get(key)
    }

    pub fn contains_key(&self, key: &AigRef) -> bool {
        self.map.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&AigRef, &AigOperand)> {
        self.map.iter()
    }
}

/// Replaces specified AIG nodes in a `GateFn` with other nodes and rebuilds the
/// function.
///
/// This function traverses the original `GateFn` (`orig_fn`) starting from its
/// outputs. It builds a new `GateFn` where each `AigRef` present as a key in
/// the `substitutions` map is replaced by the corresponding value `AigRef`.
///
/// The traversal ensures that only nodes reachable from the outputs in the new
/// graph are included, effectively removing dead code resulting from the
/// substitutions.
///
/// Tags associated with a replaced node are migrated to its substitute node in
/// the new graph. If a substitute node also has tags, the migrated tags are
/// appended.
///
/// # Arguments
///
/// * `orig_fn`: The original `GateFn` to transform.
/// * `substitutions`: A map where keys are `AigRef`s in `orig_fn` to be
///   replaced, and values are the `AigRef`s to substitute them with.
///   Substitutions are processed before the node itself is processed.
/// * `options`: `GateBuilderOptions` (e.g., folding, hashing) to use for the
///   new `GateFn`.
/// * `verify`: Whether to verify the output `GateFn` against the original, e.g.
///   I/O signatures.
///
/// # Returns
///
/// A tuple containing:
/// * The new `GateFn` with the substitutions applied.
/// * A map from original `AigRef`s to their final corresponding `AigOperand`s
///   in the new graph.
pub fn bulk_replace(
    orig_fn: &GateFn,
    substitutions: &SubstitutionMap,
    options: GateBuilderOptions,
) -> (GateFn, HashMap<AigRef, AigOperand>) {
    log::info!(
        "bulk_replace: fn {:?} with {} substitutions",
        orig_fn.name,
        substitutions.len()
    );

    #[cfg(debug_assertions)]
    {
        // Ensure no substitution chains (A->B, B->C)
        for substitute_op in substitutions.values() {
            debug_assert!(
                !substitutions.contains_key(&substitute_op.node),
                "Substitution chain detected: {:?} is a target and also a key in the substitution map. Chains are not supported.",
                substitute_op.node
            );
        }

        // Ensure no input nodes are being substituted
        let input_node_refs: HashSet<AigRef> = orig_fn
            .inputs
            .iter()
            .flat_map(|input_vec| {
                (0..input_vec.get_bit_count()).map(move |i| input_vec.bit_vector.get_lsb(i).node)
            })
            .collect();

        for orig_ref in substitutions.keys() {
            debug_assert!(
                !input_node_refs.contains(orig_ref),
                "Attempting to substitute an input node {:?}, which is not allowed.",
                orig_ref
            );
            debug_assert!(
                orig_ref.id != 0,
                "Attempting to substitute the constant literal node 0, which is not allowed."
            );
        }
    }

    let (substituted_fn, orig_to_new_map) = bulk_substitute(orig_fn, substitutions, options);

    log::info!("bulk_replace; running dce on substituted fn");
    substituted_fn.check_invariants_with_debug_assert();
    let dce_fn = dce(&substituted_fn);
    dce_fn.check_invariants_with_debug_assert();

    verify_io_signature_with_debug_assert(orig_fn, &dce_fn);

    log::info!("bulk_replace: done");
    (dce_fn, orig_to_new_map)
}

/// Verifies that the input/output signature (name, bit count) is preserved
/// between the original and replaced GateFn.
fn verify_io_signature_with_debug_assert(orig_fn: &GateFn, replaced_fn: &GateFn) {
    if !cfg!(debug_assertions) {
        return;
    }
    assert_eq!(
        orig_fn.inputs.len(),
        replaced_fn.inputs.len(),
        "Input count mismatch"
    );
    for (i, (orig_in, new_in)) in orig_fn
        .inputs
        .iter()
        .zip(replaced_fn.inputs.iter())
        .enumerate()
    {
        assert_eq!(
            orig_in.name, new_in.name,
            "Input name mismatch at index {}",
            i
        );
        assert_eq!(
            orig_in.get_bit_count(),
            new_in.get_bit_count(),
            "Input bit count mismatch for {}",
            orig_in.name
        );
    }
    assert_eq!(
        orig_fn.outputs.len(),
        replaced_fn.outputs.len(),
        "Output count mismatch"
    );
    for (i, (orig_out, new_out)) in orig_fn
        .outputs
        .iter()
        .zip(replaced_fn.outputs.iter())
        .enumerate()
    {
        assert_eq!(
            orig_out.name, new_out.name,
            "Output name mismatch at index {}",
            i
        );
        assert_eq!(
            orig_out.get_bit_count(),
            new_out.get_bit_count(),
            "Output bit count mismatch for {}",
            orig_out.name
        );
    }
}

/// Worklist-based bulk substitute: applies substitutions and builds a new
/// GateFn.
pub fn bulk_substitute(
    orig_fn: &GateFn,
    substitutions: &SubstitutionMap,
    options: GateBuilderOptions,
) -> (GateFn, HashMap<AigRef, AigOperand>) {
    log::info!(
        "bulk_substitute on {} with {} substitutions",
        orig_fn.name,
        substitutions.len()
    );
    let mut new_builder = GateBuilder::new(orig_fn.name.clone(), options);
    let mut orig_to_new_map: HashMap<AigRef, AigOperand> = HashMap::new();

    // Pre-process inputs
    for orig_input in &orig_fn.inputs {
        let new_input_vec =
            new_builder.add_input(orig_input.name.clone(), orig_input.get_bit_count());
        for i in 0..orig_input.get_bit_count() {
            let orig_input_op = orig_input.bit_vector.get_lsb(i);
            let new_input_op = new_input_vec.get_lsb(i);
            debug_assert!(
                new_builder.is_valid_ref(orig_input_op.node),
                "Input orig_input_op out of bounds: {:?}",
                orig_input_op
            );
            debug_assert!(
                new_builder.is_valid_ref(new_input_op.node),
                "Input new_input_op out of bounds: {:?}",
                new_input_op
            );
            orig_to_new_map.insert(orig_input_op.node, *new_input_op);
        }
    }

    // Get post-order refs directly from GateFn method
    let mut postorder_refs = orig_fn.post_order_refs();
    // Reverse the order because the loop below uses pop(), expecting reverse
    // topological order
    postorder_refs.reverse();

    let mut processing = HashSet::new(); // For cycle detection

    // Initialize worklist directly from the reversed post-order refs
    let mut worklist = postorder_refs;

    while let Some(orig_ref) = worklist.pop() {
        if orig_to_new_map.contains_key(&orig_ref) {
            continue;
        }
        if processing.contains(&orig_ref) {
            panic!("Cycle detected in substitution map at {:?}", orig_ref);
        }
        processing.insert(orig_ref);
        // Substitution
        if let Some(subst_op) = substitutions.get(&orig_ref) {
            if !orig_to_new_map.contains_key(&subst_op.node) {
                // Dependency not processed yet, process it first
                worklist.push(orig_ref); // Re-process current node after dependency
                worklist.push(subst_op.node);
                processing.remove(&orig_ref);
                continue;
            }
            // If the substitute is a constant, use get_true/get_false
            let subst_node = &orig_fn.gates[subst_op.node.id];
            let final_op = if let AigNode::Literal(value) = subst_node {
                let base_const_op = if *value {
                    new_builder.get_true()
                } else {
                    new_builder.get_false()
                };
                if subst_op.negated {
                    base_const_op.negate()
                } else {
                    base_const_op
                }
            } else {
                let mapped = orig_to_new_map.get(&subst_op.node).copied();
                if let Some(mapped_op) = mapped {
                    if subst_op.negated {
                        new_builder.add_not(mapped_op)
                    } else {
                        mapped_op
                    }
                } else {
                    unreachable!("Substitute node should have been mapped by worklist logic");
                }
            };
            orig_to_new_map.insert(orig_ref, final_op);
            processing.remove(&orig_ref);
            continue;
        }
        // Otherwise, build the node in the new graph
        debug_assert!(
            orig_ref.id < orig_fn.gates.len(),
            "AigRef out of bounds: {:?} (gates.len() = {})",
            orig_ref,
            orig_fn.gates.len()
        );
        let orig_node = &orig_fn.gates[orig_ref.id];
        let new_op = match orig_node {
            AigNode::Input { .. } => {
                // Inputs already handled
                processing.remove(&orig_ref);
                continue;
            }
            AigNode::Literal(value) => {
                if *value {
                    new_builder.get_true()
                } else {
                    new_builder.get_false()
                }
            }
            AigNode::And2 { a, b, .. } => {
                if !orig_to_new_map.contains_key(&a.node) {
                    worklist.push(orig_ref);
                    worklist.push(a.node);
                    processing.remove(&orig_ref);
                    continue;
                }
                if !orig_to_new_map.contains_key(&b.node) {
                    worklist.push(orig_ref);
                    worklist.push(b.node);
                    processing.remove(&orig_ref);
                    continue;
                }
                let new_a = orig_to_new_map[&a.node];
                let new_a = if a.negated {
                    new_builder.add_not(new_a)
                } else {
                    new_a
                };
                let new_b = orig_to_new_map[&b.node];
                let new_b = if b.negated {
                    new_builder.add_not(new_b)
                } else {
                    new_b
                };
                // Prevent self-loop: do not allow a node to depend on itself
                let next_id = new_builder.gates.len();
                debug_assert!(
                    new_a.node.id != next_id,
                    "bulk_substitute would create self-loop in 'a' operand for node {}",
                    next_id
                );
                debug_assert!(
                    new_b.node.id != next_id,
                    "bulk_substitute would create self-loop in 'b' operand for node {}",
                    next_id
                );
                new_builder.add_and_binary(new_a, new_b)
            }
        };
        orig_to_new_map.insert(orig_ref, new_op);
        processing.remove(&orig_ref);
    }

    // Build outputs
    for orig_output in &orig_fn.outputs {
        let mut new_output_bits = Vec::new();
        for bit in orig_output.bit_vector.iter_lsb_to_msb() {
            let mapped = orig_to_new_map[&bit.node];
            let mapped = if bit.negated {
                new_builder.add_not(mapped)
            } else {
                mapped
            };
            new_output_bits.push(mapped);
        }
        let new_output_vec = AigBitVector::from_lsb_is_index_0(&new_output_bits);
        new_builder.add_output(orig_output.name.clone(), new_output_vec);
    }

    let replaced_fn = new_builder.build();

    replaced_fn.check_invariants_with_debug_assert();
    crate::topo::debug_assert_no_cycles(&replaced_fn.gates, "bulk_substitute");
    (replaced_fn, orig_to_new_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check_equivalence,
        gate_builder::GateBuilderOptions,
        get_summary_stats::{get_summary_stats, SummaryStats},
        test_utils::{
            setup_graph_for_constant_replace, setup_graph_with_more_redundancies,
            setup_graph_with_redundancies, setup_invalid_graph_with_cycle,
        },
    };

    #[test]
    fn test_replace_redundant_node_equivalence() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_data = setup_graph_with_redundancies(); // Use the redundancy graph
        let original_fn = &test_data.g;

        // Substitution: Replace redundant node 'inner1' (%5) with 'inner0' (%4)
        let node_to_replace = test_data.inner1.node;
        let substitute_node = test_data.inner0.node;
        let mut substitutions = SubstitutionMap::new();
        substitutions.add(node_to_replace, AigOperand::from(substitute_node));

        log::info!(
            "Original function with redundancies:\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt(); // Keep no_opt() as requested
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options);

        log::info!("Replaced function:\n{}", replaced_fn.to_string());

        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn)
            .expect("original and replaced functions should be equivalent");

        let stats: SummaryStats = get_summary_stats(&replaced_fn);
        let original_stats: SummaryStats = get_summary_stats(original_fn);
        assert_eq!(original_stats.live_nodes, 7);
        assert_eq!(stats.live_nodes, 6);
    }

    #[test]
    fn test_replace_multiple_redundant_nodes() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_data = setup_graph_with_more_redundancies();
        let original_fn = &test_data.g;

        let mut substitutions = SubstitutionMap::new();
        substitutions.add(
            test_data.inner1.node,
            AigOperand::from(test_data.inner0.node),
        );
        substitutions.add(
            test_data.inner2.node,
            AigOperand::from(test_data.inner0.node),
        );

        log::info!(
            "Original function with more redundancies:\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt();
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options);

        log::info!(
            "Replaced function (multiple subs):\n{}",
            replaced_fn.to_string()
        );

        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn).expect(
            "original and replaced functions should be equivalent after multiple substitutions",
        );

        let stats: SummaryStats = get_summary_stats(&replaced_fn);
        let original_stats: SummaryStats = get_summary_stats(original_fn);
        assert_eq!(original_stats.live_nodes, 9);
        assert_eq!(stats.live_nodes, 7);
    }

    #[test]
    fn test_replace_node_with_constant() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_data = setup_graph_for_constant_replace();
        let original_fn = &test_data.g;

        let node_to_replace = test_data.and_true_true.node;
        let substitute_operand = test_data.const_true;

        let mut substitutions = SubstitutionMap::new();
        substitutions.add(node_to_replace, substitute_operand);

        log::info!(
            "Original function (const replace test):\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt();
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options);

        log::info!(
            "Replaced function (const replace test):\n{}",
            replaced_fn.to_string()
        );

        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn)
            .expect("Replacing AND(true, true) with true should preserve equivalence");

        assert_eq!(replaced_fn.outputs.len(), 1);
        let final_out_op = replaced_fn.outputs[0].bit_vector.get_lsb(0);

        let final_out_node = &replaced_fn.gates[final_out_op.node.id];
        let is_const = matches!(final_out_node, AigNode::Literal(_));
        assert!(is_const, "Output should be a constant after replacement");

        if let AigNode::Literal(literal_value) = final_out_node {
            let effective_value = *literal_value ^ final_out_op.negated;
            assert!(effective_value, "Output constant should be True");
        } else {
            panic!("Output node was not a Literal as expected");
        }
    }

    #[test]
    fn test_cycle_in_substitution_map_panics() {
        let test_graph = setup_invalid_graph_with_cycle();
        let result = std::panic::catch_unwind(|| {
            let mut substitutions = SubstitutionMap::new();
            substitutions.add(test_graph.a.node, test_graph.b);
            substitutions.add(test_graph.b.node, test_graph.a);
        });
        assert!(
            result.is_err(),
            "SubstitutionMap::add should panic on cycle in substitution map"
        );
    }
}
