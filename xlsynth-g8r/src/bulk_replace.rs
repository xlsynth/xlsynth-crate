// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

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
    substitutions: &HashMap<AigRef, AigOperand>,
    options: GateBuilderOptions,
    verify: bool,
) -> (GateFn, HashMap<AigRef, AigOperand>) {
    #[cfg(debug_assertions)]
    {
        use std::collections::HashSet;
        // Precondition check 1: Ensure no substitution chains (A->B, B->C)
        for substitute_op in substitutions.values() {
            debug_assert!(
                !substitutions.contains_key(&substitute_op.node),
                "Substitution chain detected: {:?} is a target and also a key in the substitution map. Chains are not supported.",
                substitute_op.node
            );
        }

        // Precondition check 2: Ensure no input nodes are being substituted
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
        }
    }

    let mut new_builder = GateBuilder::new(orig_fn.name.clone(), options);
    let mut orig_to_new_map: HashMap<AigRef, AigOperand> = HashMap::new();

    // Pre-process inputs
    for orig_input in &orig_fn.inputs {
        let new_input_vec =
            new_builder.add_input(orig_input.name.clone(), orig_input.get_bit_count());
        for i in 0..orig_input.get_bit_count() {
            // Inputs in orig_fn are always non-negated operands
            let orig_input_op = orig_input.bit_vector.get_lsb(i);
            let new_input_op = new_input_vec.get_lsb(i);
            orig_to_new_map.insert(orig_input_op.node, *new_input_op);
        }
    }

    // Process outputs (triggers recursive processing of dependencies)
    for orig_output in &orig_fn.outputs {
        let mut new_output_bits: Vec<AigOperand> = Vec::new();
        for orig_bit_operand in orig_output.bit_vector.iter_lsb_to_msb() {
            let final_new_op = process_operand(
                *orig_bit_operand,
                orig_fn,
                substitutions,
                &mut new_builder,
                &mut orig_to_new_map,
            );
            new_output_bits.push(final_new_op);
        }
        let new_output_vec = AigBitVector::from_lsb_is_index_0(&new_output_bits);
        new_builder.add_output(orig_output.name.clone(), new_output_vec);
    }

    let replaced_fn = new_builder.build();

    if verify {
        verify_io_signature(orig_fn, &replaced_fn);
    }

    // Apply migrated tags using the final mapping
    let mut final_fn = replaced_fn; // Get mutable ownership
    for (orig_id, orig_node) in orig_fn.gates.iter().enumerate() {
        // Get tags from original node, if any
        if let Some(tags_to_migrate) = orig_node.get_tags() {
            if tags_to_migrate.is_empty() {
                continue; // No tags to actually migrate
            }

            let orig_ref = AigRef { id: orig_id };
            // Find the corresponding node in the new graph
            if let Some(final_new_op) = orig_to_new_map.get(&orig_ref) {
                let target_node_ref = final_new_op.node;
                assert!(
                    target_node_ref.id < final_fn.gates.len(),
                    "Mapped target node {:?} for original {:?} does not exist in final graph (len={})",
                    target_node_ref, orig_ref, final_fn.gates.len()
                );
                let target_gate = &mut final_fn.gates[target_node_ref.id];
                // Attempt to add the tags
                let _ = target_gate.try_add_tags(tags_to_migrate);
            }
        }
    }

    // Return the final function and the mapping
    (final_fn, orig_to_new_map)
}

/// Verifies that the input/output signature (name, bit count) is preserved
/// between the original and replaced GateFn.
fn verify_io_signature(orig_fn: &GateFn, replaced_fn: &GateFn) {
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

/// Recursively processes an `AigOperand` from the original graph, returning the
/// corresponding `AigOperand` in the new graph. Handles negation.
fn process_operand(
    orig_operand: AigOperand,
    orig_fn: &GateFn,
    substitutions: &HashMap<AigRef, AigOperand>,
    new_builder: &mut GateBuilder,
    orig_to_new_map: &mut HashMap<AigRef, AigOperand>,
) -> AigOperand {
    // Process the underlying node reference
    let new_base_operand = process_node(
        orig_operand.node,
        orig_fn,
        substitutions,
        new_builder,
        orig_to_new_map,
    );

    // Apply negation if necessary
    if orig_operand.negated {
        new_builder.add_not(new_base_operand)
    } else {
        new_base_operand
    }
}

/// Recursively processes an `AigRef` (node) from the original graph, returning
/// the corresponding non-negated `AigOperand` for the node in the new graph.
/// Handles substitutions and tag migration. Uses memoization
/// (`orig_to_new_map`).
fn process_node(
    orig_ref: AigRef,
    orig_fn: &GateFn,
    substitutions: &HashMap<AigRef, AigOperand>,
    new_builder: &mut GateBuilder,
    orig_to_new_map: &mut HashMap<AigRef, AigOperand>,
) -> AigOperand {
    // Check for substitution before checking memoization
    if let Some(substitute_op) = substitutions.get(&orig_ref) {
        // Check if the substitute is a constant
        let substitute_node_type = &orig_fn.gates[substitute_op.node.id];
        let final_op = if let AigNode::Literal(value) = substitute_node_type {
            // Handle constant substitution directly
            let base_const_op = if *value {
                new_builder.get_true()
            } else {
                new_builder.get_false()
            };
            // Apply negation based on the substitution operand
            if substitute_op.negated {
                base_const_op.negate()
            } else {
                base_const_op
            }
        } else {
            // Substitute is not a constant, process it recursively
            let final_substitute_base_op = process_node(
                substitute_op.node,
                orig_fn,
                substitutions,
                new_builder,
                orig_to_new_map,
            );
            if substitute_op.negated {
                new_builder.add_not(final_substitute_base_op)
            } else {
                final_substitute_base_op
            }
        };

        orig_to_new_map.insert(orig_ref, final_op);
        return final_op;
    }
    // Memoization check (for nodes already processed)
    if let Some(existing_new_op) = orig_to_new_map.get(&orig_ref) {
        return *existing_new_op;
    }

    // Node is not substituted and not memoized, process it normally based on its
    // type. Inputs should have been caught by the memoization check above.
    let orig_node = &orig_fn.gates[orig_ref.id];

    let new_operand = match orig_node {
        // We should not reach Literal/Input here if memoization check is correct
        AigNode::Literal(_) => panic!(
            "Literal node {:?} reached processing logic unexpectedly",
            orig_ref
        ),
        AigNode::Input { .. } => panic!(
            "Input node {:?} reached processing logic unexpectedly.",
            orig_ref
        ),

        AigNode::And2 { a, b, .. } => {
            // Tags handled post-build
            // Recursively process operands
            let new_a = process_operand(*a, orig_fn, substitutions, new_builder, orig_to_new_map);
            let new_b = process_operand(*b, orig_fn, substitutions, new_builder, orig_to_new_map);
            // Create the new `and` gate
            new_builder.add_and_binary(new_a, new_b)
        }
    };

    // Memoize the result for the original node before returning
    orig_to_new_map.insert(orig_ref, new_operand);
    new_operand
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
            setup_graph_with_redundancies,
        },
    };
    use std::collections::HashMap;

    #[test]
    fn test_replace_redundant_node_equivalence() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_data = setup_graph_with_redundancies(); // Use the redundancy graph
        let original_fn = &test_data.g;

        // Substitution: Replace redundant node 'inner1' (%5) with 'inner0' (%4)
        let node_to_replace = test_data.inner1.node;
        let substitute_node = test_data.inner0.node;
        let mut substitutions: HashMap<AigRef, AigOperand> = HashMap::new();
        substitutions.insert(node_to_replace, AigOperand::from(substitute_node));

        log::info!(
            "Original function with redundancies:\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt(); // Keep no_opt() as requested
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options, true);

        log::info!("Replaced function:\n{}", replaced_fn.to_string());

        // Equivalence Check should still pass even without optimization
        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn)
            .expect("original and replaced functions should be equivalent");

        // Check node count reduction
        let stats: SummaryStats = get_summary_stats(&replaced_fn);
        let original_stats: SummaryStats = get_summary_stats(original_fn);
        assert_eq!(original_stats.live_nodes, 7);
        assert_eq!(stats.live_nodes, 6); // Adjusted expected count for no_opt()

        // Remove check for output node ID equality, as it won't hold without
        // optimization
    }

    #[test]
    fn test_replace_multiple_redundant_nodes() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Use the setup with three redundant paths
        let test_data = setup_graph_with_more_redundancies();
        let original_fn = &test_data.g;

        // Substitution: Replace both redundant inner nodes (inner1, inner2) with inner0
        let mut substitutions: HashMap<AigRef, AigOperand> = HashMap::new();
        substitutions.insert(
            test_data.inner1.node,
            AigOperand::from(test_data.inner0.node),
        );
        substitutions.insert(
            test_data.inner2.node,
            AigOperand::from(test_data.inner0.node),
        );

        log::info!(
            "Original function with more redundancies:\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt(); // Keep no_opt() as requested
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options, true);

        log::info!(
            "Replaced function (multiple subs):\n{}",
            replaced_fn.to_string()
        );

        // Equivalence Check should still pass
        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn).expect(
            "original and replaced functions should be equivalent after multiple substitutions",
        );

        // Check node count reduction
        let stats: SummaryStats = get_summary_stats(&replaced_fn);
        let original_stats: SummaryStats = get_summary_stats(original_fn);
        assert_eq!(original_stats.live_nodes, 9);
        assert_eq!(stats.live_nodes, 7); // Adjusted expected count for no_opt()

        // Remove checks for output node ID equality
    }

    #[test]
    fn test_replace_node_with_constant() {
        let _ = env_logger::builder().is_test(true).try_init();
        let test_data = setup_graph_for_constant_replace();
        let original_fn = &test_data.g;

        // Replace the And2(true, true) node with the constant true
        let node_to_replace = test_data.and_true_true.node;
        let substitute_operand = test_data.const_true;

        let mut substitutions: HashMap<AigRef, AigOperand> = HashMap::new();
        substitutions.insert(node_to_replace, substitute_operand);

        log::info!(
            "Original function (const replace test):\n{}",
            original_fn.to_string()
        );

        let options = GateBuilderOptions::no_opt();
        let (replaced_fn, _) = bulk_replace(original_fn, &substitutions, options, true);

        log::info!(
            "Replaced function (const replace test):\n{}",
            replaced_fn.to_string()
        );

        check_equivalence::validate_same_gate_fn(original_fn, &replaced_fn)
            .expect("Replacing AND(true, true) with true should preserve equivalence");

        assert_eq!(replaced_fn.outputs.len(), 1);
        let final_out_op = replaced_fn.outputs[0].bit_vector.get_lsb(0);

        // Check if the final output operand points to a Literal node
        let final_out_node = &replaced_fn.gates[final_out_op.node.id];
        let is_const = matches!(final_out_node, AigNode::Literal(_));
        assert!(is_const, "Output should be a constant after replacement");

        // Get the literal value and apply operand negation
        if let AigNode::Literal(literal_value) = final_out_node {
            let effective_value = *literal_value ^ final_out_op.negated;
            assert!(effective_value, "Output constant should be True");
        } else {
            // Should not happen due to the is_const assertion above
            panic!("Output node was not a Literal as expected");
        }
    }
}
