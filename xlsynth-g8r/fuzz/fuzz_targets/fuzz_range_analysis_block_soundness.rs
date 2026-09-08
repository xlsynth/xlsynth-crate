// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r_fuzz::random_block::evaluate_block_cycle_observed;
use xlsynth_pir::ir_random::{
    BlockTopology, DepletableBytes, EntropySource, RandomBlockOptions, RandomFnOptions, StopPolicy,
    generate_block_package,
};
use xlsynth_pir::known_bits;
use xlsynth_pir::random_inputs::generate_mixed_values_with_rng;
use xlsynth_pir::range_analysis::analyze_block;

fuzz_target!(|data: &[u8]| {
    let mut entropy = DepletableBytes::new(data);
    // Input/state mutations consume a separate seed header, leaving graph
    // construction driven directly by the remaining generic fuzz bytes.
    let input_seed = entropy.take_u64();
    let generated = generate_block_package(
        &mut entropy,
        &RandomBlockOptions {
            topology: BlockTopology::GeneralSequential,
            max_registers: 4,
            allow_zero_width_ports_and_registers: true,
            function_options: RandomFnOptions {
                max_nodes: 48,
                max_bit_width: 257,
                max_div_mod_bit_width: Some(8),
                max_multiply_operand_bit_width: Some(16),
                max_type_depth: 2,
                max_aggregate_leaves: 8,
                max_array_length: 4,
                max_tuple_length: 4,
                allow_zero_width_bits: true,
                allow_arbitrary_width_multiply: true,
                allow_gate: true,
                allow_extension_ops: true,
                // Deliberately failing events would prevent observing every
                // data node; register writes and output sinks remain enabled.
                allow_events: false,
                ..RandomFnOptions::default()
            },
            ..RandomBlockOptions::default()
        },
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed range soundness options must generate a valid block");
    let block = generated.package.get_top_block().unwrap();
    let facts = analyze_block(block)
        .unwrap_or_else(|error| panic!("generated block must analyze: {error}\n{block}"));
    let known_bits = known_bits::analyze_block(block).unwrap_or_else(|error| {
        panic!("generated block must have known-bit facts: {error}\n{block}")
    });
    let mut rng = StdRng::seed_from_u64(input_seed);
    for trial in 0..8 {
        // Q values are arbitrary each trial, not reset values or only states
        // reachable from reset. Reset/enable input ports are sampled normally.
        let mut state = generate_mixed_values_with_rng(
            &mut rng,
            block.registers.iter().map(|register| &register.ty),
        );
        for cycle in 0..2 {
            let inputs = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|port| block.port_type(port)),
            );
            let observed = evaluate_block_cycle_observed(block, &inputs, &state);
            for (node, range) in facts.iter() {
                let value = observed.node_values[node.index].as_ref().expect(
                    "every analyzed node, including sources and dead nodes, must be observed",
                );
                assert!(
                    range.contains(value),
                    "trial={trial} cycle={cycle} node id={}: {range:?} excludes {value}; inputs={inputs:?}; state={state:?}\n{block}",
                    block.get_node(node).text_id,
                );
                let known = known_bits
                    .get(node)
                    .expect("both analyses cover every data node");
                assert!(
                    known.contains(value),
                    "trial={trial} cycle={cycle} node id={}: {known:?} excludes {value}; inputs={inputs:?}; state={state:?}\n{block}",
                    block.get_node(node).text_id,
                );
            }
            // Also check the next state from real reset/enable/write semantics,
            // with independently sampled ports on the next cycle.
            state = observed.next_state;
        }
    }
});
