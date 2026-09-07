// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, Block, NodeRef};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_random::{
    DepletableBytes, EntropySource, RandomFnOptions, StopPolicy, generate_fn,
};
use xlsynth_pir::known_bits::{KnownBitsAnalysis, analyze_block, analyze_fn};
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;

/// Checks every computed node, including aggregate leaves and dead nodes.
struct SoundnessObserver<'a, 'ir> {
    facts: &'a KnownBitsAnalysis<'ir>,
    function: &'ir ir::Fn,
    arguments: &'a [IrValue],
    trial: usize,
    seen: Vec<bool>,
}

impl EvalObserver for SoundnessObserver<'_, '_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // Node-value callbacks already include the concrete selected result.
    }

    fn on_node_value(&mut self, node: NodeRef, text_id: usize, value: &IrValue) {
        let facts = self
            .facts
            .get(node)
            .expect("evaluated nodes must have facts");
        assert!(
            facts.contains(value),
            "trial={} node id={text_id}: {facts:?} excludes {value}; arguments={:?}\n{}",
            self.trial,
            self.arguments,
            self.function,
        );
        self.seen[node.index] = true;
    }
}

/// Tests node-level claims under independently mutable concrete-input entropy.
fn check_function(function: &ir::Fn, input_seed: u64) {
    let facts = analyze_fn(function)
        .unwrap_or_else(|error| panic!("generated graph must analyze: {error}\n{function}"));
    let block = Block::from_function(function.clone(), None)
        .unwrap_or_else(|error| panic!("function must convert to a block: {error}\n{function}"));
    let block_facts = analyze_block(&block)
        .unwrap_or_else(|error| panic!("converted block must analyze: {error}\n{function}"));
    for (node, value) in facts.iter() {
        assert_eq!(
            block_facts.get(node),
            Some(value),
            "function/block facts differ at node id={}\n{function}",
            function.get_node(node).text_id,
        );
    }

    let mut rng = StdRng::seed_from_u64(input_seed);
    // Nullary functions still get checked; repeating identical executions adds
    // no coverage. Other functions receive eight independently mixed input
    // vectors from the shared structured/special/uniform sampler.
    let trials = if function.params.is_empty() { 1 } else { 8 };
    let argument_sets = generate_mixed_argument_sets_with_rng(function, &mut rng, trials);
    for (trial, arguments) in argument_sets.into_iter().enumerate() {
        let mut observer = SoundnessObserver {
            facts: &facts,
            function,
            arguments: &arguments,
            trial,
            seen: vec![false; function.nodes.len()],
        };
        // The interpreter does not issue value callbacks for parameters.
        for (&param, value) in function.params.iter().zip(&arguments) {
            observer.on_node_value(param, function.get_node(param).text_id, value);
        }
        match eval_fn_with_observer(function, &arguments, Some(&mut observer)) {
            FnEvalResult::Success(result) => assert!(
                facts
                    .get(function.ret_node_ref.unwrap())
                    .unwrap()
                    .contains(&result.value),
                "return value violates facts: trial={trial} arguments={arguments:?}\n{function}",
            ),
            FnEvalResult::Failure(error) => panic!(
                "pure generated graph must evaluate: {error:?}; trial={trial} arguments={arguments:?}\n{function}"
            ),
        }
        for (node, _) in facts.iter() {
            assert!(
                observer.seen[node.index],
                "missing value callback at node id={}; trial={trial} arguments={arguments:?}\n{function}",
                function.get_node(node).text_id,
            );
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut entropy = DepletableBytes::new(data);
    // A zero-padded seed header lets libFuzzer change concrete inputs without
    // changing graph construction. Remaining bytes directly drive the generic
    // typed graph generator, preserving useful structure under mutations.
    let input_seed = entropy.take_u64();
    let options = RandomFnOptions {
        max_params: 4,
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
        // Effects can intentionally abort execution; this target checks pure
        // dataflow so every analyzed node must have a concrete value.
        allow_events: false,
        ..RandomFnOptions::default()
    };
    let generated = generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed soundness fuzz options must generate a valid function");
    check_function(&generated.function, input_seed);
});
