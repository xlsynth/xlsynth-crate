// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially evaluates generated GateFns through g8r and Lean.

use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use xlsynth_g8r::aig::{AigBitVector, AigNode, AigOperand, GateFn, SequentialGateFn};
use xlsynth_g8r::aig_serdes::g8r::emit_g8r;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gate_random::{RandomGateOptions, generate_gate_design};
use xlsynth_ir_value::IrBits;
use xlsynth_lean::LeanEvaluator;
use xlsynth_pir::ir_random::DepletableBytes;
use xlsynth_pir::random_inputs::generate_flat_bitvector_argument_sets_with_rng;

const INPUT_SAMPLE_COUNT: usize = 32;
static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();

/// Bounds generic graph generation while permitting empty and wide ports.
fn graph_options() -> RandomGateOptions {
    RandomGateOptions {
        input_ports: 0..=4,
        input_width: 0..=80,
        output_ports: 0..=4,
        output_width: 0..=80,
        registers: 0..=0,
        max_operations: 32,
        max_aig_nodes: 256,
        ..RandomGateOptions::default()
    }
}

/// Remaps one edge while preserving its polarity.
fn remap_operand(mut operand: AigOperand, old_to_new: &[usize]) -> AigOperand {
    operand.node.id = old_to_new[operand.node.id];
    operand
}

/// Remaps every bit of an interface vector without changing bit order.
fn remap_bit_vector(vector: &AigBitVector, old_to_new: &[usize]) -> AigBitVector {
    let operands = vector
        .iter_lsb_to_msb()
        .map(|operand| remap_operand(*operand, old_to_new))
        .collect::<Vec<_>>();
    AigBitVector::from_lsb_is_index_0(&operands)
}

/// Randomly permutes storage slots while preserving the graph and slot zero.
///
/// Slot zero remains fixed to respect g8r's canonical false-node convention.
/// The other nodes may move before their dependencies, exercising export of
/// valid DAGs whose stored node IDs are not in topological order.
fn permute_nodes(graph: &mut GateFn, rng: &mut StdRng) {
    let mut new_to_old = (0..graph.gates.len()).collect::<Vec<_>>();
    if new_to_old.len() > 1 {
        new_to_old[1..].shuffle(rng);
    }
    let mut old_to_new = vec![0; graph.gates.len()];
    for (new, old) in new_to_old.iter().copied().enumerate() {
        old_to_new[old] = new;
    }

    let mut gates = new_to_old
        .into_iter()
        .map(|old| graph.gates[old].clone())
        .collect::<Vec<_>>();
    for gate in &mut gates {
        if let AigNode::And2 { a, b, .. } = gate {
            *a = remap_operand(*a, &old_to_new);
            *b = remap_operand(*b, &old_to_new);
        }
    }
    for input in &mut graph.inputs {
        input.bit_vector = remap_bit_vector(&input.bit_vector, &old_to_new);
    }
    for output in &mut graph.outputs {
        output.bit_vector = remap_bit_vector(&output.bit_vector, &old_to_new);
    }
    graph.gates = gates;
}

/// Compares one graph's output batches and retains enough context to replay it.
fn check_case(data: &[u8]) {
    let seed = *blake3::hash(data).as_bytes();
    let generated = generate_gate_design(&mut DepletableBytes::new(data), &graph_options())
        .unwrap_or_else(|error| {
            panic!(
                "bounded GateFn generation failed: {error}\nseed={seed:02x?}\nfuzz_bytes={data:02x?}"
            )
        });
    let mut graph = generated.design.try_into_gate_fn().unwrap_or_else(|error| {
        panic!(
            "combinational generator emitted an invalid design: {error}\nseed={seed:02x?}\nfuzz_bytes={data:02x?}"
        )
    });
    let mut rng = StdRng::from_seed(seed);
    permute_nodes(&mut graph, &mut rng);

    let native_graph = emit_g8r(&SequentialGateFn::from_gate_fn(graph.clone())).unwrap_or_else(
        |error| {
            panic!(
                "generated GateFn could not be serialized: {error}\nseed={seed:02x?}\nfuzz_bytes={data:02x?}\ngraph={graph:#?}"
            )
        },
    );
    let input_widths = graph
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .collect::<Vec<_>>();
    let input_cases =
        generate_flat_bitvector_argument_sets_with_rng(&mut rng, &input_widths, INPUT_SAMPLE_COUNT);
    let mut simulator = PreparedGateSim::new(&graph);
    let expected: Vec<Vec<IrBits>> = input_cases
        .iter()
        .map(|inputs| simulator.eval_outputs(inputs))
        .collect();
    let evaluator = EVALUATOR
        .get()
        .expect("fuzzer startup must initialize the Lean evaluator");
    let actual = evaluator
        .eval_gate_fn_batch(&graph, &input_cases)
        .unwrap_or_else(|error| {
            panic!(
                "Lean evaluation failed: {error}\nseed={seed:02x?}\nfuzz_bytes={data:02x?}\ng8r:\n{native_graph}\ninputs={input_cases:?}\ng8r_outputs={expected:?}"
            )
        });
    assert_eq!(
        actual, expected,
        "Lean and g8r output batches differ\nseed={seed:02x?}\nfuzz_bytes={data:02x?}\ng8r:\n{native_graph}\ninputs={input_cases:?}\ng8r_outputs={expected:?}\nlean_outputs={actual:?}"
    );
}

fuzz_target!(init: {
    let Some(lean_executable) = std::env::var_os("XLSYNTH_LEAN") else {
        eprintln!("fuzz setup failed: set XLSYNTH_LEAN to the actual Lean executable path");
        std::process::exit(2);
    };
    let evaluator = LeanEvaluator::new(PathBuf::from(lean_executable)).unwrap_or_else(|error| {
        eprintln!("fuzz setup failed: could not initialize Lean evaluator: {error}");
        std::process::exit(2);
    });
    assert!(EVALUATOR.set(evaluator).is_ok(), "Lean evaluator initialized twice");
}, |data: &[u8]| {
    check_case(data);
});
