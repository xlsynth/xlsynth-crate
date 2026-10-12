// SPDX-License-Identifier: Apache-2.0

//! Reset-reachable equivalence checks for sequential synthesis cleanup.

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig::sequential_cleanup::{SequentialCleanupStats, cleanup_sequential_gate_fn};
use xlsynth_g8r::aig_serdes::g8r::emit_g8r;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation, StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::{generate_mixed_irbits_with_rng, generate_mixed_values_with_rng};

use crate::random_block::{block_output_types, evaluate_block_cycle, flatten_value};

const STARTUP_TRIALS: usize = 8;
const CYCLES: usize = 16;

/// Limits generic block generation while requiring a physical synchronous
/// reset.
fn block_options() -> RandomBlockOptions {
    RandomBlockOptions {
        max_input_ports: 6,
        // Reserve an observable endpoint before the body consumes its budget.
        min_output_ports: 1,
        max_output_ports: 4,
        max_registers: 6,
        require_reset_on_all_registers: true,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_params: 7, // Six data inputs plus the reset port.
            max_nodes: 32,
            max_bit_width: 8,
            allow_arbitrary_width_multiply: true,
            enabled_operations: OperationSet::new(OperationSet::all_supported().iter().filter(
                |operation| !matches!(operation, RandomOperation::Umulp | RandomOperation::Smulp),
            )),
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    }
}

/// Samples implementation state independently; reset must establish
/// correspondence.
fn random_state(design: &SequentialGateFn, rng: &mut StdRng) -> SequentialState {
    let widths = design
        .registers
        .iter()
        .map(|r| design.transition.inputs[r.q.index()].get_bit_count())
        .collect::<Vec<_>>();
    SequentialState::from_register_values(design, generate_mixed_irbits_with_rng(rng, &widths))
        .unwrap()
}

/// Checks block IR, its G8R lowering, and cleaned G8R after a reset edge.
/// Pre-reset outputs are not compared against the cleaned implementation.
pub fn check_cleanup(data: &[u8]) -> SequentialCleanupStats {
    // Separate evaluation entropy from structural choices without injecting
    // directed cleanup patterns into the generic block generator.
    let (seed, graph_bytes) = data.split_at(data.len().min(8));
    let mut rng = StdRng::from_seed(*blake3::hash(seed).as_bytes());
    let generated = generate_block_package(
        &mut DepletableBytes::new(graph_bytes),
        &block_options(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed block-generation options must be valid");
    let block = generated.package.get_top_block().unwrap();
    let block_ir = generated.package.to_string();
    let reset = block.reset.as_ref().expect("generator must provide reset");
    assert!(!reset.asynchronous);
    assert!(block.registers.iter().all(|r| r.reset_value.is_some()));
    let original = block_package_to_sequential_gate_fn(
        &generated.package,
        if rng.gen_bool(0.5) {
            GatifyOptions::all_opts_enabled()
        } else {
            GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap_or_else(|error| panic!("lowering failed: {error}\n{block_ir}"));
    let result = cleanup_sequential_gate_fn(&original)
        .unwrap_or_else(|error| panic!("cleanup failed: {error}\n{block_ir}"));
    let cleaned = &result.design;
    cleaned.validate().unwrap();
    assert_eq!(cleaned.name, original.name);
    assert_eq!(cleaned.clock, original.clock);
    let input_shape = |d: &SequentialGateFn| {
        d.inputs
            .iter()
            .map(|id| {
                let port = &d.transition.inputs[id.index()];
                (port.name.clone(), port.get_bit_count())
            })
            .collect::<Vec<_>>()
    };
    let output_shape = |d: &SequentialGateFn| {
        d.outputs
            .iter()
            .map(|id| {
                let port = &d.transition.outputs[id.index()];
                (port.name.clone(), port.get_bit_count())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(input_shape(cleaned), input_shape(&original));
    assert_eq!(output_shape(cleaned), output_shape(&original));
    assert_eq!(
        result.stats.initial_register_bits,
        original
            .registers
            .iter()
            .map(|r| original.transition.inputs[r.q.index()].get_bit_count())
            .sum::<usize>()
    );
    assert_eq!(
        result.stats.final_register_bits,
        cleaned
            .registers
            .iter()
            .map(|r| cleaned.transition.inputs[r.q.index()].get_bit_count())
            .sum::<usize>()
    );
    assert_eq!(
        result.stats.initial_register_bits - result.stats.final_register_bits,
        result.stats.dead_register_bits
            + result.stats.constant_register_bits
            + result.stats.merged_register_bits
    );

    let output_types = block_output_types(block);
    for trial in 0..STARTUP_TRIALS {
        let mut block_state =
            generate_mixed_values_with_rng(&mut rng, block.registers.iter().map(|r| &r.ty));
        let original_values = block_state
            .iter()
            .zip(&block.registers)
            .map(|(value, r)| flatten_value(value, &r.ty))
            .collect();
        let mut original_state =
            SequentialState::from_register_values(&original, original_values).unwrap();
        let mut cleaned_state = random_state(cleaned, &mut rng);
        for cycle in 0..CYCLES {
            let mut inputs = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|p| block.port_type(p)),
            );
            // Always exercise the first reset edge and its release. Later
            // reassertions check that reset remains functional after cleanup.
            let asserted = cycle == 0 || (cycle > 1 && rng.gen_bool(0.1));
            for (index, port) in block.input_ports().enumerate() {
                if port == reset.port {
                    inputs[index] =
                        IrValue::make_ubits(1, u64::from(asserted ^ reset.active_low)).unwrap();
                }
            }
            let gate_inputs = inputs
                .iter()
                .zip(block.input_ports())
                .map(|(value, port)| flatten_value(value, block.port_type(port)))
                .collect::<Vec<_>>();
            let (outputs, next_state) =
                evaluate_block_cycle(block, &inputs, &block_state, &block_ir);
            let expected = outputs
                .iter()
                .zip(&output_types)
                .map(|(value, ty)| flatten_value(value, ty))
                .collect::<Vec<_>>();
            let before = simulate(&original, &[gate_inputs.clone()], original_state).unwrap();
            let after = simulate(cleaned, &[gate_inputs], cleaned_state).unwrap();
            assert_eq!(
                before.external_outputs()[0],
                expected,
                "block lowering mismatch, trial {trial}, cycle {cycle}\n{block_ir}"
            );
            let expected_state = next_state
                .iter()
                .zip(&block.registers)
                .map(|(value, r)| flatten_value(value, &r.ty))
                .collect::<Vec<_>>();
            assert_eq!(before.final_state().values(), expected_state);
            if cycle > 0 {
                assert_eq!(
                    after.external_outputs()[0],
                    expected,
                    "post-reset cleanup mismatch, trial {trial}, cycle {cycle}\n{block_ir}\nCleaned:\n{}",
                    emit_g8r(cleaned).unwrap()
                );
            }
            block_state = next_state;
            original_state = before.final_state().clone();
            cleaned_state = after.final_state().clone();
        }
    }
    let repeated = cleanup_sequential_gate_fn(&original).unwrap();
    assert_eq!(emit_g8r(&repeated.design).unwrap(), emit_g8r(cleaned).unwrap(), "determinism");
    let fixed = cleanup_sequential_gate_fn(cleaned).unwrap();
    assert_eq!(emit_g8r(&fixed.design).unwrap(), emit_g8r(cleaned).unwrap(), "idempotence");
    result.stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn smoke_random_resettable_blocks() {
        let mut rng = StdRng::seed_from_u64(0x51a7e);
        let mut observed = SequentialCleanupStats::default();
        for length in 0..256 {
            let mut bytes = vec![0; length * 8];
            rng.fill_bytes(&mut bytes);
            let stats = check_cleanup(&bytes);
            observed.dead_register_bits += stats.dead_register_bits;
            observed.constant_register_bits += stats.constant_register_bits;
            observed.merged_register_bits += stats.merged_register_bits;
        }
        assert!(
            observed.dead_register_bits > 0,
            "no dead-state removal exercised"
        );
        assert!(
            observed.constant_register_bits > 0,
            "no constant-state removal exercised"
        );
        assert!(
            observed.merged_register_bits > 0,
            "no duplicate-state removal exercised"
        );
    }
}
