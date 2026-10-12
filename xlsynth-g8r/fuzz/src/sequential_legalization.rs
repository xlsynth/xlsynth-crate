// SPDX-License-Identifier: Apache-2.0

//! Exact cycle-zero equivalence for explicit-control register legalization.

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig_serdes::g8r::{decode_g8r_binary, emit_g8r, encode_g8r_binary, parse_g8r};
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation, StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_mixed_values_with_rng;

use crate::random_block::{block_output_types, evaluate_block_cycle, flatten_value};

/// Shape coverage used by the deterministic smoke test, not by generation.
#[derive(Debug, Default)]
pub struct LegalizationCoverage {
    pub register_bits: usize,
    pub enabled: usize,
    pub resettable: usize,
    pub unreset: usize,
}

/// Generates a generic stateful graph and checks all cycles from arbitrary
/// state.
pub fn check_legalization(data: &[u8]) -> LegalizationCoverage {
    let (seed, graph_bytes) = data.split_at(data.len().min(8));
    let mut rng = StdRng::from_seed(*blake3::hash(seed).as_bytes());
    let options = RandomBlockOptions {
        max_input_ports: 6,
        min_output_ports: 1,
        max_output_ports: 4,
        max_registers: 6,
        allow_zero_width_ports_and_registers: true,
        // No reset protocol is assumed. Some registers may have no reset.
        require_reset_on_all_registers: false,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_params: 7,
            max_nodes: 32,
            max_bit_width: 8,
            allow_arbitrary_width_multiply: true,
            enabled_operations: OperationSet::new(OperationSet::all_supported().iter().filter(
                |operation| !matches!(operation, RandomOperation::Umulp | RandomOperation::Smulp),
            )),
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let generated = generate_block_package(
        &mut DepletableBytes::new(graph_bytes),
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed generation options must be valid");
    let block = generated.package.get_top_block().unwrap();
    let block_ir = generated.package.to_string();
    let source = block_package_to_sequential_gate_fn_with_controls(
        &generated.package,
        if rng.gen_bool(0.5) {
            GatifyOptions::all_opts_enabled()
        } else {
            GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap_or_else(|e| panic!("lowering failed: {e}\n{block_ir}"));
    // Exercise serialization of the unlegalized form as well. Both formats
    // must preserve controls and arbitrary-width reset values.
    let original = if rng.gen_bool(0.5) {
        {
            let text = emit_g8r(&source).unwrap();
            parse_g8r(&text)
                .unwrap_or_else(|e| panic!("text roundtrip failed: {e}\n{text}\n{block_ir}"))
        }
    } else {
        decode_g8r_binary(&encode_g8r_binary(&source).unwrap()).unwrap()
    };
    assert_eq!(original.registers, source.registers);
    assert_eq!(original.inputs, source.inputs);
    assert_eq!(original.outputs, source.outputs);
    let legalized = original.legalize_register_controls().unwrap();
    legalized.validate_legalized().unwrap();
    assert_eq!(original.registers.len(), legalized.registers.len());
    assert_eq!(original.clock, legalized.clock);
    assert_eq!(
        emit_g8r(&legalized).unwrap(),
        emit_g8r(&legalized.legalize_register_controls().unwrap()).unwrap(),
        "idempotence"
    );
    let output_types = block_output_types(block);
    for trial in 0..8 {
        let mut block_state =
            generate_mixed_values_with_rng(&mut rng, block.registers.iter().map(|r| &r.ty));
        let values = block_state
            .iter()
            .zip(&block.registers)
            .map(|(value, r)| flatten_value(value, &r.ty))
            .collect::<Vec<_>>();
        let mut original_state =
            SequentialState::from_register_values(&original, values.clone()).unwrap();
        let mut legalized_state =
            SequentialState::from_register_values(&legalized, values).unwrap();
        for cycle in 0..16 {
            // Random reset inputs, with no mandatory initial reset or warm-up.
            let inputs = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|p| block.port_type(p)),
            );
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
            let expected_state = next_state
                .iter()
                .zip(&block.registers)
                .map(|(value, r)| flatten_value(value, &r.ty))
                .collect::<Vec<_>>();
            let before = simulate(&original, &[gate_inputs.clone()], original_state).unwrap();
            let after = simulate(&legalized, &[gate_inputs], legalized_state).unwrap();
            assert_eq!(
                before.external_outputs()[0],
                expected,
                "explicit-control mismatch, trial {trial}, cycle {cycle}\n{block_ir}"
            );
            assert_eq!(
                after.external_outputs(),
                before.external_outputs(),
                "legalization mismatch, trial {trial}, cycle {cycle}\n{block_ir}"
            );
            assert_eq!(
                before.final_state().values(),
                expected_state,
                "explicit-control state mismatch, trial {trial}, cycle {cycle}\n{block_ir}"
            );
            assert_eq!(
                after.final_state(),
                before.final_state(),
                "legalized state mismatch, trial {trial}, cycle {cycle}\n{block_ir}"
            );
            block_state = next_state;
            original_state = before.final_state().clone();
            legalized_state = after.final_state().clone();
        }
    }
    LegalizationCoverage {
        register_bits: original
            .registers
            .iter()
            .map(|r| original.transition.inputs[r.q.index()].get_bit_count())
            .sum(),
        enabled: original
            .registers
            .iter()
            .filter(|r| r.enable.is_some())
            .count(),
        resettable: original
            .registers
            .iter()
            .filter(|r| r.reset.is_some())
            .count(),
        unreset: original
            .registers
            .iter()
            .filter(|r| r.reset.is_none())
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn smoke_random_stateful_legalization() {
        let mut rng = StdRng::seed_from_u64(0x1e9a11);
        let mut coverage = LegalizationCoverage::default();
        for length in 0..256 {
            let mut bytes = vec![0; length * 8];
            rng.fill_bytes(&mut bytes);
            let observed = check_legalization(&bytes);
            coverage.register_bits += observed.register_bits;
            coverage.enabled += observed.enabled;
            coverage.resettable += observed.resettable;
            coverage.unreset += observed.unreset;
        }
        assert!(coverage.register_bits > 0, "{coverage:?}");
        assert!(coverage.enabled > 0, "{coverage:?}");
        assert!(coverage.resettable > 0, "{coverage:?}");
        assert!(coverage.unreset > 0, "{coverage:?}");
    }
}
