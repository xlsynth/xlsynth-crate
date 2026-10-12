// SPDX-License-Identifier: Apache-2.0

//! Thin fuzz-input adapters around the library's single direct-G8R generator.

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::g8r::{decode_g8r_binary, emit_g8r, encode_g8r_binary, parse_g8r};
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gate_random::{RandomGateOptions, generate_gate_design};
use xlsynth_pir::ir_random::DepletableBytes;
use xlsynth_pir::random_inputs::generate_mixed_irbits_with_rng;

/// A combinational sample and its reconstruction policy.
pub struct CombinationalSample {
    pub graph: GateFn,
    pub builder_options: GateBuilderOptions,
}

/// Generates a combinational profile; invalid options are errors, not skips.
pub fn generate_combinational(
    data: &[u8],
    options: &RandomGateOptions,
) -> Result<CombinationalSample, String> {
    if *options.registers.end() != 0 {
        return Err("combinational fuzz profiles must disable registers".into());
    }
    let generated = generate_gate_design(&mut DepletableBytes::new(data), options)
        .map_err(|e| e.to_string())?;
    Ok(CombinationalSample {
        graph: generated.design.try_into_gate_fn()?,
        builder_options: generated.builder_options,
    })
}

/// Shape coverage for the direct sequential smoke campaign.
#[derive(Debug, Default)]
pub struct DirectSequentialCoverage {
    pub registers: usize,
    pub enables: usize,
    pub resets: usize,
    pub active_high_controls: usize,
    pub active_low_controls: usize,
}

/// Checks native round trips and legalization for directly generated stateful
/// AIGs.
pub fn check_sequential(data: &[u8]) -> DirectSequentialCoverage {
    let options = RandomGateOptions {
        input_width: 0..=8,
        output_width: 0..=8,
        registers: 0..=6,
        register_width: 0..=16,
        ..RandomGateOptions::default()
    };
    let source = generate_gate_design(&mut DepletableBytes::new(data), &options)
        .expect("valid bounded sequential options")
        .design;
    let text = emit_g8r(&source).unwrap();
    let decoded_text = parse_g8r(&text).unwrap();
    let decoded_binary = decode_g8r_binary(&encode_g8r_binary(&source).unwrap()).unwrap();
    for decoded in [&decoded_text, &decoded_binary] {
        assert_eq!(decoded.registers, source.registers);
        assert_eq!(decoded.inputs, source.inputs);
        assert_eq!(decoded.outputs, source.outputs);
        assert_eq!(decoded.clock, source.clock);
        assert_eq!(emit_g8r(decoded).unwrap(), text);
    }
    let legalized = source.legalize_register_controls().unwrap();
    legalized.validate_legalized().unwrap();
    assert_eq!(
        emit_g8r(&legalized.legalize_register_controls().unwrap()).unwrap(),
        emit_g8r(&legalized).unwrap()
    );
    let input_widths = source
        .inputs
        .iter()
        .map(|id| source.transition.inputs[id.index()].get_bit_count())
        .collect::<Vec<_>>();
    let register_widths = source
        .registers
        .iter()
        .map(|r| source.transition.inputs[r.q.index()].get_bit_count())
        .collect::<Vec<_>>();
    let variants = [&source, &decoded_text, &decoded_binary, &legalized];
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    for _ in 0..4 {
        let values = generate_mixed_irbits_with_rng(&mut rng, &register_widths);
        let mut states = variants
            .iter()
            .map(|design| SequentialState::from_register_values(design, values.clone()).unwrap())
            .collect::<Vec<_>>();
        for cycle in 0..12 {
            let stimulus = vec![generate_mixed_irbits_with_rng(&mut rng, &input_widths)];
            let traces = variants
                .iter()
                .zip(&states)
                .map(|(design, state)| simulate(design, &stimulus, state.clone()).unwrap())
                .collect::<Vec<_>>();
            for trace in traces.iter().skip(1) {
                assert_eq!(
                    trace.external_outputs(),
                    traces[0].external_outputs(),
                    "cycle {cycle}\n{text}"
                );
                assert_eq!(
                    trace.final_state(),
                    traces[0].final_state(),
                    "cycle {cycle}\n{text}"
                );
            }
            states = traces
                .iter()
                .map(|trace| trace.final_state().clone())
                .collect();
        }
    }
    let mut coverage = DirectSequentialCoverage {
        registers: source.registers.len(),
        ..DirectSequentialCoverage::default()
    };
    for register in &source.registers {
        coverage.enables += usize::from(register.enable.is_some());
        coverage.resets += usize::from(register.reset.is_some());
        for control in register
            .enable
            .iter()
            .chain(register.reset.iter().map(|r| &r.control))
        {
            coverage.active_low_controls += usize::from(control.active_low);
            coverage.active_high_controls += usize::from(!control.active_low);
        }
    }
    coverage
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn direct_sequential_smoke_covers_state_and_both_control_polarities() {
        let mut rng = StdRng::seed_from_u64(0x91a7);
        let mut total = DirectSequentialCoverage::default();
        for length in 0..128 {
            let mut bytes = vec![0; length * 8];
            rng.fill_bytes(&mut bytes);
            let seen = check_sequential(&bytes);
            total.registers += seen.registers;
            total.enables += seen.enables;
            total.resets += seen.resets;
            total.active_high_controls += seen.active_high_controls;
            total.active_low_controls += seen.active_low_controls;
        }
        assert!(
            total.registers > 0 && total.enables > 0 && total.resets > 0,
            "{total:?}"
        );
        assert!(
            total.active_high_controls > 0 && total.active_low_controls > 0,
            "{total:?}"
        );
    }
}
