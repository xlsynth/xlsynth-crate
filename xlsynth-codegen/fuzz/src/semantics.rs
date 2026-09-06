// SPDX-License-Identifier: Apache-2.0

//! Shared stimuli and independent PIR expectations for external RTL oracles.

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;
use xlsynth_g8r_fuzz::random_block::{
    block_output_types, evaluate_block_cycle_observed, flatten_value,
};
use xlsynth_pir::ir::Package;
use xlsynth_test_helpers::rtl_sim::{Bindings, LogicValue};

use crate::{CYCLE_COUNT, INPUT_SAMPLE_COUNT, top_block};

pub struct Sample {
    pub inputs: Bindings,
    pub outputs: Bindings,
    pub next_state: Option<Bindings>,
}

pub struct Trace {
    pub initial_state: Bindings,
    pub samples: Vec<Sample>,
    pub observed_live_behaviors: BTreeMap<String, u64>,
}

impl Trace {
    /// Computes one deterministic input/state sequence for all external
    /// oracles.
    pub fn for_package(package: &Package) -> Self {
        Self::with_seed(
            package,
            *blake3::hash(package.to_string().as_bytes()).as_bytes(),
        )
    }

    /// Uses stimulus entropy that can be mutated without changing the graph.
    pub fn with_seed(package: &Package, seed: [u8; 32]) -> Self {
        let block = top_block(package);
        let output_types = block_output_types(block);
        let mut rng = StdRng::from_seed(seed);
        let bounds = crate::stimulus::relevant_bounds(block);
        let mut state = block
            .registers
            .iter()
            .map(|r| {
                let pattern = rng.gen_range(0..16);
                crate::stimulus::value(&r.ty, &mut rng, pattern, &bounds)
            })
            .collect::<Vec<_>>();
        let initial_state = block
            .registers
            .iter()
            .zip(&state)
            .filter(|(r, _)| r.ty.bit_count() != 0)
            .map(|(r, value)| {
                (
                    r.name.clone(),
                    LogicValue::from_bits(&flatten_value(value, &r.ty)),
                )
            })
            .collect();
        let sequential = !block.registers.is_empty();
        let mut samples = Vec::new();
        let live = crate::coverage::live_nodes(block);
        let mut observed_live_behaviors = BTreeMap::new();
        for sample in 0..if sequential {
            CYCLE_COUNT
        } else {
            INPUT_SAMPLE_COUNT
        } {
            let mut inputs = crate::stimulus::inputs(block, &mut rng, sample, &bounds);
            if let Some(reset) = &block.reset {
                assert!(
                    !reset.asynchronous,
                    "cycle trace does not model asynchronous reset events"
                );
                let asserted = matches!(sample % 12, 2 | 3 | 9);
                let position = block.input_ports().position(|p| p == reset.port).unwrap();
                inputs[position] =
                    xlsynth_pir::IrValue::from_bits(&xlsynth_pir::IrBits::from_lsb_is_0(&[
                        asserted ^ reset.active_low,
                    ]));
            }
            let bindings = block
                .input_ports()
                .zip(&inputs)
                .filter(|(p, _)| block.port_type(*p).bit_count() != 0)
                .map(|(p, value)| {
                    (
                        block.port_name(p).to_string(),
                        LogicValue::from_bits(&flatten_value(value, block.port_type(p))),
                    )
                })
                .collect();
            let evaluated = evaluate_block_cycle_observed(block, &inputs, &state);
            crate::coverage::record_behaviors(
                block,
                &evaluated.node_values,
                &live,
                &mut observed_live_behaviors,
            );
            let outputs = evaluated.outputs;
            let next_state = evaluated.next_state;
            let outputs = block
                .output_ports()
                .map(|port| block.port_name(port))
                .zip(&output_types)
                .zip(&outputs)
                .filter(|((_, ty), _)| ty.bit_count() != 0)
                .map(|((name, ty), value)| {
                    (
                        name.to_string(),
                        LogicValue::from_bits(&flatten_value(value, ty)),
                    )
                })
                .collect();
            let next_bindings = sequential.then(|| {
                block
                    .registers
                    .iter()
                    .zip(&next_state)
                    .filter(|(r, _)| r.ty.bit_count() != 0)
                    .map(|(r, value)| {
                        (
                            r.name.clone(),
                            LogicValue::from_bits(&flatten_value(value, &r.ty)),
                        )
                    })
                    .collect()
            });
            samples.push(Sample {
                inputs: bindings,
                outputs,
                next_state: next_bindings,
            });
            state = next_state;
        }
        Self {
            initial_state,
            samples,
            observed_live_behaviors,
        }
    }
}
