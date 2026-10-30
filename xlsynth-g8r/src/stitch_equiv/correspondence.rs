// SPDX-License-Identifier: Apache-2.0

//! Proposes direct state-bit correspondences for formal stage proofs.

use std::collections::BTreeMap;

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::StitchObservation;
use crate::aig::SequentialGateFn;
use crate::aig_sim::gate_simd::{Vec256, eval_node_values_in_order_into};

/// One candidate relation: mapped Q equals reference Q XOR `inverted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RegisterBitPair {
    pub reference_register: usize,
    pub reference_bit: usize,
    pub mapped_register: usize,
    pub mapped_bit: usize,
    pub inverted: bool,
}

const SAMPLE_FRAMES: usize = 4;

/// Observed Q values for one bit, canonicalized so opposite polarities group
/// together.
struct BitSample {
    register: usize,
    bit: usize,
    signature: [[u64; 4]; SAMPLE_FRAMES],
    inverted: bool,
}

/// Samples 256 independent input histories together, retaining only register
/// values after fill. Both designs draw identical input bits in port-name
/// order.
fn sample_registers(
    design: &SequentialGateFn,
    latency: usize,
    controls: StitchObservation<'_>,
) -> Result<Vec<BitSample>, String> {
    let mut offsets = Vec::with_capacity(design.transition.inputs.len());
    let mut inputs = Vec::new();
    for port in &design.transition.inputs {
        offsets.push(inputs.len());
        inputs.resize(inputs.len() + port.get_bit_count(), Vec256::zero());
    }
    let mut external = design.inputs.clone();
    external.sort_by_key(|id| &design.transition.inputs[id.index()].name);
    let mut bindings = Vec::new();
    let mut samples = Vec::new();
    for (register_index, register) in design.registers.iter().enumerate() {
        let d = &design.transition.outputs[register.d.index()].bit_vector;
        for (bit, operand) in d.iter_lsb_to_msb().enumerate() {
            bindings.push((offsets[register.q.index()] + bit, *operand));
            samples.push(BitSample {
                register: register_index,
                bit,
                signature: [[0; 4]; SAMPLE_FRAMES],
                inverted: false,
            });
        }
    }
    if samples.is_empty() {
        return Ok(samples);
    }
    // An enabled register can retain its initial value across bubbles. Fill
    // every layer with transactions first so complemented encodings align
    // even though both simulations started with physical zeroes.
    let warmup = if controls.input_valid.is_some() {
        latency
    } else {
        latency.saturating_sub(1)
    };
    let frames = warmup
        .checked_add(SAMPLE_FRAMES)
        .ok_or("pipeline latency overflow")?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let order = design.transition.post_order_refs();
    let mut values = vec![Vec256::zero(); design.transition.gates.len()];
    for frame in 0..frames {
        for id in &external {
            let start = offsets[id.index()];
            let port = &design.transition.inputs[id.index()];
            let width = port.get_bit_count();
            if frame < warmup && controls.input_valid == Some(port.name.as_str()) {
                inputs[start] = Vec256::splat(true);
                continue;
            }
            if frame < warmup && controls.reset == Some(port.name.as_str()) {
                inputs[start] = Vec256::splat(controls.reset_active_low);
                continue;
            }
            for value in &mut inputs[start..start + width] {
                // Sparse and dense stimuli exercise rare conditions missed by
                // uniform data, for example exponent extremes in FP arithmetic.
                *value = Vec256::from_words([
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64() & rng.next_u64() & rng.next_u64(),
                    rng.next_u64() | rng.next_u64() | rng.next_u64(),
                ]);
            }
        }
        eval_node_values_in_order_into(&design.transition, &inputs, &order, &mut values);
        for ((q, d), sample) in bindings.iter().zip(&mut samples) {
            let next = values[d.node.id].apply_neg(d.negated);
            inputs[*q] = next;
            if frame >= warmup {
                sample.signature[frame - warmup] = next.to_array();
            }
        }
    }
    for sample in &mut samples {
        sample.inverted = sample.signature[0][0] & 1 != 0;
        if sample.inverted {
            for word in sample.signature.iter_mut().flatten() {
                *word = !*word;
            }
        }
    }
    Ok(samples)
}

/// Pairs every register bit of two ABI-checked, effective-D designs. An
/// observed match only proposes a pair. Its capture and uses must still be
/// proved over arbitrary inputs and boundary values.
pub(super) fn direct_register_pairs(
    reference: &SequentialGateFn,
    mapped: &SequentialGateFn,
    latency: usize,
    controls: StitchObservation<'_>,
) -> Result<Vec<RegisterBitPair>, String> {
    let logical = sample_registers(reference, latency, controls)?;
    let physical = sample_registers(mapped, latency, controls)?;
    let physical_by_name: BTreeMap<_, _> = physical
        .iter()
        .enumerate()
        .map(|(index, bit)| {
            (
                (mapped.registers[bit.register].name.as_str(), bit.bit),
                index,
            )
        })
        .collect();

    // Reserve compatible exact names before assigning ambiguous duplicate or
    // constant bits. For renamed flops, break ties in register/LSB order.
    let mut used = vec![false; physical.len()];
    let mut named_matches = Vec::with_capacity(logical.len());
    for bit in &logical {
        let key = (reference.registers[bit.register].name.as_str(), bit.bit);
        let candidate = physical_by_name
            .get(&key)
            .copied()
            .filter(|&index| physical[index].signature == bit.signature);
        if let Some(index) = candidate {
            used[index] = true;
        }
        named_matches.push(candidate);
    }
    let mut candidates = BTreeMap::<_, Vec<usize>>::new();
    for (index, bit) in physical.iter().enumerate().rev() {
        if !used[index] {
            candidates.entry(&bit.signature).or_default().push(index);
        }
    }
    let mut pairs = Vec::with_capacity(logical.len());
    for (bit, named_match) in logical.iter().zip(named_matches) {
        let index = named_match.or_else(|| candidates.get_mut(&bit.signature)?.pop())
            .ok_or_else(|| format!(
                "no direct mapped partner for reference boundary register '{}[{}]' ({} reference bits, {} mapped bits)",
                reference.registers[bit.register].name, bit.bit, logical.len(), physical.len()
            ))?;
        used[index] = true;
        let other = &physical[index];
        pairs.push(RegisterBitPair {
            reference_register: bit.register,
            reference_bit: bit.bit,
            mapped_register: other.register,
            mapped_bit: other.bit,
            inverted: bit.inverted != other.inverted,
        });
    }
    if let Some(index) = used.iter().position(|matched| !matched) {
        let bit = &physical[index];
        return Err(format!(
            "no direct reference boundary partner for mapped register '{}[{}]' ({} reference bits, {} mapped bits)",
            mapped.registers[bit.register].name,
            bit.bit,
            logical.len(),
            physical.len()
        ));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, SequentialPipelineOptions,
        add_output_registers, stitch_gate_fns_into_pipeline,
    };

    /// Two logical outputs share x[0]. The physical variant splits a bus into
    /// scalar flops, reverses its input ports, and inverts two state bits.
    fn register_fixture(physical: bool) -> Result<SequentialGateFn, String> {
        let mut b = GateBuilder::new("fixture".into(), GateBuilderOptions::opt());
        let (x, y) = if physical {
            let y = b.add_input("y".into(), 1);
            (b.add_input("x".into(), 2), y)
        } else {
            let x = b.add_input("x".into(), 2);
            (x, b.add_input("y".into(), 1))
        };
        if physical {
            b.add_output("alias".into(), x.get_lsb_slice(0, 1));
            b.add_output("ff_y".into(), y.get_lsb(0).negate().into());
            b.add_output("ff_x1".into(), x.get_lsb_slice(1, 1));
            b.add_output("tie_high".into(), b.get_true().into());
            b.add_output("ff_x0".into(), x.get_lsb_slice(0, 1));
        } else {
            b.add_output("bus".into(), AigBitVector::concat(y, x.clone()));
            b.add_output("alias".into(), x.get_lsb_slice(0, 1));
            b.add_output("tie_low".into(), b.get_false().into());
        }
        add_output_registers(
            &SequentialGateFn::from_gate_fn(b.build()),
            ClockPort { name: "clk".into() },
        )
    }

    #[test]
    fn pairs_renamed_permuted_bits_with_polarity_and_reserves_named_aliases() {
        let reference = register_fixture(false).unwrap();
        let mapped = register_fixture(true).unwrap();
        let pairs =
            direct_register_pairs(&reference, &mapped, 1, StitchObservation::default()).unwrap();
        let tuples: Vec<_> = pairs
            .iter()
            .map(|p| {
                (
                    p.reference_register,
                    p.reference_bit,
                    p.mapped_register,
                    p.mapped_bit,
                    p.inverted,
                )
            })
            .collect();
        assert_eq!(
            tuples,
            [
                (0, 0, 4, 0, false),
                (0, 1, 2, 0, false),
                (0, 2, 1, 0, true),
                (1, 0, 0, 0, false),
                (2, 0, 3, 0, true)
            ]
        );
    }

    #[test]
    fn uses_independent_input_frames_and_waits_for_all_layers_to_fill() {
        const LAYERS: usize = 12;
        let pipeline = |invert| -> Result<SequentialGateFn, String> {
            let mut b = GateBuilder::new("shift".into(), GateBuilderOptions::opt());
            let x = *b.add_input("x".into(), 1).get_lsb(0);
            b.add_output("out".into(), if invert { x.negate() } else { x }.into());
            let mut design = SequentialGateFn::from_gate_fn(b.build());
            for _ in 0..LAYERS {
                design = add_output_registers(&design, ClockPort { name: "clk".into() })?;
            }
            Ok(design)
        };
        let reference = pipeline(false).unwrap();
        let mut mapped = pipeline(true).unwrap();
        mapped.registers.reverse();
        for (index, register) in mapped.registers.iter_mut().enumerate() {
            register.name = format!("ff_{index}");
        }
        let pairs =
            direct_register_pairs(&reference, &mapped, LAYERS, StitchObservation::default())
                .unwrap();
        assert_eq!(pairs.len(), LAYERS);
        for (index, pair) in pairs.iter().enumerate() {
            assert_eq!(pair.reference_register, index);
            assert_eq!(pair.mapped_register, LAYERS - index - 1);
            assert!(pair.inverted);
        }
    }

    #[test]
    fn fills_enabled_registers_before_pairing_inverted_data_bits() {
        let boundary = SequentialPipelineOptions {
            name: "pipe".into(),
            clock: ClockPort { name: "clk".into() },
            flop_inputs: false,
            flop_outputs: true,
            input_valid_signal: Some("vin".into()),
            output_valid_signal: Some("vout".into()),
            reset_signal: Some("rst_n".into()),
            reset_active_low: true,
        };
        // Inverting both stages complements the intermediate data flop while
        // preserving the observable result. Each data flop holds on bubbles.
        let build = |invert: bool| {
            let mut b = GateBuilder::new("stage".into(), GateBuilderOptions::opt());
            let x = *b.add_input("x".into(), 1).get_lsb(0);
            b.add_output("out".into(), if invert { x.negate() } else { x }.into());
            let stage = b.build();
            stitch_gate_fns_into_pipeline(&[stage.clone(), stage], &boundary)
        };
        let reference = build(false).unwrap();
        let mapped = build(true).unwrap();
        let pairs =
            direct_register_pairs(&reference, &mapped, 2, StitchObservation::from(&boundary))
                .unwrap();
        assert_eq!(
            pairs
                .iter()
                .map(|p| (p.mapped_register, p.inverted))
                .collect::<Vec<_>>(),
            [(0, true), (1, false), (2, false), (3, false)]
        );
    }

    #[test]
    fn reports_missing_bits_on_either_side_and_accepts_zero_registers() {
        let registered = register_fixture(false).unwrap();
        let mut b = GateBuilder::new("combo".into(), GateBuilderOptions::opt());
        let x = b.add_input("x".into(), 2);
        b.add_input("y".into(), 1);
        b.add_output("out".into(), x);
        let empty = SequentialGateFn::from_gate_fn(b.build());
        assert!(
            direct_register_pairs(&empty, &empty, 0, StitchObservation::default())
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            direct_register_pairs(&registered, &empty, 1, StitchObservation::default())
                .unwrap_err(),
            "no direct mapped partner for reference boundary register 'p0_bus[0]' (5 reference bits, 0 mapped bits)"
        );
        assert_eq!(
            direct_register_pairs(&empty, &registered, 1, StitchObservation::default())
                .unwrap_err(),
            "no direct reference boundary partner for mapped register 'p0_bus[0]' (0 reference bits, 5 mapped bits)"
        );
    }
}
