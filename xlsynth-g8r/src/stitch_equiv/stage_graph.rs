// SPDX-License-Identifier: Apache-2.0

//! One-edge proof obligations for a bitwise register correspondence.

use std::collections::BTreeMap;

use crate::aig::import::import_gate_fn;
use crate::aig::{AigBitVector, AigOperand, GateFn, SequentialGateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

use super::correspondence::RegisterBitPair;
use super::{initial_state, matching_interface, output_name};

/// Shared combinational graph and the mismatch bits that each stage must rule
/// out.
pub(super) struct StageMiters {
    pub graph: GateFn,
    pub mismatches: Vec<Vec<AigOperand>>,
}

/// Verifies that the correspondence visits every physical state bit exactly
/// once.
fn check_bit_cover(
    design: &SequentialGateFn,
    side: &str,
    endpoints: impl Iterator<Item = (usize, usize)>,
) -> Result<(), String> {
    let mut covered: Vec<Vec<bool>> = design
        .registers
        .iter()
        .map(|reg| vec![false; design.transition.inputs[reg.q.index()].get_bit_count()])
        .collect();
    for (index, bit) in endpoints {
        let bits = covered.get_mut(index).ok_or_else(|| {
            format!("correspondence references missing {side} register index {index}")
        })?;
        let name = &design.registers[index].name;
        let width = bits.len();
        let seen = bits.get_mut(bit).ok_or_else(|| {
            format!(
                "correspondence references {side} register '{name}' bit {bit} outside width {width}"
            )
        })?;
        if *seen {
            return Err(format!(
                "duplicate correspondence for {side} register '{name}' bit {bit}"
            ));
        }
        *seen = true;
    }
    for (reg, bits) in design.registers.iter().zip(covered) {
        if let Some(bit) = bits.iter().position(|seen| !seen) {
            return Err(format!(
                "missing correspondence for {side} register '{}' bit {bit}",
                reg.name
            ));
        }
    }
    Ok(())
}

/// Validates the named interface, stage layout and complete Q bijection for
/// either staged proof encoding.
pub(super) fn validate_stage_relation<'a>(
    reference: &'a SequentialGateFn,
    mapped: &SequentialGateFn,
    pairs: &[RegisterBitPair],
    register_stages: &[usize],
    stage_count: usize,
) -> Result<super::Interface<'a>, String> {
    reference.validate_legalized()?;
    mapped.validate_legalized()?;
    if stage_count == 0 {
        return Err("a staged proof needs at least one stage".into());
    }
    if register_stages.len() != reference.registers.len() {
        return Err(format!(
            "expected a stage for each of {} reference registers, got {}",
            reference.registers.len(),
            register_stages.len()
        ));
    }
    for (reg, stage) in reference.registers.iter().zip(register_stages) {
        if *stage >= stage_count {
            return Err(format!(
                "reference register '{}' belongs to missing stage {stage}",
                reg.name
            ));
        }
    }
    if !mapped.registers.is_empty() && mapped.clock != reference.clock {
        return Err(format!(
            "register clocks differ: reference {:?}, mapped {:?}",
            reference.clock, mapped.clock
        ));
    }
    let expected = matching_interface(reference, mapped)?;
    check_bit_cover(
        reference,
        "reference",
        pairs
            .iter()
            .map(|p| (p.reference_register, p.reference_bit)),
    )?;
    check_bit_cover(
        mapped,
        "mapped",
        pairs.iter().map(|p| (p.mapped_register, p.mapped_bit)),
    )?;
    Ok(expected)
}

/// Compares all next-state bits under the proposed Q relation and compares the
/// visible outputs under the same relation. Inputs and current state are free.
pub(super) fn build_stage_miters(
    reference: &SequentialGateFn,
    mapped: &SequentialGateFn,
    pairs: &[RegisterBitPair],
    register_stages: &[usize],
    stage_count: usize,
) -> Result<StageMiters, String> {
    let expected = validate_stage_relation(reference, mapped, pairs, register_stages, stage_count)?;
    let mut builder = GateBuilder::new("stitch_stage_miters".into(), GateBuilderOptions::opt());
    let external: BTreeMap<_, _> = expected
        .inputs
        .iter()
        .enumerate()
        .map(|(index, (&name, &width))| (name, builder.add_input(format!("port_{index}"), width)))
        .collect();
    let external_bindings = |design: &SequentialGateFn| {
        let mut bindings = design
            .transition
            .inputs
            .iter()
            .map(|port| AigBitVector::zeros(port.get_bit_count()))
            .collect::<Vec<_>>();
        for id in &design.inputs {
            let name = design.transition.inputs[id.index()].name.as_str();
            bindings[id.index()] = external[name].clone();
        }
        bindings
    };
    let mut reference_bindings = external_bindings(reference);
    let mut mapped_bindings = external_bindings(mapped);
    for (reg, state) in
        reference
            .registers
            .iter()
            .zip(initial_state(reference, "state", &mut builder))
    {
        reference_bindings[reg.q.index()] = state;
    }
    for p in pairs {
        let reference_q = reference.registers[p.reference_register].q.index();
        let mapped_q = mapped.registers[p.mapped_register].q.index();
        let mut bit = *reference_bindings[reference_q].get_lsb(p.reference_bit);
        bit.negated ^= p.inverted;
        mapped_bindings[mapped_q].set_lsb(p.mapped_bit, bit);
    }

    let reference_values =
        import_gate_fn(&reference.transition, &reference_bindings, &mut builder)?;
    let mapped_values = import_gate_fn(&mapped.transition, &mapped_bindings, &mut builder)?;
    let mut mismatches = vec![Vec::new(); stage_count];
    for p in pairs {
        let reference_d = reference.registers[p.reference_register].d.index();
        let mapped_d = mapped.registers[p.mapped_register].d.index();
        let expected_bit = *reference_values[reference_d].get_lsb(p.reference_bit);
        let mut actual_bit = *mapped_values[mapped_d].get_lsb(p.mapped_bit);
        actual_bit.negated ^= p.inverted;
        mismatches[register_stages[p.reference_register]]
            .push(builder.add_xor_binary(expected_bit, actual_bit));
    }

    let mapped_outputs: BTreeMap<_, _> = mapped
        .outputs
        .iter()
        .map(|id| {
            (
                output_name(&mapped.transition.outputs[id.index()].name),
                &mapped_values[id.index()],
            )
        })
        .collect();
    for id in &reference.outputs {
        let name = output_name(&reference.transition.outputs[id.index()].name);
        let bits = reference_values[id.index()]
            .iter_lsb_to_msb()
            .zip(mapped_outputs[name].iter_lsb_to_msb());
        mismatches[stage_count - 1].extend(bits.map(|(&a, &b)| builder.add_xor_binary(a, b)));
    }
    for (index, bits) in mismatches.iter().enumerate() {
        builder.add_output(
            format!("stage_{index}"),
            AigBitVector::from_lsb_is_index_0(bits),
        );
    }
    Ok(StageMiters {
        graph: builder.build(),
        mismatches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{ClockPort, RegisterBinding, TransitionInputId, TransitionOutputId};
    use crate::aig_sim::gate_sim::PreparedGateSim;
    use xlsynth_pir::IrBits;

    /// Delays two data bits for two edges, optionally storing (!x[0], x[1]) in
    /// each physical register. The independent sideband samples current inputs.
    fn delay(encoded: bool) -> Result<SequentialGateFn, String> {
        let mut b = GateBuilder::new("delay".into(), GateBuilderOptions::opt());
        let x = b.add_input("x".into(), 2);
        let q0 = b.add_input("q0".into(), 2);
        let q1 = b.add_input("q1".into(), 2);
        let data = if encoded {
            AigBitVector::from_lsb_is_index_0(&[q1.get_lsb(1).negate(), *q1.get_lsb(0)])
        } else {
            q1
        };
        let first_d = if encoded {
            AigBitVector::from_lsb_is_index_0(&[*x.get_lsb(1), x.get_lsb(0).negate()])
        } else {
            x.clone()
        };
        b.add_output(if encoded { "out" } else { "output_value" }.into(), data);
        b.add_output("sideband".into(), x.get_lsb_slice(0, 1));
        b.add_output("d0".into(), first_d);
        b.add_output("d1".into(), q0);
        let registers = (0..2)
            .map(|index| RegisterBinding {
                name: format!("r{index}"),
                q: TransitionInputId::new(index + 1),
                d: TransitionOutputId::new(index + 2),
                enable: None,
                reset: None,
            })
            .collect();
        SequentialGateFn::new(
            "delay".into(),
            b.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0), TransitionOutputId::new(1)],
            Some(ClockPort { name: "clk".into() }),
            registers,
        )
    }

    #[test]
    fn stage_miters_bind_register_permutations_and_phases_and_compare_sidebands() {
        let reference = delay(false).unwrap();
        let mut mapped = delay(true).unwrap();
        mapped.registers.reverse();
        mapped.outputs.reverse();
        let pairs = (0..2)
            .flat_map(|reg| {
                (0..2).map(move |bit| RegisterBitPair {
                    reference_register: reg,
                    reference_bit: bit,
                    mapped_register: 1 - reg,
                    mapped_bit: 1 - bit,
                    inverted: bit == 0,
                })
            })
            .collect::<Vec<_>>();
        let miters = build_stage_miters(&reference, &mapped, &pairs, &[0, 1], 2).unwrap();
        assert_eq!(
            miters.mismatches.iter().map(Vec::len).collect::<Vec<_>>(),
            [2, 5]
        );
        let mut sim = PreparedGateSim::new(&miters.graph);
        for assignment in 0..64 {
            let inputs = (0..3)
                .map(|i| IrBits::make_ubits(2, (assignment >> (2 * i)) & 3).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                sim.eval_outputs(&inputs),
                [
                    IrBits::make_ubits(2, 0).unwrap(),
                    IrBits::make_ubits(5, 0).unwrap()
                ]
            );
        }
        // Invert the mapped capture of reference bit 0 in stage 0 and also
        // invert the sideband. Only those two obligations should be violated.
        let first_d = &mut mapped.transition.outputs[2].bit_vector;
        first_d.set_lsb(1, first_d.get_lsb(1).negate());
        let sideband = &mut mapped.transition.outputs[1].bit_vector;
        sideband.set_lsb(0, sideband.get_lsb(0).negate());
        let miters = build_stage_miters(&reference, &mapped, &pairs, &[0, 1], 2).unwrap();
        let mut sim = PreparedGateSim::new(&miters.graph);
        let inputs = vec![IrBits::make_ubits(2, 0).unwrap(); 3];
        assert_eq!(
            sim.eval_outputs(&inputs),
            [
                IrBits::make_ubits(2, 1).unwrap(),
                IrBits::make_ubits(5, 0b10000).unwrap()
            ]
        );
    }

    #[test]
    fn stage_miters_reject_incomplete_or_nonbijective_state_relations() {
        let design = delay(false).unwrap();
        let mut pairs = (0..2)
            .flat_map(|reg| {
                (0..2).map(move |bit| RegisterBitPair {
                    reference_register: reg,
                    reference_bit: bit,
                    mapped_register: reg,
                    mapped_bit: bit,
                    inverted: false,
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            build_stage_miters(&design, &design, &pairs[..3], &[0, 1], 2)
                .err()
                .unwrap(),
            "missing correspondence for reference register 'r1' bit 1"
        );
        pairs[3].mapped_bit = 0;
        assert_eq!(
            build_stage_miters(&design, &design, &pairs, &[0, 1], 2)
                .err()
                .unwrap(),
            "duplicate correspondence for mapped register 'r1' bit 0"
        );
    }
}
