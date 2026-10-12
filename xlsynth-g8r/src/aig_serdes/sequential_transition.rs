// SPDX-License-Identifier: Apache-2.0

//! Validated scalar-AIG interchange for native sequential transition functions.

use std::collections::BTreeSet;

use crate::aig::gate::GateFn;
use crate::aig::sequential_gate::SequentialGateFn;
use crate::aig_serdes::gate2ir::{GateFnInterfaceSchema, repack_gate_fn_interface_with_schema};

/// Rebinds an optimized, scalar AIGER transition to native packed state ports.
///
/// The replacement must retain the ordered scalar port symbols emitted by the
/// AIGER writers (LSB first, with `_bit` suffixes for packed ports). Zero-width
/// ports have no scalar symbols and are restored from the original schema.
/// This checks the interface, not functional equivalence: the caller is
/// responsible for ensuring the replacement implements the same transition.
/// Explicit register controls must be legalized before transition interchange.
pub fn rebind_sequential_transition(
    design: &SequentialGateFn,
    optimized_transition: &GateFn,
) -> Result<SequentialGateFn, String> {
    design.validate_legalized()?;
    validate_flattened_interface(&design.transition, optimized_transition)?;
    let schema = GateFnInterfaceSchema::from_gate_fn(&design.transition)?;
    let mut transition =
        repack_gate_fn_interface_with_schema(optimized_transition.clone(), &schema)?;
    transition.name.clone_from(&design.transition.name);
    SequentialGateFn::new(
        design.name.clone(),
        transition,
        design.inputs.clone(),
        design.outputs.clone(),
        design.clock.clone(),
        design.registers.clone(),
    )
}

/// Rejects reordered, renamed, or missing scalar transition boundary bits.
fn validate_flattened_interface(original: &GateFn, optimized: &GateFn) -> Result<(), String> {
    let original_inputs = flattened_input_names(original)?;
    let optimized_inputs = flattened_input_names(optimized)?;
    check_flattened_port_names("input", &original_inputs, &optimized_inputs)?;
    let original_outputs = flattened_output_names(original)?;
    let optimized_outputs = flattened_output_names(optimized)?;
    check_flattened_port_names("output", &original_outputs, &optimized_outputs)
}

/// Returns scalar input names in stable AIGER order.
fn flattened_input_names(graph: &GateFn) -> Result<Vec<String>, String> {
    flattened_port_names(
        "input",
        graph
            .inputs
            .iter()
            .map(|input| (input.name.as_str(), input.get_bit_count())),
    )
}

/// Returns scalar output names in stable AIGER order.
fn flattened_output_names(graph: &GateFn) -> Result<Vec<String>, String> {
    flattened_port_names(
        "output",
        graph
            .outputs
            .iter()
            .map(|output| (output.name.as_str(), output.get_bit_count())),
    )
}

/// Expands packed port names using the canonical binary AIGER convention.
fn flattened_port_names<'a>(
    kind: &str,
    ports: impl Iterator<Item = (&'a str, usize)>,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    for (name, bit_count) in ports {
        for bit_index in 0..bit_count {
            let scalar = if bit_count == 1 {
                name.to_string()
            } else {
                format!("{name}_{bit_index}")
            };
            if !seen.insert(scalar.clone()) {
                return Err(format!(
                    "transition has duplicate flattened {kind} '{scalar}'"
                ));
            }
            result.push(scalar);
        }
    }
    Ok(result)
}

/// Reports the exact scalar transition boundary that ABC failed to preserve.
fn check_flattened_port_names(
    kind: &str,
    expected: &[String],
    actual: &[String],
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "optimized transition {kind} interface has {} bits; original transition has {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            return Err(format!(
                "optimized transition {kind} bit {index} is '{actual}'; expected '{expected}'"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::sequential_gate::{ClockPort, add_input_registers, add_output_registers};
    use crate::aig_serdes::emit_aiger::emit_aiger;
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    use crate::aig_serdes::g8r::emit_g8r;
    use crate::aig_serdes::load_aiger_auto::load_aiger_auto;
    use crate::aig_sim::sequential::{SequentialState, simulate};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::{interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case};
    use xlsynth_pir::IrBits;

    /// Exercises both wire dialects and checks metadata plus cycle-zero traces.
    fn check_roundtrip(design: &SequentialGateFn) {
        let inputs = (0..8)
            .map(|cycle| {
                design
                    .inputs
                    .iter()
                    .map(|id| {
                        let width = design.transition.inputs[id.index()].get_bit_count();
                        IrBits::from_lsb_is_0(
                            &(0..width)
                                .map(|bit| (bit + cycle) % 3 == 0)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let initial = design
            .registers
            .iter()
            .map(|r| {
                IrBits::from_lsb_is_0(&vec![
                    false;
                    design.transition.inputs[r.q.index()].get_bit_count()
                ])
            })
            .collect::<Vec<_>>();
        let expected = simulate(
            design,
            &inputs,
            SequentialState::from_register_values(design, initial.clone()).unwrap(),
        )
        .unwrap();
        for bytes in [
            emit_aiger(&design.transition, true).unwrap().into_bytes(),
            emit_aiger_binary(&design.transition, true).unwrap(),
        ] {
            let flat = load_aiger_auto(&bytes, GateBuilderOptions::no_opt())
                .unwrap()
                .gate_fn;
            let rebound = rebind_sequential_transition(design, &flat).unwrap();
            assert_eq!(rebound.name, design.name);
            assert_eq!(rebound.clock, design.clock);
            assert_eq!(rebound.inputs, design.inputs);
            assert_eq!(rebound.outputs, design.outputs);
            assert_eq!(rebound.registers, design.registers);
            assert_eq!(
                GateFnInterfaceSchema::from_gate_fn(&rebound.transition).unwrap(),
                GateFnInterfaceSchema::from_gate_fn(&design.transition).unwrap()
            );
            let actual = simulate(
                &rebound,
                &inputs,
                SequentialState::from_register_values(&rebound, initial.clone()).unwrap(),
            )
            .unwrap();
            assert_eq!(actual.external_outputs(), expected.external_outputs());
            assert_eq!(actual.final_state(), expected.final_state());
            assert_eq!(
                emit_g8r(&rebound).unwrap(),
                emit_g8r(&rebind_sequential_transition(design, &flat).unwrap()).unwrap()
            );
        }
    }

    #[test]
    fn roundtrips_shared_signature_corpus_with_and_without_registers() {
        for case in interesting_ir_roundtrip_cases() {
            let design =
                SequentialGateFn::from_gate_fn(load_interesting_ir_roundtrip_case(case).gate_fn);
            check_roundtrip(&design);
            let clock = ClockPort { name: "clk".into() };
            let design = add_input_registers(&design, clock.clone()).unwrap();
            let design = add_output_registers(&design, clock).unwrap();
            check_roundtrip(&design);
        }
    }

    #[test]
    fn roundtrips_wide_state_and_complemented_outputs() {
        let mut b = GateBuilder::new("wide".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 129);
        b.add_output("x".into(), x.clone());
        b.add_output(
            "not_x".into(),
            crate::aig::AigBitVector::from_lsb_is_index_0(
                &x.iter_lsb_to_msb().map(|v| v.negate()).collect::<Vec<_>>(),
            ),
        );
        b.add_output(
            "one".into(),
            crate::aig::AigBitVector::from_lsb_is_index_0(&[b.get_true()]),
        );
        let mut design = add_output_registers(
            &SequentialGateFn::from_gate_fn(b.build()),
            ClockPort { name: "clk".into() },
        )
        .unwrap();
        // A transition output can also be externally observed.
        design.outputs.push(design.registers[0].d);
        check_roundtrip(&design);
    }

    #[test]
    fn rejects_reordered_missing_and_renamed_boundary_bits() {
        let mut b = GateBuilder::new("pair".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 2);
        b.add_output("out".into(), x);
        let design = SequentialGateFn::from_gate_fn(b.build());
        let flat = load_aiger_auto(
            &emit_aiger_binary(&design.transition, true).unwrap(),
            GateBuilderOptions::no_opt(),
        )
        .unwrap()
        .gate_fn;
        let mut changed = flat.clone();
        changed.inputs.swap(0, 1);
        assert_eq!(
            rebind_sequential_transition(&design, &changed).unwrap_err(),
            "optimized transition input bit 0 is 'x_1'; expected 'x_0'"
        );
        let mut changed = flat.clone();
        changed.outputs.pop();
        assert_eq!(
            rebind_sequential_transition(&design, &changed).unwrap_err(),
            "optimized transition output interface has 1 bits; original transition has 2"
        );
        let mut changed = flat;
        changed.outputs[0].name = "renamed".into();
        assert_eq!(
            rebind_sequential_transition(&design, &changed).unwrap_err(),
            "optimized transition output bit 0 is 'renamed'; expected 'out_0'"
        );
    }

    #[test]
    fn rejects_ambiguous_flattened_names() {
        let mut b = GateBuilder::new("collision".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 2);
        b.add_input("x_0".into(), 1);
        b.add_output("out".into(), x);
        let design = SequentialGateFn::from_gate_fn(b.build());
        assert_eq!(
            rebind_sequential_transition(&design, &design.transition).unwrap_err(),
            "transition has duplicate flattened input 'x_0'"
        );
    }
}
