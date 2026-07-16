// SPDX-License-Identifier: Apache-2.0

//! Shared CLI input, state, and output handling for sequential evaluators.

use std::path::Path;

use clap::ArgMatches;
use xlsynth::{IrBits, IrValue, NamedIrValueSet, parse_ir_values_file};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_sim::sequential::SequentialState;

fn lower_tuple_to_bits(
    tuple: &IrValue,
    port_names: &[String],
    port_widths: &[usize],
    context: &str,
) -> Result<Vec<IrBits>, String> {
    let elements = tuple
        .get_elements()
        .map_err(|error| format!("{context} is not a tuple: {error}"))?;
    if elements.len() != port_names.len() {
        return Err(format!(
            "{context} has {} values, expected {}",
            elements.len(),
            port_names.len()
        ));
    }
    elements
        .iter()
        .zip(port_names.iter().zip(port_widths))
        .enumerate()
        .map(|(index, (value, (name, expected_width)))| {
            let bits = value.to_bits().map_err(|error| {
                format!(
                    "{context} value {} ('{}') is not bits-typed: {error}",
                    index, name
                )
            })?;
            if bits.get_bit_count() != *expected_width {
                return Err(format!(
                    "{context} value {} ('{}') has width {}, expected {}",
                    index,
                    name,
                    bits.get_bit_count(),
                    expected_width
                ));
            }
            Ok(bits)
        })
        .collect()
}

fn external_input_interface(design: &SequentialGateFn) -> (Vec<String>, Vec<usize>) {
    let names = design
        .inputs
        .iter()
        .map(|id| design.transition.inputs[id.index()].name.clone())
        .collect();
    let widths = design
        .inputs
        .iter()
        .map(|id| design.transition.inputs[id.index()].get_bit_count())
        .collect();
    (names, widths)
}

fn register_interface(design: &SequentialGateFn) -> (Vec<String>, Vec<usize>) {
    let names = design
        .registers
        .iter()
        .map(|register| register.name.clone())
        .collect();
    let widths = design
        .registers
        .iter()
        .map(|register| design.transition.inputs[register.q.index()].get_bit_count())
        .collect();
    (names, widths)
}

/// Reads one external-input record per simulated cycle.
pub(crate) fn read_external_inputs(
    matches: &ArgMatches,
    design: &SequentialGateFn,
) -> Result<Vec<Vec<IrBits>>, String> {
    let (names, widths) = external_input_interface(design);
    let samples = if let Some(arg_tuple) = matches.get_one::<String>("arg_tuple") {
        vec![
            IrValue::parse_typed(arg_tuple)
                .map_err(|error| format!("failed to parse argument tuple: {error}"))?,
        ]
    } else {
        let path = matches
            .get_one::<String>("input_irvals")
            .expect("clap requires an input source");
        parse_ir_values_file(Path::new(path))
            .and_then(|file| file.into_positional_values(&names))
            .map_err(|error| error.to_string())?
    };
    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            lower_tuple_to_bits(
                sample,
                &names,
                &widths,
                &format!("input cycle {}", index + 1),
            )
        })
        .collect()
}

fn read_initial_state_file(
    path: &Path,
    design: &SequentialGateFn,
) -> Result<SequentialState, String> {
    let (names, widths) = register_interface(design);
    let mut records = parse_ir_values_file(path)
        .and_then(|file| file.into_positional_values(&names))
        .map_err(|error| error.to_string())?;
    if records.len() != 1 {
        return Err(format!(
            "--initial-state-file must contain exactly one record; got {}",
            records.len()
        ));
    }
    let values = lower_tuple_to_bits(
        &records.pop().expect("record count was checked"),
        &names,
        &widths,
        "initial state",
    )?;
    SequentialState::from_register_values(design, values)
}

/// Resolves the explicit initial-state policy for one sequential design.
pub(crate) fn resolve_initial_state(
    matches: &ArgMatches,
    design: &SequentialGateFn,
    allow_declared_initial_values: bool,
    design_kind: &str,
) -> Result<SequentialState, String> {
    let all_zeros = matches.get_flag("initial_state_all_zeros");
    let from_declared =
        allow_declared_initial_values && matches.get_flag("initial_state_from_g8r_initial_values");
    let state_file = matches.get_one::<String>("initial_state_file");
    let selected_count =
        usize::from(all_zeros) + usize::from(from_declared) + usize::from(state_file.is_some());

    if design.registers.is_empty() {
        if selected_count != 0 {
            return Err(format!(
                "initial-state options are invalid for a {design_kind} design without registers"
            ));
        }
        return SequentialState::from_register_values(design, vec![]);
    }
    if selected_count != 1 {
        let options = if allow_declared_initial_values {
            "--initial-state-all-zeros, --initial-state-from-g8r-initial-values, or --initial-state-file"
        } else {
            "--initial-state-all-zeros or --initial-state-file"
        };
        return Err(format!(
            "a {design_kind} design with registers requires exactly one of {options}"
        ));
    }
    if all_zeros {
        Ok(SequentialState::all_zeros(design))
    } else if from_declared {
        SequentialState::from_g8r_initial_values(design)
    } else {
        read_initial_state_file(
            Path::new(state_file.expect("one initial-state option was selected")),
            design,
        )
    }
}

/// Formats one cycle of external outputs as the CLI IR-value convention.
pub(crate) fn output_value(outputs: &[IrBits]) -> IrValue {
    if outputs.len() == 1 {
        IrValue::from_bits(&outputs[0])
    } else {
        IrValue::make_tuple(&outputs.iter().map(IrValue::from_bits).collect::<Vec<_>>())
    }
}

/// Writes final register state as one named IR-values record.
pub(crate) fn write_final_state(
    path: &Path,
    design: &SequentialGateFn,
    state: &SequentialState,
) -> Result<(), String> {
    let (names, _) = register_interface(design);
    let tuple = IrValue::make_tuple(
        &state
            .values()
            .iter()
            .map(IrValue::from_bits)
            .collect::<Vec<_>>(),
    );
    let named = NamedIrValueSet::from_positional_tuple(&names, &tuple)
        .map_err(|error| format!("failed to format final state: {error}"))?;
    std::fs::write(path, format!("{named}\n"))
        .map_err(|error| format!("failed to write final state to {}: {error}", path.display()))
}
