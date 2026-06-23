// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;

use clap::ArgMatches;
use xlsynth::{IrBits, IrValue, NamedIrValueSet, parse_ir_values_file};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::aig_sim::sequential::{
    SequentialState, count_sequential_toggle_activity, simulate,
};

fn lower_tuple_to_bits(
    tuple: &IrValue,
    port_names: &[String],
    port_widths: &[usize],
    context: &str,
) -> Result<Vec<IrBits>, String> {
    let elements = tuple
        .get_elements()
        .map_err(|e| format!("{context} is not a tuple: {e}"))?;
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
            let bits = value.to_bits().map_err(|e| {
                format!(
                    "{context} value {} ('{}') is not bits-typed: {e}",
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

fn read_external_inputs(
    matches: &ArgMatches,
    design: &SequentialGateFn,
) -> Result<Vec<Vec<IrBits>>, String> {
    let (names, widths) = external_input_interface(design);
    let samples = if let Some(arg_tuple) = matches.get_one::<String>("arg_tuple") {
        vec![
            IrValue::parse_typed(arg_tuple)
                .map_err(|e| format!("failed to parse argument tuple: {e}"))?,
        ]
    } else {
        let path = matches
            .get_one::<String>("input_irvals")
            .expect("clap requires an input source");
        parse_ir_values_file(Path::new(path))
            .and_then(|file| file.into_positional_values(&names))
            .map_err(|e| e.to_string())?
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
        .map_err(|e| e.to_string())?;
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

fn resolve_initial_state(
    matches: &ArgMatches,
    design: &SequentialGateFn,
) -> Result<SequentialState, String> {
    let all_zeros = matches.get_flag("initial_state_all_zeros");
    let from_g8r = matches.get_flag("initial_state_from_g8r_initial_values");
    let state_file = matches.get_one::<String>("initial_state_file");
    let selected_count =
        usize::from(all_zeros) + usize::from(from_g8r) + usize::from(state_file.is_some());

    if design.registers.is_empty() {
        if selected_count != 0 {
            return Err(
                "initial-state options are invalid for a G8R design without registers".to_string(),
            );
        }
        return SequentialState::from_register_values(design, vec![]);
    }
    if selected_count != 1 {
        return Err(
            "a G8R design with registers requires exactly one of --initial-state-all-zeros, --initial-state-from-g8r-initial-values, or --initial-state-file"
                .to_string(),
        );
    }
    if all_zeros {
        Ok(SequentialState::all_zeros(design))
    } else if from_g8r {
        SequentialState::from_g8r_initial_values(design)
    } else {
        read_initial_state_file(
            Path::new(state_file.expect("one initial-state option was selected")),
            design,
        )
    }
}

fn output_value(outputs: &[IrBits]) -> IrValue {
    if outputs.len() == 1 {
        IrValue::from_bits(&outputs[0])
    } else {
        IrValue::make_tuple(&outputs.iter().map(IrValue::from_bits).collect::<Vec<_>>())
    }
}

fn write_final_state(
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
        .map_err(|e| format!("failed to format final state: {e}"))?;
    std::fs::write(path, format!("{named}\n"))
        .map_err(|e| format!("failed to write final state to {}: {e}", path.display()))
}

fn write_toggle_activity(
    path: &Path,
    activity: &xlsynth_g8r::aig_sim::sequential::SequentialToggleActivity,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, activity)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    writeln!(writer).map_err(|e| format!("failed to finalize {}: {e}", path.display()))
}

pub fn handle_g8r_eval(matches: &ArgMatches) -> Result<(), String> {
    let g8r_file = matches
        .get_one::<String>("g8r_file")
        .expect("g8r_file is required by clap");
    let design = load_sequential_gate_fn_from_path(Path::new(g8r_file))?;
    let inputs = read_external_inputs(matches, &design)?;
    if matches.get_one::<String>("toggle_output_json").is_some() && inputs.len() < 2 {
        return Err(format!(
            "--toggle-output-json requires at least two --input-irvals cycles; got {}",
            inputs.len()
        ));
    }
    let initial_state = resolve_initial_state(matches, &design)?;
    let trace = simulate(&design, &inputs, initial_state)?;

    for outputs in trace.external_outputs() {
        println!("{}", output_value(outputs));
    }
    if let Some(path) = matches.get_one::<String>("final_state_irvals") {
        if design.registers.is_empty() {
            return Err(
                "--final-state-irvals is invalid for a G8R design without registers".to_string(),
            );
        }
        write_final_state(Path::new(path), &design, trace.final_state())?;
    }
    if let Some(path) = matches.get_one::<String>("toggle_output_json") {
        let activity = count_sequential_toggle_activity(&design, &trace)?;
        write_toggle_activity(Path::new(path), &activity)?;
    }
    Ok(())
}
