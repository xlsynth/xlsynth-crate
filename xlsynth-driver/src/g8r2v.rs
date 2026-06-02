// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::{ClockPort, add_input_registers, add_output_registers};
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;

pub fn handle_g8r2v(matches: &clap::ArgMatches) -> Result<(), String> {
    let g8r_input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let add_clk_port = matches.get_one::<String>("add-clk-port").cloned();
    let flop_inputs = matches.get_flag("flop-inputs");
    let flop_outputs = matches.get_flag("flop-outputs");
    let use_system_verilog = matches.get_flag("use-system-verilog");
    let module_name_override = matches.get_one::<String>("module-name").cloned();

    let mut design = load_sequential_gate_fn_from_path(Path::new(g8r_input_file))?;
    if let Some(name) = add_clk_port {
        let requested_clock = ClockPort { name };
        match &design.clock {
            Some(existing) if existing != &requested_clock => {
                return Err(format!(
                    "--add-clk-port '{}' does not match the stored design clock '{}'",
                    requested_clock.name, existing.name
                ));
            }
            Some(_) => {}
            None => design.clock = Some(requested_clock),
        }
    }
    if flop_inputs || flop_outputs {
        let clock = design.clock.clone().ok_or_else(|| {
            "--add-clk-port <NAME> is required when --flop-inputs or --flop-outputs is used."
                .to_string()
        })?;
        if flop_inputs {
            design = add_input_registers(&design, clock.clone())?;
        }
        if flop_outputs {
            design = add_output_registers(&design, clock)?;
        }
    }
    if let Some(module_name) = module_name_override {
        design.name = module_name;
    }

    let netlist_str =
        xlsynth_g8r::aig_serdes::emit_netlist::emit_netlist(&design, use_system_verilog)?;

    println!("{}", netlist_str);
    Ok(())
}
