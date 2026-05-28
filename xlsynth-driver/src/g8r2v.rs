// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;

pub fn handle_g8r2v(matches: &clap::ArgMatches) -> Result<(), String> {
    let g8r_input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let add_clk_port = matches.get_one::<String>("add-clk-port").cloned();
    let flop_inputs = matches.get_flag("flop-inputs");
    let flop_outputs = matches.get_flag("flop-outputs");
    let use_system_verilog = matches.get_flag("use-system-verilog");
    let module_name_override = matches.get_one::<String>("module-name").cloned();

    if (flop_inputs || flop_outputs) && add_clk_port.is_none() {
        return Err(
            "--add-clk-port <NAME> is required when --flop-inputs or --flop-outputs is used."
                .to_string(),
        );
    }

    let gate_fn = load_gate_fn_from_path(Path::new(g8r_input_file))?;

    let final_module_name = module_name_override.as_deref().unwrap_or(&gate_fn.name);

    let netlist_str = xlsynth_g8r::aig_serdes::emit_netlist::emit_netlist(
        final_module_name,
        &gate_fn,
        flop_inputs,
        flop_outputs,
        use_system_verilog,
        add_clk_port,
    )?;

    println!("{}", netlist_str);
    Ok(())
}
