// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

pub fn handle_aig2v(matches: &clap::ArgMatches) -> Result<(), String> {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let module_name = matches.get_one::<String>("module-name").unwrap();
    let add_clk_port = matches.get_one::<String>("add-clk-port").cloned();
    let flop_inputs = matches.get_flag("flop-inputs");
    let flop_outputs = matches.get_flag("flop-outputs");
    let use_system_verilog = matches.get_flag("use-system-verilog");

    if (flop_inputs || flop_outputs) && add_clk_port.is_none() {
        return Err(
            "--add-clk-port <NAME> is required when --flop-inputs or --flop-outputs is used."
                .to_string(),
        );
    }

    let gate_fn = load_aig_gate_fn(Path::new(aig_input_file))?;
    let netlist_str = xlsynth_g8r::aig_serdes::emit_netlist::emit_netlist(
        module_name,
        &gate_fn,
        flop_inputs,
        flop_outputs,
        use_system_verilog,
        add_clk_port,
    )?;

    println!("{}", netlist_str);
    Ok(())
}
