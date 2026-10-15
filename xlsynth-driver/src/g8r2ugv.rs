// SPDX-License-Identifier: Apache-2.0

pub fn handle_g8r2ugv(matches: &clap::ArgMatches) -> Result<(), String> {
    let use_system_verilog = matches.get_flag("use-system-verilog");
    let design = crate::g8r_cli::load_netlist_design(matches)?;

    let netlist_str =
        xlsynth_g8r::aig_serdes::emit_netlist::emit_netlist(&design, use_system_verilog)?;

    println!("{}", netlist_str);
    Ok(())
}
