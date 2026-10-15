// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::Duration;

use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_g8r::liberty::io_defaults::{default_output_load_ff, resolve_reference_buffer};
use xlsynth_g8r::liberty::load::load_library_from_path;
use xlsynth_g8r::netlist::abc::{
    AbcIoConstraints, AbcMappingContext, AbcMappingOptions, AbcToolchain,
};
use xlsynth_g8r::netlist::register_mapping::RegisterCellSelection;

/// Defines the thin CLI surface for direct ABC/native sequential mapping.
pub fn command() -> Command {
    Command::new("g8r2gv")
        .about("Maps a .g8r or .g8rbin design to a complete gate-level Verilog netlist on stdout using ABC and native register mapping, without Yosys")
        .arg(Arg::new("g8r_input_file").required(true).index(1).help("Input .g8r or .g8rbin design"))
        .arg(Arg::new("liberty").long("liberty").value_name("PATH").required(true).action(ArgAction::Append)
            .help("Raw uncompressed Liberty file for ABC; repeat in loading order. The first one or two files must provide the usable combinational library"))
        .arg(Arg::new("liberty-proto").long("liberty-proto").value_name("PATH").required(true)
            .help("Matching Liberty proto (.proto, .textproto, optionally .gz) for native register mapping and IO defaults"))
        .arg(Arg::new("abc-path").long("abc-path").value_name("PATH")
            .help("ABC executable; defaults to XLSYNTH_ABC_PATH or abc on PATH"))
        .arg(Arg::new("abc-script").long("abc-script").value_name("PATH")
            .help("Trusted ABC optimization/mapping passes; defaults to ABC_SPEED_SCRIPT. Library, input, constraint loading and final export remain in the wrapper"))
        .arg(Arg::new("driving-cell").long("driving-cell").value_name("CELL")
            .help("Input driver override; defaults to the proto's reference_buffer_cell"))
        .arg(Arg::new("output-load-ff").long("output-load-ff").value_name("FF")
            .value_parser(clap::value_parser!(f64)).allow_hyphen_values(true)
            .help("Output capacitance override in femtofarads; defaults to the proto's default_output_load converted to fF"))
        .arg(Arg::new("timeout-seconds").long("timeout-seconds").value_name("SECONDS").default_value("180")
            .value_parser(clap::value_parser!(u64).range(1..)).help("ABC mapping process timeout"))
        .arg(Arg::new("cleanup-registers").long("cleanup-registers").action(ArgAction::SetTrue)
            .help("Opt in to native register cleanup before mapping; otherwise preserve all registers"))
        .arg(Arg::new("register-cell").long("register-cell").value_name("CELL")
            .help("Require a specific supported FF cell; otherwise select the smallest-area compatible FF"))
        .arg(Arg::new("add-clk-port").long("add-clk-port").value_name("NAME")
            .help("Declare a clock for a clockless design, or check the name of the stored clock"))
        .arg(Arg::new("flop-inputs").long("flop-inputs").action(ArgAction::SetTrue)
            .help("Add input boundary registers; requires a stored clock or --add-clk-port"))
        .arg(Arg::new("flop-outputs").long("flop-outputs").action(ArgAction::SetTrue)
            .help("Add output boundary registers; requires a stored clock or --add-clk-port"))
        .arg(Arg::new("module-name").long("module-name").value_name("NAME").help("Override the output module name"))
        .arg(Arg::new("diagnostics-out").long("diagnostics-out").value_name("PATH")
            .help("Write successful-run diagnostics as JSON, including resolved constraints, the composed ABC script and captured output"))
}

/// Loads inputs, resolves independent IO overrides, and emits the mapped
/// netlist.
pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let design = crate::g8r_cli::load_netlist_design(matches)?;
    let library_path = matches.get_one::<String>("liberty-proto").unwrap();
    let library = load_library_from_path(Path::new(library_path)).map_err(|e| e.to_string())?;
    // Resolve only omitted fields: explicit overrides must not require the
    // corresponding proto defaults (or capacitance units) to be valid.
    let driving_cell = match matches.get_one::<String>("driving-cell") {
        Some(cell) => cell.clone(),
        None => resolve_reference_buffer(&library)?.to_owned(),
    };
    let output_load_ff = match matches.get_one::<f64>("output-load-ff") {
        Some(load) => *load,
        None => default_output_load_ff(&library)?,
    };
    let mut options = AbcMappingOptions::new(AbcIoConstraints {
        driving_cell,
        output_load_ff,
    });
    options.timeout = Duration::from_secs(*matches.get_one::<u64>("timeout-seconds").unwrap());
    options.cleanup_registers = matches.get_flag("cleanup-registers");
    if let Some(cell) = matches.get_one::<String>("register-cell") {
        options.register_selection = RegisterCellSelection::FixedCell(cell.clone());
    }
    if let Some(path) = matches.get_one::<String>("abc-script") {
        options.script = std::fs::read_to_string(path)
            .map_err(|e| format!("reading ABC script '{path}': {e}"))?;
    }
    let toolchain = match matches.get_one::<String>("abc-path") {
        Some(path) => AbcToolchain::new(path),
        None => AbcToolchain::from_env(),
    }
    .map_err(|e| e.to_string())?;
    let raw_files = matches
        .get_many::<String>("liberty")
        .unwrap()
        .map(Path::new)
        .collect::<Vec<_>>();
    let context =
        AbcMappingContext::new(toolchain, &raw_files, library).map_err(|e| e.to_string())?;
    let mapped = context
        .map_sequential(&design, &options)
        .map_err(|e| e.to_string())?;
    let verilog = mapped.emit_verilog()?;
    if let Some(path) = matches.get_one::<String>("diagnostics-out") {
        let json = serde_json::to_string_pretty(&mapped.diagnostics).map_err(|e| e.to_string())?;
        std::fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("writing ABC diagnostics '{path}': {e}"))?;
    }
    print!("{verilog}");
    Ok(())
}
