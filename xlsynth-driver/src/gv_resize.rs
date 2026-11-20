// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::{Arg, ArgAction, ArgMatches, Command, ValueEnum};
use xlsynth_g8r::liberty::Library;
use xlsynth_g8r::liberty::io_defaults::default_output_load;
use xlsynth_g8r::liberty::load::load_library_with_timing_data_from_path;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{read_gv_from_path, select_module};
use xlsynth_g8r::netlist::resize::{
    RecoveryCellMenu, RecoveryWindowCheckpoint, ResizeOptions, TimingCellMenu,
    resize_netlist_with_recovery_checkpoint,
};
use xlsynth_g8r::netlist::sta::{StaOptions, WireLoadModel};

/// CLI spellings of the compatible-cell selection policies used in area
/// recovery.
#[derive(Clone, Copy, ValueEnum)]
enum CellMenu {
    Area,
    ElectricalMinArea,
}

/// CLI spellings of timing nomination policies.
#[derive(Clone, Copy, ValueEnum)]
enum TimingMenu {
    NearestArea,
    DelayBandsFastestFirst,
}

/// Defines the cell shortlist controls shared by standalone and final sizing.
pub(crate) fn recovery_cell_args() -> [Arg; 3] {
    [
        Arg::new("timing-cell-menu")
            .long("timing-cell-menu")
            .value_name("POLICY")
            .value_parser(clap::value_parser!(TimingMenu))
            .help("How timing search shortlists compatible cells (default: nearest-area)"),
        Arg::new("recovery-cell-menu")
            .long("recovery-cell-menu")
            .value_name("POLICY")
            .value_parser(clap::value_parser!(CellMenu))
            .help("How area recovery shortlists compatible cells (default: area)"),
        Arg::new("recovery-cell-limit")
            .long("recovery-cell-limit")
            .value_name("COUNT")
            .value_parser(clap::value_parser!(usize))
            .help("Maximum compatible cell alternatives per site in area recovery (default: 8)"),
    ]
}

/// Preserves library defaults unless the user explicitly selected an override.
pub(crate) fn recovery_cell_options(matches: &ArgMatches) -> ResizeOptions {
    let mut options = ResizeOptions::default();
    if let Some(policy) = matches.get_one::<TimingMenu>("timing-cell-menu") {
        options.timing_cell_menu = match policy {
            TimingMenu::NearestArea => TimingCellMenu::NearestArea,
            TimingMenu::DelayBandsFastestFirst => TimingCellMenu::DelayBandsFastestFirst,
        };
    }
    if let Some(policy) = matches.get_one::<CellMenu>("recovery-cell-menu") {
        options.recovery_cell_menu = match policy {
            CellMenu::Area => RecoveryCellMenu::Area,
            CellMenu::ElectricalMinArea => RecoveryCellMenu::ElectricalMinArea,
        };
    }
    options.recovery_cell_limit = matches.get_one::<usize>("recovery-cell-limit").copied();
    options
}

/// Exposes bounded native resizing independently of the mapping frontend.
pub fn command() -> Command {
    let defaults = ResizeOptions::default();
    Command::new("gv-resize")
        .about("Improve timing then recover area by resizing compatible logic cells and FFs; emit GV on stdout")
        .arg(Arg::new("netlist").long("netlist").required(true))
        .arg(Arg::new("liberty_proto").long("liberty_proto").required(true))
        .arg(Arg::new("module_name").long("module_name"))
        .arg(Arg::new("primary_input_transition").long("primary_input_transition").help(format!("Primary-input transition [default: {}]", defaults.timing.primary_input_transition)).value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("module_output_load").long("module_output_load").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("wire-capacitance-constant-ff").long("wire-capacitance-constant-ff").value_parser(clap::value_parser!(f64)).help("Fixed wire capacitance per used cell-driven net, in fF (default: 0)"))
        .arg(Arg::new("wire-capacitance-per-sink-ff").long("wire-capacitance-per-sink-ff").value_parser(clap::value_parser!(f64)).help("Wire capacitance per sink pin, in fF (default: 0)"))
        .arg(Arg::new("max-area-growth").long("max-area-growth").help(format!("Fractional area allowance [default: {}]", defaults.max_area_growth)).value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("timing-evaluation-budget").long("timing-evaluation-budget").help("Instance timing/coverage propagation evaluations across timing rounds [default: max(50000, 4 * instance count)]").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("recovery-evaluation-budget").long("recovery-evaluation-budget").help("Hard total timing/coverage propagation visits for recovery; default starts at max(100000, 8 * instances) and renews productive registered sweeps").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("upsize-rounds").long("upsize-rounds").help(format!("Timing-search rounds, in either size direction [default: {}]", defaults.max_upsize_rounds)).value_parser(clap::value_parser!(usize)))
        .args(recovery_cell_args())
        .arg(Arg::new("monotone-recovery-checkpoint").long("monotone-recovery-checkpoint").action(ArgAction::SetTrue).help("Keep each timing boundary's tightest limit when an automatic area-recovery window renews"))
        .arg(Arg::new("downsize-rounds").long("downsize-rounds").help(format!("Area-recovery rounds [default: {}]", defaults.max_downsize_rounds)).value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("no-register-resize").long("no-register-resize").action(ArgAction::SetTrue))
        .arg(Arg::new("report-json").long("report-json").help("Write deterministic before/after timing, area and search-work statistics"))
}

/// Reads one flat mapped module and delegates all optimization to the library.
pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let library = load_library_with_timing_data_from_path(Path::new(
        matches.get_one::<String>("liberty_proto").unwrap(),
    ))
    .map_err(|e| e.to_string())?;
    let mut parsed = read_gv_from_path(Path::new(matches.get_one::<String>("netlist").unwrap()))
        .map_err(|e| e.to_string())?;
    let mut module = select_module(
        &parsed,
        matches.get_one::<String>("module_name").map(String::as_str),
    )
    .map_err(|e| e.to_string())?
    .clone();
    let options = resize_options(matches, &library)?;
    let checkpoint = if matches.get_flag("monotone-recovery-checkpoint") {
        RecoveryWindowCheckpoint::Tightest
    } else {
        RecoveryWindowCheckpoint::Latest
    };
    let report = resize_netlist_with_recovery_checkpoint(
        &mut module,
        &parsed.nets,
        &mut parsed.interner,
        &library,
        &options,
        checkpoint,
    )
    .map_err(|e| e.to_string())?;
    let verilog = emit_module_as_netlist_text(&module, &parsed.nets, &parsed.interner)
        .map_err(|e| e.to_string())?;
    if let Some(path) = matches.get_one::<String>("report-json") {
        let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("writing resizing report: {e}"))?;
    }
    print!("{verilog}");
    Ok(())
}

/// Applies only explicit CLI overrides to library-owned search defaults.
fn resize_options(matches: &ArgMatches, library: &Library) -> Result<ResizeOptions, String> {
    let defaults = recovery_cell_options(matches);
    let output_load = match matches.get_one::<f64>("module_output_load") {
        Some(load) => *load,
        None => default_output_load(library)?,
    };
    Ok(ResizeOptions {
        timing: StaOptions {
            primary_input_transition: matches
                .get_one::<f64>("primary_input_transition")
                .copied()
                .unwrap_or(defaults.timing.primary_input_transition),
            module_output_load: output_load,
            wire_load: WireLoadModel::from_femtofarads(
                matches
                    .get_one::<f64>("wire-capacitance-constant-ff")
                    .copied()
                    .unwrap_or(0.0),
                matches
                    .get_one::<f64>("wire-capacitance-per-sink-ff")
                    .copied()
                    .unwrap_or(0.0),
                library,
            )
            .map_err(|e| e.to_string())?,
        },
        max_area_growth: matches
            .get_one::<f64>("max-area-growth")
            .copied()
            .unwrap_or(defaults.max_area_growth),
        timing_evaluation_budget: matches
            .get_one::<usize>("timing-evaluation-budget")
            .copied()
            .or(defaults.timing_evaluation_budget),
        recovery_evaluation_budget: matches
            .get_one::<usize>("recovery-evaluation-budget")
            .copied()
            .or(defaults.recovery_evaluation_budget),
        max_upsize_rounds: matches
            .get_one::<usize>("upsize-rounds")
            .copied()
            .unwrap_or(defaults.max_upsize_rounds),
        max_downsize_rounds: matches
            .get_one::<usize>("downsize-rounds")
            .copied()
            .unwrap_or(defaults.max_downsize_rounds),
        resize_registers: defaults.resize_registers && !matches.get_flag("no-register-resize"),
        ..defaults
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_inherits_search_defaults_and_only_overrides_supplied_values() -> anyhow::Result<()> {
        let mut library = Library::default();
        library.default_output_load = Some(4.0);
        let defaults = ResizeOptions::default();
        let matches = command().try_get_matches_from([
            "gv-resize",
            "--netlist",
            "input.gv",
            "--liberty_proto",
            "cells.proto",
        ])?;
        let options = resize_options(&matches, &library).map_err(anyhow::Error::msg)?;
        assert_eq!(
            options.timing.primary_input_transition,
            defaults.timing.primary_input_transition
        );
        assert_eq!(options.timing.module_output_load, 4.0);
        assert_eq!(options.max_area_growth, defaults.max_area_growth);
        assert_eq!(
            options.timing_evaluation_budget,
            defaults.timing_evaluation_budget
        );
        assert_eq!(
            options.recovery_evaluation_budget,
            defaults.recovery_evaluation_budget
        );
        assert_eq!(options.max_upsize_rounds, defaults.max_upsize_rounds);
        assert_eq!(options.max_downsize_rounds, defaults.max_downsize_rounds);
        assert_eq!(options.max_candidate_sites, defaults.max_candidate_sites);
        assert_eq!(options.max_cell_candidates, defaults.max_cell_candidates);
        assert_eq!(options.timing_cell_menu, TimingCellMenu::NearestArea);
        assert_eq!(options.recovery_cell_menu, RecoveryCellMenu::Area);
        assert_eq!(options.recovery_cell_limit, None);
        assert_eq!(options.resize_registers, defaults.resize_registers);

        // An explicit output load must also work without library defaults.
        library.default_output_load = None;
        assert!(resize_options(&matches, &library).is_err());
        let matches = command().try_get_matches_from([
            "gv-resize",
            "--netlist",
            "input.gv",
            "--liberty_proto",
            "cells.proto",
            "--primary_input_transition",
            "0",
            "--module_output_load",
            "0",
            "--max-area-growth",
            "0",
            "--timing-evaluation-budget",
            "0",
            "--recovery-evaluation-budget",
            "0",
            "--upsize-rounds",
            "0",
            "--downsize-rounds",
            "0",
            "--recovery-cell-menu=electrical-min-area",
            "--recovery-cell-limit=64",
            "--no-register-resize",
        ])?;
        let options = resize_options(&matches, &library).map_err(anyhow::Error::msg)?;
        assert_eq!(options.timing.primary_input_transition, 0.0);
        assert_eq!(options.timing.module_output_load, 0.0);
        assert_eq!(options.max_area_growth, 0.0);
        assert_eq!(options.timing_evaluation_budget, Some(0));
        assert_eq!(options.recovery_evaluation_budget, Some(0));
        assert_eq!(options.max_upsize_rounds, 0);
        assert_eq!(options.max_downsize_rounds, 0);
        assert_eq!(
            options.recovery_cell_menu,
            RecoveryCellMenu::ElectricalMinArea
        );
        assert_eq!(options.recovery_cell_limit, Some(64));
        assert_eq!(options.max_cell_candidates, defaults.max_cell_candidates);
        assert!(!options.resize_registers);
        Ok(())
    }

    #[test]
    fn timing_menu_selection_reaches_resize_options() -> anyhow::Result<()> {
        for (command, input_args) in [
            (
                command(),
                ["--netlist=input.gv", "--liberty_proto=cells.proto"],
            ),
            (
                crate::g8r2gv::command(),
                ["input", "--liberty-proto=cells.proto"],
            ),
            (
                crate::bx::bx2gv_command(),
                ["input", "--liberty-proto=cells.proto"],
            ),
        ] {
            for (spelling, expected) in [
                (
                    "delay-bands-fastest-first",
                    TimingCellMenu::DelayBandsFastestFirst,
                ),
                ("nearest-area", TimingCellMenu::NearestArea),
            ] {
                let args = std::iter::once(command.get_name())
                    .chain(input_args)
                    .chain(["--timing-cell-menu", spelling]);
                let matches = command.clone().try_get_matches_from(args)?;
                assert_eq!(recovery_cell_options(&matches).timing_cell_menu, expected);
            }
        }
        Ok(())
    }
}
