// SPDX-License-Identifier: Apache-2.0

use std::fs::{File, Metadata, OpenOptions};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use clap::builder::ArgPredicate;
use clap::parser::ValueSource;
use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::liberty::io_defaults::{default_output_load_ff, resolve_reference_buffer};
use xlsynth_g8r::liberty::load::load_library_from_path;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::abc::{
    AbcIoConstraints, AbcMappedDesign, AbcMappingContext, AbcMappingOptions, AbcRunDiagnostics,
    AbcToolchain,
};
use xlsynth_g8r::netlist::register_mapping::RegisterCellSelection;
use xlsynth_g8r::netlist::sequential_mapping::{
    SequentialMappingOptions, map_sequential_to_area_netlist,
};

#[cfg(unix)]
use crate::mapped_netlist_output::same_unix_file;
use crate::mapped_netlist_output::{
    Format, PendingOutput, ensure_pb_handle_distinct_from_stderr, format_arg, metadata_if_exists,
    output_arg, output_paths_alias, path_aliases_stream, reset_regular_output, write_stdout,
};

/// Defines the CLI for native or ABC combinational mapping and native register
/// mapping.
pub fn command() -> Command {
    Command::new("g8r2gv")
        .about("Maps a .g8r or .g8rbin design to a gate-level netlist with native or ABC combinational mapping")
        .arg(output_arg())
        .arg(format_arg())
        .arg(Arg::new("g8r_input_file").required(true).index(1).help("Input .g8r or .g8rbin design"))
        .arg(Arg::new("mapper").long("mapper").value_name("MAPPER").value_parser(["native", "abc"])
            .default_value_if("liberty", ArgPredicate::IsPresent, "abc").default_value("native")
            .help("Combinational mapper: native area mapping by default, ABC when raw --liberty is supplied"))
        .arg(Arg::new("liberty").long("liberty").value_name("PATH").required_if_eq("mapper", "abc").action(ArgAction::Append)
            .help("Raw uncompressed Liberty file for ABC; repeat in loading order. The first one or two files must provide the usable combinational library"))
        .arg(Arg::new("liberty-proto").long("liberty-proto").value_name("PATH").required(true)
            .help("Liberty proto (.proto, .textproto, optionally .gz) for native combinational and register mapping, or matching ABC library metadata"))
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
        .arg(Arg::new("clock-gating").long("clock-gating").value_name("BOOL")
            .value_parser(clap::value_parser!(bool)).default_value("true")
            .help("Extract integrated clock gates for shared flop enables (default: true)"))
        .arg(Arg::new("clock-gate-min-flops").long("clock-gate-min-flops").value_name("N")
            .value_parser(clap::value_parser!(NonZeroUsize))
            .help("Minimum number of flop bits sharing a clock-gate enable (default: 4)"))
        .arg(Arg::new("clock-gate-max-enable-depth").long("clock-gate-max-enable-depth").value_name("N")
            .value_parser(clap::value_parser!(usize))
            .help("Maximum AND depth of a clock-gate enable, inclusive (default: 8)"))
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
            .help("Write deterministic successful-run metadata as JSON, including resolved constraints and the composed ABC script; excludes raw tool logs"))
        .arg(Arg::new("abc-stdout-out").long("abc-stdout-out").value_name("PATH")
            .help("Write raw ABC mapping stdout on success; may contain nondeterministic timing and progress output"))
        .arg(Arg::new("abc-stderr-out").long("abc-stderr-out").value_name("PATH")
            .help("Write raw ABC mapping stderr on success; may contain nondeterministic tool output"))
}

/// Pins the netlist destination while the sidecars are being written. Opening
/// a named stream waits until its turn so its reader can consume the sidecars
/// first.
enum NetlistOutput {
    Stdout,
    File(PendingOutput),
    Stream(Metadata),
}

impl NetlistOutput {
    /// Reserves regular files and records the identity of a deferred stream.
    fn open(path: &Path, pb_stderr: Option<&same_file::Handle>) -> anyhow::Result<Self> {
        if path.as_os_str() == "-" {
            return Ok(Self::Stdout);
        }
        if let Some(metadata) = metadata_if_exists(path)?
            && !metadata.is_file()
            && !metadata.is_dir()
        {
            // Use the inherited descriptor for /dev/stdout even when stat of
            // its descriptor vnode hides the redirected regular file.
            #[cfg(unix)]
            if std::fs::metadata("/dev/fd/1").is_ok_and(|fd| same_unix_file(&metadata, &fd)) {
                if let Some(stderr) = pb_stderr {
                    ensure_pb_handle_distinct_from_stderr(&same_file::Handle::stdout()?, stderr)?;
                }
                return Ok(Self::Stdout);
            }
            return Ok(Self::Stream(metadata));
        }
        let pending = PendingOutput::open(path)?;
        if let Some(stderr) = pb_stderr {
            ensure_pb_handle_distinct_from_stderr(pending.handle(), stderr)?;
        }
        Ok(Self::File(pending))
    }

    /// Compares the opened sidecar to the netlist before the sidecar is
    /// truncated.
    fn aliases_file(&self, file: &File, metadata: &Metadata) -> anyhow::Result<bool> {
        #[cfg(not(unix))]
        if !metadata.is_file() {
            // Windows supports identity handles for regular files.
            return Ok(false);
        }
        let sidecar = same_file::Handle::from_file(file.try_clone()?)?;
        Ok(match self {
            Self::Stdout => sidecar == same_file::Handle::stdout()?,
            Self::File(pending) => &sidecar == pending.handle(),
            #[cfg(unix)]
            Self::Stream(expected) => same_unix_file(expected, metadata),
            #[cfg(not(unix))]
            Self::Stream(_) => false, // A regular sidecar cannot be the deferred stream.
        })
    }

    /// Writes the netlist after the sidecars have been closed.
    fn write(
        self,
        path: &Path,
        bytes: &[u8],
        pb_stderr: Option<&same_file::Handle>,
    ) -> anyhow::Result<()> {
        match self {
            Self::Stdout => write_stdout(bytes),
            Self::File(mut pending) => Ok(pending.prepare_write()?.write_all(bytes)?),
            Self::Stream(expected) => {
                let mut file = OpenOptions::new().write(true).open(path)?;
                let actual = file.metadata()?;
                #[cfg(unix)]
                // A descriptor vnode can have a different stat identity from
                // the pipe or terminal returned by fstat on the opened file.
                let unchanged = same_unix_file(&expected, &actual)
                    || (!actual.is_file()
                        && !actual.is_dir()
                        && same_unix_file(&expected, &std::fs::metadata(path)?));
                #[cfg(not(unix))]
                let unchanged = expected.file_type() == actual.file_type();
                anyhow::ensure!(unchanged, "netlist stream '{}' changed", path.display());
                // Windows identity handles are available only for regular
                // files.
                if let Some(stderr) = pb_stderr
                    && (cfg!(unix) || actual.is_file())
                {
                    let netlist = same_file::Handle::from_file(file.try_clone()?)?;
                    ensure_pb_handle_distinct_from_stderr(&netlist, stderr)?;
                }
                Ok(file.write_all(bytes)?)
            }
        }
    }
}

/// Opens without truncating so a concurrent path replacement cannot make a
/// sidecar overwrite the netlist between preflight and writing.
fn write_sidecar(
    path: &Path,
    bytes: &[u8],
    flag: &str,
    netlist: &NetlistOutput,
) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let metadata = file.metadata()?;
    anyhow::ensure!(
        !netlist.aliases_file(&file, &metadata)?,
        "--{flag} and --output must name different destinations"
    );
    if metadata.is_file() {
        reset_regular_output(&mut file)?;
    }
    Ok(file.write_all(bytes)?)
}

/// Checks and writes the ABC sidecars before the netlist.
fn write_abc_sidecars(
    matches: &ArgMatches,
    output_path: &Path,
    netlist: &NetlistOutput,
    diagnostics: &AbcRunDiagnostics,
) -> anyhow::Result<()> {
    for flag in ["diagnostics-out", "abc-stdout-out", "abc-stderr-out"] {
        let Some(path) = matches.get_one::<String>(flag).map(Path::new) else {
            continue;
        };
        let aliases = if matches!(netlist, NetlistOutput::Stdout) {
            path_aliases_stream(
                path,
                &same_file::Handle::stdout()?,
                &["/dev/stdout", "/dev/fd/1"],
            )
        } else {
            output_paths_alias(output_path, path)
        }
        .with_context(|| {
            format!(
                "checking --{flag} '{}' against the netlist output",
                path.display()
            )
        })?;
        anyhow::ensure!(
            !aliases,
            "--{flag} and --output must name different destinations"
        );
    }
    if let Some(path) = matches.get_one::<String>("diagnostics-out") {
        let json = serde_json::to_string_pretty(diagnostics)?;
        write_sidecar(
            Path::new(path),
            format!("{json}\n").as_bytes(),
            "diagnostics-out",
            netlist,
        )
        .with_context(|| format!("writing ABC diagnostics '{path}'"))?;
    }
    for (flag, log) in [
        ("abc-stdout-out", &diagnostics.stdout),
        ("abc-stderr-out", &diagnostics.stderr),
    ] {
        if let Some(path) = matches.get_one::<String>(flag) {
            write_sidecar(Path::new(path), log.as_bytes(), flag, netlist)
                .with_context(|| format!("writing --{flag} '{path}'"))?;
        }
    }
    Ok(())
}

/// Maps the design with the selected backend and emits the requested format.
pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let output_path = matches.get_one::<PathBuf>("output").unwrap();
    let format = Format::from_matches(matches, output_path);
    let pb_stderr = format
        .check_output_stderr(output_path)
        .map_err(|e| format!("{e:#}"))?;
    let native = matches.get_one::<String>("mapper").unwrap() == "native";
    if native
        && let Some(flag) = [
            "liberty",
            "abc-path",
            "abc-script",
            "driving-cell",
            "output-load-ff",
            "timeout-seconds",
            "diagnostics-out",
            "abc-stdout-out",
            "abc-stderr-out",
        ]
        .into_iter()
        .find(|flag| matches.value_source(flag) == Some(ValueSource::CommandLine))
    {
        return Err(format!(
            "--{flag} requires the ABC mapper (--mapper abc with --liberty)"
        ));
    }
    let design = crate::g8r_cli::load_netlist_design(matches)?;
    let library_path = matches.get_one::<String>("liberty-proto").unwrap();
    let library = load_library_from_path(Path::new(library_path)).map_err(|e| e.to_string())?;
    let mut sequential_options = SequentialMappingOptions {
        cleanup_registers: matches.get_flag("cleanup-registers"),
        ..Default::default()
    };
    if !matches.get_flag("clock-gating") {
        sequential_options.clock_gating = None;
    }
    if let Some(gating) = &mut sequential_options.clock_gating {
        if let Some(min_flops) = matches.get_one::<NonZeroUsize>("clock-gate-min-flops") {
            gating.min_flops = min_flops.get();
        }
        if let Some(max_depth) = matches.get_one::<usize>("clock-gate-max-enable-depth") {
            gating.max_enable_depth = *max_depth;
        }
    }
    if let Some(cell) = matches.get_one::<String>("register-cell") {
        sequential_options.register_selection = RegisterCellSelection::FixedCell(cell.clone());
    }
    let (mapped, diagnostics) = if native {
        let mapped = map_sequential_to_area_netlist(&design, &library, &sequential_options)
            .map_err(|e| format!("{e:#}"))?;
        (mapped, None)
    } else {
        let mapped = map_with_abc(matches, &design, library, &sequential_options, format)?;
        (mapped.netlist, Some(mapped.diagnostics))
    };
    let bytes = format.encode(&mapped).map_err(|e| format!("{e:#}"))?;
    let netlist = NetlistOutput::open(output_path, pb_stderr.as_ref())
        .with_context(|| format!("opening mapped netlist '{}'", output_path.display()))
        .map_err(|e| format!("{e:#}"))?;
    if let Some(diagnostics) = diagnostics {
        write_abc_sidecars(matches, output_path, &netlist, &diagnostics)
            .map_err(|e| format!("{e:#}"))?;
    }
    netlist
        .write(output_path, &bytes, pb_stderr.as_ref())
        .with_context(|| format!("writing mapped netlist '{}'", output_path.display()))
        .map_err(|e| format!("{e:#}"))
}

/// Resolves ABC's IO constraints and toolchain, then maps the design.
fn map_with_abc(
    matches: &ArgMatches,
    design: &SequentialGateFn,
    library: Library,
    sequential_options: &SequentialMappingOptions,
    format: Format,
) -> Result<AbcMappedDesign, String> {
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
    let mut abc_options = AbcMappingOptions::new(AbcIoConstraints {
        driving_cell,
        output_load_ff,
    });
    abc_options.timeout = Duration::from_secs(*matches.get_one::<u64>("timeout-seconds").unwrap());
    if let Some(path) = matches.get_one::<String>("abc-script") {
        abc_options.script = std::fs::read_to_string(path)
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
    match format {
        Format::Gv => context.map_sequential(design, sequential_options, &abc_options),
        Format::Pb => context.map_sequential_structural(design, sequential_options, &abc_options),
    }
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{NetlistOutput, output_paths_alias, write_sidecar};
    use std::fs;

    #[test]
    fn raw_log_replaced_after_preflight_cannot_overwrite_netlist() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let output = dir.path().join("mapped.pb");
        let log = dir.path().join("abc.stdout");
        let original_log = dir.path().join("original.stdout");
        fs::write(&output, b"existing netlist")?;
        fs::write(&log, b"existing log")?;
        let netlist = NetlistOutput::open(&output, None)?;
        assert!(!output_paths_alias(&output, &log)?);

        // A diagnostics FIFO can delay the raw-log write after preflight.
        fs::rename(&log, &original_log)?;
        fs::hard_link(&output, &log)?;
        let error = write_sidecar(&log, b"new log", "abc-stdout-out", &netlist).unwrap_err();
        assert_eq!(
            error.to_string(),
            "--abc-stdout-out and --output must name different destinations"
        );
        assert_eq!(fs::read(output)?, b"existing netlist");
        assert_eq!(fs::read(original_log)?, b"existing log");
        Ok(())
    }
}
