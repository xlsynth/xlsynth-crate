// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};
use clap::{ArgMatches, parser::ValueSource};
use std::fs::File;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
use std::path::Path;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::liberty::io_defaults::default_output_load;
use xlsynth_g8r::netlist::fanout_opt::{FanoutOptimizationOptions, optimize_fanout};
use xlsynth_g8r::netlist::io::{load_liberty_from_path, load_liberty_with_timing_data_from_path};
use xlsynth_g8r::netlist::resize::{ResizeOptions, resize_netlist};
use xlsynth_g8r::netlist::sta::{StaOptions, WireLoadModel};
use xlsynth_g8r::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};
use xlsynth_g8r::techmap::delay::{
    DelayMappingLibrary, DelayMappingOptions, map_gatefn_to_delay_netlist,
};
use xlsynth_g8r::techmap::nand2::{
    CellPolicy, map_gatefn_to_structural_netlist, structural_options_from_liberty,
};

#[cfg(unix)]
use crate::mapped_netlist_output::metadata_if_exists;
use crate::mapped_netlist_output::{
    Format, PendingOutput, is_special_output, output_paths_alias, path_aliases_stream, write_stdout,
};

/// Both file outputs, opened and checked for filesystem-level aliasing.
struct OutputFiles {
    netlist: PendingOutput,
    json: PendingOutput,
}

/// Opens both destinations before truncating either, including new files.
fn open_distinct_outputs(netlist_path: &Path, json_path: &Path) -> Result<OutputFiles> {
    let netlist = PendingOutput::open(netlist_path)
        .with_context(|| format!("failed to open netlist output '{}'", netlist_path.display()))?;
    let json = PendingOutput::open(json_path)
        .with_context(|| format!("failed to open JSON summary '{}'", json_path.display()))?;
    ensure!(
        netlist.handle() != json.handle(),
        "--json_out and --output must name different files"
    );
    Ok(OutputFiles { netlist, json })
}

/// Detects an existing JSON destination that aliases the netlist's stdout.
fn ensure_distinct_from_stdout(path: &Path, stdout: &same_file::Handle) -> Result<()> {
    ensure!(
        !path_aliases_stream(path, stdout, &["/dev/stdout", "/dev/fd/1"])?,
        "--json_out must not refer to stdout when the netlist is written to stdout"
    );
    Ok(())
}

/// Detects a JSON destination that aliases the diagnostics stream.
fn ensure_distinct_from_stderr(path: &Path, stderr: &same_file::Handle) -> Result<()> {
    ensure!(
        !path_aliases_stream(path, stderr, &["/dev/stderr", "/dev/fd/2"])?,
        "--json_out must not refer to stderr"
    );
    Ok(())
}

/// Rejects a report FIFO whose reader may be waiting for stdout to close.
#[cfg(unix)]
fn ensure_json_fifo_safe_for_stdout(path: &Path, stdout: &same_file::Handle) -> Result<()> {
    if metadata_if_exists(path)?.is_some_and(|metadata| metadata.file_type().is_fifo()) {
        let kind = stdout.as_file().metadata()?.file_type();
        ensure!(
            !kind.is_fifo() && !kind.is_socket(),
            "--json_out cannot be a FIFO when stdout is a pipe or socket"
        );
    }
    Ok(())
}

/// Keeps the mapping JSON deterministic while retaining optimization work
/// counts.
fn remove_runtime(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => {
            fields.retain(|key, _| !key.ends_with("_seconds"));
            for value in fields.values_mut() {
                remove_runtime(value);
            }
        }
        serde_json::Value::Array(values) => values.iter_mut().for_each(remove_runtime),
        _ => {} // Scalar report values have no elapsed-time fields.
    }
}

/// Loads inputs, dispatches mapping and optional fanout optimization, and
/// writes outputs.
fn run_aig_tech_map(matches: &ArgMatches) -> Result<()> {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let liberty_proto_path = Path::new(matches.get_one::<String>("liberty_proto").unwrap());
    let netlist_out = matches.get_one::<String>("output").unwrap();
    let json_out = matches.get_one::<String>("json_out");
    let module_name = matches.get_one::<String>("module_name");
    let strategy = matches.get_one::<String>("strategy").unwrap().as_str();
    let fanout_opt = matches
        .get_one::<bool>("fanout_opt")
        .copied()
        .unwrap_or(strategy == "area");
    let area_recovery = matches
        .get_one::<bool>("area-recovery")
        .copied()
        .unwrap_or(strategy == "delay");
    ensure!(
        !area_recovery || strategy == "area" || strategy == "delay",
        "--area-recovery requires --strategy area or delay"
    );
    ensure!(
        area_recovery
            || ["area-recovery-rounds", "area-recovery-evaluation-budget"]
                .iter()
                .all(|name| matches.value_source(name) != Some(ValueSource::CommandLine)),
        "area recovery tuning options require --area-recovery=true"
    );
    ensure!(
        fanout_opt
            || area_recovery
            || [
                "wire-capacitance-constant-ff",
                "wire-capacitance-per-sink-ff"
            ]
            .iter()
            .all(|name| matches.value_source(name) != Some(ValueSource::CommandLine)),
        "wire capacitance options require fanout optimization or area recovery"
    );
    ensure!(
        !fanout_opt || strategy == "area" || strategy == "delay",
        "--fanout-opt requires --strategy area or delay"
    );
    ensure!(
        fanout_opt
            || area_recovery
            || strategy == "delay"
            || [
                "primary_input_transition",
                "module_output_load",
                "fanout_max_area_growth"
            ]
            .iter()
            .all(|name| matches.value_source(name) != Some(ValueSource::CommandLine)),
        "fanout tuning options require enabled area optimization; use --strategy area without --fanout-opt=false"
    );
    ensure!(
        fanout_opt
            || matches.value_source("fanout_max_area_growth") != Some(ValueSource::CommandLine),
        "--fanout-max-area-growth requires enabled fanout optimization"
    );
    ensure!(
        strategy == "nand2"
            || matches.value_source("cell_policy") != Some(ValueSource::CommandLine),
        "--cell_policy applies only to --strategy nand2"
    );
    let stderr = if let Some(json_path) = json_out {
        ensure!(
            netlist_out == "-"
                || !output_paths_alias(Path::new(json_path), Path::new(netlist_out))?,
            "--json_out and --output must name different files"
        );
        let stderr = same_file::Handle::stderr()?;
        ensure_distinct_from_stderr(Path::new(json_path), &stderr)?;
        Some(stderr)
    } else {
        None
    };
    let gate_fn =
        load_aiger_auto_from_path(Path::new(aig_input_file), GateBuilderOptions::no_opt())
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load AIG '{aig_input_file}'"))?
            .gate_fn;
    let load = if fanout_opt || area_recovery || strategy == "delay" {
        load_liberty_with_timing_data_from_path
    } else {
        load_liberty_from_path
    };
    let lib = load(liberty_proto_path).with_context(|| {
        if fanout_opt {
            format!(
                "failed to load timing-enabled Liberty proto '{}' for area fanout optimization; use --fanout-opt=false for raw mapping",
                liberty_proto_path.display()
            )
        } else {
            format!("failed to load Liberty proto '{}'", liberty_proto_path.display())
        }
    })?;

    // Resolve one model for both post-mapping stages. Delay covering still uses
    // its explicitly supplied frozen reference load, not a fanout estimate.
    let timing = if fanout_opt || area_recovery {
        Some(StaOptions {
            primary_input_transition: matches.get_one::<f64>("primary_input_transition")
                .copied().unwrap_or(FanoutOptimizationOptions::default().primary_input_transition),
            module_output_load: match matches.get_one::<f64>("module_output_load") {
                Some(load) => *load,
                None => default_output_load(&lib).map_err(anyhow::Error::msg)
                    .context("post-mapping optimization requires an output load; supply --module-output-load, or disable optimization with --fanout-opt=false --area-recovery=false")?,
            },
            wire_load: WireLoadModel::from_femtofarads(
                matches.get_one::<f64>("wire-capacitance-constant-ff").copied().unwrap_or(0.0),
                matches.get_one::<f64>("wire-capacitance-per-sink-ff").copied().unwrap_or(0.0),
                &lib)?,
        })
    } else {
        None
    };
    let (mut mapped, mut report, mut summary) = match strategy {
        "nand2" => {
            let policy = match matches.get_one::<String>("cell_policy").unwrap().as_str() {
                "small-normal-vt" => CellPolicy::SmallNormalVt,
                "max-speed" => CellPolicy::MaxSpeed,
                _ => unreachable!("cell policy is constrained by clap"),
            };
            let options = structural_options_from_liberty(&lib, policy)?;
            let mapped = map_gatefn_to_structural_netlist(&gate_fn, &options)
                .context("structural mapping failed")?;
            let summary = format!(
                "using policy='{}' INV='{}' NAND2='{}'",
                policy.as_str(),
                options.inv_cell_name,
                options.nand2_cell_name
            );
            let report = serde_json::json!({
                "cell_policy": policy.as_str(),
                "inv_cell": options.inv_cell_name, "nand2_cell": options.nand2_cell_name
            });
            (mapped, report, summary)
        }
        "delay" => {
            let options = DelayMappingOptions {
                reference_transition: *matches
                    .get_one::<f64>("primary_input_transition")
                    .ok_or_else(|| {
                        anyhow::anyhow!("delay mode requires --primary-input-transition")
                    })?,
                reference_load: *matches
                    .get_one::<f64>("module_output_load")
                    .ok_or_else(|| anyhow::anyhow!("delay mode requires --module-output-load"))?,
            };
            let library = DelayMappingLibrary::new(&lib, options)?;
            let mapped = map_gatefn_to_delay_netlist(&gate_fn, &library)?;
            let summary = format!(
                "using experimental strategy='delay' area={}",
                mapped.stats.cell_area
            );
            let report = serde_json::json!({"stats": mapped.stats,
                "timing_model": "scalar worst-edge pin delay at fixed slew/load; one state per polarity",
                "reference_transition": options.reference_transition,
                "reference_load": options.reference_load});
            (mapped.netlist, report, summary)
        }
        "area" => {
            let library = AreaMappingLibrary::new(&lib)?;
            let mapped =
                map_gatefn_to_area_netlist(&gate_fn, &library).context("area mapping failed")?;
            let summary = format!(
                "using strategy='area' area={} inverters={} duplicated_ands={}",
                mapped.stats.cell_area, mapped.stats.inverter_count, mapped.stats.duplicated_ands
            );
            let report = serde_json::json!({
                "stats": mapped.stats,
                "excluded_cells": library.excluded_cells
            });
            (mapped.netlist, report, summary)
        }
        _ => unreachable!("strategy is constrained by clap"),
    };
    if let Some(name) = module_name {
        mapped.module.name = mapped.interner.get_or_intern(name);
    }
    if fanout_opt {
        let defaults = FanoutOptimizationOptions::default();
        let options = FanoutOptimizationOptions {
            primary_input_transition: timing.unwrap().primary_input_transition,
            module_output_load: timing.unwrap().module_output_load,
            wire_load: timing.unwrap().wire_load,
            area_growth_limit: matches
                .get_one::<f64>("fanout_max_area_growth")
                .copied()
                .unwrap_or(defaults.area_growth_limit),
            ..defaults
        };
        let optimized =
            optimize_fanout(&mut mapped, &lib, options).context(
                "area fanout optimization failed; provide usable timing data or use --fanout-opt=false for raw mapping",
            )?;
        summary.push_str(&format!(
            "; fanout delay {} -> {}, area {} -> {}",
            optimized.before.worst_delay,
            optimized.after.worst_delay,
            optimized.before.area,
            optimized.after.area
        ));
        report["fanout_optimization"] = serde_json::to_value(optimized)?;
        remove_runtime(&mut report["fanout_optimization"]);
    }
    if area_recovery {
        let defaults = ResizeOptions::default();
        let options = ResizeOptions {
            timing: timing.unwrap(),
            max_area_growth: 0.0,
            max_upsize_rounds: 0,
            timing_evaluation_budget: Some(0),
            max_downsize_rounds: matches
                .get_one::<usize>("area-recovery-rounds")
                .copied()
                .unwrap_or(defaults.max_downsize_rounds),
            recovery_evaluation_budget: matches
                .get_one::<usize>("area-recovery-evaluation-budget")
                .copied(),
            ..defaults
        };
        let recovered = resize_netlist(
            &mut mapped.module,
            &mapped.nets,
            &mut mapped.interner,
            &lib,
            &options,
        )
        .context("mapped area recovery failed")?;
        summary.push_str(&format!(
            "; area recovery {} -> {}",
            recovered.initial_area, recovered.final_area
        ));
        report["area_recovery"] = serde_json::to_value(recovered)?;
        report["area_recovery_rounds"] = options.max_downsize_rounds.into();
    }
    if let Some(timing) = timing {
        report["post_mapping_timing"] = serde_json::json!({
            "primary_input_transition": timing.primary_input_transition,
            "module_output_load": timing.module_output_load,
            "wire_load": timing.wire_load
        });
    }
    report["strategy"] = strategy.into();
    report["module"] = mapped.interner.resolve(mapped.module.name).unwrap().into();
    let cell_count = mapped.module.instances.len();
    let net_count = mapped.nets.len();
    report["cell_count"] = cell_count.into();
    report["net_count"] = net_count.into();
    let format = Format::from_matches(matches, Path::new(netlist_out));
    let netlist_bytes = format.encode(&mapped.into())?;
    if let (Some(json_path), Some(stderr)) = (json_out, stderr.as_ref()) {
        ensure_distinct_from_stderr(Path::new(json_path), stderr)?;
    }
    if let Some(json_path) = json_out.filter(|_| netlist_out != "-") {
        let netlist_path = Path::new(netlist_out);
        let json_path = Path::new(json_path);
        if is_special_output(netlist_path)? || is_special_output(json_path)? {
            #[cfg(unix)]
            if metadata_if_exists(json_path)?.is_some_and(|metadata| metadata.file_type().is_fifo())
            {
                // A named path such as /dev/stdout can keep the original
                // stdout pipe open just as --output - does.
                if let Ok(stdout) = same_file::Handle::stdout() {
                    if path_aliases_stream(netlist_path, &stdout, &["/dev/stdout", "/dev/fd/1"])? {
                        ensure_json_fifo_safe_for_stdout(json_path, &stdout)?;
                    }
                }
            }
            // A consumer may wait for netlist EOF before opening the JSON
            // stream, so opening both FIFOs before writing can deadlock. Check
            // aliases again immediately before sequentially writing streams.
            // Concurrent replacement of paths cannot be made atomic here.
            ensure!(
                !output_paths_alias(netlist_path, json_path)?,
                "--json_out and --output must name different files"
            );
            std::fs::write(netlist_path, netlist_bytes)
                .with_context(|| format!("failed to write netlist output '{netlist_out}'"))?;
            let file = File::create(json_path).with_context(|| {
                format!("failed to create JSON summary '{}'", json_path.display())
            })?;
            serde_json::to_writer_pretty(file, &report).with_context(|| {
                format!("failed writing JSON summary to '{}'", json_path.display())
            })?;
        } else {
            let mut outputs = open_distinct_outputs(netlist_path, json_path)?;
            ensure!(
                Some(outputs.json.handle()) != stderr.as_ref(),
                "--json_out must not refer to stderr"
            );
            outputs
                .netlist
                .prepare_write()?
                .write_all(&netlist_bytes)
                .with_context(|| format!("failed to write netlist output '{netlist_out}'"))?;
            serde_json::to_writer_pretty(outputs.json.prepare_write()?, &report).with_context(
                || format!("failed writing JSON summary to '{}'", json_path.display()),
            )?;
        }
    } else if netlist_out == "-" {
        if let Some(json_path) = json_out {
            let json_path = Path::new(json_path);
            let stdout = same_file::Handle::stdout()?;
            ensure_distinct_from_stdout(json_path, &stdout)?;
            if is_special_output(json_path)? {
                // Stdout remains open until process exit, so a consumer
                // waiting for its EOF before opening a JSON FIFO would
                // deadlock with a blocking FIFO open here.
                #[cfg(unix)]
                ensure_json_fifo_safe_for_stdout(json_path, &stdout)?;
                write_stdout(&netlist_bytes)?;
                let file = File::create(json_path).with_context(|| {
                    format!("failed to create JSON summary '{}'", json_path.display())
                })?;
                serde_json::to_writer_pretty(file, &report).with_context(|| {
                    format!("failed writing JSON summary to '{}'", json_path.display())
                })?;
            } else {
                let mut json = PendingOutput::open(json_path).with_context(|| {
                    format!("failed to open JSON summary '{}'", json_path.display())
                })?;
                ensure!(
                    json.handle() != &stdout,
                    "--json_out must not refer to stdout when the netlist is written to stdout"
                );
                ensure!(
                    Some(json.handle()) != stderr.as_ref(),
                    "--json_out must not refer to stderr"
                );
                write_stdout(&netlist_bytes)?;
                serde_json::to_writer_pretty(json.prepare_write()?, &report).with_context(
                    || format!("failed writing JSON summary to '{}'", json_path.display()),
                )?;
            }
        } else {
            write_stdout(&netlist_bytes)?;
        }
    } else {
        std::fs::write(netlist_out, netlist_bytes)
            .with_context(|| format!("failed to write netlist output '{netlist_out}'"))?;
    }
    eprintln!(
        "aig-tech-map: mapped '{aig_input_file}' to {cell_count} instances / {net_count} nets {summary}"
    );
    Ok(())
}

/// Runs technology mapping and renders a CLI error on failure.
pub fn handle_aig_tech_map(matches: &ArgMatches) {
    if let Err(error) = run_aig_tech_map(matches) {
        eprintln!("aig-tech-map error: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::open_distinct_outputs;
    #[cfg(unix)]
    use super::{PendingOutput, ensure_distinct_from_stdout, output_paths_alias};
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::path::Path;
    #[cfg(unix)]
    use std::process::Command;
    #[cfg(unix)]
    use std::sync::mpsc;
    #[cfg(unix)]
    use std::time::Duration;

    #[cfg(unix)]
    #[test]
    fn stdout_alias_check_accepts_descriptor_vnode_identity() {
        let dir = tempfile::tempdir().unwrap();
        let unrelated = dir.path().join("unrelated");
        std::fs::write(&unrelated, "unrelated file").unwrap();
        let other_handle = same_file::Handle::from_path(&unrelated).unwrap();

        // The differing handle simulates systems where stat of /dev/stdout
        // and fstat of stdout report different identities.
        assert!(ensure_distinct_from_stdout(Path::new("/dev/stdout"), &other_handle).is_err());
    }

    #[test]
    fn opened_output_aliases_do_not_clobber_or_leave_placeholders() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output");
        assert!(open_distinct_outputs(&path, &path).is_err());
        assert!(!path.exists());

        std::fs::write(&path, "keep this output").unwrap();
        assert!(open_distinct_outputs(&path, &path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep this output");
    }

    #[test]
    fn failed_second_open_removes_only_the_new_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output");
        let invalid_path = dir.path().join("missing").join("report");
        assert!(open_distinct_outputs(&path, &invalid_path).is_err());
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_checks_do_not_require_read_permission() {
        let dir = tempfile::tempdir().unwrap();
        let netlist = dir.path().join("netlist");
        let report = dir.path().join("report");
        std::fs::write(&netlist, "netlist").unwrap();
        std::fs::write(&report, "report").unwrap();
        for path in [&netlist, &report] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o200)).unwrap();
        }
        assert!(!output_paths_alias(&netlist, &report).unwrap());
        assert!(open_distinct_outputs(&netlist, &report).is_ok());

        let placeholder_path = dir.path().join("placeholder");
        let placeholder = PendingOutput::open(&placeholder_path).unwrap();
        std::fs::set_permissions(&placeholder_path, std::fs::Permissions::from_mode(0o200))
            .unwrap();
        drop(placeholder);
        assert!(!placeholder_path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_alias_check_does_not_open_fifos() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("fifo");
        let fifo_alias = dir.path().join("fifo-alias");
        let regular = dir.path().join("report");
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        std::fs::hard_link(&fifo, &fifo_alias).unwrap();
        std::fs::write(&regular, "report").unwrap();

        let (sender, receiver) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            sender
                .send((
                    output_paths_alias(&regular, &fifo),
                    output_paths_alias(&fifo, &fifo_alias),
                ))
                .unwrap();
        });
        let (different, same) = receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("checking FIFO output paths blocked");
        assert!(!different.unwrap());
        assert!(same.unwrap());
        thread.join().unwrap();
    }
}
