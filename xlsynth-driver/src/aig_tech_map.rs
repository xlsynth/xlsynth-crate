// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};
use clap::{ArgMatches, parser::ValueSource};
use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, MetadataExt};
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::load_liberty_from_path;
use xlsynth_g8r::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};
use xlsynth_g8r::techmap::nand2::{
    CellPolicy, map_gatefn_to_structural_netlist, structural_options_from_liberty,
};

/// Resolves existing output targets or their parents before either file is
/// written.
fn resolved_output_path(path: &Path) -> Result<PathBuf> {
    match path.canonicalize() {
        Ok(resolved) => Ok(resolved),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            // A dangling file symlink still identifies its eventual write
            // target.
            if path
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink())
            {
                let target = std::fs::read_link(path)?;
                return resolved_output_path(&parent.join(target));
            }
            let name = path.file_name().context("output path has no file name")?;
            Ok(parent.canonicalize()?.join(name))
        }
        Err(error) => Err(error.into()),
    }
}

/// Detects equal destinations, including symlinks, hard links, and
/// not-yet-created files.
fn output_paths_alias(lhs: &Path, rhs: &Path) -> Result<bool> {
    if resolved_output_path(lhs)? == resolved_output_path(rhs)? {
        return Ok(true);
    }
    let Some(lhs_metadata) = metadata_if_exists(lhs)? else {
        return Ok(false);
    };
    let Some(rhs_metadata) = metadata_if_exists(rhs)? else {
        return Ok(false);
    };
    #[cfg(unix)]
    {
        Ok(same_unix_file(&lhs_metadata, &rhs_metadata))
    }
    #[cfg(not(unix))]
    {
        // Opening special files to compare their identities can block or have
        // side effects. Regular files are also checked using their output
        // handles before either is written.
        if !lhs_metadata.is_file() || !rhs_metadata.is_file() {
            return Ok(false);
        }
        match (identity_handle(lhs), identity_handle(rhs)) {
            (Ok(lhs_handle), Ok(rhs_handle)) => Ok(lhs_handle == rhs_handle),
            // Identity checking is best effort here; both outputs are checked
            // again after opening them for writing when both are regular.
            _ => Ok(false),
        }
    }
}

/// Reads metadata without treating a missing output as an error.
fn metadata_if_exists(path: &Path) -> std::io::Result<Option<std::fs::Metadata>> {
    match std::fs::metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Identifies existing stream-like outputs, excluding invalid directories.
fn is_special_output(path: &Path) -> std::io::Result<bool> {
    Ok(metadata_if_exists(path)?.is_some_and(|metadata| !metadata.is_file() && !metadata.is_dir()))
}

/// Compares Unix file identities without opening either file's data.
#[cfg(unix)]
fn same_unix_file(lhs: &std::fs::Metadata, rhs: &std::fs::Metadata) -> bool {
    (lhs.dev(), lhs.ino()) == (rhs.dev(), rhs.ino())
}

/// Opens a file for identity queries without requesting read or write access.
#[cfg(windows)]
fn identity_handle(path: &Path) -> std::io::Result<same_file::Handle> {
    let file = OpenOptions::new().access_mode(0).open(path)?;
    same_file::Handle::from_file(file)
}

/// Opens a file for identity queries on other supported platforms.
#[cfg(not(any(unix, windows)))]
fn identity_handle(path: &Path) -> std::io::Result<same_file::Handle> {
    same_file::Handle::from_path(path)
}

/// Checks that a placeholder still names the file that was created.
fn path_matches_handle(path: &Path, handle: &same_file::Handle) -> bool {
    #[cfg(unix)]
    {
        match (std::fs::metadata(path), handle.as_file().metadata()) {
            (Ok(path_metadata), Ok(handle_metadata)) => {
                same_unix_file(&path_metadata, &handle_metadata)
            }
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
            && identity_handle(path).is_ok_and(|path_handle| path_handle == *handle)
    }
}

/// An output opened without truncation, with cleanup for a new placeholder.
struct PendingOutput {
    handle: Option<same_file::Handle>,
    created_path: Option<PathBuf>,
}

impl PendingOutput {
    /// Opens a destination without destroying existing contents.
    fn open(path: &Path) -> Result<Self> {
        // Resolving dangling symlinks lets us record the actual newly created
        // target for cleanup without deleting the symlink itself.
        let resolved = resolved_output_path(path)?;
        let (file, created) = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&resolved)
        {
            Ok(file) => (file, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                (OpenOptions::new().write(true).open(&resolved)?, false)
            }
            Err(error) => return Err(error.into()),
        };
        let handle = match same_file::Handle::from_file(file) {
            Ok(handle) => handle,
            Err(error) => {
                if created {
                    let _ = std::fs::remove_file(&resolved);
                }
                return Err(error.into());
            }
        };
        Ok(Self {
            handle: Some(handle),
            created_path: created.then_some(resolved),
        })
    }

    /// Keeps the destination and truncates it immediately before writing.
    fn prepare_write(&mut self) -> Result<&mut File> {
        self.created_path = None;
        let file = self.handle.as_mut().unwrap().as_file_mut();
        if file.metadata()?.is_file() {
            file.set_len(0)?;
        }
        Ok(file)
    }
}

impl Drop for PendingOutput {
    fn drop(&mut self) {
        if let Some(path) = self.created_path.take() {
            let handle = self.handle.take().unwrap();
            let still_same_file = path_matches_handle(&path, &handle);
            drop(handle);
            if still_same_file {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

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
        netlist.handle != json.handle,
        "--json_out and --netlist_out must name different files"
    );
    Ok(OutputFiles { netlist, json })
}

/// Detects an existing path that aliases a standard output stream.
fn path_aliases_stream(
    path: &Path,
    stream: &same_file::Handle,
    #[cfg_attr(not(unix), allow(unused_variables))] stream_paths: &[&str],
) -> Result<bool> {
    let Some(metadata) = metadata_if_exists(path)? else {
        return Ok(false);
    };
    #[cfg(unix)]
    let aliases = {
        // On some Unix systems, stat of a /dev/fd entry reports the identity
        // of its descriptor vnode rather than the identity returned by fstat
        // on the descriptor itself. Compare both forms without opening a
        // potentially blocking FIFO or another special output.
        same_unix_file(&metadata, &stream.as_file().metadata()?)
            || stream_paths.iter().any(|stream_path| {
                std::fs::metadata(stream_path)
                    .is_ok_and(|stream_metadata| same_unix_file(&metadata, &stream_metadata))
            })
    };
    #[cfg(not(unix))]
    let aliases = metadata.is_file() && identity_handle(path).is_ok_and(|handle| handle == *stream);
    Ok(aliases)
}

/// Detects an existing JSON destination that aliases the netlist's stdout.
fn ensure_distinct_from_stdout(path: &Path, stdout: &same_file::Handle) -> Result<()> {
    ensure!(
        !path_aliases_stream(path, stdout, &["/dev/stdout", "/dev/fd/1"])?,
        "--json_out must not refer to stdout when --netlist_out is '-'"
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

/// Loads the requested inputs, dispatches mapping, and writes netlist/report
/// outputs.
fn run_aig_tech_map(matches: &ArgMatches) -> Result<()> {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let liberty_proto_path = Path::new(matches.get_one::<String>("liberty_proto").unwrap());
    let netlist_out = matches.get_one::<String>("netlist_out").unwrap();
    let json_out = matches.get_one::<String>("json_out");
    let module_name = matches.get_one::<String>("module_name");
    let strategy = matches.get_one::<String>("strategy").unwrap().as_str();
    ensure!(
        strategy != "area" || matches.value_source("cell_policy") != Some(ValueSource::CommandLine),
        "--cell_policy applies only to --strategy nand2"
    );
    let stderr = if let Some(json_path) = json_out {
        ensure!(
            netlist_out == "-"
                || !output_paths_alias(Path::new(json_path), Path::new(netlist_out))?,
            "--json_out and --netlist_out must name different files"
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
    let lib = load_liberty_from_path(liberty_proto_path).with_context(|| {
        format!(
            "failed to load Liberty proto '{}'",
            liberty_proto_path.display()
        )
    })?;

    let (mut mapped, mut report, summary) = match strategy {
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
                "mapper": "nand2", "cell_policy": policy.as_str(),
                "inv_cell": options.inv_cell_name, "nand2_cell": options.nand2_cell_name
            });
            (mapped, report, summary)
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
                "mapper": "g8r-tm", "stats": mapped.stats,
                "excluded_cells": library.excluded_cells
            });
            (mapped.netlist, report, summary)
        }
        _ => unreachable!("strategy is constrained by clap"),
    };
    if let Some(name) = module_name {
        mapped.module.name = mapped.interner.get_or_intern(name);
    }
    report["strategy"] = strategy.into();
    report["module"] = mapped.interner.resolve(mapped.module.name).unwrap().into();
    report["cell_count"] = mapped.module.instances.len().into();
    report["net_count"] = mapped.nets.len().into();
    let netlist_text = emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)
        .context("failed to emit mapped netlist text")?;
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
                // stdout pipe open just as --netlist_out - does.
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
                "--json_out and --netlist_out must name different files"
            );
            std::fs::write(netlist_path, netlist_text)
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
                outputs.json.handle.as_ref() != stderr.as_ref(),
                "--json_out must not refer to stderr"
            );
            outputs
                .netlist
                .prepare_write()?
                .write_all(netlist_text.as_bytes())
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
                print!("{netlist_text}");
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
                    json.handle.as_ref() != Some(&stdout),
                    "--json_out must not refer to stdout when --netlist_out is '-'"
                );
                ensure!(
                    json.handle.as_ref() != stderr.as_ref(),
                    "--json_out must not refer to stderr"
                );
                print!("{netlist_text}");
                serde_json::to_writer_pretty(json.prepare_write()?, &report).with_context(
                    || format!("failed writing JSON summary to '{}'", json_path.display()),
                )?;
            }
        } else {
            print!("{netlist_text}");
        }
    } else {
        std::fs::write(netlist_out, netlist_text)
            .with_context(|| format!("failed to write netlist output '{netlist_out}'"))?;
    }
    eprintln!(
        "aig-tech-map: mapped '{}' to {} instances / {} nets {}",
        aig_input_file,
        mapped.module.instances.len(),
        mapped.nets.len(),
        summary
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
