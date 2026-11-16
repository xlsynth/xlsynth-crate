// SPDX-License-Identifier: Apache-2.0

//! Resolve timing libraries and optional leakage characterization for the
//! adaptive resizing passes and write their optional audit report.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ArgMatches;
use serde::Serialize;
use sha2::{Digest, Sha256};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::liberty::io_defaults::convert_io_constraints;
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::io::{ParsedNetlist, read_gv_from_str};
use xlsynth_g8r::netlist::resize::adaptive::{
    AdaptiveResizeOptions, finish_mapped_netlist, prepare_sizing_view,
};
use xlsynth_g8r::netlist::resize::{RecoveryCellLeakage, ResizeOptions};
use xlsynth_g8r::netlist::sta::StaOptions;

use crate::mapped_netlist_output::{Format, output_paths_alias, path_aliases_stream};

/// A register that cleanup later removes still makes this a registered input.
pub(super) fn select(
    matches: &ArgMatches,
    design: &SequentialGateFn,
    delay_resize: bool,
    fanout: bool,
) -> anyhow::Result<(bool, bool)> {
    let registered = !design.registers.is_empty()
        || matches.get_flag("flop-inputs")
        || matches.get_flag("flop-outputs");
    let adaptive = delay_resize && registered;
    for flag in [
        "terminal-leakage-table",
        "resize-liberty-proto",
        "adaptive-resize-report",
    ] {
        anyhow::ensure!(
            adaptive || !matches.contains_id(flag),
            "--{flag} requires adaptive resizing"
        );
    }
    let refine_trees = adaptive && fanout;
    anyhow::ensure!(
        refine_trees || !matches.contains_id("refinement-report"),
        "--refinement-report requires buffer-tree refinement"
    );
    Ok((adaptive, refine_trees))
}

/// Validated finishing inputs. One caller can map both candidate covers
/// after loading these inputs without rereading any file used for sizing.
pub(super) struct Finish {
    alternate: Option<TimingLibrarySnapshot>,
    digest: String,
    table_path: Option<PathBuf>,
    table_sha256: Option<String>,
    leakage: Option<RecoveryCellLeakage>,
    options: AdaptiveResizeOptions,
}

impl Finish {
    pub(super) fn load(
        matches: &ArgMatches,
        map_library: &Library,
        map_path: &Path,
        map_digest: &str,
        mut timing: StaOptions,
        menus: &ResizeOptions,
    ) -> anyhow::Result<Self> {
        let path = matches
            .get_one::<PathBuf>("resize-liberty-proto")
            .map(PathBuf::as_path)
            .unwrap_or(map_path);
        // Only the primary timing snapshot is pinned by the batch protocol. A
        // separately requested library and leakage table may change between
        // jobs.
        let alternate =
            if path == map_path {
                None
            } else {
                Some(TimingLibrarySnapshot::from_path(path).with_context(|| {
                    format!("loading sizing timing library '{}'", path.display())
                })?)
            };
        let (library, digest) = alternate
            .as_ref()
            .map(|s| (&s.library, s.sha256.as_str()))
            .unwrap_or((map_library, map_digest));
        let table_path = match matches.get_one::<PathBuf>("terminal-leakage-table") {
            Some(path) => Some(path.clone()),
            None => sibling_leakage_table(path, digest)?,
        };
        let want_report = matches
            .get_one::<String>("adaptive-resize-report")
            .is_some();
        protect_inputs(matches, map_path, path, table_path.as_deref())?;
        let (leakage, table_sha256) = match &table_path {
            Some(sidecar) => {
                let bytes = std::fs::read(sidecar)
                    .with_context(|| format!("reading {}", sidecar.display()))?;
                let table = RecoveryCellLeakage::from_bytes(&bytes, digest, path)
                    .with_context(|| format!("validating {}", sidecar.display()))?;
                (
                    Some(table),
                    want_report.then(|| format!("{:x}", Sha256::digest(&bytes))),
                )
            }
            None => (None, None),
        };
        if alternate.is_some() {
            (timing.primary_input_transition, timing.module_output_load) = convert_io_constraints(
                map_library,
                library,
                timing.primary_input_transition,
                timing.module_output_load,
            )
            .map_err(anyhow::Error::msg)
            .context("converting I/O constraints to --resize-liberty-proto units")?;
        }
        let mut options = AdaptiveResizeOptions {
            timing,
            recovery_cell_menu: menus.recovery_cell_menu,
            recovery_cell_limit: menus.recovery_cell_limit.expect("adaptive recovery limit"),
            record_gate_hashes: want_report,
            ..Default::default()
        };
        if matches.contains_id("timing-cell-menu") {
            options.timing_cell_menu = menus.timing_cell_menu;
        }
        let digest = digest.to_owned();
        Ok(Self {
            alternate,
            digest,
            table_path,
            table_sha256,
            leakage,
            options,
        })
    }

    pub(super) fn run(
        mut self,
        mapped: &ParsedNetlist,
        map_library: &Library,
        format: Format,
    ) -> anyhow::Result<(Vec<u8>, Option<Vec<u8>>)> {
        let library = match &mut self.alternate {
            Some(view) => {
                prepare_sizing_view(mapped, map_library, &mut view.library)?;
                &view.library
            }
            None => map_library,
        };
        let result = finish_mapped_netlist(
            emit_netlist_as_text(mapped)?,
            None,
            library,
            &self.options,
            self.leakage.as_ref(),
        )
        .context("adaptive sizing of the mapped netlist")?;
        let bytes = match format {
            Format::Gv => result.gate_text.into_bytes(),
            Format::Pb => format.encode(&read_gv_from_str(&result.gate_text)?)?,
        };
        #[derive(Serialize)]
        struct Report<'a, T> {
            sizing_liberty_sha256: &'a str,
            terminal_leakage_status: &'static str,
            terminal_leakage_table: Option<&'a Path>,
            terminal_leakage_table_sha256: Option<String>,
            output_sha256: String,
            #[serde(flatten)]
            stages: T,
        }
        let report = if self.options.record_gate_hashes {
            let mut json = serde_json::to_vec_pretty(&Report {
                sizing_liberty_sha256: &self.digest,
                terminal_leakage_status: if self.leakage.is_some() {
                    "applied"
                } else {
                    "missing_table"
                },
                terminal_leakage_table: self.table_path.as_deref(),
                terminal_leakage_table_sha256: self.table_sha256,
                output_sha256: format!("{:x}", Sha256::digest(&bytes)),
                stages: result.report,
            })?;
            json.push(b'\n');
            Some(json)
        } else {
            None
        };
        Ok((bytes, report))
    }
}

/// A report or netlist can be named freely, including via a symlink, as long
/// as writing it leaves the characterized library and circuit inputs intact.
fn protect_inputs(
    matches: &ArgMatches,
    map: &Path,
    sizing: &Path,
    table: Option<&Path>,
) -> anyhow::Result<()> {
    let g8r = matches
        .try_get_one::<String>("g8r_input_file")
        .ok()
        .flatten()
        .map(Path::new);
    let bx = matches
        .try_get_one::<PathBuf>("input")
        .ok()
        .flatten()
        .map(PathBuf::as_path);
    let link_ir = matches
        .try_get_one::<PathBuf>("link-ir")
        .ok()
        .flatten()
        .map(PathBuf::as_path);
    for input in [Some(map), Some(sizing), table, g8r, bx, link_ir]
        .into_iter()
        .flatten()
    {
        let output = matches.get_one::<PathBuf>("output").unwrap();
        let alias = if output.as_os_str() == "-" {
            path_aliases_stream(
                input,
                &same_file::Handle::stdout()?,
                &["/dev/stdout", "/dev/fd/1"],
            )?
        } else {
            output_paths_alias(input, output)?
        };
        anyhow::ensure!(!alias, "--output aliases input '{}'", input.display());
        for flag in ["refinement-report", "adaptive-resize-report"] {
            if let Some(path) = matches.get_one::<String>(flag) {
                anyhow::ensure!(
                    !output_paths_alias(input, Path::new(path))?,
                    "--{flag} aliases input '{}'",
                    input.display()
                );
            }
        }
    }
    Ok(())
}

/// Published tables use the SHA-256 of the exact (possibly compressed) timing
/// proto. A present but broken link or unreadable file must surface as an
/// error.
fn sibling_leakage_table(library: &Path, digest: &str) -> anyhow::Result<Option<PathBuf>> {
    let dir = library.parent().unwrap_or(Path::new("."));
    let name = format!("{digest}.json");
    for path in [dir.join(&name), dir.join("sidecars").join(&name)] {
        match std::fs::symlink_metadata(&path) {
            Ok(_) => return Ok(Some(path)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("locating {}", path.display())),
        }
    }
    Ok(None)
}
