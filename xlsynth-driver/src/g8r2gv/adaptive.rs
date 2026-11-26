// SPDX-License-Identifier: Apache-2.0

//! Resolve timing libraries and write reports for adaptive resizing.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ArgMatches;
use serde::Serialize;
use sha2::{Digest, Sha256};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::liberty::io_defaults::{IoTimingConstraints, convert_io_constraints};
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::io::{ParsedNetlist, read_gv_from_str};
use xlsynth_g8r::netlist::resize::ResizeOptions;
use xlsynth_g8r::netlist::resize::adaptive::{AdaptiveResizeOptions, finish_mapped_netlist};
use xlsynth_g8r::netlist::resize::sizing_library::validate_and_filter_sizing_library;
use xlsynth_g8r::netlist::sta::StaOptions;

use crate::mapped_netlist_output::{Format, output_paths_alias, path_aliases_stream};

/// A register that cleanup later removes still makes this a registered input.
pub(super) fn select(
    matches: &ArgMatches,
    design: &SequentialGateFn,
    delay_resize: bool,
) -> anyhow::Result<bool> {
    let registered = !design.registers.is_empty()
        || matches.get_flag("flop-inputs")
        || matches.get_flag("flop-outputs");
    let adaptive = delay_resize && registered;
    for flag in ["resize-liberty-proto", "adaptive-resize-report"] {
        anyhow::ensure!(
            adaptive || !matches.contains_id(flag),
            "--{flag} requires adaptive resizing"
        );
    }
    Ok(adaptive)
}

/// Validated library inputs and settings for the finishing passes.
pub(super) struct Finish {
    alternate: Option<TimingLibrarySnapshot>,
    digest: String,
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
        // separately requested sizing library may change between
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
        let want_report = matches
            .get_one::<String>("adaptive-resize-report")
            .is_some();
        protect_inputs(matches, map_path, path)?;
        if alternate.is_some() {
            let converted = convert_io_constraints(
                map_library,
                library,
                IoTimingConstraints {
                    primary_input_transition: timing.primary_input_transition,
                    module_output_load: timing.module_output_load,
                },
            )
            .map_err(anyhow::Error::msg)
            .context("converting I/O constraints to --resize-liberty-proto units")?;
            timing.primary_input_transition = converted.primary_input_transition;
            timing.module_output_load = converted.module_output_load;
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
                validate_and_filter_sizing_library(mapped, map_library, &mut view.library)?;
                &view.library
            }
            None => map_library,
        };
        let result =
            finish_mapped_netlist(emit_netlist_as_text(mapped)?, None, library, &self.options)
                .context("adaptive sizing of the mapped netlist")?;
        let bytes = match format {
            Format::Gv => result.gate_text.into_bytes(),
            Format::Pb => format.encode(&read_gv_from_str(&result.gate_text)?)?,
        };
        #[derive(Serialize)]
        struct Report<'a, T> {
            sizing_liberty_sha256: &'a str,
            output_sha256: String,
            #[serde(flatten)]
            stages: T,
        }
        let report = if self.options.record_gate_hashes {
            let mut json = serde_json::to_vec_pretty(&Report {
                sizing_liberty_sha256: &self.digest,
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
fn protect_inputs(matches: &ArgMatches, map: &Path, sizing: &Path) -> anyhow::Result<()> {
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
    for input in [Some(map), Some(sizing), g8r, bx, link_ir]
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
        if let Some(path) = matches.get_one::<String>("adaptive-resize-report") {
            anyhow::ensure!(
                !output_paths_alias(input, Path::new(path))?,
                "--adaptive-resize-report aliases input '{}'",
                input.display()
            );
        }
    }
    Ok(())
}
