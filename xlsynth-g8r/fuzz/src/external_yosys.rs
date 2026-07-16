// SPDX-License-Identifier: Apache-2.0

//! Shared external Yosys and Liberty setup for fuzz targets.

use std::sync::OnceLock;

use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::yosys::YosysEnvironment;

/// Parsed Liberty data and validated Yosys configuration shared by one target.
pub struct ExternalYosysContext {
    pub liberty: Library,
    pub yosys: YosysEnvironment,
}

static EXTERNAL_YOSYS_CONTEXT: OnceLock<Result<ExternalYosysContext, String>> = OnceLock::new();
static SKIP_REPORTED: OnceLock<()> = OnceLock::new();

/// Returns the external Yosys context, or reports one infrastructure skip.
///
/// Missing Yosys or Liberty files are environmental conditions rather than
/// properties of an individual fuzz sample, so callers should early-return
/// when this returns None.
pub fn external_yosys_context() -> Option<&'static ExternalYosysContext> {
    match EXTERNAL_YOSYS_CONTEXT.get_or_init(build_external_yosys_context) {
        Ok(context) => Some(context),
        Err(error) => {
            if SKIP_REPORTED.set(()).is_ok() {
                eprintln!("skipping external Yosys/Liberty fuzz target: {error}");
            }
            None
        }
    }
}

fn build_external_yosys_context() -> Result<ExternalYosysContext, String> {
    let yosys = YosysEnvironment::from_env()?;
    let liberty = parse_liberty_files_with_payload_options(
        yosys.liberty_files().paths(),
        LibertyPayloadOptions {
            include_timing: false,
            include_power: false,
        },
    )
    .map_err(|error| format!("parse Liberty inputs: {error}"))?;
    Ok(ExternalYosysContext { liberty, yosys })
}
