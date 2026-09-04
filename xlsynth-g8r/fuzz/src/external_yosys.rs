// SPDX-License-Identifier: Apache-2.0

//! Per-process caching policy for the shared Yosys/Liberty mapping APIs.

use std::sync::OnceLock;

use xlsynth_g8r::netlist::yosys::{YosysMappingContext, YosysMappingKind};

static EXTERNAL_YOSYS_CONTEXT: OnceLock<Result<YosysMappingContext, String>> = OnceLock::new();
static COMBO_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();
static SEQUENTIAL_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();

/// Requires the configured oracle instead of treating absent tools as a skip.
pub fn required_external_yosys_context() -> Result<&'static YosysMappingContext, &'static str> {
    EXTERNAL_YOSYS_CONTEXT
        .get_or_init(YosysMappingContext::from_env)
        .as_ref()
        .map_err(String::as_str)
}

/// Runs the library mapping/import probe once per selected mode and process.
pub fn preflight_mapping(sequential: bool) -> Result<(), String> {
    let (state, kind) = if sequential {
        (&SEQUENTIAL_PREFLIGHT, YosysMappingKind::Sequential)
    } else {
        (&COMBO_PREFLIGHT, YosysMappingKind::Combinational)
    };
    state
        .get_or_init(|| {
            required_external_yosys_context()
                .map_err(str::to_owned)?
                .preflight(kind)
                .map_err(|error| error.to_string())
        })
        .clone()
}
