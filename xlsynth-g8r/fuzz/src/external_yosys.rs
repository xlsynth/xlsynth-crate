// SPDX-License-Identifier: Apache-2.0

//! Shared external Yosys and Liberty setup for fuzz targets.

use std::sync::OnceLock;

use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::liberty_model::{Library, SequentialKind};
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_netlist_aig_with_liberty,
    load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::yosys::YosysEnvironment;

/// Parsed Liberty data and validated Yosys configuration shared by one target.
pub struct ExternalYosysContext {
    pub liberty: Library,
    pub yosys: YosysEnvironment,
}

static EXTERNAL_YOSYS_CONTEXT: OnceLock<Result<ExternalYosysContext, String>> = OnceLock::new();
static COMBO_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();
static SEQUENTIAL_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();

/// Requires the configured oracle instead of treating absent tools as a skip.
pub fn required_external_yosys_context() -> Result<&'static ExternalYosysContext, &'static str> {
    EXTERNAL_YOSYS_CONTEXT
        .get_or_init(build_external_yosys_context)
        .as_ref()
        .map_err(String::as_str)
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
    .map_err(|error| {
        format!("parse Liberty inputs configured by XLSYNTH_LIBERTY_FILES: {error}")
    })?;
    if liberty.cells.is_empty() {
        return Err(
            "XLSYNTH_LIBERTY_FILES contains no cells; supply installed standard-cell Liberty files"
                .into(),
        );
    }
    Ok(ExternalYosysContext { liberty, yosys })
}

/// Validates libraries and exercises the configured mapper/importer at startup.
pub fn preflight_mapping(sequential: bool) -> Result<(), String> {
    let state = if sequential {
        &SEQUENTIAL_PREFLIGHT
    } else {
        &COMBO_PREFLIGHT
    };
    state
        .get_or_init(|| check_mapping_setup(sequential))
        .clone()
}

/// Uses a small infrastructure probe, independently of generated fuzz samples.
fn check_mapping_setup(sequential: bool) -> Result<(), String> {
    let context = required_external_yosys_context().map_err(str::to_owned)?;
    if sequential
        && !context.liberty.cells.iter().any(|cell| {
            cell.sequential
                .iter()
                .any(|state| state.kind == SequentialKind::Ff as i32)
        })
    {
        return Err("XLSYNTH_LIBERTY_FILES has no flip-flop cells; include a sequential Liberty file for this target".into());
    }
    let source = if sequential {
        "module preflight(input clk, input a, input b, output reg q); always @(posedge clk) q <= a ^ b; endmodule\n"
    } else {
        "module preflight(input a, input b, output y); assign y = a ^ b; endmodule\n"
    };
    let netlist = if sequential {
        context
            .yosys
            .synthesize_sequential_verilog_to_gv(source, "preflight")
    } else {
        context.yosys.synthesize_verilog_to_gv(source, "preflight")
    }
    .map_err(|error| format!("Yosys/ABC startup mapping check failed: {error}"))?;
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let path = directory.path().join("mapped.gv");
    std::fs::write(&path, netlist).map_err(|error| error.to_string())?;
    let options = GvEvalOptions {
        module_name: Some("preflight".into()),
        clock_port_name: sequential.then(|| "clk".into()),
    };
    if sequential {
        load_labeled_sequential_netlist_aig_with_liberty(&path, &context.liberty, &options)
            .map(|_| ())
    } else {
        load_labeled_netlist_aig_with_liberty(&path, &context.liberty, &options).map(|_| ())
    }
    .map_err(|error| {
        format!("startup mapped-netlist import failed for XLSYNTH_LIBERTY_FILES: {error}")
    })
}
