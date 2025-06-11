// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ToolchainConfig {
    /// Directory path for the XLS toolset, e.g. codegen_main, opt_main, etc.
    pub tool_path: Option<String>,
    pub dslx: Option<DslxConfig>,
    pub codegen: Option<CodegenConfig>,
}

#[derive(Deserialize, Debug)]
pub struct DslxConfig {
    pub type_inference_v2: Option<bool>,
    pub dslx_stdlib_path: Option<String>,
    pub dslx_path: Option<Vec<String>>,
    pub warnings_as_errors: Option<bool>,
    pub enable_warnings: Option<Vec<String>>,
    pub disable_warnings: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
pub struct CodegenConfig {
    pub gate_format: Option<String>,
    pub assert_format: Option<String>,
    pub use_system_verilog: Option<bool>,
}

/// Helper for extracting the DSLX standard library path from the command line
/// flag, if specified, or the toolchain config if it's present and the cmdline
/// flag isn't specified.
pub fn get_dslx_stdlib_path(
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
) -> Option<String> {
    let dslx_stdlib_path = matches.get_one::<String>("dslx_stdlib_path");
    if let Some(dslx_stdlib_path) = dslx_stdlib_path {
        Some(dslx_stdlib_path.to_string())
    } else if let Some(config) = config {
        config
            .dslx
            .as_ref()
            .and_then(|d| d.dslx_stdlib_path.clone())
    } else {
        None
    }
}

/// Helper for retrieving supplemental DSLX search paths from the command line
/// flag, if specified, or the toolchain config if it's present and the cmdline
/// flag isn't specified.
pub fn get_dslx_path(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Option<String> {
    let dslx_path = matches.get_one::<String>("dslx_path");
    if let Some(dslx_path) = dslx_path {
        Some(dslx_path.to_string())
    } else if let Some(config) = config {
        let dslx_path = config
            .dslx
            .as_ref()
            .and_then(|d| d.dslx_path.as_deref())
            .unwrap_or(&[]);
        Some(dslx_path.join(";"))
    } else {
        None
    }
}
