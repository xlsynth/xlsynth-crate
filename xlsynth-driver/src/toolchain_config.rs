// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ToolchainConfig {
    /// Path to the DSLX standard library.
    pub dslx_stdlib_path: Option<String>,

    /// Additional paths to use in the DSLX module search, i.e. as roots for
    /// import statements.
    pub dslx_path: Option<Vec<String>>,

    /// Directory path for the XLS toolset, e.g. codegen_main, opt_main, etc.
    pub tool_path: Option<String>,

    /// Treat warnings as errors.
    pub warnings_as_errors: Option<bool>,

    /// Enable warnings (versus the default warning set) for the given list of
    /// warning names.
    ///
    /// Enabling a warning that is already enabled in the default set is fine
    /// and has no effect.
    pub enable_warnings: Option<Vec<String>>,

    /// Disable warnings (versus the default warning set) for the given list of
    /// warning names.
    ///
    /// Disabling a warning that is already disabled in the default set is fine
    /// and has no effect.
    pub disable_warnings: Option<Vec<String>>,
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
        config.dslx_stdlib_path.clone()
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
        let dslx_path = config.dslx_path.as_deref().unwrap_or(&[]);
        Some(dslx_path.join(";"))
    } else {
        None
    }
}
