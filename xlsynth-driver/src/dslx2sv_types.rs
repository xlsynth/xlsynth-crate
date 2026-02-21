// SPDX-License-Identifier: Apache-2.0

//! Execution path for the `xlsynth-driver dslx2sv-types` subcommand.
//!
//! This module parses the CLI-selected enum-case naming policy and routes it to
//! the shared `SvBridgeBuilder` so callers (for example Bazel rules) can choose
//! whether generated SV enum members are emitted as case-only symbols or
//! enum-qualified symbols.

use clap::ArgMatches;

use crate::common::get_dslx_paths;
use crate::toolchain_config::ToolchainConfig;
use xlsynth::sv_bridge_builder::SvEnumCaseNamingPolicy;

/// Converts DSLX type definitions to SV type declarations and writes the result
/// to stdout.
///
/// This function is intentionally thin: it wires file contents, import search
/// paths, and the caller-selected [`SvEnumCaseNamingPolicy`] into the shared
/// bridge implementation. Passing `Unqualified` for a module whose enums reuse
/// case names across types will surface as a generation-time collision error in
/// the builder.
pub fn dslx2sv_types(
    input_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    search_paths: &[&std::path::Path],
    enum_case_naming_policy: SvEnumCaseNamingPolicy,
) {
    log::info!("dslx2sv_types");
    let dslx = std::fs::read_to_string(input_file).unwrap();

    let dslx_stdlib_path = dslx_stdlib_path;

    let additional_search_path_views = search_paths;

    let mut import_data =
        xlsynth::dslx::ImportData::new(dslx_stdlib_path, additional_search_path_views);
    let mut builder =
        xlsynth::sv_bridge_builder::SvBridgeBuilder::with_enum_case_policy(enum_case_naming_policy);
    xlsynth::dslx_bridge::convert_leaf_module(&mut import_data, &dslx, input_file, &mut builder)
        .unwrap();
    let sv = builder.build();
    println!("{}", sv);
}

/// Handles the `dslx2sv-types` subcommand from the top-level Clap dispatch.
///
/// The CLI definition in `main.rs` requires and validates
/// `--sv_enum_case_naming_policy`, so this function can map the parsed string to
/// [`SvEnumCaseNamingPolicy`] and then delegate to [`dslx2sv_types`]. Calling
/// this with `ArgMatches` from another subcommand would panic because the code
/// unwraps/`expect`s required `dslx2sv-types` arguments.
pub fn handle_dslx2sv_types(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx2sv_types");
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    // Clap guarantees presence and allowed values for this flag in the
    // `dslx2sv-types` subcommand definition.
    let enum_case_naming_policy = match matches
        .get_one::<String>("sv_enum_case_naming_policy")
        .map(|s| s.as_str())
        .expect("clap should require sv_enum_case_naming_policy")
    {
        "unqualified" => SvEnumCaseNamingPolicy::Unqualified,
        "enum_qualified" => SvEnumCaseNamingPolicy::EnumQualified,
        other => unreachable!("unexpected clap-validated sv enum case naming policy: {other}"),
    };

    let paths = get_dslx_paths(matches, config);
    let dslx_stdlib_path = paths.stdlib_path.as_ref().map(|p| p.as_path());
    let search_views = paths.search_path_views();
    dslx2sv_types(
        input_path,
        dslx_stdlib_path,
        &search_views,
        enum_case_naming_policy,
    );
}
