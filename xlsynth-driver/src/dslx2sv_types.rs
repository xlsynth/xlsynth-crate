// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};

pub fn dslx2sv_types(
    input_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    dslx_path: Option<&str>,
) {
    log::info!("dslx2sv_types");
    let dslx = std::fs::read_to_string(input_file).unwrap();

    let dslx_stdlib_path_buf: Option<std::path::PathBuf> =
        dslx_stdlib_path.map(|s| std::path::Path::new(s).to_path_buf());
    let dslx_stdlib_path = dslx_stdlib_path_buf.as_ref().map(|p| p.as_path());

    let mut additional_search_path_bufs: Vec<std::path::PathBuf> = vec![];
    if let Some(dslx_path) = dslx_path {
        for path in dslx_path.split(';') {
            additional_search_path_bufs.push(std::path::Path::new(path).to_path_buf());
        }
    }

    // We need the `Path` view type instead of `PathBuf`.
    let additional_search_path_views: Vec<&std::path::Path> = additional_search_path_bufs
        .iter()
        .map(|p| p.as_path())
        .collect::<Vec<_>>();

    let mut import_data =
        xlsynth::dslx::ImportData::new(dslx_stdlib_path, &additional_search_path_views);
    let mut builder = xlsynth::sv_bridge_builder::SvBridgeBuilder::new();
    xlsynth::dslx_bridge::convert_leaf_module(&mut import_data, &dslx, input_file, &mut builder)
        .unwrap();
    let sv = builder.build();
    println!("{}", sv);
}

pub fn handle_dslx2sv_types(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx2sv_types");
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path: Option<std::path::PathBuf> =
        dslx_stdlib_path.map(|s| std::path::Path::new(&s).to_path_buf());
    let dslx_stdlib_path = dslx_stdlib_path.as_ref().map(|p| p.as_path());

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    // Stub function for DSLX to SV type conversion
    dslx2sv_types(input_path, dslx_stdlib_path, dslx_path);
}
