// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::get_dslx_paths;
use crate::toolchain_config::ToolchainConfig;

pub fn dslx2sv_types(
    input_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    search_paths: &[&std::path::Path],
) {
    log::info!("dslx2sv_types");
    let dslx = std::fs::read_to_string(input_file).unwrap();

    let dslx_stdlib_path = dslx_stdlib_path;

    let additional_search_path_views = search_paths;

    let mut import_data =
        xlsynth::dslx::ImportData::new(dslx_stdlib_path, additional_search_path_views);
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

    let paths = get_dslx_paths(matches, config);
    let dslx_stdlib_path = paths.stdlib_path.as_ref().map(|p| p.as_path());
    let search_views = paths.search_path_views();
    dslx2sv_types(input_path, dslx_stdlib_path, &search_views);
}
