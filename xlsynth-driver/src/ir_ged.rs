// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_pir::edit_distance;
use xlsynth_pir::ir_parser;

use crate::toolchain_config::ToolchainConfig;

fn ir_ged(
    lhs: &std::path::Path,
    lhs_ir_top: Option<&str>,
    rhs: &std::path::Path,
    rhs_ir_top: Option<&str>,
    json: bool,
) {
    log::info!("ir_ged");

    let lhs_pkg = ir_parser::parse_path_to_package(lhs).unwrap();
    let rhs_pkg = ir_parser::parse_path_to_package(rhs).unwrap();

    let lhs_fn = match lhs_ir_top {
        Some(top) => lhs_pkg.get_fn(top).unwrap(),
        None => lhs_pkg.get_top_fn().unwrap(),
    };

    let rhs_fn = match rhs_ir_top {
        Some(top) => rhs_pkg.get_fn(top).unwrap(),
        None => rhs_pkg.get_top_fn().unwrap(),
    };

    let distance = edit_distance::compute_edit_distance(lhs_fn, rhs_fn);
    if json {
        println!("{{\"distance\": {}}}", distance);
    } else {
        println!("Distance: {}", distance);
    }
}

pub fn handle_ir_ged(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_ged");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let lhs_path = std::path::Path::new(lhs);
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let rhs_path = std::path::Path::new(rhs);
    let lhs_ir_top = matches.get_one::<String>("lhs_ir_top");
    let rhs_ir_top = matches.get_one::<String>("rhs_ir_top");
    let json = match matches.get_one::<String>("json").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    ir_ged(
        lhs_path,
        lhs_ir_top.map(|s| s.as_str()),
        rhs_path,
        rhs_ir_top.map(|s| s.as_str()),
        json,
    );
}
