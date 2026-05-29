// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::blif::load_blif_from_path;
use xlsynth_g8r::aig_serdes::g8r::emit_g8r;

use crate::common::write_stdout;

pub fn handle_blif2g8r(matches: &clap::ArgMatches) -> Result<(), String> {
    let input_file = matches.get_one::<String>("blif_input_file").unwrap();
    let design = load_blif_from_path(Path::new(input_file))?;
    write_stdout(&emit_g8r(&design));
    Ok(())
}
