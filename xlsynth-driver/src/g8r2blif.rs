// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::blif::emit_blif;
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;

use crate::common::write_stdout;

pub fn handle_g8r2blif(matches: &clap::ArgMatches) -> Result<(), String> {
    let input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let design = load_sequential_gate_fn_from_path(Path::new(input_file))?;
    let text = emit_blif(&design).map_err(|e| {
        format!(
            "g8r2blif error: failed to emit BLIF for {}: {}",
            input_file, e
        )
    })?;
    write_stdout(&text);
    Ok(())
}
