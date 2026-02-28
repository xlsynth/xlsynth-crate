// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::ArgMatches;
use xlsynth_autocov::{
    resolve_entry_fn_from_ir_path, run_ir_fn_autocov_with_writers, IrFnAutocovRunConfig,
};

use crate::common::parse_bool_flag_or;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_fn_autocov(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_input_file = PathBuf::from(
        matches
            .get_one::<String>("ir_input_file")
            .expect("ir_input_file is required"),
    );
    let corpus_file = PathBuf::from(
        matches
            .get_one::<String>("corpus_file")
            .expect("corpus_file is required"),
    );
    let seed = *matches
        .get_one::<u64>("seed")
        .expect("seed has a default value");
    let max_iters = matches.get_one::<u64>("max_iters").copied();
    let max_corpus_len = matches.get_one::<usize>("max_corpus_len").copied();
    let progress_every = *matches
        .get_one::<u64>("progress_every")
        .expect("progress_every has a default value");
    let threads = matches.get_one::<usize>("threads").copied();
    let seed_two_hot_max_bits = *matches
        .get_one::<usize>("seed_two_hot_max_bits")
        .expect("seed_two_hot_max_bits has a default value");
    let no_mux_space = parse_bool_flag_or(matches, "no_mux_space", false);
    let seed_structured = parse_bool_flag_or(matches, "seed_structured", true);

    let entry_fn = match resolve_entry_fn_from_ir_path(
        &ir_input_file,
        matches.get_one::<String>("ir_top").map(String::as_str),
    ) {
        Ok(entry_fn) => entry_fn,
        Err(e) => {
            report_cli_error_and_exit(&e, Some("ir-fn-autocov"), vec![]);
        }
    };

    let run_config = IrFnAutocovRunConfig {
        ir_input_file,
        entry_fn,
        corpus_file,
        seed,
        max_iters,
        max_corpus_len,
        progress_every,
        no_mux_space,
        threads,
        seed_structured,
        seed_two_hot_max_bits,
        install_signal_handlers: true,
    };

    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let mut stdout = stdout.lock();
    let mut stderr = stderr.lock();

    if let Err(e) = run_ir_fn_autocov_with_writers(&run_config, &mut stdout, &mut stderr) {
        report_cli_error_and_exit(&e, Some("ir-fn-autocov"), vec![]);
    }
}
