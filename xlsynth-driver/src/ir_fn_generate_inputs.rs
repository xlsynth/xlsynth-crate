// SPDX-License-Identifier: Apache-2.0

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::ArgMatches;
use xlsynth_pir::ir_fn_generate_inputs::{
    generate_ir_fn_inputs_from_ir_path, FloatParamSpec, IrFnGenerateInputsConfig,
};

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_fn_generate_inputs(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_input_file = PathBuf::from(
        matches
            .get_one::<String>("ir_input_file")
            .expect("ir_input_file is required"),
    );
    let count = *matches
        .get_one::<usize>("count")
        .expect("count has a default value");
    let seed = *matches
        .get_one::<u64>("seed")
        .expect("seed has a default value");
    let float_params = matches
        .get_many::<String>("float_param")
        .into_iter()
        .flatten()
        .map(|spec| {
            FloatParamSpec::parse(spec).unwrap_or_else(|e| {
                report_cli_error_and_exit(&e, Some("ir-fn-generate-inputs"), vec![])
            })
        })
        .collect::<Vec<_>>();
    let config = IrFnGenerateInputsConfig {
        count,
        seed,
        float_params,
    };
    let tuples = generate_ir_fn_inputs_from_ir_path(
        &ir_input_file,
        matches.get_one::<String>("ir_top").map(String::as_str),
        &config,
    )
    .unwrap_or_else(|e| report_cli_error_and_exit(&e, Some("ir-fn-generate-inputs"), vec![]));

    let stdout = std::io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    for tuple in tuples {
        writeln!(writer, "{}", tuple).unwrap_or_else(|e| {
            report_cli_error_and_exit(
                &format!("failed to write generated input tuple: {}", e),
                Some("ir-fn-generate-inputs"),
                vec![],
            )
        });
    }
}
