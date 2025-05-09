// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::{extract_codegen_flags, extract_pipeline_spec, CodegenFlags, PipelineSpec};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_codegen_pipeline;

pub fn handle_ir2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = extract_pipeline_spec(matches);

    let codegen_flags = extract_codegen_flags(matches, config.as_ref());

    ir2pipeline(
        input_path,
        delay_model,
        &pipeline_spec,
        &codegen_flags,
        config,
    );
}

/// To convert an IR file to a pipeline we run the codegen_main command and give
/// it a number of pipeline stages.
fn ir2pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir2pipeline");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_codegen_pipeline(
            input_file,
            delay_model,
            pipeline_spec,
            codegen_flags,
            tool_path,
        );
        println!("{}", output);
    } else {
        todo!("ir2pipeline subcommand using runtime APIs")
    }
}
