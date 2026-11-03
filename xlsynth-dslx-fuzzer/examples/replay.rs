// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, generate_with_grammar};

/// Expands a libFuzzer input into source files and both IR programs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let (grammar, input, output) = match args.as_slice() {
        [input, output] => (Grammar::Mission, input, output),
        [flag, input, output] if flag == "--local" => (Grammar::MissionLocal, input, output),
        [flag, input, output] if flag == "--mission-v1" => (Grammar::MissionV1, input, output),
        [flag, input, output] if flag == "--mission-v2" => (Grammar::MissionV2, input, output),
        [flag, input, output] if flag == "--mission-local-v2" => (Grammar::MissionLocalV2, input, output),
        [flag, input, output] if flag == "--mission-v4" => (Grammar::MissionV4, input, output),
        [flag, input, output] if flag == "--mission-local-v4" => (Grammar::MissionLocalV4, input, output),
        [flag, input, output] if flag == "--experimental" => (Grammar::Experimental, input, output),
        _ => return Err(
            "usage: replay [--local|--mission-v1|--mission-v2|--mission-local-v2|--mission-v4|--mission-local-v4|--experimental] <input-bytes> <output-directory>"
                .into(),
        ),
    };
    let bytes = std::fs::read(input)?;
    let sample = generate_with_grammar(&bytes, GenerateOptions::default(), grammar)?;
    let directory = PathBuf::from(output);
    std::fs::create_dir_all(&directory)?;
    // Write the independent side first so converter failures retain a
    // reproducer.
    for module in sample.sources()? {
        std::fs::write(directory.join(format!("{}.x", module.name)), module.text)?;
    }
    std::fs::write(
        directory.join("expected.ir"),
        sample.expected_ir()?.to_string(),
    )?;
    let prepared = sample.prepare()?;
    std::fs::write(directory.join("converted.ir"), prepared.converted_ir)?;
    Ok(())
}
