// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use xlsynth_dslx_fuzzer::generate;

/// Expands a libFuzzer input into source files and both IR programs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: replay <input-bytes> <output-directory>".into());
    }
    let bytes = std::fs::read(&args[0])?;
    let sample = generate(&bytes)?;
    let directory = PathBuf::from(&args[1]);
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
