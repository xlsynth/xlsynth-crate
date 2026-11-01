// SPDX-License-Identifier: Apache-2.0

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::path::PathBuf;
use xlsynth_dslx_fuzzer::generate;
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;

/// Exports generated or saved fuzz inputs for the standalone C++ interpreter.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let typecheck_only = args.last().is_some_and(|arg| arg == "--typecheck-only");
    if typecheck_only {
        args.pop();
    }
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: coverage_corpus <output-directory> <random-sample-count> [corpus-directory] [--typecheck-only]"
                .into(),
        );
    }
    let directory = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    let random_count: usize = args[1].to_str().ok_or("invalid sample count")?.parse()?;
    let mut data_sets = Vec::new();
    if let Some(corpus) = args.get(2) {
        let mut paths = std::fs::read_dir(corpus)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort();
        for path in paths {
            if path.is_file() {
                data_sets.push(std::fs::read(path)?);
            }
        }
    }
    let mut graph_rng = StdRng::from_seed([37; 32]);
    for _ in 0..random_count {
        let mut data = vec![0; 1024];
        graph_rng.fill_bytes(&mut data);
        data_sets.push(data);
    }
    if data_sets.is_empty() {
        return Err("no input samples were selected".into());
    }
    let mut value_rng = StdRng::from_seed([41; 32]);
    let mut manifest = String::new();
    for (index, data) in data_sets.iter().enumerate() {
        let sample = generate(data)?;
        let reference = sample.expected_ir()?;
        let top = reference.get_top_fn().ok_or("reference has no top")?;
        let modules = if typecheck_only {
            sample.sources()?
        } else {
            let vectors = generate_mixed_argument_sets_with_rng(top, &mut value_rng, 16);
            sample.interpreter_sources(&vectors)?
        };
        let sample_directory = directory.join(format!("sample_{index:05}"));
        std::fs::create_dir_all(&sample_directory)?;
        std::fs::write(sample_directory.join("input.bin"), data)?;
        for module in modules {
            std::fs::write(
                sample_directory.join(format!("{}.x", module.name)),
                module.text,
            )?;
        }
        manifest.push_str(
            sample_directory
                .join("sample.x")
                .to_str()
                .ok_or("non-UTF8 source path")?,
        );
        manifest.push('\n');
    }
    std::fs::write(directory.join("manifest.txt"), manifest)?;
    println!(
        "Exported {} samples ({}) to {}",
        data_sets.len(),
        if typecheck_only {
            "constexpr only"
        } else {
            "16 argument vectors each"
        },
        directory.display()
    );
    Ok(())
}
