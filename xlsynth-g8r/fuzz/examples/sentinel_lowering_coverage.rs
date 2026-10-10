// SPDX-License-Identifier: Apache-2.0

//! Replay a sentinel-lowering corpus and require completed semantic coverage.

use std::path::PathBuf;

use xlsynth_g8r_fuzz::sentinel_lowering::validation_inputs;
use xlsynth_g8r_fuzz::sentinel_lowering::{Coverage, check_input};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--write-corpus") {
        if args.len() != 2 {
            return Err("usage: --write-corpus DIRECTORY".into());
        }
        let directory = PathBuf::from(&args[1]);
        std::fs::create_dir_all(&directory)?;
        for (index, input) in validation_inputs().iter().enumerate() {
            std::fs::write(directory.join(format!("case-{index:03}")), input)?;
        }
        return Ok(());
    }
    let inputs = if args.is_empty() {
        validation_inputs()
    } else {
        let mut paths = Vec::new();
        for path in args.iter().map(PathBuf::from) {
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let path = entry?.path();
                    if path.is_file() {
                        paths.push(path);
                    }
                }
            } else {
                paths.push(path);
            }
        }
        paths.sort();
        paths
            .iter()
            .map(std::fs::read)
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut coverage = Coverage::default();
    for (index, input) in inputs.iter().enumerate() {
        coverage.accumulate(
            check_input(input)
                .map_err(|error| format!("replay input {index}, bytes={input:02x?}: {error}"))?,
        );
    }
    println!("{coverage:#?}");
    coverage.validate()?;
    Ok(())
}
