// SPDX-License-Identifier: Apache-2.0

//! Reports a corpus or reproducible generated samples, optionally checking
//! semantics.

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_codegen_fuzz::coverage::{CoverageReport, Outcome};
use xlsynth_codegen_fuzz::{
    Profile,
    input::{FuzzCase, mark_versioned},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut profile = Profile::NativeSemantics;
    let mut corpus = None;
    let mut write_corpus = None;
    let mut samples = 1000usize;
    let mut seed = 1u64;
    let mut byte_count = 2048usize;
    let mut check = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                profile = match args.next().as_deref() {
                    Some("native") => Profile::NativeSemantics,
                    Some("stock-xls") => Profile::StockXls,
                    _ => return Err("--profile requires native or stock-xls".into()),
                }
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(
                    args.next().ok_or("--corpus needs a file or directory")?,
                ))
            }
            "--write-corpus" => {
                write_corpus = Some(PathBuf::from(
                    args.next().ok_or("--write-corpus needs a directory")?,
                ))
            }
            "--samples" => samples = args.next().ok_or("--samples needs a count")?.parse()?,
            "--seed" => seed = args.next().ok_or("--seed needs an integer")?.parse()?,
            "--bytes" => byte_count = args.next().ok_or("--bytes needs a count")?.parse()?,
            "--check" => check = true,
            "--help" => {
                println!(
                    "coverage [--profile native|stock-xls] [--corpus PATH | --samples N --seed N --bytes N] [--check] [--write-corpus DIR]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if check {
        xlsynth_codegen_fuzz::preflight::validate(profile.into())?;
    }
    let mut paths = Vec::new();
    if let Some(path) = &corpus {
        if path.is_file() {
            paths.push(path.clone());
        } else {
            collect_inputs(path, &mut paths)?;
        }
    }
    paths.sort();
    if let Some(path) = &write_corpus {
        std::fs::create_dir_all(path)?;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut report = CoverageReport::default();
    for index in 0..if corpus.is_some() {
        paths.len()
    } else {
        samples
    } {
        let data = if corpus.is_some() {
            std::fs::read(&paths[index])?
        } else {
            let mut bytes = vec![0; byte_count];
            rng.fill_bytes(&mut bytes);
            mark_versioned(&mut bytes);
            bytes
        };
        // Save before checking so a failing sample is directly replayable.
        if let Some(path) = &write_corpus {
            std::fs::write(path.join(blake3::hash(&data).to_hex().as_str()), &data)?;
        }
        let case = match FuzzCase::decode(&data, profile) {
            Ok(case) => case,
            Err(reason) if reason == "unsupported block fuzz input format version" => {
                // Rejected format versions never describe a generated block.
                *report.rejected_inputs.entry(reason).or_default() += 1;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let checked = check.then(|| case.check());
        let outcome = if let Some(checked) = &checked {
            checked.coverage_outcome()
        } else {
            Outcome::GeneratedOnly
        };
        let trace = checked.as_ref().and_then(|c| c.checked_trace());
        report.record_case(&case, outcome, trace);
    }
    println!("{}", report.json(profile));
    Ok(())
}

/// Ignores symlinks so a corpus census stays within its explicitly supplied
/// tree.
fn collect_inputs(path: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_inputs(&entry.path(), paths)?;
        } else if kind.is_file() {
            paths.push(entry.path());
        }
    }
    Ok(())
}
