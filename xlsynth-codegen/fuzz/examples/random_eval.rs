// SPDX-License-Identifier: Apache-2.0

//! Fresh PRNG-driven graphs using the same oracles as coverage-guided fuzzing.

use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use xlsynth_codegen_fuzz::{
    Profile,
    coverage::{CoverageReport, Outcome},
    input::{FuzzCase, GENERATOR_VERSION},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = Profile::NativeSemantics;
    let mut seed = 1u64;
    let mut case_seed = None;
    let mut samples = u64::MAX;
    let mut duration = 3600u64;
    let mut directory = None;
    let mut check = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                profile = match args.next().as_deref() {
                    Some("native") => Profile::NativeSemantics,
                    Some("stock-xls") => Profile::StockXls,
                    _ => return Err("--profile requires native or stock-xls".into()),
                }
            }
            "--seed" => seed = args.next().ok_or("missing seed")?.parse()?,
            "--case-seed" => {
                case_seed = Some(args.next().ok_or("missing case seed")?.parse::<u64>()?)
            }
            "--samples" => samples = args.next().ok_or("missing sample count")?.parse()?,
            "--duration" => duration = args.next().ok_or("missing duration in seconds")?.parse()?,
            "--artifact-dir" => {
                directory = Some(PathBuf::from(
                    args.next().ok_or("missing artifact directory")?,
                ))
            }
            "--no-check" => check = false,
            "--help" => {
                println!(
                    "random_eval --profile native|stock-xls [--seed N | --case-seed N] [--samples N] [--duration SECONDS] [--artifact-dir PATH] [--no-check]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if case_seed.is_some() {
        samples = 1;
    }
    // Validate before generating samples, including zero-sample runs.
    if check {
        xlsynth_codegen_fuzz::preflight::validate(profile.into())?;
    }
    if let Some(dir) = &directory {
        std::fs::create_dir_all(dir)?;
    }
    let started = Instant::now();
    let mut last_report = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut report = CoverageReport::default();
    for index in 0..samples {
        if started.elapsed() >= Duration::from_secs(duration) {
            break;
        }
        let this_seed = case_seed.unwrap_or_else(|| rng.next_u64());
        if let Some(dir) = &directory {
            xlsynth_codegen_fuzz::input::clear_case_artifacts(dir)?;
            // Save before graph generation so even a generator panic is replayable.
            let progress = serde_json::json!({"generator_version":GENERATOR_VERSION,
                "profile":profile.name(), "worker_seed":seed, "case_seed":this_seed,
                "case_index":index, "started_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64()});
            std::fs::write(dir.join("progress.tmp"), progress.to_string())?;
            std::fs::rename(dir.join("progress.tmp"), dir.join("progress.json"))?;
        }
        let case = FuzzCase::random(this_seed, profile);
        let checked = check.then(|| case.check_with_artifacts(directory.as_deref()));
        let outcome = match &checked {
            None => Outcome::GeneratedOnly,
            Some(c) => c.coverage_outcome(),
        };
        let trace = checked.as_ref().and_then(|c| c.checked_trace());
        let inconclusive = matches!(outcome, Outcome::Inconclusive(_));
        report.record_case(&case, outcome, trace);
        if inconclusive
            || index == 0
            || (index + 1) % 4096 == 0
            || last_report.elapsed() >= Duration::from_secs(30)
        {
            println!("codegen-coverage {}", report.json(profile));
            last_report = Instant::now();
        }
    }
    println!("codegen-coverage {}", report.json(profile));
    println!("stat::number_of_executed_units: {}", report.generated_cases);
    Ok(())
}
