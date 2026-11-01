// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Arg, ArgGroup, ArgMatches, Command, value_parser};
use xlsynth_g8r::stitch_equiv::{StitchVectorResult, StitchVectors};
use xlsynth_pir::{IrValuesFile, parse_ir_values_file};

use crate::dslx_stitch_pipeline::resolve_stitch_spec;
use crate::stitch_equiv::design_args;

/// Defines the concrete DSLX versus mapped gate check.
pub fn command() -> Command {
    Command::new("stitch-check-vectors")
        .about("Checks concrete input rows against the behavioral DSLX stages and mapped gates")
        .args(design_args())
        .arg(
            Arg::new("input-irvals")
                .long("input-irvals")
                .value_name("PATH")
                .value_parser(value_parser!(PathBuf))
                .help("One named packed input record per cycle"),
        )
        .arg(
            Arg::new("random-samples")
                .long("random-samples")
                .value_name("COUNT")
                .value_parser(value_parser!(NonZeroUsize))
                .help("Generate this many input rows"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("SEED")
                .requires("random-samples")
                .value_parser(value_parser!(u64))
                .help("Seed for random input generation (default 0)"),
        )
        .group(
            ArgGroup::new("vectors")
                .args(["input-irvals", "random-samples"])
                .required(true),
        )
}

/// Runs the selected stimulus and reports the first concrete disagreement.
pub fn handle(matches: &ArgMatches) -> Result<()> {
    let module = matches.get_one::<String>("module-name").map(String::as_str);
    let seed = matches.get_one::<u64>("seed").copied().unwrap_or(0);
    let random_count = matches.get_one::<NonZeroUsize>("random-samples");
    let vectors = if let Some(count) = random_count {
        StitchVectors::Random {
            count: *count,
            seed,
        }
    } else {
        let path = matches.get_one::<PathBuf>("input-irvals").unwrap();
        match parse_ir_values_file(path)? {
            IrValuesFile::NamedValueSequence(rows) => StitchVectors::Named(rows),
            IrValuesFile::ValueSequence(rows) if rows.is_empty() => bail!("no input rows"),
            IrValuesFile::ValueSequence(_) => bail!("--input-irvals requires named records"),
        }
    };
    let seed_note = random_count.map_or_else(String::new, |_| format!(" Random seed {seed}."));
    let reference = resolve_stitch_spec(matches.get_one::<PathBuf>("spec").unwrap())?;
    match reference.check_vectors_against_gv(
        matches.get_one::<PathBuf>("netlist").unwrap(),
        matches.get_one::<PathBuf>("liberty-proto").unwrap(),
        module,
        vectors,
    )? {
        StitchVectorResult::Checked(counts) => println!(
            "Concrete check passed for {} (input rows {}, data comparisons {}, output-valid comparisons {}).{seed_note}",
            module.unwrap_or(reference.name()),
            counts.rows,
            counts.data,
            counts.valid,
        ),
        StitchVectorResult::Mismatch(mismatch) => {
            if let Some(count) = random_count {
                bail!("{mismatch}\nReplay with --random-samples {count} --seed {seed}");
            }
            bail!("{mismatch}");
        }
    }
    Ok(())
}
