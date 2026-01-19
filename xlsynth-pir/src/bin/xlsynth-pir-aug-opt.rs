// SPDX-License-Identifier: Apache-2.0

//! CLI wrapper for `xlsynth_pir::aug_opt`.
//!
//! Intent: "opt_main"-like usage for debugging and corpus scans. For end-user
//! workflows, prefer wiring this into `xlsynth-driver`.

use std::io::Read;

use clap::ArgAction;
use clap::Parser;
use xlsynth_pir::{AugOptOptions, run_aug_opt_over_ir_text};

#[derive(Debug, Parser)]
#[command(
    name = "xlsynth-pir-aug-opt",
    about = "Co-recursive augmented optimization: libxls opt → PIR rewrite → libxls opt.",
    version
)]
struct Args {
    /// Input IR package path. Use '-' to read from stdin.
    #[arg(value_name = "INPUT_IR", default_value = "-")]
    input_ir: String,

    /// Top function name (required).
    #[arg(long, value_name = "NAME")]
    top: String,

    /// Number of co-recursive rounds (default 1).
    #[arg(long, value_name = "N", default_value_t = 1)]
    rounds: usize,

    /// Whether to run libxls optimization after the PIR rewrite step (default
    /// true).
    ///
    /// When set to false, this helps inspect the direct effect of PIR rewrites
    /// without libxls re-canonicalization.
    #[arg(
        long,
        value_name = "BOOL",
        default_value = "true",
        value_parser = ["true", "false"],
        num_args = 1,
        action = ArgAction::Set
    )]
    run_xlsynth_opt_after: String,
}

fn read_ir_text(input: &str) -> Result<String, String> {
    let mut s = String::new();
    if input == "-" {
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        return Ok(s);
    }
    let mut f = std::fs::File::open(input).map_err(|e| format!("failed to open {input}: {e}"))?;
    f.read_to_string(&mut s)
        .map_err(|e| format!("failed to read {input}: {e}"))?;
    Ok(s)
}

fn main() {
    let args = Args::parse();

    let input_text = match read_ir_text(&args.input_ir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let out = match run_aug_opt_over_ir_text(
        &input_text,
        Some(args.top.as_str()),
        AugOptOptions {
            enable: true,
            rounds: args.rounds,
            run_xlsynth_opt_before: true,
            run_xlsynth_opt_after: args.run_xlsynth_opt_after == "true",
        },
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    print!("{out}");
}
