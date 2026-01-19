// SPDX-License-Identifier: Apache-2.0

//! CLI wrapper for `xlsynth_pir::aug_opt`.
//!
//! Intent: "opt_main"-like usage for debugging and corpus scans. For end-user
//! workflows, prefer wiring this into `xlsynth-driver`.

use std::io::Read;

use xlsynth_pir::{AugOptOptions, run_aug_opt_over_ir_text};

fn usage_and_exit(msg: Option<&str>) -> ! {
    if let Some(m) = msg {
        eprintln!("error: {m}\n");
    }
    eprintln!(
        "usage: xlsynth-pir-aug-opt <input_ir|- > --top <NAME> [--rounds <N>]\n\
         \n\
         Examples:\n\
           xlsynth-pir-aug-opt input.ir --top cone > out.ir\n\
           cat input.ir | xlsynth-pir-aug-opt - --top cone --rounds 1 > out.ir\n"
    );
    std::process::exit(2);
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
    let mut args: Vec<String> = std::env::args().collect();
    let _prog = args.remove(0);

    if args.is_empty() {
        usage_and_exit(None);
    }

    let input_ir = args.remove(0);
    let mut top: Option<String> = None;
    let mut rounds: usize = 1;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--top" => {
                if i + 1 >= args.len() {
                    usage_and_exit(Some("--top requires a value"));
                }
                top = Some(args[i + 1].clone());
                i += 2;
            }
            "--rounds" => {
                if i + 1 >= args.len() {
                    usage_and_exit(Some("--rounds requires a value"));
                }
                rounds = args[i + 1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| usage_and_exit(Some("--rounds must be an integer")));
                i += 2;
            }
            "--help" | "-h" => usage_and_exit(None),
            other => usage_and_exit(Some(&format!("unknown argument: {other}"))),
        }
    }

    let top = top.unwrap_or_else(|| usage_and_exit(Some("--top is required")));

    let input_text = match read_ir_text(&input_ir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let out = match run_aug_opt_over_ir_text(
        &input_text,
        Some(top.as_str()),
        AugOptOptions {
            enable: true,
            rounds,
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
