// SPDX-License-Identifier: Apache-2.0

//! PIR MCMC driver.
//!
//! This binary parses an XLS IR file into PIR, runs MCMC optimization over the
//! top function using `xlsynth-mcmc-pir`, and writes the resulting optimized IR
//! back out as text.

use anyhow::Result;
use clap::Command;
use xlsynth_mcmc_pir::driver_cli::{add_pir_mcmc_args, parse_pir_mcmc_args, run_pir_mcmc_driver};

fn main() -> Result<()> {
    let _ = env_logger::try_init();

    let cmd = Command::new("pir-mcmc-driver")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("Optimize PIR IR with MCMC and emit best artifacts");
    let matches = add_pir_mcmc_args(cmd).get_matches();
    let cli = parse_pir_mcmc_args(&matches);
    run_pir_mcmc_driver(cli, |msg| eprintln!("{msg}"))
}
