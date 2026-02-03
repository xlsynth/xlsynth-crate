// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::report_cli_error::report_cli_error_and_exit;

pub fn handle_ir_mcmc_opt(matches: &ArgMatches) {
    let cli = xlsynth_mcmc_pir::driver_cli::parse_pir_mcmc_args(matches);
    if let Err(e) = xlsynth_mcmc_pir::driver_cli::run_pir_mcmc_driver(cli) {
        let message = e.to_string();
        report_cli_error_and_exit(&message, Some("ir-mcmc-opt"), vec![]);
    }
}
