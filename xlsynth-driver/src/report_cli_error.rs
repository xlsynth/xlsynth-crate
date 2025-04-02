// SPDX-License-Identifier: Apache-2.0

use colored::Colorize;

pub fn report_cli_error_and_exit(
    message: &str,
    subcommand: Option<&str>,
    details: Vec<(&str, &str)>,
) -> ! {
    let subcommand_str = if let Some(subcommand) = subcommand {
        format!("{}: ", subcommand)
    } else {
        String::new()
    };
    eprintln!("xlsynth-driver: {}{}", subcommand_str, message.red().bold());
    for (key, value) in details {
        eprintln!("  {}: {}", key, value);
    }
    std::process::exit(1);
}
