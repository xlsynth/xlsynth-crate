// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_fn_eval(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_path = matches.get_one::<String>("ir_file").unwrap();
    let entry_fn = matches.get_one::<String>("entry_fn").unwrap();
    let arg_tuple = matches.get_one::<String>("arg_tuple").unwrap();

    let package = match xlsynth::IrPackage::parse_ir_from_path(std::path::Path::new(ir_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to parse IR file: {}", e);
            std::process::exit(1);
        }
    };

    let func = match package.get_function(entry_fn) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to get function '{}': {}", entry_fn, e);
            std::process::exit(1);
        }
    };

    let args_value = match xlsynth::IrValue::parse_typed(arg_tuple) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to parse argument tuple: {}", e);
            std::process::exit(1);
        }
    };

    let args = match args_value.get_elements() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Argument value is not a tuple: {}", e);
            std::process::exit(1);
        }
    };

    let result = match func.interpret(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Interpretation failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("{}", result);
}
