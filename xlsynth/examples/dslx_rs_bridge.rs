// SPDX-License-Identifier: Apache-2.0

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use xlsynth::dslx;
use xlsynth::dslx_bridge;
use xlsynth::rust_bridge_builder::RustBridgeBuilder;

fn usage(program: &str) -> String {
    format!("usage: {program} <input.x> <dslx_search_path> <function_name> [dependency.x ...]")
}

fn convert_file(
    input_path: &Path,
    search_path: &Path,
    function_name: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let dslx_text = std::fs::read_to_string(input_path)?;
    let search_path_views = [search_path];
    let mut import_data = dslx::ImportData::new(None, &search_path_views);
    let mut builder = if let Some(function_name) = function_name {
        RustBridgeBuilder::with_function_signature_aliases(function_name)
    } else {
        RustBridgeBuilder::new()
    };
    dslx_bridge::convert_leaf_module(&mut import_data, &dslx_text, input_path, &mut builder)?;
    Ok(builder.build())
}

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() >= 4 {
        let input_path = PathBuf::from(&args[1]);
        let search_path = PathBuf::from(&args[2]);
        let function_name = &args[3];
        let dependency_outputs = args[4..]
            .iter()
            .map(|path| convert_file(Path::new(path), &search_path, None))
            .collect::<Result<Vec<_>, _>>()?;
        let main_output = convert_file(&input_path, &search_path, Some(function_name))?;
        let output = dependency_outputs
            .into_iter()
            .chain(std::iter::once(main_output))
            .collect::<Vec<_>>()
            .join("\n\n");
        println!("{output}");
        Ok(())
    } else {
        Err(Box::new(Error::new(
            ErrorKind::InvalidInput,
            usage(&args[0]),
        )))
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if let Err(error) = run(&args) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
