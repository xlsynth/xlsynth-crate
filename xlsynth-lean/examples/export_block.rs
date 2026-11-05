// SPDX-License-Identifier: Apache-2.0

//! Generate a raw block model from XLS block IR.
//! This example exercises the library; it does not check the generated proofs.

use std::{env, fs, path::Path};

use anyhow::{Context, Result, ensure};
use xlsynth_lean::block;

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: export_block BLOCK_IR NEW_OUTPUT_DIRECTORY"
    );
    let ir = fs::read_to_string(&args[0]).context("reading block IR")?;
    xlsynth::IrPackage::parse_ir(&ir, None)?.verify()?;
    let package = xlsynth_pir::ir_parser::Parser::new(&ir).parse_and_validate_package()?;
    let model = block::emit(&package, block::select_block(&package, None)?)?;
    let directory = Path::new(&args[1]);
    fs::create_dir(directory).context("create a new output directory")?;
    fs::write(directory.join("Generated.lean"), model)?;
    Ok(())
}
