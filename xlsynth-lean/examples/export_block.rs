// SPDX-License-Identifier: Apache-2.0

//! Generate a raw block model, or its five source-named modules.
//! This example exercises the library; it does not check the generated proofs.

use std::{env, fs, path::Path};

use anyhow::{Context, Result, ensure};
use xlsynth_lean::block::{self, Attachment};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    ensure!(
        args.len() == 2 || args.len() == 5,
        "usage: export_block BLOCK_IR NEW_OUTPUT_DIRECTORY [DSLX_SOURCE REGISTER SOURCE_TYPE]"
    );
    let ir = fs::read_to_string(&args[0]).context("read block IR")?;
    let files = if args.len() == 2 {
        let parsed = block::parse(&ir, None)?;
        vec![("Generated.lean".to_owned(), block::emit(&parsed)?)]
    } else {
        let source_path = Path::new(&args[2]);
        let source = fs::read_to_string(source_path).context("read DSLX source")?;
        let attachments = [Attachment {
            register: args[3].clone(),
            source_type: args[4].clone(),
        }];
        let map = block::collect_source_map(&ir, &source, source_path, &attachments)?;
        let checked = block::validate_source_map(&map, &ir, &source, source_path)?;
        block::emit_readable(&checked)?.files
    };
    // Generate everything before creating the output directory, so invalid
    // input does not leave a partial bundle that a consumer could mistake for
    // it.
    let directory = Path::new(&args[1]);
    fs::create_dir(directory).context("create a new output directory")?;
    for (name, text) in files {
        fs::write(directory.join(name), text).context("write generated Lean module")?;
    }
    Ok(())
}
