// SPDX-License-Identifier: Apache-2.0

//! File and CLI adaptation for the library-owned Lean block translation.
//!
//! Clap enforces the raw/readable option combinations. The library verifies
//! inputs and generates the complete model before this adapter opens outputs.
//! Generation errors preserve existing files; write errors can leave partial
//! output. Readable bundles require a new directory to avoid mixing models.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use clap::ArgMatches;
use xlsynth_lean::block::{self, Attachment};

/// Parse attachment syntax; the library checks register and type semantics.
///
/// This clap value parser accepts exactly one nonempty `REGISTER=TYPE` pair.
/// A syntactically valid pair can still name nonexistent or incompatible
/// declarations; source-map collection and validation check that later.
pub fn parse_type_attachment(text: &str) -> Result<Attachment, String> {
    match text.split_once('=') {
        Some((register, source_type))
            if !register.is_empty() && !source_type.is_empty() && !source_type.contains('=') =>
        {
            Ok(Attachment {
                register: register.to_owned(),
                source_type: source_type.to_owned(),
            })
        }
        _ => Err("expected REGISTER=TYPE with nonempty register and type names".to_owned()),
    }
}

/// Generate all Lean text before creating or truncating any output files.
///
/// Call with matches validated by the `block2lean` command definition, which
/// requires all readable options together and excludes raw-only options.
/// IO errors retain input/output path context; filesystem errors during writes
/// can leave partial output. This operation never invokes Lean.
pub fn handle_block2lean(matches: &ArgMatches) -> Result<()> {
    let input_path = matches
        .get_one::<String>("input_file")
        .expect("clap requires an input file");
    let input = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read block IR file '{input_path}'"))?;
    if let Some(source_path) = matches.get_one::<String>("dslx_source") {
        let source = std::fs::read_to_string(source_path)
            .with_context(|| format!("failed to read DSLX source '{source_path}'"))?;
        let attachments: Vec<_> = matches
            .get_many::<Attachment>("type_attachment")
            .expect("clap requires attachments with DSLX source")
            .cloned()
            .collect();
        let source_path = Path::new(source_path);
        let map = block::collect_source_map(&input, &source, source_path, &attachments)?;
        let checked = block::validate_source_map(&map, &input, &source, source_path)?;
        let bundle = block::emit_readable(&checked)?;
        let directory = matches
            .get_one::<String>("output_dir")
            .expect("clap requires an output directory with DSLX source");
        std::fs::create_dir(directory)
            .with_context(|| format!("failed to create new output directory '{directory}'"))?;
        for (name, text) in bundle.files {
            let path = Path::new(directory).join(name);
            std::fs::write(&path, text).with_context(|| {
                format!("failed to write generated Lean module '{}'", path.display())
            })?;
        }
    } else {
        let top = matches.get_one::<String>("top").map(String::as_str);
        let parsed = block::parse(&input, top)?;
        let text = block::emit(&parsed)?;
        if let Some(path) = matches.get_one::<String>("output") {
            std::fs::write(path, text)
                .with_context(|| format!("failed to write generated Lean file '{path}'"))?;
        } else {
            std::io::stdout()
                .lock()
                .write_all(text.as_bytes())
                .context("failed to write generated Lean to stdout")?;
        }
    }
    Ok(())
}
