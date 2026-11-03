// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::Path;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::process_ir_path::WordChoices;

pub(super) fn require_new_destination(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!("{} already exists", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("checking {}", path.display())),
    }
}

/// Stage the complete choice set before publishing it. A fresh directory is
/// required even when the invocation produces no offers.
pub(super) fn publish(destination: &Path, choices: &WordChoices) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix(".g8r-word-choices-")
        .tempdir_in(parent)?;
    let mut candidates = Vec::new();
    for choice in &choices.candidates {
        let key = match (choice.scope, choice.source_pairs.as_slice()) {
            ("all", _) => format!("{}-all", choice.mode),
            ("source-pair", [[x, y]]) => format!("{}-source-{x}-{y}", choice.mode),
            _ => bail!("invalid word choice source scope"),
        };
        let file = format!("{key}.aig");
        let path = staging.path().join(&file);
        let bytes = emit_aiger_binary(&choice.graph, true).map_err(anyhow::Error::msg)?;
        fs::File::create_new(&path)?.write_all(&bytes)?;
        candidates.push(json!({
            "key": key, "file": file, "mode": choice.mode, "scope": choice.scope,
            "source_pairs": choice.source_pairs, "output_bits": choice.output_bits,
        }));
    }
    let manifest = json!({
        "schema_version": 1, "stage": "native-terminal", "status": choices.status,
        "construction_limited": choices.construction_limited, "candidates": candidates,
    });
    fs::write(
        staging.path().join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    // Reserve the name exclusively, including against another concurrent
    // export. Only a complete generated directory may replace this empty
    // reservation.
    #[cfg(not(windows))]
    {
        fs::create_dir(destination)
            .with_context(|| format!("creating fresh directory {}", destination.display()))?;
        if let Err(err) = fs::rename(staging.path(), destination) {
            fs::remove_dir(destination).context("removing the empty word choice reservation")?;
            return Err(err).context("publishing the staged word choices");
        }
    }
    // Windows refuses to replace a destination directory with a moved
    // directory.
    #[cfg(windows)]
    fs::rename(staging.path(), destination).context("publishing choices to a fresh directory")?;
    Ok(())
}
