// SPDX-License-Identifier: Apache-2.0

//! Checks generated bundles in isolated Lean projects for native tests and
//! fuzzing.

use std::fs;

#[path = "block_project.rs"]
mod block_project;

use anyhow::Result;
use xlsynth_lean::block::LeanBlockBundle;

/// Writes the bundle to a temporary project and elaborates its files in bundle
/// order.
///
/// Append client modules after the generated modules they import.
///
/// # Errors
///
/// Returns I/O errors and Lean compilation failures, including failed client
/// guards.
pub fn check_bundle(bundle: &LeanBlockBundle) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let mut sources = Vec::new();
    for (name, source) in &bundle.files {
        let path = root.join(name);
        fs::write(&path, source)?;
        sources.push(path);
    }
    block_project::compile_project(root, sources)
}
