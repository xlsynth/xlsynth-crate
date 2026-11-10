// SPDX-License-Identifier: Apache-2.0

use std::fs;

#[path = "block_project.rs"]
mod block_project;

use anyhow::Result;
use xlsynth_lean::block::LeanBlockBundle;

/// Elaborate every generated file in dependency order with the requested Lean.
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
