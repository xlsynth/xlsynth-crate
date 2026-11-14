// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

fn compile_lean(root: &Path, imports: &[&Path], source: &Path, object: &Path) -> Result<()> {
    let lean =
        std::env::var_os("XLSYNTH_LEAN").context("set XLSYNTH_LEAN to the Lean executable")?;
    let output = std::process::Command::new(lean)
        .current_dir(root)
        .env_remove("LEAN_SYSROOT")
        .env("LEAN_PATH", std::env::join_paths(imports)?)
        .arg("-o")
        .arg(object)
        .arg(source)
        .output()?;
    ensure!(
        output.status.success(),
        "Lean failed to compile {}:\n{}\n{}",
        source.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

pub fn compile_project(root: &Path, sources: impl IntoIterator<Item = PathBuf>) -> Result<()> {
    // Compile the repository module once, outside the model output directory.
    let support = tempfile::tempdir()?;
    let object = support.path().join("XlsynthLean/IR.olean");
    std::fs::create_dir(object.parent().unwrap())?;
    let source = object.with_extension("lean");
    std::fs::write(&source, include_str!("../../lean/XlsynthLean/IR.lean"))?;
    compile_lean(support.path(), &[], &source, &object)?;
    for source in sources {
        compile_lean(
            root,
            &[root, support.path()],
            &source,
            &source.with_extension("olean"),
        )?;
    }
    Ok(())
}
