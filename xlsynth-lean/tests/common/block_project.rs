// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

fn compile_lean(root: &Path, imports: &[&Path], source: &Path, object: &Path) {
    let lean = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable");
    let output = std::process::Command::new(lean)
        .current_dir(root)
        .env_remove("LEAN_SYSROOT")
        .env("LEAN_PATH", std::env::join_paths(imports).unwrap())
        .arg("-o")
        .arg(object)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Lean failed to compile {}:\n{}\n{}",
        source.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

pub fn compile_project(root: &Path, sources: impl IntoIterator<Item = PathBuf>) {
    // Compile the repository module once, outside the model output directory.
    let support = tempfile::tempdir().unwrap();
    let library = Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
    let object = support.path().join("XlsynthLean/IR.olean");
    std::fs::create_dir(object.parent().unwrap()).unwrap();
    compile_lean(&library, &[], &library.join("XlsynthLean/IR.lean"), &object);
    for source in sources {
        compile_lean(
            root,
            &[root, support.path()],
            &source,
            &source.with_extension("olean"),
        );
    }
}
