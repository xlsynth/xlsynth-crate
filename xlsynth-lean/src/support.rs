// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use crate::LeanError;

/// Companion Lean support files embedded in the Rust crate. Module entries are
/// listed in dependency order.
const LEAN_SUPPORT_FILES: &[(&str, &str)] = &[
    (
        "XlsynthLean/Aig.lean",
        include_str!("../lean/XlsynthLean/Aig.lean"),
    ),
    (
        "XlsynthLean/Aag.lean",
        include_str!("../lean/XlsynthLean/Aag.lean"),
    ),
    ("XlsynthLean.lean", include_str!("../lean/XlsynthLean.lean")),
    ("EvalMain.lean", include_str!("../lean/EvalMain.lean")),
    ("lean-toolchain", include_str!("../lean/lean-toolchain")),
];

macro_rules! proof_module {
    ($name:literal) => {
        (
            concat!("XlsynthLean/Proof/", $name, ".lean"),
            include_str!(concat!("../lean/XlsynthLean/Proof/", $name, ".lean")),
        )
    };
}

/// Proof-only modules, compiled in dependency order.
const PROOF_SUPPORT_FILES: &[(&str, &str)] = &[
    proof_module!("Claim"),
    proof_module!("Miter"),
    proof_module!("Decide"),
    proof_module!("Outputs"),
    proof_module!("Rebuild"),
    proof_module!("Leaves"),
    proof_module!("Rewrite"),
    proof_module!("Domain"),
    proof_module!("Substitute"),
    proof_module!("Cut"),
    proof_module!("Sparse"),
    (
        "XlsynthLean/Proof.lean",
        include_str!("../lean/XlsynthLean/Proof.lean"),
    ),
];

/// Returns support module paths in dependency order.
pub(crate) fn lean_support_module_paths() -> impl Iterator<Item = &'static str> {
    lean_support_paths()
        .filter(|relative_path| relative_path.ends_with(".lean"))
        .chain(PROOF_SUPPORT_FILES.iter().map(|(path, _)| *path))
}

/// Returns the support artifacts shared with the circuit evaluator.
pub(crate) fn lean_support_paths() -> impl Iterator<Item = &'static str> {
    LEAN_SUPPORT_FILES
        .iter()
        .map(|(relative_path, _)| *relative_path)
}

/// Writes the companion Lean support files into `root` using portable relative
/// paths.
pub fn write_lean_support(root: &Path) -> Result<(), LeanError> {
    write_evaluation_support(root)?;
    write_proof_support(root)
}

/// Writes the modules and toolchain pin needed by the circuit evaluator.
pub(crate) fn write_evaluation_support(root: &Path) -> Result<(), LeanError> {
    write_support_files(root, LEAN_SUPPORT_FILES)
}

/// Stages the tactics shipped with the proof checker.
pub(crate) fn write_proof_support(root: &Path) -> Result<(), LeanError> {
    write_support_files(root, PROOF_SUPPORT_FILES)
}

fn write_support_files(root: &Path, files: &[(&str, &str)]) -> Result<(), LeanError> {
    for (relative_path, contents) in files {
        write_file(root, Path::new(relative_path), contents.as_bytes())?;
    }
    Ok(())
}

/// Writes one generated artifact, creating its parent directories.
pub(crate) fn write_file(
    root: &Path,
    relative_path: &Path,
    contents: &[u8],
) -> Result<(), LeanError> {
    let path = root.join(relative_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| LeanError::Io {
            operation: "creating directory",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(&path, contents).map_err(|source| LeanError::Io {
        operation: "writing file",
        path,
        source,
    })
}
