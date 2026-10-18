// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use crate::LeanError;

/// Companion Lean support files embedded in the Rust crate.
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

/// Writes the companion Lean support files into `root` using portable relative
/// paths.
pub fn write_lean_support(root: &Path) -> Result<(), LeanError> {
    for (relative_path, contents) in LEAN_SUPPORT_FILES {
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
