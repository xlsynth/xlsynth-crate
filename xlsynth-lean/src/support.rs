// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use crate::LeanError;

type Module = (&'static str, &'static [u8]);

const AIG: Module = (
    "XlsynthLean/Aig.lean",
    include_bytes!("../lean/XlsynthLean/Aig.lean"),
);
const AAG: Module = (
    "XlsynthLean/Aag.lean",
    include_bytes!("../lean/XlsynthLean/Aag.lean"),
);
const XLSYNTH_LEAN: Module = (
    "XlsynthLean.lean",
    include_bytes!("../lean/XlsynthLean.lean"),
);
const EVAL_MAIN: Module = ("EvalMain.lean", include_bytes!("../lean/EvalMain.lean"));
const IR: Module = (
    "XlsynthLean/IR.lean",
    include_bytes!("../lean/XlsynthLean/IR.lean"),
);
const TOOLCHAIN_PATH: &str = "lean-toolchain";

/// The modules a consumer needs, in Lean compilation order, plus the common
/// toolchain pin.
pub(crate) struct LeanSupport(&'static [Module]);

// This is also the on-disk contract for previously prepared AIG proof targets.
pub(crate) const AIG_PROOF_SUPPORT: LeanSupport = LeanSupport(&[AIG, AAG, XLSYNTH_LEAN, EVAL_MAIN]);
pub(crate) const IR_FUNCTION_SUPPORT: LeanSupport = LeanSupport(&[IR]);
pub(crate) const EVALUATOR_SUPPORT: LeanSupport = LeanSupport(&[IR, AIG, AAG, EVAL_MAIN]);

impl LeanSupport {
    pub(crate) fn module_paths(&self) -> impl Iterator<Item = &'static str> {
        self.0.iter().map(|&(path, _)| path)
    }

    pub(crate) fn paths(&self) -> impl Iterator<Item = &'static str> {
        self.module_paths().chain([TOOLCHAIN_PATH])
    }

    pub(crate) fn write_to(&self, root: &Path) -> Result<(), LeanError> {
        write_support(root, self.0.iter().copied())
    }
}

/// Writes all companion Lean sources and their toolchain pin.
pub fn write_lean_support(root: &Path) -> Result<(), LeanError> {
    let modules = AIG_PROOF_SUPPORT.0.iter().chain(IR_FUNCTION_SUPPORT.0);
    write_support(root, modules.copied())
}

fn write_support(root: &Path, modules: impl Iterator<Item = Module>) -> Result<(), LeanError> {
    for (path, source) in modules {
        write_file(root, Path::new(path), source)?;
    }
    write_toolchain(root)
}

pub(crate) fn write_toolchain(root: &Path) -> Result<(), LeanError> {
    write_file(
        root,
        Path::new(TOOLCHAIN_PATH),
        include_bytes!("../lean/lean-toolchain"),
    )
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
