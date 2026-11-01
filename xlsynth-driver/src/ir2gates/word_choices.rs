// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::fs;
use std::io::Write;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
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

    rename_new_directory(staging.path(), destination)
        .with_context(|| format!("publishing choices to {}", destination.display()))?;
    // The staging name is free for reuse after the rename.
    let _ = staging.keep();
    Ok(())
}

/// Publishes the complete directory only if its name is still available.
#[cfg(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    target_os = "redox"
))]
fn rename_new_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        source,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    Ok(())
}

#[cfg(windows)]
fn rename_new_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    let wide_path = |p: &Path| {
        let mut wide: Vec<u16> = p.as_os_str().encode_wide().collect();
        if wide.contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "NUL in path",
            ));
        }
        wide.push(0);
        Ok(wide)
    };
    let from = wide_path(source)?;
    let to = wide_path(destination)?;
    // SAFETY: Both buffers are NUL-terminated and live for the call. With
    // flags zero, MoveFileExW only moves to a fresh name on the same volume.
    let moved = unsafe {
        windows_sys::Win32::Storage::FileSystem::MoveFileExW(from.as_ptr(), to.as_ptr(), 0)
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(any(
    windows,
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    target_os = "redox"
)))]
fn rename_new_directory(_source: &Path, _destination: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic directory publication without replacement is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn manifest_is_readable_as_soon_as_the_destination_appears() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("choices");
        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                publish(
                    &dest,
                    &WordChoices {
                        candidates: Vec::new(),
                        status: "no-offers",
                        construction_limited: false,
                    },
                )
            });
            while !dest.exists() && !writer.is_finished() {
                thread::yield_now();
            }
            let manifest = fs::read(dest.join("manifest.json"));
            writer.join().unwrap().unwrap();
            let manifest: serde_json::Value = serde_json::from_slice(&manifest.unwrap()).unwrap();
            assert_eq!(manifest["status"], "no-offers");
        });
    }

    #[test]
    fn publication_requires_a_fresh_destination_at_rename_time() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("choices");
        require_new_destination(&dest).unwrap();
        fs::create_dir(&dest).unwrap();
        let choices = WordChoices {
            candidates: Vec::new(),
            status: "no-offers",
            construction_limited: false,
        };
        assert!(publish(&dest, &choices).is_err());
        assert_eq!(fs::read_dir(dest).unwrap().count(), 0);
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 1);
    }
}
