// SPDX-License-Identifier: Apache-2.0

//! Regression test ensuring the macOS XLS dynamic library advertises an
//! absolute install-name, guarding against regressions in the build script
//! logic.

use std::ffi::OsStr;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Command;

/// Resolve the concrete `.dylib` path on macOS by inspecting `XLS_DSO_PATH`.
///
/// When a directory is provided, we gather every `libxls-*.dylib` candidate,
/// sort them lexicographically, and select the last entry. The filenames embed
/// version numbers, so this yields the most recent artifact in a stable way.
fn resolve_macos_dso_path() -> PathBuf {
    let configured_path = PathBuf::from(xlsynth_sys::XLS_DSO_PATH);
    if configured_path.is_file() {
        return configured_path;
    }

    if configured_path.is_dir() {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(&configured_path)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to read XLS_DSO_PATH directory {}: {}",
                    configured_path.display(),
                    err
                )
            })
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter(|path| {
                path.file_name()
                    .and_then(OsStr::to_str)
                    .map(|name| name.starts_with("libxls-") && name.ends_with(".dylib"))
                    .unwrap_or(false)
            })
            .collect();

        assert!(
            !candidates.is_empty(),
            "No libxls-*.dylib candidates found in XLS_DSO_PATH directory: {}",
            configured_path.display()
        );

        candidates.sort();
        return candidates
            .last()
            .expect("candidates vector cannot be empty after is_empty check")
            .to_path_buf();
    }

    panic!(
        "XLS_DSO_PATH must be a directory or file; got: {}",
        configured_path.display()
    );
}

#[test]
fn install_name_is_absolute() {
    if !cfg!(target_os = "macos") {
        eprintln!("Skipping macOS install-name regression test on non-macOS target.");
        return;
    }

    let dso_path = resolve_macos_dso_path();

    let output = Command::new("otool")
        .arg("-D")
        .arg(&dso_path)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "Failed to launch otool -D for {}: {}",
                dso_path.display(),
                err
            )
        });

    assert!(
        output.status.success(),
        "otool -D {} exited with {:?}: {}",
        dso_path.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    let install_name_line = stdout_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.ends_with(':'))
        .unwrap_or_else(|| {
            panic!(
                "otool -D output missing install-name line.\nDSO path: {}\notool stdout:\n{}",
                dso_path.display(),
                stdout_text
            )
        })
        .to_string();

    assert!(
        install_name_line.starts_with('/'),
        "Install-name must start with '/'.\nDSO path: {}\nInstall-name: {}\notool stdout:\n{}",
        dso_path.display(),
        install_name_line,
        stdout_text
    );
    assert!(
        !install_name_line.starts_with('@'),
        "Install-name must not use @-prefixed rpath references.\nDSO path: {}\nInstall-name: {}\notool stdout:\n{}",
        dso_path.display(),
        install_name_line,
        stdout_text
    );

    let install_path = PathBuf::from(&install_name_line);
    let canonical_dso = std::fs::canonicalize(&dso_path);
    let canonical_install = std::fs::canonicalize(&install_path);

    match (canonical_dso, canonical_install) {
        (Ok(expected), Ok(actual)) => {
            assert_eq!(
                actual,
                expected,
                "Install-name resolved to {} but canonical DSO path is {}.\nDSO path: {}\notool stdout:\n{}",
                actual.display(),
                expected.display(),
                dso_path.display(),
                stdout_text
            );
        }
        _ => {
            let install_metadata = std::fs::metadata(&install_path).unwrap_or_else(|err| {
                panic!(
                    "Install-name {} could not be opened: {}",
                    install_path.display(),
                    err
                )
            });
            let dso_metadata = std::fs::metadata(&dso_path).unwrap_or_else(|err| {
                panic!(
                    "Resolved DSO {} could not be opened: {}",
                    dso_path.display(),
                    err
                )
            });

            #[cfg(target_family = "unix")]
            {
                assert_eq!(
                    install_metadata.dev(),
                    dso_metadata.dev(),
                    "Install-name {} resolved to device {} but DSO resides on device {}.",
                    install_path.display(),
                    install_metadata.dev(),
                    dso_metadata.dev()
                );
                assert_eq!(
                    install_metadata.ino(),
                    dso_metadata.ino(),
                    "Install-name {} resolved to inode {} but DSO inode is {}.",
                    install_path.display(),
                    install_metadata.ino(),
                    dso_metadata.ino()
                );
            }

            #[cfg(not(target_family = "unix"))]
            {
                assert_eq!(
                    install_metadata.len(),
                    dso_metadata.len(),
                    "Install-name {} resolved to file length {} but DSO length is {}.",
                    install_path.display(),
                    install_metadata.len(),
                    dso_metadata.len()
                );
            }
        }
    }
}
