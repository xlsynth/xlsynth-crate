// SPDX-License-Identifier: Apache-2.0
//! Exercises native and Bazel-declared standalone runtime link modes through
//! nested Cargo smoke crates.
//!
//! These tests deliberately invoke Cargo rather than unit-testing parsing:
//! the observable contract spans build-script directives and the crate's FFI
//! link annotation, either of which can make a final binary request the native
//! standalone runtime archive.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Writes one file used to assemble a throwaway nested Cargo package.
fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents)
        .unwrap_or_else(|err| panic!("Failed to write {}: {}", path.display(), err));
}

/// Creates the smallest dependent crate that compiles this runtime package.
fn write_smoke_crate(crate_dir: &Path, runtime_manifest: &Path) {
    let src_dir = crate_dir.join("src");
    fs::create_dir_all(&src_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {}", src_dir.display(), err));
    write_file(
        &crate_dir.join("Cargo.toml"),
        &format!(
            concat!(
                "[package]\n",
                "name = \"xlsynth-aot-runtime-link-mode-smoke\"\n",
                "version = \"0.1.0\"\n",
                "edition = \"2021\"\n\n",
                "[dependencies]\n",
                "xlsynth-aot-runtime = {{ path = {:?} }}\n"
            ),
            runtime_manifest.display().to_string(),
        ),
    );
    write_file(
        &src_dir.join("main.rs"),
        concat!(
            "fn main() {\n",
            "    let _ = xlsynth_aot_runtime::SUPPORTED_ARTIFACT_ABI_VERSION;\n",
            "}\n",
        ),
    );
}

/// Creates isolated filesystem state for a single nested Cargo invocation.
fn make_temp_dir() -> TempDir {
    tempfile::Builder::new()
        .prefix("xlsynth-aot-runtime-link-mode-test-")
        .tempdir()
        .unwrap_or_else(|err| panic!("Failed to create temporary test directory: {}", err))
}

/// Executes a nested offline Cargo command with one selected link contract.
///
/// Separate target directories prevent a build in one mode from reusing a
/// prior build-script result from the other mode.
fn run_nested_cargo(
    crate_dir: &Path,
    target_dir: &Path,
    cargo_subcommand: &str,
    envs: &[(&str, &Path)],
    link_mode: Option<&str>,
) -> std::process::Output {
    let cargo_binary = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut command = Command::new(&cargo_binary);
    command
        .arg(cargo_subcommand)
        .arg("-vv")
        .arg("--offline")
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_TARGET_DIR", target_dir)
        .current_dir(crate_dir);
    for (key, value) in envs {
        command.env(key, value);
    }
    if let Some(link_mode) = link_mode {
        command.env("XLS_AOT_RUNTIME_LINK_MODE", link_mode);
    }
    command.output().unwrap_or_else(|err| {
        panic!(
            "Failed to run nested cargo {} in {}: {}",
            cargo_subcommand,
            crate_dir.display(),
            err,
        )
    })
}

/// Renders nested Cargo output in assertion messages for actionable failures.
fn output_text(output: &std::process::Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Writes the minimal native-mode archive and platform link configuration.
fn write_native_runtime_inputs(root: &Path) -> (PathBuf, PathBuf) {
    let archive_path = root.join("libxls_aot_runtime.a");
    write_file(&archive_path, "");
    let target_os = if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    let link_config_path = root.join("libxls_aot_runtime_link.toml");
    write_file(
        &link_config_path,
        &format!(
            concat!(
                "format_version = 1\n\n",
                "[targets.{target_os}]\n",
                "system_libraries = []\n",
                "frameworks = []\n",
            ),
            target_os = target_os,
        ),
    );
    (archive_path, link_config_path)
}

// Verifies: Native mode tells Cargo to link the released standalone archive.
// Catches: Removing the default direct-Cargo runtime dependency.
#[test]
fn native_mode_emits_static_archive_link_directive() {
    let temp_dir = make_temp_dir();
    let crate_dir = temp_dir.path().join("native-smoke");
    let runtime_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    write_smoke_crate(&crate_dir, &runtime_manifest);
    let (archive_path, link_config_path) = write_native_runtime_inputs(temp_dir.path());

    let output = run_nested_cargo(
        &crate_dir,
        &temp_dir.path().join("native-target"),
        "check",
        &[
            ("XLS_AOT_RUNTIME_PATH", archive_path.as_path()),
            (
                "XLS_AOT_RUNTIME_LINK_CONFIG_PATH",
                link_config_path.as_path(),
            ),
        ],
        None,
    );
    let output_text = output_text(&output);
    assert!(
        output.status.success(),
        "Nested native-mode cargo check failed with status {:?}.\n{}",
        output.status,
        output_text,
    );
    assert!(
        output_text.contains("cargo:rustc-link-lib=static=xls_aot_runtime"),
        "Native mode did not emit the standalone runtime archive link directive.\n{}",
        output_text,
    );
}

// Verifies: Declared mode emits no native runtime archive request.
// Catches: Bazel and Cargo both linking the runtime.
#[test]
fn declared_mode_builds_without_native_archive_link_requests() {
    let temp_dir = make_temp_dir();
    let crate_dir = temp_dir.path().join("declared-smoke");
    let runtime_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    write_smoke_crate(&crate_dir, &runtime_manifest);

    let output = run_nested_cargo(
        &crate_dir,
        &temp_dir.path().join("declared-target"),
        "build",
        &[],
        Some("declared"),
    );
    let output_text = output_text(&output);
    assert!(
        output.status.success(),
        "Nested declared-mode cargo build failed without a native runtime archive.\n{}",
        output_text,
    );
    assert!(
        output_text.contains(
            "cargo:info=Skipping native link directives because XLS_AOT_RUNTIME_LINK_MODE=declared"
        ),
        "Declared mode did not report the skipped native directives.\n{}",
        output_text,
    );
    assert!(
        !output_text.contains("cargo:rustc-link-lib=static=xls_aot_runtime"),
        "Declared mode still emitted a flattened runtime archive link directive.\n{}",
        output_text,
    );
}
