// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn resolve_config_dso_path() -> PathBuf {
    let configured_path = PathBuf::from(xlsynth_sys::XLS_DSO_PATH);
    if configured_path.is_file() {
        return configured_path;
    }

    if configured_path.is_dir() {
        let expected_suffix = if cfg!(target_os = "macos") {
            ".dylib"
        } else {
            ".so"
        };
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
                    .map(|name| name.starts_with("libxls-") && name.ends_with(expected_suffix))
                    .unwrap_or(false)
            })
            .collect();
        candidates.sort();
        return candidates.pop().unwrap_or_else(|| {
            panic!(
                "No XLS DSO candidates found in {}",
                configured_path.display()
            )
        });
    }

    panic!(
        "XLS_DSO_PATH must be a directory or file; got: {}",
        configured_path.display()
    );
}

fn make_temp_dir() -> PathBuf {
    let unique_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "xlsynth-sys-artifact-config-test-{}-{unique_id}",
        std::process::id()
    ));
    fs::create_dir_all(&temp_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create temporary test directory {}: {}",
            temp_dir.display(),
            err
        )
    });
    temp_dir
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents)
        .unwrap_or_else(|err| panic!("Failed to write {}: {}", path.display(), err));
}

fn write_smoke_crate(temp_crate_dir: &Path, manifest_path: &Path) {
    let temp_src_dir = temp_crate_dir.join("src");
    fs::create_dir_all(&temp_src_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create temporary crate directory {}: {}",
            temp_src_dir.display(),
            err
        )
    });
    write_file(
        &temp_crate_dir.join("Cargo.toml"),
        &format!(
            concat!(
                "[package]\n",
                "name = \"artifact-config-smoke\"\n",
                "version = \"0.1.0\"\n",
                "edition = \"2021\"\n\n",
                "[dependencies]\n",
                "xlsynth-sys = {{ path = {:?} }}\n"
            ),
            manifest_path.display().to_string()
        ),
    );
    write_file(
        &temp_src_dir.join("main.rs"),
        concat!(
            "fn main() {\n",
            "    println!(\"XLS_DSO_PATH={}\", xlsynth_sys::XLS_DSO_PATH);\n",
            "    println!(\"DSLX_STDLIB_PATH={}\", xlsynth_sys::DSLX_STDLIB_PATH);\n",
            "}\n"
        ),
    );
}

fn copy_dir_recursive(source_dir: &Path, target_dir: &Path) {
    fs::create_dir_all(target_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create directory {} while copying {}: {}",
            target_dir.display(),
            source_dir.display(),
            err
        )
    });
    for entry in fs::read_dir(source_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to read directory {} while copying to {}: {}",
            source_dir.display(),
            target_dir.display(),
            err
        )
    }) {
        let entry = entry.unwrap_or_else(|err| {
            panic!(
                "Failed to read entry in {} while copying to {}: {}",
                source_dir.display(),
                target_dir.display(),
                err
            )
        });
        let source_path = entry.path();
        let target_path = target_dir.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target_path);
        } else if source_path.is_file() {
            fs::copy(&source_path, &target_path).unwrap_or_else(|err| {
                panic!(
                    "Failed to copy file {} to {}: {}",
                    source_path.display(),
                    target_path.display(),
                    err
                )
            });
        } else {
            panic!(
                "Expected directory tree entry to be a file or directory: {}",
                source_path.display()
            );
        }
    }
}

fn copy_config_artifacts(config_artifacts_dir: &Path) -> (PathBuf, PathBuf) {
    fs::create_dir_all(config_artifacts_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create config artifacts directory {}: {}",
            config_artifacts_dir.display(),
            err
        )
    });

    let source_dso_path = resolve_config_dso_path();
    let source_stdlib_path = PathBuf::from(xlsynth_sys::DSLX_STDLIB_PATH);
    let expected_dso_path = config_artifacts_dir.join(
        source_dso_path
            .file_name()
            .unwrap_or_else(|| panic!("DSO path missing filename: {}", source_dso_path.display())),
    );
    fs::copy(&source_dso_path, &expected_dso_path).unwrap_or_else(|err| {
        panic!(
            "Failed to copy DSO {} to {}: {}",
            source_dso_path.display(),
            expected_dso_path.display(),
            err
        )
    });

    let expected_stdlib_path = config_artifacts_dir.join("stdlib");
    copy_dir_recursive(&source_stdlib_path, &expected_stdlib_path);
    (expected_dso_path, expected_stdlib_path)
}

fn write_artifact_config(config_path: &Path, dso_path: &str, dslx_stdlib_path: &str) {
    let config_dir = config_path.parent().unwrap_or_else(|| {
        panic!(
            "Artifact config path should have a parent directory: {}",
            config_path.display()
        )
    });
    fs::create_dir_all(config_dir).unwrap_or_else(|err| {
        panic!(
            "Failed to create artifact config directory {}: {}",
            config_dir.display(),
            err
        )
    });
    write_file(
        config_path,
        &format!(
            "dso_path = {:?}\ndslx_stdlib_path = {:?}\n",
            dso_path, dslx_stdlib_path
        ),
    );
}

fn run_nested_cargo_with_artifact_config(
    temp_crate_dir: &Path,
    target_dir: &Path,
    artifact_config_path: &OsStr,
) -> std::process::Output {
    let cargo_binary = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    Command::new(&cargo_binary)
        .arg("run")
        .arg("--quiet")
        .arg("--offline")
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_TARGET_DIR", target_dir)
        .env("XLSYNTH_ARTIFACT_CONFIG", artifact_config_path)
        .env("XLS_DSO_PATH", "/definitely/not/the/configured/libxls.so")
        .env("DSLX_STDLIB_PATH", "/definitely/not/the/configured/stdlib/")
        .current_dir(temp_crate_dir)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "Failed to run nested cargo for artifact-config test: {}",
                err
            )
        })
}

#[test]
fn artifact_config_resolves_relative_toml_paths_from_absolute_config_path() {
    let temp_dir = make_temp_dir();
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_crate_dir = temp_dir.join("artifact-config-smoke");
    let config_root_dir = temp_dir.join("config-root");
    let config_artifacts_dir = config_root_dir.join("artifacts");
    let (expected_dso_path, expected_stdlib_path) = copy_config_artifacts(&config_artifacts_dir);
    let config_path = config_root_dir.join("artifact-config.toml");
    write_artifact_config(
        &config_path,
        &format!(
            "artifacts/{}",
            expected_dso_path
                .file_name()
                .unwrap_or_else(|| panic!(
                    "Expected copied DSO path to have filename: {}",
                    expected_dso_path.display()
                ))
                .to_string_lossy()
        ),
        "artifacts/stdlib",
    );
    write_smoke_crate(&temp_crate_dir, &manifest_path);

    let target_dir = temp_dir.join("target");
    let output = run_nested_cargo_with_artifact_config(
        &temp_crate_dir,
        &target_dir,
        config_path.as_os_str(),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Nested cargo run failed with status {:?}.\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
    let expected_dso_line = format!("XLS_DSO_PATH={}", expected_dso_path.display());
    let expected_stdlib_line = format!("DSLX_STDLIB_PATH={}", expected_stdlib_path.display());
    assert!(
        stdout.contains(&expected_dso_line),
        "Nested cargo output did not include expected DSO path.\nExpected line: {}\nstdout:\n{}\nstderr:\n{}",
        expected_dso_line,
        stdout,
        stderr
    );
    assert!(
        stdout.contains(&expected_stdlib_line),
        "Nested cargo output did not include expected DSLX stdlib path.\nExpected line: {}\nstdout:\n{}\nstderr:\n{}",
        expected_stdlib_line,
        stdout,
        stderr
    );
    assert!(
        !stdout.contains("/definitely/not/the/configured/"),
        "Nested cargo output should come from XLSYNTH_ARTIFACT_CONFIG and its relative-path resolution, not the paired env override.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );

    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn artifact_config_requires_absolute_config_path() {
    let temp_dir = make_temp_dir();
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_crate_dir = temp_dir.join("artifact-config-relative-smoke");
    let config_root_dir = temp_crate_dir.join("config");
    let config_artifacts_dir = temp_crate_dir.join("artifacts");
    let (expected_dso_path, _expected_stdlib_path) = copy_config_artifacts(&config_artifacts_dir);
    let config_path = config_root_dir.join("artifact-config.toml");
    write_artifact_config(
        &config_path,
        &format!(
            "../artifacts/{}",
            expected_dso_path
                .file_name()
                .unwrap_or_else(|| panic!(
                    "Expected copied DSO path to have filename: {}",
                    expected_dso_path.display()
                ))
                .to_string_lossy()
        ),
        "../artifacts/stdlib",
    );
    write_smoke_crate(&temp_crate_dir, &manifest_path);

    let target_dir = temp_dir.join("relative-target");
    let output = run_nested_cargo_with_artifact_config(
        &temp_crate_dir,
        &target_dir,
        OsStr::new("config/artifact-config.toml"),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "Nested cargo run unexpectedly succeeded with relative XLSYNTH_ARTIFACT_CONFIG.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    assert!(
        stderr.contains("XLSYNTH_ARTIFACT_CONFIG must be an absolute path"),
        "Nested cargo stderr did not explain the absolute-path requirement.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    assert!(
        stderr.contains("dso_path and dslx_stdlib_path values inside that TOML may still be relative"),
        "Nested cargo stderr did not preserve the relative-path guidance for TOML entries.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    assert!(
        !stderr.contains("No such file or directory"),
        "Nested cargo stderr should reject relative XLSYNTH_ARTIFACT_CONFIG directly instead of failing with a file lookup error.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );

    fs::remove_dir_all(&temp_dir).ok();
}
