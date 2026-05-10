// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

const XLS_AOT_RUNTIME_PATH_ENV: &str = "XLS_AOT_RUNTIME_PATH";
const XLSYNTH_ARTIFACT_CONFIG_ENV: &str = "XLSYNTH_ARTIFACT_CONFIG";

fn path_from_artifact_config(config_path: &Path) -> PathBuf {
    if !config_path.is_absolute() {
        panic!(
            "{XLSYNTH_ARTIFACT_CONFIG_ENV} must be absolute, got {}",
            config_path.display()
        );
    }
    let config_text = std::fs::read_to_string(config_path).unwrap_or_else(|error| {
        panic!(
            "failed to read {XLSYNTH_ARTIFACT_CONFIG_ENV} {}: {error}",
            config_path.display()
        )
    });
    let config = config_text.parse::<toml::Table>().unwrap_or_else(|error| {
        panic!(
            "failed to parse {XLSYNTH_ARTIFACT_CONFIG_ENV} {}: {error}",
            config_path.display()
        )
    });
    let raw_path = config
        .get("aot_runtime_path")
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| {
            panic!(
                "{XLSYNTH_ARTIFACT_CONFIG_ENV} {} is missing string field aot_runtime_path",
                config_path.display()
            )
        });
    let raw_path = PathBuf::from(raw_path);
    if raw_path.is_absolute() {
        raw_path
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| {
                panic!(
                    "{XLSYNTH_ARTIFACT_CONFIG_ENV} path has no parent: {}",
                    config_path.display()
                )
            })
            .join(raw_path)
    }
}

fn resolve_archive_path() -> PathBuf {
    if let Some(path) = std::env::var_os(XLS_AOT_RUNTIME_PATH_ENV) {
        PathBuf::from(path)
    } else if let Some(config_path) = std::env::var_os(XLSYNTH_ARTIFACT_CONFIG_ENV) {
        path_from_artifact_config(Path::new(&config_path))
    } else {
        panic!(
            "standalone AOT runtime linking requires either {} or {}",
            XLS_AOT_RUNTIME_PATH_ENV, XLSYNTH_ARTIFACT_CONFIG_ENV
        );
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={XLS_AOT_RUNTIME_PATH_ENV}");
    println!("cargo:rerun-if-env-changed={XLSYNTH_ARTIFACT_CONFIG_ENV}");

    let archive_path = resolve_archive_path();
    if archive_path.file_name().and_then(|name| name.to_str()) != Some("libxls_aot_runtime.a") {
        panic!(
            "standalone AOT runtime archive must be named libxls_aot_runtime.a, got {}",
            archive_path.display()
        );
    } else if !archive_path.is_file() {
        panic!(
            "standalone AOT runtime archive does not exist: {}",
            archive_path.display()
        );
    }

    let archive_dir = archive_path.parent().unwrap_or_else(|| {
        panic!(
            "standalone AOT runtime archive has no parent directory: {}",
            archive_path.display()
        )
    });
    println!("cargo:rerun-if-changed={}", archive_path.display());
    println!("cargo:rustc-link-search=native={}", archive_dir.display());
    println!("cargo:rustc-link-lib=static=xls_aot_runtime");

    // The XLS-owned archive packages real C++ runtime code instead of the old
    // Rust-side shim, so final Rust binaries must link the platform support it
    // would receive automatically from a native C++ driver.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=c++abi");
        println!("cargo:rustc-link-lib=dylib=unwind");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
