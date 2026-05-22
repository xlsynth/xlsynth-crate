// SPDX-License-Identifier: Apache-2.0
//! Selects the native linking contract for the standalone AOT runtime crate.
//!
//! Direct Cargo consumers use the released native archive and companion link
//! configuration. Bazel consumers set declared-link mode because their final
//! target declares and links the source-backed runtime dependency itself; in
//! that mode this build script must not issue duplicate native link requests.

use std::path::{Path, PathBuf};

const XLS_AOT_RUNTIME_PATH_ENV: &str = "XLS_AOT_RUNTIME_PATH";
const XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV: &str = "XLS_AOT_RUNTIME_LINK_CONFIG_PATH";
/// Selects whether Cargo or the enclosing build graph owns native runtime
/// linking.
const XLS_AOT_RUNTIME_LINK_MODE_ENV: &str = "XLS_AOT_RUNTIME_LINK_MODE";
const XLSYNTH_ARTIFACT_CONFIG_ENV: &str = "XLSYNTH_ARTIFACT_CONFIG";

/// Identifies which build system is responsible for providing native runtime
/// symbols.
#[derive(Clone, Copy)]
enum LinkDirectiveMode {
    /// Cargo links the released `libxls_aot_runtime.a` archive and its system
    /// dependencies.
    Native,
    /// The enclosing build graph supplies the runtime dependency and Cargo
    /// emits no native links.
    Declared,
}

struct RuntimeInputs {
    archive_path: PathBuf,
    link_config_path: PathBuf,
}

struct RuntimeLinkConfig {
    system_libraries: Vec<String>,
    frameworks: Vec<String>,
}

/// Parses the link-ownership contract, preserving native linking as the
/// default.
///
/// Declared mode is opt-in because ordinary Cargo builds do not otherwise have
/// a Bazel target available to satisfy the standalone runtime symbols.
fn parse_link_directive_mode() -> LinkDirectiveMode {
    match std::env::var(XLS_AOT_RUNTIME_LINK_MODE_ENV) {
        Ok(value) => match value.as_str() {
            "native" => LinkDirectiveMode::Native,
            "declared" => LinkDirectiveMode::Declared,
            _ => panic!(
                "{XLS_AOT_RUNTIME_LINK_MODE_ENV} must be one of 'native' or 'declared'; got {:?}",
                value
            ),
        },
        Err(std::env::VarError::NotPresent) => LinkDirectiveMode::Native,
        Err(std::env::VarError::NotUnicode(value)) => {
            panic!(
                "{XLS_AOT_RUNTIME_LINK_MODE_ENV} must be valid UTF-8; got {:?}",
                value
            )
        }
    }
}

fn parse_toml_file(path: &Path, label: &str) -> toml::Table {
    let config_text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {label} {}: {error}", path.display()));
    config_text
        .parse::<toml::Table>()
        .unwrap_or_else(|error| panic!("failed to parse {label} {}: {error}", path.display()))
}

fn resolve_artifact_config_path(config_path: &Path, config: &toml::Table, key: &str) -> PathBuf {
    let raw_path = config
        .get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| {
            panic!(
                "{XLSYNTH_ARTIFACT_CONFIG_ENV} {} is missing string field {key}",
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

fn runtime_inputs_from_artifact_config(config_path: &Path) -> RuntimeInputs {
    if !config_path.is_absolute() {
        panic!(
            "{XLSYNTH_ARTIFACT_CONFIG_ENV} must be absolute, got {}",
            config_path.display()
        );
    }
    println!("cargo:rerun-if-changed={}", config_path.display());
    let config = parse_toml_file(config_path, XLSYNTH_ARTIFACT_CONFIG_ENV);
    RuntimeInputs {
        archive_path: resolve_artifact_config_path(config_path, &config, "aot_runtime_path"),
        link_config_path: resolve_artifact_config_path(
            config_path,
            &config,
            "aot_runtime_link_config_path",
        ),
    }
}

fn resolve_runtime_inputs() -> RuntimeInputs {
    let archive_path = std::env::var_os(XLS_AOT_RUNTIME_PATH_ENV);
    let link_config_path = std::env::var_os(XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV);
    if archive_path.is_some() || link_config_path.is_some() {
        let archive_path = archive_path.unwrap_or_else(|| {
            panic!(
                "{} is required when {} is set",
                XLS_AOT_RUNTIME_PATH_ENV, XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV
            )
        });
        let link_config_path = link_config_path.unwrap_or_else(|| {
            panic!(
                "{} is required when {} is set",
                XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV, XLS_AOT_RUNTIME_PATH_ENV
            )
        });
        RuntimeInputs {
            archive_path: PathBuf::from(archive_path),
            link_config_path: PathBuf::from(link_config_path),
        }
    } else if let Some(config_path) = std::env::var_os(XLSYNTH_ARTIFACT_CONFIG_ENV) {
        runtime_inputs_from_artifact_config(Path::new(&config_path))
    } else {
        panic!(
            "standalone AOT runtime linking requires either {} + {} or {}",
            XLS_AOT_RUNTIME_PATH_ENV,
            XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV,
            XLSYNTH_ARTIFACT_CONFIG_ENV
        );
    }
}

fn read_string_array(table: &toml::Table, path: &Path, field_name: &str) -> Vec<String> {
    table
        .get(field_name)
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "standalone AOT runtime link config {} is missing array field {field_name}",
                path.display()
            )
        })
        .iter()
        .map(|value| {
            value.as_str().unwrap_or_else(|| {
                panic!(
                    "standalone AOT runtime link config {} has non-string item in {field_name}",
                    path.display()
                )
            })
        })
        .map(str::to_owned)
        .collect()
}

fn read_link_config(path: &Path, target_os: &str) -> RuntimeLinkConfig {
    let config = parse_toml_file(path, "standalone AOT runtime link config");
    let format_version = config
        .get("format_version")
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| {
            panic!(
                "standalone AOT runtime link config {} is missing integer field format_version",
                path.display()
            )
        });
    if format_version != 1 {
        panic!(
            "unsupported standalone AOT runtime link config version {} in {}",
            format_version,
            path.display()
        );
    }
    let target_table = config
        .get("targets")
        .and_then(toml::Value::as_table)
        .and_then(|targets| targets.get(target_os))
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| {
            panic!(
                "standalone AOT runtime link config {} has no targets.{target_os} table",
                path.display()
            )
        });
    RuntimeLinkConfig {
        system_libraries: read_string_array(target_table, path, "system_libraries"),
        frameworks: read_string_array(target_table, path, "frameworks"),
    }
}

/// Emits all native linker inputs required by direct Cargo consumers.
///
/// This must not be invoked in declared-link mode: doing so would cause both
/// Cargo and the enclosing Bazel link graph to provide the runtime dependency.
fn emit_native_link_directives() {
    let inputs = resolve_runtime_inputs();
    let archive_path = &inputs.archive_path;
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

    if !inputs.link_config_path.is_file() {
        panic!(
            "standalone AOT runtime link config does not exist: {}",
            inputs.link_config_path.display()
        );
    }

    let archive_dir = archive_path.parent().unwrap_or_else(|| {
        panic!(
            "standalone AOT runtime archive has no parent directory: {}",
            archive_path.display()
        )
    });
    println!("cargo:rerun-if-changed={}", archive_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        inputs.link_config_path.display()
    );
    println!("cargo:rustc-link-search=native={}", archive_dir.display());
    println!("cargo:rustc-link-lib=static=xls_aot_runtime");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let link_config = read_link_config(&inputs.link_config_path, target_os.as_str());
    for library in link_config.system_libraries {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    for framework in link_config.frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

/// Applies the selected native-link ownership contract to this crate build.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={XLS_AOT_RUNTIME_PATH_ENV}");
    println!("cargo:rerun-if-env-changed={XLS_AOT_RUNTIME_LINK_CONFIG_PATH_ENV}");
    println!("cargo:rerun-if-env-changed={XLS_AOT_RUNTIME_LINK_MODE_ENV}");
    println!("cargo:rerun-if-env-changed={XLSYNTH_ARTIFACT_CONFIG_ENV}");
    println!("cargo:rustc-check-cfg=cfg(xls_aot_runtime_declared_link)");

    match parse_link_directive_mode() {
        LinkDirectiveMode::Native => emit_native_link_directives(),
        LinkDirectiveMode::Declared => {
            println!("cargo:rustc-cfg=xls_aot_runtime_declared_link");
            println!(
                "cargo:info=Skipping native link directives because {XLS_AOT_RUNTIME_LINK_MODE_ENV}=declared"
            );
        }
    }
}
