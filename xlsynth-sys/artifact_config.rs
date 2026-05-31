// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Resolved paths to the XLS artifacts supplied by the caller.
pub struct ArtifactPaths {
    pub dso_path: String,
    pub dslx_stdlib_path: String,
}

/// Linker inputs derived from an explicit XLS DSO path.
pub struct ExplicitDsoPath {
    pub link_name: String,
    pub path: PathBuf,
    pub parent_dir: PathBuf,
}

/// Validates the optional `XLSYNTH_ARTIFACT_CONFIG` environment value.
pub fn parse_artifact_config_env_path(
    config_path: Option<OsString>,
) -> Result<Option<PathBuf>, String> {
    let Some(config_path) = config_path else {
        return Ok(None);
    };
    let config_path = PathBuf::from(config_path);
    if config_path.is_absolute() {
        Ok(Some(config_path))
    } else {
        Err(format!(
            concat!(
                "XLSYNTH_ARTIFACT_CONFIG must be an absolute path; got relative path: {}.\n",
                "The dso_path and dslx_stdlib_path values inside that TOML may still be relative ",
                "to the TOML file's directory."
            ),
            config_path.display()
        ))
    }
}

/// Loads and resolves artifact paths relative to an artifact-config TOML file.
pub fn load_artifact_paths_from_config_path(config_path: &Path) -> Result<ArtifactPaths, String> {
    let config_contents = std::fs::read_to_string(config_path).map_err(|err| {
        format!(
            "Failed to read XLSYNTH_ARTIFACT_CONFIG {}: {}",
            config_path.display(),
            err
        )
    })?;
    let config_value: toml::Value = toml::from_str(&config_contents).map_err(|err| {
        format!(
            "Failed to parse XLSYNTH_ARTIFACT_CONFIG {} as TOML: {}",
            config_path.display(),
            err
        )
    })?;
    let config_table = config_value.as_table().ok_or_else(|| {
        format!(
            "XLSYNTH_ARTIFACT_CONFIG {} must contain a top-level TOML table",
            config_path.display()
        )
    })?;

    Ok(ArtifactPaths {
        dso_path: parse_artifact_config_path(config_table, config_path, "dso_path")?,
        dslx_stdlib_path: parse_artifact_config_path(
            config_table,
            config_path,
            "dslx_stdlib_path",
        )?,
    })
}

/// Validates an explicit shared-library path and derives its linker inputs.
pub fn validate_explicit_dso_path(dso_path: &Path) -> Result<ExplicitDsoPath, String> {
    let dso_dir = dso_path.parent().ok_or_else(|| {
        format!(
            "Explicit XLS DSO path must have a parent directory: {}",
            dso_path.display()
        )
    })?;
    let dso_name = dso_path.file_name().ok_or_else(|| {
        format!(
            "Explicit XLS DSO path must name a shared library file: {}",
            dso_path.display()
        )
    })?;
    let dso_name = dso_name.to_str().ok_or_else(|| {
        format!(
            "Explicit XLS DSO path must be valid UTF-8: {}",
            dso_path.display()
        )
    })?;
    let (dso_name, ext) = dso_name.rsplit_once('.').ok_or_else(|| {
        format!(
            "Explicit XLS DSO path must end with a shared library extension: {}",
            dso_path.display()
        )
    })?;
    match ext {
        "dylib" | "so" => {}
        _ => return Err(format!("Expected shared library extension: {:?}", ext)),
    }
    if !dso_name.starts_with("lib") {
        return Err(format!(
            "DSO name should start with 'lib'; dso_name: {:?}",
            dso_name
        ));
    }
    Ok(ExplicitDsoPath {
        link_name: dso_name[3..].to_string(),
        path: dso_path.to_path_buf(),
        parent_dir: dso_dir.to_path_buf(),
    })
}

fn parse_artifact_config_path(
    config_table: &toml::Table,
    config_path: &Path,
    key: &str,
) -> Result<String, String> {
    let raw_path = PathBuf::from(parse_artifact_config_string(
        config_table,
        config_path,
        key,
    )?);
    let resolved_path = if raw_path.is_absolute() {
        raw_path
    } else {
        resolve_artifact_config_dir(config_path)?.join(raw_path)
    };
    Ok(resolved_path.display().to_string())
}

fn parse_artifact_config_string(
    config_table: &toml::Table,
    config_path: &Path,
    key: &str,
) -> Result<String, String> {
    config_table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            format!(
                "XLSYNTH_ARTIFACT_CONFIG {} must define string key {:?}",
                config_path.display(),
                key
            )
        })
}

fn resolve_artifact_config_dir(config_path: &Path) -> Result<&Path, String> {
    config_path.parent().ok_or_else(|| {
        format!(
            "XLSYNTH_ARTIFACT_CONFIG should have a parent directory: {}",
            config_path.display()
        )
    })
}
