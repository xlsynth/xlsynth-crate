// SPDX-License-Identifier: Apache-2.0

//! Test that the current package version reflected in Cargo.toml is more than
//! the released version -- this help make sure we bump the version
//! appropriately after a release is performed.
//!
//! This is done for all released crates in the workspace.

use curl::easy::Easy;
use std::collections::BTreeMap;

const USER_AGENT: &str = "xlsynth_crate_unit_test";

fn get_workspace_root() -> std::path::PathBuf {
    let workspace_dir = cargo_metadata::MetadataCommand::new()
        .exec()
        .unwrap()
        .workspace_root;
    workspace_dir.into()
}

/// Fetches the local version of a package given the path to a `Cargo.toml`
/// file.
fn fetch_local_version(dirpath: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let cargo_toml = std::fs::read_to_string(dirpath.join("Cargo.toml"))?;
    let cargo_toml: toml::Value = toml::from_str(&cargo_toml)?;
    let version = cargo_toml["package"]["version"]
        .as_str()
        .ok_or_else(|| {
            format!(
                "Failed to parse local version: {}",
                cargo_toml["package"]["version"]
            )
        })?
        .to_string();
    Ok(version)
}

/// Builds a map from (major, minor) -> latest released patch for that pair.
///
/// Yanked versions are ignored.
fn get_latest_patch_versions(
    crate_name: &str,
) -> Result<BTreeMap<(u64, u64), u64>, Box<dyn std::error::Error>> {
    let url = format!("https://crates.io/api/v1/crates/{crate_name}");
    let mut data = Vec::new();
    let mut easy = curl::easy::Easy::new();
    easy.url(&url)?;
    easy.useragent(USER_AGENT)?;
    {
        let mut transfer = easy.transfer();
        transfer.write_function(|new_data| {
            data.extend_from_slice(new_data);
            Ok(new_data.len())
        })?;
        transfer.perform()?;
    }
    let response: serde_json::Value = serde_json::from_slice(&data)?;
    log::trace!("Response: {response:?}");
    let versions = response["versions"].as_array().ok_or_else(|| {
        std::io::Error::other("Failed to parse versions array from crates.io response")
    })?;

    let mut mm_to_patch: BTreeMap<(u64, u64), u64> = BTreeMap::new();
    for v in versions {
        let is_yanked = v["yanked"].as_bool().unwrap_or(false);
        if is_yanked {
            continue;
        }
        let num = match v["num"].as_str() {
            Some(s) => s,
            None => continue,
        };
        let parsed = match semver::Version::parse(num) {
            Ok(ver) => ver,
            Err(_) => continue,
        };
        let key = (parsed.major, parsed.minor);
        let patch = parsed.patch;
        mm_to_patch
            .entry(key)
            .and_modify(|p| *p = (*p).max(patch))
            .or_insert(patch);
    }
    log::debug!("Latest patch versions: {mm_to_patch:?}");
    Ok(mm_to_patch)
}

fn validate_local_version_is_latest_patch_version(
    crate_name: &str,
    workspace_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Fetch the newest version for logging continuity with previous behavior.
    let local_version = fetch_local_version(workspace_path)?;
    let local_semver = semver::Version::parse(&local_version)?;

    log::info!("crate: {crate_name} local_version: {local_version}");
    let latest_patches = get_latest_patch_versions(crate_name)?;

    if let Some(max_patch) = latest_patches.get(&(local_semver.major, local_semver.minor)) {
        if local_semver.patch > *max_patch {
            Ok(())
        } else {
            Err(Box::new(std::io::Error::other(format!(
                "Local version {local_version} must have a patch greater than any released {major}.{minor}.x (max released patch is {max_patch})",
                major = local_semver.major,
                minor = local_semver.minor,
                max_patch = max_patch
            ))))
        }
    } else {
        // No released versions for this major.minor; vacuously the latest patch.
        Ok(())
    }
}

#[test]
fn test_xlsynth_crate_version() {
    let _ = env_logger::builder().is_test(true).try_init();
    if std::env::var("CARGO_NET_OFFLINE").is_ok() {
        eprintln!("CARGO_NET_OFFLINE set - skipping network dependent test");
        return;
    }
    let workspace_root = get_workspace_root();
    let workspace_path = workspace_root.join("xlsynth");
    validate_local_version_is_latest_patch_version("xlsynth", workspace_path.as_path()).unwrap();
}

#[test]
fn test_xlsynth_sys_crate_version() {
    let _ = env_logger::builder().is_test(true).try_init();
    if std::env::var("CARGO_NET_OFFLINE").is_ok() {
        eprintln!("CARGO_NET_OFFLINE set - skipping network dependent test");
        return;
    }
    let workspace_root = get_workspace_root();
    let workspace_path = workspace_root.join("xlsynth-sys");
    validate_local_version_is_latest_patch_version("xlsynth-sys", workspace_path.as_path())
        .unwrap();
}

#[test]
fn test_xlsynth_driver_crate_version() {
    let _ = env_logger::builder().is_test(true).try_init();
    if std::env::var("CARGO_NET_OFFLINE").is_ok() {
        eprintln!("CARGO_NET_OFFLINE set - skipping network dependent test");
        return;
    }
    let workspace_root = get_workspace_root();
    let workspace_path = workspace_root.join("xlsynth-driver");
    validate_local_version_is_latest_patch_version("xlsynth-driver", workspace_path.as_path())
        .unwrap();
}

#[test]
fn test_crate_versions_are_equal() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = get_workspace_root();
    let released_crate_dirs = [
        workspace_root.join("xlsynth"),
        workspace_root.join("xlsynth-sys"),
        workspace_root.join("xlsynth-driver"),
    ];
    let mut local_versions = vec![];
    for dir in released_crate_dirs {
        local_versions.push(fetch_local_version(dir.as_path()).unwrap());
    }
    // Check all the local versions are the same.
    let first_version = &local_versions[0];
    for version in &local_versions[1..] {
        assert_eq!(version, first_version);
    }
}
