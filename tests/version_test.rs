// SPDX-License-Identifier: Apache-2.0

//! Test that the current package version reflected in Cargo.toml is more than the released
//! version -- this help make sure we bump the version appropriately after a release is performed.

use curl::easy::Easy;

const CRATE_NAME: &str = "xlsynth";
const USER_AGENT: &str = "xlsynth_crate_unit_test";

fn fetch_latest_version(crate_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("https://crates.io/api/v1/crates/{}", crate_name);
    let mut data = Vec::new();
    let mut easy = Easy::new();
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
    log::info!("Response: {:?}", response);
    let newest_version = response["crate"]["newest_version"].as_str();
    let latest_version = newest_version
        .ok_or(format!("Failed to parse latest version: {:?}", newest_version))?
        .to_string();
    Ok(latest_version)
}

fn fetch_local_version() -> Result<String, Box<dyn std::error::Error>> {
    let cargo_toml = std::fs::read_to_string("Cargo.toml")?;
    let cargo_toml: toml::Value = toml::from_str(&cargo_toml)?;
    let version = cargo_toml["package"]["version"]
        .as_str()
        .ok_or(format!("Failed to parse local version: {}", cargo_toml["package"]["version"]))?
        .to_string();
    Ok(version)
}

#[test]
fn test_version_is_greater_than_latest_released() {
    let latest_version = fetch_latest_version(CRATE_NAME).expect("Failed to fetch latest version");
    let local_version = fetch_local_version().expect("Failed to fetch local version");

    let latest_semver = semver::Version::parse(&latest_version).expect("Invalid latest version");
    let local_semver = semver::Version::parse(&local_version).expect("Invalid local version");

    assert!(
        local_semver > latest_semver,
        "Local version {} is not greater than the latest version {}",
        local_version,
        latest_version
    );
}