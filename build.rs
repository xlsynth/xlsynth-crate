// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;

const DSO_VERSION_TAG: &str = "v0.0.51";

fn main() {
    // URL of the DSO release on GitHub
    #[allow(unused_assignments)]
    let mut dso_extension: Option<&'static str> = None;

    #[cfg(target_os = "macos")]
    {
        dso_extension = Some("dylib");
    }

    #[cfg(target_os = "linux")]
    {
        dso_extension = Some("so");
    }

    let url_base = format!(
        "https://github.com/xlsynth/xlsynth/releases/download/{}/",
        DSO_VERSION_TAG
    );
    let dso_url = format!("{}libxls.{}", url_base, dso_extension.unwrap());
    let tarball_url = format!("{}/dslx_stdlib.tar.gz", url_base);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dso_path = PathBuf::from(&out_dir).join(format!(
        "libxls-{}.{}",
        DSO_VERSION_TAG,
        dso_extension.unwrap()
    ));

    // Check if the DSO has already been downloaded
    if dso_path.exists() {
        println!(
            "cargo:info=DSO already downloaded to: {}",
            dso_path.display()
        );
    } else {
        println!(
            "cargo:info=Downloading DSO from: {} to {}",
            dso_url,
            dso_path.display()
        );

        // Download the DSO
        let status = Command::new("curl")
            .arg("-L")
            .arg("-o")
            .arg(&dso_path)
            .arg(dso_url)
            .status()
            .expect("Failed to download DSO");

        if !status.success() {
            // Remove the output file path.
            std::fs::remove_file(&dso_path).expect("Failed to remove file");
            panic!("Download failed with status: {:?}", status);
        }
    }

    let stdlib_path = PathBuf::from(&out_dir).join(format!("dslx_stdlib_{}", DSO_VERSION_TAG));
    if stdlib_path.exists() {
        println!(
            "cargo:info=DSLX stdlib path already downloaded to: {}",
            stdlib_path.display()
        );
    } else {
        let tarball_path = PathBuf::from(&out_dir).join("dslx_stdlib.tar.gz");
        let status = Command::new("curl")
            .arg("-L")
            .arg("-o")
            .arg(&tarball_path)
            .arg(tarball_url)
            .status()
            .expect("Failed to download DSO");

        if !status.success() {
            // Remove the output file path.
            std::fs::remove_file(&tarball_path).expect("Failed to remove file");
            panic!("Download failed with status: {:?}", status);
        }
        let tar_gz = std::fs::File::open(tarball_path).unwrap();
        let tar = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(tar);
        archive.unpack(&stdlib_path).unwrap();
    }

    // Ensure the DSO is copied to the correct location
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib={}-{}", "xls", DSO_VERSION_TAG);
    println!("cargo:rustc-env=XLS_DSO_VERSION_TAG={}", DSO_VERSION_TAG);
    println!("cargo:rustc-env=XLS_DSO_PATH={}", out_dir);    
    println!(
        "cargo:rustc-env=DSLX_STDLIB_PATH={}/xls/dslx/stdlib/",
        stdlib_path.display()
    );
}
