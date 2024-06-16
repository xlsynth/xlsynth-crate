// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;

const VERSION_TAG: &str = "v0.0.41";

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

    let dso_url = format!(
        "https://github.com/xlsynth/xlsynth/releases/download/{}/libxls.{}",
        VERSION_TAG,
        dso_extension.unwrap()
    );
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dso_path =
        PathBuf::from(&out_dir).join(format!("libxls-{}.{}", VERSION_TAG, dso_extension.unwrap()));

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

    // Ensure the DSO is copied to the correct location
    println!("cargo:rustc-link-search=native={}", out_dir);
}
