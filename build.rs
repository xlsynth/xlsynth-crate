// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;

const DSO_VERSION_TAG: &str = "v0.0.58";

fn main() {
    // URL of the DSO release on GitHub
    #[allow(unused_assignments)]
    let mut dso_extension: Option<&'static str> = None;
    #[allow(unused_assignments)]
    let mut dso_download_suffix = "";

    #[cfg(target_os = "macos")]
    {
        dso_extension = Some("dylib");
    }

    #[cfg(target_os = "linux")]
    {
        dso_extension = Some("so");
        dso_download_suffix = "-ubuntu20.04";
    }

    let url_base = format!(
        "https://github.com/xlsynth/xlsynth/releases/download/{}/",
        DSO_VERSION_TAG
    );
    let dso_url = format!("{url_base}libxls{dso_download_suffix}.{}", dso_extension.unwrap());
    let tarball_url = format!("{url_base}/dslx_stdlib.tar.gz");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dso_name = format!(
        "libxls-{DSO_VERSION_TAG}.{}",
        dso_extension.unwrap()
    );
    let dso_path = PathBuf::from(&out_dir).join(&dso_name);

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

	#[cfg(target_os = "macos")]
	{
	    println!("cargo:info=Fixing DSO id: to {}", dso_name);
            // Download the DSO id
            let status = Command::new("install_name_tool")
		.arg("-id")
		.arg(format!("@rpath/{}", &dso_name))
		.arg(&dso_path)
		.status()
		.expect("Failed to fix DSO id");
	    
            if !status.success() {
		panic!("Fixing DSO id failed with status: {:?}", status);
            }
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
    println!("cargo:rustc-link-lib=dylib=xls-{}", DSO_VERSION_TAG);
    println!("cargo:rustc-env=XLS_DSO_VERSION_TAG={}", DSO_VERSION_TAG);
    println!("cargo:rustc-env=XLS_DSO_PATH={}", out_dir);
    println!(
        "cargo:rustc-env=DSLX_STDLIB_PATH={}/xls/dslx/stdlib/",
        stdlib_path.display()
    );
}
