use std::path::PathBuf;
use std::process::Command;

const VERSION_TAG: &str = "v0.0.10";

fn main() {
    // URL of the DSO release on GitHub
    let dso_url = format!(
        "https://github.com/xlsynth/xlsynth/releases/download/{}/libxls.dylib",
        VERSION_TAG
    );
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dso_path = PathBuf::from(&out_dir).join(format!("libxls-{}.dylib", VERSION_TAG));

    // Check if the DSO has already been downloaded
    if !dso_path.exists() {
        // Download the DSO
        let status = Command::new("curl")
            .arg("-L")
            .arg("-o")
            .arg(&dso_path)
            .arg(dso_url)
            .status()
            .expect("Failed to download DSO");

        if !status.success() {
            panic!("Download failed with status: {:?}", status);
        }
    }

    // Ensure the DSO is copied to the correct location
    println!("cargo:rustc-link-search=native={}", out_dir);
}
