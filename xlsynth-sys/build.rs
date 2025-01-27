// SPDX-License-Identifier: Apache-2.0

use std::io::BufRead;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const RELEASE_LIB_VERSION_TAG: &str = "v0.0.126";

struct DsoInfo {
    extension: &'static str,
    lib_suffix: &'static str,
}

impl DsoInfo {
    fn get_dso_filename(&self) -> String {
        format!(
            "libxls-{RELEASE_LIB_VERSION_TAG}-{}.{}",
            self.lib_suffix, self.extension
        )
    }

    fn get_dso_name(&self) -> String {
        format!("xls-{RELEASE_LIB_VERSION_TAG}-{}", self.lib_suffix)
    }

    fn get_dso_url(&self, url_base: &str) -> String {
        format!("{url_base}libxls-{}.{}", self.lib_suffix, self.extension)
    }
}

fn is_rocky() -> bool {
    // Define the path to /etc/os-release
    let os_release_path = Path::new("/etc/os-release");

    // Check if the file exists
    if !os_release_path.exists() {
        println!("cargo:info=OS release path does not exist");
        return false;
    }

    // Open the file
    let file = std::fs::File::open(os_release_path);
    if let Ok(file) = file {
        // Read through the lines in the file
        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            if let Ok(line) = line {
                // Check if the line contains `ID="rocky"`
                if line.contains("ID=\"rocky\"") {
                    return true;
                }
            }
        }
        println!("cargo:info=Did not find rocky ID line in OS release data");
    } else {
        println!("cargo:info=Could not open OS release data file");
    }

    // Return false if `ID="rocky"` is not found
    false
}

fn get_dso_info() -> DsoInfo {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let extension = match target_os.as_str() {
        "macos" => "dylib",
        "linux" => "so",
        _ => panic!("Unhandled target_os: {:?}", target_os),
    };

    let lib_suffix = match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "x86_64") => "x64",
        ("macos", "aarch64") => "arm64",
        ("linux", "x86_64") => {
            if is_rocky() {
                "rocky8"
            } else {
                "ubuntu2004"
            }
        }
        _ => panic!(
            "Unhandled combination; target_os: {} target_arch: {}",
            target_os, target_arch
        ),
    };

    DsoInfo {
        extension,
        lib_suffix,
    }
}

/// Downloads the dynamic shared object for XLS from the release page if it does
/// not already exist.
fn download_dso_if_dne(url_base: &str, out_dir: &str) -> DsoInfo {
    let dso_info: DsoInfo = get_dso_info();
    let dso_url = dso_info.get_dso_url(url_base);
    let dso_path = PathBuf::from(&out_dir).join(&dso_info.get_dso_filename());

    // Check if the DSO has already been downloaded
    if dso_path.exists() {
        println!(
            "cargo:info=DSO already downloaded to: {}",
            dso_path.display()
        );
        return dso_info;
    }

    println!(
        "cargo:info=Downloading DSO from: {} to {}",
        dso_url,
        dso_path.display()
    );

    // Download the DSO
    let status = Command::new("curl")
        .arg("-L")
        .arg("--fail")
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

    if cfg!(target_os = "macos") {
        let dso_filename = dso_info.get_dso_filename();
        println!("cargo:info=Fixing DSO id: to {}", dso_filename);
        // Download the DSO id
        let status = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@rpath/{}", &dso_filename))
            .arg(&dso_path)
            .status()
            .expect("Failed to fix DSO id");

        if !status.success() {
            panic!("Fixing DSO id failed with status: {:?}", status);
        }
    }

    dso_info
}

fn download_stdlib_if_dne(url_base: &str, out_dir: &str) -> PathBuf {
    let stdlib_path =
        PathBuf::from(&out_dir).join(format!("dslx_stdlib_{}", RELEASE_LIB_VERSION_TAG));
    if stdlib_path.exists() {
        println!(
            "cargo:info=DSLX stdlib path already downloaded to: {}",
            stdlib_path.display()
        );
        return stdlib_path;
    }
    let tarball_path = PathBuf::from(&out_dir).join("dslx_stdlib.tar.gz");
    let tarball_url = format!("{url_base}/dslx_stdlib.tar.gz");
    let status = Command::new("curl")
        .arg("-L")
        .arg("--fail")
        .arg("-o")
        .arg(&tarball_path)
        .arg(&tarball_url)
        .status()
        .expect("Failed to download DSO");

    if !status.success() {
        // Remove the output file path if it got created -- if not it's ok.
        let _ = std::fs::remove_file(&tarball_path);
        panic!(
            "Download of {tarball_url:?} failed with status: {:?}",
            status
        );
    }
    let tar_gz = std::fs::File::open(tarball_path).unwrap();
    let tar = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(&stdlib_path).unwrap();
    stdlib_path
}

fn main() {
    // Detect if building on docs.rs
    if std::env::var("DOCS_RS").is_ok() {
        println!("cargo:warning=Skipping dynamic library download on docs.rs");
        println!("cargo:rustc-env=XLS_DSO_PATH=/does/not/exist/libxls.so");
        println!("cargo:rustc-env=DSLX_STDLIB_PATH=/does/not/exist/stdlib/");
        return;
    }

    let url_base = format!(
        "https://github.com/xlsynth/xlsynth/releases/download/{}/",
        RELEASE_LIB_VERSION_TAG
    );
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let stdlib_path: PathBuf = download_stdlib_if_dne(&url_base, &out_dir);
    let stdlib_path_full = format!("{}/xls/dslx/stdlib/", stdlib_path.display());
    println!("cargo:rustc-env=DSLX_STDLIB_PATH={stdlib_path_full}");

    if std::env::var("DEV_XLS_DSO_WORKSPACE").is_ok() {
        // This points at a XLS workspace root.
        // Grab the DSO from the build artifacts.
        let workspace = std::env::var("DEV_XLS_DSO_WORKSPACE").unwrap();
        // The DSO is in the workspace subdir bazel-bin/xls/public/libxls.so
        const DSO_RELPATH: &str = if cfg!(target_os = "macos") {
            "bazel-bin/xls/public/libxls.dylib"
        } else {
            "bazel-bin/xls/public/libxls.so"
        };
        let dso_path = PathBuf::from(workspace).join(DSO_RELPATH);
        let dso_info = DsoInfo {
            extension: "so",
            lib_suffix: "ubuntu2004",
        };
        let dso_filename = dso_info.get_dso_filename();
        let dso_dest = PathBuf::from(&out_dir).join(&dso_filename);

        // Symlink to the artifact in the workspace.
        assert!(
            cfg!(unix),
            "DEV_XLS_DSO_WORKSPACE env var only supported in UNIX-like environments"
        );
        std::fs::remove_file(&dso_dest).ok();

        println!(
            "cargo:info=Symlinking DSO from workspace; src: {} dst symlink: {}",
            dso_path.display(),
            dso_dest.display()
        );

        #[cfg(unix)]
        std::os::unix::fs::symlink(&dso_path, &dso_dest).unwrap();

        println!(
            "cargo:info=Using DSO from workspace; src: {} dst symlink: {}",
            dso_path.display(),
            dso_dest.display()
        );

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir);
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib={}", dso_info.get_dso_name());
        println!("cargo:rustc-env=XLS_DSO_PATH={}", out_dir);
        return;
    }

    let dso_info = download_dso_if_dne(&url_base, &out_dir);

    // Ensure the DSO is copied to the correct location
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=dylib={}", dso_info.get_dso_name());
    println!("cargo:rustc-env=XLS_DSO_PATH={}", out_dir);
}
