// SPDX-License-Identifier: Apache-2.0

use sha2::Digest;
use std::io::BufRead;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const RELEASE_LIB_VERSION_TAG: &str = "v0.0.168";

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

/// Performs a "high integrity" download of a file from a URL by doing the
/// following:
/// - Downloading a checksum file first.
/// - Downloading the file not to the target destination path but to a temporary
///   location.
/// - Verifying the checksum of the downloaded file against the checksum file.
/// - If the checksum is correct, move the file to the target destination path.
/// - If the checksum is incorrect, return an error.
///
/// The checksum URL is assumed to be the original URL with a `.sha256` suffix.
///
/// `out_path` should be a file path where we ultimately want to place the
/// downloaded file, not a directory path.
fn high_integrity_download(
    url: &str,
    out_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let env_tmp_dir = PathBuf::from(std::env::temp_dir());
    assert!(
        env_tmp_dir.exists(),
        "environment-based temp directory {} does not exist",
        env_tmp_dir.display()
    );

    let tmp_dir = env_tmp_dir.join(format!("xlsynth-sys-tmp-{}", std::process::id()));
    // Make the temp dir.
    std::fs::create_dir_all(&tmp_dir).expect("create temp directory should succeed");

    // Download the sha256 checksum file to the temp directory.
    let checksum_url = format!("{}.sha256", url);

    let filename = out_path.file_name().unwrap();
    let checksum_path = tmp_dir.join(format!("{}.sha256", filename.to_str().unwrap()));
    println!(
        "cargo:info=downloading checksum from {} to {}",
        checksum_url,
        checksum_path.display()
    );

    download_file_via_https(&checksum_url, &checksum_path)?;

    let want_checksum_str = std::fs::read_to_string(&checksum_path)?;
    let want_checksum_str = want_checksum_str.split_whitespace().next().unwrap();
    println!(
        "cargo:info=want checksum for {} to be {}",
        filename.to_str().unwrap(),
        want_checksum_str
    );

    // Download the URL with the file itself to the temp directory.
    let tmp_out_path = tmp_dir.join(filename);
    println!(
        "cargo:info=downloading file from {} to {}",
        url,
        tmp_out_path.display()
    );

    download_file_via_https(url, &tmp_out_path)?;

    if !tmp_out_path.exists() {
        return Err(format!(
            "Failed to download file {}; file does not exist after request completed",
            tmp_out_path.display()
        )
        .into());
    }

    println!(
        "cargo:info=downloaded file to {}; verifying checksum...",
        tmp_out_path.display()
    );
    let sha256 = sha2::Sha256::digest(std::fs::read(&tmp_out_path)?);
    let got_checksum_str = format!("{:x}", sha256);

    if want_checksum_str != got_checksum_str {
        return Err(format!(
            "Checksum mismatch for file: {} want: {} got: {}",
            out_path.display(),
            want_checksum_str,
            got_checksum_str
        )
        .into());
    }

    // Checksum matches expectation, now we can move the file to its target
    // destination.
    println!(
        "cargo:info=checksums match; copying file from {} to {}",
        tmp_out_path.display(),
        out_path.display()
    );
    assert!(
        tmp_out_path.exists(),
        "temp file {} does not exist",
        tmp_out_path.display()
    );

    let out_path_dir = out_path.parent().unwrap();
    assert!(
        out_path_dir.exists(),
        "output directory {} does not exist",
        out_path_dir.display()
    );

    std::fs::copy(&tmp_out_path, out_path)?;
    std::fs::remove_file(&tmp_out_path)?;
    Ok(())
}

/// Simple helper that downloads a file to the given path using HTTPS with
/// reqwest.
fn download_file_via_https(
    url: &str,
    dest: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(url)?.error_for_status()?;
    let mut file = std::fs::File::create(dest)?;
    let bytes = response.bytes()?;
    std::io::copy(&mut bytes.as_ref(), &mut file)?;
    Ok(())
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
    high_integrity_download(&dso_url, &dso_path).expect("download of DSO should succeed");

    if cfg!(target_os = "macos") {
        let dso_filename = dso_info.get_dso_filename();
        println!("cargo:info=Fixing DSO id: to {}", dso_filename);
        // Fix the DSO id so it can be found via the rpath.
        let status = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@rpath/{}", &dso_filename))
            .arg(&dso_path)
            .status()
            .expect("fixing DSO id should succeed");

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
    high_integrity_download(&tarball_url, &tarball_path)
        .expect("download of stdlib tarball should succeed");

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

    // Ensure the out directory exists.
    if !std::fs::metadata(&out_dir).unwrap().is_dir() {
        std::fs::create_dir_all(&out_dir).unwrap();
    }

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
