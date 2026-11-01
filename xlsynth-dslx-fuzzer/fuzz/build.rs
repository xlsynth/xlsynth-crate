// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Links optional native instrumentation without changing ordinary fuzz builds.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rustc-check-cfg=cfg(xlsynth_dslx_native)");
    println!("cargo:rerun-if-env-changed=DSLX_FUZZ_NATIVE_CONFIG");
    let Some(config_path) = std::env::var_os("DSLX_FUZZ_NATIVE_CONFIG") else {
        return Ok(());
    };
    if std::env::var("CARGO_CFG_TARGET_OS")? != "linux" {
        return Err("native DSLX feedback currently requires Linux ELF libraries".into());
    }
    let config_path = PathBuf::from(config_path).canonicalize()?;
    println!("cargo:rerun-if-changed={}", config_path.display());
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let library = PathBuf::from(
        config["library"]
            .as_str()
            .ok_or("native config requires library")?,
    )
    .canonicalize()?;
    let stdlib = PathBuf::from(
        config["stdlib"]
            .as_str()
            .ok_or("native config requires stdlib")?,
    )
    .canonicalize()?;
    if library.file_name().and_then(|name| name.to_str()) != Some("libdslx_fuzz_native.so")
        || !library.is_file()
    {
        return Err("native library must name an existing libdslx_fuzz_native.so".into());
    }
    if !stdlib.join("std.x").is_file() {
        return Err("native stdlib must contain std.x".into());
    }
    let directory = library.parent().ok_or("native library has no directory")?;
    for path in [&library, &stdlib] {
        let path = path.to_str().ok_or("native build paths must be UTF-8")?;
        if path
            .chars()
            .any(|character| matches!(character, '\n' | '\r' | '\0'))
        {
            return Err("native build paths cannot contain line breaks or NUL".into());
        }
    }
    println!("cargo:rerun-if-changed={}", library.display());
    println!("cargo:rustc-cfg=xlsynth_dslx_native");
    println!(
        "cargo:rustc-env=DSLX_FUZZ_NATIVE_STDLIB={}",
        stdlib.display()
    );
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=dylib=dslx_fuzz_native");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
    // The DSO's compiler-generated constructors/callbacks use this executable's
    // libFuzzer runtime, so its registrations feed the actual mutation loop.
    println!("cargo:rustc-link-arg=-Wl,--export-dynamic-symbol=__sanitizer_cov_*");
    println!("cargo:rustc-link-arg=-Wl,--undefined=__sanitizer_cov_8bit_counters_init");
    Ok(())
}
