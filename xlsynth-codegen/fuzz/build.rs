// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // libFuzzer looks up this optional hook dynamically on macOS. Retain it
    // even for ordinary `cargo build/test`, without cargo-fuzz's link flags.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-bins=-Wl,-u,_LLVMFuzzerInitialize");
    }
}
