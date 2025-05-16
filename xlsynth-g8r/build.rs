// SPDX-License-Identifier: Apache-2.0

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let descriptor_path = std::path::Path::new(&out_dir).join("liberty.bin");
    prost_build::Config::new()
        .file_descriptor_set_path(&descriptor_path)
        .compile_protos(&["proto/liberty.proto"], &["proto"])
        .expect("Failed to compile proto");
    println!("cargo:rerun-if-changed=proto/liberty.proto");
}
