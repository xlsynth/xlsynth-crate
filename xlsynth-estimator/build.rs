// SPDX-License-Identifier: Apache-2.0

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let descriptor_path = std::path::PathBuf::from(out_dir.clone()).join("descriptors.bin");

    prost_build::Config::new()
        // AlmaLinux 9's `protobuf-compiler` is old enough that proto3 `optional`
        // fields require this flag.
        .protoc_arg("--experimental_allow_proto3_optional")
        .out_dir(out_dir.clone())
        .file_descriptor_set_path(descriptor_path)
        .compile_protos(
            &["proto/estimator_model.proto", "proto/sample_node.proto"],
            &["proto"],
        )
        .expect("compilation of proto");
    println!("cargo:rerun-if-changed=proto/estimator_model.proto");
    println!("cargo:rerun-if-changed=proto/sample_node.proto");
}
