// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use xlsynth_pir_compiler::aot::TypedIrAotPackageBuilder;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let metadata_path = manifest_dir.join("src/native_aot_tests_aot_metadata.json");

    let generated = TypedIrAotPackageBuilder::new("native_aot_tests", metadata_path)
        .build()
        .unwrap_or_else(|error| panic!("typed native PIR AOT package should build: {error}"));
    assert!(
        generated.object_file.is_file(),
        "typed IR AOT package should emit one native object"
    );

    println!("cargo:rerun-if-changed=build.rs");
}
