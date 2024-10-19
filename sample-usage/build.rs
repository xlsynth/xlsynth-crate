// SPDX-License-Identifier: Apache-2.0

use xlsynth::dslx;
use xlsynth::dslx_bridge;

fn main() {
    let mut import_data = dslx::ImportData::default();
    let path = std::path::PathBuf::from("src/sample_with_enum_def.x");
    let sample_with_enum_def = std::fs::read_to_string(&path).unwrap();

    // Generate the bridge code.
    let mut builder = dslx_bridge::RustBridgeBuilder::new();
    dslx_bridge::convert_leaf_module(&mut import_data, &sample_with_enum_def, &path, &mut builder)
        .expect("expect bridge building success");
    let sample_with_enum_def_rs = builder.build();

    // Write this out to sample_with_enum_def.rs in the OUT_DIR.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = std::path::PathBuf::from(out_dir).join("sample_with_enum_def.rs");
    std::fs::write(&out_path, sample_with_enum_def_rs).unwrap();

    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_ENUM_DEF_PATH={}",
        out_path.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
