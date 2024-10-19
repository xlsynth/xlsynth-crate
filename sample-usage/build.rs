// SPDX-License-Identifier: Apache-2.0

use xlsynth::dslx;
use xlsynth::dslx_bridge;
use xlsynth::rust_bridge_builder;

fn x_path_to_rs_filename(path: &std::path::Path) -> String {
    let mut out = path.file_stem().unwrap().to_str().unwrap().to_string();
    out.push_str(".rs");
    out
}

fn x_path_to_bridge(relpath: &str) -> std::path::PathBuf {
    let mut import_data = dslx::ImportData::default();
    let path = std::path::PathBuf::from(relpath);
    let dslx =
        std::fs::read_to_string(&path).expect(&format!("DSLX file should be readable: {path:?}"));

    // Generate the bridge code.
    let mut builder = rust_bridge_builder::RustBridgeBuilder::new();
    dslx_bridge::convert_leaf_module(&mut import_data, &dslx, &path, &mut builder)
        .expect("expect bridge building success");
    let sample_with_enum_def_rs = builder.build();

    // Write this out to sample_with_enum_def.rs in the OUT_DIR.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR should be set");
    let out_path = std::path::PathBuf::from(out_dir).join(x_path_to_rs_filename(&path));
    std::fs::write(&out_path, sample_with_enum_def_rs).unwrap();
    out_path
}

fn main() {
    let sample_with_enum_def_rs_path = x_path_to_bridge("src/sample_with_enum_def.x");
    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_ENUM_DEF_PATH={}",
        sample_with_enum_def_rs_path.display()
    );

    let sample_with_enum_def_rs_path = x_path_to_bridge("src/structure_zoo.x");
    println!(
        "cargo:rustc-env=DSLX_STRUCTURE_ZOO={}",
        sample_with_enum_def_rs_path.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
