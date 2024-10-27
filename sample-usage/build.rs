// SPDX-License-Identifier: Apache-2.0

fn main() {
    let sample_with_enum_def_rs_path =
        xlsynth::x_path_to_rs_bridge_via_env("src/sample_with_enum_def.x");
    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_ENUM_DEF_PATH={}",
        sample_with_enum_def_rs_path.display()
    );

    let rs_path = xlsynth::x_path_to_rs_bridge_via_env("../xlsynth/tests/structure_zoo.x");
    println!("cargo:rustc-env=DSLX_STRUCTURE_ZOO={}", rs_path.display());

    let rs_path = xlsynth::x_path_to_rs_bridge_via_env("src/sample_with_struct_def.x");
    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_STRUCT_DEF={}",
        rs_path.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
