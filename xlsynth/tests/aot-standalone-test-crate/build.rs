// SPDX-License-Identifier: Apache-2.0

use xlsynth::aot_builder::{emit_aot_module_from_ir_text, AotBuildSpec};

fn main() {
    let add_one_ir = r#"package standalone_aot_tests

top fn add_one(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1)
  ret out: bits[8] = add(x, one)
}
"#;
    let output = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "add_one",
        ir_text: add_one_ir,
        top: "add_one",
    })
    .expect("standalone add-one AOT compile should succeed");
    println!(
        "cargo:rustc-env=XLSYNTH_STANDALONE_AOT_ADD_ONE_RS={}",
        output.rust_file.display()
    );
}
