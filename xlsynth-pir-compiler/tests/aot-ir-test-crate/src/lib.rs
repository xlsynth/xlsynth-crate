// SPDX-License-Identifier: Apache-2.0

//! Native PIR compiler AOT package emitted from checked-in IR plus type
//! metadata.

pub mod native_aot_tests_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/native_aot_tests_typed_ir_pir_aot_package.rs"
    ));
}
