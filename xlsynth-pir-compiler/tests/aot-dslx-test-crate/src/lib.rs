// SPDX-License-Identifier: Apache-2.0

//! Native PIR compiler AOT package emitted from DSLX by the build script.

pub mod native_dslx_tests_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/native_dslx_tests_typed_dslx_pir_aot_package.rs"
    ));
}
