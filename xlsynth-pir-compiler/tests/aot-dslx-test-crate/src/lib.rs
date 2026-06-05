// SPDX-License-Identifier: Apache-2.0

//! Native PIR compiler AOT wrappers emitted from DSLX by the build script.

pub mod gizmo_frob_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/gizmo_frob_typed_dslx_pir_aot_wrapper.rs"
    ));
}

pub mod parametric_forms_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/parametric_forms_typed_dslx_pir_aot_wrapper.rs"
    ));
}

pub mod parametric_imports_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/parametric_imports_typed_dslx_pir_aot_wrapper.rs"
    ));
}

pub mod duplicate_widget_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/duplicate_widget_typed_dslx_pir_aot_wrapper.rs"
    ));
}

pub mod namespaced_doodle_package_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/namespaced_doodle_package_typed_dslx_pir_aot_package.rs"
    ));
}
