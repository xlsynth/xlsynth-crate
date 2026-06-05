// SPDX-License-Identifier: Apache-2.0

//! Native PIR compiler AOT wrappers emitted from PIR text by the build script.

pub mod add_one_aot {
    include!(concat!(env!("OUT_DIR"), "/add_one_pir_aot_wrapper.rs"));
}

pub mod add_inputs_aot {
    include!(concat!(env!("OUT_DIR"), "/add_inputs_pir_aot_wrapper.rs"));
}

pub mod compound_shapes_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/compound_shapes_pir_aot_wrapper.rs"
    ));
}

pub mod empty_tuple_aot {
    include!(concat!(env!("OUT_DIR"), "/empty_tuple_pir_aot_wrapper.rs"));
}

pub mod wide_runtime_ops_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/wide_runtime_ops_pir_aot_wrapper.rs"
    ));
}

pub mod events_aot {
    include!(concat!(env!("OUT_DIR"), "/events_pir_aot_wrapper.rs"));
}

pub mod assumed_in_bounds_aot {
    include!(concat!(
        env!("OUT_DIR"),
        "/assumed_in_bounds_pir_aot_wrapper.rs"
    ));
}
