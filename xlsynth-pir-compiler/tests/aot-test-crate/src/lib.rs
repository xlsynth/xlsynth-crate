// SPDX-License-Identifier: Apache-2.0

//! Native PIR compiler AOT wrappers emitted by the build script.

pub mod add_one_aot {
    include!(env!("XLSYNTH_PIR_AOT_ADD_ONE_RS"));
}

pub mod add_inputs_aot {
    include!(env!("XLSYNTH_PIR_AOT_ADD_INPUTS_RS"));
}

pub mod compound_shapes_aot {
    include!(env!("XLSYNTH_PIR_AOT_COMPOUND_SHAPES_RS"));
}

pub mod empty_tuple_aot {
    include!(env!("XLSYNTH_PIR_AOT_EMPTY_TUPLE_RS"));
}

pub mod events_aot {
    include!(env!("XLSYNTH_PIR_AOT_EVENTS_RS"));
}

pub mod widget_frob_aot {
    include!(env!("XLSYNTH_PIR_AOT_WIDGET_FROB_RS"));
}

pub mod parametric_imports_aot {
    include!(env!("XLSYNTH_PIR_AOT_PARAMETRIC_IMPORTS_RS"));
}

pub mod shared_widget_echo_aot {
    include!(env!("XLSYNTH_PIR_AOT_SHARED_WIDGET_ECHO_RS"));
}

pub mod shared_widget_package_aot {
    include!(env!("XLSYNTH_PIR_AOT_SHARED_WIDGET_PACKAGE_RS"));
}
