// SPDX-License-Identifier: Apache-2.0

pub mod add_inputs_aot {
    include!(env!("XLSYNTH_AOT_ADD_INPUTS_RS"));
}

pub mod add_one_aot {
    include!(env!("XLSYNTH_AOT_ADD_ONE_RS"));
}

pub mod compound_shapes_aot {
    include!(env!("XLSYNTH_AOT_COMPOUND_SHAPES_RS"));
}

pub mod empty_tuple_aot {
    include!(env!("XLSYNTH_AOT_EMPTY_TUPLE_RS"));
}

pub mod wide_sizes_aot {
    include!(env!("XLSYNTH_AOT_WIDE_SIZES_RS"));
}

pub mod trace_assert_aot {
    include!(env!("XLSYNTH_AOT_TRACE_ASSERT_RS"));
}
