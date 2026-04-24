// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use xlsynth_aot_test_crate::{
    add_inputs_aot, add_one_aot, compound_shapes_aot, empty_tuple_aot, large_array_tuple_aot,
    trace_assert_aot, wide_bits_tuple_aot, wide_sizes_aot,
};
use xlsynth_test_helpers::compare_golden_text;

fn generated_wrapper_golden_cases() -> [(&'static str, &'static str); 16] {
    [
        (
            env!("XLSYNTH_AOT_ADD_ONE_RS"),
            "tests/goldens/add_one_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_ADD_INPUTS_RS"),
            "tests/goldens/add_inputs_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_COMPOUND_SHAPES_RS"),
            "tests/goldens/compound_shapes_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_EMPTY_TUPLE_RS"),
            "tests/goldens/empty_tuple_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_WIDE_SIZES_RS"),
            "tests/goldens/wide_sizes_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_LARGE_ARRAY_TUPLE_RS"),
            "tests/goldens/large_array_tuple_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_WIDE_BITS_TUPLE_RS"),
            "tests/goldens/wide_bits_tuple_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_TRACE_ASSERT_RS"),
            "tests/goldens/trace_assert_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_AOT_WIDGET_FROB_RS"),
            "tests/goldens/widget_frob_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_SELF_ALIAS_WIDGET_RS"),
            "tests/goldens/self_alias_widget_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_PARAMETRIC_BOX_RS"),
            "tests/goldens/parametric_box_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_PARAMETRIC_SHAPES_RS"),
            "tests/goldens/parametric_shapes_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_PARAMETRIC_ARRAYS_RS"),
            "tests/goldens/parametric_arrays_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_PARAMETRIC_VALUES_RS"),
            "tests/goldens/parametric_values_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_PARAMETRIC_IMPORTS_RS"),
            "tests/goldens/parametric_imports_wrapper.rs",
        ),
        (
            env!("XLSYNTH_AOT_DUPLICATE_WIDGET_RS"),
            "tests/goldens/duplicate_widget_wrapper.rs",
        ),
    ]
}

// Verifies: generated AOT wrapper source matches checked-in goldens.
// Catches: unreviewed generated source drift in AOT wrapper emission.
#[test]
fn generated_wrappers_match_golden_references() {
    fs::create_dir_all(Path::new("tests/goldens")).expect("goldens directory should exist");

    for (generated_path, golden_path) in generated_wrapper_golden_cases() {
        let generated = fs::read_to_string(generated_path).expect("generated wrapper should exist");
        compare_golden_text(&generated, golden_path);
    }
}

// Verifies: a scalar IR-only generated runner links and executes.
// Catches: regressions in basic AOT runner construction or scalar ABI calls.
#[test]
fn add_one_generated_runner_executes() {
    let mut runner = add_one_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run(&41).expect("run should succeed");
    assert_eq!(output, 42);
}

// Verifies: generated entrypoints coexist in one test crate.
// Catches: symbol or artifact naming collisions between wrappers.
#[test]
fn multiple_generated_entrypoints_can_link_and_run() {
    let mut runner = add_inputs_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run(&10, &20).expect("run should succeed");
    assert_eq!(output, 30);
}

// Verifies: generated runners round-trip tuple and array aggregate shapes.
// Catches: ABI packing regressions for nested tuple and array fields.
#[test]
fn generated_runner_supports_compound_shapes() {
    let mut runner = compound_shapes_aot::new_runner().expect("runner creation should succeed");
    let lhs = compound_shapes_aot::Input0 {
        field0: 10,
        field1: [1000, 2000],
    };
    let rhs = compound_shapes_aot::Input1 {
        field0: 300,
        field1: [7, 9],
    };

    let output = runner.run(&lhs, &rhs).expect("run should succeed");
    assert_eq!(output.field0, [1007, 2300]);
    assert_eq!(output.field1.field0, 9);
    assert_eq!(output.field1.field1, 300);
    assert_eq!(output.field2, [17, 19]);
}

// Verifies: generated runners support zero-argument empty-tuple outputs.
// Catches: ABI edge cases for empty input and output layouts.
#[test]
fn generated_runner_supports_empty_tuple() {
    let mut runner = empty_tuple_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run().expect("run should succeed");
    assert_eq!(output, empty_tuple_aot::Output {});
}

// Verifies: generated runners handle scalar, wide, and array-backed bits.
// Catches: ABI packing regressions for non-byte-aligned wide values.
#[test]
fn generated_runner_supports_varied_bit_widths_including_wide_values() {
    let mut runner = wide_sizes_aot::new_runner().expect("runner creation should succeed");
    let input = wide_sizes_aot::Input0 {
        field0: true,
        field1: 0x55,
        field2: 0xAB,
        field3: 0x1234,
        field4: 0x89AB_CDEF,
        field5: 0x0123_4567_89AB_CDEF,
        field6: [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x01],
        field7: [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x11, 0x01],
        field8: [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
            0xEE, 0x7F,
        ],
        field9: [
            [0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x01],
            [0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x90, 0x00],
        ],
    };

    let output = runner.run(&input).expect("run should succeed");
    assert!(output.field0);
    assert_eq!(output.field1, 0x55);
    assert_eq!(output.field2, 0xAB);
    assert_eq!(output.field3, 0x1234);
    assert_eq!(output.field4, 0x89AB_CDEF);
    assert_eq!(output.field5, 0x0123_4567_89AB_CDEF);
    assert_eq!(
        output.field6,
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x01]
    );
    assert_eq!(
        output.field7,
        [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x11, 0x01]
    );
    assert_eq!(
        output.field8,
        [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
            0xEE, 0x7F,
        ]
    );
    assert_eq!(
        output.field9,
        [
            [0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x01],
            [0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x90, 0x00]
        ]
    );
}

// Verifies: generated runners preserve large fixed array fields.
// Catches: ABI traversal regressions that truncate or mis-index arrays.
#[test]
fn generated_runner_supports_large_array_tuple_structs() {
    let mut runner = large_array_tuple_aot::new_runner().expect("runner creation should succeed");
    let mut field1 = [0u16; 128];
    field1[0] = 0x1234;
    field1[127] = 0xABCD;
    let input = large_array_tuple_aot::Input0 {
        field0: 0x5A,
        field1,
    };

    let output = runner.run(&input).expect("run should succeed");
    assert_eq!(output.field0, 0x5A);
    assert_eq!(output.field1[0], 0x1234);
    assert_eq!(output.field1[127], 0xABCD);
    assert!(output.field1[1..127].iter().all(|value| *value == 0));
}

// Verifies: generated runners support bit values wider than scalar integers.
// Catches: byte packing regressions for wide bits tuple fields.
#[test]
fn generated_runner_supports_wide_bits_tuple_structs() {
    let mut runner = wide_bits_tuple_aot::new_runner().expect("runner creation should succeed");
    let mut expected_wide_bits = [0u8; 33];
    expected_wide_bits[0] = 0x5A;
    expected_wide_bits[32] = 0x01;
    let input = wide_bits_tuple_aot::Input0 {
        field0: 0xC3,
        field1: expected_wide_bits,
    };

    let output = runner.run(&input).expect("run should succeed");
    assert_eq!(output.field0, 0xC3);
    assert_eq!(output.field1, expected_wide_bits);
    assert!(output.field1[1..32].iter().all(|value| *value == 0));
}

// Verifies: generated runners surface trace messages and assertion messages.
// Catches: event plumbing regressions between AOT execution and runner APIs.
#[test]
fn generated_runner_run_with_events_surfaces_trace_and_assert_messages() {
    let mut runner = trace_assert_aot::new_runner().expect("runner creation should succeed");
    let token = trace_assert_aot::Token {};

    let ok_input = trace_assert_aot::Input1 {
        field0: 2,
        field1: 3,
    };
    let ok_result = runner
        .run_with_events(&token, &ok_input)
        .expect("run with events should succeed");
    assert_eq!(ok_result.output.field0, 5);
    assert_eq!(ok_result.output.field1, 3);
    assert_eq!(ok_result.trace_messages.len(), 1);
    assert_eq!(ok_result.assert_messages.len(), 0);
    assert_eq!(ok_result.trace_messages[0].message, "sum: 5");

    let ok_output = runner.run(&token, &ok_input).expect("run should succeed");
    assert_eq!(ok_output.field0, 5);
    assert_eq!(ok_output.field1, 3);

    let bad_input = trace_assert_aot::Input1 {
        field0: 2,
        field1: 2,
    };
    let bad_result = runner
        .run_with_events(&token, &bad_input)
        .expect("run with events should succeed");
    assert_eq!(bad_result.output.field0, 4);
    assert_eq!(bad_result.output.field1, 2);
    assert_eq!(bad_result.trace_messages.len(), 1);
    assert_eq!(bad_result.assert_messages.len(), 1);
    assert_eq!(bad_result.trace_messages[0].message, "sum: 4");
    assert_eq!(bad_result.assert_messages[0], "sum must be >= 5");

    let run_err = runner
        .run(&token, &bad_input)
        .expect_err("run should fail on assert");
    assert!(run_err.to_string().contains("XLS assertion failed"));
    assert!(run_err.to_string().contains("sum must be >= 5"));
}
