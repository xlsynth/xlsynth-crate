// SPDX-License-Identifier: Apache-2.0

use xlsynth_aot_test_crate::{
    add_inputs_aot, add_one_aot, compound_shapes_aot, empty_tuple_aot, trace_assert_aot,
    wide_sizes_aot,
};

#[test]
fn add_one_generated_runner_executes() {
    let mut runner = add_one_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run(&41).expect("run should succeed");
    assert_eq!(output, 42);
}

#[test]
fn multiple_generated_entrypoints_can_link_and_run() {
    let mut runner = add_inputs_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run(&10, &20).expect("run should succeed");
    assert_eq!(output, 30);
}

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

#[test]
fn generated_runner_supports_empty_tuple() {
    let mut runner = empty_tuple_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run().expect("run should succeed");
    assert_eq!(output, empty_tuple_aot::Output {});
}

#[test]
fn generated_runner_supports_varied_bit_widths_including_wide_values() {
    let mut runner = wide_sizes_aot::new_runner().expect("runner creation should succeed");
    let input = wide_sizes_aot::Input0 {
        field0: 1,
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
    assert_eq!(output.field0, 1);
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
