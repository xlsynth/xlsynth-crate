// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path, process::Command};

use xlsynth_pir_compiler_aot_ir_test_crate::native_aot_tests_aot;
use xlsynth_pir_compiler_aot_ir_test_crate::native_aot_tests_aot::native_aot_tests;
use xlsynth_pir_compiler_runtime::{
    AssumptionFailureKind, ExecutionOptions, ExecutionResult, SignedWideBits, UnsignedWideBits,
};

fn generated_package_golden_cases() -> Vec<(&'static str, &'static str)> {
    vec![(
        concat!(
            env!("OUT_DIR"),
            "/native_aot_tests_typed_ir_pir_aot_package.rs"
        ),
        "tests/goldens/native_aot_tests_typed_ir_package.golden.txt",
    )]
}

fn compare_golden_text(generated: &str, golden_path: &str) {
    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() {
        fs::write(golden_path, generated).expect("golden should update");
        return;
    }
    let golden = fs::read_to_string(golden_path).expect("golden should exist");
    assert_eq!(
        generated, golden,
        "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
    );
}

fn assert_unsigned_wide_limbs<const BIT_COUNT: usize, const LIMB_COUNT: usize>(
    actual: &UnsignedWideBits<BIT_COUNT, LIMB_COUNT>,
    expected: [u64; LIMB_COUNT],
) {
    assert_eq!(actual.limbs(), &expected);
}

fn assert_signed_wide_limbs<const BIT_COUNT: usize, const LIMB_COUNT: usize>(
    actual: &SignedWideBits<BIT_COUNT, LIMB_COUNT>,
    expected: [u64; LIMB_COUNT],
) {
    assert_eq!(actual.limbs(), &expected);
}

fn cover_count(events: &ExecutionResult, label: &str) -> u64 {
    events
        .cover_counts
        .iter()
        .find(|cover| cover.label == label)
        .unwrap_or_else(|| panic!("missing cover count for label {label}"))
        .count
}

fn assert_consumer_binary_has_no_runtime_libxls_dependency() {
    let executable = std::env::current_exe().expect("current test executable should resolve");
    let output = if cfg!(target_os = "macos") {
        Command::new("otool")
            .arg("-L")
            .arg(executable)
            .output()
            .expect("otool should inspect the test executable")
    } else if cfg!(target_os = "linux") {
        Command::new("ldd")
            .arg(executable)
            .output()
            .expect("ldd should inspect the test executable")
    } else {
        return;
    };
    let dependencies = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "dependency inspection should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !dependencies.contains("libxls"),
        "native PIR compiler AOT consumer unexpectedly depends on libxls:\n{}",
        dependencies
    );
}

// Verifies: generated PIR native AOT package source matches checked-in goldens.
// Catches: unreviewed public API, layout, metadata, or runtime glue changes.
#[test]
fn generated_pir_package_matches_golden_references() {
    fs::create_dir_all(Path::new("tests/goldens")).expect("goldens directory should exist");
    for (generated_path, golden_path) in generated_package_golden_cases() {
        let generated = fs::read_to_string(generated_path).expect("generated package should exist");
        compare_golden_text(&generated, golden_path);
    }
}

// Verifies: generated scalar wrappers use the native bits API and execute
// linked Cranelift AOT object code.
#[test]
fn scalar_generated_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = native_aot_tests::aot_add_one::new_runner()?;
    let input = native_aot_tests_aot::U42::new(41);
    let mut output = native_aot_tests_aot::U42::all_zeros();
    runner.run(&input, &mut output)?;
    assert_eq!(output.to_u64(), 42);

    let mut caller_owned_output = native_aot_tests_aot::U42::all_zeros();
    runner.run(&input, &mut caller_owned_output)?;
    assert_eq!(caller_owned_output.to_u64(), 42);
    Ok(())
}

// Verifies: multiple generated package entrypoints can coexist in one
// downstream crate without symbol collisions.
#[test]
fn multiple_generated_entrypoints_link_and_run() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = native_aot_tests::aot_add_inputs::new_runner()?;
    let lhs = native_aot_tests_aot::U8::new(10);
    let rhs = native_aot_tests_aot::U8::new(20);
    let mut output = native_aot_tests_aot::U8::all_zeros();
    runner.run(&lhs, &rhs, &mut output)?;
    assert_eq!(output.to_u8(), 30);
    Ok(())
}

// Verifies: a generated AOT runner accepts zero-byte native input storage and
// executes an operation over the sole bits[0] value.
#[test]
fn generated_runner_accepts_zero_bit_input() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(std::mem::size_of::<native_aot_tests_aot::U0>(), 0);
    let input = native_aot_tests_aot::U0::try_from(0_u64)?;
    assert!(native_aot_tests_aot::U0::try_from(1_u64).is_err());

    let mut runner = native_aot_tests::aot_zero_bit_input::new_runner()?;
    let mut output = native_aot_tests_aot::U1::all_zeros();
    runner.run(&input, &mut output)?;
    assert!(output.to_bool());
    Ok(())
}

// Verifies: public aggregate types have C layout and allow caller-owned native
// arrays and wide-bit limbs to be passed without packing buffers.
#[test]
fn generated_runner_accepts_native_aggregates_and_wide_bits()
-> Result<(), Box<dyn std::error::Error>> {
    let input = native_aot_tests::CompoundInput {
        base: native_aot_tests_aot::U42::new(100),
        limbs: [
            native_aot_tests_aot::U65::from_limbs([0x0123_4567_89ab_cdef, 1])?,
            native_aot_tests_aot::U65::from_limbs([0xfedc_ba98_7654_3210, 0])?,
        ],
    };
    let increment = native_aot_tests_aot::U42::new(23);
    let mut runner = native_aot_tests::aot_compound_shapes::new_runner()?;
    let mut output = native_aot_tests::CompoundOutput::all_zeros();
    runner.run(&input, &increment, &mut output)?;

    assert_eq!(output.sum.to_u64(), 123);
    assert_eq!(output.limbs, input.limbs);
    assert_eq!(
        std::mem::offset_of!(native_aot_tests::CompoundInput, limbs),
        8
    );
    assert_eq!(std::mem::size_of::<native_aot_tests::CompoundInput>(), 40);
    Ok(())
}

// Verifies: a zero-sized PIR value has a usable native generated interface.
#[test]
fn generated_runner_supports_empty_tuple_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = native_aot_tests::aot_empty_tuple::new_runner()?;
    let mut output = native_aot_tests::EmptyOutput::all_zeros();
    runner.run(&mut output)?;
    assert_eq!(output, native_aot_tests::EmptyOutput {});
    Ok(())
}

// Verifies: linked AOT object code executes runtime-backed arbitrary-width
// operations through the public native limb storage API.
#[test]
fn generated_runner_executes_runtime_backed_wide_operations()
-> Result<(), Box<dyn std::error::Error>> {
    let x = native_aot_tests_aot::U129::from_limbs([3, 0, 0])?;
    let y = native_aot_tests_aot::U129::from_limbs([5, 0, 0])?;
    let shift = native_aot_tests_aot::U8::new(1);
    let replacement = native_aot_tests_aot::U73::from_limbs([0x55, 0])?;
    let mut runner = native_aot_tests::aot_wide_runtime_ops::new_runner()?;
    let mut output = native_aot_tests::WideRuntimeOutput::all_zeros();
    runner.run(&x, &y, &shift, &replacement, &mut output)?;

    assert_unsigned_wide_limbs(&output.product, [15, 0, 0]);
    assert_signed_wide_limbs(&output.signed_product, [15, 0, 0]);
    assert_unsigned_wide_limbs(&output.quotient, [0, 0, 0]);
    assert_unsigned_wide_limbs(&output.left, [6, 0, 0]);
    assert_unsigned_wide_limbs(&output.right, [1, 0, 0]);
    assert_unsigned_wide_limbs(&output.slice, [1, 0]);
    assert_eq!(output.low_slice.to_u32(), 1);
    assert_unsigned_wide_limbs(&output.updated, [0xab, 0, 0]);
    assert_unsigned_wide_limbs(&output.unsigned_sum, [15, 0, 0]);
    assert_signed_wide_limbs(&output.signed_sum, [15, 0, 0]);
    assert_unsigned_wide_limbs(&output.hot, [1, 0, 0]);
    assert_eq!(output.encoded.to_u8(), 0);
    assert_unsigned_wide_limbs(&output.decoded, [2, 0, 0]);

    let negative_x = native_aot_tests_aot::U129::from_limbs([u64::MAX - 7, u64::MAX, 1])?;
    let mut arithmetic = native_aot_tests::WideRuntimeOutput::all_zeros();
    runner.run(
        &negative_x,
        &y,
        &native_aot_tests_aot::U8::new(2),
        &replacement,
        &mut arithmetic,
    )?;
    assert_signed_wide_limbs(&arithmetic.signed_product, [u64::MAX - 39, u64::MAX, 1]);
    assert_unsigned_wide_limbs(&arithmetic.right, [u64::MAX - 1, u64::MAX, 1]);
    Ok(())
}

// Verifies: an AOT runner collects enabled and disabled cover/trace/assert
// records in the Rust runtime while direct output remains native storage.
#[test]
fn generated_runner_collects_events() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = native_aot_tests::aot_events::new_runner()?;
    let x = native_aot_tests_aot::U8::new(0xa5);
    let y = native_aot_tests_aot::U8::new(0x3c);
    let emit = native_aot_tests_aot::U1::new(true);
    let suppress = native_aot_tests_aot::U1::new(false);
    let passed = native_aot_tests_aot::U1::new(true);
    let failed = native_aot_tests_aot::U1::new(false);

    let mut successful_output = native_aot_tests_aot::U8::all_zeros();
    let successful = runner.run_with_events(
        &x,
        &y,
        &passed,
        &emit,
        &mut successful_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(successful_output.to_u8(), 0xe1);
    assert!(successful.assertion_failures.is_empty());
    assert_eq!(cover_count(&successful, "covered"), 1);
    assert_eq!(cover_count(&successful, "accepted"), 1);
    assert_eq!(
        successful
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["x=165 y=3c", "ok=1"]
    );

    let mut suppressed_output = native_aot_tests_aot::U8::all_zeros();
    let suppressed = runner.run_with_events(
        &x,
        &y,
        &passed,
        &suppress,
        &mut suppressed_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(cover_count(&suppressed, "covered"), 0);
    assert_eq!(cover_count(&suppressed, "accepted"), 1);
    assert_eq!(
        suppressed
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["ok=1"]
    );

    let mut failure_output = native_aot_tests_aot::U8::all_zeros();
    let with_failure = runner.run_with_events(
        &x,
        &y,
        &failed,
        &emit,
        &mut failure_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(with_failure.assertion_failures[0].message, "bad condition");
    assert_eq!(with_failure.assertion_failures[0].label, "A");
    assert_eq!(cover_count(&with_failure, "covered"), 1);
    assert_eq!(cover_count(&with_failure, "accepted"), 0);
    assert_eq!(
        with_failure
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["x=165 y=3c"]
    );
    let mut rejected_output = native_aot_tests_aot::U8::all_zeros();
    assert!(
        runner
            .run(&x, &y, &failed, &emit, &mut rejected_output)
            .unwrap_err()
            .to_string()
            .contains("compiled assertion failed at node 8: bad condition")
    );
    Ok(())
}

// Verifies: retained assumed-in-bounds array nodes record failures through
// generated AOT wrappers while preserving safe compiled execution.
#[test]
fn generated_runner_reports_assumed_in_bounds_failures() -> Result<(), Box<dyn std::error::Error>> {
    let values = [
        native_aot_tests_aot::U8::new(10),
        native_aot_tests_aot::U8::new(11),
    ];
    let value = native_aot_tests_aot::U8::new(99);
    let in_bounds = native_aot_tests_aot::U2::new(1);
    let out_of_bounds = native_aot_tests_aot::U2::new(3);
    let mut runner = native_aot_tests::aot_assumed_in_bounds::new_runner()?;

    let mut safe_output = native_aot_tests_aot::U8::all_zeros();
    let safe = runner.run_with_events(
        &values,
        &value,
        &in_bounds,
        &mut safe_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(safe_output.to_u8(), 99);
    assert!(safe.assumption_failures.is_empty());

    let mut failed_output = native_aot_tests_aot::U8::all_zeros();
    let failed = runner.run_with_events(
        &values,
        &value,
        &out_of_bounds,
        &mut failed_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(failed_output.to_u8(), 99);
    let mut failures = failed
        .assumption_failures
        .iter()
        .map(|failure| (failure.node_text_id, failure.kind))
        .collect::<Vec<_>>();
    failures.sort();
    assert_eq!(
        failures,
        vec![
            (4, AssumptionFailureKind::ArrayIndexOutOfBounds),
            (5, AssumptionFailureKind::ArrayUpdateOutOfBounds),
        ]
    );

    let mut caller_owned_output = native_aot_tests_aot::U8::all_zeros();
    let events = runner.run_with_events(
        &values,
        &value,
        &out_of_bounds,
        &mut caller_owned_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(caller_owned_output.to_u8(), 99);
    assert_eq!(events.assumption_failures.len(), 2);
    let mut rejected_output = native_aot_tests_aot::U8::all_zeros();
    assert!(
        runner
            .run(&values, &value, &out_of_bounds, &mut rejected_output)
            .unwrap_err()
            .to_string()
            .contains(
                "compiled assumed-in-bounds condition failed at node 5: ArrayUpdateOutOfBounds"
            )
    );
    Ok(())
}

// Verifies: AOT object emission links reachable package callees used by
// ordinary invokes and DSLX-for-loop `counted_for` bodies.
#[test]
fn generated_runner_executes_invokes_and_counted_for() -> Result<(), Box<dyn std::error::Error>> {
    let init = native_aot_tests_aot::U8::new(10);
    let increment = native_aot_tests_aot::U8::new(1);
    let mut runner = native_aot_tests::aot_invokes_and_counted_for::new_runner()?;
    let mut output = native_aot_tests_aot::U8::all_zeros();
    runner.run(&init, &increment, &mut output)?;
    assert_eq!(output.to_u8(), 20);
    Ok(())
}

// Verifies: consuming PIR-started compiler AOT output needs only the dedicated
// runtime at execution/link time, not libxls.
#[test]
fn pir_consumer_binary_has_no_runtime_libxls_dependency() {
    assert_consumer_binary_has_no_runtime_libxls_dependency();
}
