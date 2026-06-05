// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path, process::Command};

use xlsynth_pir_compiler_aot_ir_test_crate::{
    add_inputs_aot, add_one_aot, assumed_in_bounds_aot, compound_shapes_aot, empty_tuple_aot,
    events_aot, wide_runtime_ops_aot,
};
use xlsynth_pir_compiler_runtime::{AssumptionFailureKind, ExecutionResult, WideBits};

fn generated_wrapper_golden_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            concat!(env!("OUT_DIR"), "/add_one_pir_aot_wrapper.rs"),
            "tests/goldens/add_one_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/add_inputs_pir_aot_wrapper.rs"),
            "tests/goldens/add_inputs_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/compound_shapes_pir_aot_wrapper.rs"),
            "tests/goldens/compound_shapes_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/empty_tuple_pir_aot_wrapper.rs"),
            "tests/goldens/empty_tuple_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/wide_runtime_ops_pir_aot_wrapper.rs"),
            "tests/goldens/wide_runtime_ops_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/events_pir_aot_wrapper.rs"),
            "tests/goldens/events_wrapper.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/assumed_in_bounds_pir_aot_wrapper.rs"),
            "tests/goldens/assumed_in_bounds_wrapper.golden.txt",
        ),
    ]
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

fn assert_wide_limbs<const BIT_COUNT: usize, const LIMB_COUNT: usize>(
    actual: &WideBits<BIT_COUNT, LIMB_COUNT>,
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

// Verifies: generated PIR native AOT wrapper source matches checked-in goldens.
// Catches: unreviewed public API, layout, metadata, or runtime glue changes.
#[test]
fn generated_pir_wrappers_match_golden_references() {
    fs::create_dir_all(Path::new("tests/goldens")).expect("goldens directory should exist");
    for (generated_path, golden_path) in generated_wrapper_golden_cases() {
        let generated = fs::read_to_string(generated_path).expect("generated wrapper should exist");
        compare_golden_text(&generated, golden_path);
    }
}

// Verifies: generated scalar wrappers use the native bits API and execute
// linked Cranelift AOT object code.
#[test]
fn scalar_generated_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = add_one_aot::new_runner()?;
    let input = add_one_aot::BitsInU64::<42>::new(41)?;
    let output = runner.run(&input)?;
    assert_eq!(output.to_u64(), 42);

    let mut caller_owned_output = add_one_aot::BitsInU64::<42>::default();
    runner.run_into(&input, &mut caller_owned_output)?;
    assert_eq!(caller_owned_output.to_u64(), 42);
    Ok(())
}

// Verifies: separately generated object entrypoints can coexist in one
// downstream crate without symbol collisions.
#[test]
fn multiple_generated_entrypoints_link_and_run() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = add_inputs_aot::new_runner()?;
    let lhs = add_inputs_aot::BitsInU8::<8>::new(10)?;
    let rhs = add_inputs_aot::BitsInU8::<8>::new(20)?;
    assert_eq!(runner.run(&lhs, &rhs)?.to_u64(), 30);
    Ok(())
}

// Verifies: public aggregate types have C layout and allow caller-owned native
// arrays and wide-bit limbs to be passed without packing buffers.
#[test]
fn generated_runner_accepts_native_aggregates_and_wide_bits()
-> Result<(), Box<dyn std::error::Error>> {
    let input = compound_shapes_aot::Input0 {
        field0: compound_shapes_aot::BitsInU64::<42>::new(100)?,
        field1: [
            compound_shapes_aot::WideBits::<65, 2>::from_limbs([0x0123_4567_89ab_cdef, 1])?,
            compound_shapes_aot::WideBits::<65, 2>::from_limbs([0xfedc_ba98_7654_3210, 0])?,
        ],
    };
    let increment = compound_shapes_aot::BitsInU64::<42>::new(23)?;
    let mut runner = compound_shapes_aot::new_runner()?;
    let output = runner.run(&input, &increment)?;

    assert_eq!(output.field0.to_u64(), 123);
    assert_eq!(output.field1, input.field1);
    assert_eq!(std::mem::offset_of!(compound_shapes_aot::Input0, field1), 8);
    assert_eq!(std::mem::size_of::<compound_shapes_aot::Input0>(), 40);
    Ok(())
}

// Verifies: a zero-sized PIR value has a usable native generated interface.
#[test]
fn generated_runner_supports_empty_tuple_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = empty_tuple_aot::new_runner()?;
    assert_eq!(runner.run()?, empty_tuple_aot::Output {});
    Ok(())
}

// Verifies: linked AOT object code executes runtime-backed arbitrary-width
// operations through the public native limb storage API.
#[test]
fn generated_runner_executes_runtime_backed_wide_operations()
-> Result<(), Box<dyn std::error::Error>> {
    let x = wide_runtime_ops_aot::WideBits::<129, 3>::from_limbs([3, 0, 0])?;
    let y = wide_runtime_ops_aot::WideBits::<129, 3>::from_limbs([5, 0, 0])?;
    let shift = wide_runtime_ops_aot::BitsInU8::<8>::new(1)?;
    let replacement = wide_runtime_ops_aot::WideBits::<73, 2>::from_limbs([0x55, 0])?;
    let mut runner = wide_runtime_ops_aot::new_runner()?;
    let output = runner.run(&x, &y, &shift, &replacement)?;

    assert_wide_limbs(&output.field0, [15, 0, 0]);
    assert_wide_limbs(&output.field1, [15, 0, 0]);
    assert_wide_limbs(&output.field2, [0, 0, 0]);
    assert_wide_limbs(&output.field3, [6, 0, 0]);
    assert_wide_limbs(&output.field4, [1, 0, 0]);
    assert_wide_limbs(&output.field5, [1, 0]);
    assert_eq!(output.field6.to_u64(), 1);
    assert_wide_limbs(&output.field7, [0xab, 0, 0]);
    assert_wide_limbs(&output.field8, [15, 0, 0]);
    assert_wide_limbs(&output.field9, [15, 0, 0]);
    assert_wide_limbs(&output.field10, [1, 0, 0]);
    assert_eq!(output.field11.to_u64(), 0);
    assert_wide_limbs(&output.field12, [2, 0, 0]);

    let negative_x =
        wide_runtime_ops_aot::WideBits::<129, 3>::from_limbs([u64::MAX - 7, u64::MAX, 1])?;
    let arithmetic = runner.run(
        &negative_x,
        &y,
        &wide_runtime_ops_aot::BitsInU8::<8>::new(2)?,
        &replacement,
    )?;
    assert_wide_limbs(&arithmetic.field1, [u64::MAX - 39, u64::MAX, 1]);
    assert_wide_limbs(&arithmetic.field4, [u64::MAX - 1, u64::MAX, 1]);
    Ok(())
}

// Verifies: an AOT runner collects enabled and disabled cover/trace/assert
// records in the Rust runtime while direct output remains native storage.
#[test]
fn generated_runner_collects_events() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = events_aot::new_runner()?;
    let x = events_aot::BitsInU8::<8>::new(0xa5)?;
    let y = events_aot::BitsInU8::<8>::new(0x3c)?;
    let emit = events_aot::BitsInU8::<1>::new(1)?;
    let suppress = events_aot::BitsInU8::<1>::new(0)?;
    let passed = events_aot::BitsInU8::<1>::new(1)?;
    let failed = events_aot::BitsInU8::<1>::new(0)?;

    let successful = runner.run_with_events(&x, &y, &passed, &emit)?;
    assert_eq!(successful.output.to_u64(), 0xe1);
    assert!(successful.events.assertion_failures.is_empty());
    assert_eq!(cover_count(&successful.events, "covered"), 1);
    assert_eq!(cover_count(&successful.events, "accepted"), 1);
    assert_eq!(
        successful
            .events
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["x=165 y=3c", "ok=1"]
    );

    let suppressed = runner.run_with_events(&x, &y, &passed, &suppress)?;
    assert_eq!(cover_count(&suppressed.events, "covered"), 0);
    assert_eq!(cover_count(&suppressed.events, "accepted"), 1);
    assert_eq!(
        suppressed
            .events
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["ok=1"]
    );

    let with_failure = runner.run_with_events(&x, &y, &failed, &emit)?;
    assert_eq!(
        with_failure.events.assertion_failures[0].message,
        "bad condition"
    );
    assert_eq!(with_failure.events.assertion_failures[0].label, "A");
    assert_eq!(cover_count(&with_failure.events, "covered"), 1);
    assert_eq!(cover_count(&with_failure.events, "accepted"), 0);
    assert_eq!(
        with_failure
            .events
            .trace_messages
            .iter()
            .map(|trace| trace.message.as_str())
            .collect::<Vec<_>>(),
        vec!["x=165 y=3c"]
    );
    assert!(
        runner
            .run(&x, &y, &failed, &emit)
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
        assumed_in_bounds_aot::BitsInU8::<8>::new(10)?,
        assumed_in_bounds_aot::BitsInU8::<8>::new(11)?,
    ];
    let value = assumed_in_bounds_aot::BitsInU8::<8>::new(99)?;
    let in_bounds = assumed_in_bounds_aot::BitsInU8::<2>::new(1)?;
    let out_of_bounds = assumed_in_bounds_aot::BitsInU8::<2>::new(3)?;
    let mut runner = assumed_in_bounds_aot::new_runner()?;

    let safe = runner.run_with_events(&values, &value, &in_bounds)?;
    assert_eq!(safe.output.to_u64(), 99);
    assert!(safe.events.assumption_failures.is_empty());

    let failed = runner.run_with_events(&values, &value, &out_of_bounds)?;
    assert_eq!(failed.output.to_u64(), 99);
    let mut failures = failed
        .events
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

    let mut caller_owned_output = assumed_in_bounds_aot::BitsInU8::<8>::default();
    let events =
        runner.run_into_with_events(&values, &value, &out_of_bounds, &mut caller_owned_output)?;
    assert_eq!(caller_owned_output.to_u64(), 99);
    assert_eq!(events.assumption_failures.len(), 2);
    assert!(
        runner
            .run(&values, &value, &out_of_bounds)
            .unwrap_err()
            .to_string()
            .contains(
                "compiled assumed-in-bounds condition failed at node 5: ArrayUpdateOutOfBounds"
            )
    );
    Ok(())
}

// Verifies: consuming PIR-started compiler AOT output needs only the dedicated
// runtime at execution/link time, not libxls.
#[test]
fn pir_consumer_binary_has_no_runtime_libxls_dependency() {
    assert_consumer_binary_has_no_runtime_libxls_dependency();
}
