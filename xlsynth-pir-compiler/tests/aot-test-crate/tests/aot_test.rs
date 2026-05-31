// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path, process::Command};

use xlsynth_pir_compiler_aot_test_crate::{
    add_inputs_aot, add_one_aot, compound_shapes_aot, empty_tuple_aot, events_aot,
    shared_widget_echo_aot, shared_widget_package_aot, widget_frob_aot,
};

fn generated_wrapper_golden_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            env!("XLSYNTH_PIR_AOT_ADD_ONE_RS"),
            "tests/goldens/add_one_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_ADD_INPUTS_RS"),
            "tests/goldens/add_inputs_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_COMPOUND_SHAPES_RS"),
            "tests/goldens/compound_shapes_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_EMPTY_TUPLE_RS"),
            "tests/goldens/empty_tuple_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_EVENTS_RS"),
            "tests/goldens/events_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_WIDGET_FROB_RS"),
            "tests/goldens/widget_frob_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_PARAMETRIC_IMPORTS_RS"),
            "tests/goldens/parametric_imports_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_SHARED_WIDGET_ECHO_RS"),
            "tests/goldens/shared_widget_echo_wrapper.golden.txt",
        ),
        (
            env!("XLSYNTH_PIR_AOT_SHARED_WIDGET_PACKAGE_RS"),
            "tests/goldens/shared_widget_package.golden.txt",
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

// Verifies: generated native PIR compiler wrapper source matches checked-in
// goldens.
// Catches: unreviewed public API, layout, metadata, or runtime glue changes.
#[test]
fn generated_wrappers_match_golden_references() {
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
    let input = add_one_aot::Bits42::new(41)?;
    let output = runner.run(&input)?;
    assert_eq!(output.to_u64(), 42);

    let mut caller_owned_output = add_one_aot::Bits42::default();
    runner.run_into(&input, &mut caller_owned_output)?;
    assert_eq!(caller_owned_output.to_u64(), 42);
    Ok(())
}

// Verifies: separately generated object entrypoints can coexist in one
// downstream crate without symbol collisions.
#[test]
fn multiple_generated_entrypoints_link_and_run() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = add_inputs_aot::new_runner()?;
    let lhs = add_inputs_aot::Bits8::new(10)?;
    let rhs = add_inputs_aot::Bits8::new(20)?;
    assert_eq!(runner.run(&lhs, &rhs)?.to_u64(), 30);
    Ok(())
}

// Verifies: public aggregate types have C layout and allow caller-owned native
// arrays and wide-bit limbs to be passed without packing buffers.
#[test]
fn generated_runner_accepts_native_aggregates_and_wide_bits()
-> Result<(), Box<dyn std::error::Error>> {
    let input = compound_shapes_aot::Input0 {
        field0: compound_shapes_aot::Bits42::new(100)?,
        field1: [
            compound_shapes_aot::Bits65::from_limbs([0x0123_4567_89ab_cdef, 1])?,
            compound_shapes_aot::Bits65::from_limbs([0xfedc_ba98_7654_3210, 0])?,
        ],
    };
    let increment = compound_shapes_aot::Bits42::new(23)?;
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

// Verifies: an AOT runner collects cover/trace/assert records in the Rust
// runtime while direct output remains native storage.
#[test]
fn generated_runner_collects_events() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = events_aot::new_runner()?;
    let x = events_aot::Bits8::new(0xa5)?;
    let emit = events_aot::Bits1::new(1)?;
    let passed = events_aot::Bits1::new(1)?;
    let failed = events_aot::Bits1::new(0)?;

    let successful = runner.run_with_events(&x, &passed, &emit)?;
    assert_eq!(successful.output.to_u64(), 0xa5);
    assert!(successful.events.assertion_failures.is_empty());
    assert_eq!(successful.events.cover_counts[0].label, "covered");
    assert_eq!(successful.events.cover_counts[0].count, 1);
    assert_eq!(successful.events.trace_messages[0].message, "x=165");

    let with_failure = runner.run_with_events(&x, &failed, &emit)?;
    assert_eq!(
        with_failure.events.assertion_failures[0].message,
        "bad condition"
    );
    assert_eq!(with_failure.events.assertion_failures[0].label, "A");
    assert!(runner.run(&x, &failed, &emit).is_err());
    Ok(())
}

// Verifies: DSLX-facing wrappers preserve nominal structs, enums, signed
// carriers, aliases, and arrays while directly invoking native code.
// Catches: DSLX bridge layout drift and enum representation regressions.
#[test]
fn typed_dslx_widget_runner_executes_without_marshalling() -> Result<(), Box<dyn std::error::Error>>
{
    use widget_frob_aot::widget_types::{SignedWidgetMode, Widget, WidgetMode, WidgetTuning};

    let widget = Widget {
        widget_id: widget_frob_aot::NativeBits8::new(9)?,
        mode: WidgetMode::Frob,
        frobs: [
            widget_frob_aot::NativeBits8::new(1)?,
            widget_frob_aot::NativeBits8::new(2)?,
            widget_frob_aot::NativeBits8::new(3)?,
        ],
        tuning: WidgetTuning {
            frob_bias: widget_frob_aot::NativeBits8::new(4)?,
            wobble_trim: widget_frob_aot::NativeBits8::wrapping(15),
        },
        wobble: widget_frob_aot::NativeBits8::new(3)?,
        signed_mode: SignedWidgetMode::Negative,
    };
    let garnish = [
        widget_frob_aot::NativeBits8::new(5)?,
        widget_frob_aot::NativeBits8::new(6)?,
        widget_frob_aot::NativeBits8::new(7)?,
    ];

    let mut runner = widget_frob_aot::widget_types::new_runner()?;
    let output = runner.run(&widget, &garnish)?;
    assert_eq!(output.next_widget_id.to_u64(), 10);
    assert_eq!(output.selected_frob.to_u64(), 13);
    assert_eq!(output.mode, WidgetMode::Frob);
    assert_eq!(output.adjusted_wobble.to_u64(), 4);
    assert_eq!(output.signed_mode, SignedWidgetMode::Negative);
    Ok(())
}

// Verifies: an imported DSLX struct remains the public Rust boundary type for
// a native AOT wrapper.
// Catches: imported nominal paths being replaced by structural tuple aliases.
#[test]
fn typed_dslx_imported_nominal_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    let widget = shared_widget_echo_aot::shared_widget_types::Widget {
        widget_id: shared_widget_echo_aot::NativeBits8::new(23)?,
    };
    let mut runner = shared_widget_echo_aot::shared_widget_echo::new_runner()?;
    assert_eq!(runner.run(&widget)?.widget_id.to_u64(), 23);
    Ok(())
}

// Verifies: package wrappers emit imported nominal types once and use that
// exact Rust type in independently compiled entrypoint runners.
// Catches: package generation minting incompatible per-entrypoint struct types.
#[test]
fn typed_dslx_package_runners_share_nominal_types() -> Result<(), Box<dyn std::error::Error>> {
    let widget = shared_widget_package_aot::shared_widget_types::Widget {
        widget_id: shared_widget_package_aot::NativeBits8::new(41)?,
    };
    let mut echo =
        shared_widget_package_aot::shared_widget_echo::aot_shared_package_echo::new_runner()?;
    let mut bump =
        shared_widget_package_aot::shared_widget_bump::aot_shared_package_bump::new_runner()?;
    let echoed = echo.run(&widget)?;
    let bumped = bump.run(&echoed)?;
    assert_eq!(bumped.widget_id.to_u64(), 42);
    Ok(())
}

// Verifies: consuming new compiler AOT output needs only the dedicated
// runtime at execution/link time, not libxls.
#[test]
fn consumer_binary_has_no_runtime_libxls_dependency() {
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
