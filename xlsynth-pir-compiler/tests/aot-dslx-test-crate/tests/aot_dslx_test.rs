// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path, process::Command};

use xlsynth_pir_compiler_aot_dslx_test_crate::native_dslx_tests_aot as dslx_aot;
use xlsynth_pir_compiler_runtime::{ExecutionOptions, ExecutionResult};

fn generated_package_golden_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            concat!(
                env!("OUT_DIR"),
                "/native_dslx_tests_typed_dslx_pir_aot_package.rs"
            ),
            "tests/goldens/native_dslx_tests_package.golden.txt",
        ),
        (
            concat!(env!("OUT_DIR"), "/native_dslx_tests_aot_metadata.json"),
            "tests/goldens/native_dslx_tests_aot_metadata.golden.json",
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

fn cover_count(events: &ExecutionResult, label: &str) -> u64 {
    events
        .cover_counts
        .iter()
        .find(|cover| cover.label == label)
        .unwrap_or_else(|| {
            panic!(
                "missing cover count for label {label}; observed {:?}",
                events.cover_counts
            )
        })
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

// Verifies: generated DSLX-started native AOT package source matches checked-in
// goldens.
// Catches: unreviewed public API, layout, metadata, or runtime glue changes.
#[test]
fn generated_dslx_package_matches_golden_references() {
    fs::create_dir_all(Path::new("tests/goldens")).expect("goldens directory should exist");
    for (generated_path, golden_path) in generated_package_golden_cases() {
        let generated = fs::read_to_string(generated_path).expect("generated package should exist");
        compare_golden_text(&generated, golden_path);
    }
}

// Verifies: DSLX-facing wrappers preserve nominal structs, enums, signed
// carriers, aliases, and arrays while directly invoking native code.
// Catches: DSLX bridge layout drift and enum representation regressions.
#[test]
fn typed_dslx_gizmo_runner_executes_without_marshalling() -> Result<(), Box<dyn std::error::Error>>
{
    use dslx_aot::gizmo_types::{Gizmo, GizmoMode, GizmoTuning, SignedGizmoMode};

    let gizmo = Gizmo {
        gizmo_id: dslx_aot::U8::new(9),
        mode: GizmoMode::Frob,
        frobs: [
            dslx_aot::U4::new(1),
            dslx_aot::U4::new(2),
            dslx_aot::U4::new(3),
        ],
        tuning: GizmoTuning {
            frob_bias: dslx_aot::U4::new(4),
            wobble_trim: dslx_aot::S4::new(-1),
        },
        wobble: dslx_aot::S4::new(3),
        signed_mode: SignedGizmoMode::Negative,
    };
    let garnish = [
        dslx_aot::U4::new(5),
        dslx_aot::U4::new(6),
        dslx_aot::U4::new(7),
    ];

    let mut runner = dslx_aot::gizmo_types::aot_gizmo_frob::new_runner()?;
    let mut output = dslx_aot::gizmo_types::GizmoOutcome::all_zeros();
    runner.run(&gizmo, &garnish, &mut output)?;
    assert_eq!(output.next_gizmo_id.to_u8(), 10);
    assert_eq!(output.selected_frob.to_u8(), 13);
    assert_eq!(output.mode, GizmoMode::Frob);
    assert_eq!(output.adjusted_wobble.to_i8(), 4);
    assert_eq!(output.signed_mode, SignedGizmoMode::Negative);
    Ok(())
}

// Verifies: local DSLX parametric aliases preserve evaluated value bindings,
// arrays of structs, structs of arrays, matrix shapes, and wide bits through
// one generated native AOT wrapper.
#[test]
fn typed_dslx_parametric_forms_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    use dslx_aot::parametric_forms::{
        ArrayBox4, Box__N_8, Box8, Box8Array4, Box16, ExprBox8, HugeTag, Matrix2x3, NegativeTag,
        WidePair,
    };

    let box8 = Box8 {
        value: dslx_aot::U8::new(8),
    };
    let box16 = Box16 {
        value: dslx_aot::U16::new(16),
    };
    let matrix = Matrix2x3 {
        rows: [
            [
                dslx_aot::U8::new(1),
                dslx_aot::U8::new(2),
                dslx_aot::U8::new(3),
            ],
            [
                dslx_aot::U8::new(4),
                dslx_aot::U8::new(5),
                dslx_aot::U8::new(6),
            ],
        ],
    };
    let bits8 = dslx_aot::U8::new;
    let array_box = ArrayBox4 {
        items: [bits8(10), bits8(11), bits8(12), bits8(13)],
    };
    let box_array: Box8Array4 = [
        Box__N_8 { value: bits8(20) },
        Box__N_8 { value: bits8(21) },
        Box__N_8 { value: bits8(22) },
        Box__N_8 { value: bits8(23) },
    ];
    let expr_box = ExprBox8 { value: bits8(77) };
    let negative = NegativeTag { payload: bits8(88) };
    let huge = HugeTag { payload: bits8(99) };
    let wide_pair = WidePair {
        unsigned_value: dslx_aot::U65::from_limbs([0x0123_4567_89ab_cdef, 1])?,
        signed_value: dslx_aot::S65::from_limbs([u64::MAX, 1])?,
    };
    let mut runner = dslx_aot::parametric_forms::aot_parametric_forms::new_runner()?;
    let mut result = dslx_aot::parametric_forms::ParametricFormsResult::all_zeros();
    runner.run(
        &box8,
        &box16,
        &matrix,
        &array_box,
        &box_array,
        &expr_box,
        &negative,
        &huge,
        &wide_pair,
        &mut result,
    )?;
    assert_eq!(result.box8.value.to_u8(), 8);
    assert_eq!(result.box16.value.to_u16(), 16);
    assert_eq!(result.matrix.rows[0][2].to_u8(), 3);
    assert_eq!(result.matrix.rows[1][1].to_u8(), 5);
    assert_eq!(result.array_box.items[3].to_u8(), 13);
    assert_eq!(result.box_array[2].value.to_u8(), 22);
    assert_eq!(result.expr_box.value.to_u8(), 77);
    assert_eq!(result.negative.payload.to_u8(), 88);
    assert_eq!(result.huge.payload.to_u8(), 99);
    assert_eq!(result.wide_pair, wide_pair);
    Ok(())
}

// Verifies: anonymous DSLX tuple params and returns lower to generated C-layout
// public Rust wrapper structs instead of Rust tuple ABI.
#[test]
fn typed_dslx_tuple_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    let pair = dslx_aot::XlsynthPirAotTuple0 {
        field0: dslx_aot::U8::new(10),
        field1: dslx_aot::U16::new(1000),
    };
    let increment = dslx_aot::U8::new(7);
    let mut runner = dslx_aot::tuple_shapes::aot_tuple_shapes::new_runner()?;
    let mut output = dslx_aot::XlsynthPirAotTuple0::all_zeros();
    runner.run(&pair, &increment, &mut output)?;

    assert_eq!(output.field0.to_u8(), 17);
    assert_eq!(output.field1.to_u16(), 1007);
    assert_eq!(
        std::mem::offset_of!(dslx_aot::XlsynthPirAotTuple0, field1),
        2
    );
    assert_eq!(std::mem::size_of::<dslx_aot::XlsynthPirAotTuple0>(), 4);
    Ok(())
}

// Verifies: DSLX-started AOT follows helper calls and `for` loop bodies through
// package-level invokes when emitting the native object.
#[test]
fn typed_dslx_invokes_and_for_loop_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    let init = dslx_aot::U8::new(10);
    let increment = dslx_aot::U8::new(1);
    let mut runner = dslx_aot::invokes_and_loop::aot_invokes_and_loop::new_runner()?;
    let mut output = dslx_aot::U8::all_zeros();
    runner.run(&init, &increment, &mut output)?;
    assert_eq!(output.to_u8(), 18);
    Ok(())
}

// Verifies: DSLX-started AOT preserves DSLX trace/assert/cover builtins through
// IR lowering and records them through the generated native runner.
#[test]
fn typed_dslx_event_runner_collects_trace_assert_and_cover()
-> Result<(), Box<dyn std::error::Error>> {
    let x = dslx_aot::U8::new(0xa5);
    let y = dslx_aot::U8::new(0x3c);
    let emit = dslx_aot::U1::new(true);
    let suppress = dslx_aot::U1::new(false);
    let passed = dslx_aot::U1::new(true);
    let failed = dslx_aot::U1::new(false);
    let mut runner = dslx_aot::events::aot_events::new_runner()?;

    let mut successful_output = dslx_aot::U8::all_zeros();
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
        vec!["accepted x=165", "x=165 y=3c"]
    );

    let mut suppressed_output = dslx_aot::U8::all_zeros();
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
        vec!["accepted x=165"]
    );

    let mut failure_output = dslx_aot::U8::all_zeros();
    let with_failure = runner.run_with_events(
        &x,
        &y,
        &failed,
        &emit,
        &mut failure_output,
        ExecutionOptions::collect_all(),
    )?;
    assert_eq!(failure_output.to_u8(), 0xe1);
    assert!(
        with_failure.assertion_failures[0]
            .message
            .contains("Assertion failure via assert!")
    );
    assert_eq!(with_failure.assertion_failures[0].label, "bad_condition");
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
    let mut rejected_output = dslx_aot::U8::all_zeros();
    assert!(
        runner
            .run(&x, &y, &failed, &emit, &mut rejected_output)
            .unwrap_err()
            .to_string()
            .contains("Assertion failure via assert!")
    );
    Ok(())
}

// Verifies: imported plain and concrete parametric DSLX structs remain the
// public Rust boundary types for a generated native AOT wrapper.
#[test]
fn typed_dslx_imported_parametric_structs_execute_without_marshalling()
-> Result<(), Box<dyn std::error::Error>> {
    use dslx_aot::{parametric_imports, parametric_lib};

    let bits8 = dslx_aot::U8::new;
    let remote = parametric_lib::RemotePlain { id: bits8(40) };
    let imported_direct = parametric_lib::RemoteBox__N_8 { value: bits8(60) };
    let imported_pair = parametric_lib::RemotePair__A_8__B_23 {
        left: bits8(61),
        right: dslx_aot::U23::new(62),
    };

    let mut runner = parametric_imports::aot_parametric_imports::new_runner()?;
    let mut result = parametric_imports::ParametricImportsResult::all_zeros();
    runner.run(&remote, &imported_direct, &imported_pair, &mut result)?;

    assert_eq!(result.remote.id.to_u8(), 40);
    assert_eq!(result.imported_direct.value.to_u8(), 60);
    assert_eq!(result.imported_pair.left.to_u8(), 61);
    assert_eq!(result.imported_pair.right.to_u32(), 62);
    Ok(())
}

// Verifies: same-named imported DSLX structs stay on canonical nested Rust
// paths when the generated AOT interface crosses module boundaries.
#[test]
fn typed_dslx_duplicate_imported_names_use_canonical_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let widget = dslx_aot::foo::widget::Widget {
        widget_id: dslx_aot::U8::new(41),
    };
    let mut runner = dslx_aot::frobber::aot_duplicate_widget::new_runner()?;
    let mut output = dslx_aot::bar::widget::Widget::all_zeros();
    runner.run(&widget, &mut output)?;
    assert_eq!(output.widget_id.to_u8(), 42);
    Ok(())
}

// Verifies: package wrappers share imported nominal types and preserve nested
// DSLX module paths across different entrypoint modules.
#[test]
fn typed_dslx_namespaced_package_runners_preserve_module_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let doodle = dslx_aot::types::shared_types::Doodle {
        doodle_id: dslx_aot::U8::new(51),
    };
    let mut echo = dslx_aot::foo::my_file::aot_namespaced_package_echo::new_runner()?;
    let mut bump = dslx_aot::bar::your_file::aot_namespaced_package_bump::new_runner()?;
    let mut echoed = dslx_aot::types::shared_types::Doodle::all_zeros();
    echo.run(&doodle, &mut echoed)?;
    let mut bumped = dslx_aot::types::shared_types::Doodle::all_zeros();
    bump.run(&echoed, &mut bumped)?;
    assert_eq!(bumped.doodle_id.to_u8(), 52);
    Ok(())
}

// Verifies: consuming DSLX-started compiler AOT output needs only the dedicated
// runtime at execution/link time, not libxls.
#[test]
fn dslx_consumer_binary_has_no_runtime_libxls_dependency() {
    assert_consumer_binary_has_no_runtime_libxls_dependency();
}
