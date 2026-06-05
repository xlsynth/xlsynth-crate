// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path, process::Command};

use xlsynth_pir_compiler_aot_dslx_test_crate::{
    duplicate_widget_aot, gizmo_frob_aot, namespaced_doodle_package_aot,
    parametric_forms_aot, parametric_imports_aot,
};

fn generated_wrapper_golden_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            concat!(
                env!("OUT_DIR"),
                "/gizmo_frob_typed_dslx_pir_aot_wrapper.rs"
            ),
            "tests/goldens/gizmo_frob_wrapper.golden.txt",
        ),
        (
            concat!(
                env!("OUT_DIR"),
                "/parametric_forms_typed_dslx_pir_aot_wrapper.rs"
            ),
            "tests/goldens/parametric_forms_wrapper.golden.txt",
        ),
        (
            concat!(
                env!("OUT_DIR"),
                "/parametric_imports_typed_dslx_pir_aot_wrapper.rs"
            ),
            "tests/goldens/parametric_imports_wrapper.golden.txt",
        ),
        (
            concat!(
                env!("OUT_DIR"),
                "/duplicate_widget_typed_dslx_pir_aot_wrapper.rs"
            ),
            "tests/goldens/duplicate_widget_wrapper.golden.txt",
        ),
        (
            concat!(
                env!("OUT_DIR"),
                "/namespaced_doodle_package_typed_dslx_pir_aot_package.rs"
            ),
            "tests/goldens/namespaced_doodle_package.golden.txt",
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

// Verifies: generated DSLX-started native AOT wrapper source matches checked-in
// goldens.
// Catches: unreviewed public API, layout, metadata, or runtime glue changes.
#[test]
fn generated_dslx_wrappers_match_golden_references() {
    fs::create_dir_all(Path::new("tests/goldens")).expect("goldens directory should exist");
    for (generated_path, golden_path) in generated_wrapper_golden_cases() {
        let generated = fs::read_to_string(generated_path).expect("generated wrapper should exist");
        compare_golden_text(&generated, golden_path);
    }
}

// Verifies: DSLX-facing wrappers preserve nominal structs, enums, signed
// carriers, aliases, and arrays while directly invoking native code.
// Catches: DSLX bridge layout drift and enum representation regressions.
#[test]
fn typed_dslx_gizmo_runner_executes_without_marshalling() -> Result<(), Box<dyn std::error::Error>>
{
    use gizmo_frob_aot::gizmo_types::{Gizmo, GizmoMode, GizmoTuning, SignedGizmoMode};

    let gizmo = Gizmo {
        gizmo_id: gizmo_frob_aot::BitsInU8::new(9)?,
        mode: GizmoMode::Frob,
        frobs: [
            gizmo_frob_aot::BitsInU8::new(1)?,
            gizmo_frob_aot::BitsInU8::new(2)?,
            gizmo_frob_aot::BitsInU8::new(3)?,
        ],
        tuning: GizmoTuning {
            frob_bias: gizmo_frob_aot::BitsInU8::new(4)?,
            wobble_trim: gizmo_frob_aot::BitsInU8::wrapping(15),
        },
        wobble: gizmo_frob_aot::BitsInU8::new(3)?,
        signed_mode: SignedGizmoMode::Negative,
    };
    let garnish = [
        gizmo_frob_aot::BitsInU8::new(5)?,
        gizmo_frob_aot::BitsInU8::new(6)?,
        gizmo_frob_aot::BitsInU8::new(7)?,
    ];

    let mut runner = gizmo_frob_aot::gizmo_types::new_runner()?;
    let output = runner.run(&gizmo, &garnish)?;
    assert_eq!(output.next_gizmo_id.to_u64(), 10);
    assert_eq!(output.selected_frob.to_u64(), 13);
    assert_eq!(output.mode, GizmoMode::Frob);
    assert_eq!(output.adjusted_wobble.to_u64(), 4);
    assert_eq!(output.signed_mode, SignedGizmoMode::Negative);
    Ok(())
}

// Verifies: local DSLX parametric aliases preserve evaluated value bindings,
// arrays of structs, structs of arrays, matrix shapes, and wide bits through
// one generated native AOT wrapper.
#[test]
fn typed_dslx_parametric_forms_runner_executes() -> Result<(), Box<dyn std::error::Error>> {
    use parametric_forms_aot::parametric_forms::{
        ArrayBox4, Box__N_8, Box8, Box8Array4, Box16, ExprBox8, HugeTag, Matrix2x3, NegativeTag,
        WidePair,
    };

    let box8 = Box8 {
        value: parametric_forms_aot::BitsInU8::new(8)?,
    };
    let box16 = Box16 {
        value: parametric_forms_aot::BitsInU16::new(16)?,
    };
    let matrix = Matrix2x3 {
        rows: [
            [
                parametric_forms_aot::BitsInU8::new(1)?,
                parametric_forms_aot::BitsInU8::new(2)?,
                parametric_forms_aot::BitsInU8::new(3)?,
            ],
            [
                parametric_forms_aot::BitsInU8::new(4)?,
                parametric_forms_aot::BitsInU8::new(5)?,
                parametric_forms_aot::BitsInU8::new(6)?,
            ],
        ],
    };
    let bits8 = parametric_forms_aot::BitsInU8::<8>::new;
    let array_box = ArrayBox4 {
        items: [bits8(10)?, bits8(11)?, bits8(12)?, bits8(13)?],
    };
    let box_array: Box8Array4 = [
        Box__N_8 { value: bits8(20)? },
        Box__N_8 { value: bits8(21)? },
        Box__N_8 { value: bits8(22)? },
        Box__N_8 { value: bits8(23)? },
    ];
    let expr_box = ExprBox8 { value: bits8(77)? };
    let negative = NegativeTag {
        payload: bits8(88)?,
    };
    let huge = HugeTag {
        payload: bits8(99)?,
    };
    let wide_pair = WidePair {
        unsigned_value: parametric_forms_aot::WideBits::from_limbs([0x0123_4567_89ab_cdef, 1])?,
        signed_value: parametric_forms_aot::WideBits::from_limbs([u64::MAX, 1])?,
    };
    let mut runner = parametric_forms_aot::parametric_forms::new_runner()?;
    let result = runner.run(
        &box8, &box16, &matrix, &array_box, &box_array, &expr_box, &negative, &huge, &wide_pair,
    )?;
    assert_eq!(result.box8.value.to_u64(), 8);
    assert_eq!(result.box16.value.to_u64(), 16);
    assert_eq!(result.matrix.rows[0][2].to_u64(), 3);
    assert_eq!(result.matrix.rows[1][1].to_u64(), 5);
    assert_eq!(result.array_box.items[3].to_u64(), 13);
    assert_eq!(result.box_array[2].value.to_u64(), 22);
    assert_eq!(result.expr_box.value.to_u64(), 77);
    assert_eq!(result.negative.payload.to_u64(), 88);
    assert_eq!(result.huge.payload.to_u64(), 99);
    assert_eq!(result.wide_pair, wide_pair);
    Ok(())
}

// Verifies: imported plain and concrete parametric DSLX structs remain the
// public Rust boundary types for a generated native AOT wrapper.
#[test]
fn typed_dslx_imported_parametric_structs_execute_without_marshalling()
-> Result<(), Box<dyn std::error::Error>> {
    use parametric_imports_aot::{parametric_imports, parametric_lib};

    let bits8 = parametric_imports_aot::BitsInU8::<8>::new;
    let remote = parametric_lib::RemotePlain { id: bits8(40)? };
    let imported_direct = parametric_lib::RemoteBox__N_8 { value: bits8(60)? };
    let imported_pair = parametric_lib::RemotePair__A_8__B_23 {
        left: bits8(61)?,
        right: parametric_imports_aot::BitsInU32::<23>::new(62)?,
    };

    let mut runner = parametric_imports::new_runner()?;
    let result = runner.run(&remote, &imported_direct, &imported_pair)?;

    assert_eq!(result.remote.id.to_u64(), 40);
    assert_eq!(result.imported_direct.value.to_u64(), 60);
    assert_eq!(result.imported_pair.left.to_u64(), 61);
    assert_eq!(result.imported_pair.right.to_u64(), 62);
    Ok(())
}

// Verifies: same-named imported DSLX structs stay on canonical nested Rust
// paths when the generated AOT interface crosses module boundaries.
#[test]
fn typed_dslx_duplicate_imported_names_use_canonical_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let widget = duplicate_widget_aot::foo::widget::Widget {
        widget_id: duplicate_widget_aot::BitsInU8::new(41)?,
    };
    let mut runner = duplicate_widget_aot::frobber::new_runner()?;
    assert_eq!(runner.run(&widget)?.widget_id.to_u64(), 42);
    Ok(())
}

// Verifies: package wrappers share imported nominal types and preserve nested
// DSLX module paths across different entrypoint modules.
#[test]
fn typed_dslx_namespaced_package_runners_preserve_module_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let doodle = namespaced_doodle_package_aot::types::shared_types::Doodle {
        doodle_id: namespaced_doodle_package_aot::BitsInU8::new(51)?,
    };
    let mut echo =
        namespaced_doodle_package_aot::foo::my_file::aot_namespaced_package_echo::new_runner()?;
    let mut bump =
        namespaced_doodle_package_aot::bar::your_file::aot_namespaced_package_bump::new_runner()?;
    let echoed = echo.run(&doodle)?;
    let bumped = bump.run(&echoed)?;
    assert_eq!(bumped.doodle_id.to_u64(), 52);
    Ok(())
}

// Verifies: consuming DSLX-started compiler AOT output needs only the dedicated
// runtime at execution/link time, not libxls.
#[test]
fn dslx_consumer_binary_has_no_runtime_libxls_dependency() {
    assert_consumer_binary_has_no_runtime_libxls_dependency();
}
