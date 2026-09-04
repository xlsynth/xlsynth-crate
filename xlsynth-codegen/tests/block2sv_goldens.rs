// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
#[allow(dead_code)]
#[path = "support/block2sv_goldens.rs"]
mod fixtures;
use fixtures::*;
use std::fs;
use std::path::Path;
use xlsynth_pir::ir_parser::Parser;

/// Runs every independently authored combined block-IR/SystemVerilog fixture.
#[test]
fn block2sv_golden_fixtures() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let mut paths = Vec::new();
    collect_golden_fixtures(&directory, &mut paths)
        .unwrap_or_else(|error| panic!("failed to collect {}: {error}", directory.display()));
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no combined golden fixtures found in {}",
        directory.display()
    );

    let mut failures = Vec::new();
    for path in paths {
        if let Err(error) = run_golden_fixture(&path) {
            failures.push(format!("{}:\n{error}", path.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} golden fixtures failed:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// Ensures fixture source remains directly consumable as package-form block IR.
#[test]
fn combined_golden_fixtures_are_valid_ir() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let mut paths = Vec::new();
    collect_golden_fixtures(&directory, &mut paths).unwrap();
    paths.sort();
    assert!(!paths.is_empty());

    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        let fixture = parse_golden_fixture(&path, &text).unwrap();
        Parser::new(&fixture.source)
            .parse_package()
            .unwrap_or_else(|error| panic!("invalid block IR in {}: {error}", path.display()));
        Parser::new(&text).parse_package().unwrap_or_else(|error| {
            panic!("fixture comments break IR in {}: {error}", path.display())
        });
    }
}

/// Rejects ambiguous register-write graphs across layout and reset variants.
#[test]
fn block2sv_rejects_multiple_register_write_nodes() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv/errors");
    for (relative_path, block) in [
        (
            "register_multiple_writes.golden.ir",
            "multiple_register_writes",
        ),
        (
            "register_multiple_writes_asynchronous_reset.golden.ir",
            "multiple_asynchronous_writes",
        ),
        (
            "register_multiple_writes_pipeline_reset.golden.ir",
            "multiple_pipeline_writes",
        ),
    ] {
        let path = directory.join(relative_path);
        let text = fs::read_to_string(&path).expect("read multiple-write error fixture");
        let fixture = parse_golden_fixture(&path, &text).expect("parse multiple-write error");
        assert!(fixture.expects_error());
        let error = execute_golden_fixture(&fixture).expect("run multiple-write error fixture");
        assert_eq!(
            error,
            format!("register `state` in block `{block}` requires exactly one write, found 2\n")
        );
    }
}

/// Preserves native sharing, signedness, and pipeline boundaries.
#[test]
fn native_codegen_preserves_sharing_signedness_and_pipeline_boundaries() {
    let native = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");

    let signed_path = native.join("arithmetic/unsigned_comparison_of_signed_products.golden.ir");
    let signed_text = fs::read_to_string(&signed_path).expect("read signed-comparison fixture");
    let signed_fixture =
        parse_golden_fixture(&signed_path, &signed_text).expect("parse signed-comparison fixture");
    let signed = execute_golden_fixture(&signed_fixture).expect("emit signed-comparison fixture");
    let signed_wires = regex::Regex::new(r"(?m)^\s*logic \[7:0\] smul_\d+;$")
        .expect("signed-multiply declaration expression is valid");
    assert_eq!(signed_wires.find_iter(&signed).count(), 2);
    assert!(signed.contains("$unsigned("));

    let array_path = native.join("arrays/chained_unnamed_updates.golden.ir");
    let array_text = fs::read_to_string(&array_path).expect("read chained array-update fixture");
    let array_fixture =
        parse_golden_fixture(&array_path, &array_text).expect("parse chained array-update fixture");
    let arrays = execute_golden_fixture(&array_fixture).expect("emit chained array-update fixture");
    let array_wires = regex::Regex::new(r"(?m)^\s*logic \[3:0\]\[7:0\] array_update_\d+;$")
        .expect("array-update declaration expression is valid");
    assert_eq!(array_wires.find_iter(&arrays).count(), 3);
    assert_eq!(arrays.matches("for (genvar ").count(), 3);

    let nested_path = native.join("arrays/multidimensional_update.golden.ir");
    let nested_text = fs::read_to_string(&nested_path).expect("read nested array-update fixture");
    let nested_fixture = parse_golden_fixture(&nested_path, &nested_text)
        .expect("parse nested array-update fixture");
    let nested = execute_golden_fixture(&nested_fixture).expect("emit nested array-update fixture");
    assert_eq!(nested.matches("for (genvar ").count(), 2);

    let slice_path = native.join("arrays/wide_slice.golden.ir");
    let slice_text = fs::read_to_string(&slice_path).expect("read generated array-slice fixture");
    let slice_fixture = parse_golden_fixture(&slice_path, &slice_text)
        .expect("parse generated array-slice fixture");
    let slice = execute_golden_fixture(&slice_fixture).expect("emit generated array-slice fixture");
    assert_eq!(slice.matches("for (genvar ").count(), 1);

    let arithmetic_path = native.join("arithmetic/unnamed_arithmetic_intermediates.golden.ir");
    let arithmetic_text =
        fs::read_to_string(&arithmetic_path).expect("read arithmetic-intermediate fixture");
    let arithmetic_fixture = parse_golden_fixture(&arithmetic_path, &arithmetic_text)
        .expect("parse arithmetic-intermediate fixture");
    let arithmetic =
        execute_golden_fixture(&arithmetic_fixture).expect("emit arithmetic-intermediate fixture");
    for name in ["add_4", "sub_5", "umul_6", "udiv_7"] {
        assert!(
            arithmetic.contains(&format!("logic [7:0] {name};")),
            "arithmetic node must be materialized: {name}"
        );
    }

    let pipeline_path = native.join("registers/two_stage_pipeline.golden.ir");
    let pipeline_text = fs::read_to_string(&pipeline_path).expect("read native pipeline fixture");
    let pipeline_fixture = parse_golden_fixture(&pipeline_path, &pipeline_text)
        .expect("parse native pipeline fixture");
    let pipeline = execute_golden_fixture(&pipeline_fixture).expect("emit native pipeline fixture");
    for marker in [
        "// ===== Pipe stage 0:",
        "// Registers for pipe stage 0:",
        "// ===== Pipe stage 1:",
        "// Registers for pipe stage 1:",
    ] {
        assert!(pipeline.contains(marker));
    }
}

#[test]
fn every_checked_in_fixture_roundtrips_without_rewriting_source_or_options() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let mut paths = Vec::new();
    collect_golden_fixtures(&root, &mut paths).unwrap();
    for path in paths {
        let original = fs::read_to_string(&path).unwrap();
        let fixture = parse_golden_fixture(&path, &original).unwrap();
        assert_eq!(
            fixture
                .with_expected_output(fixture.expected_output())
                .unwrap(),
            original,
            "{}",
            path.display()
        );
    }
}

#[test]
fn update_mode_cannot_turn_unexpected_success_or_failure_into_a_golden() {
    let fixture = GoldenFixture::parse("package p\n// EXPECT-SV: placeholder\n").unwrap();
    assert!(execute_golden_fixture(&fixture).is_err());
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/goldens/block2sv/structure/identity.golden.ir"),
    )
    .unwrap();
    let fixture =
        GoldenFixture::parse(&source.replace("// EXPECT-SV:", "// EXPECT-ERROR:")).unwrap();
    assert!(execute_golden_fixture(&fixture).is_err());
}
