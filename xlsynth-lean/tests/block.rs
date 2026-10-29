// SPDX-License-Identifier: Apache-2.0

//! Public block emission contracts, including port and sequential semantics.

use xlsynth_lean::block::{emit, select_block};
use xlsynth_pir::{ir::Package, ir_parser::Parser};

const BLOCK: &str = include_str!("fixtures/block_step.ir");

fn package(text: &str) -> Package {
    Parser::new(text).parse_and_validate_package().unwrap()
}

// Verifies: Raw emission preserves old outputs, simultaneous writes, and
// assertions. Catches: Port adaptation that changes clock-step semantics or
// drops assertions.
#[test]
fn raw_step_preserves_old_outputs_simultaneous_writes_and_disconnected_assertions() {
    let package = package(BLOCK);
    let source = emit(&package, package.get_top_block().unwrap()).unwrap();
    // The golden reads output/a and next/b from old_a, next/a from old_b,
    // gives reset precedence over enable, and retains the unused assertion.
    // The checked-in fixture's license is metadata, not generated user text.
    let golden = include_str!("fixtures/block_step.lean")
        .strip_prefix("-- SPDX-License-Identifier: Apache-2.0\n\n")
        .unwrap();
    pretty_assertions::assert_eq!(source, golden);
}

// Verifies: Named, top, and sole-block selection, including invalid choices.
#[test]
fn selects_top_or_explicit_block_and_reports_ambiguity() {
    let another = BLOCK
        .split_once("top block")
        .unwrap()
        .1
        .replace("swap(", "other(")
        .replace("id=", "id=10");
    let multiple = format!("{BLOCK}\nblock{another}");
    let with_top = package(&multiple);
    assert_eq!(select_block(&with_top, None).unwrap().name, "swap");
    assert_eq!(
        select_block(&with_top, Some("other")).unwrap().name,
        "other"
    );
    assert!(select_block(&with_top, Some("missing")).is_err());
    let ambiguous = package(&multiple.replace("top block", "block"));
    assert!(select_block(&ambiguous, None).is_err());
    let sole = package(&BLOCK.replace("top block", "block"));
    assert_eq!(select_block(&sole, None).unwrap().name, "swap");
}

// A same-named block from a different package would use the wrong file table.
#[test]
fn emission_rejects_block_from_another_package() {
    let original = package(BLOCK);
    let copy = original.clone();
    let error = emit(&original, copy.get_top_block().unwrap()).unwrap_err();
    assert!(error.to_string().contains("does not belong to package"));
}

// Negative test: Unsupported resets and operations return contextual emission
// errors.
#[test]
fn emission_rejects_asynchronous_reset_and_unsupported_operations() {
    let asynchronous = package(&BLOCK.replace("asynchronous=false", "asynchronous=true"));
    let error = emit(&asynchronous, asynchronous.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("asynchronous reset"));
    let unsupported = BLOCK.replace("package swap", "package swap\n\nfile_number 0 \"counter.x\"").replace(
        "  out: () = output_port(old_a,",
        "  reversed: bits[2] = reverse(old_a, id=10, pos=[(0,1,2)])\n  out: () = output_port(reversed,",
    );
    let unsupported = package(&unsupported);
    let error = emit(&unsupported, unsupported.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported Lean operation reverse"));
    assert!(format!("{error:#}").contains("counter.x:2:3"));
}
