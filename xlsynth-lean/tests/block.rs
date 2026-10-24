// SPDX-License-Identifier: Apache-2.0

//! Public block emission contracts, including port and sequential semantics.

use xlsynth_lean::block::{emit, parse};

const BLOCK: &str = include_str!("fixtures/block_step.ir");

// Verifies: Raw emission preserves old outputs, simultaneous writes, and
// assertions. Catches: Port adaptation that changes clock-step semantics or
// drops assertions.
#[test]
fn raw_step_preserves_old_outputs_simultaneous_writes_and_disconnected_assertions() {
    let source = emit(&parse(BLOCK, None).unwrap()).unwrap();
    // The golden reads output/a and next/b from old_a, next/a from old_b,
    // gives reset precedence over enable, and retains the unused assertion.
    // The checked-in fixture's license is metadata, not generated user text.
    let golden = include_str!("fixtures/block_step.lean")
        .strip_prefix("-- SPDX-License-Identifier: Apache-2.0\n\n")
        .unwrap();
    pretty_assertions::assert_eq!(source, golden);
}

// Verifies: Top and explicit selection work while invalid selections and IR
// fail. Catches: Wrong-block selection or bypassed package validation.
#[test]
fn parsing_selects_top_or_explicit_block_and_reports_ambiguity() {
    let another = BLOCK
        .split_once("top block")
        .unwrap()
        .1
        .replace("swap(", "other(")
        .replace("id=", "id=10");
    let multiple = format!("{BLOCK}\nblock{another}");
    assert!(
        emit(&parse(&multiple, None).unwrap())
            .unwrap()
            .contains("block swap")
    );
    assert!(
        emit(&parse(&multiple, Some("other")).unwrap())
            .unwrap()
            .contains("block other")
    );
    assert!(parse(&multiple.replace("top block", "block"), None).is_err());
    assert!(parse(BLOCK, Some("missing")).is_err());
    assert!(parse(&BLOCK.replace("load_enable=en", "load_enable=old_a"), None).is_err());
}

// Negative test: Unsupported resets and operations return contextual emission
// errors.
#[test]
fn emission_rejects_asynchronous_reset_and_unsupported_operations() {
    let asynchronous = parse(
        &BLOCK.replace("asynchronous=false", "asynchronous=true"),
        None,
    )
    .unwrap();
    let error = emit(&asynchronous).unwrap_err();
    assert!(format!("{error:#}").contains("asynchronous reset"));
    let unsupported = BLOCK.replace(
        "  out: () = output_port(old_a,",
        "  reversed: bits[2] = reverse(old_a, id=10)\n  out: () = output_port(reversed,",
    );
    let error = emit(&parse(&unsupported, None).unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported Lean operation reverse"));
}
