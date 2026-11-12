// SPDX-License-Identifier: Apache-2.0

//! End-to-end soundness checks for bounded and inductive BX properties.

#![cfg(feature = "has-bitwuzla")]

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::Result;

const PARITY: &str = r#"#[invariant]
block parity_equivalent(clk, rst, toggle: bool) -> bool {
    reg lhs: u2 = u2:0;
    reg rhs: bool = false;
    #[lemma]
    let lhs_is_live = lhs < u2:2;
    if toggle {
        if lhs_is_live { lhs := lhs ^ u2:1; }
        rhs := !rhs;
    }
    (lhs == u2:1) == rhs
}
"#;

fn run(
    root: &Path,
    source: &str,
    command: &str,
    flags: &[&str],
    success: bool,
) -> Result<(String, String)> {
    fs::write(root.join("suite.bx"), source)?;
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(root)
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1")
        .args([command, "suite.bx"])
        .args(flags)
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;
    assert_eq!(output.status.success(), success, "{stdout}\n{stderr}");
    Ok((stdout, stderr))
}

#[test]
fn independently_proved_lemma_closes_equivalence() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (proved, _) = run(
        dir.path(),
        PARITY,
        "bx-prove-invariant",
        &["--engine=bitwuzla"],
        true,
    )?;
    assert_eq!(
        proved,
        "invariant parity_equivalent ... proved at k=1\n1 proved; 0 bounded; 0 disproved; 0 inconclusive; 0 errors\n"
    );
    // Demoting the annotation preserves the value and both machines' behavior.
    let bare = PARITY.replace("#[lemma]", "");
    let (open, diagnostic) = run(
        dir.path(),
        &bare,
        "bx-prove-invariant",
        &["--max-depth=4"],
        false,
    )?;
    assert_eq!(
        open,
        "invariant parity_equivalent ... INCONCLUSIVE\n0 proved; 0 bounded; 0 disproved; 1 inconclusive; 0 errors\n"
    );
    assert!(
        diagnostic.contains("property induction step remains open at k=4"),
        "{diagnostic}"
    );
    let (bounded, _) = run(
        dir.path(),
        &bare,
        "bx-prove-invariant",
        &["--mode=bmc", "--max-depth=4"],
        true,
    )?;
    assert_eq!(
        bounded,
        "invariant parity_equivalent ... passed 4 bounded cycles\n0 proved; 1 bounded; 0 disproved; 0 inconclusive; 0 errors\n"
    );
    run(
        dir.path(),
        PARITY,
        "bx-test",
        &["--invariant-cycles=4", "--quickcheck-trials=4"],
        true,
    )?;
    Ok(())
}

#[test]
fn reachable_false_helpers_cannot_strengthen_a_proof() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"#[invariant]
block bad_helper(clk, rst) -> bool {
    reg count: u2 = u2:0;
    count := count + u2:1;
    #[lemma]
    let stays_small = count < u2:2;
    true
}
"#;
    let (_, diagnostic) = run(
        dir.path(),
        source,
        "bx-prove-invariant",
        &["--max-depth=2"],
        false,
    )?;
    assert!(
        diagnostic.contains("lemma group induction step remains open at k=2"),
        "{diagnostic}"
    );
    for mode in ["--mode=induction", "--mode=bmc"] {
        let (status, trace) = run(
            dir.path(),
            source,
            "bx-prove-invariant",
            &[mode, "--max-depth=3"],
            false,
        )?;
        assert_eq!(
            status,
            "invariant bad_helper ... DISPROVED\n0 proved; 0 bounded; 1 disproved; 0 inconclusive; 0 errors\n"
        );
        assert!(trace.contains("stays_small"), "{trace}");
    }
    let vacuous = source
        .replace("count < u2:2", "false")
        .replace("    true", "    false");
    let (_, trace) = run(dir.path(), &vacuous, "bx-prove-invariant", &[], false)?;
    assert!(trace.contains("stays_small"), "{trace}");
    Ok(())
}

#[test]
fn arbitrary_power_up_and_repeated_resets_are_in_the_initialized_check() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"#[invariant]
block arbitrary_power_up(clk, rst, write: bool) -> bool {
    reg held: bool;
    if write { held := false; }
    !held
}
#[invariant]
block reset_can_recur(clk, rst) -> bool {
    reg sampled: bool = false;
    sampled := true;
    !(rst && sampled)
}
#[invariant]
block before_first_clock(clk) -> bool {
    reg sampled: bool;
    sampled := false;
    !sampled
}
"#;
    let (status, _) = run(
        dir.path(),
        source,
        "bx-prove-invariant",
        &["--max-depth=2", "--reset-active-low"],
        false,
    )?;
    assert_eq!(
        status,
        "invariant arbitrary_power_up ... DISPROVED\ninvariant reset_can_recur ... DISPROVED\ninvariant before_first_clock ... DISPROVED\n0 proved; 0 bounded; 3 disproved; 0 inconclusive; 0 errors\n"
    );
    Ok(())
}

#[test]
fn longer_induction_and_jointly_inductive_lemmas() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"#[invariant]
block shift(clk, rst) -> bool {
    reg a: bool = false;
    reg b: bool = false;
    a := b;
    b := false;
    !a
}
#[invariant]
block mutually_preserved(clk, rst) -> bool {
    reg a: bool = false;
    reg b: bool = false;
    a := b;
    b := a;
    #[lemma]
    let a_clear = !a;
    #[lemma]
    let b_clear = !b;
    !(a || b)
}
"#;
    let (status, _) = run(
        dir.path(),
        source,
        "bx-prove-invariant",
        &["--max-depth=2"],
        true,
    )?;
    assert_eq!(
        status,
        "invariant shift ... proved at k=2\ninvariant mutually_preserved ... proved at k=1\n2 proved; 0 bounded; 0 disproved; 0 inconclusive; 0 errors\n"
    );
    Ok(())
}

#[test]
fn compares_imported_dslx_steps_on_a_shared_typed_input() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("parity.x"),
        r#"pub fn f(i: bool, s: u2) -> (bool, u2) {
    let next = if s < u2:2 && i { s ^ u2:1 } else { s };
    (s == u2:1, next)
}
pub fn g(i: bool, s: bool) -> (bool, bool) {
    (s, s ^ i)
}
"#,
    )?;
    let source = r#"use parity;
enum Command : u2 { Hold = 0, Toggle = 3 }
struct Drive { cmd: Command }
#[invariant]
block parity_from_library(clk, rst, drive: Drive) -> bool {
    reg lhs: u2 = u2:0;
    reg rhs: bool = false;
    #[lemma]
    let valid_command = drive.cmd == Command::Hold || drive.cmd == Command::Toggle;
    #[lemma]
    let lhs_is_live = lhs < u2:2;
    let toggle = drive.cmd == Command::Toggle;
    let (a, next_a) = parity::f(toggle, lhs);
    let (b, next_b) = parity::g(toggle, rhs);
    lhs := next_a;
    rhs := next_b;
    a == b
}
"#;
    let (status, _) = run(
        dir.path(),
        source,
        "bx-prove-invariant",
        &["--max-depth=1"],
        true,
    )?;
    assert_eq!(
        status,
        "invariant parity_from_library ... proved at k=1\n1 proved; 0 bounded; 0 disproved; 0 inconclusive; 0 errors\n"
    );
    Ok(())
}

#[test]
fn proves_observational_equivalence_examples_and_distinguishes_failures() -> Result<()> {
    let examples =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/bx-observational-equivalence");
    let dir = tempfile::tempdir()?;
    for module in [
        "add_vs_widening.x",
        "binary_vs_one_hot_latched_detector.x",
        "unreachable_state_elimination.x",
    ] {
        fs::copy(examples.join(module), dir.path().join(module))?;
    }
    let harness = fs::read_to_string(examples.join("equivalence.bx"))?;
    let flags = ["--engine=bitwuzla", "--max-depth=3"];
    let (proved, _) = run(dir.path(), &harness, "bx-prove-invariant", &flags, true)?;
    assert_eq!(
        proved,
        r#"invariant widened_accumulator ... proved at k=1
invariant latched_detector ... proved at k=3
invariant parity ... proved at k=1
3 proved; 0 bounded; 0 disproved; 0 inconclusive; 0 errors
"#
    );

    // The adder disagrees at cycle 0. Without its lemma, parity can start an
    // induction step in a dead state.
    let adder = dir.path().join("add_vs_widening.x");
    let wrong_sum = fs::read_to_string(&adder)?.replace("i_wide + s", "i_wide + s + u16:1");
    fs::write(adder, wrong_sum)?;
    let no_parity_lemma = harness.replace("#[lemma]", "");
    let (status, diagnostic) = run(
        dir.path(),
        &no_parity_lemma,
        "bx-prove-invariant",
        &flags,
        false,
    )?;
    assert_eq!(
        status,
        r#"invariant widened_accumulator ... DISPROVED
invariant latched_detector ... proved at k=3
invariant parity ... INCONCLUSIVE
1 proved; 0 bounded; 1 disproved; 1 inconclusive; 0 errors
"#
    );
    assert!(
        diagnostic.contains("invariant 'widened_accumulator' at cycle 0:"),
        "{diagnostic}"
    );
    Ok(())
}
