// SPDX-License-Identifier: Apache-2.0

//! Exercise bounded BX proofs, including concrete replay of solver witnesses.

#![cfg(feature = "has-bitwuzla")]

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::Result;

/// Runs the command on a BX module and returns its report and diagnostics.
fn prove(root: &Path, source: &str, args: &[&str], success: bool) -> Result<(String, String)> {
    fs::write(root.join("suite.bx"), source)?;
    let mut arguments = vec!["bx-prove-quickcheck", "suite.bx", "--solver=bitwuzla"];
    arguments.extend_from_slice(args);
    run(root, &arguments, success)
}

fn run(root: &Path, args: &[&str], success: bool) -> Result<(String, String)> {
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(root)
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1")
        .args(args)
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;
    assert_eq!(output.status.success(), success, "{stdout}\n{stderr}");
    Ok((stdout, stderr))
}

const COUNTER: &str = r#"struct Drive { increment: u8, enable: bool }
block counter(clk, rst, drive: Drive) -> u8 {
    reg total: u8 = u8:0;
    if drive.enable { total := total + drive.increment; }
    total
}
#[quickcheck]
block counter_matches(trace: Drive[4]) -> bool {
    reg drive: Drive;
    reg expected: u8;
    let total = counter(drive);
    tick!(rst: true);
    for row in trace {
        drive := row;
        if row.enable { expected := expected + row.increment; }
        tick!();
        assert_eq(total, expected);
    }
    total == expected
}
"#;

#[test]
fn proves_counter_and_replays_broken_enable_and_reset() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (report, _) = prove(dir.path(), COUNTER, &[], true)?;
    assert_eq!(
        report,
        "quickcheck counter_matches ... proved\n1 proved; 0 disproved; 0 inconclusive; 0 errors\n"
    );

    for (old, new, expected_replay) in [
        (
            "if drive.enable { total := total + drive.increment; }",
            "total := total + drive.increment;",
            "bits[1]:0), tick!()]", // First mismatch must be a disabled increment.
        ),
        (
            "reg total: u8 = u8:0;",
            "reg total: u8;",
            "at tick 2: assert_eq failed", // A nonzero power-up value survives reset.
        ),
    ] {
        let (report, witness) = prove(dir.path(), &COUNTER.replace(old, new), &[], false)?;
        assert_eq!(
            report,
            "quickcheck counter_matches ... DISPROVED\n0 proved; 1 disproved; 0 inconclusive; 0 errors\n"
        );
        for text in [
            "suite.bx:17:",
            "at tick ",
            "inputs: [trace=",
            "concrete initial registers:",
            "stimulus trace:",
            "tick!(rst: true)",
            "drive :=",
            expected_replay,
        ] {
            assert!(witness.contains(text), "missing {text:?} in {witness}");
        }
    }
    Ok(())
}

#[test]
fn quantifies_each_initial_register_and_checks_samples_and_the_final_value() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"block latch(clk, rst, en: bool, d: u8) -> u8 {
    reg q: u8 = u8:0;
    if en { q := d; }
    q
}
#[quickcheck]
block sampled(d: u8, reset: bool) -> bool {
    reg en: bool;
    reg data: u8;
    let q = latch(en, data);
    en := false;
    let before = q;
    tick!();
    assert_eq(q, before);
    en := true;
    data := d;
    tick!(rst: reset);
    let sample = q;
    data := !d;
    tick!();
    assert_eq(q, !d);
    if reset { assert_eq(sample, u8:0); } else { assert_eq(sample, d); }
    en := reset;
    if en { en := false; } else { assert!(!reset); }
    assert!(!en);
    for i in u8:2..u8:5 {
        en := true;
        data := i;
        tick!();
        assert_eq(q, i);
    }
    true
}
#[quickcheck]
block independent_power_up() -> bool {
    let a = latch(false, u8:0);
    let b = latch(false, u8:0);
    tick!();
    assert_eq(a, b);
    true
}
#[quickcheck]
block never_three(n: u8) -> bool {
    let q = latch(true, n);
    tick!();
    q != u8:3
}
"#;
    let (report, witness) = prove(dir.path(), source, &["--reset-active-low"], false)?;
    assert_eq!(
        report,
        "quickcheck sampled ... proved\nquickcheck independent_power_up ... DISPROVED\nquickcheck never_three ... DISPROVED\n1 proved; 2 disproved; 0 inconclusive; 0 errors\n"
    );
    assert!(
        witness.contains("suite.bx:39:5: test 'independent_power_up' at tick 1: assert_eq failed")
            && witness.contains("test 'never_three' at tick 1")
            && witness.contains("inputs: [n=bits[8]:3]"),
        "{witness}"
    );
    Ok(())
}

#[test]
fn captures_loop_array_at_entry_and_restricts_nested_enums_to_declared_values() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"enum Op : u3 { Hold = 1, Load = 5 }
struct Row { op: Op, data: u8 }
#[quickcheck]
block rows_match(rows: Row[3]) -> bool {
    reg pending: Row[3] = rows;
    reg i: u32;
    for row in pending {
        assert!(row.op == Op::Hold || row.op == Op::Load);
        assert_eq(row, rows[i]);
        pending := Row[3]:[Row { op: Op::Hold, data: u8:0 }, ...];
        i := i + u32:1;
    }
    true
}
"#;
    let (report, _) = prove(dir.path(), source, &[], true)?;
    assert_eq!(
        report,
        "quickcheck rows_match ... proved\n1 proved; 0 disproved; 0 inconclusive; 0 errors\n"
    );
    Ok(())
}

#[test]
fn rejects_a_tick_nested_under_a_runtime_condition() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"#[quickcheck]
block conditional(run: bool, rows: u8[2]) -> bool {
    if run {
        for row in rows {
            tick!();
        }
    }
    true
}
#[quickcheck] block fixed() -> bool { tick!(); true }
"#;
    let (report, diagnostics) = prove(dir.path(), source, &[], false)?;
    assert_eq!(
        report,
        "quickcheck conditional ... INCONCLUSIVE\nquickcheck fixed ... proved\n1 proved; 0 disproved; 1 inconclusive; 0 errors\n"
    );
    assert!(
        diagnostics.contains("suite.bx:5:13: conditional tick!"),
        "{diagnostics}"
    );
    run(
        dir.path(),
        &["bx-test", "suite.bx", "--quickcheck-trials=4"],
        true,
    )?;
    Ok(())
}

#[test]
fn bounds_expansion_of_both_the_test_schedule_and_reachable_hardware_loops() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = r#"#[quickcheck]
block too_many_edges() -> bool {
    for i in u32:0..u32:2048 {
        for j in u32:0..u32:2048 {
            tick!();
        }
    }
    true
}
"#;
    let (_, diagnostics) = prove(dir.path(), source, &[], false)?;
    assert!(
        diagnostics.contains("suite.bx:4:9:") && diagnostics.contains("symbolic unrolling limit"),
        "{diagnostics}"
    );
    fs::write(
        dir.path().join("long.ir"),
        r#"package long
fn step(i: bits[32] id=1, acc: bits[8] id=2) -> bits[8] {
  ret inverted: bits[8] = not(acc, id=3)
}
fn many(x: bits[8] id=4) -> bits[8] {
  ret result: bits[8] = counted_for(x, trip_count=100000000, stride=1, body=step, id=5)
}
"#,
    )?;
    let source = r#"extern fn many(x: u8) -> u8;
#[quickcheck]
block expensive(x: u8) -> bool {
    many(x) == x
}
"#;
    let (_, diagnostics) = prove(dir.path(), source, &["--link-ir=long.ir"], false)?;
    assert!(
        diagnostics.contains("suite.bx:4:") && diagnostics.contains("symbolic unrolling limit"),
        "{diagnostics}"
    );
    Ok(())
}

#[test]
fn bounds_block_instance_expansion_before_flattening_only_the_selected_hierarchy() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut source = String::from(
        r#"block level0(clk, rst, x: bool) -> bool {
    reg q: bool;
    q := x;
    q
}
"#,
    );
    for level in 1..=20 {
        let child = level - 1;
        source.push_str(&format!(
            r#"block level{level}(clk, rst, x: bool) -> bool {{
    let a = level{child}(x);
    let b = level{child}(!x);
    a ^ b
}}
"#,
        ));
    }
    source.push_str(
        r#"#[quickcheck] block large(x: bool) -> bool {
    let out = level20(x);
    tick!();
    out == x
}
#[quickcheck] block small() -> bool { true }
"#,
    );
    let (report, diagnostics) = prove(dir.path(), &source, &[], false)?;
    assert_eq!(
        report,
        "quickcheck large ... INCONCLUSIVE\nquickcheck small ... proved\n1 proved; 0 disproved; 1 inconclusive; 0 errors\n"
    );
    assert!(diagnostics.contains("block hierarchy exceeds the symbolic unrolling limit of 1048576 work units before flattening"), "{diagnostics}");
    Ok(())
}

#[test]
fn rejects_reachable_linked_contracts_including_counted_loop_bodies() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("linked.ir"),
        r#"package linked
fn lookup(a: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
  ret selected: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=3)
}
fn step(i: bits[2] id=4, acc: bits[8] id=5) -> bits[8] {
  t: token = after_all(id=6)
  bad: bits[1] = literal(value=0, id=7)
  checked: token = assert(t, bad, message="bad step", label="step_valid", id=8)
  ret result: bits[8] = identity(acc, id=9)
}
block through_loop(clk: clock, x: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=10)
  result: bits[8] = counted_for(x, trip_count=2, stride=1, body=step, id=11)
  out: () = output_port(result, name=out, id=12)
}
"#,
    )?;
    let source = r#"extern fn lookup(a: u8[2], i: u8) -> u8;
#[quickcheck]
block lookup_contract(i: u8) -> bool {
    let a = [u8:1, u8:2];
    lookup(a, i) == lookup(a, i)
}
"#;
    let (_, diagnostics) = prove(dir.path(), source, &["--link-ir=linked.ir"], false)?;
    assert!(
        diagnostics.contains("lookup_contract") && diagnostics.contains("assumed_in_bounds=true"),
        "{diagnostics}"
    );

    let loop_property = r#"extern block through_loop(clk, x: u8) -> u8;
#[quickcheck]
block loop_contract(x: u8) -> bool {
    let result = through_loop(x);
    tick!();
    result == x
}
"#;
    let (_, diagnostics) = prove(dir.path(), loop_property, &["--link-ir=linked.ir"], false)?;
    assert!(
        diagnostics.contains("loop_contract")
            && diagnostics.contains("hardware assertion 'step_valid'"),
        "{diagnostics}"
    );
    prove(
        dir.path(),
        "#[quickcheck] block unrelated() -> bool { true }",
        &["--link-ir=linked.ir"],
        true,
    )?;
    Ok(())
}
