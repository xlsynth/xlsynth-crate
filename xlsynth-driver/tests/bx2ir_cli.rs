// SPDX-License-Identifier: Apache-2.0

//! Exercise BX source, linked IR, RTL, and mapped netlists at the driver
//! boundary.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use anyhow::{Context, Result};
use prost::Message;
use tempfile::TempDir;
use xlsynth_g8r::netlist::proto::pb;
use xlsynth_pir::ir::MemberType;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_test_helpers::compare_golden_text;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

fn driver(directory: &Path, arguments: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1")
        .args(arguments)
        .output()
}

/// Returns successful stdout and includes the command's diagnostics on failure.
fn stdout(directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = driver(directory, arguments)?;
    anyhow::ensure!(
        output.status.success(),
        "{arguments:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?)
}

#[test]
fn bx_test_runs_root_tests_with_imports_and_linked_functions() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let library = root.join("lib");
    fs::create_dir(&library)?;
    fs::write(
        library.join("math.bx"),
        r#"pub fn twice(x: u8) -> u8 { x + x }
#[test]
fn library_only() { assert!(false); }
"#,
    )?;
    fs::write(
        root.join("externs.ir"),
        r#"package arithmetic
fn triple(x: bits[8]) -> bits[8] {
    doubled: bits[8] = add(x, x)
    ret result: bits[8] = add(doubled, x)
}"#,
    )?;
    fs::write(
        root.join("suite.bx"),
        r#"use math;
extern fn triple(x: u8) -> u8;

#[test]
fn arithmetic() {
    assert_eq(math::twice(u8:129), u8:2);
    assert_eq(triple(u8:7), u8:21);
}

struct Drive { increment: u8, enable: bool }
struct Expect { total: u8 }
const CASES: (Drive, Expect)[3] = [
    (Drive { increment: u8:2, enable: true }, Expect { total: u8:2 }),
    (Drive { increment: u8:8, enable: false }, Expect { total: u8:2 }),
    (Drive { increment: u8:3, enable: true }, Expect { total: u8:5 }),
];

block counter(clk, drive: Drive) -> Expect {
    reg total_q: u8 = u8:0;
    if drive.enable {
        total_q := total_q + drive.increment;
    }
    Expect { total: total_q }
}

#[test]
block counter_steps() {
    reg drive: Drive = zero!<Drive>();
    let output = counter(drive);
    tick!(rst: true);
    assert_eq(output, Expect { total: u8:0 });
    for (to_drive, to_expect) in CASES {
        drive := to_drive;
        tick!();
        assert_eq(output, to_expect);
    }
}
"#,
    )?;
    let mut args = vec![
        "bx-test",
        "suite.bx",
        "--dslx-path=lib",
        "--link-ir=externs.ir",
    ];
    let expected = "test arithmetic ... ok\ntest counter_steps ... ok\n2 passed; 0 failed\n";
    assert_eq!(stdout(root, &args)?, expected);
    args.push("--reset-active-low");
    assert_eq!(stdout(root, &args)?, expected);
    Ok(())
}

#[test]
fn bx_quickcheck_compares_each_sample_and_reports_four_valued_outcomes() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("counter.bx"),
        r#"
enum Op : u3 { Hold = 1, Load = 5 }
struct Drive { op: Op, data: uN[71] }
block capture(clk, d: Drive) -> uN[71] {
    reg q: uN[71] = uN[71]:7;
    if d.op == Op::Load { q := d.data; }
    q
}
#[quickcheck]
block checked(trace: Drive[4]) -> bool {
    reg d: Drive = Drive { op: Op::Hold, data: uN[71]:0 };
    reg expected: uN[71] = uN[71]:7;
    let q = capture(d);
    tick!(rst: true);
    assert_eq(q, expected);
    for row in trace {
        assert!(row.op == Op::Hold || row.op == Op::Load);
        d := row;
        if row.op == Op::Load { expected := row.data; }
        tick!();
        assert_eq(q, expected);
    }
    q == expected
}
#[quickcheck]
block observes_power_up() -> bool {
    reg d: Drive = Drive { op: Op::Hold, data: uN[71]:0 };
    let q = capture(d);
    if q == uN[71]:0 { assert!(true); }
    true
}
"#,
    )?;
    for active_low in [false, true] {
        let mut args = vec![
            "bx-test",
            "counter.bx",
            "--quickcheck-trials",
            "12",
            "--seed",
            "38",
            "--cross-check-vastly",
        ];
        if active_low {
            args.push("--reset-active-low");
        }
        let result = driver(directory.path(), &args)?;
        anyhow::ensure!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8(result.stdout)?,
            "quickcheck checked: 12 concrete trials (seed 38)\n  Vastly from X: 12 true, 0 false, 0 indeterminate\ntest checked ... ok\nquickcheck observes_power_up: 12 concrete trials (seed 38)\n  Vastly from X: 0 true, 0 false, 12 indeterminate\ntest observes_power_up ... ok\n2 passed; 0 failed\n"
        );
    }
    Ok(())
}

#[test]
fn bx_quickcheck_replays_a_concrete_counterexample() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("replay.bx"),
        r#"block delay(clk, d: u16) -> u16 {
    reg q: u16;
    q := d;
    q
}
#[quickcheck] block fails(d: u16) -> bool {
    let q = delay(d);
    assert_eq(d, d);
    tick!();
    q != d
}
#[test] fn after() { assert!(true); }
"#,
    )?;
    let args = [
        "bx-test",
        "replay.bx",
        "--quickcheck-trials",
        "5",
        "--seed",
        "3",
    ];
    let first = driver(directory.path(), &args)?;
    assert_eq!(first.status.code(), Some(1));
    assert_eq!(first.stdout, b"quickcheck fails: 1 concrete trials (seed 3)\ntest fails ... FAILED\ntest after ... ok\n1 passed; 1 failed\n");
    let replay = driver(directory.path(), &args)?;
    assert_eq!(first.stderr, replay.stderr);
    compare_golden_text(
        &String::from_utf8(first.stderr)?,
        "tests/bx_quickcheck_counterexample.golden.txt",
    );
    let mut cross_check_args = args.to_vec();
    cross_check_args.push("--cross-check-vastly");
    let from_x = driver(directory.path(), &cross_check_args)?;
    assert_eq!(from_x.status.code(), Some(1));
    assert_eq!(from_x.stdout, b"quickcheck fails: 1 concrete trials (seed 3)\n  Vastly from X: 0 true, 1 false, 0 indeterminate\ntest fails ... FAILED\ntest after ... ok\n1 passed; 1 failed\n");
    Ok(())
}

#[test]
fn bx_test_reports_the_failed_edge_and_continues_with_the_next_test() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("failing.bx"),
        r#"block capture(clk, d: u8) -> u8 {
    reg q: u8 = u8:0;
    q := d;
    q
}

#[test]
block wrong_output() {
    reg drive: u8 = u8:0;
    let output = capture(drive);
    tick!(rst: true);
    drive := u8:42;
    tick!();
    assert_eq(output, u8:41);
}

#[test]
fn still_runs() { assert!(true); }
"#,
    )?;
    let output = driver(directory.path(), &["bx-test", "failing.bx"])?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(output.stdout)?,
        "test wrong_output ... FAILED\ntest still_runs ... ok\n1 passed; 1 failed\n"
    );
    compare_golden_text(
        &String::from_utf8(output.stderr)?,
        "tests/bx_test_failed_edge.golden.txt",
    );
    Ok(())
}

#[test]
fn bx_test_applies_the_bx_warning_policy() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("warning.bx"),
        r#"block identity(clk, x: u8) -> u8 { x }
#[test]
fn sanity() { assert_eq(u8:1, u8:1); }
"#,
    )?;
    let mut args = vec!["bx-test", "warning.bx"];
    let strict = driver(directory.path(), &args)?;
    assert_eq!(strict.status.code(), Some(1));
    assert!(strict.stdout.is_empty());

    args.push("--warnings-as-errors=false");
    let demoted = driver(directory.path(), &args)?;
    assert!(
        demoted.status.success(),
        "{}",
        String::from_utf8_lossy(&demoted.stderr)
    );
    assert_eq!(
        String::from_utf8(demoted.stdout)?,
        "test sanity ... ok\n1 passed; 0 failed\n"
    );
    compare_golden_text(
        &String::from_utf8(demoted.stderr)?,
        "tests/bx_test_combinational_warning.golden.txt",
    );
    Ok(())
}

const EFFECT_ARGS: &[&str] = &["bx2ir", "pipeline.bx", "--dslx-path=source files"];

/// Build a named BX callback whose transitive DSLX multiplier has the given
/// checks.
fn multiplier_design(checks: &str) -> Result<TempDir> {
    let directory = tempfile::tempdir()?;
    let sources = directory.path().join("source files/math");
    fs::create_dir_all(&sources)?;
    fs::write(
        sources.join("multiplier.x"),
        format!(
            r#"pub fn unused(x: u8) -> u8 {{
    assert!(x != u8:0, "not_on_path");
    x
}}

pub fn mul<N: u32>(x: uN[N]) -> uN[N] {{
{checks}
    x * x
}}
"#,
        ),
    )?;
    fs::write(
        sources.join("stage.x"),
        r#"import math.multiplier;
pub fn apply(x: u8) -> u8 { multiplier::mul(x) }
"#,
    )?;
    fs::write(
        directory.path().join("pipeline.bx"),
        r#"use math::stage;
block advance<In, Out, F: fn(In) -> Out>(clk, x: In) -> Out { F(x) }
pub block pipeline(clk, x: u8) -> u8 {
    let y = advance<stage::apply>(x);
    y
}
"#,
    )?;
    Ok(directory)
}

/// Return an import rejection after verifying the CLI produced no partial IR.
fn multiplier_error(directory: &TempDir) -> Result<String> {
    let output = driver(directory.path(), EFFECT_ARGS)?;
    anyhow::ensure!(
        output.status.code() == Some(1),
        "expected an import error, got {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    anyhow::ensure!(output.stdout.is_empty(), "rejected import emitted IR");
    Ok(String::from_utf8(output.stderr)?)
}

#[test]
fn bx_library_next_to_the_input_is_visible_from_another_working_directory() -> Result<()> {
    let directory = TempDir::new()?;
    let source_dir = directory.path().join("design");
    fs::create_dir(&source_dir)?;
    fs::write(
        source_dir.join("stages.bx"),
        "pub block identity<T>(clk, x: T) -> T { x }",
    )?;
    fs::write(
        source_dir.join("circuit.bx"),
        r#"
use stages;
block circuit(clk, x: u8) -> u8 { let y = stages::identity(x); y }
"#,
    )?;
    let ir = stdout(directory.path(), &["bx2ir", "design/circuit.bx"])?;
    let package = Parser::new(&ir).parse_and_verify_package()?;
    assert_eq!(
        package.get_top_block().map(|b| b.name.as_str()),
        Some("circuit")
    );
    Ok(())
}

#[test]
fn bx2ir_reports_each_combinational_block_once_across_source_files() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let library = "pub block invert(clk, x: u8) -> u8 { !x }";
    let source = r#"use ops;
block passthrough(clk, x: u8) -> u8 { x }
pub block circuit(clk, x: u8) -> u8 {
    let first = ops::invert(x);
    let second = ops::invert(first);
    second
}
"#;
    fs::write(root.join("ops.bx"), library)?;
    fs::write(root.join("circuit.bx"), source)?;
    let demoted = driver(root, &["bx2ir", "circuit.bx", "--warnings-as-errors=false"])?;
    let warnings = String::from_utf8(demoted.stderr)?;
    assert!(demoted.status.success(), "{warnings}");
    compare_golden_text(&warnings, "tests/bx2ir_combinational_warnings.golden.txt");

    let strict = driver(root, &["bx2ir", "circuit.bx"])?;
    assert_eq!(strict.status.code(), Some(1));
    assert!(strict.stdout.is_empty());
    assert_eq!(
        String::from_utf8(strict.stderr)?,
        format!("{warnings}xlsynth-driver: bx2ir: BX warnings treated as errors\n")
    );

    fs::write(
        root.join("ops.bx"),
        format!("#[allow(combinational_block)]\n{library}"),
    )?;
    fs::write(
        root.join("circuit.bx"),
        source.replace(
            "block passthrough",
            "#[allow(combinational_block)]\nblock passthrough",
        ),
    )?;
    let suppressed = driver(root, &["bx2ir", "circuit.bx", "--warnings-as-errors"])?;
    assert!(
        suppressed.status.success(),
        "{}",
        String::from_utf8_lossy(&suppressed.stderr)
    );
    assert!(suppressed.stderr.is_empty());
    assert_eq!(suppressed.stdout, demoted.stdout);
    Ok(())
}

#[test]
fn bx2gv_applies_the_bx_warning_policy_before_writing_the_netlist() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(root.join("cells.lib"), MAPPING_CELLS)?;
    stdout(root, &["lib2proto", "cells.lib", "--output=cells.proto"])?;
    fs::write(
        root.join("design.bx"),
        r#"
block invert(clk, d: bool) -> bool { !d }
pub block capture(clk, d: bool) -> bool {
    reg q: bool;
    let inv = invert(d);
    q := inv;
    q
}
"#,
    )?;
    let mut args = vec![
        "bx2gv",
        "design.bx",
        "--liberty-proto=cells.proto",
        "-o",
        "mapped.pb",
    ];
    let strict = driver(root, &args)?;
    assert_eq!(strict.status.code(), Some(1));
    assert!(strict.stdout.is_empty());
    assert!(!root.join("mapped.pb").try_exists()?);
    args.push("--warnings-as-errors=false");
    let demoted = driver(root, &args)?;
    let warnings = String::from_utf8(demoted.stderr)?;
    assert!(demoted.status.success(), "{warnings}");
    assert_eq!(
        String::from_utf8(strict.stderr)?,
        format!("{warnings}xlsynth-driver: bx2gv: BX warnings treated as errors\n")
    );
    assert!(demoted.stdout.is_empty());
    let netlist = pb::Netlist::decode(fs::read(root.join("mapped.pb"))?.as_slice())?;
    assert_eq!(netlist.modules.len(), 1);
    Ok(())
}

#[test]
fn transitive_runtime_width_assertions_point_at_the_multiplier() -> Result<()> {
    let directory = multiplier_design(
        r#"    assert!(N >= u32:2, "min_width");
    assert!(N <= u32:32, "max_width");"#,
    )?;
    compare_golden_text(
        &multiplier_error(&directory)?,
        "tests/bx2ir_import_width_assertions.golden.txt",
    );
    Ok(())
}

#[test]
fn transitive_assertion_in_a_for_loop_points_at_the_callee() -> Result<()> {
    let directory = multiplier_design(r#"    assert!(N >= u32:2, "min_width");"#)?;
    for body in [
        "for (_, acc) in u32:0..u32:1 { multiplier::mul(acc) }(x)",
        r#"let y = for (_, acc) in u32:0..u32:1 {
        let next = multiplier::mul(acc);
        next
    }(x);
    y"#,
    ] {
        fs::write(
            directory.path().join("source files/math/stage.x"),
            format!(
                r#"import math.multiplier;
pub fn apply(x: u8) -> u8 {{
    {body}
}}
"#,
            ),
        )?;
        compare_golden_text(
            &multiplier_error(&directory)?,
            "tests/bx2ir_import_loop_assertion.golden.txt",
        );
    }
    Ok(())
}

#[test]
fn uncertain_loop_effect_order_keeps_the_verified_call_path() -> Result<()> {
    let directory = multiplier_design("")?;
    for body in [
        "for (_, acc) in u32:0..u32:1 { checked(acc) }(seed(x))",
        "for (_, acc) in u32:0..u32:0 { checked(acc) }(x)",
        "for (_, acc) in u32:0..u32:1 { if false { seed(acc) } else { checked(acc) } }(x)",
        r#"for (_, acc) in u32:0..u32:1 {
        let y: uN[seed(u8:8) as u32] = checked(acc);
        y
    }(x)"#,
        r#"let y: uN[seed(u8:8) as u32] = for (_, acc) in u32:0..u32:1 { checked(acc) }(x);
    y"#,
        r#"for (_, acc) in u32:0..u32:1 {
        assert_fmt!(acc != u8:0, "seed {}", seed(u8:1));
        checked(acc)
    }(x)"#,
    ] {
        fs::write(
            directory.path().join("source files/math/multiplier.x"),
            format!(
                r#"fn seed(x: u8) -> u8 {{
    assert!(x != u8:0, "seed");
    x
}}
fn checked(x: u8) -> u8 {{
    assert!(x != u8:0, "body");
    x
}}
pub fn mul(x: u8) -> u8 {{
    {body}
}}
"#
            ),
        )?;
        compare_golden_text(
            &multiplier_error(&directory)?,
            "tests/bx2ir_import_implicit_token_path.golden.txt",
        );
    }
    Ok(())
}

#[test]
fn first_effect_follows_dslx_call_order() -> Result<()> {
    let directory = multiplier_design(r#"    assert!(N >= u32:2, "min_width");"#)?;
    let call = "    let y = multiplier::mul(x);";
    let assertion = r#"    assert!(x != u8:0, "stage_check");"#;
    for (first, second, expected_location) in [
        (call, assertion, "source files/math/multiplier.x:7:5"),
        (assertion, call, "source files/math/stage.x:7:5"),
        (
            r#"    assert!(multiplier::mul(x) != u8:0, "stage_check");"#,
            "    let y = x;",
            "source files/math/multiplier.x:7:5",
        ),
        (
            "    let y = checked(x);",
            "    let _ = for (_, checked) in u32:0..u32:1 { checked }(y);",
            "source files/math/stage.x:3:5",
        ),
    ] {
        fs::write(
            directory.path().join("source files/math/stage.x"),
            format!(
                r#"import math.multiplier;
fn checked(x: u8) -> u8 {{
    assert!(x != u8:0, "helper");
    x
}}
pub fn apply(x: u8) -> u8 {{
{first}
{second}
    y
}}
"#
            ),
        )?;
        let error = multiplier_error(&directory)?;
        let first_location = error
            .lines()
            .find_map(|line| line.trim().strip_prefix("--> "));
        assert_eq!(first_location, Some(expected_location), "{error}");
    }
    Ok(())
}

#[test]
fn compile_time_width_checks_use_native_dslx_validation() -> Result<()> {
    let pure = multiplier_design(
        r#"    const_assert!(N >= u32:2);
    const_assert!(N <= u32:32);"#,
    )?;
    let ir = stdout(pure.path(), EFFECT_ARGS)?;
    assert_eq!(
        Parser::new(&ir).parse_and_verify_package()?.top,
        Some(("pipeline".into(), MemberType::Block)),
    );

    let invalid = multiplier_design("    const_assert!(N >= u32:16);")?;
    let error = multiplier_error(&invalid)?;
    assert!(
        error.contains("source files/math/multiplier.x:7:5-7:32"),
        "{error}"
    );
    assert!(error.contains("const_assert! failure"), "{error}");
    assert!(error.contains("N: u32:8"), "{error}");
    Ok(())
}

#[test]
fn imported_coverage_reports_its_call_path() -> Result<()> {
    let directory = multiplier_design(r#"    cover!("nonzero", x != uN[N]:0);"#)?;
    compare_golden_text(
        &multiplier_error(&directory)?,
        "tests/bx2ir_import_cover.golden.txt",
    );
    Ok(())
}

#[test]
fn effect_in_an_anonymous_map_reports_the_verified_dslx_path() -> Result<()> {
    let map = r#"    let _ = map([x], |a: uN[N]| { assert!(a == uN[N]:0, "lambda"); a });"#;
    for later in ["", r#"    assert!(N >= u32:2, "later_width_check");"#] {
        let directory = multiplier_design(&format!("{map}\n{later}"))?;
        compare_golden_text(
            &multiplier_error(&directory)?,
            "tests/bx2ir_import_implicit_token_path.golden.txt",
        );
    }
    Ok(())
}

#[test]
fn compile_time_invocations_can_contribute_to_an_implicit_token() -> Result<()> {
    let directory = multiplier_design("")?;
    for (extra_param, statement) in [
        ("", r#"const K = assert!(true, "constant");"#),
        ("", "const_assert!(helper(u8:1) != u8:0);"),
        ("", "let _ = id<{helper(u8:1) as u32}>(x as u8);"),
        (", M: u32 = {helper(u8:1) as u32}", ""),
    ] {
        fs::write(
            directory.path().join("source files/math/multiplier.x"),
            format!(
                r#"fn helper(x: u8) -> u8 {{ assert!(x != u8:0, "helper_data"); x }}
fn id<K: u32>(x: u8) -> u8 {{ x }}
pub fn mul<N: u32{extra_param}>(x: uN[N]) -> u8 {{
    {statement}
    assert!(N > u32:0, "later_width");
    helper(x as u8)
}}"#,
            ),
        )?;
        compare_golden_text(
            &multiplier_error(&directory)?,
            "tests/bx2ir_import_implicit_token_path.golden.txt",
        );
    }
    Ok(())
}

#[test]
fn repeated_builtin_or_callee_cannot_blame_a_later_width_assertion() -> Result<()> {
    let check_diagnostic = |directory: &TempDir, later_line: usize| -> Result<()> {
        let error = multiplier_error(directory)?;
        let first_location = error
            .lines()
            .find_map(|line| line.trim().strip_prefix("--> "));
        let later_check = format!("source files/math/multiplier.x:{later_line}:5");
        assert_ne!(first_location, Some(later_check.as_str()), "{error}");
        assert!(!error.contains("const_assert!("), "{error}");
        assert!(
            error.contains("math.stage::apply -> math.multiplier::mul"),
            "{error}"
        );
        Ok(())
    };
    let earlier_loop = r#"    let _ = for (_, acc): (u32, uN[N]) in u32:0..u32:1 {
        assert!(acc != uN[N]:0, "loop_data");
        acc
    }(x);"#;
    let width_check = r#"    assert!(N >= u32:2, "later_width");"#;
    let loop_case = multiplier_design(&format!("{earlier_loop}\n{width_check}"))?;
    check_diagnostic(&loop_case, 11)?;
    let call_case = multiplier_design("")?;
    for (formals, first_call, last_call) in [
        ("<K: u32>", "helper<{N + u32:1}>(x as u8)", "helper<N>(y)"),
        ("", "((helper))(x as u8)", "helper(y)"),
        ("", "map([x as u8], (helper))[u32:0]", "helper(y)"),
    ] {
        fs::write(
            call_case.path().join("source files/math/multiplier.x"),
            format!(
                r#"fn helper{formals}(x: u8) -> u8 {{
    assert!(x != u8:0, "helper_data");
    x
}}
pub fn mul<N: u32>(x: uN[N]) -> u8 {{
    let y = {first_call};
    assert!(N >= u32:2, "later_width");
    {last_call}
}}"#
            ),
        )?;
        check_diagnostic(&call_case, 7)?;
    }
    Ok(())
}

#[test]
fn a_runtime_guard_on_a_transitive_call_suppresses_the_compile_time_hint() -> Result<()> {
    let directory = multiplier_design(r#"    assert!(N >= u32:16, "min_width");"#)?;
    fs::write(
        directory.path().join("source files/math/stage.x"),
        r#"import math.multiplier;
pub fn apply(x: u8) -> u8 { if x != u8:0 { multiplier::mul(x) } else { x } }"#,
    )?;
    let error = multiplier_error(&directory)?;
    assert!(
        error.contains("source files/math/multiplier.x:7:5"),
        "{error}"
    );
    assert!(!error.contains("const_assert!("), "{error}");
    Ok(())
}

#[test]
fn implicit_token_diagnostic_handles_nested_tuple_patterns() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("effects.x"),
        r#"pub fn inner(x: ((u8,), u8)) -> u8 {
    let ((a,), b) = x;
    assert!(a == b, "x");
    a + b
}
"#,
    )?;
    fs::write(
        directory.path().join("pipeline.bx"),
        "use effects; pub block pipeline(clk, x: ((u8,), u8)) -> u8 { effects::inner(x) }",
    )?;
    let output = driver(directory.path(), &["bx2ir", "pipeline.bx", "--dslx-path=."])?;
    let error = String::from_utf8(output.stderr)?;
    assert_eq!(output.status.code(), Some(1), "{error}");
    assert!(error.contains("requires an implicit token"), "{error}");
    Ok(())
}

#[test]
fn data_dependent_assertions_and_failures_report_the_original_operation() -> Result<()> {
    for (checks, operation) in [
        (
            r#"    assert!(x != uN[N]:0, "nonzero_data");"#,
            r#"assert!(x != uN[N]:0, "nonzero_data")"#,
        ),
        (
            r#"    if x == uN[N]:0 { fail!("zero_data", x) } else { x };"#,
            r#"fail!("zero_data", x)"#,
        ),
        (
            r#"    if x != uN[N]:0 { assert!(N >= u32:2, "guarded"); x } else { x };"#,
            r#"assert!(N >= u32:2, "guarded")"#,
        ),
    ] {
        let directory = multiplier_design(checks)?;
        let error = multiplier_error(&directory)?;
        assert!(
            error.contains("source files/math/multiplier.x:7:"),
            "{error}"
        );
        assert!(error.contains(operation), "{error}");
        // Each assertion predicate or its execution depends on runtime input x.
        assert!(!error.contains("const_assert!("), "{error}");
    }
    Ok(())
}

#[test]
fn imported_dslx_logic_runs_at_the_bx_register_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("empty"))?;
    fs::create_dir(root.join("library"))?;
    fs::write(
        root.join("library/arithmetic.x"),
        "pub fn increment(x: u8) -> u8 { x + u8:1 }",
    )?;
    fs::write(
        root.join("design.bx"),
        r#"use arithmetic::increment;
block registered_sum(clk, x: u8) -> u8 {
    reg q: u8;
    q := increment(x);
    q
}"#,
    )?;
    let ir = stdout(
        root,
        &["bx2ir", "design.bx", "--dslx-path", "empty;library"],
    )?;
    let package = Parser::new(&ir).parse_and_verify_package()?;
    assert_eq!(package.name, "design");
    assert_eq!(
        package.top,
        Some(("registered_sum".into(), MemberType::Block))
    );

    fs::write(root.join("design.ir"), ir)?;
    let rtl = stdout(root, &["block2sv", "design.ir"])?;
    let module = compile_pipeline_module(&rtl).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let x = runner.port_bindings().input_with_width("x", 8).unwrap();
    let mut previous = None;
    for (input, expected) in [(7, 8), (255, 0)] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| drive.set_u64(&x, input))
            .unwrap();
        assert_eq!(edge.pre_outputs["out"].to_u64_if_known(), previous);
        assert_eq!(edge.outputs["out"].to_u64_if_known(), Some(expected));
        previous = Some(expected);
    }
    Ok(())
}

#[test]
fn inferred_dslx_struct_can_share_its_name_with_functions_and_constants() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(root.join("wire.x"), "pub struct Leaf { z: u8 }")?;
    fs::write(
        root.join("factory.x"),
        "import wire; pub fn make(x: u8) -> (wire::Leaf,) { (wire::Leaf { z: x },) }",
    )?;
    fs::write(
        root.join("design.bx"),
        r#"use decoy; use factory;
#[allow(combinational_block)]
block b(clk, x: u8) -> u8 { factory::make(x).0.z }"#,
    )?;
    // The generated type probe for decoy::Leaf fails. The subsequent probe
    // for wire::Leaf must still see an intact DSLX inference graph.
    for decoy in ["pub fn Leaf(x: u8) -> u8 { x }", "pub const Leaf = u32:2;"] {
        fs::write(root.join("decoy.x"), decoy)?;
        let ir = stdout(root, &["bx2ir", "design.bx"])?;
        let package = Parser::new(&ir).parse_and_verify_package()?;
        assert!(package.get_fn("__factory__make").is_some());
    }
    Ok(())
}

#[test]
fn dslx_path_replaces_the_configured_import_directories() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("configured"))?;
    fs::create_dir(root.join("empty"))?;
    fs::write(
        root.join("tools.toml"),
        r#"[toolchain.dslx]
dslx_path = ["configured"]
"#,
    )?;
    fs::write(
        root.join("configured/custom_math.x"),
        "pub fn twice(x: u8) -> u8 { x + x }",
    )?;
    fs::write(
        root.join("math_pipe.bx"),
        r#"use custom_math;
#[allow(combinational_block)]
block math_pipe(clk, x: u8) -> u8 { custom_math::twice(x) }"#,
    )?;

    let mut args = vec!["--toolchain=tools.toml", "bx2ir", "math_pipe.bx"];
    let ir = stdout(root, &args)?;
    Parser::new(&ir).parse_and_verify_package()?;

    args.push("--dslx-path=empty");
    let output = driver(root, &args)?;
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("custom_math"), "{stderr}");
    Ok(())
}

#[test]
fn extern_function_links_into_the_dslx_package() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("arithmetic.x"),
        "pub fn twice(x: u8) -> u8 { x + x }",
    )?;
    fs::write(
        root.join("arithmetic.ir"),
        stdout(root, &["dslx2ir", "--dslx_input_file=arithmetic.x"])?,
    )?;
    fs::write(
        root.join("doubler.bx"),
        r#"extern fn arithmetic::twice(x: u8) -> u8;
#[allow(combinational_block)]
block doubler(clk, x: u8) -> u8 { arithmetic::twice(x) }"#,
    )?;
    let actual = stdout(root, &["bx2ir", "doubler.bx", "--link-ir=arithmetic.ir"])?;
    assert_eq!(actual, include_str!("bx2ir_linked.golden.ir"));

    // Handwritten XLS source can leave the parameter and add IDs implicit.
    fs::write(
        root.join("arithmetic.ir"),
        r#"package arithmetic
file_number 0 "arithmetic.x"
fn __arithmetic__twice(x: bits[8]) -> bits[8] {
  ret sum: bits[8] = add(x, x, pos=[(0,0,28)])
}"#,
    )?;
    let from_source = stdout(root, &["bx2ir", "doubler.bx", "--link-ir=arithmetic.ir"])?;
    let package = xlsynth::IrPackage::parse_ir(&from_source, None)?;
    let twice = package.get_function("__arithmetic__twice")?;
    assert_eq!(
        twice.interpret(&[xlsynth::XlsIrValue::make_ubits(8, 21)?])?,
        xlsynth::XlsIrValue::make_ubits(8, 42)?
    );
    Ok(())
}

#[test]
fn top_override_and_reset_flags_apply_to_a_standalone_module() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("blocks.bx"),
        r#"#[allow(combinational_block)]
pub block passthrough(clk, x: u8) -> u8 { x }
block delay(clk, x: u8) -> u8 {
    reg q: u8 = 0;
    q := x;
    q
}"#,
    )?;
    let default_ir = stdout(root, &["bx2ir", "blocks.bx"])?;
    let package = Parser::new(&default_ir).parse_and_verify_package()?;
    assert_eq!(package.top, Some(("passthrough".into(), MemberType::Block)));
    let default_reset = package
        .get_block("delay")
        .context("missing delay")?
        .reset
        .as_ref()
        .context("missing reset")?;
    assert!(!default_reset.active_low);
    assert!(!default_reset.asynchronous);

    let ir = stdout(
        root,
        &[
            "bx2ir",
            "blocks.bx",
            "--top=delay",
            "--reset-active-low",
            "--reset-asynchronous=true",
        ],
    )?;
    let package = Parser::new(&ir).parse_and_verify_package()?;
    assert_eq!(package.top, Some(("delay".into(), MemberType::Block)));
    let top = package.get_top_block().context("missing top")?;
    let reset = top.reset.as_ref().context("missing reset")?;
    assert!(reset.active_low);
    assert!(reset.asynchronous);
    Ok(())
}

#[test]
fn bx2lean_emits_the_selected_block_and_preserves_existing_output() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("design.bx"),
        r#"pub block passthrough(clk, x: u8) -> u8 { x }
struct Outputs { seen: u8, valid: bool }
block counter(clk, rst, x: u8) -> Outputs {
    reg count: u8 = 0;
    count := x;
    Outputs { seen: count, valid: true }
}"#,
    )?;
    let options = ["design.bx", "--top=counter", "--reset-active-low"];
    let args = [&["bx2lean"][..], &options, &["--output-dir=lean"]].concat();
    assert!(stdout(root, &args)?.is_empty());
    let mut files = fs::read_dir(root.join("lean"))?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    assert_eq!(
        files,
        [
            "Generated.lean",
            "ReadableCorrespondence.lean",
            "ReadableGenerated.lean",
            "ReadableRepresentation.lean",
            "ReadableTypes.lean",
        ]
    );
    fs::write(
        root.join("saved.ir"),
        stdout(root, &[&["bx2ir"][..], &options].concat())?,
    )?;
    assert_eq!(
        fs::read_to_string(root.join("lean/Generated.lean"))?,
        stdout(root, &["block2lean", "saved.ir"])?
    );
    fs::write(root.join("lean/Generated.lean"), "keep")?;
    assert!(!driver(root, &args)?.status.success());
    assert_eq!(
        fs::read_to_string(root.join("lean/Generated.lean"))?,
        "keep"
    );
    let missing = driver(
        root,
        &[
            "bx2lean",
            "design.bx",
            "--top=unknown",
            "--output-dir=missing",
        ],
    )?;
    assert!(!missing.status.success());
    assert!(String::from_utf8(missing.stderr)?.contains("unknown"));
    assert!(!root.join("missing").try_exists()?);
    Ok(())
}

#[test]
fn ambiguous_and_missing_top_fail_without_emitting_ir() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("choices.bx"),
        r#"block a(clk, x: u8) -> u8 { x }
block b(clk, x: u8) -> u8 { x + x }"#,
    )?;
    for (arguments, diagnostic) in [
        (vec!["bx2ir", "choices.bx"], "--top"),
        (vec!["bx2ir", "choices.bx", "--top=absent"], "absent"),
    ] {
        let output = driver(root, &arguments)?;
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stderr.contains(diagnostic), "{stderr}");
    }
    Ok(())
}

#[test]
fn mapped_protobuf_matches_the_staged_pipeline_with_synchronous_reset_and_enable() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("modules"))?;
    fs::write(
        root.join("modules/math.x"),
        "pub fn invert(x: u2) -> u2 { !x }",
    )?;
    fs::write(
        root.join("pipeline.bx"),
        r#"use math;
block pipeline(clk, x: u2, en: bool) -> u2 {
    reg q: u2 = 0;
    if en { q := math::invert(x); }
    q
}"#,
    )?;
    fs::write(root.join("cells.lib"), MAPPING_CELLS)?;
    stdout(root, &["lib2proto", "cells.lib", "--output=cells.proto"])?;
    let bx_options = ["pipeline.bx", "--dslx-path=modules", "--reset-active-low"];
    let map_options = ["--liberty-proto=cells.proto", "--module-name=mapped"];

    fs::write(
        root.join("pipeline.ir"),
        stdout(root, &[&["bx2ir"][..], &bx_options].concat())?,
    )?;
    fs::write(
        root.join("pipeline.opt.ir"),
        stdout(root, &["ir2opt", "pipeline.ir", "--top=pipeline"])?,
    )?;
    fs::write(
        root.join("pipeline.g8r"),
        stdout(
            root,
            &[
                "ir2g8r",
                "pipeline.opt.ir",
                "--enable-formal-array-alias-analysis=false",
            ],
        )?,
    )?;
    stdout(
        root,
        &[
            &["g8r2gv", "pipeline.g8r", "--strategy=area"][..],
            &map_options,
            &["-o", "staged.pb"],
        ]
        .concat(),
    )?;
    let staged = fs::read(root.join("staged.pb"))?;

    let direct = [&["bx2gv"][..], &bx_options, &map_options].concat();
    let mut file_args = direct.clone();
    file_args.extend(["-o", "direct.pb"]);
    assert!(stdout(root, &file_args)?.is_empty());
    assert_eq!(fs::read(root.join("direct.pb"))?, staged);
    let mut stdout_args = direct.clone();
    stdout_args.push("--netlist-format=pb");
    let output = driver(root, &stdout_args)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, staged);
    let netlist = pb::Netlist::decode(staged.as_slice())?;
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("expected one mapped module")
    };
    assert_eq!(module.name, "mapped");
    assert_eq!(
        module
            .instances
            .iter()
            .filter(|instance| netlist.interfaces[instance.interface as usize].cell == "DFF")
            .count(),
        2
    );

    let mut async_args = direct;
    async_args.extend(["--reset-asynchronous", "-o", "async.pb"]);
    let output = driver(root, &async_args)?;
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!root.join("async.pb").try_exists()?);
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("asynchronous reset"), "{stderr}");
    Ok(())
}

const MAPPING_CELLS: &str = r#"library(test) {
  cell(BUF) { area: 1; pin(A) { direction: input; }
    pin(Y) { direction: output; function: "A"; } }
  cell(INV) { area: 1; pin(A) { direction: input; }
    pin(Y) { direction: output; function: "!A"; } }
  cell(NAND2) { area: 2; pin(A) { direction: input; } pin(B) { direction: input; }
    pin(Y) { direction: output; function: "!(A & B)"; } }
  cell(DFF) { area: 3; pin(D) { direction: input; } pin(CLK) { direction: input; clock: true; }
    pin(Q) { direction: output; function: "IQ"; }
    ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
}"#;

#[test]
fn bx_child_register_uses_parent_clock_despite_linked_port_name() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("delay.ir"),
        r#"package cells
block delay(CK: clock, d: bits[1], out: bits[1]) {
  reg q(bits[1])
  d: bits[1] = input_port(name=d, id=1)
  register_write.2: () = register_write(d, register=q, id=2)
  register_read.3: bits[1] = register_read(register=q, id=3)
  out: () = output_port(register_read.3, name=out, id=4)
}"#,
    )?;
    fs::write(
        root.join("parent.bx"),
        r#"extern block delay(clk, d: bool) -> bool;
block parent(clk, x: bool) -> bool {
    let delayed = delay(x);
    delayed
}"#,
    )?;
    fs::write(root.join("cells.lib"), MAPPING_CELLS)?;
    stdout(root, &["lib2proto", "cells.lib", "--output=cells.proto"])?;
    stdout(
        root,
        &[
            "bx2gv",
            "parent.bx",
            "--link-ir=delay.ir",
            "--liberty-proto=cells.proto",
            "-o",
            "parent.pb",
        ],
    )?;
    let netlist = pb::Netlist::decode(fs::read(root.join("parent.pb"))?.as_slice())?;
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("expected one flattened module")
    };
    let [flop] = module.instances.as_slice() else {
        anyhow::bail!("expected one mapped register")
    };
    let interface = &netlist.interfaces[flop.interface as usize];
    assert_eq!(interface.cell, "DFF");
    let clock_pin = interface
        .pins
        .iter()
        .position(|pin| pin == "CLK")
        .context("DFF is missing CLK")?;
    let clock_net = flop.nets[clock_pin];
    assert_eq!(module.nets[clock_net as usize].name, "clk");
    assert!(module.ports.iter().any(|port| port.net == clock_net));
    Ok(())
}
