// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::process::Command;

/// Toolchain loading happens after argument parsing and must produce the same
/// execution-error report as setup inside a proof command.
#[test]
fn toolchain_setup_failures_write_proof_reports() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(std::ffi::OsStr::from_bytes(b"proof-\xff"));
    fs::create_dir(&path).unwrap();
    fs::write(path.join("xlsynth-toolchain.toml"), "[toolchain\n").unwrap();
    fs::write(path.join("type.toml"), "[toolchain]\ntool_path = true\n").unwrap();
    fs::create_dir(path.join("directory.toml")).unwrap();
    for (args, toolchain) in [
        (&["ir-equiv", "lhs.ir", "rhs.ir"][..], Some("missing.toml")),
        (
            &["dslx-equiv", "lhs.x", "rhs.x", "--dslx_top=main"][..],
            None,
        ),
        (
            &["prove-quickcheck", "--dslx_input_file=qc.x"][..],
            Some("type.toml"),
        ),
        (
            &[
                "dslx-fn-prove-assertions",
                "--dslx_input_file=entry.x",
                "--dslx_top=main",
            ][..],
            Some("directory.toml"),
        ),
    ] {
        let result_path = path.join("result.json");
        fs::write(&result_path, r#"{"status":"proved","success":true}"#).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        cmd.current_dir(&path).args(args);
        if let Some(toolchain) = toolchain {
            cmd.args(["--toolchain", toolchain]);
        }
        let output = cmd.arg("--output_json=result.json").output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(&result_path).unwrap()).unwrap();
        assert_eq!(report["status"], "execution-error", "{args:?}: {report}");
        assert_eq!(report["success"], false);
        assert!(
            report["error_str"]
                .as_str()
                .unwrap()
                .contains(toolchain.unwrap_or("xlsynth-toolchain.toml"))
        );
    }
}

/// A directory in a TOML or JSON list is a single import root even when its
/// name contains the driver's CLI separator.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn worker_never_splits_a_literal_import_root() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    for (root, predicate) in [("prod;bad", "x != u8:0"), ("prod", "x == x")] {
        fs::create_dir(path.join(root)).unwrap();
        fs::write(
            path.join(root).join("helper.x"),
            format!("pub fn pred(x: u8) -> bool {{ {predicate} }}"),
        )
        .unwrap();
    }
    fs::write(
        path.join("tools.toml"),
        r#"[toolchain.dslx]
dslx_path = ["prod;bad"]
"#,
    )
    .unwrap();
    fs::write(
        path.join("qc.x"),
        "import helper; #[quickcheck] fn check(x: u8) -> bool { helper::pred(x) }",
    )
    .unwrap();
    fs::write(
        path.join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();
    fs::write(path.join("plan.json"), r#"{"kind":"prove-quickcheck","dslx_input_file":"qc.x","solver":"bitwuzla","dslx_path":["prod;bad"]}"#).unwrap();
    for (index, (args, expected)) in [
        (
            &[
                "prove-quickcheck",
                "--dslx_input_file=qc.x",
                "--solver=bitwuzla",
            ][..],
            "disproved",
        ),
        (
            &[
                "prove-quickcheck",
                "--dslx_input_file=qc.x",
                "--solver=bitwuzla",
                "--tactic_json=solve.json",
            ][..],
            "execution-error",
        ),
        (
            &["prover", "--plan_json_file=plan.json"][..],
            "execution-error",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let report_file = format!("report_{index}.json");
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(path)
            .arg("--toolchain=tools.toml")
            .args(args)
            .args(["--output_json", &report_file])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(report_file)).unwrap()).unwrap();
        assert_eq!(report["status"], expected, "{args:?}: {report}");
    }
}

/// Workers must honor an explicit toolchain file even when the tool directory
/// is absent from their environment.
#[test]
fn explicit_toolchain_reaches_tactic_and_plan_workers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let tools = std::env::var("XLSYNTH_TOOLS").expect("toolchain tests require XLSYNTH_TOOLS");
    fs::write(
        path.join("selected tools.toml"),
        format!(
            r#"[toolchain]
tool_path = {tools:?}
[toolchain.dslx]
dslx_path = ["deps", "more deps"]
disable_warnings = ["unused_definition"]
"#
        ),
    )
    .unwrap();
    fs::create_dir(path.join("deps")).unwrap();
    fs::create_dir(path.join("more deps")).unwrap();
    fs::write(
        path.join("more deps/helper.x"),
        "pub fn bump(x: u8) -> u8 { x + u8:1 }",
    )
    .unwrap();
    fs::write(
        path.join("lhs.x"),
        "import helper; pub fn main(x: u8) -> u8 { helper::bump(x) }",
    )
    .unwrap();
    fs::write(path.join("rhs.x"), "pub fn main(x: u8) -> u8 { x + u8:1 }").unwrap();
    fs::write(
        path.join("qc.x"),
        "import helper; #[quickcheck] fn property(x: u8) -> bool { helper::bump(x) == x + u8:1 }",
    )
    .unwrap();
    fs::write(
        path.join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();
    fs::write(
        path.join("plan.json"),
        r#"{"kind":"prove-quickcheck","dslx_input_file":"qc.x","solver":"toolchain"}"#,
    )
    .unwrap();
    let check = |args: &[&str], flags: &[&str], expected: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(path)
            .env_remove("XLSYNTH_TOOLS")
            .arg("--toolchain=selected tools.toml")
            .args(args)
            .args(flags)
            .arg("--output_json=result.json")
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            expected == "proved",
            "{args:?} {flags:?}: {output:?}"
        );
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("result.json")).unwrap()).unwrap();
        assert_eq!(report["status"], expected, "{args:?} {flags:?}: {report}");
        fs::remove_file(path.join("result.json")).unwrap();
    };
    let equiv = [
        "dslx-equiv",
        "lhs.x",
        "rhs.x",
        "--dslx_top=main",
        "--solver=toolchain",
        "--assume-enum-in-bound=false",
        "--assertion-semantics=ignore",
    ];
    let solve = "--tactic_json=solve.json";
    check(
        &[
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--solver=toolchain",
        ],
        &[solve],
        "proved",
    );
    check(&equiv, &[], "proved");
    check(&equiv, &[solve], "proved");
    // The imported helper exists only under the configured search roots.
    check(&equiv, &["--dslx_path="], "execution-error");
    check(&equiv, &["--dslx_path=", solve], "execution-error");

    let plan = ["prover", "--plan_json_file=plan.json"];
    check(&plan, &[], "proved");
    fs::write(
        path.join("plan.json"),
        r#"{"kind":"prove-quickcheck","dslx_input_file":"qc.x","solver":"toolchain","dslx_path":[]}"#,
    )
    .unwrap();
    check(&plan, &[], "execution-error");
}
