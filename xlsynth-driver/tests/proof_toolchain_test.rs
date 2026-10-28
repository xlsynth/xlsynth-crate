// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::fs;
use std::process::Command;

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
    for args in [
        vec![
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--solver=toolchain",
            "--tactic_json=solve.json",
        ],
        vec![
            "dslx-equiv",
            "lhs.x",
            "rhs.x",
            "--dslx_top=main",
            "--solver=toolchain",
            "--assume-enum-in-bound=false",
            "--assertion-semantics=ignore",
            "--tactic_json=solve.json",
        ],
        vec!["prover", "--plan_json_file=plan.json"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(path)
            .env_remove("XLSYNTH_TOOLS")
            .arg("--toolchain=selected tools.toml")
            .args(&args)
            .arg("--output_json=result.json")
            .output()
            .unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("result.json")).unwrap()).unwrap();
        assert_eq!(report["status"], "proved");
        fs::remove_file(path.join("result.json")).unwrap();
    }
}
