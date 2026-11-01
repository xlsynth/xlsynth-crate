// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::fs;
use std::process::Command;

use test_case::test_case;

/// Splitting a directory name at a colon would substitute a different helper
/// and prove a property that fails on the actual input.
#[test_case("dslx-equiv", &["lhs.x", "rhs.x", "--dslx_top=main", "--assume-enum-in-bound=false"]; "equivalence")]
#[test_case("prove-quickcheck", &["--dslx_input_file=qc.x"]; "quickcheck")]
fn conflicting_import_paths(command: &str, args: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    for name in ["unused", "deps:prod", "deps"] {
        fs::create_dir(path.join(name)).unwrap();
    }
    for (name, body) in [("deps:prod", "x != u8:0"), ("deps", "x == x")] {
        fs::write(
            path.join(name).join("helper.x"),
            format!("pub fn pred(x: u8) -> bool {{ {body} }}"),
        )
        .unwrap();
    }
    fs::write(
        path.join("lhs.x"),
        "import helper; pub fn main(x: u8) -> bool { helper::pred(x) }",
    )
    .unwrap();
    fs::write(path.join("rhs.x"), "pub fn main(x: u8) -> bool { x == x }").unwrap();
    fs::write(
        path.join("qc.x"),
        "import helper; #[quickcheck] fn property(x: u8) -> bool { helper::pred(x) }",
    )
    .unwrap();
    let tools = std::env::var("XLSYNTH_TOOLS").expect("toolchain tests require XLSYNTH_TOOLS");
    fs::write(
        path.join("tools.toml"),
        format!("[toolchain]\ntool_path = {tools:?}\n"),
    )
    .unwrap();

    let run = |solver: &str, toolchain: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        if toolchain {
            cmd.arg("--toolchain=tools.toml");
        }
        let output = cmd
            .current_dir(path)
            .arg(command)
            .args(args)
            .arg(format!("--solver={solver}"))
            .args(["--dslx_path=unused;deps:prod", "--output_json=result.json"])
            .output()
            .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("result.json")).unwrap()).unwrap();
        fs::remove_file(path.join("result.json")).unwrap();
        assert_eq!(output.status.code(), Some(1), "{result}");
        result
    };

    let external = run("toolchain", true);
    assert_eq!(external["status"], "execution-error", "{external}");
    assert_eq!(
        external["error_str"], r#"DSLX search path contains the separator ':': "deps:prod""#,
        "{external}"
    );
    #[cfg(feature = "has-bitwuzla")]
    {
        let native = run("bitwuzla", false);
        assert_eq!(native["status"], "disproved", "{native}");
    }
}
