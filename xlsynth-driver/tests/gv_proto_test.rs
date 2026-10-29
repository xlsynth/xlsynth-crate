// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;
use std::process::Command;

use anyhow::{Result, ensure};
use flate2::Compression;
use flate2::write::GzEncoder;
use xlsynth_test_helpers::compare_golden_text;

/// Runs a conversion with a named input and captures its exact stdout bytes.
fn convert(name: &str, input: &Path, output: Option<&Path>) -> Result<Vec<u8>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    command.arg(name).arg(input);
    if let Some(path) = output {
        command.arg("-o").arg(path);
    }
    let result = command.output()?;
    ensure!(
        result.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(result.stdout)
}

#[test]
fn round_trips_mapped_hierarchy_via_files_and_stdout() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gv = dir.path().join("design.gv");
    let pb = dir.path().join("design.pb");
    let restored = dir.path().join("restored.gv");
    std::fs::write(
        &gv,
        r#"`timescale 1ns/1ps module leaf(input [3:0] a, output y); INV gate0(.A(a[2]), .Y(y)); endmodule
module top(input [1:0] a, output y);
  wire [3:0] bus;
  assign bus = {a, 2'H1};
  (* note = "literal: \" *) // text" *) leaf inst(.a(bus), .y(from_leaf));
  (* note = "trailing backslash: \\" *) assign y = from_leaf;
endmodule
"#,
    )?;

    assert!(convert("gv2pb", &gv, Some(&pb))?.is_empty());
    assert_eq!(std::fs::read(&pb)?, convert("gv2pb", &gv, None)?);
    assert!(convert("pb2gv", &pb, Some(&restored))?.is_empty());
    let text = std::fs::read_to_string(&restored)?;
    compare_golden_text(&text, "tests/gv_proto.golden.v");
    assert_eq!(text.as_bytes(), convert("pb2gv", &pb, None)?);
    convert("gv2pb", &restored, Some(&pb))?;
    assert_eq!(text.as_bytes(), convert("pb2gv", &pb, None)?);
    Ok(())
}

#[test]
fn rejects_unrepresentable_gv_before_replacing_output() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gv = dir.path().join("design.gv");
    let pb = dir.path().join("design.pb");
    std::fs::write(&pb, b"previous output")?;
    for statement in [
        "assign y = 4'b10x1;",
        "assign y = 4294967295;",
        "tran bridge(y[0], a[0]);",
        "tran bridges[3:0](y, a);",
        "endmodule /* unclosed block comment\nmodule truncated();",
    ] {
        std::fs::write(
            &gv,
            format!("module top(input [3:0] a, output [3:0] y); {statement} endmodule"),
        )?;
        let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("gv2pb")
            .arg(&gv)
            .arg("--output")
            .arg(&pb)
            .output()?;
        assert!(
            !result.status.success(),
            "unexpectedly accepted {statement}"
        );
        assert!(result.stdout.is_empty());
        assert_eq!(std::fs::read(&pb)?, b"previous output");
    }
    Ok(())
}

#[test]
fn round_trips_octal_digits_at_the_declared_width() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gv = dir.path().join("octal.gv");
    let pb = dir.path().join("octal.pb");
    std::fs::write(
        &gv,
        "module top(output [7:0] y); (* value = 8'o70 *) assign y = 8'O4_70; endmodule",
    )?;
    convert("gv2pb", &gv, Some(&pb))?;
    let text = String::from_utf8(convert("pb2gv", &pb, None)?)?;
    assert_eq!(
        text,
        "module top(y);\n  output [7:0] y;\n  assign y = 8'b00111000;\nendmodule\n"
    );
    Ok(())
}

#[test]
fn rejects_deep_concatenations_with_a_diagnostic() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gv = dir.path().join("deep.gv");
    let source = format!(
        "module top(output y); assign y = {}1'b1{}; endmodule",
        "{".repeat(8192),
        "}".repeat(8192)
    );
    std::fs::write(&gv, source)?;
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("gv2pb")
        .arg(&gv)
        .output()?;
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8(result.stderr)?.contains("48 nested concatenations"));
    Ok(())
}

#[test]
fn verifies_gzip_input_through_the_trailer() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gv = dir.path().join("design.gv.gz");
    let pb = dir.path().join("design.pb");
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(b"module top(input a, output y); assign y = a; endmodule\n")?;
    let bytes = gz.finish()?;
    std::fs::write(&gv, &bytes)?;
    convert("gv2pb", &gv, Some(&pb))?;
    let good_output = std::fs::read(&pb)?;

    std::fs::write(&gv, &bytes[..bytes.len() - 1])?;
    assert!(convert("gv2pb", &gv, Some(&pb)).is_err());
    assert_eq!(std::fs::read(&pb)?, good_output);
    Ok(())
}
