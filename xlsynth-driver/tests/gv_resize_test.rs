// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use prost::Message;
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_test_helpers::compare_golden_text;

/// Write a two-strength Liberty model and an initially slow combinational gate.
fn write_resize_case(dir: &Path) -> anyhow::Result<()> {
    let raw = dir.join("cells.lib");
    std::fs::write(
        &raw,
        r#"library(test) {
        cell(BUF1) { area: 1; pin(A) { direction: input; capacitance: 1; }
            pin(Y) { direction: output; function: "A";
                timing() { related_pin: "A"; timing_sense: positive_unate;
                    cell_rise(scalar) { values("2"); } cell_fall(scalar) { values("2"); }
                    rise_transition(scalar) { values("1"); } fall_transition(scalar) { values("1"); } } } }
        cell(BUF2) { area: 2; pin(A) { direction: input; capacitance: 2; }
            pin(Y) { direction: output; function: "A";
                timing() { related_pin: "A"; timing_sense: positive_unate;
                    cell_rise(scalar) { values("1"); } cell_fall(scalar) { values("1"); }
                    rise_transition(scalar) { values("1"); } fall_transition(scalar) { values("1"); } } } }
    }"#,
    )?;
    let mut library = parse_liberty_files(&[&raw]).map_err(anyhow::Error::msg)?;
    library.default_output_load = Some(4.0);
    let proto = dir.join("cells.proto");
    std::fs::write(&proto, library_to_proto(library)?.encode_to_vec())?;
    let input = dir.join("input.gv");
    std::fs::write(
        &input,
        "module m(input a, output y); BUF1 b(.A(a), .Y(y)); endmodule",
    )?;
    Ok(())
}

#[test]
fn resizes_with_proto_load_default_and_writes_verified_report() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    write_resize_case(dir.path())?;
    let proto = dir.path().join("cells.proto");
    let input = dir.path().join("input.gv");
    let report = dir.path().join("report.json");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .env_remove("RUST_LOG")
            .arg("gv-resize")
            .arg("--netlist")
            .arg(&input)
            .arg("--liberty_proto")
            .arg(&proto)
            .args(["--max-area-growth", "1", "--report-json"])
            .arg(&report)
            .args(args)
            .output()
    };
    // This combinational fixture never renews a recovery window. The
    // registered finisher's explicit controls must reproduce both goldens.
    for flags in [
        &[][..],
        &[
            "--recovery-cell-menu=electrical-min-area",
            "--recovery-cell-limit=8",
            "--monotone-recovery-checkpoint",
        ],
    ] {
        let output = run(flags)?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        compare_golden_text(
            &String::from_utf8(output.stdout)?,
            "tests/gv_resize.golden.v",
        );
        compare_golden_text(
            &std::fs::read_to_string(&report)?,
            "tests/gv_resize_report.golden.txt",
        );
    }
    let invalid = run(&["--recovery-cell-limit=0"])?;
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(invalid.stderr)?,
        "xlsynth-driver: gv-resize: candidate limits must be positive\n"
    );
    Ok(())
}

// A successful request precedes a request with the wrong Liberty path.
// The first request's output must survive, and the batch must stop.
#[test]
fn batch_preserves_completed_files_and_stops_on_wrong_library() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    write_resize_case(dir.path())?;
    let request = |output: &str, report: &str, library: &str| {
        serde_json::json!({"args":["--netlist","input.gv","--liberty_proto",library,
                            "--max-area-growth","1","--report-json",report],"output":output})
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .env_remove("RUST_LOG")
        .args(["gv-resize-batch", "--liberty-proto", "cells.proto"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let mut input = child.stdin.take().unwrap();
        writeln!(
            input,
            "{}",
            request("first.gv", "first.report", "cells.proto")
        )?;
        writeln!(
            input,
            "{}",
            request("second.gv", "second.report", "cells.proto")
        )?;
        writeln!(
            input,
            "{}",
            request("third.gv", "third.report", "other.proto")
        )?;
    }
    let result = child.wait_with_output()?;
    assert_eq!(result.stdout, b"{}\n{}\n");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr)?,
        "xlsynth-driver: gv-resize-batch: gv-resize-batch request 3: request must use the batch Liberty path cells.proto\n"
    );
    for name in ["first", "second"] {
        compare_golden_text(
            &std::fs::read_to_string(dir.path().join(format!("{name}.gv")))?,
            "tests/gv_resize.golden.v",
        );
        compare_golden_text(
            &std::fs::read_to_string(dir.path().join(format!("{name}.report")))?,
            "tests/gv_resize_report.golden.txt",
        );
    }
    assert!(!dir.path().join("third.gv").exists());
    assert!(!dir.path().join("third.report").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn batch_rolls_back_on_report_collision_or_partial_write() -> anyhow::Result<()> {
    for (report, operation) in [
        ("./first.gv", "reserving new batch output"),
        ("last.json", "writing batch output"),
    ] {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        write_resize_case(dir)?;
        // The netlist fits in one block, the report exceeds it. Ignoring
        // SIGXFSZ lets the worker receive the error from the second write.
        let mut child = Command::new("sh")
            .current_dir(dir)
            .args([
                "-c",
                "trap '' XFSZ && ulimit -f 1 && exec \"$@\"",
                "sh",
                env!("CARGO_BIN_EXE_xlsynth-driver"),
            ])
            .args(["gv-resize-batch", "--liberty-proto", "cells.proto"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let first = serde_json::json!({
            "args": ["--netlist", "input.gv", "--liberty_proto", "cells.proto", "--max-area-growth=1"],
            "output": "first.gv"
        });
        let last = serde_json::json!({
            "args": ["--netlist", "first.gv", "--liberty_proto", "cells.proto", "--report-json", report],
            "output": "last.gv"
        });
        writeln!(child.stdin.take().unwrap(), "{first}\n{last}")?;
        let result = child.wait_with_output()?;
        let error = String::from_utf8(result.stderr)?;
        assert_eq!(result.status.code(), Some(1), "{error}");
        assert_eq!(result.stdout, b"{}\n", "{error}");
        assert!(
            error.contains(&format!("request 2: {operation} '{report}'")),
            "{error}"
        );
        compare_golden_text(
            &std::fs::read_to_string(dir.join("first.gv"))?,
            "tests/gv_resize.golden.v",
        );
        assert!(!dir.join("last.gv").exists());
        assert!(!dir.join("last.json").exists());
    }
    Ok(())
}

// Replace the Library file after the first acknowledgement. A sidecar
// for the replacement must be rejected against the original snapshot.
#[test]
fn batch_rejects_leakage_prices_for_a_replaced_library() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    write_resize_case(root)?;
    let path = root.join("cells.proto");
    let mut child = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .args(["gv-resize-batch", "--liberty-proto", "cells.proto"])
        .current_dir(root)
        .env_remove("RUST_LOG")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut requests = child.stdin.take().unwrap();
    let mut replies = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        requests,
        r#"{{"args":["--netlist","input.gv","--liberty_proto","cells.proto"],"output":"first.gv"}}"#
    )?;
    let mut reply = String::new();
    replies.read_line(&mut reply)?;
    assert_eq!(reply, "{}\n");

    // Install a valid model at the same path after the first reply proves the
    // worker loaded the original. The new model's prices must be rejected.
    let mut replacement = TimingLibrarySnapshot::from_path(&path)?.library;
    replacement.default_output_load = Some(8.0);
    let next = root.join("replacement.proto");
    std::fs::write(&next, library_to_proto(replacement)?.encode_to_vec())?;
    let digest = TimingLibrarySnapshot::from_path(&next)?.sha256;
    std::fs::rename(next, path)?;
    std::fs::write(
        root.join("prices.json"),
        serde_json::to_vec(&serde_json::json!({
               "liberty_proto_sha256": digest, "cell_leakage_nw": {"BUF1": 2.0, "BUF2": 1.0}
        }))?,
    )?;
    writeln!(
        requests,
        "{}",
        serde_json::json!({
            "args":["--netlist","first.gv","--liberty_proto","cells.proto","--upsize-rounds","0",
                    "--downsize-rounds","0","--terminal-leakage-table","prices.json","--report-json","terminal.json"],
            "output":"terminal.gv"
        })
    )?;
    drop(requests);
    reply.clear();
    assert_eq!(replies.read_line(&mut reply)?, 0);
    let result = child.wait_with_output()?;
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr)?,
        "xlsynth-driver: gv-resize-batch: gv-resize-batch request 2: cell leakage table does not match timing library cells.proto\n"
    );
    assert!(!root.join("terminal.gv").exists());
    assert!(!root.join("terminal.json").exists());
    Ok(())
}

#[cfg(not(unix))]
#[test]
fn batch_rejects_unavailable_cpu_accounting_on_startup() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .args([
            "gv-resize-batch",
            "--liberty-proto",
            "unused.proto",
            "--report-process-cpu",
        ])
        .current_dir(dir.path())
        .env_remove("RUST_LOG")
        .output()?;
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        String::from_utf8(result.stderr)?,
        "xlsynth-driver: gv-resize-batch: process CPU accounting requires Unix\n"
    );
    Ok(())
}
