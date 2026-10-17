// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use prost::Message;
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_test_helpers::compare_golden_text;

#[test]
fn resizes_with_proto_load_default_and_writes_verified_report() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let raw = dir.path().join("cells.lib");
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
    let proto = dir.path().join("cells.proto");
    std::fs::write(&proto, library_to_proto(library)?.encode_to_vec())?;
    let input = dir.path().join("input.gv");
    std::fs::write(
        &input,
        "module m(input a, output y); BUF1 b(.A(a), .Y(y)); endmodule",
    )?;
    let report = dir.path().join("report.json");
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("gv-resize")
        .arg("--netlist")
        .arg(&input)
        .arg("--liberty_proto")
        .arg(&proto)
        .args(["--max-area-growth", "1", "--report-json"])
        .arg(&report)
        .output()?;
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
    Ok(())
}
