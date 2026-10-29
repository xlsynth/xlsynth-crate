// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::fmt::Write;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::read_gv_for_roundtrip_from_path;
use xlsynth_g8r::netlist::proto::{decode_netlist, encode_netlist};
use xlsynth_test_helpers::iverilog::required_iverilog_toolchain;

/// Generates GV text directly so the original never passes through the emitter.
fn mapped_gv(data: &[u8]) -> (String, usize) {
    let (&layout, gates) = data.split_first().unwrap_or((&0, &[]));
    let gates = gates.chunks_exact(3).take(16);
    let width = gates.len() + 4;
    let mut gv = String::new();
    if layout & 1 != 0 {
        gv.push_str("`timescale 1ns/1ps ");
    }
    if layout & 2 == 0 {
        writeln!(gv, "module dut(input [3:0] a, output [{}:0] y);", width - 1).unwrap();
    } else {
        writeln!(
            gv,
            "module dut(a, y); output [{}:0] y; input [3:0] a;",
            width - 1
        )
        .unwrap();
    }
    writeln!(gv, "  assign y[3:0] = {{a[1:0], a[3:2]}};").unwrap();
    let mut sources: Vec<String> = (0..4).map(|bit| format!("a[{bit}]")).collect();
    sources.extend(["1'b0", "1'o1", "1'hx"].map(str::to_string));
    for (i, gate) in gates.enumerate() {
        let a = &sources[usize::from(gate[1]) % sources.len()];
        let b = &sources[usize::from(gate[2]) % sources.len()];
        let cell = ["AND2", "XOR2", "ANDNOT2", r"\tran "][usize::from(gate[0] & 3)];
        if gate[0] & 4 != 0 {
            writeln!(gv, "  wire n{i};").unwrap();
        }
        if gate[0] & 8 == 0 {
            writeln!(gv, "  {cell} u{i} (.A({a}), .B({b}), .Y(n{i}));").unwrap();
        } else {
            writeln!(gv, "  {cell} u{i} (.Y(n{i}), .B({b}), .A({a}));").unwrap();
        }
        writeln!(gv, "  assign y[{}] = n{i};", i + 4).unwrap();
        sources.push(format!("n{i}"));
    }
    gv.push_str("endmodule\n");
    (gv, width)
}

/// Elaborates both designs with independent cell models and compares every
/// observable output, including X/Z, for each binary input combination.
fn check(source: &str, width: usize) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("mapped.v");
    std::fs::write(&path, source)?;
    let parsed = read_gv_for_roundtrip_from_path(&path)?;
    let restored = decode_netlist(&encode_netlist(&parsed)?)?;
    let restored_gv = restored
        .modules
        .iter()
        .map(|module| emit_module_as_netlist_text(module, &restored.nets, &restored.interner))
        .collect::<Result<String>>()?;
    let original = source.replacen("module dut", "module original", 1);
    let bench = format!(
        r#"
module AND2(input A, B, output Y); assign Y = A & B; endmodule
module XOR2(input A, B, output Y); assign Y = A ^ B; endmodule
module ANDNOT2(input A, B, output Y); assign Y = A & ~B; endmodule
module \tran (input A, B, output Y); assign Y = A | B; endmodule
module bench;
  reg [3:0] a;
  wire [{msb}:0] expected, observed;
  original gold(a, expected);
  dut restored(a, observed);
  integer i;
  initial for (i = 0; i < 16; i = i + 1) begin
    a = i;
    #1;
    if (expected !== observed)
      $fatal(1, "a=%b original=%b restored=%b", a, expected, observed);
  end
endmodule
"#,
        msb = width - 1
    );
    std::fs::write(
        dir.path().join("miter.v"),
        format!("{original}\n{restored_gv}\n{bench}"),
    )?;
    let tools = required_iverilog_toolchain().map_err(|e| anyhow!(e))?;
    let timeout = Duration::from_secs(10);
    let result = tools
        .compile(dir.path(), &["miter.v"], "bench", &[], timeout)
        .and_then(|executable| tools.run(dir.path(), &executable, timeout));
    result.with_context(|| format!("restored GV:\n{restored_gv}"))?;
    Ok(())
}

fuzz_target!(init: {
    required_iverilog_toolchain().expect("mapped netlist round trips require iverilog and vvp");
}, |data: &[u8]| {
    let (source, width) = mapped_gv(data);
    check(&source, width).unwrap_or_else(|e| panic!("original GV:\n{source}\n{e:#}"));
});
