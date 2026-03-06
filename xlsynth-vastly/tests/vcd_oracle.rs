// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

use std::io::Write;
use std::process::Command;
use std::time::SystemTime;

use xlsynth_vastly::Stimulus;
use xlsynth_vastly::Value4;

pub fn require_iverilog() {
    let ok = Command::new("iverilog")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        panic!("iverilog is required for these tests but was not found on PATH");
    }
}

pub fn run_iverilog_and_collect_vcd(
    dut_src: &str,
    stimulus: &Stimulus,
    initial_state: &std::collections::BTreeMap<String, Value4>,
    out_vcd_path: &std::path::Path,
) {
    require_iverilog();
    let td = mk_temp_dir();
    let sv_path = td.join("tb.sv");
    let out_path = td.join("a.out");

    let tb = build_tb(dut_src, stimulus, initial_state, out_vcd_path);
    {
        let mut f = std::fs::File::create(&sv_path).expect("create tb.sv");
        f.write_all(tb.as_bytes()).expect("write tb.sv");
    }

    let iverilog = Command::new("iverilog")
        .arg("-g2012")
        .arg("-o")
        .arg(&out_path)
        .arg(&sv_path)
        .output()
        .expect("run iverilog");
    assert!(
        iverilog.status.success(),
        "iverilog failed\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&iverilog.stdout),
        String::from_utf8_lossy(&iverilog.stderr),
        tb
    );

    let vvp = Command::new("vvp")
        .arg(&out_path)
        .output()
        .expect("run vvp");
    assert!(
        vvp.status.success(),
        "vvp failed\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&vvp.stdout),
        String::from_utf8_lossy(&vvp.stderr),
        tb
    );
}

fn build_tb(
    dut_src: &str,
    stimulus: &Stimulus,
    initial_state: &std::collections::BTreeMap<String, Value4>,
    out_vcd_path: &std::path::Path,
) -> String {
    let mut s = String::new();
    // Ensure iverilog produces a VCD timescale compatible with our writer.
    // Force VCD timescale unit to remain in ns for exact diffing.
    s.push_str("`timescale 1ns/1ns\n");
    s.push_str(dut_src);
    s.push_str("\nmodule tb;\n");
    s.push_str("  logic clk;\n");

    // Declare all stimulus input signals (excluding clk) + state outputs as wires.
    let mut input_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for cyc in &stimulus.cycles {
        for name in cyc.inputs.keys() {
            input_names.insert(name.clone());
        }
    }
    for name in &input_names {
        // v1: assume 1-bit inputs unless overridden by cycle values; pick max width
        // seen.
        let mut w: u32 = 1;
        for cyc in &stimulus.cycles {
            if let Some(v) = cyc.inputs.get(name) {
                w = w.max(v.width);
            }
        }
        s.push_str(&format!("  logic [{}:0] {};\n", w - 1, name));
    }
    for (name, v) in initial_state {
        s.push_str(&format!("  wire [{}:0] {};\n", v.width - 1, name));
    }

    s.push_str("  m dut(.clk(clk)");
    for name in &input_names {
        s.push_str(&format!(", .{}({})", name, name));
    }
    for name in initial_state.keys() {
        s.push_str(&format!(", .{}({})", name, name));
    }
    s.push_str(");\n");

    s.push_str(&format!(
        "  initial begin\n    $dumpfile(\"{}\");\n    $dumpvars(0, tb);\n",
        out_vcd_path.display()
    ));
    s.push_str("    clk = 0;\n");

    // Initialize inputs at t=0 to their cycle0 values if provided, else X.
    if let Some(c0) = stimulus.cycles.first() {
        for (name, v) in &c0.inputs {
            s.push_str(&format!("    {} = {};\n", name, to_verilog_literal(v)));
        }
    }
    // Initialize sequential state in dut at t=0 (for deterministic comparisons).
    for (name, v) in initial_state {
        s.push_str(&format!("    dut.{} = {};\n", name, to_verilog_literal(v)));
    }

    // Drive cycles: inputs at +1, posedge at +half_period, negedge at
    // +2*half_period.
    for (i, cyc) in stimulus.cycles.iter().enumerate() {
        let base = (i as u64) * stimulus.half_period * 2;
        let input_t = base + 1;
        s.push_str(&format!(
            "    #{};\n",
            if i == 0 { 1u64 } else { input_t - (base) }
        ));
        // If i>0, we are already at base from last negedge, so delay is 1.
        // Apply inputs:
        for (name, v) in &cyc.inputs {
            s.push_str(&format!("    {} = {};\n", name, to_verilog_literal(v)));
        }
        // Advance to posedge at half_period.
        let to_pos = stimulus.half_period.saturating_sub(1);
        s.push_str(&format!("    #{}; clk = 1;\n", to_pos));
        // Advance to negedge at next half period.
        s.push_str(&format!("    #{}; clk = 0;\n", stimulus.half_period));
    }
    s.push_str("    #1; $finish;\n  end\nendmodule\n");
    s
}

fn to_verilog_literal(v: &Value4) -> String {
    let bits = v.to_bit_string_msb_first();
    format!("{}'b{}", v.width, bits)
}

fn mk_temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_vcd_oracle_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
