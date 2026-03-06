// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "reference-sim-tests")]

mod reference_sim_iverilog;

use std::io::Write;
use std::process::Command;
use std::time::SystemTime;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_module;

fn vbits(width: u32, signedness: Signedness, msb: &str) -> Value4 {
    assert_eq!(msb.len(), width as usize);
    let mut bits = Vec::with_capacity(width as usize);
    for c in msb.chars().rev() {
        bits.push(match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            'x' | 'X' => LogicBit::X,
            'z' | 'Z' => LogicBit::Z,
            _ => panic!("bad bit char {c}"),
        });
    }
    Value4::new(width, signedness, bits)
}

#[test]
fn module_step_matches_iverilog_simple_counter() {
    reference_sim_iverilog::require_iverilog();

    let src = r#"
module m(input logic clk, input logic en, output logic [3:0] q);
  always_ff @(posedge clk) begin
    if (en) q <= q + 4'd1;
  end
endmodule
"#;

    let cm = compile_module(src).unwrap();
    let mut state = cm.initial_state_x(); // time-zero X requirement

    // Override initial state to a known value so we can compare deterministically.
    state.insert("q".to_string(), vbits(4, Signedness::Unsigned, "0011"));

    let mut inputs = Env::new();
    inputs.insert("en", vbits(1, Signedness::Unsigned, "1"));

    let ours_next = cm.step(&inputs, &state).unwrap();
    let oracle_next = run_module_oracle(src, &inputs, &state, "q");

    assert_eq!(
        ours_next.get("q").unwrap().to_bit_string_msb_first(),
        oracle_next
    );
}

#[test]
fn module_step_part_select_and_concat() {
    reference_sim_iverilog::require_iverilog();

    let src = r#"
module m(input logic clk, input logic [3:0] a, output logic [7:0] q);
  always_ff @(posedge clk) begin
    q[3:0] <= a;
    q[7:4] <= {a[1:0], a[3:2]};
  end
endmodule
"#;
    let cm = compile_module(src).unwrap();
    let mut state = cm.initial_state_x();
    state.insert("q".to_string(), vbits(8, Signedness::Unsigned, "00000000"));

    let mut inputs = Env::new();
    inputs.insert("a", vbits(4, Signedness::Unsigned, "1010"));

    let ours_next = cm.step(&inputs, &state).unwrap();
    let oracle_next = run_module_oracle(src, &inputs, &state, "q");
    assert_eq!(
        ours_next.get("q").unwrap().to_bit_string_msb_first(),
        oracle_next
    );
}

fn run_module_oracle(
    src: &str,
    inputs: &Env,
    state: &xlsynth_vastly::State,
    watch: &str,
) -> String {
    let td = mk_temp_dir();
    let sv_path = td.join("dut.sv");
    let out_path = td.join("a.out");

    let tb = build_tb(src, inputs, state, watch);
    {
        let mut f = std::fs::File::create(&sv_path).expect("create dut.sv");
        f.write_all(tb.as_bytes()).expect("write dut.sv");
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

    let out = String::from_utf8_lossy(&vvp.stdout);
    for line in out.lines() {
        if let Some(x) = line.trim().strip_prefix("Q=") {
            return x.to_string();
        }
    }
    panic!("no Q= line in vvp output:\n{out}\nsource:\n{tb}");
}

fn build_tb(dut_src: &str, inputs: &Env, state: &xlsynth_vastly::State, watch: &str) -> String {
    // We rely on "time-zero X" for regs by *not* initializing q unless provided in
    // `state`. For deterministic tests we do drive initial state explicitly.
    let mut s = String::new();
    s.push_str(dut_src);
    s.push_str("\nmodule tb;\n");
    s.push_str("  logic clk;\n");

    // Declare inputs and state/watch regs with simple logic vectors matching
    // provided widths. v1: assume identifiers in `inputs` and `state` are
    // declared in the DUT module ports/regs.
    for (name, v) in inputs.iter() {
        s.push_str(&format!("  logic [{}:0] {};\n", v.width - 1, name));
    }
    for (name, v) in state.iter() {
        // Do not drive DUT outputs from the testbench; treat these as observed wires.
        s.push_str(&format!("  wire [{}:0] {};\n", v.width - 1, name));
    }

    s.push_str("  m dut(.clk(clk)");
    for (name, _) in inputs.iter() {
        s.push_str(&format!(", .{}({})", name, name));
    }
    for (name, _) in state.iter() {
        // For tests, we assume output reg name matches state reg name (like q).
        s.push_str(&format!(", .{}({})", name, name));
    }
    s.push_str(");\n");

    s.push_str("  initial begin\n");
    s.push_str("    clk = 0;\n");
    for (name, v) in inputs.iter() {
        s.push_str(&format!("    {} = {};\n", name, to_verilog_literal(v)));
    }
    for (name, v) in state.iter() {
        // Override sequential state directly inside the DUT at time 0.
        s.push_str(&format!("    dut.{} = {};\n", name, to_verilog_literal(v)));
    }
    s.push_str("    #1; clk = 1;\n");
    s.push_str("    #1; $display(\"Q=%b\", dut.");
    s.push_str(watch);
    s.push_str(");\n");
    s.push_str("    $finish;\n");
    s.push_str("  end\n");
    s.push_str("endmodule\n");
    s
}

fn to_verilog_literal(v: &Value4) -> String {
    let mut msb = String::with_capacity(v.width as usize);
    for b in v.bits_lsb_first().iter().rev() {
        msb.push(b.as_char());
    }
    format!("{}'b{}", v.width, msb)
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
        let p = base.join(format!("vastly_mod_oracle_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
