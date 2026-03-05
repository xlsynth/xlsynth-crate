// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "iverilog-tests")]

use std::collections::BTreeMap;
use std::io::Write;
use std::process::Command;
use std::time::SystemTime;

use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Vcd;
use xlsynth_vastly::VcdDiffOptions;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::diff_vcd_exact;
use xlsynth_vastly::eval_combo_seeded;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::run_pipeline_and_write_vcd;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::State;
use xlsynth_vastly::Value4;

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
fn vcd_matches_iverilog_for_pipeline_with_combo_outputs() {
    require_iverilog();

    let dut = r#"
module m(
  input logic clk,
  input logic [3:0] a,
  output logic [3:0] q,
  output wire [3:0] out
);
  wire [3:0] t;
  assign t = q ^ a;
  assign out = t + 4'd1;
  always_ff @(posedge clk) begin
    q <= q + 4'd1;
  end
endmodule
"#;

    let cm = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [("a".to_string(), vbits(4, Signedness::Unsigned, "0001"))]
                    .into_iter()
                    .collect(),
            },
            PipelineCycle {
                inputs: [("a".to_string(), vbits(4, Signedness::Unsigned, "0010"))]
                    .into_iter()
                    .collect(),
            },
            PipelineCycle {
                inputs: [("a".to_string(), vbits(4, Signedness::Unsigned, "0011"))]
                    .into_iter()
                    .collect(),
            },
        ],
    };

    let mut init: State = BTreeMap::new();
    init.insert("q".to_string(), vbits(4, Signedness::Unsigned, "0101"));

    // Quick invariant check: the combo path can compute `out` and `t` from seeded
    // env.
    let plan = plan_combo_eval(&cm.combo).unwrap();
    let mut seed = xlsynth_vastly::Env::new();
    seed.insert("q".to_string(), init.get("q").unwrap().clone());
    seed.insert(
        "a".to_string(),
        stimulus.cycles[0].inputs.get("a").unwrap().clone(),
    );
    seed.insert("clk".to_string(), vbits(1, Signedness::Unsigned, "0"));
    let vals = eval_combo_seeded(&cm.combo, &plan, &seed).unwrap();
    assert_eq!(vals.get("t").unwrap().to_bit_string_msb_first(), "0100");
    assert_eq!(vals.get("out").unwrap().to_bit_string_msb_first(), "0101");

    let td = mk_temp_dir();
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("iverilog.vcd");

    run_pipeline_and_write_vcd(&cm, &stimulus, &init, &ours_vcd).unwrap();
    run_iverilog_pipeline_and_collect_vcd(dut, &cm, &stimulus, &init, &iv_vcd);

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
}

#[test]
fn vcd_matches_iverilog_for_pipeline_with_two_stateful_stages() {
    require_iverilog();

    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic input_valid,
  input logic [7:0] x,
  output logic output_valid,
  output logic [7:0] out
);
  logic s0_valid;
  logic [7:0] s0_x;
  logic s1_valid;
  logic [7:0] s1_out;

  always_ff @(posedge clk) begin
    if (rst) begin
      s0_valid <= 1'b0;
      s0_x <= 8'h00;
    end else begin
      s0_valid <= input_valid;
      s0_x <= input_valid ? x : s0_x;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      s1_valid <= 1'b0;
      s1_out <= 8'h00;
    end else begin
      s1_valid <= s0_valid;
      s1_out <= s0_valid ? (s0_x + 8'h01) : s1_out;
    end
  end

  assign output_valid = s1_valid;
  assign out = s1_out;
endmodule
"#;

    let cm = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "0"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "0"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "1"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000001")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "1"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000010")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "1"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000011")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "0"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "input_valid".to_string(),
                        vbits(1, Signedness::Unsigned, "0"),
                    ),
                    ("x".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::from([
        ("s0_valid".to_string(), vbits(1, Signedness::Unsigned, "0")),
        (
            "s0_x".to_string(),
            vbits(8, Signedness::Unsigned, "00000000"),
        ),
        ("s1_valid".to_string(), vbits(1, Signedness::Unsigned, "0")),
        (
            "s1_out".to_string(),
            vbits(8, Signedness::Unsigned, "00000000"),
        ),
    ]);

    let td = mk_temp_dir();
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("iverilog.vcd");

    run_pipeline_and_write_vcd(&cm, &stimulus, &init, &ours_vcd).unwrap();
    run_iverilog_pipeline_and_collect_vcd(dut, &cm, &stimulus, &init, &iv_vcd);

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
}

fn require_iverilog() {
    let ok = Command::new("iverilog")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        panic!("iverilog is required for these tests but was not found on PATH");
    }
}

fn run_iverilog_pipeline_and_collect_vcd(
    dut_src: &str,
    cm: &xlsynth_vastly::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    initial_state: &State,
    out_vcd_path: &std::path::Path,
) {
    let td = mk_temp_dir();
    let sv_path = td.join("tb.sv");
    let out_path = td.join("a.out");

    let tb = build_tb(dut_src, cm, stimulus, initial_state, out_vcd_path);
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
    cm: &xlsynth_vastly::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    initial_state: &State,
    out_vcd_path: &std::path::Path,
) -> String {
    assert!(!stimulus.cycles.is_empty());
    let mut s = String::new();
    s.push_str("`timescale 1ns/1ns\n");
    s.push_str(dut_src);
    s.push_str("\nmodule tb;\n");
    s.push_str("  logic clk;\n");

    for p in &cm.combo.input_ports {
        s.push_str(&format!("  logic [{}:0] {};\n", p.width - 1, p.name));
    }
    for p in &cm.combo.output_ports {
        s.push_str(&format!("  wire [{}:0] {};\n", p.width - 1, p.name));
    }

    s.push_str("  m dut(.clk(clk)");
    for p in &cm.combo.input_ports {
        s.push_str(&format!(", .{}({})", p.name, p.name));
    }
    for p in &cm.combo.output_ports {
        s.push_str(&format!(", .{}({})", p.name, p.name));
    }
    s.push_str(");\n");

    s.push_str(&format!(
        "  initial begin\n    $dumpfile(\"{}\");\n    $dumpvars(0, tb);\n",
        out_vcd_path.display()
    ));
    s.push_str("    clk = 0;\n");

    // Cycle 0 inputs at t=0.
    let c0 = stimulus.cycles.first().unwrap();
    for p in &cm.combo.input_ports {
        let v = c0.inputs.get(&p.name).expect("provide all inputs");
        s.push_str(&format!("    {} = {};\n", p.name, to_verilog_literal(v)));
    }

    // Initialize sequential state in dut at t=0 (for deterministic comparisons).
    for (name, v) in initial_state {
        s.push_str(&format!("    dut.{} = {};\n", name, to_verilog_literal(v)));
    }

    for (i, cyc) in stimulus.cycles.iter().enumerate() {
        if i == 0 {
            s.push_str(&format!("    #{}; clk = 1;\n", stimulus.half_period));
        } else {
            s.push_str("    #1;\n");
            for p in &cm.combo.input_ports {
                let v = cyc.inputs.get(&p.name).expect("provide all inputs");
                s.push_str(&format!("    {} = {};\n", p.name, to_verilog_literal(v)));
            }
            let to_pos = stimulus.half_period.saturating_sub(1);
            s.push_str(&format!("    #{}; clk = 1;\n", to_pos));
        }
        s.push_str(&format!("    #{}; clk = 0;\n", stimulus.half_period));
    }
    s.push_str("    #1; $finish;\n  end\nendmodule\n");
    s
}

fn to_verilog_literal(v: &Value4) -> String {
    format!("{}'b{}", v.width, v.to_bit_string_msb_first())
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
        let p = base.join(format!("vastly_pipeline_vcd_cmp_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
