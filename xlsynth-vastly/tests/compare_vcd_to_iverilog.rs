// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "iverilog-tests")]

mod vcd_oracle;

use std::time::SystemTime;

use xlsynth_vastly::Cycle;
use xlsynth_vastly::Stimulus;
use xlsynth_vastly::Vcd;
use xlsynth_vastly::VcdDiffOptions;
use xlsynth_vastly::compile_module;
use xlsynth_vastly::diff_vcd_exact;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
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
fn vcd_matches_iverilog_for_simple_counter() {
    vcd_oracle::require_iverilog();

    let dut = r#"
module m(input logic clk, input logic en, output logic [3:0] q);
  always_ff @(posedge clk) begin
    if (en) q <= q + 4'd1;
  end
endmodule
"#;
    let cm = compile_module(dut).unwrap();

    let stimulus = Stimulus {
        half_period: 5,
        cycles: vec![
            Cycle {
                inputs: [("en".to_string(), vbits(1, Signedness::Unsigned, "1"))]
                    .into_iter()
                    .collect(),
            },
            Cycle {
                inputs: [("en".to_string(), vbits(1, Signedness::Unsigned, "0"))]
                    .into_iter()
                    .collect(),
            },
            Cycle {
                inputs: [("en".to_string(), vbits(1, Signedness::Unsigned, "1"))]
                    .into_iter()
                    .collect(),
            },
        ],
    };

    let mut init = cm.initial_state_x();
    // Deterministic start value.
    init.insert("q".to_string(), vbits(4, Signedness::Unsigned, "0011"));

    let td = mk_temp_dir();
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("iverilog.vcd");

    xlsynth_vastly::run_and_write_vcd(&cm, &stimulus, &init, &ours_vcd).unwrap();
    vcd_oracle::run_iverilog_and_collect_vcd(dut, &stimulus, &init, &iv_vcd);

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
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
        let p = base.join(format!("vastly_vcd_cmp_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
