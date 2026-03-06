// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

use std::io::Write;
use std::process::Command;
use std::time::SystemTime;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Value4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleResult {
    pub width: u32,
    pub value_bits_msb: String,
    pub ext_bits_msb: String,
}

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

pub fn run_oracle(expr: &str, env: &Env) -> OracleResult {
    require_iverilog();

    let sv = build_sv(expr, env);
    let td = mk_temp_dir();

    let sv_path = td.join("oracle.sv");
    let out_path = td.join("oracle.out");

    {
        let mut f = std::fs::File::create(&sv_path).expect("create oracle.sv");
        f.write_all(sv.as_bytes()).expect("write oracle.sv");
    }

    let iverilog = Command::new("iverilog")
        .arg("-g2012")
        .arg("-o")
        .arg(&out_path)
        .arg(&sv_path)
        .output()
        .expect("run iverilog");
    if !iverilog.status.success() {
        panic!(
            "iverilog failed\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
            String::from_utf8_lossy(&iverilog.stdout),
            String::from_utf8_lossy(&iverilog.stderr),
            sv
        );
    }

    let vvp = Command::new("vvp")
        .arg(&out_path)
        .output()
        .expect("run vvp");
    if !vvp.status.success() {
        panic!(
            "vvp failed\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
            String::from_utf8_lossy(&vvp.stdout),
            String::from_utf8_lossy(&vvp.stderr),
            sv
        );
    }

    let out = String::from_utf8_lossy(&vvp.stdout);
    parse_oracle_output(&out)
        .unwrap_or_else(|| panic!("failed to parse oracle output:\n{out}\nsource:\n{sv}"))
}

pub fn infer_signedness_from_ext(oracle: &OracleResult) -> Option<bool> {
    // If MSB of the value is known 1 and EXT MSB is known, we can tell:
    // - signed: EXT_MSB == 1
    // - unsigned: EXT_MSB == 0
    let v_msb = oracle.value_bits_msb.chars().next()?;
    let ext_msb = oracle.ext_bits_msb.chars().next()?;
    if v_msb == '1' {
        match ext_msb {
            '1' => Some(true),
            '0' => Some(false),
            _ => None,
        }
    } else {
        None
    }
}

fn parse_oracle_output(out: &str) -> Option<OracleResult> {
    for line in out.lines() {
        let line = line.trim();
        if !line.starts_with("W=") {
            continue;
        }
        let mut width: Option<u32> = None;
        let mut v: Option<String> = None;
        let mut ext: Option<String> = None;
        for part in line.split_whitespace() {
            if let Some(x) = part.strip_prefix("W=") {
                width = x.parse::<u32>().ok();
            } else if let Some(x) = part.strip_prefix("V=") {
                v = Some(x.to_string());
            } else if let Some(x) = part.strip_prefix("EXT=") {
                ext = Some(x.to_string());
            }
        }
        if let (Some(width), Some(v), Some(ext)) = (width, v, ext) {
            return Some(OracleResult {
                width,
                value_bits_msb: v,
                ext_bits_msb: ext,
            });
        }
    }
    None
}

fn build_sv(expr: &str, env: &Env) -> String {
    let mut s = String::new();
    s.push_str("module oracle;\n");
    for (name, v) in env.iter() {
        let decl = match v.signedness {
            xlsynth_vastly::Signedness::Signed => format!(
                "  localparam logic signed [{}:0] {} = {};\n",
                v.width - 1,
                name,
                to_verilog_literal(v)
            ),
            xlsynth_vastly::Signedness::Unsigned => format!(
                "  localparam logic [{}:0] {} = {};\n",
                v.width - 1,
                name,
                to_verilog_literal(v)
            ),
        };
        s.push_str(&decl);
    }

    s.push_str(&format!("  localparam int W = $bits({expr});\n"));
    s.push_str("  logic [W-1:0] V;\n");
    s.push_str("  logic [W:0] EXT;\n");
    s.push_str(&format!("  assign V = {expr};\n"));
    s.push_str(&format!("  assign EXT = {expr};\n"));
    s.push_str("  initial begin\n");
    s.push_str("    $display(\"W=%0d V=%b EXT=%b\", W, V, EXT);\n");
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
        let p = base.join(format!("vastly_oracle_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp oracle dir: {e:?}"),
        }
    }
    panic!("failed to create unique temp oracle dir after many attempts");
}

#[allow(dead_code)]
fn _debug_bits(v: &Value4) -> Vec<LogicBit> {
    v.bits_lsb_first().to_vec()
}
