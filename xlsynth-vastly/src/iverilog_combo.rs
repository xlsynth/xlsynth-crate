// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::Write;
use std::process::Command;
use std::time::SystemTime;

use crate::Error;
use crate::Result;
use crate::Value4;
use crate::combo_compile::CompiledComboModule;

pub struct IverilogComboRun {
    pub out_vcd_path: std::path::PathBuf,
}

pub fn run_iverilog_combo_and_collect_vcd(
    dut_path: &std::path::Path,
    m: &CompiledComboModule,
    vectors: &[BTreeMap<String, Value4>],
    out_vcd_path: &std::path::Path,
) -> Result<IverilogComboRun> {
    require_iverilog()?;
    if vectors.is_empty() {
        return Err(Error::Parse("no input vectors provided".to_string()));
    }

    let td = mk_temp_dir()?;
    let result = (|| {
        let tb_path = td.join("tb.sv");
        let out_path = td.join("a.out");

        let tb = build_tb(dut_path, m, vectors, out_vcd_path)?;
        {
            let mut f = std::fs::File::create(&tb_path)
                .map_err(|e| Error::Parse(format!("io error: {e}")))?;
            f.write_all(tb.as_bytes())
                .map_err(|e| Error::Parse(format!("io error: {e}")))?;
        }

        let iverilog = Command::new("iverilog")
            .arg("-g2012")
            .arg("-o")
            .arg(&out_path)
            .arg(&tb_path)
            .arg(dut_path)
            .output()
            .map_err(|e| Error::Parse(format!("failed to run iverilog: {e}")))?;
        if !iverilog.status.success() {
            return Err(Error::Parse(format!(
                "iverilog failed\nstdout:\n{}\nstderr:\n{}\ntb:\n{}",
                String::from_utf8_lossy(&iverilog.stdout),
                String::from_utf8_lossy(&iverilog.stderr),
                tb
            )));
        }

        let vvp = Command::new("vvp")
            .arg(&out_path)
            .output()
            .map_err(|e| Error::Parse(format!("failed to run vvp: {e}")))?;
        if !vvp.status.success() {
            return Err(Error::Parse(format!(
                "vvp failed\nstdout:\n{}\nstderr:\n{}\ntb:\n{}",
                String::from_utf8_lossy(&vvp.stdout),
                String::from_utf8_lossy(&vvp.stderr),
                tb
            )));
        }

        Ok(IverilogComboRun {
            out_vcd_path: out_vcd_path.to_path_buf(),
        })
    })();
    let _ = std::fs::remove_dir_all(&td);
    result
}

fn require_iverilog() -> Result<()> {
    let ok = Command::new("iverilog")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        return Err(Error::Parse(
            "iverilog is required but was not found on PATH".to_string(),
        ));
    }
    Ok(())
}

fn build_tb(
    dut_path: &std::path::Path,
    m: &CompiledComboModule,
    vectors: &[BTreeMap<String, Value4>],
    out_vcd_path: &std::path::Path,
) -> Result<String> {
    let mut s = String::new();
    s.push_str("`timescale 1ns/1ns\n");
    s.push_str("module tb;\n");

    for p in &m.input_ports {
        s.push_str(&format!("  logic [{}:0] {};\n", p.width - 1, p.name));
    }
    for p in &m.output_ports {
        s.push_str(&format!("  wire [{}:0] {};\n", p.width - 1, p.name));
    }

    s.push_str(&format!("  {} dut(", m.module_name));
    let mut first = true;
    for p in m.input_ports.iter().chain(m.output_ports.iter()) {
        if !first {
            s.push_str(", ");
        }
        first = false;
        s.push_str(&format!(".{}({})", p.name, p.name));
    }
    s.push_str(");\n");

    s.push_str("  initial begin\n");
    s.push_str(&format!(
        "    $dumpfile(\"{}\");\n    $dumpvars(0, tb);\n",
        out_vcd_path.display()
    ));

    // Initialize vector0 at t=0.
    let v0 = vectors
        .first()
        .ok_or_else(|| Error::Parse("no vectors".to_string()))?;
    for p in &m.input_ports {
        let v = v0
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("missing input `{}`", p.name)))?;
        s.push_str(&format!("    {} = {};\n", p.name, to_verilog_literal(v)));
    }

    for vec_inputs in vectors.iter().skip(1) {
        s.push_str("    #1;\n");
        for p in &m.input_ports {
            let v = vec_inputs
                .get(&p.name)
                .ok_or_else(|| Error::Parse(format!("missing input `{}`", p.name)))?;
            s.push_str(&format!("    {} = {};\n", p.name, to_verilog_literal(v)));
        }
    }

    // Give combinational logic a tick and finish.
    s.push_str("    #1; $finish;\n");
    s.push_str("  end\nendmodule\n");

    // Silence unused warnings for dut_path in string-based tb debugging.
    let _ = dut_path;
    Ok(s)
}

fn to_verilog_literal(v: &Value4) -> String {
    let bits = v.to_bit_string_msb_first();
    format!("{}'b{}", v.width, bits)
}

fn mk_temp_dir() -> Result<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_iverilog_combo_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(Error::Parse(format!("create temp dir: {e:?}"))),
        }
    }
    Err(Error::Parse("failed to create temp dir".to_string()))
}
