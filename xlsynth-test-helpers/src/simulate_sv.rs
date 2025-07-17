// SPDX-License-Identifier: Apache-2.0

//! Helpers for compiling and simulating SystemVerilog/Verilog sources via
//! iverilog when it is present via system installation.
//!
//! These helpers mirror the functionality of `assert_valid_sv` except they
//! build the given sources with `iverilog`, run the resulting simulation via
//! `vvp`, and return the collected VCD waveform contents so tests can make
//! assertions about dynamic behaviour.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Command as StdCommand;

use tempfile::tempdir;

use crate::assert_valid_sv::FlistEntry;
use xlsynth::IrBits;

/// Error type returned by [`simulate_sv_flist`] when the simulation cannot be
/// performed.
#[derive(Debug)]
pub enum SimulateSvError {
    /// Icarus Verilog (`iverilog` + `vvp`) is not available in the caller's
    /// `PATH`.
    IverilogUnavailable,
    /// `iverilog` returned a non-zero exit code.
    CompileFailed {
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
    /// `vvp` returned a non-zero exit code.
    SimulationFailed {
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
    /// Generic I/O error (e.g. writing sources or reading the VCD).
    Io(std::io::Error),
}

impl PartialEq for SimulateSvError {
    fn eq(&self, other: &Self) -> bool {
        use SimulateSvError::*;
        match (self, other) {
            (IverilogUnavailable, IverilogUnavailable) => true,
            (
                CompileFailed {
                    status: s1,
                    stdout: o1,
                    stderr: e1,
                },
                CompileFailed {
                    status: s2,
                    stdout: o2,
                    stderr: e2,
                },
            ) => s1 == s2 && o1 == o2 && e1 == e2,
            (
                SimulationFailed {
                    status: s1,
                    stdout: o1,
                    stderr: e1,
                },
                SimulationFailed {
                    status: s2,
                    stdout: o2,
                    stderr: e2,
                },
            ) => s1 == s2 && o1 == o2 && e1 == e2,
            // We intentionally do not attempt to compare `Io` variants
            // because `std::io::Error` does not implement `PartialEq`.
            (Io(_), Io(_)) => false,
            _ => false,
        }
    }
}

impl Eq for SimulateSvError {}

impl std::fmt::Display for SimulateSvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimulateSvError::IverilogUnavailable => {
                write!(f, "Icarus Verilog (iverilog) not found in PATH")
            }
            SimulateSvError::CompileFailed { status, .. } => {
                write!(f, "iverilog failed with status {:?}", status)
            }
            SimulateSvError::SimulationFailed { status, .. } => {
                write!(f, "vvp failed with status {:?}", status)
            }
            SimulateSvError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for SimulateSvError {}

impl From<std::io::Error> for SimulateSvError {
    fn from(e: std::io::Error) -> Self {
        SimulateSvError::Io(e)
    }
}

/// Attempts to locate the `iverilog` binary using the caller's `PATH`.
///
/// The function executes `which iverilog` and returns the absolute path if the
/// tool is found. If `iverilog` is not available in `PATH` the function returns
/// `None`.
fn find_iverilog() -> Option<PathBuf> {
    // Probe the user's PATH for an `iverilog` executable without introducing
    // any additional dependencies.
    if let Ok(output) = StdCommand::new("which").arg("iverilog").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }
    None
}

/// Runs `iverilog` with the given sources and returns the path to the produced
/// `*.vvp` executable inside `work_dir`.
fn compile_with_iverilog(
    work_dir: &Path,
    sources: &[PathBuf],
    top_module: &str,
) -> Result<PathBuf, SimulateSvError> {
    let iverilog_bin =
        find_iverilog().expect("iverilog binary not found; ensure it is available in PATH");
    let out_path = work_dir.join("sim.vvp");
    let mut cmd = Command::new(&iverilog_bin);
    cmd.current_dir(work_dir)
        .arg("-g2012") // Enable SystemVerilog-2012 features – fine for pure Verilog too.
        .arg("-o")
        .arg(&out_path)
        .arg("-s")
        .arg(top_module);
    for src in sources {
        cmd.arg(src);
    }

    // Log the command we are about to run.
    log::info!("Running: {:?}", cmd);

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(SimulateSvError::CompileFailed {
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into(),
            stderr: String::from_utf8_lossy(&output.stderr).into(),
        });
    }
    log::info!("iverilog finished OK, output {:?}", out_path);
    Ok(out_path)
}

/// Executes the given `*.vvp` simulation and waits for completion.
fn run_vvp(vvp_path: &Path, work_dir: &Path) -> Result<(), SimulateSvError> {
    // Rely on PATH lookup for `vvp` instead of an environment variable.
    let mut cmd = Command::new("vvp");
    cmd.current_dir(work_dir).arg(vvp_path);

    log::info!("Running: {:?}", cmd);

    let output = cmd.current_dir(work_dir).output()?;

    if !output.status.success() {
        return Err(SimulateSvError::SimulationFailed {
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into(),
            stderr: String::from_utf8_lossy(&output.stderr).into(),
        });
    }
    log::info!("vvp simulation completed successfully");
    Ok(())
}

/// Compiles and simulates the given SystemVerilog/Verilog sources and returns
/// the contents of the VCD file produced by the simulation.
///
/// * `files` – List of filename/contents pairs that make up the design +
///   optional test bench. The filenames are recreated verbatim inside a fresh
///   temporary directory so relative includes work as expected.
/// * `top_module` – Name of the module that should be treated as simulation top
///   (passed via `iverilog -s`). In most setups this is the name of the test
///   bench.
/// * `vcd_name` – Path (relative to the temporary work dir) of the VCD file the
///   test bench writes to, typically something like `dump.vcd`.
///
/// If Icarus is not available the function logs a warning and returns an empty
/// string so tests can gracefully skip.
pub fn simulate_sv_flist(
    files: &[FlistEntry],
    top_module: &str,
    vcd_name: &str,
) -> Result<String, SimulateSvError> {
    // Write all sources to a temporary work directory.
    let temp_dir = tempdir().expect("create temp dir");

    // Pre-compute the VCD path (handy for the log below and later reading).
    let vcd_path = temp_dir.path().join(vcd_name);

    // Emit the location up-front so it is visible even if an early error
    // occurs – still prefer logging over stdout.
    log::info!("VCD dump will be written to: {}", vcd_path.display());

    // Write all design and testbench sources into the temporary directory.
    let mut src_paths = vec![];
    for entry in files {
        let path = temp_dir.path().join(&entry.filename);
        std::fs::write(&path, &entry.contents)?;
        log::info!("wrote {} to {}", entry.filename, path.display());
        // Emit an easy-to-spot log for the DUT file, conventionally named
        // `dut.sv` in most tests.
        if entry.filename == "dut.sv" {
            log::info!("DUT source written to: {}", path.display());
        }
        src_paths.push(path);
    }

    // Compile.
    let vvp_path = compile_with_iverilog(temp_dir.path(), &src_paths, top_module)?;

    // Run simulation.
    run_vvp(&vvp_path, temp_dir.path())?;

    // Read and return the VCD contents.
    let vcd = std::fs::read_to_string(&vcd_path).map_err(SimulateSvError::Io)?;

    Ok(vcd)
}

/// Simulates a pipeline design with parameterized signal names.
///
/// * `input_valid_signal` / `output_valid_signal` – names of the handshake
///   pins.
/// * `reset_signal` – reset pin name.
/// * `reset_active_low` – if true, reset asserts when `0`.
pub fn simulate_pipeline_single_pulse_custom(
    pipeline_sv: &str,
    module_name: &str,
    inputs: &[(&str, IrBits)],
    expected_output: &IrBits,
    latency: usize,
    input_valid_signal: &str,
    output_valid_signal: Option<&str>,
    reset_signal: &str,
    reset_active_low: bool,
) -> Result<String, SimulateSvError> {
    use xlsynth::ir_value::IrFormatPreference;

    let mut reg_decls = String::new();
    let mut port_conns = Vec::new();
    let mut assign_values = String::new();

    for (name, value) in inputs {
        let width = value.get_bit_count() - 1;
        let hex = value
            .to_string_fmt(IrFormatPreference::Hex, false)
            .trim_start_matches("0x")
            .to_string();
        reg_decls.push_str(&format!("  reg [{width}:0] {name} = 0;\n"));
        port_conns.push(format!(".{name}({name})"));
        assign_values.push_str(&format!(
            "    {name} = {width_plus_one}'h{hex};\n",
            width_plus_one = width + 1
        ));
    }

    let ports = port_conns.join(", ");

    let out_width_minus_one = expected_output.get_bit_count() - 1;
    let out_width_minus_one_plus_one = out_width_minus_one + 1;
    let exp_hex = expected_output
        .to_string_fmt(IrFormatPreference::Hex, false)
        .trim_start_matches("0x")
        .to_string();

    // Handshake and reset regs are declared separately in the TB header below to
    // avoid duplicates.

    // Reset literal values depending on polarity.
    let initial_reset_val = if reset_active_low { "0" } else { "1" };
    let deassert_reset_val = if reset_active_low { "1" } else { "0" };

    // Optional user data ports segment (prefixed with a comma when non-empty).
    let user_ports_part = if ports.is_empty() {
        String::new()
    } else {
        format!(", {ports}")
    };

    // Build optional output-valid port snippet.
    let (output_valid_decl, output_valid_port, output_valid_asserts) = if let Some(ov) =
        output_valid_signal
    {
        (
            format!("  wire {ov};\n"),
            format!(", .{ov}({ov})"),
            format!(
                "    if ({ov} !== 1'b1) $fatal(1, \"{ov} not asserted\");\n    @(posedge clk);\n    #1;\n    if ({ov} !== 1'b0) $fatal(1, \"{ov} did not deassert\");\n"
            ),
        )
    } else {
        (String::new(), String::new(), String::new())
    };

    let tb = format!(
        r#"`timescale 1ns/1ps
module tb;
  reg clk = 0;
  always #5 clk = ~clk;
  reg {reset_signal} = {initial_reset_val};
  reg {input_valid_signal} = 0;
{reg_decls}{output_valid_decl}  wire [{out_width_minus_one}:0] out;
  {module_name} dut(.clk(clk), .{reset_signal}({reset_signal}), .{input_valid_signal}({input_valid_signal}){user_ports_part}{output_valid_port}, .out(out));
  integer i;
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, tb);
    {reset_signal} = {initial_reset_val};
    for (i = 0; i < 2; i = i + 1) @(posedge clk);
    {reset_signal} = {deassert_reset_val};
    @(posedge clk);
{assign_values}    {input_valid_signal} = 1;
    @(posedge clk);
    {input_valid_signal} = 0;
    for (i = 0; i < {latency}; i = i + 1) @(posedge clk);
    #1;
{output_valid_asserts}    if (out !== {out_width_minus_one_plus_one}'h{exp_hex}) $fatal(1, "unexpected output");
    @(posedge clk);
    #1;
    $finish;
  end
endmodule"#
    );

    let files = vec![
        FlistEntry {
            filename: "dut.sv".into(),
            contents: pipeline_sv.to_string(),
        },
        FlistEntry {
            filename: "tb.sv".into(),
            contents: tb,
        },
    ];

    simulate_sv_flist(&files, "tb", "dump.vcd")
}

/// Backward-compat wrapper that uses default signal names.
pub fn simulate_pipeline_single_pulse(
    pipeline_sv: &str,
    module_name: &str,
    inputs: &[(&str, IrBits)],
    expected_output: &IrBits,
    latency: usize,
) -> Result<String, SimulateSvError> {
    simulate_pipeline_single_pulse_custom(
        pipeline_sv,
        module_name,
        inputs,
        expected_output,
        latency,
        "input_valid",
        Some("output_valid"),
        "rst",
        true,
    )
}

/// Parses the given VCD text and returns the first timestamp and value of
/// `data_sig` at the moment `valid_sig` is asserted (changes to 1'b1).
/// The value is returned as a Vec<bool> with LSB at index 0.
pub fn capture_value_on_valid(
    vcd_text: &str,
    valid_sig: &str,
    data_sig: &str,
) -> Option<(u64, xlsynth::IrBits)> {
    // First pass: build mapping from signal name to id code (one or more chars)
    let mut code_for_name: HashMap<String, String> = HashMap::new();
    for line in vcd_text.lines() {
        if line.starts_with("$var ") {
            // Example: $var wire 1 ! output_valid $end
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                let code = parts[3].to_string();
                let name = parts[4].to_string();
                code_for_name.insert(name, code);
            }
        }
        if line.starts_with("$enddefinitions") {
            break;
        }
    }

    let valid_code = code_for_name.get(valid_sig)?.clone();
    let data_code = code_for_name.get(data_sig)?.clone();

    let mut cur_time: u64 = 0;
    let mut cur_data: Vec<bool> = Vec::new();
    let mut valid_high = false;

    for line in vcd_text.lines() {
        if line.starts_with('#') {
            // timestamp
            if let Ok(t) = line[1..].trim().parse::<u64>() {
                cur_time = t;
                if valid_high {
                    // Convert bits vec (LSB first) into IrBits.
                    let bit_count = cur_data.len();
                    let mut value: u64 = 0;
                    for (idx, bit) in cur_data.iter().enumerate().take(64) {
                        if *bit {
                            value |= 1u64 << idx;
                        }
                    }
                    let irb = if bit_count <= 64 {
                        xlsynth::IrBits::make_ubits(bit_count, value).ok()?
                    } else {
                        // Fallback: build literal string.
                        let bin: String = cur_data
                            .iter()
                            .rev()
                            .map(|b| if *b { '1' } else { '0' })
                            .collect();
                        let typed = format!("bits[{}]:0b{}", bit_count, bin);
                        xlsynth::IrValue::parse_typed(&typed).ok()?.to_bits().ok()?
                    };
                    return Some((cur_time, irb));
                }
            }
        } else if line.starts_with('b') {
            // vector change: b1010 <code>
            let mut parts = line[1..].split_whitespace();
            if let (Some(bits_str), Some(code)) = (parts.next(), parts.next()) {
                if code == data_code {
                    cur_data = bits_str
                        .chars()
                        .rev() // LSB first
                        .map(|c| c == '1')
                        .collect();
                }
            }
        } else {
            // scalar change: 1<code> or 0<code>
            if line.ends_with(&valid_code) {
                let val_char = line.chars().next()?;
                valid_high = val_char == '1';
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_simple_adder_tb() {
        let _ = env_logger::builder().is_test(true).try_init();

        // Design under test + simple test-bench that writes a VCD named
        // `dump.vcd`.
        let files = vec![
            FlistEntry {
                filename: "adder.sv".into(),
                contents:
                    "module adder(input [7:0] a, b, output [7:0] y); assign y = a + b; endmodule"
                        .into(),
            },
            FlistEntry {
                filename: "tb.sv".into(),
                contents: r#"`timescale 1ns/1ps
module tb;
  reg clk = 0;
  always #5 clk = ~clk;
  reg  [7:0] a, b;
  wire [7:0] y;
  adder dut(.a(a), .b(b), .y(y));
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, tb);
    a = 8'h3; b = 8'h5;
    #10;
    a = 8'ha; b = 8'h1;
    #10;
    $finish;
  end
endmodule"#
                    .into(),
            },
        ];

        let vcd = simulate_sv_flist(&files, "tb", "dump.vcd").expect("simulation succeeds");
        assert!(vcd.contains("$var"), "expect VCD wave dump contents");
    }
}
