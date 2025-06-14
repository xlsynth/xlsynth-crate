// SPDX-License-Identifier: Apache-2.0

//! Helpers for compiling and simulating SystemVerilog/Verilog sources via
//! iverilog when it is present via system installation.
//!
//! These helpers mirror the functionality of `assert_valid_sv` except they
//! build the given sources with `iverilog`, run the resulting simulation via
//! `vvp`, and return the collected VCD waveform contents so tests can make
//! assertions about dynamic behaviour.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Command as StdCommand;

use tempfile::tempdir;

use crate::assert_valid_sv::FlistEntry;

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
