// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use regex::Regex;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;
use xlsynth::ir_value::{IrFormatPreference, IrValue};

/// Describes a port found in the module header.
#[derive(Debug, Clone)]
struct PortInfo {
    name: String,
    width: usize, // at least 1
    is_input: bool,
}

/// Parse the top-level module name (the *last* module declaration in the file)
/// together with its port list.
///
/// Returns: (module_name, Vec<PortInfo>)
fn parse_verilog_top_module(verilog: &str) -> anyhow::Result<(String, Vec<PortInfo>)> {
    // Write the input Verilog to a temporary file so that `slang` can read it.
    let tmp_file = tempfile::Builder::new().suffix(".sv").tempfile()?;
    std::fs::write(tmp_file.path(), verilog)?;

    // Locate the `slang` executable.
    let slang_path = which::which("slang")
        .map_err(|_| anyhow::anyhow!("`slang` executable not found in PATH"))?;

    // Invoke slang to obtain the AST in JSON form on stdout.
    let output = Command::new(slang_path)
        .arg("--quiet")
        .arg("--single-unit")
        .arg("--ast-json")
        .arg("-")
        .arg(tmp_file.path())
        .output()?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "slang failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let json: Value = serde_json::from_slice(&output.stdout)?;

    // Navigate JSON: design -> members[] (looking for Instance) -> body ->
    // members[] -> kind=="Port".
    let design = json
        .get("design")
        .ok_or_else(|| anyhow::anyhow!("Missing `design` in Slang JSON"))?;
    let members = design
        .get("members")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing `members` array in design"))?;

    // Find the last "Instance" member – assumed to be the top module instance.
    let inst = members
        .iter()
        .filter(|m| m.get("kind").and_then(|k| k.as_str()) == Some("Instance"))
        .last()
        .ok_or_else(|| anyhow::anyhow!("No Instance found in design members"))?;

    let module_name = inst
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Instance missing name"))?
        .to_string();

    let body_members = inst
        .get("body")
        .and_then(|b| b.get("members"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing body members for instance"))?;

    let mut ports = Vec::new();
    for m in body_members {
        if m.get("kind").and_then(|k| k.as_str()) != Some("Port") {
            continue;
        }
        let name = m
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Port missing name"))?;
        let dir = m.get("direction").and_then(|v| v.as_str()).unwrap_or("In");
        let type_str = m.get("type").and_then(|v| v.as_str()).unwrap_or("logic");

        // Infer bit width from the type string, defaulting to 1.
        let width = if let Some(lb) = type_str.find('[') {
            if let Some(rb) = type_str[lb + 1..].find(']') {
                let inside = &type_str[lb + 1..lb + 1 + rb];
                // Expect forms like 31:0 or 0:0
                let parts: Vec<&str> = inside.split(':').collect();
                if parts.len() == 2 {
                    if let (Ok(msb), Ok(lsb)) = (
                        parts[0].trim().parse::<i32>(),
                        parts[1].trim().parse::<i32>(),
                    ) {
                        (msb - lsb).abs() as usize + 1
                    } else {
                        1
                    }
                } else {
                    1
                }
            } else {
                1
            }
        } else {
            1
        };

        ports.push(PortInfo {
            name: name.to_string(),
            width,
            is_input: dir.eq_ignore_ascii_case("In"),
        });
    }

    if ports.is_empty() {
        return Err(anyhow::anyhow!("No ports found in top module"));
    }

    Ok((module_name, ports))
}

/// Compiles the given sources with iverilog and runs them via vvp, returning
/// the captured stdout.
fn compile_and_run(work_dir: &Path, sources: &[PathBuf]) -> anyhow::Result<String> {
    // Locate iverilog.
    let iverilog_path =
        which::which("iverilog").map_err(|_| anyhow::anyhow!("iverilog not found"))?;
    let vvp_out = work_dir.join("sim.vvp");
    // Build.
    let mut cmd_compile = Command::new(&iverilog_path);
    cmd_compile
        .current_dir(work_dir)
        .arg("-g2012")
        .arg("-o")
        .arg(&vvp_out)
        .arg("-s")
        .arg("tb");
    for src in sources {
        cmd_compile.arg(src);
    }
    let out_compile = cmd_compile.output()?;
    if !out_compile.status.success() {
        return Err(anyhow::anyhow!(
            "iverilog failed: {}",
            String::from_utf8_lossy(&out_compile.stderr)
        ));
    }

    // Run simulation.
    let out_sim = Command::new("vvp")
        .current_dir(work_dir)
        .arg(&vvp_out)
        .output()?;
    if !out_sim.status.success() {
        return Err(anyhow::anyhow!(
            "vvp failed: {}",
            String::from_utf8_lossy(&out_sim.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out_sim.stdout).to_string())
}

pub fn handle_run_verilog_pipeline(matches: &ArgMatches) {
    let _ = env_logger::try_init();
    // Obtain SV source: either from file path argument or stdin ("-").
    let sv_path_arg = matches
        .get_one::<String>("sv_path")
        .map(String::as_str)
        .unwrap_or("-");
    let sv_input = if sv_path_arg == "-" {
        let mut buf = String::new();
        if std::io::stdin().read_to_string(&mut buf).is_err() || buf.is_empty() {
            eprintln!(
                "run-verilog-pipeline: expected SystemVerilog source on stdin (or specify a file path)"
            );
            std::process::exit(1);
        }
        buf
    } else {
        match std::fs::read_to_string(sv_path_arg) {
            Ok(contents) => contents,
            Err(e) => {
                eprintln!(
                    "run-verilog-pipeline: failed to read '{}': {}",
                    sv_path_arg, e
                );
                std::process::exit(1);
            }
        }
    };

    // Parse module and ports.
    let (module_name, ports) = match parse_verilog_top_module(&sv_input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("run-verilog-pipeline: failed to parse module header: {}", e);
            std::process::exit(1);
        }
    };

    // Extract CLI args.
    let input_valid_signal = matches
        .get_one::<String>("input_valid_signal")
        .map(String::as_str);
    let output_valid_signal = matches
        .get_one::<String>("output_valid_signal")
        .map(String::as_str);
    let reset_signal = matches.get_one::<String>("reset").map(String::as_str);
    let reset_active_low = matches
        .get_one::<String>("reset_active_low")
        .map(|s| s == "true")
        .unwrap_or(false);
    let latency_opt = matches.get_one::<String>("latency");
    let waves_path = matches.get_one::<String>("waves");

    if output_valid_signal.is_none() && latency_opt.is_none() {
        eprintln!("run-verilog-pipeline: --latency must be provided when --output_valid_signal is not used");
        std::process::exit(1);
    }
    let latency: usize = latency_opt.map(|s| s.parse().unwrap()).unwrap_or(0);

    // Require an explicit clock port named `clk`.
    if ports
        .iter()
        .find(|p| p.name == "clk" && p.is_input)
        .is_none()
    {
        eprintln!("run-verilog-pipeline: top module must have an input port named `clk`");
        std::process::exit(1);
    }

    // Determine clk and handshake names to exclude from data lists.
    let mut data_inputs: Vec<&PortInfo> = ports
        .iter()
        .filter(|p| {
            if !p.is_input {
                return false;
            }
            if p.name == "clk" {
                return false;
            }
            if let Some(s) = input_valid_signal {
                if p.name == s {
                    return false;
                }
            }
            if let Some(s) = reset_signal {
                if p.name == s {
                    return false;
                }
            }
            true
        })
        .collect();

    if data_inputs.is_empty() {
        eprintln!("No data input ports detected – at least one is required");
        std::process::exit(1);
    }

    // Parse the XLS IR value provided on the command line. It may be either a
    // single bits value (for a single data input) *or* a tuple whose arity
    // must match the number of data input ports detected above.
    let input_value_str = matches
        .get_one::<String>("input_value")
        .expect("input_value arg");
    let input_value = match IrValue::parse_typed(input_value_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "Failed to parse input XLS IR value '{}': {}",
                input_value_str, e
            );
            std::process::exit(1);
        }
    };

    // Map each data input port to a corresponding IrBits value.
    let mut input_port_bits: Vec<(&PortInfo, xlsynth::IrBits)> = Vec::new();
    if data_inputs.len() == 1 {
        // Expect a single bits value.
        match input_value.to_bits() {
            Ok(bits) => {
                input_port_bits.push((data_inputs[0], bits));
            }
            Err(_) => {
                eprintln!(
                    "For a single data input port the <INPUT_VALUE> argument must be a bits value; got: {}",
                    input_value_str
                );
                std::process::exit(1);
            }
        }
    } else {
        // Expect a tuple with arity equal to the number of data input ports.
        let elems = match input_value.get_elements() {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "With {} data input ports the <INPUT_VALUE> argument must be a tuple of equal arity; got non-tuple value {}",
                    data_inputs.len(), input_value_str
                );
                std::process::exit(1);
            }
        };
        if elems.len() != data_inputs.len() {
            eprintln!(
                "Tuple arity ({}) does not match number of data input ports ({})",
                elems.len(),
                data_inputs.len()
            );
            std::process::exit(1);
        }
        for (port, elem) in data_inputs.iter().zip(elems.iter()) {
            match elem.to_bits() {
                Ok(bits) => input_port_bits.push((*port, bits)),
                Err(_) => {
                    eprintln!(
                        "Tuple element for port '{}' is not a bits value: {}",
                        port.name, elem
                    );
                    std::process::exit(1);
                }
            }
        }
    }

    // Determine data outputs (exclude handshake outputs like output_valid_signal)
    let data_outputs: Vec<&PortInfo> = ports
        .iter()
        .filter(|p| {
            if p.is_input {
                return false;
            }
            if let Some(s) = output_valid_signal {
                if p.name == s {
                    return false;
                }
            }
            true
        })
        .collect();

    if data_outputs.is_empty() {
        eprintln!("No data output ports detected");
        std::process::exit(1);
    }

    // Build testbench.
    let mut tb_src = String::new();
    tb_src.push_str("`timescale 1ns/1ps\nmodule tb;\n  reg clk = 0;\n  always #5 clk = ~clk;\n");

    // Declarations for data input ports.
    for (port, _) in &input_port_bits {
        tb_src.push_str(&format!(
            "  reg [{}:0] {} = 0;\n",
            port.width - 1,
            port.name
        ));
    }
    // Handshake/reg declarations.
    if let Some(in_valid) = input_valid_signal {
        tb_src.push_str(&format!("  reg {} = 0;\n", in_valid));
    }
    if let Some(reset) = reset_signal {
        tb_src.push_str(&format!(
            "  reg {} = {};\n",
            reset,
            if reset_active_low { 0 } else { 1 }
        ));
    }
    if let Some(out_valid) = output_valid_signal {
        tb_src.push_str(&format!("  wire {} ;\n", out_valid));
    }
    for outp in &data_outputs {
        tb_src.push_str(&format!("  wire [{}:0] {};\n", outp.width - 1, outp.name));
    }

    // Instantiate DUT.
    tb_src.push_str(&format!("  {} dut(.clk(clk)", module_name));
    if let Some(reset) = reset_signal {
        tb_src.push_str(&format!(", .{}({})", reset, reset));
    }
    if let Some(in_valid) = input_valid_signal {
        tb_src.push_str(&format!(", .{}({})", in_valid, in_valid));
    }
    for outp in &data_outputs {
        tb_src.push_str(&format!(", .{}({})", outp.name, outp.name));
    }
    // Data input connections.
    for (port, _) in &input_port_bits {
        tb_src.push_str(&format!(", .{}({})", port.name, port.name));
    }
    if let Some(out_valid) = output_valid_signal {
        tb_src.push_str(&format!(", .{}({})", out_valid, out_valid));
    }
    tb_src.push_str(");\n");

    // Start initial block.
    tb_src.push_str(
        "  integer i;\n  initial begin\n    $dumpfile(\"dump.vcd\");\n    $dumpvars(0, tb);\n",
    );

    if let Some(reset) = reset_signal {
        let init_val = if reset_active_low { 0 } else { 1 };
        let deassert_val = if reset_active_low { 1 } else { 0 };
        tb_src.push_str(&format!("    {} = {}'b{};\n", reset, 1, init_val));
        tb_src.push_str("    for (i = 0; i < 2; i = i + 1) @(posedge clk);\n");
        tb_src.push_str(&format!("    {} = {}'b{};\n", reset, 1, deassert_val));
    }

    // Apply input after one negedge so it aligns with clk.
    tb_src.push_str("    @(negedge clk);\n");
    for (port, bits) in &input_port_bits {
        let hex_val = bits
            .to_string_fmt(IrFormatPreference::Hex, false)
            .trim_start_matches("0x")
            .to_string();
        tb_src.push_str(&format!(
            "    {} = {}'h{};\n",
            port.name, port.width, hex_val
        ));
    }
    if let Some(in_valid) = input_valid_signal {
        tb_src.push_str(&format!("    {} = 1'b1;\n", in_valid));
        if let Some(out_valid) = output_valid_signal {
            tb_src.push_str(&format!("    wait ({});\n", out_valid));
        } else {
            // Fallback: wait latency cycles when no explicit output_valid.
            tb_src.push_str(&format!(
                "    for (i = 0; i < {}; i = i + 1) @(posedge clk);\n",
                latency
            ));
        }
        tb_src.push_str(&format!("    #1;\n    {} = 1'b0;\n", in_valid));
    } else if output_valid_signal.is_none() {
        // No handshake: wait latency cycles.
        tb_src.push_str(&format!(
            "    for (i = 0; i < {}; i = i + 1) @(posedge clk);\n    #1;\n",
            latency
        ));
    }

    // Display outputs.
    for outp in &data_outputs {
        tb_src.push_str(&format!(
            "    $display(\"{}: bits[{}]:%0d\", {});\n",
            outp.name, outp.width, outp.name
        ));
    }

    tb_src.push_str("    $finish;\n  end\n");

    // Timeout watchdog: if the simulation exceeds a generous cycle budget, abort.
    // Each clock cycle is 10 ns (clk toggles every 5 ns), so we wait
    // (latency+50)*10 ns.
    let timeout_cycles = latency + 50;
    tb_src.push_str(&format!(
        "  initial begin\n    #{};\n    $fatal(1, \"Simulation timed out\");\n  end\n",
        timeout_cycles * 10
    ));

    tb_src.push_str("endmodule\n");

    // Write sources to temp dir and run.
    let temp_dir = tempdir().expect("tempdir");
    let dut_path = temp_dir.path().join("dut.sv");
    let tb_path = temp_dir.path().join("tb.sv");
    std::fs::write(&dut_path, sv_input).expect("write dut");
    std::fs::write(&tb_path, tb_src).expect("write tb");

    // Run compile+sim.
    let stdout_sim = match compile_and_run(temp_dir.path(), &[dut_path, tb_path]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Simulation failed: {}", e);
            std::process::exit(1);
        }
    };

    // Copy waves if requested.
    if let Some(path) = waves_path {
        let vcd_src = temp_dir.path().join("dump.vcd");
        if let Err(e) = std::fs::copy(&vcd_src, Path::new(path)) {
            eprintln!("Warning: failed to copy wave VCD to {}: {}", path, e);
        }
    }

    // Filter and print mapping lines.
    let re_map = Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*: bits\[\d+\]:").unwrap();
    for line in stdout_sim.lines() {
        if re_map.is_match(line.trim()) {
            println!("{}", line.trim());
        }
    }
}
