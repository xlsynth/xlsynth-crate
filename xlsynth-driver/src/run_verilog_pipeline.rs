// SPDX-License-Identifier: Apache-2.0

use crate::common::{execute_command_with_context, find_and_verify_executable};
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
    log::debug!("parse_verilog_top_module: Starting");
    // Write the input Verilog to a temporary file so that `slang` can read it.
    log::debug!("Creating temporary file for slang input");
    let tmp_file = tempfile::Builder::new().suffix(".sv").tempfile()?;
    log::debug!(
        "Writing {} bytes to temp file: {:?}",
        verilog.len(),
        tmp_file.path()
    );
    std::fs::write(tmp_file.path(), verilog)?;
    log::debug!("Successfully wrote temp file");
    log::debug!(
        "Temp file content preview: {}",
        &verilog[..verilog.len().min(200)]
    );

    // Locate and verify the `slang` executable.
    log::debug!("Looking for slang executable");
    let slang_path = find_and_verify_executable(
        "slang",
        "Please ensure slang is installed and available in PATH. \
         You can download it from https://github.com/xlsynth/slang-rs/releases or install it via your package manager."
    )?;
    log::debug!("Found slang at: {:?}", slang_path);

    // Invoke slang to obtain the AST in JSON form on stdout.
    log::debug!("Building slang command");
    let mut cmd = Command::new(&slang_path);
    cmd.arg("--single-unit")
        .arg("--quiet") // Suppress status messages like "Top level design units:" and "Build succeeded:" that would
        // break JSON parsing
        .arg("--ast-json")
        .arg("-") // Tell slang to write JSON AST to stdout (this is the output file parameter for --ast-json)
        .arg(tmp_file.path());
    // Verify temp file exists and is readable
    if let Ok(metadata) = std::fs::metadata(tmp_file.path()) {
        log::debug!("Temp file size on disk: {} bytes", metadata.len());
    } else {
        log::debug!("Warning: temp file doesn't seem to exist!");
    }

    log::debug!("About to execute slang command: {:?}", cmd);
    let output = execute_command_with_context(
        cmd,
        &format!(
            "Failed to execute slang binary at '{}'",
            slang_path.display()
        ),
    )?;
    log::debug!(
        "Slang execution completed, stdout size: {} bytes",
        output.stdout.len()
    );

    if !output.status.success() {
        log::debug!("Slang stderr: {}", String::from_utf8_lossy(&output.stderr));
        log::debug!("Slang stdout: {}", String::from_utf8_lossy(&output.stdout));
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
    // Locate and verify iverilog.
    let iverilog_path = find_and_verify_executable(
        "iverilog",
        "Please install iverilog. On Ubuntu/Debian: 'sudo apt-get install iverilog', on macOS: 'brew install icarus-verilog'"
    )?;
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
    let out_compile = execute_command_with_context(cmd_compile, "Failed to execute iverilog")?;
    if !out_compile.status.success() {
        return Err(anyhow::anyhow!(
            "iverilog failed: {}",
            String::from_utf8_lossy(&out_compile.stderr)
        ));
    }

    // Run simulation.
    let mut vvp_cmd = Command::new("vvp");
    vvp_cmd.current_dir(work_dir).arg(&vvp_out);
    let out_sim =
        execute_command_with_context(vvp_cmd, "Failed to execute vvp (Verilog simulator)")?;
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
    log::debug!("Starting run_verilog_pipeline");
    // Obtain SV source: either from file path argument or stdin ("-").
    let sv_path_arg = matches
        .get_one::<String>("sv_path")
        .map(String::as_str)
        .unwrap_or_else(|| {
            eprintln!("run-verilog-pipeline: missing required SV_PATH argument");
            eprintln!("Usage: xlsynth-driver run-verilog-pipeline <SV_PATH> [INPUT_VALUE]");
            std::process::exit(1);
        });
    log::debug!("Reading input from: {}", sv_path_arg);
    let sv_input = if sv_path_arg == "-" {
        log::debug!("Reading from stdin");
        let mut buf = String::new();
        if std::io::stdin().read_to_string(&mut buf).is_err() || buf.is_empty() {
            eprintln!(
                "run-verilog-pipeline: expected SystemVerilog source on stdin (or specify a file path)"
            );
            std::process::exit(1);
        }
        buf
    } else {
        log::debug!("Reading from file: {}", sv_path_arg);
        match std::fs::read_to_string(sv_path_arg) {
            Ok(contents) => {
                log::debug!("Successfully read {} bytes from file", contents.len());
                contents
            }
            Err(e) => {
                eprintln!(
                    "run-verilog-pipeline: failed to read '{}': {}",
                    sv_path_arg, e
                );
                std::process::exit(1);
            }
        }
    };
    log::debug!("Input size: {} bytes", sv_input.len());

    // Parse module and ports.
    log::debug!("About to parse verilog module");
    let (module_name, ports) = match parse_verilog_top_module(&sv_input) {
        Ok(v) => {
            log::debug!("Successfully parsed module: {}", v.0);
            v
        }
        Err(e) => {
            eprintln!("run-verilog-pipeline: failed to parse module header: {}", e);
            std::process::exit(1);
        }
    };
    log::debug!("Found module '{}' with {} ports", module_name, ports.len());

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
    let latency: usize = if let Some(latency_str) = latency_opt {
        match latency_str.parse() {
            Ok(val) => val,
            Err(e) => {
                eprintln!(
                    "run-verilog-pipeline: invalid latency value '{}': {}",
                    latency_str, e
                );
                std::process::exit(1);
            }
        }
    } else {
        0
    };

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
    let data_inputs: Vec<&PortInfo> = ports
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

    // Parse or generate the XLS IR value. If not provided, generate zero values.
    let input_value = match matches.get_one::<String>("input_value") {
        Some(input_value_str) => {
            // User provided an input value, parse it
            match IrValue::parse_typed(input_value_str) {
                Ok(v) => {
                    println!("Using provided input: {}", input_value_str);
                    v
                }
                Err(e) => {
                    eprintln!(
                        "Failed to parse input XLS IR value '{}': {}",
                        input_value_str, e
                    );
                    eprintln!("Expected an XLS IR value like 'bits[32]:5' or 'tuple(bits[8]:1, bits[16]:2)'.");
                    eprintln!("Usage: xlsynth-driver run-verilog-pipeline <SV_PATH> [INPUT_VALUE]");
                    std::process::exit(1);
                }
            }
        }
        None => {
            // Generate zero values based on data input ports
            let zero_value = if data_inputs.len() == 1 {
                // Single input: create a zero bits value
                let port = data_inputs[0];
                match IrValue::parse_typed(&format!("bits[{}]:0", port.width)) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "Failed to generate zero value for port '{}': {}",
                            port.name, e
                        );
                        std::process::exit(1);
                    }
                }
            } else {
                // Multiple inputs: create a zero tuple
                let mut zero_elements = Vec::new();
                for port in &data_inputs {
                    match IrValue::parse_typed(&format!("bits[{}]:0", port.width)) {
                        Ok(v) => zero_elements.push(v),
                        Err(e) => {
                            eprintln!(
                                "Failed to generate zero value for port '{}': {}",
                                port.name, e
                            );
                            std::process::exit(1);
                        }
                    }
                }
                IrValue::make_tuple(&zero_elements)
            };
            let zero_str = zero_value.to_string();
            println!("No input value provided, using zero values: {}", zero_str);
            println!(
                "To use different values, run: xlsynth-driver run-verilog-pipeline {} \"{}\"",
                sv_path_arg, zero_str
            );
            zero_value
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
                    input_value.to_string()
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
                    data_inputs.len(), input_value.to_string()
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
        // Assert the input valid signal for exactly one clock cycle.
        tb_src.push_str(&format!("    {} = 1'b1;\n", in_valid));
        if let Some(out_valid) = output_valid_signal {
            tb_src.push_str("    fork\n");
            // Thread 0: wait for valid handshake to complete
            tb_src.push_str("      begin\n");
            tb_src.push_str(&format!("        wait ({});\n", out_valid));
            tb_src.push_str(&format!("        wait (!{});\n", out_valid));
            tb_src.push_str("      end\n");
            // Thread 1: keep in_valid asserted for exactly one cycle
            tb_src.push_str("      begin\n");
            tb_src.push_str("        @(posedge clk);\n");
            tb_src.push_str("        #1;\n");
            // De-assert valid and immediately clear data inputs.
            tb_src.push_str(&format!("        {} = 1'b0;\n", in_valid));
            for (port, _) in &input_port_bits {
                tb_src.push_str(&format!("        {} = {}'h0;\n", port.name, port.width));
            }
            tb_src.push_str("      end\n");
            tb_src.push_str("    join\n");
        } else {
            // No output_valid: assert for one cycle then wait latency.
            tb_src.push_str("    @(posedge clk);\n");
            tb_src.push_str("    #1;\n");
            tb_src.push_str(&format!("    {} = 1'b0;\n", in_valid));
            tb_src.push_str(&format!(
                "    for (i = 0; i < {}; i = i + 1) @(posedge clk);\n",
                latency
            ));
        }
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

    // Let the design run for one extra cycle so waveforms capture the stable
    // outputs and any handshake de-assertion before ending.
    tb_src.push_str("    @(posedge clk);\n    $finish;\n  end\n");

    // Timeout watchdog: if the simulation exceeds a generous cycle budget, abort.
    // Each clock cycle is 10 ns (clk toggles every 5 ns).  Deep pipelines with
    // handshake signalling can legitimately run for hundreds of cycles before
    // `out_valid` drops, so budget much more generously.
    let timeout_cycles = latency + 500;
    tb_src.push_str(&format!(
        "  initial begin\n    #{};\n    $fatal(1, \"Simulation timed out\");\n  end\n",
        timeout_cycles * 10
    ));

    tb_src.push_str("endmodule\n");

    // Write sources to temp dir and run.
    let temp_dir = match tempdir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!(
                "run-verilog-pipeline: failed to create temporary directory: {}",
                e
            );
            std::process::exit(1);
        }
    };
    let dut_path = temp_dir.path().join("dut.sv");
    let tb_path = temp_dir.path().join("tb.sv");
    if let Err(e) = std::fs::write(&dut_path, sv_input) {
        eprintln!("run-verilog-pipeline: failed to write DUT file: {}", e);
        std::process::exit(1);
    }
    if let Err(e) = std::fs::write(&tb_path, tb_src) {
        eprintln!(
            "run-verilog-pipeline: failed to write testbench file: {}",
            e
        );
        std::process::exit(1);
    }

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
    let re_map = Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*: bits\[\d+\]:")
        .expect("regex pattern should be valid");
    for line in stdout_sim.lines() {
        if re_map.is_match(line.trim()) {
            println!("{}", line.trim());
        }
    }
}
