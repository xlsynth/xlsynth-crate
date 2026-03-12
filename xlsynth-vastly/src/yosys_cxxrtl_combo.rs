// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::process::Command;
use std::time::SystemTime;

use crate::CompiledComboModule;
use crate::Env;
use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::compile_combo_module;
use crate::env_is_two_value_safe;

/// Returns whether the local Yosys/CXXRTL toolchain required for the
/// two-valued reference-implementation path is available on `PATH`.
pub fn has_yosys_cxxrtl_toolchain() -> bool {
    command_works("yosys", &["-V"])
        && command_works("g++", &["--version"])
        && cxxrtl_runtime_include_dir().is_some()
}

/// Evaluates a combinational Verilog module through Yosys+CXXRTL for a single
/// concrete two-valued input vector and returns the top-level output bindings.
pub fn eval_yosys_cxxrtl_combo(
    src: &str,
    module_name: &str,
    inputs: &BTreeMap<String, Value4>,
) -> Result<BTreeMap<String, Value4>> {
    require_yosys_cxxrtl_toolchain()?;
    let m = compile_combo_module(src)?;
    ensure_two_value_inputs(inputs)?;
    validate_inputs(&m, inputs)?;

    let td = mk_temp_dir()?;
    let result = (|| {
        let dut_path = td.join("dut.v");
        let cxxrtl_cc_path = td.join("dut.cc");
        let driver_cc_path = td.join("driver.cc");
        let sim_path = td.join("sim");

        std::fs::write(&dut_path, src)
            .map_err(|e| Error::Parse(format!("failed to write Yosys DUT source: {e}")))?;
        std::fs::write(&driver_cc_path, render_driver_cpp(module_name, &m, inputs))
            .map_err(|e| Error::Parse(format!("failed to write CXXRTL driver source: {e}")))?;

        let yosys_script = format!(
            "read_verilog {dut}; rename {top} top; hierarchy -top top; proc; memory_memx; opt_expr -keepdc; opt_clean; write_cxxrtl -header {out}",
            dut = yosys_path(&dut_path),
            top = module_name,
            out = yosys_path(&cxxrtl_cc_path),
        );
        let yosys = Command::new("yosys")
            .current_dir(&td)
            .arg("-Q")
            .arg("-p")
            .arg(&yosys_script)
            .output()
            .map_err(|e| Error::Parse(format!("failed to run yosys: {e}")))?;
        if !yosys.status.success() {
            return Err(Error::Parse(format!(
                "yosys failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&yosys.stdout),
                String::from_utf8_lossy(&yosys.stderr),
            )));
        }

        let cxxrtl_include_dir = cxxrtl_runtime_include_dir().ok_or_else(|| {
            Error::Parse("failed to locate the Yosys CXXRTL runtime include directory".to_string())
        })?;
        let compile = Command::new("g++")
            .current_dir(&td)
            .arg("-std=c++17")
            .arg("-O3")
            .arg(format!("-I{}", cxxrtl_include_dir.display()))
            .arg("-o")
            .arg(&sim_path)
            .arg(&driver_cc_path)
            .arg(&cxxrtl_cc_path)
            .output()
            .map_err(|e| Error::Parse(format!("failed to compile CXXRTL driver: {e}")))?;
        if !compile.status.success() {
            return Err(Error::Parse(format!(
                "CXXRTL compile failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&compile.stdout),
                String::from_utf8_lossy(&compile.stderr),
            )));
        }

        let run = Command::new(&sim_path)
            .current_dir(&td)
            .output()
            .map_err(|e| Error::Parse(format!("failed to run compiled CXXRTL binary: {e}")))?;
        if !run.status.success() {
            return Err(Error::Parse(format!(
                "CXXRTL execution failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&run.stdout),
                String::from_utf8_lossy(&run.stderr),
            )));
        }

        parse_driver_output(&m, &String::from_utf8_lossy(&run.stdout))
    })();
    let _ = std::fs::remove_dir_all(&td);
    result
}

fn command_works(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn require_yosys_cxxrtl_toolchain() -> Result<()> {
    if has_yosys_cxxrtl_toolchain() {
        return Ok(());
    }
    Err(Error::Parse(
        "yosys with CXXRTL runtime headers and g++ are required for Yosys/CXXRTL reference simulation".to_string(),
    ))
}

fn cxxrtl_runtime_include_dir() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("YOSYS_CXXRTL_INCLUDE_DIR") {
        let path = std::path::PathBuf::from(path);
        if path.join("cxxrtl/capi/cxxrtl_capi.h").is_file() {
            return Some(path);
        }
    }

    let yosys = find_on_path("yosys")?;
    let prefix = yosys.parent()?.parent()?;
    let path = prefix.join("share/yosys/include/backends/cxxrtl/runtime");
    if path.join("cxxrtl/capi/cxxrtl_capi.h").is_file() {
        return Some(path);
    }
    None
}

fn find_on_path(binary: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for entry in std::env::split_paths(&path) {
        let candidate = entry.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn ensure_two_value_inputs(inputs: &BTreeMap<String, Value4>) -> Result<()> {
    let mut env = Env::new();
    for (name, value) in inputs {
        env.insert(name.clone(), value.clone());
    }
    if env_is_two_value_safe(&env) {
        return Ok(());
    }
    Err(Error::Parse(
        "Yosys/CXXRTL reference simulation requires concrete two-valued inputs".to_string(),
    ))
}

fn validate_inputs(m: &CompiledComboModule, inputs: &BTreeMap<String, Value4>) -> Result<()> {
    if m.input_ports.len() != inputs.len() {
        return Err(Error::Parse(format!(
            "input port count mismatch: module has {} inputs but vector has {}",
            m.input_ports.len(),
            inputs.len()
        )));
    }
    for p in &m.input_ports {
        let value = inputs
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("missing expected input `{}`", p.name)))?;
        if value.width != p.width {
            return Err(Error::Parse(format!(
                "input width mismatch for `{}`: module width={} vector width={}",
                p.name, p.width, value.width
            )));
        }
    }
    Ok(())
}

fn render_driver_cpp(
    _module_name: &str,
    m: &CompiledComboModule,
    inputs: &BTreeMap<String, Value4>,
) -> String {
    let mut s = String::new();
    s.push_str("#include \"dut.h\"\n");
    s.push_str("#include <cstddef>\n");
    s.push_str("#include <cstdint>\n");
    s.push_str("#include <cstring>\n");
    s.push_str("#include <iostream>\n");
    s.push_str("#include <stdexcept>\n");
    s.push_str("#include <string>\n\n");
    s.push_str("template<size_t Bits>\n");
    s.push_str("void set_hex(cxxrtl::value<Bits> &dst, const char *hex) {\n");
    s.push_str("  for (size_t i = 0; i < cxxrtl::value<Bits>::chunks; ++i) dst.data[i] = 0;\n");
    s.push_str("  const size_t len = std::strlen(hex);\n");
    s.push_str("  size_t bit_index = 0;\n");
    s.push_str("  for (size_t n = 0; n < len; ++n) {\n");
    s.push_str("    const char ch = hex[len - 1 - n];\n");
    s.push_str("    unsigned nibble = 0;\n");
    s.push_str("    if (ch >= '0' && ch <= '9') nibble = unsigned(ch - '0');\n");
    s.push_str("    else if (ch >= 'a' && ch <= 'f') nibble = 10u + unsigned(ch - 'a');\n");
    s.push_str("    else if (ch >= 'A' && ch <= 'F') nibble = 10u + unsigned(ch - 'A');\n");
    s.push_str("    else throw std::runtime_error(\"bad hex input\");\n");
    s.push_str("    for (size_t k = 0; k < 4 && bit_index < Bits; ++k, ++bit_index) {\n");
    s.push_str("      if ((nibble & (1u << k)) == 0) continue;\n");
    s.push_str("      const size_t chunk_index = bit_index / cxxrtl::value<Bits>::chunk::bits;\n");
    s.push_str("      const size_t chunk_offset = bit_index % cxxrtl::value<Bits>::chunk::bits;\n");
    s.push_str("      dst.data[chunk_index] |= cxxrtl::chunk_t(1) << chunk_offset;\n");
    s.push_str("    }\n");
    s.push_str("  }\n");
    s.push_str("}\n\n");
    s.push_str("template<size_t Bits>\n");
    s.push_str("std::string to_bits(const cxxrtl::value<Bits> &value) {\n");
    s.push_str("  std::string out;\n");
    s.push_str("  out.reserve(Bits);\n");
    s.push_str("  for (size_t n = 0; n < Bits; ++n) {\n");
    s.push_str("    const size_t bit_index = Bits - 1 - n;\n");
    s.push_str("    const size_t chunk_index = bit_index / cxxrtl::value<Bits>::chunk::bits;\n");
    s.push_str("    const size_t chunk_offset = bit_index % cxxrtl::value<Bits>::chunk::bits;\n");
    s.push_str(
        "    const bool bit = ((value.data[chunk_index] >> chunk_offset) & cxxrtl::chunk_t(1)) != 0;\n",
    );
    s.push_str("    out.push_back(bit ? '1' : '0');\n");
    s.push_str("  }\n");
    s.push_str("  return out;\n");
    s.push_str("}\n\n");
    writeln!(&mut s, "int main() {{\n  cxxrtl_design::p_top top;\n").unwrap();
    for input in &m.input_ports {
        let value = inputs.get(&input.name).unwrap();
        let hex = value.to_hex_string_if_known().unwrap();
        writeln!(
            &mut s,
            "  set_hex(top.p_{name}, \"{hex}\");",
            name = input.name
        )
        .unwrap();
    }
    s.push_str("  top.step();\n");
    for output in &m.output_ports {
        writeln!(
            &mut s,
            "  std::cout << \"OUT name={name} bits=\" << to_bits(top.p_{name}) << \"\\n\";",
            name = output.name
        )
        .unwrap();
    }
    s.push_str("  return 0;\n}\n");
    s
}

fn parse_driver_output(m: &CompiledComboModule, stdout: &str) -> Result<BTreeMap<String, Value4>> {
    let mut out = BTreeMap::new();
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with("OUT ") {
            continue;
        }
        let mut name = None;
        let mut bits = None;
        for part in line.split_whitespace() {
            if let Some(value) = part.strip_prefix("name=") {
                name = Some(value.to_string());
            } else if let Some(value) = part.strip_prefix("bits=") {
                bits = Some(value.to_string());
            }
        }
        let name =
            name.ok_or_else(|| Error::Parse(format!("malformed CXXRTL output line: {line}")))?;
        let bits =
            bits.ok_or_else(|| Error::Parse(format!("malformed CXXRTL output line: {line}")))?;
        out.insert(name, value4_from_two_value_bits(&bits)?);
    }
    for output in &m.output_ports {
        if !out.contains_key(&output.name) {
            return Err(Error::Parse(format!(
                "missing output `{}` in CXXRTL output",
                output.name
            )));
        }
    }
    Ok(out)
}

fn value4_from_two_value_bits(bits: &str) -> Result<Value4> {
    let mut out = Vec::with_capacity(bits.len());
    for c in bits.chars().rev() {
        out.push(match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            _ => {
                return Err(Error::Parse(format!(
                    "unexpected non-two-value CXXRTL bit `{c}` in `{bits}`"
                )));
            }
        });
    }
    Ok(Value4::new(out.len() as u32, Signedness::Unsigned, out))
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
        let p = base.join(format!("vastly_cxxrtl_combo_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(Error::Parse(format!("create temp dir failed: {e}"))),
        }
    }
    Err(Error::Parse(
        "failed to create unique temp dir for Yosys/CXXRTL combo run".to_string(),
    ))
}

fn yosys_path(path: &std::path::Path) -> String {
    path.to_string_lossy().into_owned()
}
