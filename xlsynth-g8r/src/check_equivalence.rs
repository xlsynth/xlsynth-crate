// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use crate::{gate, gate2ir, xls_ir::ir};

pub fn check_equivalence(orig_package: &str, gate_package: &str) -> Result<(), String> {
    check_equivalence_with_top(orig_package, gate_package, None, false)
}

pub fn check_equivalence_with_top(
    orig_package: &str,
    gate_package: &str,
    top_fn_name: Option<&str>,
    keep_temp_dir: bool,
) -> Result<(), String> {
    let tempdir = tempfile::tempdir().unwrap();
    let dirpath = if keep_temp_dir {
        tempdir.keep()
    } else {
        tempdir.path().to_path_buf()
    };
    let orig_path = dirpath.join("orig.ir");
    let gate_path = dirpath.join("gate.ir");
    std::fs::write(orig_path.clone(), orig_package).unwrap();
    std::fs::write(gate_path.clone(), gate_package).unwrap();
    let tools_dir_str =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS env var should be set");
    let tools_dirpath = std::path::PathBuf::from(tools_dir_str);
    assert!(
        tools_dirpath.exists(),
        "XLSYNTH_TOOLS environment variable does not exist"
    );
    let check_ir_equivalence_main_path = tools_dirpath.join("check_ir_equivalence_main");
    assert!(
        check_ir_equivalence_main_path.exists(),
        "check_ir_equivalence_main not found in XLSYNTH_TOOLS"
    );

    let mut command = std::process::Command::new(check_ir_equivalence_main_path);
    // Optional: a flag like "--alsologtostderr" is useful while debugging to mirror
    // logs to stderr, but not required for functionality.
    command.arg(orig_path.to_str().unwrap());
    command.arg(gate_path.to_str().unwrap());
    if let Some(top) = top_fn_name {
        command.arg("--top");
        command.arg(top);
    }
    log::info!("check_equivalence_with_top; running command: {:?}", command);
    let start = Instant::now();
    let output = command.output().unwrap();
    let elapsed = start.elapsed();
    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let err_kind = if stderr_str.contains("DEADLINE_EXCEEDED")
            || stderr_str.contains("SIGINT")
            || stderr_str.contains("interrupted")
        {
            "TimedOutOrInterrupted"
        } else {
            "SolverFailed"
        };
        return Err(format!(
            "{}: check_ir_equivalence_main failed with retcode {}\nstdout: {:?}\nstderr: {:?}",
            err_kind, output.status, stdout_str, stderr_str
        ));
    }
    log::info!("check_equivalence_with_top; successful in {:?}", elapsed);
    Ok(())
}

fn get_fn_signature(f: &ir::Fn) -> String {
    let mut signature = String::new();
    signature.push_str("fn ");
    signature.push_str(&f.name);
    signature.push_str("(");
    for (i, param) in f.params.iter().enumerate() {
        signature.push_str(&format!("{}: {}", param.name, param.ty));
        if i + 1 != f.params.len() {
            signature.push_str(", ");
        }
    }
    signature.push_str(")");
    if !f.ret_ty.is_nil() {
        signature.push_str(&format!(" -> {}", f.ret_ty));
    }
    signature
}

pub fn validate_same_signature(orig_fn: &ir::Fn, gate_fn: &gate::GateFn) -> Result<(), String> {
    let gate_signature = gate_fn.get_signature();
    let orig_signature = get_fn_signature(orig_fn);
    if orig_signature != gate_signature {
        return Err(format!(
            "signature mismatch: original fn: `{}` != gate fn: `{}`",
            orig_signature, gate_signature
        ));
    }
    Ok(())
}

/// Note: if the original IR function has a different signature than the gate
/// function (because gate functions generally have flattened signatures into a
/// bit vector for each parameter / result tuple element) then we adjust in the
/// conversion from "gate IR" to "XLS IR" to use the original function's
/// signature so we can check XLS IR equivalence directly.
pub fn validate_same_fn(orig_fn: &ir::Fn, gate_fn: &gate::GateFn) -> Result<(), String> {
    let orig_ir_fn_text: String = orig_fn.to_string();
    let xlsynth_package_ir: String =
        gate2ir::gate_fn_to_xlsynth_ir(gate_fn, "gate", &orig_fn.get_type())
            .unwrap()
            .to_string();
    log::info!("xlsynth_package_ir:\n{}", xlsynth_package_ir);
    let orig_ir_pkg_text: String = format!("package orig\n\ntop {}", orig_ir_fn_text);
    let result = check_equivalence(&orig_ir_pkg_text, &xlsynth_package_ir);
    result
}

/// Structured result for IR-level equivalence checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrCheckResult {
    /// The two GateFns were proven equivalent.
    Equivalent,
    /// A concrete counter-example was found – they are *not* equivalent.
    NotEquivalent,
    /// The external solver timed out or was interrupted (SIGINT, etc.).
    TimedOutOrInterrupted,
    /// Some other infrastructure failure (I/O, missing tool etc.). Payload is
    /// explanatory message.
    OtherProcessError(String),
}

impl IrCheckResult {
    pub fn is_equivalent(&self) -> bool {
        matches!(self, IrCheckResult::Equivalent)
    }
}

/// Run the external `check_ir_equivalence_main` tool and get a structured
/// result. This is similar to the logic inside `check_equivalence_with_top`
/// but returns an `IrCheckResult` instead of `Result<(), String>`.
fn run_external_ir_tool(orig_pkg: &str, gate_pkg: &str) -> IrCheckResult {
    let tempdir = tempfile::tempdir().unwrap();
    let dirpath = tempdir.path();
    let orig_path = dirpath.join("orig.ir");
    let gate_path = dirpath.join("gate.ir");
    if std::fs::write(&orig_path, orig_pkg).is_err() {
        return IrCheckResult::OtherProcessError("failed to write orig temp file".to_string());
    }
    if std::fs::write(&gate_path, gate_pkg).is_err() {
        return IrCheckResult::OtherProcessError("failed to write gate temp file".to_string());
    }

    let tools_dir = match std::env::var("XLSYNTH_TOOLS") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            return IrCheckResult::OtherProcessError("XLSYNTH_TOOLS env var not set".to_string());
        }
    };
    let exe = tools_dir.join("check_ir_equivalence_main");
    if !exe.exists() {
        return IrCheckResult::OtherProcessError(format!(
            "check_ir_equivalence_main not found in {}",
            tools_dir.display()
        ));
    }

    let output = std::process::Command::new(exe)
        // Optional debug flag can mirror logs to stderr; not required for functionality.
        .arg(orig_path)
        .arg(gate_path)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => return IrCheckResult::OtherProcessError(format!("failed to spawn process: {e}")),
    };

    if output.status.success() {
        return IrCheckResult::Equivalent;
    }

    let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();

    if stderr_str.contains("DEADLINE_EXCEEDED")
        || stderr_str.contains("SIGINT")
        || stderr_str.contains("interrupted")
    {
        return IrCheckResult::TimedOutOrInterrupted;
    }

    // Heuristic: if the tool reports a counterexample it usually exits 1 and prints
    // something like "NOT EQUIVALENT" or "counterexample".
    if stderr_str.to_lowercase().contains("not equivalent")
        || stderr_str.to_lowercase().contains("counterexample")
    {
        return IrCheckResult::NotEquivalent;
    }

    IrCheckResult::OtherProcessError(format!("retcode {} stderr: {}", output.status, stderr_str))
}

/// Structured variant of `prove_same_gate_fn_via_ir`.
pub fn prove_same_gate_fn_via_ir_status(lhs: &gate::GateFn, rhs: &gate::GateFn) -> IrCheckResult {
    let lhs_type = lhs.get_flat_type();
    let rhs_type = rhs.get_flat_type();
    if lhs_type != rhs_type {
        return IrCheckResult::OtherProcessError("type mismatch".to_string());
    }
    let lhs_ir: xlsynth::IrPackage =
        crate::gate2ir::gate_fn_to_xlsynth_ir(lhs, "lhs", &lhs_type).unwrap();
    let rhs_ir: xlsynth::IrPackage =
        crate::gate2ir::gate_fn_to_xlsynth_ir(rhs, "rhs", &rhs_type).unwrap();

    run_external_ir_tool(&lhs_ir.to_string(), &rhs_ir.to_string())
}

/// Backwards-compatibility wrapper that preserves the original
/// Result<(),String> behaviour used elsewhere in the codebase.
pub fn prove_same_gate_fn_via_ir(lhs: &gate::GateFn, rhs: &gate::GateFn) -> Result<(), String> {
    match prove_same_gate_fn_via_ir_status(lhs, rhs) {
        IrCheckResult::Equivalent => Ok(()),
        IrCheckResult::NotEquivalent => Err("not equivalent".to_string()),
        IrCheckResult::TimedOutOrInterrupted => Err("TimedOutOrInterrupted".to_string()),
        IrCheckResult::OtherProcessError(msg) => Err(msg),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gate::AigBitVector,
        gate_builder::{GateBuilder, GateBuilderOptions},
        xls_ir::ir_parser,
    };

    use super::*;

    #[test]
    fn test_validate_same_signature_simple_one_bit() {
        let simple_xor_ir = "package simple_xor
 top fn my_xor(a: bits[1], b: bits[1]) -> bits[1] {
     ret xor.3: bits[1] = xor(a, b, id=3)
 }
";
        let mut parser = ir_parser::Parser::new(simple_xor_ir);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_top = ir_package.get_top().unwrap();

        // Now we make a simple one bit gate fn.
        let mut gate_builder = GateBuilder::new("my_xor".to_string(), GateBuilderOptions::opt());
        let a = gate_builder
            .add_input("a".to_string(), 1)
            .get_lsb(0)
            .clone();
        let b = gate_builder
            .add_input("b".to_string(), 1)
            .get_lsb(0)
            .clone();
        let xor = gate_builder.add_xor_binary(a, b);
        gate_builder.add_output("output_value".to_string(), AigBitVector::from_bit(xor));
        let gate_fn = gate_builder.build();

        validate_same_signature(&ir_top, &gate_fn).unwrap();
    }
}
