// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use crate::{gate, gate2ir, xls_ir::ir};

pub fn check_equivalence(orig_package: &str, gate_package: &str) -> Result<(), String> {
    let tempdir = tempfile::tempdir().unwrap();
    let temp_path = tempdir.into_path(); // This prevents auto-deletion
    let orig_path = temp_path.join("orig.ir");
    let gate_path = temp_path.join("gate.ir");
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
    command.arg("--alsologtostderr");
    command.arg(orig_path.to_str().unwrap());
    command.arg(gate_path.to_str().unwrap());
    log::info!("check_equivalence; running command: {:?}", command);
    let start = Instant::now();
    let output = command.output().unwrap();
    let elapsed = start.elapsed();
    if !output.status.success() {
        return Err(format!(
            "check_ir_equivalence_main failed with retcode {}\nstdout: {:?}\nstderr: {:?}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    log::info!("check_equivalence; successful in {:?}", elapsed);
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

pub fn validate_same_gate_fn(lhs: &gate::GateFn, rhs: &gate::GateFn) -> Result<(), String> {
    let lhs_type = lhs.get_flat_type();
    let rhs_type = rhs.get_flat_type();
    if lhs_type != rhs_type {
        return Err(format!("type mismatch: {:?} != {:?}", lhs_type, rhs_type));
    }
    let lhs_ir_fn_text: xlsynth::IrPackage =
        gate2ir::gate_fn_to_xlsynth_ir(lhs, "lhs", &lhs_type).unwrap();
    let rhs_ir_fn_text: xlsynth::IrPackage =
        gate2ir::gate_fn_to_xlsynth_ir(rhs, "rhs", &rhs_type).unwrap();
    let lhs_ir_pkg_text: String = lhs_ir_fn_text.to_string();
    let rhs_ir_pkg_text: String = rhs_ir_fn_text.to_string();
    let result = check_equivalence(&lhs_ir_pkg_text, &rhs_ir_pkg_text);
    result
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
        let ir_package = parser.parse_package().unwrap();
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
