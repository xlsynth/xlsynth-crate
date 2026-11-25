// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::path::Path;
use std::process::Command;

use xlsynth_bx::{ResetBehavior, compile, empty_package};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog_with_register_names};
use xlsynth_pir::ir::NodePayload;

const FLOW_REG: &str = r#"
struct Output { pop_valid: bool }

pub block flow_reg(clk, push_valid: bool) -> Output {
    reg pop_valid: bool = false;
    pop_valid := push_valid;
    Output { pop_valid }
}
"#;

#[test]
fn register_can_share_an_output_port_name() -> Result<(), Box<dyn Error>> {
    let package = compile(FLOW_REG, empty_package("flow"), ResetBehavior::default())?;
    let block = package.get_block("flow_reg").unwrap();
    let output = block.output_ports().next().unwrap();
    assert_eq!(block.port_name(output), "pop_valid");
    assert!(matches!(
        &block.get_node(block.output_value(output)).payload,
        NodePayload::RegisterRead { register } if register == "pop_valid__1"
    ));
    let input = block.get_input_port("push_valid").unwrap();
    assert!(block.nodes.iter().any(|node| matches!(
        &node.payload,
        NodePayload::RegisterWrite { register, arg, .. }
            if register == "pop_valid__1" && *arg == input
    )));

    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let rtl = emit_system_verilog_with_register_names(&package, &BlockCodegenOptions::default())?;
    assert_eq!(rtl.top_register_names["pop_valid__1"], "pop_valid__1");
    Ok(())
}

// Parsing alone misses the name collision. CI supplies the XLS Verilog emitter.
#[test]
fn xls_emits_verilog_for_shared_register_and_port_name() -> Result<(), Box<dyn Error>> {
    let Some(tools) = std::env::var_os("XLSYNTH_TOOLS") else {
        return Ok(());
    };
    let package = compile(FLOW_REG, empty_package("flow"), ResetBehavior::default())?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("flow.ir");
    std::fs::write(&path, package.to_string())?;
    let result = Command::new(Path::new(&tools).join("block_to_verilog_main"))
        .arg(path)
        .output()?;
    assert!(
        result.status.success(),
        "block_to_verilog_main failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

#[test]
fn register_suffix_skips_later_registers_ports_and_guarded_labels() -> Result<(), String> {
    let source = r#"
struct Output { valid: bool, valid__2: bool }
pub block flow(clk, x: bool) -> Output {
    reg valid: bool;
    reg valid__1: bool;
    valid := x;
    valid__1 := valid;
    if x { assert!(valid, "valid__3"); }
    Output { valid, valid__2: valid__1 }
}
"#;
    let package = compile(source, empty_package("flow"), ResetBehavior::default())?;
    let block = package.get_block("flow").unwrap();
    assert_eq!(block.registers[0].name, "valid__4");
    assert_eq!(block.registers[1].name, "valid__1");
    Ok(())
}

#[test]
fn input_and_output_port_names_must_be_distinct() {
    let source = r#"
struct Output { valid: bool }
pub block flow(valid: bool) -> Output { Output { valid } }
"#;
    let error = compile(source, empty_package("flow"), ResetBehavior::default()).unwrap_err();
    assert_eq!(error, "block 'flow': duplicate name 'valid'");
}
