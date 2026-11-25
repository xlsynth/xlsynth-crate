// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use xlsynth_bx::{ResetBehavior, compile, empty_package};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog_with_register_names};
use xlsynth_pir::ir::NodePayload;

#[test]
fn register_can_share_an_output_port_name() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Output { pop_valid: bool }

pub block flow_reg(clk, push_valid: bool) -> Output {
    reg pop_valid: bool = false;
    pop_valid := push_valid;
    Output { pop_valid }
}
"#;
    let package = compile(source, empty_package("flow"), ResetBehavior::default())?;
    let block = package.get_block("flow_reg").unwrap();
    let output = block.output_ports().next().unwrap();
    assert_eq!(block.port_name(output), "pop_valid");
    assert!(matches!(
        &block.get_node(block.output_value(output)).payload,
        NodePayload::RegisterRead { register } if register == "pop_valid"
    ));

    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let rtl = emit_system_verilog_with_register_names(&package, &BlockCodegenOptions::default())?;
    assert_eq!(rtl.top_register_names["pop_valid"], "pop_valid__1");
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
