// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::netlist::gv2ir::{
    Gv2IrOptions, convert_gv2ir_paths, convert_gv2ir_paths_with_options,
};

const LIBERTY_TEXTPROTO: &str = r#"
cells: {
  name: "BUF"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "A" }
  area: 1.0
}
"#;

#[test]
fn test_gv2ir_rejects_default_top_function_name() {
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  BUF u0 (.A(a), .Y(y));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2ir_paths(netlist_file.path(), liberty_file.path(), true)
        .expect_err("expected default top function name to be rejected");
    assert!(
        err.to_string()
            .contains("gv2ir would emit XLS IR function name 'top'")
    );
    assert!(err.to_string().contains("--output_function_name"));
}

#[test]
fn test_gv2ir_output_function_name_override_supports_top_module_with_assigns() {
    let netlist = r#"
module top (a, y);
  input [1:0] a;
  output y;
  wire [1:0] a;
  wire y;
  wire [1:0] tmp;
  assign tmp = {a[0], a[1]};
  BUF u0 (.A(tmp[1]), .Y(y));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let ir = convert_gv2ir_paths_with_options(
        netlist_file.path(),
        liberty_file.path(),
        &Gv2IrOptions {
            module_name: None,
            collapse_sequential: true,
            output_function_name: Some("top_fn".to_string()),
        },
    )
    .expect("override should allow a top-named netlist module");
    assert!(
        ir.contains("fn top_fn("),
        "expected overridden function name in IR:\n{}",
        ir
    );
}

#[test]
fn test_gv2ir_rejects_invalid_output_function_name_override() {
    let netlist = r#"
module not_top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  BUF u0 (.A(a), .Y(y));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2ir_paths_with_options(
        netlist_file.path(),
        liberty_file.path(),
        &Gv2IrOptions {
            module_name: None,
            collapse_sequential: true,
            output_function_name: Some("bad-name".to_string()),
        },
    )
    .expect_err("invalid function name override should be rejected");
    assert!(
        err.to_string()
            .contains("invalid XLS IR function name 'bad-name'")
    );
}
