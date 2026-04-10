// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::netlist::gv2ir::convert_gv2ir_paths;

const LIBERTY_TEXTPROTO: &str = r#"
cells: {
  name: "BUF"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "A" }
  area: 1.0
}
"#;

#[test]
fn test_gv2ir_rejects_preserved_assigns() {
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  assign y = a & a;
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2ir_paths(netlist_file.path(), liberty_file.path(), true)
        .expect_err("expected preserved assigns to be rejected");
    assert!(
        err.to_string()
            .contains("gv2ir does not support preserved continuous assigns")
    );
}
