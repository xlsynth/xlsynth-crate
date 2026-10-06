// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::netlist::integrity::validate_structural_assign_module;
use xlsynth_g8r::netlist::io::{read_gv_from_str, read_ugv_from_str};

const TRAN_ERROR: &str = "UGV does not support bidirectional tran connections; use continuous assignments for directional wiring";

#[test]
fn ugv_reader_rejects_tran_in_both_terminal_orders() {
    for terminals in ["y, a", "a, y"] {
        let source = format!(
            r#"module top(a, y);
  input a;
  output y;
  tran({terminals});
endmodule
"#
        );
        let error = read_ugv_from_str(&source)
            .err()
            .expect("UGV must reject bidirectional tran connections");
        assert_eq!(
            error.to_string(),
            format!(
                r#"UGV reader: {TRAN_ERROR} @ <string>:4:3..4:14
  tran({terminals});
  ^
UGV supports combinational Boolean assignments; procedural registers and leaf cells are unsupported."#
            )
        );
    }
}

#[test]
fn structural_validator_rejects_tran_after_gv_parsing() {
    for terminals in ["y, a", "a, y"] {
        let source = format!(
            r#"module top(a, y);
  input a;
  output y;
  tran({terminals});
endmodule
"#
        );
        let parsed = read_gv_from_str(&source).expect("GV must continue to accept tran");
        let error =
            validate_structural_assign_module(&parsed.modules[0], &parsed.nets, &parsed.interner)
                .expect_err("the public structural validator must reject tran");
        assert_eq!(
            error.to_string(),
            "UGV does not support bidirectional tran connections"
        );
    }
}
