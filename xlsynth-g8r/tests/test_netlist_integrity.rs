// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::liberty_proto::{Cell, Library, Pin, PinDirection};
use xlsynth_g8r::netlist::integrity::{IntegrityFinding, IntegritySummary, check_module};
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

fn build_simple_lib() -> Library {
    Library {
        cells: vec![Cell {
            name: "INVX1".to_string(),
            area: 1.0,
            pins: vec![
                Pin {
                    name: "A".to_string(),
                    direction: PinDirection::Input as i32,
                    function: String::new(),
                },
                Pin {
                    name: "Y".to_string(),
                    direction: PinDirection::Output as i32,
                    function: String::new(),
                },
            ],
        }],
    }
}

#[test]
fn test_clean_netlist() {
    let netlist = r#"module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INVX1 u1 (.A(a), .Y(y));
endmodule"#;
    let scanner = TokenScanner::with_line_lookup(netlist.as_bytes(), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let lib = build_simple_lib();
    let summary = check_module(&modules[0], &parser.nets, &parser.interner, &lib);
    assert!(matches!(summary, IntegritySummary::Clean));
}

#[test]
fn test_findings_netlist() {
    let netlist = r#"module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n1;
  INVX1 u1 (.A(y), .Y(y));
endmodule"#;
    let scanner = TokenScanner::with_line_lookup(netlist.as_bytes(), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let lib = build_simple_lib();
    let summary = check_module(&modules[0], &parser.nets, &parser.interner, &lib);
    match summary {
        IntegritySummary::Findings(f) => {
            assert!(f.contains(&IntegrityFinding::UnusedInput("a".into())));
            assert!(f.contains(&IntegrityFinding::UnusedWire("n1".into())));
        }
        _ => panic!("expected findings"),
    }
}
