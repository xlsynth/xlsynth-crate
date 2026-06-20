// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::liberty_model::{Cell, Library, LibraryBuilder, Pin, PinDirection};
use xlsynth_g8r::netlist::integrity::{IntegrityFinding, IntegritySummary, check_module};
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

fn build_simple_lib() -> Library {
    let mut builder = LibraryBuilder::new();
    let a = builder.intern_string("A").unwrap();
    let y = builder.intern_string("Y").unwrap();
    let empty = builder.intern_string("").unwrap();
    builder.cells = vec![Cell {
        name: "INV".to_string(),
        area: 1.0,
        pins: vec![
            Pin {
                name: a,
                direction: PinDirection::Input as i32,
                function: empty,
                is_clocking_pin: false,
                ..Default::default()
            },
            Pin {
                name: y,
                direction: PinDirection::Output as i32,
                function: empty,
                is_clocking_pin: false,
                ..Default::default()
            },
        ],
        sequential: vec![],
        clock_gate: None,
        ..Default::default()
    }];
    builder.finish()
}

#[test]
fn test_clean_netlist() {
    let netlist = r#"module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u1 (.A(a), .Y(y));
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
  INV u1 (.A(y), .Y(y));
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

#[test]
fn test_assign_driven_netlist_is_clean() {
    let netlist = r#"module top (a, y);
  assign y = a;
  input a;
  output y;
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
fn test_assign_chain_marks_internal_wire_driven_and_used() {
    let netlist = r#"module top (a, y);
  input a;
  output y;
  wire n;
  assign n = a;
  assign y = n;
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
fn test_partial_assign_output_reports_undriven_output() {
    let netlist = r#"module top (a, y);
  input a;
  output [3:0] y;
  assign y[0] = a;
endmodule"#;
    let scanner = TokenScanner::with_line_lookup(netlist.as_bytes(), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let lib = build_simple_lib();
    let summary = check_module(&modules[0], &parser.nets, &parser.interner, &lib);
    match summary {
        IntegritySummary::Findings(f) => {
            assert!(f.contains(&IntegrityFinding::UndrivenOutput("y".into())));
        }
        _ => panic!("expected findings"),
    }
}

#[test]
fn test_self_referential_assign_does_not_count_as_driven() {
    let netlist = r#"module top (y);
  output y;
  assign y = y;
endmodule"#;
    let scanner = TokenScanner::with_line_lookup(netlist.as_bytes(), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let lib = build_simple_lib();
    let summary = check_module(&modules[0], &parser.nets, &parser.interner, &lib);
    match summary {
        IntegritySummary::Findings(f) => {
            assert!(f.contains(&IntegrityFinding::UndrivenOutput("y".into())));
        }
        _ => panic!("expected findings"),
    }
}
