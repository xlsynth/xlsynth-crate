// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::liberty::load::load_library_with_timing_data_from_path;
use xlsynth_g8r::netlist::io::parse_netlist_from_path;
use xlsynth_g8r::netlist::sta::{StaOptions, analyze_combinational_max_arrival};

const SYNTHETIC_TIMING_LIBRARY: &str = r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT capacitance: 1.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!A"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 2.0 }
      tables: { kind: "cell_fall" values: 3.0 }
      tables: { kind: "rise_transition" values: 0.2 }
      tables: { kind: "fall_transition" values: 0.3 }
    }
  }
}
cells: {
  name: "NAND2"
  pins: { name: "A" direction: INPUT capacitance: 2.0 }
  pins: { name: "B" direction: INPUT capacitance: 2.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!(A*B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 4.0 }
      tables: { kind: "cell_fall" values: 5.0 }
      tables: { kind: "rise_transition" values: 0.4 }
      tables: { kind: "fall_transition" values: 0.5 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 4.0 }
      tables: { kind: "cell_fall" values: 5.0 }
      tables: { kind: "rise_transition" values: 0.4 }
      tables: { kind: "fall_transition" values: 0.5 }
    }
  }
}
cells: {
  name: "NOR2"
  pins: { name: "A" direction: INPUT capacitance: 2.0 }
  pins: { name: "B" direction: INPUT capacitance: 2.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "!(A+B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 6.0 }
      tables: { kind: "cell_fall" values: 7.0 }
      tables: { kind: "rise_transition" values: 0.6 }
      tables: { kind: "fall_transition" values: 0.7 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "negative_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 6.0 }
      tables: { kind: "cell_fall" values: 7.0 }
      tables: { kind: "rise_transition" values: 0.6 }
      tables: { kind: "fall_transition" values: 0.7 }
    }
  }
}
cells: {
  name: "XOR2"
  pins: { name: "A" direction: INPUT capacitance: 2.0 }
  pins: { name: "B" direction: INPUT capacitance: 2.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "(A*!B)+(!A*B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "non_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 8.0 }
      tables: { kind: "cell_fall" values: 9.0 }
      tables: { kind: "rise_transition" values: 0.8 }
      tables: { kind: "fall_transition" values: 0.9 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "non_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 8.0 }
      tables: { kind: "cell_fall" values: 9.0 }
      tables: { kind: "rise_transition" values: 0.8 }
      tables: { kind: "fall_transition" values: 0.9 }
    }
  }
}
cells: {
  name: "XNOR2"
  pins: { name: "A" direction: INPUT capacitance: 2.0 }
  pins: { name: "B" direction: INPUT capacitance: 2.0 }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "(A*B)+(!A*!B)"
    timing_arcs: {
      related_pin: "A"
      timing_sense: "non_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 10.0 }
      tables: { kind: "cell_fall" values: 11.0 }
      tables: { kind: "rise_transition" values: 1.0 }
      tables: { kind: "fall_transition" values: 1.1 }
    }
    timing_arcs: {
      related_pin: "B"
      timing_sense: "non_unate"
      timing_type: "combinational"
      tables: { kind: "cell_rise" values: 10.0 }
      tables: { kind: "cell_fall" values: 11.0 }
      tables: { kind: "rise_transition" values: 1.0 }
      tables: { kind: "fall_transition" values: 1.1 }
    }
  }
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
"#;

struct StaCase {
    name: &'static str,
    netlist: &'static str,
    expected_worst_arrival: f64,
}

fn write_temp_file(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(contents.as_bytes())
        .expect("write temp file");
    file
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual={} expected={}",
        actual,
        expected
    );
}

#[test]
fn sta_small_synthetic_cases_match_expected_arrivals() {
    let liberty_file = write_temp_file(SYNTHETIC_TIMING_LIBRARY);
    let library = load_library_with_timing_data_from_path(liberty_file.path())
        .expect("load synthetic timing library");
    let cases = [
        StaCase {
            name: "inv_chain2",
            netlist: r#"
module inv_chain2 (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  INV u0 ( .A(a), .Y(n0) );
  INV u1 ( .A(n0), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 5.0,
        },
        StaCase {
            name: "reconvergent_nand",
            netlist: r#"
module reconvergent_nand (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  wire n1;
  INV u0 ( .A(a), .Y(n0) );
  INV u1 ( .A(n0), .Y(n1) );
  NAND2 u2 ( .A(n1), .B(b), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 10.0,
        },
        StaCase {
            name: "fanout_load",
            netlist: r#"
module fanout_load (a, b, c, d, y0, y1, y2);
  input a;
  input b;
  input c;
  input d;
  output y0;
  output y1;
  output y2;
  wire a;
  wire b;
  wire c;
  wire d;
  wire y0;
  wire y1;
  wire y2;
  wire n0;
  INV u0 ( .A(a), .Y(n0) );
  NAND2 u1 ( .A(n0), .B(b), .Y(y0) );
  NAND2 u2 ( .A(n0), .B(c), .Y(y1) );
  NAND2 u3 ( .A(n0), .B(d), .Y(y2) );
endmodule
"#,
            expected_worst_arrival: 7.0,
        },
        StaCase {
            name: "single_xor",
            netlist: r#"
module single_xor (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  XOR2 u0 ( .A(a), .B(b), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 9.0,
        },
        StaCase {
            name: "single_xnor",
            netlist: r#"
module single_xnor (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  XNOR2 u0 ( .A(a), .B(b), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 11.0,
        },
        StaCase {
            name: "xor_chain",
            netlist: r#"
module xor_chain (a, b, c, y);
  input a;
  input b;
  input c;
  output y;
  wire a;
  wire b;
  wire c;
  wire y;
  wire n0;
  XOR2 u0 ( .A(a), .B(b), .Y(n0) );
  XOR2 u1 ( .A(n0), .B(c), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 18.0,
        },
        StaCase {
            name: "mixed_logic",
            netlist: r#"
module mixed_logic (a, b, c, d, y);
  input a;
  input b;
  input c;
  input d;
  output y;
  wire a;
  wire b;
  wire c;
  wire d;
  wire y;
  wire n0;
  wire n1;
  wire n2;
  NAND2 u0 ( .A(a), .B(b), .Y(n0) );
  NOR2  u1 ( .A(c), .B(d), .Y(n1) );
  XNOR2 u2 ( .A(n0), .B(n1), .Y(n2) );
  INV   u3 ( .A(n2), .Y(y) );
endmodule
"#,
            expected_worst_arrival: 20.0,
        },
    ];

    for case in cases {
        let netlist_file = write_temp_file(case.netlist);
        let parsed = parse_netlist_from_path(netlist_file.path())
            .unwrap_or_else(|e| panic!("parse case '{}': {e}", case.name));
        let module = parsed
            .modules
            .iter()
            .find(|m| parsed.interner.resolve(m.name) == Some(case.name))
            .unwrap_or_else(|| panic!("missing module '{}'", case.name));
        let report = analyze_combinational_max_arrival(
            module,
            parsed.nets.as_slice(),
            &parsed.interner,
            &library,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .unwrap_or_else(|e| panic!("sta failed for '{}': {e}", case.name));
        assert_close(report.worst_output_arrival, case.expected_worst_arrival);
    }
}

#[test]
fn sta_uses_selected_module_net_scope_when_names_repeat() {
    let liberty_file = write_temp_file(SYNTHETIC_TIMING_LIBRARY);
    let library = load_library_with_timing_data_from_path(liberty_file.path())
        .expect("load synthetic timing library");
    let netlist_file = write_temp_file(
        r#"
module fast (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 ( .A(a), .Y(y) );
endmodule

module slow (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  INV u0 ( .A(a), .Y(n0) );
  INV u1 ( .A(n0), .Y(y) );
endmodule
"#,
    );
    let parsed = parse_netlist_from_path(netlist_file.path()).expect("parse netlist");
    let module = parsed
        .modules
        .iter()
        .find(|m| parsed.interner.resolve(m.name) == Some("fast"))
        .expect("find fast module");
    let report = analyze_combinational_max_arrival(
        module,
        parsed.nets.as_slice(),
        &parsed.interner,
        &library,
        StaOptions {
            primary_input_transition: 0.0,
            module_output_load: 0.0,
        },
    )
    .expect("sta should succeed");

    assert_close(report.worst_output_arrival, 3.0);
}
