// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module_with_defines;

#[test]
fn ifdef_simulation_includes_observer_when_defined() {
    let sv = concat!(
        "module m(input logic clk, input logic a, output wire y);\n",
        "  assign y = a;\n",
        "`ifdef SIMULATION\n",
        "  always_ff @ (posedge clk) begin\n",
        "    if (a) begin\n",
        "      $display(\"a_is_true: %d\", a);\n",
        "    end\n",
        "  end\n",
        "`endif\n",
        "endmodule\n",
    );
    let mut defs: BTreeSet<String> = BTreeSet::new();
    defs.insert("SIMULATION".to_string());
    let m = compile_pipeline_module_with_defines(sv, &defs).unwrap();
    assert_eq!(m.observers.len(), 1);
}

#[test]
fn ifdef_simulation_excludes_observer_when_not_defined() {
    let sv = concat!(
        "module m(input logic clk, input logic a, output wire y);\n",
        "  assign y = a;\n",
        "`ifdef SIMULATION\n",
        "  always_ff @ (posedge clk) begin\n",
        "    if (a) begin\n",
        "      $display(\"a_is_true: %d\", a);\n",
        "    end\n",
        "  end\n",
        "`endif\n",
        "endmodule\n",
    );
    let defs: BTreeSet<String> = BTreeSet::new();
    let m = compile_pipeline_module_with_defines(sv, &defs).unwrap();
    assert_eq!(m.observers.len(), 0);
}

#[test]
fn multiple_observer_always_ff_blocks_are_supported() {
    let sv = concat!(
        "module m(input logic clk, input logic a, output wire y);\n",
        "  assign y = a;\n",
        "`ifdef SIMULATION\n",
        "  always_ff @ (posedge clk) begin if (a) begin $display(\"one: %d\", a); end end\n",
        "  always_ff @ (posedge clk) begin if (~a) begin $display(\"two: %d\", a); end end\n",
        "`endif\n",
        "endmodule\n",
    );
    let mut defs: BTreeSet<String> = BTreeSet::new();
    defs.insert("SIMULATION".to_string());
    let m = compile_pipeline_module_with_defines(sv, &defs).unwrap();
    assert_eq!(m.observers.len(), 2);
}

#[test]
fn observer_else_branch_executes_when_if_condition_is_x() {
    let sv = concat!(
        "module m(input logic clk, input logic a, output wire y);\n",
        "  assign y = a;\n",
        "`ifdef SIMULATION\n",
        "  always_ff @ (posedge clk) begin\n",
        "    if (a) begin\n",
        "      $display(\"then\");\n",
        "    end else begin\n",
        "      $display(\"else\");\n",
        "    end\n",
        "  end\n",
        "`endif\n",
        "endmodule\n",
    );
    let mut defs: BTreeSet<String> = BTreeSet::new();
    defs.insert("SIMULATION".to_string());
    let m = compile_pipeline_module_with_defines(sv, &defs).unwrap();
    assert_eq!(m.observers.len(), 2);

    let mut env = Env::new();
    env.insert("a", Value4::new(1, Signedness::Unsigned, vec![LogicBit::X]));

    let lines: Vec<String> = m
        .observers
        .iter()
        .filter_map(|obs| obs.eval_and_format(&env).unwrap())
        .collect();
    assert_eq!(lines, vec!["else".to_string()]);
}
