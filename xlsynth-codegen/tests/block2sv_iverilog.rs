// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
#[allow(dead_code)]
#[path = "support/block2sv_goldens.rs"]
mod fixtures;
use fixtures::*;
use std::fs;
use std::path::Path;
use std::time::Duration;
use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;

/// Compiles every successful fixture with a second SystemVerilog frontend.
#[test]
fn block2sv_generated_fixtures_compile_with_iverilog() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let mut paths = Vec::new();
    collect_golden_fixtures(&directory, &mut paths).unwrap();
    paths.sort();

    let temporary_directory = tempfile::tempdir().expect("create iverilog input directory");
    let verilog_path = temporary_directory.path().join("generated.sv");
    let support_types =
        "package types; typedef logic [7:0] input_t; typedef logic [7:0] output_t; endpackage\n";
    let mut failures = Vec::new();
    for path in paths {
        let contents = fs::read_to_string(&path).unwrap();
        let fixture = parse_golden_fixture(&path, &contents).unwrap();
        if fixture.expects_error() {
            continue;
        }
        let output = match execute_golden_fixture(&fixture) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        fs::write(&verilog_path, format!("{support_types}{output}"))
            .expect("write generated SystemVerilog");
        if let Err(error) = tools.check_syntax(
            temporary_directory.path(),
            &[&verilog_path],
            &["SYNTHESIS"],
            Duration::from_secs(60),
        ) {
            failures.push(format!("{}: {error}", path.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} generated SystemVerilog fixtures failed to compile:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Simulates packed ports across flat connections, hierarchy, and custom types.
#[test]
fn packed_array_ports_connect_to_flat_vectors() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let cases = [
        (
            "arrays/multidimensional_update.golden.ir",
            r#"
module tb;
  logic [47:0] data = 48'h665544332211;
  logic [1:0] row = 0, column = 0;
  logic [7:0] value = 8'haa;
  logic [47:0] out;
  nested_update dut(.*);
  initial begin
    #1; if (out !== 48'h6655443322aa) $fatal(1, "element zero");
    row = 1; column = 2;
    #1; if (out !== 48'haa5544332211) $fatal(1, "last element");
    row = 3;
    #1; if (out !== data) $fatal(1, "invalid row");
    row = 0; column = 3;
    #1; if (out !== data) $fatal(1, "invalid column");
    $finish;
  end
endmodule
"#,
        ),
        (
            "hierarchy/packed_array_ports.golden.ir",
            r#"
module tb;
  logic [15:0] data = 16'h2211;
  logic [1:0] index = 0;
  logic [7:0] out;
  array_parent dut(.*);
  initial begin
    #1; if (out !== 8'h11) $fatal(1, "child element zero");
    index = 1;
    #1; if (out !== 8'h22) $fatal(1, "child element one");
    index = 3;
    #1; if (out !== 8'h22) $fatal(1, "clamped child output");
    $finish;
  end
endmodule
"#,
        ),
        (
            "arrays/custom_typed_array.golden.ir",
            r#"
module tb;
  logic [7:0] data = 8'h21;
  logic [1:0] index = 0;
  logic [3:0] replacement = 4'hf;
  logic [7:0] out;
  custom_typed_array dut(.*);
  initial begin
    #1; if (out !== 8'h2f) $fatal(1, "typed element zero");
    index = 1;
    #1; if (out !== 8'hf1) $fatal(1, "typed element one");
    index = 2;
    #1; if (out !== data) $fatal(1, "typed invalid index");
    $finish;
  end
endmodule
"#,
        ),
    ];
    let temporary = tempfile::tempdir().unwrap();
    let source_path = temporary.path().join("packed.sv");
    for (relative, testbench) in cases {
        let path = directory.join(relative);
        let fixture = parse_golden_fixture(&path, &fs::read_to_string(&path).unwrap()).unwrap();
        let generated = execute_golden_fixture(&fixture).unwrap();
        let support = "package types; typedef logic [7:0] input_t; typedef logic [7:0] output_t; endpackage\n";
        fs::write(&source_path, format!("{support}{generated}{testbench}")).unwrap();
        let binary_path = tools
            .compile(
                source_path.parent().unwrap(),
                &[&source_path],
                "tb",
                &[],
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        tools
            .run(
                source_path.parent().unwrap(),
                &binary_path,
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
    }
}

/// Confirms both asynchronous reset polarities update state before a clock
/// edge.
#[test]
fn block2sv_asynchronous_resets_activate_between_clock_edges() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let temporary_directory = tempfile::tempdir().expect("create asynchronous reset testbench");
    let source_path = temporary_directory.path().join("asynchronous_reset.sv");
    let cases = [
        (
            "registers/asynchronous_active_high_reset.golden.ir",
            "asynchronous_high",
            "rst",
            8,
            "1'b1",
            "1'b0",
            "8'h07",
        ),
        (
            "registers/asynchronous_active_low_reset.golden.ir",
            "asynchronous_low",
            "rst_n",
            16,
            "1'b0",
            "1'b1",
            "16'h00ff",
        ),
    ];

    for (relative, module, reset_port, width, asserted, released, reset_value) in cases {
        let path = directory.join(relative);
        let text = fs::read_to_string(&path).expect("read asynchronous-reset fixture");
        let fixture = parse_golden_fixture(&path, &text).expect("parse asynchronous-reset fixture");
        let generated = execute_golden_fixture(&fixture).expect("generate asynchronous-reset RTL");
        let testbench = format!(
            r#"module asynchronous_reset_tb;
  logic clock;
  logic reset;
  logic [{msb}:0] data;
  wire [{msb}:0] result;
  {module} dut(.clk(clock), .{reset_port}(reset), .data(data), .out(result));

  initial begin
    clock = 1'b0;
    reset = {released};
    data = '1;
    #1;
    clock = 1'b1;
    #1;
    if (result !== '1) $fatal(1, "failed to capture a non-reset value");
    clock = 1'b0;
    #1;
    reset = {asserted};
    #1;
    if (result !== {reset_value}) $fatal(1, "reset did not take effect between clock edges");
    data = '0;
    #1;
    if (result !== {reset_value}) $fatal(1, "reset value was not held");
    reset = {released};
    #1;
    if (result !== {reset_value}) $fatal(1, "reset release changed state without a clock");
    clock = 1'b1;
    #1;
    if (result !== '0) $fatal(1, "normal capture did not resume after reset");
    $finish;
  end
endmodule
"#,
            msb = width - 1
        );
        fs::write(&source_path, format!("{generated}\n{testbench}"))
            .expect("write asynchronous-reset testbench");
        let binary_path = tools
            .compile(
                source_path.parent().unwrap(),
                &[&source_path],
                "asynchronous_reset_tb",
                &["SYNTHESIS"],
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        tools
            .run(
                source_path.parent().unwrap(),
                &binary_path,
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
    }
}

/// Checks extension operators against the independently implemented PIR
/// evaluator.
#[test]
fn block2sv_extension_operators_match_pir_evaluation() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv/arithmetic");
    let examples = [
        "extended_carry_out.golden.ir",
        "extended_priority_encode.golden.ir",
        "extended_leading_zeros.golden.ir",
        "extended_normalize.golden.ir",
        "extended_nary_add.golden.ir",
    ];
    let temporary_directory = tempfile::tempdir().expect("create extension simulation directory");
    let source_path = temporary_directory.path().join("extension_equivalence.sv");

    for relative_path in examples {
        let path = directory.join(relative_path);
        let text = fs::read_to_string(&path).expect("read extension fixture");
        let fixture = parse_golden_fixture(&path, &text).expect("parse extension fixture");
        let package = Parser::new(&fixture.source)
            .parse_package()
            .expect("parse extension package");
        let Some(PackageMember::Block { func, metadata }) = package.get_top_block() else {
            panic!("extension fixture has no top block");
        };
        assert_eq!(metadata.output_names.len(), 1);
        let generated = execute_golden_fixture(&fixture).expect("generate extension RTL");

        let output_width = func.ret_ty.bit_count();
        let mut testbench = format!(
            "module extension_tb;\n  wire [{}:0] actual;\n",
            output_width - 1
        );
        let mut connections = Vec::new();
        for parameter in &func.params {
            let width = parameter.ty.bit_count();
            testbench.push_str(&format!(
                "  logic [{}:0] stimulus_{};\n",
                width - 1,
                parameter.name
            ));
            connections.push(format!(".{}(stimulus_{})", parameter.name, parameter.name));
        }
        connections.push(format!(".{}(actual)", metadata.output_names[0]));
        testbench.push_str(&format!(
            "  {} dut({});\n  initial begin\n",
            func.name,
            connections.join(", ")
        ));

        let vectors = if func.params.is_empty() { 1 } else { 260 };
        let mut state = 0x6d2b_79f5_u64;
        for vector in 0..vectors {
            let mut arguments = Vec::new();
            for (index, parameter) in func.params.iter().enumerate() {
                let width = parameter.ty.bit_count();
                assert!(width < 64, "extension fixture inputs must fit in u64");
                let mask = (1_u64 << width) - 1;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = if func.params.len() == 1 && vector < 256 {
                    (vector as u64) & mask
                } else {
                    match vector {
                        0 => 0,
                        1 => mask,
                        2 if index == 0 => 1_u64 << (width - 1),
                        _ => state & mask,
                    }
                };
                testbench.push_str(&format!(
                    "    stimulus_{} = {width}'h{value:x};\n",
                    parameter.name
                ));
                arguments.push(
                    xlsynth::IrValue::make_ubits(width, value)
                        .expect("construct extension evaluator input"),
                );
            }
            let result = match eval_fn_in_package(&package, func, &arguments) {
                FnEvalResult::Success(result) => result.value,
                FnEvalResult::Failure(failure) => panic!(
                    "extension evaluator rejected {relative_path} vector {vector}: {:?}",
                    failure.assertion_failures
                ),
            };
            let expected = flatten_reference_value(&result, &func.ret_ty);
            testbench.push_str(&format!(
                "    #1; if (actual !== {output_width}'h{expected:x}) $fatal(1, \"vector {vector}: actual=%h expected={expected:x}\", actual);\n"
            ));
        }
        testbench.push_str("    $finish;\n  end\nendmodule\n");
        fs::write(&source_path, format!("{generated}\n{testbench}"))
            .expect("write extension equivalence testbench");

        let binary_path = tools
            .compile(
                source_path.parent().unwrap(),
                &[&source_path],
                "extension_tb",
                &["SYNTHESIS"],
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        tools
            .run(
                source_path.parent().unwrap(),
                &binary_path,
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
    }
}

/// Checks hierarchical clock/reset wiring and child-aware pipeline latency.
#[test]
fn block2sv_hierarchy_preserves_child_latency_and_reset() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv");
    let temporary_directory =
        tempfile::tempdir().expect("create hierarchical simulation directory");
    let source_path = temporary_directory.path().join("hierarchy_equivalence.sv");
    let cases = [
        (
            "hierarchy/child_reset_and_clock.golden.ir",
            r#"module hierarchy_tb;
  logic clock;
  logic reset;
  logic [7:0] data;
  wire [7:0] result;
  reset_parent dut(.parent_clk(clock), .parent_rst(reset), .data(data), .out(result));

  initial begin
    clock = 1'b0;
    reset = 1'b1;
    data = 8'ha5;
    #1;
    clock = 1'b1;
    #1;
    if (result !== 8'h00) $fatal(1, "child did not receive parent reset");
    clock = 1'b0;
    reset = 1'b0;
    #1;
    if (result !== 8'h00) $fatal(1, "child state changed without a parent clock");
    clock = 1'b1;
    #1;
    if (result !== 8'ha5) $fatal(1, "child did not receive parent clock");
    clock = 1'b0;
    data = 8'h3c;
    #1;
    if (result !== 8'ha5) $fatal(1, "child changed between parent clock edges");
    clock = 1'b1;
    #1;
    if (result !== 8'h3c) $fatal(1, "child data connectivity is incorrect");
    clock = 1'b0;
    reset = 1'b1;
    #1;
    if (result !== 8'h3c) $fatal(1, "synchronous child reset acted asynchronously");
    clock = 1'b1;
    #1;
    if (result !== 8'h00) $fatal(1, "forwarded child reset did not clear state");
    $finish;
  end
endmodule
"#,
        ),
        (
            "hierarchy/pipelined_parent.golden.ir",
            r#"module hierarchy_tb;
  logic clock;
  logic [7:0] data;
  wire [7:0] result;
  pipelined_parent dut(.clk(clock), .data(data), .out(result));

  task tick(input [7:0] value);
    begin data = value; #1; clock = 1'b1; #1; clock = 1'b0; #1; end
  endtask

  initial begin
    clock = 1'b0;
    tick(8'h10);
    if (result !== 8'hxx) $fatal(1, "combinational child output arrived early");
    tick(8'h20);
    if (result !== 8'h11) $fatal(1, "combinational child latency is not two cycles");
    tick(8'h30);
    if (result !== 8'h21) $fatal(1, "combinational child pipeline dropped data");
    $finish;
  end
endmodule
"#,
        ),
        (
            "hierarchy/pipelined_stateful_child.golden.ir",
            r#"module hierarchy_tb;
  logic clock;
  logic [7:0] data;
  wire [7:0] result;
  stateful_pipeline_parent dut(.parent_clk(clock), .data(data), .out(result));

  task tick(input [7:0] value);
    begin data = value; #1; clock = 1'b1; #1; clock = 1'b0; #1; end
  endtask

  initial begin
    clock = 1'b0;
    tick(8'ha1);
    if (result !== 8'hxx) $fatal(1, "stateful child output arrived after one cycle");
    tick(8'hb2);
    if (result !== 8'hxx) $fatal(1, "stateful child output arrived after two cycles");
    tick(8'hc3);
    if (result !== 8'ha1) $fatal(1, "stateful child latency is not three cycles");
    tick(8'hd4);
    if (result !== 8'hb2) $fatal(1, "stateful child pipeline dropped data");
    data = 8'hee;
    #2;
    if (result !== 8'hb2) $fatal(1, "stateful child changed without a clock");
    $finish;
  end
endmodule
"#,
        ),
        (
            "hierarchy/registered_child_feedback.golden.ir",
            r#"module hierarchy_tb;
  logic clock;
  logic reset;
  logic [7:0] step;
  wire [7:0] result;
  registered_feedback dut(.parent_clk(clock), .parent_reset(reset), .step(step), .out(result));

  task tick;
    begin #1; clock = 1'b1; #1; clock = 1'b0; #1; end
  endtask

  initial begin
    clock = 1'b0;
    reset = 1'b1;
    step = 8'h03;
    tick();
    if (result !== 8'h00) $fatal(1, "registered feedback did not reset");
    reset = 1'b0;
    tick();
    if (result !== 8'h03) $fatal(1, "registered feedback did not advance");
    tick();
    if (result !== 8'h06) $fatal(1, "registered feedback did not retain state");
    step = 8'h01;
    tick();
    if (result !== 8'h07) $fatal(1, "registered feedback ignored updated input");
    reset = 1'b1;
    tick();
    if (result !== 8'h00) $fatal(1, "registered feedback reset lost priority");
    $finish;
  end
endmodule
"#,
        ),
        (
            "extern/pipelined_external.golden.ir",
            r#"module hierarchy_tb;
  logic clock;
  logic [7:0] data;
  wire [7:0] result;
  external_pipeline dut(.clk(clock), .data(data), .out(result));

  task tick(input [7:0] value);
    begin data = value; #1; clock = 1'b1; #1; clock = 1'b0; #1; end
  endtask

  initial begin
    clock = 1'b0;
    tick(8'h12);
    if (result !== 8'hxx) $fatal(1, "external output arrived early");
    tick(8'h34);
    if (result !== 8'hed) $fatal(1, "external combinational latency is not two cycles");
    tick(8'h56);
    if (result !== 8'hcb) $fatal(1, "external pipeline dropped data");
    $finish;
  end
endmodule
"#,
        ),
    ];

    for (relative_path, testbench) in cases {
        let path = directory.join(relative_path);
        let text = fs::read_to_string(&path).expect("read hierarchical fixture");
        let fixture = parse_golden_fixture(&path, &text).expect("parse hierarchical fixture");
        let generated = execute_golden_fixture(&fixture).expect("generate hierarchical RTL");
        fs::write(&source_path, format!("{generated}\n{testbench}"))
            .expect("write hierarchical testbench");
        let binary_path = tools
            .compile(
                source_path.parent().unwrap(),
                &[&source_path],
                "hierarchy_tb",
                &["SYNTHESIS"],
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        tools
            .run(
                source_path.parent().unwrap(),
                &binary_path,
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
    }
}

/// Confirms custom reset/enable templates preserve clocking and reset priority.
#[test]
fn block2sv_custom_register_template_preserves_reset_and_enable() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens/block2sv/registers/custom_reset_and_enable_template.golden.ir");
    let text = fs::read_to_string(&path).expect("read custom register fixture");
    let fixture = parse_golden_fixture(&path, &text).expect("parse custom register fixture");
    let generated = execute_golden_fixture(&fixture).expect("generate custom register RTL");
    let testbench = r#"module register_template_tb;
  logic clock;
  logic reset;
  logic enable;
  logic [7:0] data;
  wire [7:0] result;
  custom_reset_enable dut(.clk(clock), .reset(reset), .enable(enable), .data(data), .out(result));

  task tick;
    begin #1; clock = 1'b1; #1; clock = 1'b0; #1; end
  endtask

  initial begin
    clock = 1'b0;
    reset = 1'b1;
    enable = 1'b0;
    data = 8'ha5;
    tick();
    if (result !== 8'h05) $fatal(1, "custom reset value is incorrect");
    reset = 1'b0;
    tick();
    if (result !== 8'h05) $fatal(1, "disabled custom register changed");
    enable = 1'b1;
    tick();
    if (result !== 8'ha5) $fatal(1, "enabled custom register did not capture data");
    enable = 1'b0;
    data = 8'h3c;
    tick();
    if (result !== 8'ha5) $fatal(1, "disabled custom register lost its value");
    reset = 1'b1;
    enable = 1'b1;
    tick();
    if (result !== 8'h05) $fatal(1, "custom reset did not override enable");
    reset = 1'b0;
    tick();
    if (result !== 8'h3c) $fatal(1, "capture did not resume after custom reset");
    $finish;
  end
endmodule
"#;
    let directory = tempfile::tempdir().expect("create custom register testbench");
    let source_path = directory.path().join("custom_register.sv");
    fs::write(&source_path, format!("{generated}\n{testbench}"))
        .expect("write custom register testbench");
    let binary_path = tools
        .compile(
            source_path.parent().unwrap(),
            &[&source_path],
            "register_template_tb",
            &["SYNTHESIS"],
            Duration::from_secs(60),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
    tools
        .run(
            source_path.parent().unwrap(),
            &binary_path,
            Duration::from_secs(60),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
}

/// Compares actual simulated trace text with XLS format-directive semantics.
#[test]
fn block2sv_trace_formats_produce_expected_simulation_output() {
    let tools = xlsynth_test_helpers::iverilog::required_iverilog_toolchain()
        .expect("iverilog tests require iverilog and vvp");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/block2sv/debug");
    let temporary_directory = tempfile::tempdir().expect("create trace-format testbench");
    let source_path = temporary_directory.path().join("trace_formats.sv");
    let cases = [
        (
            "trace_formats.golden.ir",
            r#"module trace_tb;
  logic clock;
  logic flag;
  logic [7:0] data;
  logic [7:0] signed_value;
  wire [7:0] result;
  trace_formats dut(.clk(clock), .flag(flag), .data(data), .signed_value(signed_value), .out(result));

  initial begin
    clock = 1'b0;
    flag = 1'b0;
    data = 8'h2b;
    signed_value = 8'hfb;
    #1;
    clock = 1'b1;
    #1;
    clock = 1'b0;
    flag = 1'b1;
    #1;
    clock = 1'b1;
    #1;
    $finish(0);
  end
endmodule
"#,
            "plain=43 hex=2b padded=2b prefixed=0x2b binary=101011 padded_binary=00101011 prefixed_binary=0b00101011 signed=-5 unsigned=43",
        ),
        (
            "trace_escaped_literals.golden.ir",
            r#"module trace_tb;
  logic clock;
  logic flag;
  logic [11:0] data;
  wire [11:0] result;
  trace_escaped_literals dut(.clk(clock), .flag(flag), .data(data), .out(result));

  initial begin
    clock = 1'b0;
    flag = 1'b1;
    data = 12'h02b;
    #1;
    clock = 1'b1;
    #1;
    $finish(0);
  end
endmodule
"#,
            "progress 100% {value} = 02b",
        ),
    ];

    for (relative_path, testbench, expected) in cases {
        let path = directory.join(relative_path);
        let text = fs::read_to_string(&path).expect("read trace-format fixture");
        let fixture = parse_golden_fixture(&path, &text).expect("parse trace-format fixture");
        let generated = execute_golden_fixture(&fixture).expect("generate trace-format RTL");
        fs::write(&source_path, format!("{generated}\n{testbench}"))
            .expect("write trace-format testbench");
        let binary_path = tools
            .compile(
                source_path.parent().unwrap(),
                &[&source_path],
                "trace_tb",
                &[],
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        let simulation = tools
            .run(
                source_path.parent().unwrap(),
                &binary_path,
                Duration::from_secs(60),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
        assert_eq!(
            simulation,
            format!("{expected}\n"),
            "trace output differs for {relative_path}"
        );
    }
}
