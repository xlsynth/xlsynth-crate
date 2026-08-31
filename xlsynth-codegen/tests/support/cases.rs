// SPDX-License-Identifier: Apache-2.0

use super::support::TestRtl;
use std::collections::BTreeMap;
use xlsynth::IrBits;
use xlsynth_codegen::{BlockCodegenError, BlockCodegenOptions, Layout, emit_system_verilog};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_test_helpers::rtl_sim::LogicValue;

pub(super) const PASSTHROUGH: &str = r#"package public_passthrough

top block passthrough(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}
"#;

pub(super) const ADD: &str = r#"package public_add

top block add_bytes(lhs: bits[8], rhs: bits[8], result: bits[8]) {
  lhs: bits[8] = input_port(name=lhs, id=1)
  rhs: bits[8] = input_port(name=rhs, id=2)
  sum: bits[8] = add(lhs, rhs, id=3)
  result: () = output_port(sum, name=result, id=4)
}
"#;

pub(super) const PIPELINE: &str = r#"package public_pipeline

top block pipeline(clk: clock, data: bits[8], result: bits[8]) {
  reg first(bits[8])
  reg second(bits[8])
  data: bits[8] = input_port(name=data, id=1)
  first_write: () = register_write(data, register=first, id=2)
  first_read: bits[8] = register_read(register=first, id=3)
  one: bits[8] = literal(value=1, id=4)
  incremented: bits[8] = add(first_read, one, id=5)
  second_write: () = register_write(incremented, register=second, id=6)
  second_read: bits[8] = register_read(register=second, id=7)
  result: () = output_port(second_read, name=result, id=8)
}
"#;

pub(super) const HIERARCHY: &str = r#"package public_hierarchy

block child(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block parent(first: bits[8], second: bits[8], result: bits[8]) {
  instantiation left(block=child, kind=block)
  instantiation right(block=child, kind=block)
  first: bits[8] = input_port(name=first, id=3)
  second: bits[8] = input_port(name=second, id=4)
  left_input: () = instantiation_input(first, instantiation=left, port_name=value, id=5)
  right_input: () = instantiation_input(second, instantiation=right, port_name=value, id=6)
  left_value: bits[8] = instantiation_output(instantiation=left, port_name=result, id=7)
  right_value: bits[8] = instantiation_output(instantiation=right, port_name=result, id=8)
  combined: bits[8] = xor(left_value, right_value, id=9)
  result: () = output_port(combined, name=result, id=10)
}
"#;

pub(super) const EXTERNAL: &str = r#"package public_external

#[ffi_proto("""code_template: "external_cell {fn} (.source({value}), .destination({return}));"
""")]
fn external_identity(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}

top block external_wrapper(data: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=external_identity, kind=extern)
  data: bits[8] = input_port(name=data, id=2)
  connected: () = instantiation_input(data, instantiation=external_instance, port_name=value, id=3)
  received: bits[8] = instantiation_output(instantiation=external_instance, port_name=return, id=4)
  result: () = output_port(received, name=result, id=5)
}
"#;

/// Parses and verifies independently authored package-form block IR.
pub(super) fn package(ir: &str) -> Package {
    Parser::new(ir)
        .parse_and_verify_package()
        .unwrap_or_else(|error| panic!("public test IR must be valid:\n{ir}\n{error}"))
}

/// Confirms that an edge-case fixture is accepted by the upstream XLS parser.
pub(super) fn assert_stock_xls_accepts(ir: &str) {
    xlsynth::IrPackage::parse_ir(ir, None).unwrap_or_else(|error| {
        panic!("upstream XLS rejected the public edge-case IR:\n{ir}\n{error}")
    });
}

/// Emits SystemVerilog using the requested public backend configuration.
pub(super) fn emit(ir: &str, options: &BlockCodegenOptions) -> TestRtl {
    TestRtl::emit(&package(ir), options)
}

/// Checks unsupported calls in both layouts, even when their result is unused.
pub(super) fn assert_invoke_rejected(ir: &str, block: &str) {
    for layout in [Layout::None, Layout::Pipeline] {
        let error = emit_system_verilog(
            &package(ir),
            &BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            BlockCodegenError::Unsupported(format!(
                "invoke is not supported in `{block}`; inline function calls before block2sv code generation"
            ))
        );
    }
}

/// Evaluates generated combinational output with independently compiled RTL.
#[cfg(feature = "iverilog-tests")]
pub(super) fn evaluate(
    source: &TestRtl,
    inputs: &[(&str, u32, u64)],
) -> BTreeMap<String, LogicValue> {
    let inputs = inputs
        .iter()
        .filter(|(_, width, _)| *width != 0)
        .map(|(name, width, value)| ((*name).to_owned(), LogicValue::from_u64(*width, *value)))
        .collect();
    source
        .evaluate(&inputs)
        .unwrap_or_else(|e| panic!("Icarus evaluation failed:\\n{source}\\n{e}"))
}

/// Extracts one completely known output no wider than 64 bits.
pub(super) fn output(values: &BTreeMap<String, LogicValue>, name: &str) -> u64 {
    values
        .get(name)
        .unwrap_or_else(|| panic!("generated RTL did not produce output `{name}`"))
        .to_u64_if_known()
        .unwrap_or_else(|| panic!("generated output `{name}` contains unknown values"))
}

/// Converts an arbitrary-width XLS value to simulator transport bits.
pub(super) fn logic_from_ir_bits(bits: &IrBits) -> LogicValue {
    LogicValue::from_bits(bits)
}

/// Rejects unknown RTL bits before comparing against concrete IR semantics.
pub(super) fn ir_bits_from_logic(value: &LogicValue) -> IrBits {
    value.to_bits().expect("known RTL output")
}

/// Produces a resettable register with independently configurable controls.
pub(super) fn register_ir(asynchronous: bool, active_low: bool, feedback: bool) -> String {
    let reset = if active_low { "rst_n" } else { "rst" };
    let next = if feedback {
        "  next: bits[8] = add(current, data, id=5)\n"
    } else {
        ""
    };
    let next_value = if feedback { "next" } else { "data" };
    format!(
        r#"package public_register

top block register_block(clk: clock, {reset}: bits[1], data: bits[8], enable: bits[1], result: bits[8]) {{
  #![reset(port="{reset}", asynchronous={asynchronous}, active_low={active_low})]
  reg state(bits[8], reset_value=3)
  {reset}: bits[1] = input_port(name={reset}, id=1)
  data: bits[8] = input_port(name=data, id=2)
  enable: bits[1] = input_port(name=enable, id=3)
  current: bits[8] = register_read(register=state, id=4)
{next}  state_write: () = register_write({next_value}, register=state, load_enable=enable, reset={reset}, id=6)
  result: () = output_port(current, name=result, id=7)
}}
"#
    )
}

/// Creates one state register updated by two mutually exclusive enabled writes.
pub(super) fn multiwrite_register_ir(reset: Option<(bool, bool)>) -> String {
    let (reset_port, reset_attribute, reset_value, reset_input, reset_operand) = if let Some((
        asynchronous,
        active_low,
    )) = reset
    {
        let name = if active_low { "rst_n" } else { "rst" };
        (
            format!("{name}: bits[1], "),
            format!(
                "  #![reset(port=\"{name}\", asynchronous={asynchronous}, active_low={active_low})]\n"
            ),
            ", reset_value=42".to_owned(),
            format!("  {name}: bits[1] = input_port(name={name}, id=1)\n"),
            format!(", reset={name}"),
        )
    } else {
        Default::default()
    };
    format!(
        r#"package public_multiwrite

top block multiwrite(clk: clock, {reset_port}first: bits[8], second: bits[8], first_enable: bits[1], second_enable: bits[1], result: bits[8]) {{
{reset_attribute}  reg state(bits[8]{reset_value})
{reset_input}  first: bits[8] = input_port(name=first, id=2)
  second: bits[8] = input_port(name=second, id=3)
  first_enable: bits[1] = input_port(name=first_enable, id=4)
  second_enable: bits[1] = input_port(name=second_enable, id=5)
  current: bits[8] = register_read(register=state, id=6)
  first_write: () = register_write(first, register=state, load_enable=first_enable{reset_operand}, id=7)
  second_write: () = register_write(second, register=state, load_enable=second_enable{reset_operand}, id=8)
  result: () = output_port(current, name=result, id=9)
}}
"#
    )
}

/// Creates one binary-operation package with explicit result width.
pub(super) fn binary_ir(operation: &str, result_width: usize) -> String {
    format!(
        r#"package public_binary

top block binary(lhs: bits[8], rhs: bits[8], result: bits[{result_width}]) {{
  lhs: bits[8] = input_port(name=lhs, id=1)
  rhs: bits[8] = input_port(name=rhs, id=2)
  computed: bits[{result_width}] = {operation}(lhs, rhs, id=3)
  result: () = output_port(computed, name=result, id=4)
}}
"#
    )
}

/// Constructs a parent with one child between two explicit register cuts.
pub(super) fn child_pipeline_ir(registered_child: bool) -> String {
    let child = if registered_child {
        r#"block child(clk: clock, value: bits[8], result: bits[8]) {
  reg inner(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  written: () = register_write(value, register=inner, id=2)
  current: bits[8] = register_read(register=inner, id=3)
  result: () = output_port(current, name=result, id=4)
}"#
    } else {
        r#"block child(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  one: bits[8] = literal(value=1, id=2)
  incremented: bits[8] = add(value, one, id=3)
  result: () = output_port(incremented, name=result, id=4)
}"#
    };
    format!(
        r#"package public_hierarchy

{child}

top block parent(clk: clock, value: bits[8], result: bits[8]) {{
  reg before(bits[8])
  reg after(bits[8])
  instantiation component(block=child, kind=block)
  value: bits[8] = input_port(name=value, id=5)
  before_write: () = register_write(value, register=before, id=6)
  before_read: bits[8] = register_read(register=before, id=7)
  connected: () = instantiation_input(before_read, instantiation=component, port_name=value, id=8)
  received: bits[8] = instantiation_output(instantiation=component, port_name=result, id=9)
  after_write: () = register_write(received, register=after, id=10)
  after_read: bits[8] = register_read(register=after, id=11)
  result: () = output_port(after_read, name=result, id=12)
}}
"#
    )
}

/// Extracts the visible stage headers belonging to one emitted module.
pub(super) fn pipeline_stage_comments<'source>(
    source: &'source str,
    module: &str,
) -> Vec<&'source str> {
    let marker = format!("module {module}(");
    let body = source
        .split(&marker)
        .nth(1)
        .unwrap_or_else(|| panic!("generated source is missing module `{module}`"))
        .split("endmodule")
        .next()
        .expect("split always has an initial segment");
    body.lines()
        .filter(|line| line.trim_start().starts_with("// ===== Pipe stage"))
        .collect()
}

/// Exhausts small multiplier operands and samples arbitrary-width sign
/// boundaries.
pub(super) fn multiplier_input_samples(width: usize) -> Vec<IrBits> {
    if width <= 5 {
        return (0..(1 << width))
            .map(|value| IrBits::make_ubits(width, value).unwrap())
            .collect();
    }
    let mut top = vec![false; width];
    top[width - 1] = true;
    vec![
        IrBits::make_ubits(width, 0).unwrap(),
        IrBits::make_ubits(width, 1).unwrap(),
        IrBits::all_ones(width),
        IrBits::from_lsb_is_0(&top),
        IrBits::from_lsb_is_0(&(0..width).map(|bit| bit % 2 == 0).collect::<Vec<_>>()),
    ]
}

/// Builds a two-stage enabled pipeline with a shared register-data expression.
pub(super) fn pipeline_with_shared_data_ir(definition: &str, early_use: &str) -> String {
    let (consumer, first_next) = match early_use {
        "none" => ("", "data"),
        "dead" => ("  dead: bits[8] = not(shared, id=12)", "data"),
        "live" => ("  early: bits[8] = or(shared, data, id=12)", "early"),
        _ => unreachable!(),
    };
    format!(
        r#"package public_shared_pipeline_data
top block shared_data(clk: clock, rst: bits[1], data: bits[8], result: bits[8]) {{
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg r0(bits[8], reset_value=0)
  reg r1(bits[8], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  r0_q: bits[8] = register_read(register=r0, id=3)
  r1_q: bits[8] = register_read(register=r1, id=4)
{definition}
{consumer}
  enabled: bits[1] = bit_slice(r0_q, start=0, width=1, id=13)
  r0_d: () = register_write({first_next}, register=r0, reset=rst, id=14)
  r1_d: () = register_write(shared, register=r1, load_enable=enabled, reset=rst, id=15)
  result: () = output_port(r1_q, name=result, id=16)
}}
"#
    )
}
