// SPDX-License-Identifier: Apache-2.0

mod support;

use std::error::Error;
use std::fmt::Write;
use std::fs;
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{ResetBehavior, compile, compile_with_options, empty_package};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn struct_type_inference_distinguishes_omitted_and_empty_arguments() -> Result<(), String> {
    let source = r#"
struct Wrap<T> { value: T }
#[allow(combinational_block)]
block sample(clk, value: u8) -> u8 { (Wrap { value }).value }
"#;
    compile(source, empty_package("sample"), ResetBehavior::default())?;
    let explicit = source.replace("Wrap { value }", "Wrap<> { value }");
    assert_eq!(
        compile(&explicit, empty_package("sample"), ResetBehavior::default()).unwrap_err(),
        "block 'sample': result: struct 'Wrap' expects 1 type arguments, got 0"
    );
    Ok(())
}

#[test]
fn contextual_struct_fields_can_use_later_type_witnesses() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Wrap<T> { value: T }
struct Outer<T> { inner: Wrap<T>, tag: T }
struct Duo<T> { a: T, b: T }

fn combine<T>(wrapped: Wrap<T>, witness: T) -> (T, T) {
    (wrapped.value, witness)
}

fn first<T>(a: T, b: T) -> T { a }
fn get_width<N: u32>(wrapped: Wrap<uN[N]>, witness: uN[N]) -> uN[N] {
    wrapped.value
}
fn get_array_width<N: u32>(wrapped: Wrap<uN[N]>[1], witness: uN[N]) -> uN[N] {
    wrapped[u32:0].value
}

#[allow(combinational_block)]
block forward<T>(clk, wrapped: Wrap<T>, witness: T) -> (T, T) {
    (wrapped.value, witness)
}

#[allow(combinational_block)]
pub block pipeline(clk, x: u8, select: bool) -> (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8) {
    let nested = Outer { inner: Wrap { value: 1 }, tag: x };
    let from_fn = combine(Wrap { value: 2 }, x);
    let from_block = forward(Wrap { value: 3 }, x);
    let from_branch = combine(if select { Wrap { value: 4 } } else { Wrap { value: 5 } }, x);
    let from_array = first([Wrap { value: 6 }], [Wrap { value: x }]);
    let from_fields = Duo { a: [Wrap { value: 7 }], b: [Wrap { value: x }] };
    let from_width = get_width(Wrap { value: 8 }, x);
    let from_array_width = get_array_width([Wrap { value: 9 }], x);
    let from_branch_width = get_width(if select { Wrap { value: 10 } } else { Wrap { value: 11 } }, x);
    let from_tuple = first((if select { Wrap { value: 12 } } else { Wrap { value: 13 } },), (Wrap { value: x },));
    (nested.inner.value, from_fn.0, from_block.0, from_branch.0,
     from_array[u32:0].value, from_fields.a[u32:0].value, from_width, from_array_width,
     from_branch_width, from_tuple.0.value)
}
"#;
    let package = compile(source, empty_package("context"), ResetBehavior::default())?;
    let top = package.get_top_block().expect("top block");
    let ports: Vec<_> = top
        .output_ports()
        .map(|port| top.port_type(port).to_string())
        .collect();
    assert_eq!(ports, ["bits[8]"; 10]);
    Ok(())
}

#[test]
fn dimension_witness_preserves_the_common_constructor_type() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Wrap<T> { value: T }

fn length<N: u32>(items: Wrap<u8>[N]) -> u8[N] {
    zero!<u8[N]>()
}
fn array_width<N: u32>(items: (Wrap<u8>, uN[N])[2]) -> uN[N] {
    items[u32:1].1
}
fn tuple_width<N: u32>(item: (Wrap<u8>, uN[N])) -> uN[N] {
    item.1
}

#[allow(combinational_block)]
pub block pipeline(clk, x: u8, select: bool) -> (u8[1], u8, u8) {
    let wrap = Wrap { value: x };
    (length(if select { [Wrap { value: x }] } else { [wrap] }),
     array_width([(Wrap { value: x }, 1), (wrap, x)]),
     tuple_width(if select { (Wrap { value: x }, 1) } else { (wrap, x) }))
}
"#;
    let package = compile(
        source,
        empty_package("dimensions"),
        ResetBehavior::default(),
    )?;
    let top = package.get_top_block().expect("top block");
    let ports: Vec<_> = top
        .output_ports()
        .map(|port| top.port_type(port).to_string())
        .collect();
    assert_eq!(ports, ["bits[8][1]", "bits[8]", "bits[8]"]);
    Ok(())
}

#[test]
fn generic_block_specializations_keep_named_output_ports() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct FlowOut<T> { pop_valid: bool, pop_data: T }
struct Packet { tag: u8, word: u16 }

block flow_result<T>(clk, rst, valid: bool, data: T) -> FlowOut<T> {
    reg valid_q: bool = false;
    reg data_q: T = zero!<T>();
    valid_q := valid;
    if valid { data_q := data; }
    FlowOut { pop_data: data_q, pop_valid: valid_q }
}

pub block pipeline(clk, rst, a_valid: bool, a: u8, b_valid: bool, tag: u8, word: u16)
    -> (bool, u8, bool, u8, u16) {
    let scalar = flow_result(a_valid, a);
    let packet = flow_result(b_valid, Packet { tag, word });
    (scalar.pop_valid, scalar.pop_data, packet.pop_valid, packet.pop_data.tag, packet.pop_data.word)
}
"#;
    let mut package = compile(source, empty_package("flows"), ResetBehavior::default())?;
    let top = package.get_top_block().expect("pipeline is the top");
    let [scalar, packet] = top.instantiations.as_slice() else {
        panic!("expected one instance per payload");
    };
    assert_ne!(scalar.block, packet.block);
    for (instance, expected_payload) in [(scalar, "bits[8]"), (packet, "(bits[8], bits[16])")] {
        let block = package
            .get_block(&instance.block)
            .expect("specialized child");
        let ports: Vec<_> = block
            .output_ports()
            .map(|port| (block.port_name(port), block.port_type(port).to_string()))
            .collect();
        assert_eq!(
            ports,
            [
                ("pop_valid", "bits[1]".to_owned()),
                ("pop_data", expected_payload.to_owned()),
            ]
        );
    }

    inline_all_blocks_in_package(&mut package)?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let rst = ports.input("rst").unwrap();
    let a_valid = ports.input("a_valid").unwrap();
    let b_valid = ports.input("b_valid").unwrap();
    let a = ports.input("a").unwrap();
    let tag = ports.input("tag").unwrap();
    let word = ports.input("word").unwrap();
    for (reset, a_enable, b_enable, [byte, id, data], expected) in [
        (true, true, true, [10, 20, 0x1234], [0, 0, 0, 0, 0]),
        (false, true, false, [11, 21, 0x2345], [1, 11, 0, 0, 0]),
        (false, false, true, [12, 22, 0x3456], [0, 11, 1, 22, 0x3456]),
        (false, true, false, [13, 23, 0x4567], [1, 13, 0, 22, 0x3456]),
        (true, false, true, [14, 24, 0x5678], [0, 0, 0, 0, 0]),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&rst, reset)?;
                drive.set_bool(&a_valid, a_enable)?;
                drive.set_bool(&b_valid, b_enable)?;
                drive.set_u64(&a, byte)?;
                drive.set_u64(&tag, id)?;
                drive.set_u64(&word, data)
            })
            .unwrap();
        let actual = ["out_0", "out_1", "out_2", "out_3", "out_4"]
            .map(|port| edge.outputs[port].to_u64_if_known());
        assert_eq!(actual, expected.map(Some));
    }
    Ok(())
}

#[test]
fn imported_struct_specializes_with_a_nested_type_argument() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("flow.bx"),
        r#"
pub struct FlowOut<T: type> { pop_valid: bool, pop_data: T }

#[allow(combinational_block)]
pub block forward<T>(clk, valid: bool, data: T) -> FlowOut<T> {
    FlowOut::<T> { pop_valid: valid, pop_data: data }
}
"#,
    )?;
    let source = r#"
use flow;
struct Packet<U> { words: U }

pub block pipeline(clk, valid: bool, words: u16[2]) -> flow::FlowOut<Packet<u16[2]>> {
    let packet = Packet<u16[2]> { words };
    let stage = flow::forward(valid, packet);
    stage
}
"#;
    let package = compile_with_options(
        source,
        empty_package("flows"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().expect("pipeline is the top");
    let ports: Vec<_> = top
        .output_ports()
        .map(|port| (top.port_name(port), top.port_type(port).to_string()))
        .collect();
    assert_eq!(
        ports,
        [
            ("pop_valid", "bits[1]".to_owned()),
            ("pop_data", "(bits[16][2])".to_owned()),
        ]
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn equal_layout_specializations_keep_distinct_nominal_types() {
    let source = r#"
struct Token<T> { value: u8 }
block rename(clk, x: Token<u8>) -> Token<s8> { x }
"#;
    let error = compile(source, empty_package("flows"), ResetBehavior::default()).unwrap_err();
    assert!(
        error.contains("expected Token<s8>, got Token<u8>"),
        "{error}"
    );
}

#[test]
fn struct_arguments_are_inferred_through_block_and_function_inputs() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Packet<T> { payload: T }

fn read<T>(packet: Packet<T>) -> T { packet.payload }

fn invert<N: u32>(packet: Packet<uN[N]>) -> Packet<uN[N]> {
    Packet::<uN[N]> { payload: !packet.payload }
}

#[allow(combinational_block)]
block unpack<T>(clk, packet: Packet<T>) -> T { read(packet) }

pub block pipeline(clk, x: u8) -> Packet<u8> {
    let unpacked = unpack(Packet { payload: x });
    invert(Packet { payload: unpacked })
}
"#;
    let package = compile(source, empty_package("flows"), ResetBehavior::default())?;
    let top = package.get_top_block().expect("pipeline is the top");
    let mut outputs = top.output_ports();
    let output = outputs.next().expect("expected a payload output");
    assert!(outputs.next().is_none());
    assert_eq!(top.port_name(output), "payload");
    assert_eq!(top.port_type(output).to_string(), "bits[8]");
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn callback_result_infers_a_struct_type_argument() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Wrapper<T> { value: T }

fn wrap(x: u8) -> Wrapper<u8> { Wrapper { value: x } }
fn apply<T, F: fn(u8) -> Wrapper<T>>(x: u8) -> T { F(x).value }

#[allow(combinational_block)]
pub block pipeline(clk, x: u8) -> u8 { apply<wrap>(x) }
"#;
    let package = compile(source, empty_package("flows"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn nested_phantom_arguments_obey_the_specialization_budget() -> Result<(), Box<dyn Error>> {
    let mut source = r#"
struct Tag<T> { value: u8 }
fn wrap<T>(x: T) -> Tag<(T, T)> { Tag { value: u8:0 } }

#[allow(combinational_block)]
pub block pipeline(clk, x: u8) -> u8 {
    let x0 = x;
"#
    .to_owned();
    for index in 0..24 {
        writeln!(source, "    let x{} = wrap(x{index});", index + 1)?;
    }
    source.push_str("    x24.value\n}\n");
    let error = compile(&source, empty_package("flows"), ResetBehavior::default()).unwrap_err();
    assert!(
        error.contains("struct specializations exceed the limit"),
        "{error}"
    );
    Ok(())
}

#[test]
fn unused_struct_arguments_do_not_count_as_physical_fields() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Tag<A, B> { value: u8 }

#[allow(combinational_block)]
block child<T>(clk, x: T) -> Tag<T, T> { Tag { value: u8:1 } }

pub block pipeline(clk, x: uN[600000]) -> u8 {
    let tagged = child(x);
    tagged.value
}
"#;
    let package = compile(source, empty_package("flows"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}
