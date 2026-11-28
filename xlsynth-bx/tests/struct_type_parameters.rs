// SPDX-License-Identifier: Apache-2.0

mod support;

use std::error::Error;
use std::fmt::Write;
use std::fs;
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{
    ResetBehavior, check_module, compile, compile_with_options, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
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

block flow_result<T>(clk, valid: bool, data: T) -> FlowOut<T> {
    reg valid_q: bool = false;
    reg data_q: T = zero!<T>();
    valid_q := valid;
    if valid { data_q := data; }
    FlowOut { pop_data: data_q, pop_valid: valid_q }
}

pub block pipeline(clk, a_valid: bool, a: u8, b_valid: bool, tag: u8, word: u16)
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

#[test]
fn mixed_struct_arguments_are_inferred_and_keep_phantom_values() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Packet<T, N: u32, Tag: u32 = {N + u32:1}> { lanes: T[N], valid: bool }

fn describe<T, N: u32, Tag: u32>(p: Packet<T, N, Tag>) -> (T, u32, u32) {
    (p.lanes[u32:0], N, Tag)
}

fn sample() -> ((u8, u32, u32), (s4, u32, u32), (u8, u32, u32)) {
    let a = Packet { lanes: [u8:7, u8:8], valid: true };
    let b = Packet::<s4> { lanes: [s4:-2], valid: false };
    let custom: Packet<u8, 2, 9> = Packet<u8, 2> { lanes: a.lanes, valid: false };
    let updated = Packet { valid: true, ..custom };
    (describe(a), describe(b), describe(updated))
}
const RESULT = sample();
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed(
            "((bits[8]:7, bits[32]:2, bits[32]:3), (bits[4]:14, bits[32]:1, bits[32]:2), (bits[8]:7, bits[32]:2, bits[32]:9))"
        )?
    );
    Ok(())
}

#[test]
fn mixed_struct_value_formal_can_use_an_earlier_type() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Payload { valid: bool, tag: u8 }
struct Stamp<N:u32, T:type, Init:T> { lanes:T[N] }
fn read_init<T, N:u32, Init:T>(p:Stamp<N, T, Init>) -> T { Init }
fn sample() -> (bool, u8) {
    const INITIAL:Payload = Payload { valid: true, tag: u8:42 };
    type Row = Stamp<2, Payload, INITIAL>;
    let p: Row = Stamp { lanes: zero!<Payload[2]>() };
    let expected = read_init(p);
    (expected.valid, expected.tag)
}
const RESULT = sample();
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[1]:1, bits[8]:42)")?
    );
    Ok(())
}

#[test]
fn scoped_fields_contribute_mixed_struct_witnesses() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct S<N:u32, T:type> { xs:T[N] }
struct O<T:type, N:u32> { a:S<N,T>, tag:T }
fn sample(x:u8) -> (u8, u8, u8, u8) {
    let inferred = O { a:{ let _dummy = u8:8; S { xs:[1,2] } }, tag:x };
    let prefixed = O {
        a:{ type Len = u32; const V:Len = u32:2; S<{V as Len}> { xs:[3,4] } },
        tag:x
    };
    let indexed = [u8:12,u8:13][O { a:{ x; S { xs:[1,2] } }, tag:x }.a.xs[u32:0]];
    let update = O { a:S { ..{ const V = u32:2; S<V> { xs:[5,6] } } }, tag:x };
    (inferred.a.xs[u32:0], prefixed.a.xs[u32:1], indexed, update.a.xs[u32:1])
}
const RESULT = sample(u8:42);
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[8]:1,bits[8]:4,bits[8]:13,bits[8]:6)")?
    );
    Ok(())
}

#[test]
fn scalar_value_formals_keep_each_calls_signedness() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Tag<T, N:u32, V:T={zero!<T>()}> { lanes:T[N] }
fn tag<T, N:u32, V:T>(p:Tag<T,N,V>) -> T { V }
fn read(a:s8[2], b:u8[1]) -> (s8,u8) {
    let s = Tag<s8> { lanes:a };
    let u:Tag<u8,1,u8:255> = Tag<u8,1,u8:255> { lanes:b };
    (tag(s), tag(u))
}
const RESULT = read([s8:-1,s8:0], [u8:3]);
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[8]:0, bits[8]:255)")?
    );
    Ok(())
}

#[test]
fn mixed_structs_preserve_imported_bindings_and_named_ports() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("frames.bx"),
        r#"
const EXTRA = u32:1;
pub struct Frame<N:u32, T, M:u32={N + EXTRA}> { payload:T[N], tags:u8[M] }
pub fn wrap<T, N:u32>(payload:T[N], tags:u8[N+EXTRA]) -> Frame<N,T> {
    Frame { payload, tags }
}
"#,
    )?;
    fs::write(
        directory.path().join("stages.bx"),
        r#"
use frames::Frame;
pub struct Stage<T, N:u32> { frame:Frame<N,T> }
pub fn stage<T,N:u32>(frame:Frame<N,T>) -> Stage<T,N> { Stage::<T,N> { frame } }
"#,
    )?;
    let source = r#"
use frames;
use stages;
const EXTRA = u32:9;
struct Value { x: u4 }
#[allow(combinational_block)]
pub block pipeline(data_in:Value[2], tags_in:u8[3]) -> frames::Frame<2,Value> {
    let input = frames::wrap(data_in, tags_in);
    let s: stages::Stage<Value, 2> = stages::stage(input);
    frames::Frame::<2,Value> { payload: s.frame.payload, tags: s.frame.tags }
}
"#;
    let package = compile_with_options(
        source,
        empty_package("pipeline"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().expect("pipeline is the top");
    let outputs: Vec<_> = top
        .output_ports()
        .map(|p| (top.port_name(p), top.port_type(p).to_string()))
        .collect();
    assert_eq!(
        outputs,
        [
            ("payload", "(bits[4])[2]".into()),
            ("tags", "bits[8][3]".into())
        ]
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn mixed_struct_specializations_check_both_argument_kinds() -> Result<(), String> {
    let template = "struct Tag<N:u32, T> { x:u8 }";
    for wrong in ["Tag<3,u8>", "Tag<2,s8>"] {
        let source = format!("{template} fn bad(v:Tag<2,u8>)->{wrong} {{ v }}");
        assert!(
            parse_module(&source)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{source}"
        );
    }
    for source in [
        "struct Bad<N:T, T> { data:T }",
        "struct Tag<T,N:u32> { x:u8 } block bad<T>(clk, p:Tag<T>) -> T { zero!() }",
        "struct Tag<T,N:u32> { x:u8 } block bad<T>(clk, p:Tag<T,MISSING>) -> T { zero!() }",
        "type A = u16; struct Tag<T,N:u32> { x:u8 } block bad<T>(clk, A:T, p:Tag<T,{bit_count<A>()}>) -> T { A }",
        "struct Bad<T, T:u32> { data:T }",
        "struct Tag<T,N:u32> { x:u8 } type Bad = Tag<u8>;",
        "struct Tag<T,N:u32> { x:u8 } fn bad(x:u8)->u8 { let v = Tag { x }; v.x }",
        "struct Tag<T,N:u32> { x:u8 } fn bad(x:u32)->u8 { let v = Tag<u8,x> { x:u8:1 }; v.x }",
    ] {
        assert!(
            parse_module(source).and_then(|m| check_module(&m)).is_err(),
            "{source}"
        );
    }
    let runtime = "const n = u32:2; struct Tag<T,N:u32> { x:u8 } block bad<T>(clk, n:u32, p:Tag<T,n>) -> T { zero!() }";
    check_module(&parse_module(runtime)?)?;
    let called =
        format!("{runtime} block top(clk, p:Tag<u8,2>)->u8 {{ let q = bad(u32:2, p); q }}");
    let error = check_module(&parse_module(&called)?).unwrap_err();
    assert!(
        error.contains("requires a compile-time constant"),
        "{error}"
    );
    Ok(())
}

#[test]
fn mixed_struct_type_arguments_follow_enclosing_declaration_visibility() -> Result<(), String> {
    let source = r#"
struct Tag<T, N:u32> { data:T }
struct Concrete { wrapped:Tag<Rec,2> }
struct Generic<A> { wrapped:Tag<(Rec,A),2> }
#[allow(combinational_block)]
block sample(x:Tag<Alias,2>, y:Concrete, z:Generic<bool>) -> (u8, u8, bool) {
    (x.data, y.wrapped.data.value, z.wrapped.data.1)
}
struct Rec { value:u8 }
type Alias = u8;
"#;
    let module = parse_module(source)?;
    check_module(&module)?;
    let strict = parse_module(&format!("#![require_define_before_use]\n{source}"))?;
    assert_eq!(
        check_module(&strict).unwrap_err(),
        "in 'Concrete': 'Rec' is declared later in the module"
    );

    for function in [
        "fn read(x:Tag<Rec,2>) -> u8 { x.data }",
        "fn read(x:u8) -> u8 { let t = Tag<Rec,2> { data:x }; t.data }",
        "fn read<T>(x:Tag<(Rec,T),2>) -> u8 { x.data }",
    ] {
        let module = parse_module(&format!(
            "struct Tag<T,N:u32> {{ data:u8 }} {function} struct Rec {{value:u8}}"
        ))?;
        let error = check_module(&module).unwrap_err();
        assert!(
            error.contains("Rec") && error.contains("declared later"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn mixed_struct_results_specialize_clocked_children() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Packet<T, N:u32> { valid:bool, data:T[N] }
block stage<T>(clk, d:T[2]) -> Packet<T,2> {
    reg store:T[2] = zero!();
    store := d;
    Packet { valid:true, data:store }
}
pub block pipeline(clk, xs:u8[2], flags:bool[2]) -> (u8[2], bool[2]) {
    let bytes = stage(xs);
    let bits_stage = stage(flags);
    (bytes.data, bits_stage.data)
}
"#;
    let package = compile(
        source,
        empty_package("packet_stage"),
        ResetBehavior::default(),
    )?;
    let parent = package.get_top_block().expect("top block");
    let [bytes, flags] = parent.instantiations.as_slice() else {
        panic!("expected one instance per array shape");
    };
    assert_ne!(bytes.block, flags.block);
    for (instance, expected) in [(bytes, "bits[8][2]"), (flags, "bits[1][2]")] {
        let block = package
            .get_block(&instance.block)
            .expect("specialized child");
        let outputs: Vec<_> = block
            .output_ports()
            .map(|p| (block.port_name(p), block.port_type(p).to_string()))
            .collect();
        assert_eq!(
            outputs,
            [("valid", "bits[1]".into()), ("data", expected.into())]
        );
    }
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn mixed_struct_siblings_infer_each_other_fields() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct S<T, N:u32, U> { xs:T[N], tag:U }
fn compare(a:u8, b:u16, pick:bool) -> (u8,u16,u8,u16,u8,u16) {
    let pair = [S { xs:[a], tag:1 }, S { tag:b, xs:[2] }];
    let chosen = if pick { S { xs:[a], tag:3 } } else { S { tag:b, xs:[4] } };
    let arm = match pick {
        true => S { xs:[a], tag:5 },
        _ => S { tag:b, xs:[6] },
    };
    (pair[u32:1].xs[u32:0], pair[u32:0].tag, chosen.xs[u32:0], chosen.tag, arm.xs[u32:0], arm.tag)
}
const RESULT = compare(u8:9, u16:77, false);
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed(
            "(bits[8]:2, bits[16]:1, bits[8]:4, bits[16]:77, bits[8]:6, bits[16]:77)"
        )?
    );
    Ok(())
}

#[test]
fn mixed_struct_siblings_keep_the_final_dimension_witness() -> Result<(), String> {
    let source = r#"
struct R<T,N:u32> { data:T, width_witness:uN[N] }
fn sample(y:u8) -> u8 {
    let rows = [R { data:y, width_witness:0 }, R { data:y, width_witness:y }];
    rows[u32:0].width_witness
}
"#;
    check_module(&parse_module(source)?)?;
    let reversed = source.replace(
        "R { data:y, width_witness:0 }, R { data:y, width_witness:y }",
        "R { data:y, width_witness:y }, R { data:y, width_witness:0 }",
    );
    assert!(check_module(&parse_module(&reversed)?).is_err());
    Ok(())
}
