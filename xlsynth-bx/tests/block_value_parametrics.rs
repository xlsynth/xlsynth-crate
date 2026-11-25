// SPDX-License-Identifier: Apache-2.0

mod support;

use std::error::Error;
use std::fs;
use std::path::Path;
use support::search_paths;

use tempfile::tempdir;
use xlsynth::{DslxConvertOptions, XlsIrValue, convert_dslx_to_ir};
use xlsynth_bx::{
    ResetBehavior, check_module, check_module_with_options, compile, compile_with_options,
    empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn mixed_struct_port_types_bind_dimensions_inside_the_type_argument() -> Result<(), Box<dyn Error>>
{
    let source = r#"
struct Tagged<T, Tag: u32> { data: T }

block stage<W: u32, N: u32>(clk, input: Tagged<bits[W][N], 7>) -> Tagged<bits[W][N], 7> {
    reg q: bits[W][N];
    q := input.data;
    Tagged { data: q }
}

pub block capture(clk, words: Tagged<u12[3], 7>) -> u12[3] {
    let saved = stage<12, 3>(words);
    saved.data
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let capture = package.get_top_block().ok_or("missing capture")?;
    let [instance] = capture.instantiations.as_slice() else {
        panic!("expected one stage instance");
    };
    let stage = package.get_block(&instance.block).ok_or("missing stage")?;
    let data = stage.get_output_port("data").ok_or("missing data field")?;
    assert_eq!(stage.port_type(data), &Type::new_array(Type::Bits(12), 3));
    Ok(())
}

#[test]
fn unused_dimension_dependent_ports_check_independent_struct_arguments() -> Result<(), String> {
    for (formals, arguments, expected) in [
        ("T, Tag: u32", "u8, true", "u32"),
        ("T, Tag: u32", "bits[W], true", "u32"),
        ("T, Tag: u32", "bits[ceillog2(W)], true", "u32"),
        ("T, U, Tag: U", "bits[W], u8, true", "u8"),
        ("T, N: u32, Tag: bits[N]", "bits[W], 8, true", "u8"),
    ] {
        let source = format!(
            r#"
struct Tagged<{formals}> {{ data: T }}
block stage<W: u32>(clk, x: Tagged<{arguments}>) -> bool {{ true }}
pub block top(clk, x: bool) -> bool {{ x }}
"#,
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(
            error,
            format!(
                "block 'stage': struct 'Tagged' parametric 'Tag': expected {expected}, got bool"
            )
        );
    }
    Ok(())
}

#[test]
fn unused_dimension_dependent_ports_check_concrete_struct_defaults() -> Result<(), String> {
    for (formals, expected) in [
        ("T, Tag: u32 = {true}", "u32"),
        ("T, N: u32 = {8}, Tag: bits[N] = {true}", "u8"),
        ("T, N: u32 = {bit_count<T>()}, Tag: u32 = {true}", "u32"),
    ] {
        let source = format!(
            r#"
struct Tagged<{formals}> {{ data: T }}
block stage<W: u32>(clk, x: Tagged<bits[W]>) -> bool {{ true }}
pub block top(clk, x: bool) -> bool {{ x }}
"#,
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(
            error,
            format!(
                "block 'stage': struct 'Tagged' default for 'Tag': expected {expected}, got bool"
            )
        );
    }
    Ok(())
}

#[test]
fn mixed_struct_value_type_can_depend_on_a_block_port_dimension() -> Result<(), String> {
    let source = r#"
struct Stamp<T, Init: T, Width: u32 = {bit_count<T>()}> { data: T }
block stage<W: u32>(clk, x: Stamp<bits[W], u8:1>) -> bits[W] { x.data }
pub block top(clk, x: Stamp<u8, u8:1, 8>) -> u8 {
    let result = stage<8>(x);
    result
}
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn imported_delay_specializes_by_depth_and_matches_dslx_cycles() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("stages.bx"),
        r#"
const FIRST = u32:1;
fn end_of_delay(n: u32) -> u32 { FIRST + n }

block stage<T>(clk, data: T) -> T {
    reg q: T = zero!<T>();
    q := data;
    q
}

pub block delay<T, N: u32>(clk, data: T) -> T {
    unroll_for! (_, prev): (u32, T) in FIRST..end_of_delay(N) {
        let next = stage(prev);
        next
    }(data)
}
"#,
    )?;
    let source = r#"
use stages::delay;
const DEPTH = u32:3;

pub block pipeline(clk, x: u8, y: u8) -> (u8, u8, u8, u8) {
    let bypass = delay<0>(x);
    let short = delay<1>(x);
    let deep = delay<DEPTH>(x);
    let other = delay<{u32:1 + u32:2}>(y);
    (bypass, short, deep, other)
}
"#;
    let mut package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().expect("pipeline is the top");
    let [bypass, short, deep, other] = top.instantiations.as_slice() else {
        panic!("expected four delay instances");
    };
    for (instance, depth) in [(bypass, 0), (short, 1), (deep, 3), (other, 3)] {
        let delay = package
            .get_block(&instance.block)
            .expect("specialized delay");
        assert_eq!(delay.instantiations.len(), depth);
        assert_eq!(delay.reset.is_some(), depth > 0);
    }
    assert_eq!(deep.block, other.block);

    let reference = convert_dslx_to_ir(
        r#"
fn shift<N: u32>(rst: bool, data: u8, prev: u8[N]) -> u8[N] {
    for (i, next): (u32, u8[N]) in u32:0..N {
        let stage = if rst { u8:0 } else if i == u32:0 { data } else { prev[i - u32:1] };
        update(next, i, stage)
    }(prev)
}

fn step(rst: bool, x: u8, y: u8, prev: (u8[1], u8[3], u8[3])) -> (u8[1], u8[3], u8[3]) {
    (shift<1>(rst, x, prev.0), shift<3>(rst, x, prev.1), shift<3>(rst, y, prev.2))
}
"#,
        Path::new("delay_model.x"),
        &DslxConvertOptions::default(),
    )?;
    let step = reference.ir.get_function("__delay_model__step")?;
    let mut state = XlsIrValue::parse_typed(
        "([bits[8]:0], [bits[8]:0, bits[8]:0, bits[8]:0], [bits[8]:0, bits[8]:0, bits[8]:0])",
    )?;
    inline_all_blocks_in_package(&mut package)?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let reset = ports.input("rst").unwrap();
    let x = ports.input("x").unwrap();
    let y = ports.input("y").unwrap();
    for (rst, lhs, rhs) in [
        (true, 0x11, 0xaa),
        (false, 0x22, 0xbb),
        (false, 0x33, 0xcc),
        (false, 0x44, 0xdd),
        (false, 0x55, 0xee),
        (true, 0x66, 0xff),
        (false, 0x77, 0x99),
    ] {
        state = step.interpret(&[
            XlsIrValue::make_ubits(1, u64::from(rst))?,
            XlsIrValue::make_ubits(8, lhs)?,
            XlsIrValue::make_ubits(8, rhs)?,
            state,
        ])?;
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&reset, rst)?;
                drive.set_u64(&x, lhs)?;
                drive.set_u64(&y, rhs)
            })
            .unwrap();
        let expected = [
            lhs,
            state.get_element(0)?.get_element(0)?.to_u64()?,
            state.get_element(1)?.get_element(2)?.to_u64()?,
            state.get_element(2)?.get_element(2)?.to_u64()?,
        ];
        let actual =
            ["out_0", "out_1", "out_2", "out_3"].map(|port| edge.outputs[port].to_u64_if_known());
        assert_eq!(actual, expected.map(Some));
    }
    Ok(())
}

#[test]
fn value_and_callable_arguments_follow_declaration_order() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn inc(x: u8) -> u8 { x + u8:1 }
fn double(x: u32) -> u32 { x + x }

#[allow(combinational_block)]
block apply<T, F: fn(T) -> T, N: u32, G: fn(u32) -> u32>(clk, x: T) -> (T, u32) {
    (F(x), G(N))
}

block relay<T, OFFSET: u32>(clk, x: T) -> (T, u32) {
    let offset: uN[OFFSET + u32:1] = OFFSET as uN[OFFSET + u32:1];
    let result = apply<inc, {(offset as u32) + u32:2}, double>(x);
    result
}

pub block pipeline(clk, x: u8) -> (u8, u32) {
    let depth = double(u32:1) + u32:1;
    let result = relay<depth>(x);
    result
}
"#;
    let mut package = compile(source, empty_package("design"), ResetBehavior::default())?;
    inline_all_blocks_in_package(&mut package)?;
    let top = package.get_top_block().expect("pipeline is the top");
    let function = combinational_block_to_fn(top)?;
    let input = IrValue::make_ubits(8, 19)?;
    let expected =
        IrValue::make_tuple(&[IrValue::make_ubits(8, 20)?, IrValue::make_ubits(32, 10)?]);
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[input]),
        expected
    );
    Ok(())
}

#[test]
fn block_value_formals_resolve_u32_aliases_in_their_declaration_scope() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let options = search_paths(&[directory.path()]);
    fs::write(directory.path().join("shape.bx"), "pub type Dim = u32;")?;
    fs::write(directory.path().join("native.x"), "pub type Count = u32;")?;
    fs::write(
        directory.path().join("pipe.bx"),
        r#"
use shape;
type Word = shape::Dim;
#[allow(combinational_block)]
pub block bump<N: Word>(clk, x:u32) -> u32 { x + N }
"#,
    )?;
    let source = r#"
use pipe;
use native::Count;
use shape;
type Word = u8;
block relay<N: shape::Dim, M: Count>(clk, x:u32) -> u32 {
    let first = pipe::bump<N>(x);
    let second = pipe::bump<M>(first);
    second
}
block circuit(clk, x:u32) -> u32 { let q = relay<3, u32:5>(x); q }
"#;
    let mut package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )?;
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().ok_or("missing circuit")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(32, 10)?]),
        IrValue::make_ubits(32, 18)?
    );
    for source in [
        "type Dim = s32; block b<N:Dim>(clk, x:u32)->u32 { x }",
        "type Dim = (u32,); block b<N:Dim>(clk, x:u32)->u32 { x }",
        "type Dim = u32; block b<Dim,N:Dim>(clk,x:Dim)->Dim { x }",
        "use shape; block b<shape:u32,N:shape::Dim>(clk,x:u32)->u32 { x }",
        "use shape; block b<shape:fn(u8)->u8,N:shape::Dim>(clk,x:u32)->u32 { x }",
        "use shape::Dim; block b<Dim:fn(u8)->u8,N:Dim>(clk,x:u32)->u32 { x }",
        "type Dim = u32; block b<N:Dim>(clk,x:u32)->u32 { x } block c(clk,x:u32)->u32 { let q=b<u8:1>(x); q }",
    ] {
        assert!(
            check_module_with_options(&parse_module(source)?, &options).is_err(),
            "{source}"
        );
    }
    let forward = "block b<N:Dim>(clk,x:u32)->u32 { x } type Dim=u32;";
    check_module(&parse_module(forward)?)?;
    assert_eq!(
        check_module(&parse_module(&format!(
            "#![require_define_before_use]\n{forward}"
        ))?)
        .unwrap_err(),
        "in 'b': 'Dim' is declared later in the module"
    );
    Ok(())
}

#[test]
fn boolean_block_defaults_use_declaration_scope_and_share_specializations()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("stages.bx"),
        r#"
type Flag = bool;
const BASE = u32:2;
fn offset(split: bool) -> u32 { if split { BASE + u32:3 } else { BASE } }
#[allow(combinational_block)]
pub block stage<SPLIT: Flag = {false}, N: u32 = {offset(SPLIT)}>(x: u32) -> u32 {
    x + N
}
"#,
    )?;
    let source = r#"
use stages::stage;
pub block circuit(x: u32) -> (u32, u32, u32, u32) {
    let BASE = u32:9;
    let a = stage(x);
    let b = stage<false, 2>(x);
    let c = stage<true>(x);
    let d = stage<true, BASE>(x);
    (a, b, c, d)
}
"#;
    let mut package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().ok_or("missing circuit")?;
    let [a, b, c, d] = top.instantiations.as_slice() else {
        panic!("expected four stage instances");
    };
    assert_eq!(a.block, b.block);
    assert_ne!(a.block, c.block);
    assert_ne!(c.block, d.block);
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().ok_or("missing circuit")?)?;
    let expected = IrValue::parse_typed("(bits[32]:12, bits[32]:12, bits[32]:15, bits[32]:19)")?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(32, 10)?]),
        expected
    );
    Ok(())
}

#[test]
fn overridden_block_default_omits_its_native_helper_from_ir() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("config.x"),
        "pub fn split() -> bool { false }",
    )?;
    let source = r#"
use config;
#[allow(combinational_block)]
block stage<SPLIT: bool = {config::split()}>(x: bool) -> bool { SPLIT && x }
pub block pipeline(x: bool) -> bool { let q = stage<true>(x); q }
"#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    assert!(package.get_fn("__config__split").is_none());
    Ok(())
}

#[test]
fn omitted_block_defaults_follow_required_callbacks_in_calls_and_skipped_loops()
-> Result<(), Box<dyn Error>> {
    let source = r#"
fn invert(x: bool) -> bool { !x }
#[allow(combinational_block)]
block select<S: bool = {false}, F: fn(bool) -> bool, ENABLE: bool = {S}>(x: bool) -> bool {
    if ENABLE { F(x) } else { x }
}
block circuit(x: bool) -> bool {
    let v = select<true, invert>(x);
    unroll_for! (_, prev): (u32, bool) in u32:0..u32:0 {
        let next = select<false, invert>(prev);
        next
    }(v)
}
"#;
    let mut package = compile(source, empty_package("design"), ResetBehavior::default())?;
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().ok_or("missing circuit")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(1, 1)?]),
        IrValue::make_ubits(1, 0)?
    );
    Ok(())
}

#[test]
fn unused_block_defaults_resolve_names_at_the_declaration() -> Result<(), String> {
    for default in ["MISSING", "LATER", "AFTER", "input"] {
        let source = format!(
            "block stage<S:bool={{{default}}}, LATER:bool={{true}}>(input:bool)->bool {{ input }} const AFTER = true;"
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains("default for 'S'"), "{error}");
        assert!(error.contains(default), "{error}");
    }
    Ok(())
}

#[test]
fn block_defaults_preserve_required_positions_and_types() -> Result<(), String> {
    for (parametrics, call, expected) in [
        (
            "S: bool = {false}, F: fn(bool) -> bool",
            "stage(x)",
            "expects 2 explicit parametric arguments",
        ),
        (
            "S: bool = {false}, N: u32",
            "stage<true>(x)",
            "expects 2 explicit parametric arguments",
        ),
        (
            "S: bool = {false}",
            "stage<x>(x)",
            "requires a compile-time constant",
        ),
        (
            "S: bool = {false}",
            "stage<u32:1>(x)",
            "expected bool, got u32",
        ),
        ("S: bool = {u32:1}", "stage(x)", "default for 'S'"),
        ("S: bool = {u32:1}", "stage<true>(x)", "default for 'S'"),
        ("S: bool = {x}", "stage(x)", "default for 'S'"),
    ] {
        let source = format!(
            "block stage<{parametrics}>(x:bool)->bool {{ S }} block top(x:bool)->bool {{ let y={call}; y }}"
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains(expected), "{source}: {error}");
    }
    Ok(())
}

#[test]
fn block_value_actuals_must_be_explicit_compile_time_u32_values() -> Result<(), String> {
    for (call, expected) in [
        (
            "configured(x)",
            "expects 1 explicit parametric arguments, got 0",
        ),
        (
            "configured<x>(x)",
            "value parametric requires a compile-time constant",
        ),
        ("configured<u8:1>(x)", "expected u32, got u8"),
        (
            "configured<1, 2>(x)",
            "expects 1 explicit parametric arguments, got 2",
        ),
    ] {
        let source = format!(
            r#"
block configured<N: u32>(clk, x: u32) -> u32 {{ x + N }}
pub block pipeline(clk, x: u32) -> u32 {{ let result = {call}; result }}
"#,
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains(expected), "{call}: {error}");
    }
    Ok(())
}

#[test]
fn skipped_unroll_checks_required_values_for_a_block_named_like_a_port() -> Result<(), String> {
    let source = r#"
block configured<N: u32>(clk, x: u32) -> u32 { x + N }
pub block pipeline(clk, configured: u32) -> u32 {
    unroll_for! (_, x): (u32, u32) in u32:0..u32:0 {
        let value = configured(x);
        value
    }(configured)
}
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert!(
        error.contains("expects 1 explicit parametric arguments, got 0"),
        "{error}"
    );
    Ok(())
}

#[test]
fn imported_block_value_formals_preserve_qualified_extern_calls() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("stages.bx"),
        r#"
extern fn chip::step(x: u8) -> u8;
#[allow(combinational_block)]
pub block stage<chip: u32>(clk, x: u8) -> u8 { chip::step(x) + (chip as u8) }
"#,
    )?;
    let module = parse_module(
        r#"
use stages::stage;
pub block pipeline(clk, x: u8) -> u8 {
    let result = stage<3>(x);
    result
}
"#,
    )?;
    check_module_with_options(&module, &search_paths(&[directory.path()]))?;
    Ok(())
}

#[test]
fn block_value_parametrics_determine_port_and_register_shapes() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("banks.bx"),
        r#"
const W = u32:31;
const N = u32:7;

pub block storage<W: u32, N: u32>(clk, data: bits[W][N]) -> bits[W][N] {
    reg q: bits[W][N] = zero!<bits[W][N]>();
    q := data;
    q
}

pub block bank<T, N: u32>(clk, data: T[N]) -> T[N] {
    reg q: T[N];
    q := data;
    q
}
"#,
    )?;
    let source = r#"
use banks;
const W = u32:5;
const N = u32:3;

pub block circuit(clk, small: u5[3], large: u13[2]) -> (u5[3], u13[2], u13[2]) {
    let a = banks::storage<W, N>(small);
    let b = banks::storage<13, 2>(large);
    let c = banks::bank<2>(b);
    (a, b, c)
}
"#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().ok_or("missing circuit")?;
    let [small, large, bank] = top.instantiations.as_slice() else {
        panic!("expected three storage instances");
    };
    for (instance, width, depth) in [(small, 5, 3), (large, 13, 2), (bank, 13, 2)] {
        let child = package.get_block(&instance.block).ok_or("missing child")?;
        let shape = Type::new_array(Type::Bits(width), depth);
        let data = child.get_input_port("data").ok_or("missing data port")?;
        let out = child.get_output_port("out").ok_or("missing out port")?;
        assert_eq!(child.port_type(data), &shape);
        assert_eq!(child.port_type(out), &shape);
        let [q] = child.registers.as_slice() else {
            panic!("expected one array register");
        };
        assert_eq!(q.ty, shape);
    }
    Ok(())
}

#[test]
fn block_port_dimensions_specialize_nested_struct_arguments() -> Result<(), String> {
    let source = r#"
struct Bank<T> { words: T }

block save<W: u32, N: u32>(clk, bank: Bank<bits[W][N]>) -> Bank<bits[W][N]> {
    reg q: Bank<bits[W][N]>;
    q := bank;
    q
}

block circuit(clk, bank: Bank<u9[4]>) -> Bank<u9[4]> {
    let saved = save<9, 4>(bank);
    saved
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing circuit")?;
    let output = top.get_output_port("words").ok_or("missing words port")?;
    assert_eq!(top.port_type(output), &Type::new_array(Type::Bits(9), 4));
    Ok(())
}

#[test]
fn dimension_dependent_ports_preserve_the_nominal_type_behind_an_alias() -> Result<(), String> {
    let source = r#"
struct Record { byte: u8 }
type Saved = Record;
struct Wrap<U, Tag: u32> { data: U }
block stage<Record, L: u32>(clk, t: Record, w: Wrap<Saved[L], 7>) -> u8 {
    w.data[u32:0].byte
}
pub block circuit(clk, x: u16, w: Wrap<Saved[1], 7>) -> u8 {
    let q = stage<1>(x, w);
    q
}
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn unused_block_checks_output_names_from_a_shadowed_struct_alias() -> Result<(), String> {
    let source = r#"
struct Record { byte: u8 }
type Saved = Record;
block stage<Record, W: u32>(clk, t: Record, x: bits[W], byte: u8) -> Saved {
    Saved { byte }
}
pub block top(clk, x: u8) -> u8 { x }
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert_eq!(error, "block 'stage': duplicate name 'byte'");
    Ok(())
}

#[test]
fn block_value_parametric_ports_require_the_specialized_input_shape() -> Result<(), String> {
    for (width, depth) in [(4, 3), (5, 2)] {
        let source = format!(
            r#"
block storage<W: u32, N: u32>(clk, data: bits[W][N]) -> bits[W][N] {{
    reg q: bits[W][N];
    q := data;
    q
}}
pub block circuit(clk, data: u{width}[{depth}]) -> u5[3] {{
    let q = storage<5, 3>(data);
    q
}}
"#,
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(
            error,
            format!(
                "block 'circuit': let 'q': input 'data', call to block 'storage': expected u5[3], got u{width}[{depth}]"
            )
        );
    }
    Ok(())
}

#[test]
fn block_port_length_defaults_can_use_the_inferred_element_type() -> Result<(), String> {
    let source = r#"
type Dim = u32;
block bank<T, N: Dim = {bit_count<T>()}>(clk, data: T[N]) -> T[N] {
    reg q: T[N];
    q := data;
    q
}
block circuit(clk, a: u3[3], b: u3[2]) -> (u3[3], u3[2]) {
    let from_type = bank(a);
    let explicit = bank<2>(b);
    (from_type, explicit)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let circuit = package.get_top_block().ok_or("missing circuit")?;
    assert_eq!(circuit.instantiations.len(), 2);
    for (instance, length) in circuit.instantiations.iter().zip([3, 2]) {
        let bank = package.get_block(&instance.block).ok_or("missing bank")?;
        assert_eq!(bank.registers[0].ty, Type::new_array(Type::Bits(3), length));
    }
    Ok(())
}

#[test]
fn boolean_block_parameters_cannot_supply_port_sizes() -> Result<(), String> {
    for ty in ["bits[FLAG]", "u8[FLAG]"] {
        let source =
            format!("type Flag = bool; block bank<FLAG: Flag>(clk, x: {ty}) -> {ty} {{ x }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(
            error,
            "block 'bank': 'FLAG' names a local value in this type dimension"
        );
    }
    Ok(())
}

#[test]
fn block_port_dimension_expressions_specialize_in_the_declaring_module()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("banks.bx"),
        r#"
const N = u32:32;
const PAD = u32:2;
fn slots(n: u32) -> u32 { n + PAD }
struct Words<T, Tag: u32> { words: T }

pub block bank<N: u32>(clk, data: uN[ceillog2(N)][slots(N)]) -> Words<bits[ceillog2(N)][N + PAD], 7> {
    reg q: bits[ceillog2(N)][N + PAD];
    q := data;
    Words { words: q }
}
"#,
    )?;
    let source = r#"
use banks::bank;
const N = u32:8;
const PAD = u32:10;
fn slots(n: u32) -> u32 { n + PAD }

pub block circuit(clk, empty_bits: uN[0][3], small: u3[10], large: u4[11]) -> (uN[0][3], u3[10], u4[11]) {
    let z = bank<1>(empty_bits);
    let a = bank<N>(small);
    let b = bank<9>(large);
    (z.words, a.words, b.words)
}
"#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let circuit = package.get_top_block().ok_or("missing circuit")?;
    let [empty_bits, small, large] = circuit.instantiations.as_slice() else {
        panic!("expected three bank instances");
    };
    for (instance, width, length) in [(empty_bits, 0, 3), (small, 3, 10), (large, 4, 11)] {
        let bank = package.get_block(&instance.block).ok_or("missing bank")?;
        let input = bank.get_input_port("data").ok_or("missing data input")?;
        let output = bank
            .get_output_port("words")
            .ok_or("missing words output")?;
        let ty = Type::new_array(Type::Bits(width), length);
        assert_eq!(bank.port_type(input), &ty);
        assert_eq!(bank.port_type(output), &ty);
    }
    Ok(())
}

#[test]
fn computed_block_input_width_rejects_a_mismatched_argument() -> Result<(), String> {
    let source = r#"
block index<N: u32>(clk, x: bits[ceillog2(N)]) -> bits[ceillog2(N)] { x }
pub block circuit(clk, x: u4) -> u3 { let y = index<5>(x); y }
"#;
    assert_eq!(
        check_module(&parse_module(source)?).unwrap_err(),
        "block 'circuit': let 'y': input 'x', call to block 'index': expected u3, got u4",
    );
    Ok(())
}

#[test]
fn unused_dependent_block_dimensions_check_all_names_in_declaration_scope() -> Result<(), String> {
    for name in ["missing", "input"] {
        for signature in [
            format!("(clk, input: u32, x: bits[N + {name}]) -> u32 {{ input }}"),
            format!("(clk, input: u32) -> u8[N + {name}] {{ zero!() }}"),
        ] {
            let source = format!(
                "const input = u32:4; block bank<N: u32>{signature} pub block top(clk, x: u8) -> u8 {{ x }}"
            );
            let error = check_module(&parse_module(&source)?).unwrap_err();
            assert!(error.contains(name), "{signature}: {error}");
        }
    }
    Ok(())
}

#[test]
fn unused_dependent_block_dimensions_reject_shadowed_helpers() -> Result<(), String> {
    let helpers = r#"
fn bump(x: u32) -> u32 { x + u32:1 }
fn apply<F: fn(u32) -> u32>(x: u32) -> u32 { F(x) }
"#;
    for (port, width, diagnostic) in [
        (
            "apply",
            "apply<bump>(N)",
            "'apply' names a local value here",
        ),
        ("bump", "apply<bump>(N)", "'bump' names a local value here"),
        (
            "map",
            "N + map([u32:1], bump)[u32:0]",
            "'map' names a local value in this type argument",
        ),
        (
            "bump",
            "N + map([u32:1], bump)[u32:0]",
            "'bump' names a local value here",
        ),
    ] {
        let source = format!(
            r#"
{helpers}
block stage<N: u32>(clk, {port}: bool, y: bits[{width}]) -> bool {{ true }}
pub block circuit(clk, x: u8) -> u8 {{ x }}
"#
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(error, format!("block 'stage': {diagnostic}"), "{width}");
    }
    Ok(())
}

#[test]
fn computed_port_dimensions_can_use_inferred_types_and_boolean_values() -> Result<(), String> {
    let source = r#"
block wire<T, EXTRA: bool>(clk, sample: T, data: bits[bit_count<T>() + (EXTRA as u32)])
    -> bits[bit_count<T>() + (EXTRA as u32)] {
    reg q: bits[bit_count<T>() + (EXTRA as u32)];
    q := data;
    q
}
pub block circuit(clk, a: u3, b: u4) -> (u3, u4) {
    let narrow = wire<false>(a, a);
    let wide = wire<true>(a, b);
    (narrow, wide)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let circuit = package.get_top_block().ok_or("missing circuit")?;
    let [narrow, wide] = circuit.instantiations.as_slice() else {
        panic!("expected two wire instances");
    };
    for (instance, width) in [(narrow, 3), (wide, 4)] {
        let wire = package.get_block(&instance.block).ok_or("missing wire")?;
        assert_eq!(wire.registers[0].ty, Type::Bits(width));
    }
    Ok(())
}

#[test]
fn scoped_port_dimensions_respect_local_bindings() -> Result<(), String> {
    let source = r#"
block stage<N: u32>(clk, x: bits[{ let N = N + u32:1; N }]) -> bits[{ let N = u32:3; N }] {
    x as u3
}
pub block circuit(clk, x: u5) -> u3 { let q = stage<4>(x); q }
"#;
    check_module(&parse_module(source)?)?;

    let unused = r#"
block stage<N: u32>(clk, x: bits[{ let N = true; N }]) -> bool { true }
pub block circuit(clk, x: bool) -> bool { x }
"#;
    let error = check_module(&parse_module(unused)?).unwrap_err();
    assert_eq!(error, "block 'stage': expected u32, got bool");
    Ok(())
}

#[test]
fn a_match_pattern_compares_against_the_block_parameter_when_computing_a_width()
-> Result<(), String> {
    let source = r#"
block select<N: u32>(clk, x: bits[match u32:1 { N => u32:4, _ => u32:7 }]) -> u8 { x as u8 }
pub block circuit(clk, a: u4, b: u7) -> (u8, u8) {
    let matched = select<1>(a);
    let unmatched = select<2>(b);
    (matched, unmatched)
}
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn computed_mixed_struct_dimensions_respect_alias_layout_bounds() -> Result<(), String> {
    let source = r#"
type Lane = u16;
struct Bank<T, Tag: u32> { words: T }
block make<N: u32>(clk, x: bool) -> Bank<Lane[N + u32:1], 7> {
    Bank { words: zero!<Lane[N + u32:1]>() }
}
pub block circuit(clk, x: bool) -> bool { let q = make<65535>(x); x }
"#;
    check_module(&parse_module(&source.replace("make<65535>", "make<7>"))?)?;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert_eq!(
        error,
        "block 'circuit': let 'q': call to block 'make': type u16[65536] exceeds maximum concrete type size 1048576"
    );
    Ok(())
}
