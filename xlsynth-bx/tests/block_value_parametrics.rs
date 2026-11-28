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
fn specializing_a_block_checks_the_types_of_forwarded_struct_arguments()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let template = "struct Tag<N: u32, Value: uN[N]> { data: u8 }";
    fs::write(directory.path().join("words.x"), format!("pub {template}"))?;
    let options = search_paths(&[directory.path()]);
    for (decl, name, context) in [
        (template, "Tag", "struct 'Tag' parametric 'Value': "),
        ("use words;", "words::Tag", ""),
    ] {
        let source = format!(
            "{decl} block stage<N: u32, S: bool>(clk, x: {name}<32, S>) -> bool {{ true }}"
        );
        check_module_with_options(&parse_module(&source)?, &options)?;
        let called =
            format!("{source} pub block top(clk) -> bool {{ let q = stage<1, true>(zero!()); q }}");
        assert_eq!(
            check_module_with_options(&parse_module(&called)?, &options).unwrap_err(),
            format!("block 'top': let 'q': call to block 'stage': {context}expected u32, got bool"),
        );
        // N may be one, in which case S is a valid argument for Value.
        let dependent = source.replace("<32, S>", "<N, S>");
        check_module_with_options(&parse_module(&dependent)?, &options)?;
    }
    Ok(())
}

#[test]
fn unused_block_checks_fixed_types_of_omitted_dslx_struct_arguments() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    fs::write(
        directory.path().join("words.x"),
        "pub struct Tag<N: u32, Flag: bool = {N != u32:0}, X: u8 = {u8:2}> { data: uN[N] }",
    )?;
    let options = search_paths(&[directory.path()]);
    for (width, context) in [("N", ""), ("32", "words::Tag<u32:32>: ")] {
        let source = format!(
            "use words; block stage<N: u32>(clk, x: words::Tag<{width}>) -> bool {{ true }}"
        );
        assert_eq!(
            check_module_with_options(&parse_module(&source)?, &options).unwrap_err(),
            format!(
                "block 'stage': {context}DSLX struct parametric 'X' of 'words::Tag' has unsupported type 'uN[8]'; expected bool or u32"
            ),
        );
    }
    Ok(())
}

#[test]
fn unused_block_checks_dslx_struct_types_through_default_chains() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("words.x"),
        r#"
const BASE = u32:3;
fn width(v: u32) -> u32 { v + u32:5 }
pub struct Known<N: u32, V: u32 = {BASE}, W: u32 = {width(V)}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Dependent<N: u32, V: u32 = {N}, W: u32 = {width(V)}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
"#,
    )?;
    let options = search_paths(&[directory.path()]);
    let source = "use words; block stage<N: u32>(clk, x: words::Known<N>) -> bool { true }";
    assert_eq!(
        check_module_with_options(&parse_module(source)?, &options).unwrap_err(),
        "block 'stage': DSLX struct parametric 'X' of 'words::Known' has unsupported type 'uN[8]'; expected bool or u32",
    );
    let dependent = source.replace("::Known", "::Dependent");
    check_module_with_options(&parse_module(&dependent)?, &options)?;
    Ok(())
}

#[test]
fn dslx_struct_parametrics_resolve_formal_names_in_their_lexical_scope()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let templates = r#"
const N = u32:8;
struct Dims { N: u32 }
pub struct Local<N: u32, W: u32 = {{ let N = u32:8; N }}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Field<N: u32, W: u32 = {(Dims { N: u32:8 }).N}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct FieldType<N: u32, V: uN[(Dims { N: u32:8 }).N]> { data: uN[N] }
pub struct ShadowDependent<N: u32, M: uN[N], W: u32 = {{ let M = u32:8; M }}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Forward<N: u32, W: u32 = {N}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Shorthand<N: u32, W: u32 = {(Dims { N }).N}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Select<N: u32, W: u32 = {const if N == u32:32 { u32:32 } else { u32:8 }}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
"#;
    let options = search_paths(&[directory.path()]);
    for inference in ["type_inference_v2", "type_inference_v1"] {
        fs::write(
            directory.path().join("words.x"),
            format!("#![feature({inference})]{templates}"),
        )?;
        for (name, arguments, formal) in [
            ("Local", "N", "X"),
            ("Field", "N", "X"),
            ("FieldType", "N, true", "V"),
            ("ShadowDependent", "N, 0", "X"),
        ] {
            let source = format!(
                "use words; block stage<N: u32>(clk, x: words::{name}<{arguments}>) -> bool {{ true }}"
            );
            assert_eq!(
                check_module_with_options(&parse_module(&source)?, &options).err(),
                Some(format!(
                    "block 'stage': DSLX struct parametric '{formal}' of 'words::{name}' has unsupported type 'uN[8]'; expected bool or u32"
                )),
                "{name} {inference}"
            );
        }
        for name in ["Forward", "Shorthand", "Select"] {
            let source = format!(
                r#"
use words;
block stage<N: u32>(clk, x: words::{name}<N>) -> bool {{ true }}
pub block top(clk, x: words::{name}<32>) -> bool {{ let result = stage<32>(x); result }}
"#,
            );
            check_module_with_options(&parse_module(&source)?, &options)?;
        }
    }
    Ok(())
}

#[test]
fn imported_struct_parametric_probes_respect_declaration_order() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let file = directory.path().join("words.x");
    let templates = br#"
pub struct Word<N: u32, W: u32 = {clz(u32:1) + u32:1}, X: uN[W] = {uN[W]:0}> { data: uN[N] }
pub struct Annotated<N: u32, X: uN[clz(u32:1) + u32:1]> { data: uN[N] }
"#;
    let shadow = b"fn clz(x: u32) -> u32 { u32:7 }\n";
    let concrete = r#"
use words;
#[allow(combinational_block)]
pub block circuit(clk, a: words::Word<8>, b: words::Annotated<8, 0>) -> u8 { a.data + b.data }
"#;
    let generic = r#"
use words;
#[allow(combinational_block)]
block stage<N: u32>(clk, a: words::Word<N>, b: words::Annotated<N, 0>) -> bits[N] { a.data + b.data }
#[allow(combinational_block)]
pub block circuit(clk, a: words::Word<8>, b: words::Annotated<8, 0>) -> u8 {
    let result = stage<8>(a, b);
    result
}
"#;
    let options = search_paths(&[directory.path()]);
    for inference in ["type_inference_v2", "type_inference_v1"] {
        let mut prefix = format!("#![feature({inference})]\n").into_bytes();
        // This DSLX character occupies one raw byte before both declarations.
        prefix.extend_from_slice(b"fn byte() -> u8 { '\xff' }\n");
        fs::write(&file, [prefix.as_slice(), templates, shadow].concat())?;
        for source in [concrete, generic] {
            compile_with_options(
                source,
                empty_package("design"),
                ResetBehavior::default(),
                &options,
            )
            .map_err(|e| format!("{inference}: {e}"))?;
        }

        // When clz is already in scope, both expressions produce width 8.
        fs::write(&file, [prefix.as_slice(), shadow, templates].concat())?;
        for (name, args) in [("Word", "N"), ("Annotated", "N, 0")] {
            let stage = format!(
                "use words; block stage<N: u32>(clk, word: words::{name}<{args}>) -> bits[N] {{ word.data }}"
            );
            assert_eq!(
                check_module_with_options(&parse_module(&stage)?, &options).err(),
                Some(format!(
                    "block 'stage': DSLX struct parametric 'X' of 'words::{name}' has unsupported type 'uN[8]'; expected bool or u32"
                )),
                "{inference}"
            );
        }
    }
    Ok(())
}

#[test]
fn imported_struct_default_lambda_captures_a_fixed_or_forwarded_block_port_argument()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let word = r#"
pub struct Word<N: u32, BASE: u32,
                W: u32 = {map([u32:0], |x: u32| x + BASE)[u32:0]},
                X: uN[W] = {uN[W]:0}> { data: uN[N], tag: uN[W] }
"#;
    let source = r#"
use words;
block fixed<N: u32>(clk, word: words::Word<N, 32>) -> u32 { word.tag }
block forwarded<N: u32, BASE: u32>(clk, word: words::Word<N, BASE>) -> bits[BASE] { word.tag }
pub block tags(clk, wide: words::Word<8, 32>, narrow: words::Word<32, 1>) -> (u32, u1) {
    let a = fixed<8>(wide);
    let b = forwarded<32, 1>(narrow);
    (a, b)
}
"#;
    let options = search_paths(&[directory.path()]);
    let module = parse_module(source)?;
    let unsupported = parse_module(
        "use words; block fixed<N: u32>(clk, word: words::Word<N, 8>) -> u8 { word.tag }",
    )?;
    for inference in ["type_inference_v2", "type_inference_v1"] {
        fs::write(
            directory.path().join("words.x"),
            format!("#![feature({inference})]{word}"),
        )?;
        check_module_with_options(&module, &options).map_err(|e| format!("{inference}: {e}"))?;
        assert_eq!(
            check_module_with_options(&unsupported, &options)
                .err()
                .as_deref(),
            Some(
                "block 'fixed': DSLX struct parametric 'X' of 'words::Word' has unsupported type 'uN[8]'; expected bool or u32"
            ),
            "{inference}"
        );
    }
    Ok(())
}

#[test]
fn imported_struct_literal_defaults_keep_their_contextual_types() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let word = r#"
pub struct Word<TAG: u32, N: u32, V: uN[N] = {0},
                X: uN[if V == uN[N]:0 { N } else { u32:8 }] = {0}> { data: uN[N] }
"#;
    let module = parse_module(
        r#"
use words;
block stage<TAG: u32>(clk, flag: words::Word<TAG, 1>, value: words::Word<TAG, 32>)
    -> (bool, u32) { (flag.data, value.data) }
"#,
    )?;
    let options = search_paths(&[directory.path()]);
    for inference in ["type_inference_v2", "type_inference_v1"] {
        fs::write(
            directory.path().join("words.x"),
            format!("#![feature({inference})]{word}"),
        )?;
        check_module_with_options(&module, &options).map_err(|e| format!("{inference}: {e}"))?;
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
fn record_block_options_specialize_array_ports_and_register_behavior() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    fs::write(
        directory.path().join("buffers.bx"),
        r#"
pub struct BufferOpts { depth: u32, bypass: bool }
type Options = BufferOpts;
const DEFAULT = BufferOpts { depth: u32:2, bypass: false };

block hold<T>(clk, data: T, enable: bool) -> T {
    reg saved: T = zero!<T>();
    if enable { saved := data; }
    saved
}
#[allow(combinational_block)]
pub block buffer<OPTS: Options = {DEFAULT}>(clk, data: u8[OPTS.depth], enable: bool) -> u8[OPTS.depth] {
    let result = const if OPTS.bypass { data } else { hold(data, enable) };
    result
}
"#,
    )?;
    let source = r#"
use buffers::BufferOpts;
use buffers::buffer;
const PIPE = BufferOpts { bypass: false, depth: u32:2 };
const DEFAULT = BufferOpts { bypass: true, depth: u32:2 };
const WIDE = BufferOpts { depth: u32:3, bypass: false };
pub block circuit(clk, a: u8[2], b: u8[3], enable: bool) -> (u8[2], u8[2], u8[2], u8[3]) {
    let implicit = buffer(a, enable);
    let explicit = buffer<PIPE>(a, enable);
    let direct = buffer<DEFAULT>(a, enable);
    let wide = buffer<WIDE>(b, enable);
    (implicit, explicit, direct, wide)
}
"#;
    let mut package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &search_paths(&[directory.path()]),
    )?;
    let top = package.get_top_block().ok_or("missing circuit")?;
    let [implicit, explicit, direct, wide] = top.instantiations.as_slice() else {
        panic!("expected four buffer instances");
    };
    assert_eq!(implicit.block, explicit.block);
    assert_ne!(implicit.block, direct.block);
    assert_ne!(implicit.block, wide.block);
    for (instance, depth, children) in [(implicit, 2, 1), (direct, 2, 0), (wide, 3, 1)] {
        let buffer = package.get_block(&instance.block).ok_or("missing buffer")?;
        let data = buffer.get_input_port("data").ok_or("missing data")?;
        let out = buffer.get_output_port("out").ok_or("missing out")?;
        let ty = Type::new_array(Type::Bits(8), depth);
        assert_eq!(buffer.port_type(data), &ty);
        assert_eq!(buffer.port_type(out), &ty);
        assert_eq!(buffer.instantiations.len(), children);
    }

    inline_all_blocks_in_package(&mut package)?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let reset = ports.input("rst").unwrap();
    let enable = ports.input("enable").unwrap();
    let a = ports.input("a").unwrap();
    let b = ports.input("b").unwrap();
    for (rst, en, words, wide_words, saved, wide_saved) in [
        (true, true, 0x1234, 0xabcdef, 0, 0),
        (false, true, 0x5678, 0x010203, 0x5678, 0x010203),
        (false, false, 0x9abc, 0x040506, 0x5678, 0x010203),
        (false, true, 0xff00, 0x070809, 0xff00, 0x070809),
        (true, false, 0x4321, 0x654321, 0, 0),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&reset, rst)?;
                drive.set_bool(&enable, en)?;
                drive.set_u64(&a, words)?;
                drive.set_u64(&b, wide_words)
            })
            .unwrap();
        let actual =
            ["out_0", "out_1", "out_2", "out_3"].map(|port| edge.outputs[port].to_u64_if_known());
        assert_eq!(actual, [saved, saved, words, wide_saved].map(Some));
    }
    Ok(())
}

#[test]
fn record_block_actuals_require_nominally_typed_constants() -> Result<(), String> {
    for (arg, diagnostic) in [
        ("OTHER", "expected BufferOpts, got Other"),
        (
            "runtime",
            "value parametric requires a compile-time constant",
        ),
    ] {
        let source = format!(
            r#"
struct BufferOpts {{ depth: u32 }}
struct Other {{ depth: u32 }}
const OTHER = Other {{ depth: u32:3 }};
block buffer<C: BufferOpts>(x: u32) -> u32 {{ C.depth + x }}
pub block circuit(runtime: BufferOpts, x: u32) -> u32 {{ let result = buffer<{arg}>(x); result }}
"#
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert_eq!(
            error,
            format!(
                "block 'circuit': let 'result': value parametric 'C' of block 'buffer': {diagnostic}"
            )
        );
    }
    Ok(())
}

#[test]
fn concrete_record_block_parametrics_preserve_wide_fields() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Mask<N: u32> { value: bits[N] }
const LOW = Mask<128> { value: uN[128]:1 };
const HIGH = Mask<128> { value: uN[128]:0x10000000000000001 };
#[allow(combinational_block)]
block masked<M: Mask<128>>(x: uN[128]) -> uN[128] { x ^ M.value }
pub block circuit(x: uN[128]) -> (uN[128], uN[128]) {
    let a = masked<LOW>(x);
    let b = masked<HIGH>(x);
    (a, b)
}
"#;
    let mut package = compile(source, empty_package("design"), ResetBehavior::default())?;
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().ok_or("missing circuit")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(128, 3)?]),
        IrValue::parse_typed("(bits[128]:2, bits[128]:0x10000000000000002)")?
    );
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
    let tuple = "type Dim = (u32,); block b<N:Dim>(clk, x:u32)->u32 { x }";
    assert_eq!(
        check_module(&parse_module(tuple)?).unwrap_err(),
        "block 'b': block value parametric 'N' must have type u32, bool, or a concrete struct"
    );
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
fn block_ports_forward_nested_struct_value_arguments_and_defaults() -> Result<(), String> {
    let source = r#"
struct Word<N: u32, Wide: bool = {N == u32:8}> { data: uN[N] }
struct Envelope<T> { item: T }

block hold<N: u32 = {7}>(clk, x: Envelope<Word<{N + u32:1}>>)
    -> Envelope<Word<{N + u32:1}>> {
    reg q: Envelope<Word<{N + u32:1}>>;
    q := x;
    q
}
pub block circuit(clk, a: Envelope<Word<8, true>>, b: Envelope<Word<17, false>>) -> (u8, u17) {
    let byte = hold(a);
    let wide = hold<16>(b);
    (byte.item.data, wide.item.data)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let circuit = package.get_top_block().ok_or("missing circuit")?;
    let [byte, wide] = circuit.instantiations.as_slice() else {
        panic!("expected two hold instances");
    };
    for (instance, width) in [(byte, 8), (wide, 17)] {
        let hold = package.get_block(&instance.block).ok_or("missing hold")?;
        let output = hold.get_output_port("item").ok_or("missing item")?;
        assert_eq!(
            hold.port_type(output),
            &Type::Tuple(vec![Box::new(Type::Bits(width))])
        );
    }
    // The phantom argument gives these inputs different types despite their
    // identical layouts.
    let wrong = source.replace("a: Envelope<Word<8, true>>", "a: Envelope<Word<8, false>>");
    assert!(check_module(&parse_module(&wrong)?).is_err());
    // Only the specialized child port gives the nested unsized literal its
    // width.
    let literal = source.replace("hold(a)", "hold(Envelope { item: Word { data: 1 } })");
    check_module(&parse_module(&literal)?)?;
    Ok(())
}

#[test]
fn mixed_block_port_infers_type_and_forwards_value_from_separate_sources() -> Result<(), String> {
    let source = r#"
struct Stamp<T, Tag: u32> { payload: T }
block save<T, N: u32>(clk, packet: Stamp<T, N>) -> Stamp<T, N> {
    reg q: T;
    q := packet.payload;
    Stamp { payload: q }
}
pub block circuit(clk, x: Stamp<s11, 5>) -> s11 {
    let saved = save<5>(x);
    saved.payload
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing circuit")?;
    let [instance] = top.instantiations.as_slice() else {
        panic!("expected one save instance");
    };
    let save = package.get_block(&instance.block).ok_or("missing save")?;
    assert_eq!(save.registers[0].ty, Type::Bits(11));
    Ok(())
}

#[test]
fn block_port_signedness_controls_extension_at_each_call() -> Result<(), Box<dyn Error>> {
    let source = r#"
#[allow(combinational_block)]
block widen<S: bool>(x: xN[S][8]) -> xN[S][16] { x as xN[S][16] }
#[allow(combinational_block)]
pub block circuit(x: u8) -> (s16, u16) {
    let signed = widen<true>(x as s8);
    let unsigned = widen<false>(x);
    (signed, unsigned)
}
"#;
    let mut package = compile(source, empty_package("design"), ResetBehavior::default())?;
    inline_all_blocks_in_package(&mut package)?;
    let circuit = combinational_block_to_fn(package.get_top_block().ok_or("missing circuit")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &circuit, &[IrValue::make_ubits(8, 255)?]),
        IrValue::make_tuple(&[IrValue::make_sbits(16, -1)?, IrValue::make_ubits(16, 255)?]),
    );
    let wrong = source.replace("widen<true>(x as s8)", "widen<true>(x)");
    assert_eq!(
        check_module(&parse_module(&wrong)?).unwrap_err(),
        "block 'circuit': let 'signed': input 'x', call to block 'widen': expected s8, got u8",
    );
    Ok(())
}

#[test]
fn unused_block_with_dependent_signedness_checks_its_known_layout_size() -> Result<(), String> {
    let source = r#"
block oversized<S: bool>(clk, x: xN[S][600000][2]) -> bool { true }
pub block top(clk, x: bool) -> bool { x }
"#;
    let err = check_module(&parse_module(source)?).unwrap_err();
    assert!(err.contains("exceeds maximum concrete type size"), "{err}");
    Ok(())
}

#[test]
fn port_width_can_be_a_literal_typed_by_a_block_parameter() -> Result<(), String> {
    let source = r#"
block stage<T>(clk, sample: T, data: bits[T:0x8]) -> bits[T:0x8] {
    reg q: bits[T:0x8];
    q := data;
    q
}
pub block circuit(clk, x: u8) -> u8 { let q = stage(u32:0, x); q }
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let circuit = package.get_top_block().ok_or("missing circuit")?;
    let out = circuit.get_output_port("out").ok_or("missing output")?;
    assert_eq!(circuit.port_type(out), &Type::Bits(8));
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
fn unused_block_checks_known_output_names_on_value_parametric_structs() -> Result<(), String> {
    let source = r#"
struct Word<N: u32> { data: uN[N] }
block stage<N: u32>(clk, data: uN[N]) -> Word<N> { Word { data } }
pub block top(clk, x: u8) -> u8 { x }
"#;
    assert_eq!(
        check_module(&parse_module(source)?).unwrap_err(),
        "block 'stage': duplicate name 'data'",
    );
    Ok(())
}

#[test]
fn specializing_a_struct_port_checks_its_expanded_array_layout() -> Result<(), String> {
    let source = r#"
struct Word<Tag: u32> { payload: u32 }
block stage<N: u32>(clk, words: Word<N>[33000]) -> bool { true }
pub block top(clk) -> bool {
    let result = stage<1>(zero!());
    result
}
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert!(
        error.contains("exceeds maximum concrete type size"),
        "{error}"
    );
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
fn specializing_a_port_checks_the_type_of_its_size_or_signedness() -> Result<(), String> {
    for (ty, value, formal, diagnostic) in [
        ("bits[V]", "true", "bool", "expected u32, got bool"),
        ("u8[V]", "true", "bool", "expected u32, got bool"),
        ("xN[V][8]", "1", "u32", "expected bool, got u32"),
    ] {
        let source = format!("block bank<V: {formal}>(clk, x: {ty}) -> {ty} {{ x }}");
        check_module(&parse_module(&source)?)?;
        let called =
            format!("{source} pub block top(clk) -> u8 {{ let q = bank<{value}>(u8:0); q }}");
        let error = check_module(&parse_module(&called)?).unwrap_err();
        assert_eq!(
            error,
            format!("block 'top': let 'q': call to block 'bank': {diagnostic}")
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
    for signature in [
        "(clk, input: u32, x: bits[N + input]) -> u32 { input }",
        "(clk, input: u32) -> u8[N + input] { zero!() }",
        "(clk, input: u32, x: Word<{N + input}>) -> u32 { input }",
        "(clk, input: u32) -> xN[N == input][8] { zero!() }",
    ] {
        let header = "const input = u32:4; struct Word<W: u32> { data: uN[W] } block bank<N: u32>";
        check_module(&parse_module(&format!("{header}{signature}"))?)?;
        let missing = signature
            .replace("+ input", "+ missing")
            .replace("== input", "== missing");
        let error = check_module(&parse_module(&format!("{header}{missing}"))?).unwrap_err();
        assert_eq!(error, "block 'bank': unknown value 'missing'");
    }
    Ok(())
}

#[test]
fn generic_port_operands_use_earlier_inputs_as_runtime_values() -> Result<(), String> {
    let operand = "if N == u32:4 { u32:5 } else { x }";
    for (result, concrete) in [
        (format!("uN[{operand}]"), "u5"),
        (format!("u8[{operand}]"), "u8[5]"),
        (format!("xN[{operand} == u32:5][8]"), "s8"),
        (format!("Word<{{{operand}}}>"), "Word<5>"),
    ] {
        // The runtime port shadows the module constant even for a literal
        // argument at the call site.
        let decl = format!(
            "const x = u32:5; struct Word<W: u32> {{ data: bits[W] }} block stage<N: u32>(clk, x: u32, value: {result}) -> {result} {{ value }}"
        );
        let used = format!(
            "{decl} pub block top(clk, a: {concrete}) -> {concrete} {{ let q = stage<4>(u32:5, a); q }}"
        );
        check_module(&parse_module(&used)?)?;
        let runtime = used.replace("stage<4>(", "stage<3>(");
        let error = check_module(&parse_module(&runtime)?).unwrap_err();
        assert!(
            error.contains("requires a compile-time constant"),
            "{result}: {error}"
        );
        let missing = decl.replace("else { x }", "else { missing }");
        assert_eq!(
            check_module(&parse_module(&missing)?).unwrap_err(),
            "block 'stage': unknown value 'missing'",
            "{result}"
        );
    }
    Ok(())
}

#[test]
fn direct_runtime_width_waits_until_block_specialization() -> Result<(), String> {
    let decl = "const x = u32:8; block stage<N: u32>(clk, x: u32) -> uN[x] { zero!() }";
    check_module(&parse_module(decl)?)?;
    let called = format!("{decl} pub block top(clk) -> u8 {{ let q = stage<1>(u32:8); q }}");
    assert_eq!(
        check_module(&parse_module(&called)?).unwrap_err(),
        "block 'top': let 'q': call to block 'stage': value parametric requires a compile-time constant",
    );
    Ok(())
}

#[test]
fn current_port_shadows_module_values_before_its_type_is_available() -> Result<(), String> {
    let decl = "const x = u32:8; block stage<N: u32>(clk, x: uN[x]) -> bool { true }";
    check_module(&parse_module(decl)?)?;
    let called = format!("{decl} block top(clk) -> bool {{ let q = stage<4>(u8:1); q }}");
    let error = check_module(&parse_module(&called)?).unwrap_err();
    assert!(error.contains("'x'"), "{error}");
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
