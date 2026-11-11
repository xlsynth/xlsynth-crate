// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fs;
use std::path::Path;

use tempfile::tempdir;
use xlsynth::{DslxConvertOptions, XlsIrValue, convert_dslx_to_ir};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_module_with_options, compile,
    compile_with_options, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn imported_delay_specializes_by_depth_and_matches_dslx_cycles() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("stages.bx"),
        r#"
const FIRST = u32:1;
fn end_of_delay(n: u32) -> u32 { FIRST + n }

block stage<T>(clk, rst, data: T) -> T {
    reg q: T = zero!<T>();
    q := data;
    q
}

pub block delay<T, N: u32>(clk, rst, data: T) -> T {
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

pub block pipeline(clk, rst, x: u8, y: u8) -> (u8, u8, u8, u8) {
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
        &ImportOptions {
            additional_search_paths: vec![directory.path()],
            ..Default::default()
        },
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
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
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
    check_module_with_options(
        &module,
        &ImportOptions {
            additional_search_paths: vec![directory.path()],
            ..Default::default()
        },
    )?;
    Ok(())
}
