// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs};
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{ResetBehavior, compile, compile_with_options, empty_package};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn compile_time_assertions_in_a_library_leave_block_ir_unchanged() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("limits.x"), "pub const MIN_WIDTH = u32:4;")?;
    let library = r#"
use limits::MIN_WIDTH;
fn valid_width(n: u32) -> bool { n >= MIN_WIDTH }
const_assert!(valid_width(u32:8));
pub fn bump<N: u32>(x: uN[N]) -> uN[N] {
    const_assert!({ let width = N; valid_width(width) });
    x + uN[N]:1
}
"#;
    let root = r#"
use math::bump;
block increment(clk, x: u8) -> u8 {
    reg q: u8;
    q := bump(x);
    q
}
"#;
    let options = search_paths(&[dir.path()]);
    let compile_root = || {
        compile_with_options(
            root,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        )
    };
    fs::write(dir.path().join("math.bx"), library)?;
    let with_assertions = compile_root()?;
    let without_assertions = library
        .lines()
        .filter(|line| !line.trim_start().starts_with("const_assert!"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(dir.path().join("math.bx"), without_assertions)?;
    assert_eq!(with_assertions.to_string(), compile_root()?.to_string());
    Ok(())
}

#[test]
fn an_assertion_can_typecheck_an_unused_dslx_call_without_linking_it() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        "pub fn nonzero(x: u8) -> bool { x != u8:0 }",
    )?;
    let options = search_paths(&[dir.path()]);
    let source = r#"
use math;
fn identity(x: u8) -> u8 {
    const_assert!(if true { true } else { math::nonzero(x) });
    x
}
"#;
    let compiled = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )?;
    let reference = compile(
        "fn identity(x: u8) -> u8 { x }",
        empty_package("design"),
        ResetBehavior::default(),
    )?;
    assert_eq!(compiled.to_string(), reference.to_string());
    Ok(())
}

#[test]
fn constexpr_dependencies_preserve_ir_for_maps_and_tuple_patterns() -> Result<(), String> {
    let compile_ir =
        |source: &str| compile(source, empty_package("design"), ResetBehavior::default());
    for source in [
        r#"
fn inc(x: u8) -> u8 { x + u8:1 }
fn apply(xs: u8[2], other: u8) -> u8[2] {
    const_assert!(true);
    map(xs, |e| { other; inc(e) })
}
"#,
        "fn f(x: u8) -> u8 { const_assert!(true); let (a, _) = (u8:1, x); u8:2 }",
    ] {
        let reference = source.replace("const_assert!(true);", "");
        assert_eq!(
            compile_ir(source)?.to_string(),
            compile_ir(&reference)?.to_string(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn local_functions_resolve_qualified_types_in_source_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("wire.x"), "pub struct T { x: u8 }")?;
    let options = search_paths(&[dir.path()]);
    for function in [
        "fn probe(x: (wire::T,)) -> u8 { x.0.x }",
        "fn probe(x: u8) -> u8 { let t = wire::T { x }; t.x }",
    ] {
        let early = format!("use wire;\n{function}");
        compile_with_options(
            &early,
            xlsynth_bx::empty_package("probe"),
            ResetBehavior::default(),
            &options,
        )?;
        let late = format!("{function}\nuse wire;");
        let error = compile_with_options(
            &late,
            xlsynth_bx::empty_package("probe"),
            ResetBehavior::default(),
            &options,
        )
        .unwrap_err();
        assert!(
            error.contains("type 'wire::T' is declared later"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn local_function_calls_remain_distinct_from_imports_and_externs() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        r#"
fn offset(x: u8) -> u8 { x + u8:2 }
pub fn transform(x: u8) -> u8 { offset(x) * u8:3 }
"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let linked = Parser::new(
        r#"package design
fn invert(x: bits[8] id=1) -> bits[8] { ret not.2: bits[8] = not(x, id=2) }
"#,
    )
    .parse_and_verify_package()?;
    let source = r#"
use math;
extern fn flip(x: u8) -> u8 = "invert";

fn __math__offset(x: u8) -> u8 { x + u8:9 }
fn collect(x: u8) -> (u8, u8, u8) { (__math__offset(x), math::transform(x), flip(x)) }
#[allow(combinational_block)]
block results(clk, x: u8) -> (u8, u8, u8) { collect(x) }
"#;
    let package = compile_with_options(source, linked, ResetBehavior::default(), &options)?;
    let top = package.get_top_block().ok_or("expected a top block")?;
    let function = combinational_block_to_fn(top)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(8, 5)?]),
        IrValue::parse_typed("(bits[8]:14, bits[8]:21, bits[8]:250)")?
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn local_function_rejects_an_occupied_destination_symbol() -> Result<(), Box<dyn Error>> {
    for member in [
        "fn occupied() -> bits[8] { ret literal.1: bits[8] = literal(value=1, id=1) }",
        "block occupied() {}",
    ] {
        let linked =
            Parser::new(&format!("package design\n{member}")).parse_and_verify_package()?;
        let error = compile(
            "fn occupied() -> u8 { u8:2 }",
            linked,
            ResetBehavior::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            r#"fn 'occupied': duplicate IR identifier: "occupied""#
        );
    }
    Ok(())
}

#[test]
fn local_function_and_parameter_names_can_be_ir_keywords() -> Result<(), Box<dyn Error>> {
    let linked = Parser::new(
        r#"package design
fn stage__bx_2() -> bits[8] { ret literal.1: bits[8] = literal(value=42, id=1) }
"#,
    )
    .parse_and_verify_package()?;
    let source = r#"
fn stage(next: u8, next__bx_1: u8) -> u8 { next - next__bx_1 }
fn stage__bx_1(x: u8) -> u8 { x + u8:7 }
#[allow(combinational_block)]
block results(clk, x: u8) -> u8 { stage(stage__bx_1(x), x) }
"#;
    let package = compile(source, linked, ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("expected a top block")?;
    let function = combinational_block_to_fn(top)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(8, 12)?]),
        IrValue::make_ubits(8, 7)?
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn block_typed_zero_preserves_the_intrinsic_with_a_same_named_import() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    fs::write(dir.path().join("wire.x"), "pub fn zero!() -> u8 { u8:7 }")?;
    fs::write(
        dir.path().join("values.bx"),
        "pub fn zero!() -> u8 { u8:7 }",
    )?;
    fs::write(
        dir.path().join("lib.bx"),
        "pub fn cleared() -> u8 { zero!() }",
    )?;
    let options = search_paths(&[dir.path()]);
    for (source, expected) in [
        (
            "use wire::zero!; #[allow(combinational_block)] block probe(clk) -> u8 { zero!<u8>() }",
            0,
        ),
        (
            "use values::zero!; #[allow(combinational_block)] block probe(clk) -> u8 { zero!() }",
            7,
        ),
        (
            "use lib::cleared; fn zero!() -> u8 { u8:7 } #[allow(combinational_block)] block probe(clk) -> u8 { cleared() }",
            0,
        ),
        (
            "#[allow(combinational_block)] block probe(clk) -> u8 { zero!<u8>() } fn zero!() -> u8 { u8:7 }",
            7,
        ),
        (
            "#[allow(combinational_block)] block probe(clk) -> u8 { zero!() } fn zero!() -> u8 { u8:7 }",
            7,
        ),
    ] {
        let package = compile_with_options(
            source,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        )?;
        let top = package.get_top_block().ok_or("expected a top block")?;
        let function = combinational_block_to_fn(top)?;
        assert_eq!(
            eval_fn_in_package_value(&package, &function, &[]),
            IrValue::make_ubits(8, expected)?,
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn contextual_zero_uses_the_register_type_and_checks_enum_members() -> Result<(), Box<dyn Error>> {
    let source = r#"enum Mode : u2 { Idle = 0, Busy = 1 }
struct Payload { mode: Mode, data: u8[2] }
block delay(clk, input: Payload) -> Payload {
    reg q: Payload = zero!();
    q := input;
    q
}"#;
    let compile_reset =
        |source: &str| compile(source, empty_package("design"), ResetBehavior::default());
    let explicit = source.replace("zero!()", "zero!<Payload>()");
    assert_eq!(
        compile_reset(source)?.to_string(),
        compile_reset(&explicit)?.to_string()
    );
    // A specialization used only to compute the reset leaves the same hardware.
    let factory = format!(
        "fn cleared<T>() -> T {{ zero!() }}\n{}",
        source.replace("= zero!();", "= cleared();")
    );
    assert_eq!(
        compile_reset(&factory)?.to_string(),
        compile_reset(&explicit)?.to_string()
    );
    let no_zero = source.replace("Idle = 0", "Idle = 2");
    assert_eq!(
        compile_reset(&no_zero).unwrap_err(),
        compile_reset(&no_zero.replace("zero!()", "zero!<Payload>()")).unwrap_err()
    );
    Ok(())
}

#[test]
fn register_reset_replays_a_pure_call_bound_to_a_let() -> Result<(), Box<dyn Error>> {
    let source = r#"fn bump(x: u8) -> u8 { x + u8:1 }
block counter(clk, x: u8) -> u8 {
    let initial = bump(u8:6);
    reg q: u8 = initial;
    q := x;
    q
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing counter block")?;
    assert_eq!(
        top.registers[0].reset_value,
        Some(IrValue::make_ubits(8, 7)?)
    );
    let dynamic = source.replace("bump(u8:6)", "bump(x)");
    assert_eq!(
        compile(&dynamic, empty_package("design"), ResetBehavior::default()).unwrap_err(),
        "block 'counter': reset of register 'q' must be a compile-time constant"
    );
    Ok(())
}

#[test]
fn register_reset_checks_a_constant_array_index_supplied_by_a_helper() -> Result<(), Box<dyn Error>>
{
    let compile_block =
        |source: &str| compile(source, empty_package("design"), ResetBehavior::default());
    for (bindings, initializer) in [
        ("", "[u8:3, u8:7][idx()]"),
        ("let i = idx();", "[u8:3, u8:7][i]"),
        ("let initial = [u8:3, u8:7][idx()];", "initial"),
        (
            "let i = idx(); let initial = [u8:3, u8:7][i]; reg saved: u32 = i; saved := i;",
            "initial",
        ),
    ] {
        let source = format!(
            "fn idx() -> u32 {{ u32:1 }} block b(clk, x: u8) -> u8 {{ {bindings} reg q: u8 = {initializer}; q := x; q }}"
        );
        let package = compile_block(&source)?;
        let block = package.get_top_block().ok_or("missing block b")?;
        let q = block
            .registers
            .iter()
            .find(|r| r.name == "q")
            .ok_or("missing q")?;
        assert_eq!(q.reset_value, Some(IrValue::make_ubits(8, 7)?), "{source}");
        assert_eq!(
            compile_block(&source.replace("u32:1", "u32:2")).unwrap_err(),
            "block 'b': reset of register 'q' must be a compile-time constant",
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn bx_array_reads_preserve_clamping_for_runtime_indices() -> Result<(), Box<dyn Error>> {
    let source = r#"fn idx() -> u32 { u32:2 }
#[allow(combinational_block)]
block b(clk) -> u8 { [u8:3, u8:7][idx()] }"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing block b")?;
    let function = combinational_block_to_fn(top)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[]),
        IrValue::make_ubits(8, 7)?
    );

    let source = r#"block b(clk, x: u32) -> u8 {
    let i = if true { u32:2 } else { x };
    reg q: u8 = [u8:3, u8:7][i];
    q := u8:0;
    q
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing block b")?;
    assert_eq!(
        top.registers[0].reset_value,
        Some(IrValue::make_ubits(8, 7)?)
    );
    Ok(())
}

#[test]
fn runtime_array_reads_skip_replaying_an_index_that_requires_a_helper() -> Result<(), Box<dyn Error>>
{
    let source = r#"fn f() -> u32 { u32:2 }
#[allow(combinational_block)]
block b(clk, x: u8) -> (u8, u8, u8) {
    let i = (for (_, a) in u32:0..u32:100000 { a + u32:1 }(u32:0)) + f();
    ([x, x][i], [x, x][i], [x, x][i])
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let block = package.get_top_block().ok_or("missing block b")?;
    assert!(block.registers.is_empty());
    Ok(())
}

#[test]
fn a_reset_using_a_local_does_not_change_runtime_array_index_validation()
-> Result<(), Box<dyn Error>> {
    let source = r#"fn idx() -> u32 { u32:2 }
block b(clk) -> (u8, u8, u32) {
    let i = idx();
    let before = [u8:3, u8:7][i];
    let copy = i;
    reg r: u32 = copy;
    r := i;
    (before, [u8:3, u8:7][copy], r)
}"#;
    let with_builtin = source.replace("reg r:", "let width = ceillog2(u32:16); reg r:");
    for source in [source, &with_builtin] {
        let package = compile(source, empty_package("design"), ResetBehavior::default())?;
        let eval = BlockEvaluator::prepare(&package, "b")?;
        let state = eval.initial_state(RegisterInit::AllZeros)?;
        let reset = eval.step(&state, &[IrValue::bool(true)])?;
        let result = eval.step(&reset.next_state, &[IrValue::bool(false)])?;
        assert_eq!(
            result.outputs,
            [
                IrValue::make_ubits(8, 7)?,
                IrValue::make_ubits(8, 7)?,
                IrValue::make_ubits(32, 2)?,
            ]
        );
    }
    Ok(())
}

#[test]
fn finding_a_helper_late_in_a_reset_does_not_repeat_loop_work() -> Result<(), Box<dyn Error>> {
    let count = "for (_, a) in u32:0..u32:140000 { a + u32:1 }(u32:0)";
    for (first, second, expected) in [
        (count, "f()", "(bits[32]:140000, bits[32]:3)"),
        ("f()", count, "(bits[32]:3, bits[32]:140000)"),
    ] {
        let source = format!(
            r#"fn f() -> u32 {{ u32:3 }}
block b(clk, x: (u32, u32)) -> (u32, u32) {{
    reg q: (u32, u32) = ({first}, {second}); q := x; q
}}"#
        );
        let package = compile(&source, empty_package("design"), ResetBehavior::default())?;
        let block = package.get_top_block().ok_or("missing block b")?;
        assert_eq!(
            block.registers[0].reset_value,
            Some(IrValue::parse_typed(expected)?)
        );
    }
    Ok(())
}

#[test]
fn deferred_let_without_helpers_stays_available_after_a_register_reset()
-> Result<(), Box<dyn Error>> {
    let source = r#"block b(clk, x: u32) -> (u32, u32) {
    let spent = for (i, a) in uN[1048576]:0..uN[1048576]:128 { a + (i as u32) }(u32:0);
    let k = for (_, a) in u32:0..u32:170000 { a + u32:1 }(u32:0);
    reg r: u32 = k; reg s: u32 = k;
    r := x; s := x; (r, s)
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let block = package.get_top_block().ok_or("missing block b")?;
    let initial = IrValue::make_ubits(32, 170000)?;
    assert_eq!(block.registers.len(), 2);
    assert!(
        block
            .registers
            .iter()
            .all(|r| r.reset_value.as_ref() == Some(&initial))
    );
    Ok(())
}

#[test]
fn identical_register_resets_share_the_evaluation_of_a_pure_helper() -> Result<(), Box<dyn Error>> {
    let mut source = String::from(
        r#"fn initial() -> u32 {
    for (_, a) in u32:0..u32:170000 { a + u32:1 }(u32:0)
}
block b(clk, x: u32) -> u32 {
"#,
    );
    for i in 0..32 {
        if i == 16 {
            // Checking this builtin enables ordinary constexpr calls too.
            source.push_str("let width = ceillog2(u32:16);\n");
        }
        source.push_str(&format!("reg q{i}: u32 = initial(); q{i} := x;\n"));
    }
    source.push_str("q0 }");
    let package = compile(&source, empty_package("design"), ResetBehavior::default())?;
    let block = package.get_top_block().ok_or("missing block b")?;
    let initial = IrValue::make_ubits(32, 170000)?;
    assert_eq!(block.registers.len(), 32);
    for reg in &block.registers {
        assert_eq!(reg.reset_value.as_ref(), Some(&initial));
    }
    Ok(())
}

#[test]
fn siblings_supply_factory_result_types_in_either_order() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn cleared<T>() -> T { zero!() }
fn width<N: u32>() -> uN[N] { N as uN[N] }
fn id(x: u8) -> u8 { x }
fn apply<T, F: fn(T) -> T>() -> T { F(zero!()) }
fn probe(x: u8) -> (bool, bool, u8[2], u8[2], u8, u8, bool) {
    let a = [cleared(), u8:1];
    let b = [u8:1, cleared()];
    let c = if true { width() } else { x };
    let d = if true { x } else { width() };
    (zero!() == x, x == zero!(), a, b, c, d, apply<id>() == x)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let probe = package.get_fn("probe").ok_or("missing probe")?;
    assert_eq!(
        eval_fn_in_package_value(&package, probe, &[IrValue::make_ubits(8, 6)?]),
        IrValue::parse_typed(
            "(bits[1]:0, bits[1]:0, [bits[8]:0, bits[8]:1], [bits[8]:1, bits[8]:0], bits[8]:8, bits[8]:6, bits[1]:0)"
        )?
    );

    let source = r#"
fn zero!() -> u16 { u16:42 }
fn id<T>(x: T) -> T { x }
fn probe() -> (u16[2], u16[2]) {
    let a = [zero!(), 1];
    let b = [id(u16:42), 1];
    (a, b)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let probe = package.get_fn("probe").ok_or("missing probe")?;
    assert_eq!(
        eval_fn_in_package_value(&package, probe, &[]),
        IrValue::parse_typed("([bits[16]:42, bits[16]:1], [bits[16]:42, bits[16]:1])")?
    );
    Ok(())
}

#[test]
fn bare_zero_without_a_result_type_suggests_an_annotation() {
    let error = compile(
        "fn unused() -> bool { let value = zero!(); true }",
        empty_package("design"),
        ResetBehavior::default(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        "fn 'unused': result: cannot infer the result type of 'zero!()'; provide a type annotation or use zero!<T>()"
    );
}
