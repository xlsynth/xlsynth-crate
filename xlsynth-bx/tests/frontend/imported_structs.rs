// SPDX-License-Identifier: Apache-2.0

use super::support::search_paths;
use std::{error::Error, fs, io, path::Path};

use tempfile::{TempDir, tempdir};
use xlsynth_bx::check::CheckedStmt;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module_with_options, empty_package, lower_module,
    parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::libxls_bridge::value_to_libxls;

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{
    BlockEvaluator, ExpressionEvaluator, bx_source, compile_bundled_module, dslx_source,
};

#[test]
fn std_ordering_retains_its_nominal_type_and_signed_encoding() -> Result<(), Box<dyn Error>> {
    let bx = r#"
use std;
use std::Ordering;
block sample(clk, x:u8, y:u8)->(bool, s8) {
    let order: Ordering = std::compare(x,y);
    (order == std::Ordering::Less, order as s8)
}
"#;
    let native = r#"
import std;
fn sample(x:u8, y:u8)->(bool, s8) {
    let order = std::compare(x,y);
    (order == std::Ordering::Less, order as s8)
}
"#;
    let evaluator = ExpressionEvaluator::compile_block(
        bx,
        &ImportOptions::default(),
        compile_dslx_module(native, &[]),
    )?;
    for (x, y, expected) in [
        (0, 255, "(bits[1]:1,bits[8]:255)"),
        (7, 7, "(bits[1]:0,bits[8]:0)"),
        (255, 0, "(bits[1]:0,bits[8]:1)"),
    ] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, x)?, IrValue::make_ubits(8, y)?])?,
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn enum_value_paths_follow_dslx_identifier_rules() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let declarations =
        "pub enum self:u2 { A=0 } pub enum _:u2 { A=1 } pub enum E:u2 { self=1, _=2 }";
    fs::write(dir.path().join("protocol.bx"), declarations)?;
    let options = search_paths(&[dir.path()]);
    for (uses, path, valid) in [
        ("", "_::A", false),
        ("", "E::self", false),
        ("", "self::A", true),
        ("", "E::_", true),
        ("use protocol;", "protocol::self::A", false),
        ("use protocol;", "protocol::E::self", false),
        ("use protocol;", "protocol::_::A", true),
        ("use protocol;", "protocol::E::_", true),
        ("use protocol;", "_::A", false),
    ] {
        let source = format!("{uses} {declarations} block sample(clk)->u2 {{ {path} as u2 }}");
        let result = check_module_with_options(&parse_module(&source)?, &options);
        assert_eq!(result.is_ok(), valid, "{source}: {result:?}");
    }
    Ok(())
}

#[test]
fn reflected_enum_names_require_a_visible_use_binding() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("protocol.x"),
        "pub enum Mode:u2 { Idle=0, Busy=1 } pub fn get()->Mode { Mode::Busy }",
    )?;
    let options = search_paths(&[dir.path()]);
    for body in [
        "get() == protocol::Mode::Busy",
        "protocol::Mode::Busy == get()",
        "let x = get(); match x { protocol::Mode::Busy => true, _ => false }",
        "let x = get(); unroll_for! (_, a) in u8:0..u8:0 { x == protocol::Mode::Busy }(true)",
        "let x = get(); unroll_for! (_, a) in u8:0..u8:0 { let m:protocol::Mode = x; a }(true)",
    ] {
        let source = format!("use protocol::get; block sample(clk)->bool {{ {body} }}");
        assert!(
            check_module_with_options(&parse_module(&source)?, &options).is_err(),
            "{source}"
        );
        let bound_module = format!("use protocol; {source}");
        check_module_with_options(&parse_module(&bound_module)?, &options)?;
        let bound_type = format!(
            "use protocol::Mode; {}",
            source.replace("protocol::Mode", "Mode")
        );
        check_module_with_options(&parse_module(&bound_type)?, &options)?;
    }
    Ok(())
}

#[test]
fn enum_imports_distinguish_equal_layouts_and_preserve_aliases() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    for subject in ["first", "second"] {
        fs::write(
            dir.path().join(format!("{subject}.x")),
            r#"
pub enum Mode { Neg = -2, Zero = 0 }
pub enum Maybe { None, Some(u8) }
pub struct Packet { modes: Mode[2] }
pub fn packet(m: Mode) -> Packet { Packet { modes: [m, Mode::Zero] } }
pub fn read(p: Packet) -> Mode { p.modes[u32:0] }
"#,
        )?;
    }
    fs::write(
        dir.path().join("aliases.x"),
        "import first; pub type Alias = first::Mode;",
    )?;
    let options = search_paths(&[dir.path()]);
    let bx = r#"
use aliases::Alias;
use first;
use second;
block sample(clk, pick:bool)->s8 {
    let selected: Alias = if pick { first::Mode::Neg } else { Alias::Zero };
    first::read(first::packet(selected)) as s8
}
"#;
    let native = r#"
import aliases;
import first;
fn sample(pick:bool)->s8 {
    let selected: aliases::Alias = if pick { first::Mode::Neg } else { aliases::Alias::Zero };
    first::read(first::packet(selected)) as s8
}
"#;
    let evaluator = ExpressionEvaluator::compile_block(
        bx,
        &options,
        compile_dslx_module(native, &[dir.path()]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::bool(true)])?,
        IrValue::make_ubits(8, 254)?
    );
    assert_eq!(
        evaluator.compare(&[IrValue::bool(false)])?,
        IrValue::make_ubits(8, 0)?
    );
    let mismatch = bx.replace("first::Mode::Neg", "second::Mode::Neg");
    assert!(check_module_with_options(&parse_module(&mismatch)?, &options).is_err());
    let sum = parse_module("use first; fn id(x:first::Maybe)->first::Maybe { x }")?;
    let error = check_module_with_options(&sum, &options).unwrap_err();
    assert!(error.contains("sum enum"), "{error}");
    Ok(())
}

#[test]
fn generic_dslx_calls_preserve_each_enums_discriminants() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        r#"#![feature(generics)]
pub fn classify<T: type>(x: T) -> bool { x == T::A }
pub fn classify__BXEnum0() -> bool { false }
pub fn classify__Mode(x: u2) -> bool { false }
"#,
    )?;
    fs::write(
        dir.path().join("first.x"),
        r#"import ops;
pub enum Mode : u2 { A=0, B=1 }
pub fn classified(x: Mode) -> bool { ops::classify(x) }"#,
    )?;
    fs::write(
        dir.path().join("outer.x"),
        r#"import first;
import second;
import ops;
pub fn hidden() -> (bool, bool, bool) {
    (ops::classify(first::Mode::A), ops::classify(second::Alias::A), ops::classify__BXEnum0())
}"#,
    )?;
    let bx = r#"use ops; use first; use second; use outer;
block sample(clk, x: first::Mode, y: second::Alias) -> (bool, bool, bool, bool, (bool,bool,bool)) {
    (ops::classify(x), ops::classify(y), first::classified(x), second::classified(y), outer::hidden())
}"#;
    let options = search_paths(&[dir.path()]);
    for width in [2, 3] {
        fs::write(
            dir.path().join("second.x"),
            format!(
                r#"import ops;
pub enum Mode : u{width} {{ A=1, B=0 }}
pub type Alias = Mode;
pub fn classified(x: Alias) -> bool {{ ops::classify(x) }}"#
            ),
        )?;
        let checked = check_module_with_options(&parse_module(bx)?, &options)?;
        let evaluator = BlockEvaluator::lower(&checked, bx)?;
        for (input, first, second) in [(0, true, false), (1, false, true)] {
            assert_eq!(
                evaluator.evaluate(&[
                    IrValue::make_ubits(2, input)?,
                    IrValue::make_ubits(width, input)?,
                ])?,
                IrValue::make_tuple(&[
                    IrValue::bool(first),
                    IrValue::bool(second),
                    IrValue::bool(first),
                    IrValue::bool(second),
                    IrValue::make_tuple(&[
                        IrValue::bool(true),
                        IrValue::bool(true),
                        IrValue::bool(false)
                    ])
                ])
            );
        }
    }
    let single = r#"use first; use ops;
block sample(clk, x: first::Mode) -> (bool, bool) {
    (ops::classify(x), ops::classify__Mode(x as u2))
}"#;
    let checked = check_module_with_options(&parse_module(single)?, &options)?;
    let evaluator = BlockEvaluator::lower(&checked, single)?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::make_ubits(2, 0)?])?,
        IrValue::make_tuple(&[IrValue::bool(true), IrValue::bool(false)])
    );
    Ok(())
}

#[test]
fn nul_character_literals_preserve_imported_dslx_comparisons() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        r#"
pub enum Mode:u2 { A=0, B=3 }
fn copy<N:u32>(x:uN[N])->uN[N] { x }
pub fn safe(x:Mode)->bool { let _ = '\x00'; (x as u2) < u2:1 }
pub fn shift(x:Mode)->u2 { let _ = '\x00'; (x as u2) >> u1:1 }
pub fn sibling(x:Mode)->bool { copy<u32:2>(x as u2) >= u2:1 }
"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let bx = "use ops; block sample(clk,x:ops::Mode)->(bool,u2,bool) { (ops::safe(x), ops::shift(x), ops::sibling(x)) }";
    let checked = check_module_with_options(&parse_module(bx)?, &options)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, bx)?.evaluate(&[IrValue::make_ubits(2, 3)?])?,
        IrValue::parse_typed("(bits[1]:0,bits[2]:1,bits[1]:1)")?
    );
    Ok(())
}

#[test]
fn imported_enum_order_diagnostic_uses_the_original_byte_columns() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("helper.x"), "pub fn id(x:u8)->u8{x}")?;
    let path = dir.path().join("ops.x");
    let line = b"import helper; pub fn f(x:Mode)->bool { let _ = helper::id('\xff'); x < Mode::B }";
    let mut source = b"pub enum Mode:u2 { A=0, B=1 }\n".to_vec();
    source.extend_from_slice(line);
    fs::write(&path, source)?;
    let bx = parse_module("use ops; block sample(clk,x:ops::Mode)->bool { ops::f(x) }")?;
    let options = search_paths(&[dir.path()]);
    let error = check_module_with_options(&bx, &options).unwrap_err();
    let column = line.iter().position(|b| *b == b'<').unwrap() + 1;
    assert_eq!(
        error,
        format!(
            "checking use 'ops': xlsynth error: INVALID_ARGUMENT: TypeInferenceError: {}:2:{column}-2:{} Operator '<' requires bits-typed operands, got Mode.",
            path.display(),
            column + 1
        )
    );
    Ok(())
}

#[test]
fn imported_constants_can_order_enum_backing_values() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        r#"
pub enum Mode : u2 { A = 0, B = 1 }
pub fn before(a:Mode, b:Mode)->bool { (a as u2) < (b as u2) }
"#,
    )?;
    fs::write(
        dir.path().join("constants.x"),
        r#"
import ops;
const ORDERED = ops::before(ops::Mode::A, ops::Mode::B);
const WIDE = if ORDERED { sN[96]:-0x1_0000_0000_0000_0001 } else { sN[96]:0 };
const_assert!(ORDERED);
pub fn wide(x:ops::Mode)->sN[96] { if ops::before(x, ops::Mode::B) { WIDE } else { sN[96]:0 } }
"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let bx = "use ops; use constants; block sample(clk,x:ops::Mode)->sN[96] { constants::wide(x) }";
    let native =
        "import ops; import constants; fn sample(x:ops::Mode)->sN[96] { constants::wide(x) }";
    let evaluator = ExpressionEvaluator::compile_block(
        bx,
        &options,
        compile_dslx_module(native, &[dir.path()]),
    )?;
    for input in [0, 1] {
        evaluator.compare(&[IrValue::make_ubits(2, input)?])?;
    }
    Ok(())
}

#[test]
fn independent_imports_and_inert_generic_syntax_preserve_value_parametric_dispatch()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("other.x"),
        r#"#![feature(generics)]
pub enum Other : u2 { A=0 }
pub fn identity<T: type>(x:T)->T { x }"#,
    )?;
    let options = search_paths(&[dir.path()]);
    for (header, helpers) in [
        (
            "",
            r#"// #![feature(generics)] <T: type>
pub enum Mode : u2 { A=0 }
const generics = "T: type";"#,
        ),
        ("#![feature(generics)]", "fn identity<T:type>(x:T)->T{x}"),
    ] {
        fs::write(
            dir.path().join("ops.x"),
            format!(
                r#"{header}
{helpers}
struct Pair<N:u32> {{ v:uN[N] }}
impl Pair<N> {{ fn read(x:u2)->u2 {{ x }} }}
pub fn via<N:u32>(v:uN[N], x:u2)->u2 {{
    type P=Pair<N>;
    P::read(x)
}}"#
            ),
        )?;
        let bx = "use other; use ops; block sample(clk,x:u2)->u2 { ops::via(x,x) }";
        let checked = check_module_with_options(&parse_module(bx)?, &options)?;
        assert_eq!(
            BlockEvaluator::lower(&checked, bx)?.evaluate(&[IrValue::make_ubits(2, 2)?])?,
            IrValue::make_ubits(2, 2)?
        );
    }
    Ok(())
}

#[test]
fn colliding_nested_tuple_specializations_are_diagnosed_before_conversion()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        r#"#![feature(generics)]
pub enum Mode : u2 { A=0 }
pub fn eq<T: type>(x: T) -> bool { x == x }
pub fn hidden(x: Mode) -> (bool, bool) {
    (eq((x, (u8:0, u8:1), u8:2)), eq((x, (u8:0, u8:1, u8:2))))
}
pub fn mapped(x: Mode) -> (bool, bool) {
    map([x], hidden)[u32:0]
}
pub fn lambda(x: Mode) -> (bool, bool) {
    map([x], |v: Mode| -> (bool,bool) {
        (eq((v, (u8:0, u8:1), u8:2)), eq((v, (u8:0, u8:1, u8:2))))
    })[u32:0]
}
struct Pair<N:u32> { v:uN[N] }
impl Pair<N> {
    fn compare(x:Mode)->(bool,bool) { hidden(x) }
}
type Two=Pair<u32:2>;
pub fn via<N:u32>(v:uN[N], x:Mode)->(bool,bool) {
    let _ = '\x00';
    Two::compare(x)
}
pub fn mapped_via<N:u32>(v:uN[N], x:Mode)->(bool,bool) {
    map([x], Two::compare)[u32:0]
}
fn ordinary(x:Mode)->bool { x == Mode::A }
pub fn mapped_ok<N:u32>(v:uN[N], x:Mode)->bool {
    map([x], ordinary)[u32:0]
}
pub fn low_byte<T:type>(x:T)->bool {
    let b = '\x00';
    b == u8:0
}
pub fn high_byte<T:type>(x:T)->bool {
    let b = '\xff';
    b == u8:255
}"#,
    )?;
    let options = search_paths(&[dir.path()]);
    for body in [
        "(ops::eq((x,(u8:0,u8:1),u8:2)),ops::eq((x,(u8:0,u8:1,u8:2))))",
        "ops::hidden(x)",
        "ops::mapped(x)",
        "ops::lambda(x)",
    ] {
        let bx = format!("use ops; block sample(clk,x:ops::Mode)->(bool,bool) {{ {body} }}");
        assert_eq!(
            check_module_with_options(&parse_module(&bx)?, &options).unwrap_err(),
            "compiling DSLX import 'ops': distinct specializations of DSLX function 'eq' have the same native IR symbol"
        );
    }
    for function in ["via", "mapped_via"] {
        let bx = format!(
            "use ops; block sample(clk,x:ops::Mode)->(bool,bool) {{ ops::{function}(u2:0,x) }}"
        );
        assert_eq!(
            check_module_with_options(&parse_module(&bx)?, &options).unwrap_err(),
            format!(
                "compiling DSLX import 'ops': libxls does not expose the concrete method dispatch to 'compare' inside DSLX function '{function}'; BX cannot verify its native specialization symbols"
            )
        );
    }
    for body in [
        "ops::eq((x,(u8:0,u8:1),u8:2))",
        "ops::mapped_ok(u2:0,x)",
        "ops::low_byte(x)",
        "ops::high_byte(x)",
    ] {
        let single = format!("use ops; block sample(clk,x:ops::Mode)->bool {{ {body} }}");
        let checked = check_module_with_options(&parse_module(&single)?, &options)?;
        assert_eq!(
            BlockEvaluator::lower(&checked, &single)?.evaluate(&[IrValue::make_ubits(2, 0)?])?,
            IrValue::bool(true)
        );
    }
    Ok(())
}

#[test]
fn invalid_enum_ordering_is_diagnosed_when_the_dslx_module_is_imported()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("ops.x");
    let options = search_paths(&[dir.path()]);
    // Importing a valid sibling still typechecks the invalid comparison.
    let bx = parse_module("use ops; block sample(clk,x:ops::Mode)->bool { ops::equal(x,x) }")?;
    for operator in ["<", "<=", ">", ">="] {
        let source = format!(
            "pub enum Mode : u2 {{ Idle=0, Busy=1 }}\npub fn equal(a:Mode,b:Mode)->bool {{ a == b }}\npub fn ordered(a:Mode,b:Mode)->bool {{ a {operator} b }}\n"
        );
        fs::write(&path, source)?;
        let error = check_module_with_options(&bx, &options).unwrap_err();
        assert_eq!(
            error,
            format!(
                "checking use 'ops': xlsynth error: INVALID_ARGUMENT: TypeInferenceError: {}:3:41-3:{} Operator '{operator}' requires bits-typed operands, got Mode.",
                path.display(),
                41 + operator.len()
            )
        );
    }
    Ok(())
}

/// Install the declarations used on both sides of each import boundary test.
fn struct_modules() -> io::Result<TempDir> {
    let root = tempdir()?;
    fs::create_dir(root.path().join("fabric"))?;
    fs::write(
        root.path().join("fabric/wire.x"),
        r#"
pub struct Header { first: u8, second: u8, bias: s4 }
pub struct Frame { headers: Header[2] }
pub struct Holder { reg: u8 }
pub struct Wrapped { value: Holder }
struct Private { field: u8 }
pub struct Sized<N: u32> { data: uN[N] }
pub fn sized(x: u8) -> Sized<u32:8> { Sized<u32:8> { data: x } }
"#,
    )?;
    fs::write(
        root.path().join("facade.x"),
        r#"
import fabric.wire;
pub type Header = wire::Header;
pub fn swap(h: Header) -> Header { Header { second: h.first, first: h.second, bias: h.bias } }
pub fn echo(f: wire::Frame) -> wire::Frame { f }
pub fn reverse(h: Header[2]) -> Header[2] { [h[u32:1], h[u32:0]] }
"#,
    )?;
    fs::write(
        root.path().join("factory.x"),
        r#"
import fabric.wire;
import other;
pub fn make(x: u8) -> wire::Header { wire::Header { first: x, second: x + u8:1, bias: s4:-2 } }
pub fn second(h: wire::Header) -> u8 { h.second }
pub fn pair(x: u8) -> (wire::Header, (other::Header, u8)) {
    (make(x), (other::Header { first: x, second: u8:42, bias: s4:0 }, x))
}
pub fn other_second(h: other::Header) -> u8 { h.second }
"#,
    )?;
    fs::write(
        root.path().join("other.x"),
        "pub struct Header { first: u8, second: u8, bias: s4 }",
    )?;
    Ok(root)
}

/// Compare the same expression through BX and DSLX with equivalent bindings.
fn struct_expression(
    root: &Path,
    bx_declarations: &str,
    dslx_declarations: &str,
    params: &str,
    result: &str,
    body: &str,
) -> Result<ExpressionEvaluator, String> {
    let reference = format!("{dslx_declarations}\n{}", dslx_source(params, result, body));
    let source = format!("{bx_declarations}\n{}", bx_source(params, result, body));
    let options = search_paths(&[root]);
    ExpressionEvaluator::compile_block(&source, &options, compile_dslx_module(&reference, &[root]))
}

#[test]
fn cross_module_struct_aliases_preserve_field_order_and_signedness() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let expr = struct_expression(
        root.path(),
        r#"
use fabric::wire;
use facade;
struct Local { data: facade::Header }
"#,
        r#"
import fabric.wire;
import facade;
struct Local { data: facade::Header }
"#,
        "x: u8",
        "(u8, u8, s16)",
        r#"
let h: facade::Header = wire::Header { second: x + u8:1, bias: s4:-2, first: x };
let local = Local { data: facade::swap(h) };
let y: facade::Header = local.data;
let frame = wire::Frame { headers: [h, y] };
let echoed = facade::echo(frame);
let reversed = facade::reverse(echoed.headers);
(reversed[u32:0].first, reversed[u32:0].second, reversed[u32:0].bias as s16)
"#,
    )?;
    for x in [0u8, 127, 255] {
        let actual = expr.compare(&[IrValue::make_ubits(8, u64::from(x))?])?;
        assert_eq!(
            actual,
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, u64::from(x.wrapping_add(1)))?,
                IrValue::make_ubits(8, u64::from(x))?,
                IrValue::make_sbits(16, -2)?,
            ])
        );
    }
    Ok(())
}

#[test]
fn grouped_use_binds_nested_modules_public_functions_and_struct_aliases()
-> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let source = r#"
use fabric::{wire, wire::Frame};
use facade::{Header, swap, echo};
block sample(clk, x: u8) -> u8 {
    let h: Header = wire::Header { first: x, second: x + u8:1, bias: s4:-1 };
    let f = Frame { headers: [h, swap(h)] };
    echo(f).headers[u32:1].first
}"#;
    let reference = r#"
import fabric.wire;
import facade;
fn sample(x: u8) -> u8 {
    let h: facade::Header = wire::Header { first: x, second: x + u8:1, bias: s4:-1 };
    let f = wire::Frame { headers: [h, facade::swap(h)] };
    facade::echo(f).headers[u32:1].first
}"#;
    let paths = [root.path()];
    let options = search_paths(&paths);
    let expr = ExpressionEvaluator::compile_block(
        source,
        &options,
        compile_dslx_module(reference, &paths),
    )?;
    assert_eq!(
        expr.compare(&[IrValue::make_ubits(8, 12)?])?,
        IrValue::make_ubits(8, 13)?
    );
    Ok(())
}

#[test]
fn local_values_shadow_use_bindings_in_declaration_order() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let paths = [root.path()];
    let options = search_paths(&paths);
    let used_wire = ("use fabric::wire;", "import fabric.wire;");
    // Both frontends call the same DSLX function. The reference gives it a
    // callable local name so a value binding can shadow it.
    let used_swap = (
        "use fabric::wire; use facade::swap;",
        r#"
import fabric.wire;
import facade;
fn swap(h: wire::Header) -> wire::Header { facade::swap(h) }
"#,
    );
    let input = IrValue::parse_typed("(bits[8]:3, bits[8]:2, bits[4]:0)")?;
    let control = struct_expression(
        root.path(),
        used_swap.0,
        used_swap.1,
        "h: wire::Header",
        "u8",
        "swap(h).second",
    )?;
    assert_eq!(
        control.compare(std::slice::from_ref(&input))?,
        IrValue::make_ubits(8, 3)?
    );
    for ((bx_decl, dslx_decl), params, body) in [
        (used_wire, "wire: u8, h: wire::Header", "h.first"),
        (
            used_wire,
            "x: u8",
            "let wire = x; wire::Header { first: x, second: x, bias: s4:0 }.first",
        ),
        (
            used_wire,
            "x: u8",
            "let wire = x; wire::Sized<8> { data: x }.data",
        ),
        (
            used_swap,
            "h: wire::Header",
            "let swap = h.first; swap(h).second",
        ),
    ] {
        let source = format!("{bx_decl}\n{}", bx_source(params, "u8", body));
        let reference = format!("{dslx_decl}\n{}", dslx_source(params, "u8", body));
        assert!(
            check_module_with_options(&parse_module(&source)?, &options).is_err(),
            "{source}"
        );
        assert!(
            compile_dslx_module(&reference, &paths).is_err(),
            "{reference}"
        );
    }
    let expr = struct_expression(
        root.path(),
        used_wire.0,
        used_wire.1,
        "h: wire::Header, wire: u8",
        "u8",
        "h.first + wire",
    )?;
    assert_eq!(
        expr.compare(&[
            IrValue::parse_typed("(bits[8]:3, bits[8]:2, bits[4]:0)")?,
            IrValue::make_ubits(8, 1)?
        ])?,
        IrValue::make_ubits(8, 4)?,
    );
    let expr = struct_expression(
        root.path(),
        used_swap.0,
        used_swap.1,
        "x: u8",
        "u8",
        "let swap = x; swap",
    )?;
    let x = IrValue::make_ubits(8, 37)?;
    assert_eq!(expr.compare(std::slice::from_ref(&x))?, x);
    Ok(())
}

#[test]
fn grouped_struct_type_bindings_match_native_dslx_use() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let decls = "use fabric::wire::{Header, Frame};";
    let body = "let h: Header = frame.headers[u32:1]; h.second";
    let source = format!("{decls}\n{}", bx_source("frame: Frame", "u8", body));
    let reference = format!(
        "#![feature(use_syntax)]\n{decls}\n{}",
        dslx_source("frame: Frame", "u8", body)
    );
    let paths = [root.path()];
    let options = search_paths(&paths);
    // The pinned frontend resolves nominal use bindings in type annotations.
    // The checkout's v2 converter reports this annotation as unimplemented.
    let expr = ExpressionEvaluator::compile_block(
        &source,
        &options,
        compile_bundled_module(&reference, &paths),
    )?;
    let frame = IrValue::make_tuple(&[IrValue::parse_typed(
        "[(bits[8]:1, bits[8]:2, bits[4]:0), (bits[8]:3, bits[8]:4, bits[4]:0)]",
    )?]);
    assert_eq!(expr.compare(&[frame])?, IrValue::make_ubits(8, 4)?);
    Ok(())
}

#[test]
fn a_dslx_field_can_use_a_bx_keyword() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let expr = struct_expression(
        root.path(),
        "use fabric::wire;",
        "import fabric.wire;",
        "h: wire::Holder",
        "wire::Wrapped",
        "wire::Wrapped { value: wire::Holder { reg: h.reg + u8:1 } }",
    )?;
    let h = IrValue::make_tuple(&[IrValue::make_ubits(8, 6)?]);
    assert_eq!(
        expr.compare(&[h])?,
        IrValue::make_tuple(&[IrValue::make_tuple(&[IrValue::make_ubits(8, 7)?])])
    );

    // A struct result exposes its fields as physical XLS IR port names.
    let reference = format!(
        "import fabric.wire;\n{}",
        dslx_source("h: wire::Holder", "wire::Holder", "h")
    );
    compile_dslx_module(&reference, &[root.path()])?;
    let bx = format!(
        "use fabric::wire;\n{}",
        bx_source("h: wire::Holder", "wire::Holder", "h")
    );
    let options = search_paths(&[root.path()]);
    assert_eq!(
        check_module_with_options(&parse_module(&bx)?, &options).unwrap_err(),
        "block 'sample': output port 'reg' is not a valid XLS IR identifier"
    );
    Ok(())
}

#[test]
fn pinned_dslx_v1_inferred_struct_result_through_a_use_tree() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::create_dir(root.path().join("device"))?;
    fs::write(
        root.path().join("device/fields'.x"),
        "pub struct Item { data: u8 } pub struct Extra' { tag: bool }",
    )?;
    fs::write(
        root.path().join("device/constants.x"),
        "pub const BIAS = u8:1;",
    )?;
    fs::write(
        root.path().join("factory.x"),
        r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use device::{fields'::{Extra', Item}, constants::BIAS};
type Alias = Item;
pub fn make(_x: u8) -> Alias { zero!<Alias>() }
pub fn read(item: Alias) -> u8 { item.data + BIAS }"#,
    )?;
    let body = "let item = factory::make(x); factory::read(item) + x";
    let reference = format!("import factory;\n{}", dslx_source("x: u8", "u8", body));
    let source = format!("use factory;\n{}", bx_source("x: u8", "u8", body));
    let paths = [root.path()];
    let options = search_paths(&paths);
    // The pinned frontend supports nominal use bindings with type inference
    // v1. The v2-only converter reports this annotation as unimplemented.
    let dslx = compile_bundled_module(&reference, &paths);
    let expr = ExpressionEvaluator::compile_block(&source, &options, dslx)?;
    assert_eq!(
        expr.compare(&[IrValue::make_ubits(8, 17)?])?,
        IrValue::make_ubits(8, 18)?
    );
    Ok(())
}

#[test]
fn inferred_struct_result_keeps_its_transitive_definition() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let expr = struct_expression(
        root.path(),
        "use factory;",
        "import factory;",
        "x: u8",
        "u8",
        "let h = factory::make(x); factory::second(h)",
    )?;
    assert_eq!(
        expr.compare(&[IrValue::make_ubits(8, 7)?])?,
        IrValue::make_ubits(8, 8)?
    );
    Ok(())
}

#[test]
fn inferred_nested_tuples_keep_same_named_structs_distinct() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let expr = struct_expression(
        root.path(),
        "use factory;",
        "import factory;",
        "x: u8",
        "(u8, u8)",
        "let p = factory::pair(x); let r = p.1; (factory::second(p.0), factory::other_second(r.0))",
    )?;
    assert_eq!(
        expr.compare(&[IrValue::make_ubits(8, 7)?])?,
        IrValue::make_tuple(&[IrValue::make_ubits(8, 8)?, IrValue::make_ubits(8, 42)?])
    );
    Ok(())
}

#[test]
fn struct_registers_keep_the_checked_dslx_layout_after_sources_are_removed()
-> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let init = "wire::Header { bias: s4:-2, second: u8:9, first: u8:3 }";
    let source = format!(
        r#"
use fabric::wire;
use facade;
block capture(clk, input: wire::Header) -> facade::Header {{
    reg saved: facade::Header = {init};
    saved := facade::swap(input);
    saved
}}
"#
    );
    let checked =
        check_module_with_options(&parse_module(&source)?, &search_paths(&[root.path()]))?;
    let reference = format!("import fabric.wire; fn sample() -> wire::Header {{ {init} }}");
    let expected = compile_dslx_module(&reference, &[root.path()])?.interpret(&[])?;
    root.close()?;
    let package = lower_module(&checked, empty_package("test"), ResetBehavior::default())?;
    let block = package.get_block("capture").ok_or("missing block")?;
    let reset = block.registers[0]
        .reset_value
        .as_ref()
        .ok_or("missing reset")?;
    assert_eq!(value_to_libxls(reset)?, expected);
    assert_eq!(
        block
            .output_ports()
            .map(|p| block.port_name(p))
            .collect::<Vec<_>>(),
        ["first", "second", "bias"]
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn public_structs_preserve_private_nested_values() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("wire.x"),
        r#"
struct Leaf { x: u8 }
struct Internal { pair: (Leaf, bool) }
pub struct Outer { inner: Internal }
pub fn echo(s: Outer) -> Outer { s }
pub fn read(value: Internal) -> u8 { value.pair.0.x }
"#,
    )?;
    let outer = IrValue::parse_typed("((((bits[8]:42), bits[1]:1)))")?;
    for (result, body, expected) in [
        ("wire::Outer", "wire::echo(s)", outer.clone()),
        (
            "u8",
            "wire::echo(s).inner.pair.0.x",
            IrValue::make_ubits(8, 42)?,
        ),
    ] {
        let expr = struct_expression(
            root.path(),
            "use wire;",
            "import wire;",
            "s: wire::Outer",
            result,
            body,
        )?;
        assert_eq!(expr.compare(std::slice::from_ref(&outer))?, expected);
    }
    let options = search_paths(&[root.path()]);
    let private = parse_module("use wire; block b(clk, s: wire::Internal) -> u8 { s.pair.0.x }")?;
    assert!(check_module_with_options(&private, &options).is_err());
    let call =
        parse_module("use wire; block b(clk, s: wire::Outer) -> u8 { wire::read(s.inner) }")?;
    assert_eq!(
        check_module_with_options(&call, &options).unwrap_err(),
        "block 'b': result: cannot pass DSLX struct 'Internal' to an imported function without a public type name"
    );

    fs::write(
        root.path().join("wire.x"),
        r#"
struct Leaf { x: u8 }
struct Internal { pair: (Leaf, bool) }
pub type Payload = Internal;
pub struct Outer { inner: Internal }
pub fn read(value: Internal) -> u8 { value.pair.0.x }
"#,
    )?;
    let expr = struct_expression(
        root.path(),
        "use wire;",
        "import wire;",
        "s: wire::Outer",
        "u8",
        "let p: wire::Payload = s.inner; wire::read(p)",
    )?;
    assert_eq!(expr.compare(&[outer])?, IrValue::make_ubits(8, 42)?);
    Ok(())
}

#[test]
fn both_frontends_reject_private_names_and_layout_compatible_impostors()
-> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let options = search_paths(&[root.path()]);
    let local = (
        "use facade; struct Header { first: u8, second: u8, bias: s4 }",
        "import facade; struct Header { first: u8, second: u8, bias: s4 }",
    );
    let imported = (
        "use fabric::wire; use other; use facade;",
        "import fabric.wire; import other; import facade;",
    );
    for ((bx_decls, dslx_decls), params, expr) in [
        (imported, "h: other::Header", "facade::swap(h).first"),
        (
            imported,
            "h: other::Header",
            "let y: wire::Header = h; y.first",
        ),
        (local, "h: Header", "facade::swap(h).first"),
        (
            ("use factory;", "import factory;"),
            "x: u8",
            "let p = factory::pair(x); let r = p.1; factory::second(r.0)",
        ),
        (
            ("use fabric::wire;", "import fabric.wire;"),
            "h: wire::Private",
            "h.field",
        ),
    ] {
        let bx = format!("{bx_decls}\n{}", bx_source(params, "u8", expr));
        let dslx = format!("{dslx_decls}\n{}", dslx_source(params, "u8", expr));
        assert!(
            check_module_with_options(&parse_module(&bx)?, &options).is_err(),
            "{bx}"
        );
        assert!(
            compile_dslx_module(&dslx, &[root.path()]).is_err(),
            "{dslx}"
        );
    }
    Ok(())
}

#[test]
fn same_named_templates_require_the_correct_declared_result() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let template = r#"
enum Event { Idle, Data(u8), }
pub struct Tag<N: u32> { value: u8 }
pub fn tag<N: u32>(x: Tag<N>) -> u32 { N + (x.value as u32) }
"#;
    fs::write(root.path().join("left.x"), template)?;
    fs::write(root.path().join("right.x"), template)?;
    fs::write(
        root.path().join("factory.x"),
        r#"#![feature(generics)]
import right;
pub fn make<T: type = right::Tag<2>>(_x: u8) -> T { zero!<T>() }
pub fn declared(x: u8) -> right::Tag<2> { right::Tag<2> { value: x } }
"#,
    )?;
    let source = "use left; use right; use factory; \
        block sample(clk, x: u8) -> u32 { right::tag(factory::make(x)) }";
    let options = search_paths(&[root.path()]);
    let declared = source.replace("factory::make", "factory::declared");
    let checked = check_module_with_options(&parse_module(&declared)?, &options)?;
    let block = BlockEvaluator::lower(&checked, &declared)?;
    assert_eq!(
        block.evaluate(&[IrValue::make_ubits(8, 40)?])?,
        IrValue::make_ubits(32, 42)?
    );
    let wrong = declared.replace("right::tag(", "left::tag(");
    assert!(check_module_with_options(&parse_module(&wrong)?, &options).is_err());
    let error = check_module_with_options(&parse_module(source)?, &options)
        .err()
        .ok_or("expected an opaque template diagnostic")?;
    assert!(
        error.contains("name the concrete struct or a public alias"),
        "{error}"
    );
    Ok(())
}

#[test]
fn inferred_parametric_struct_result_exposes_its_specialized_field() -> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let expr = struct_expression(
        root.path(),
        "use fabric::wire;",
        "import fabric.wire;",
        "x: u8",
        "u8",
        "wire::sized(x).data",
    )?;
    let x = IrValue::make_ubits(8, 42)?;
    assert_eq!(expr.compare(std::slice::from_ref(&x))?, x);
    Ok(())
}

#[test]
fn apfloat_arithmetic_infers_a_new_format_from_concrete_alias_ports() -> Result<(), Box<dyn Error>>
{
    let source = r#"
use float32;
use bfloat16;
use apfloat;
block sample(clk, x: float32::F32, y: bfloat16::BF16) -> u40 {
    let sum = apfloat::add(x, float32::one(false));
    let product = apfloat::full_precision_mul(sum, y);
    apfloat::flatten(product)
}"#;
    let checked = check_module_with_options(&parse_module(source)?, &ImportOptions::default())?;
    let block = BlockEvaluator::lower(&checked, source)?;
    // (0.5 + 1.0) * 2.0 = 3.0, with an 8-bit exponent and 31 fraction bits.
    let x = IrValue::parse_typed("(bits[1]:0, bits[8]:126, bits[23]:0)")?;
    let y = IrValue::parse_typed("(bits[1]:0, bits[8]:128, bits[7]:0)")?;
    assert_eq!(
        block.evaluate(&[x, y])?,
        IrValue::make_ubits(40, 0x0040_4000_0000)?
    );
    Ok(())
}

#[test]
fn apfloat_direct_spelling_and_float32_alias_share_a_function_and_block_signature()
-> Result<(), Box<dyn Error>> {
    let helper = r#"fn copy(p: float32::F32) -> apfloat::APFloat<u32:8, 23> {
    apfloat::APFloat<8, u32:23> { fraction: p.fraction, sign: p.sign, bexp: p.bexp }
}"#;
    let params = "x: apfloat::APFloat<8, 23>";
    let result = "apfloat::APFloat<u32:8, u32:23>";
    let body = "let copied: float32::F32 = copy(x); float32::negate(copied)";
    let source = format!(
        "use float32; use apfloat;\n{helper}\n{}",
        bx_source(params, result, body)
    );
    let native = format!(
        "import float32; import apfloat;\n{helper}\n{}",
        dslx_source(params, result, body)
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &source,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    let x = IrValue::parse_typed("(bits[1]:1, bits[8]:128, bits[23]:0x123456)")?;
    assert_eq!(
        evaluator.compare(&[x])?,
        IrValue::parse_typed("(bits[1]:0, bits[8]:128, bits[23]:0x123456)")?
    );
    Ok(())
}

#[test]
fn parametric_struct_aliases_preserve_bindings_across_nested_results_and_lowering()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("tags.x"),
        r#"
pub struct Tagged<W: u32, N: u32 = {W + u32:1}> { x: uN[W] }
pub type First = Tagged<u32:8, u32:9>;
pub type Second = Tagged<u32:8, u32:10>;
pub fn tag<W: u32, N: u32>(_value: Tagged<W, N>) -> u32 { N }
pub fn pair(a: First, b: Second) -> (First[1], Second) { ([a], b) }
"#,
    )?;
    fs::write(
        root.path().join("facade.x"),
        "import tags; pub type Again = tags::First;",
    )?;
    let options = search_paths(&[root.path()]);
    let source = r#"
use tags;
use tags::Tagged;
use facade::Again;
fn local_tag<W: u32, N: u32>(_value: Tagged<W, N>) -> u32 { N }
fn local_width<W: u32, N: u32>(_value: tags::Tagged<W, N>) -> u32 { W }
block sample(clk, a: Again, b: Tagged<u32:8, 10>) -> (u32, u32, u32, u32, u32, u32) {
    let p = tags::pair(a, b);
    let first: tags::Tagged<8> = p.0[u32:0];
    let copy = Tagged<8, u32:9> { x: first.x };
    (tags::tag(first), tags::tag(copy), tags::tag(p.1), local_tag(a), local_tag(p.1), local_width(copy))
}"#;
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    assert_eq!(
        checked
            .imported_calls()
            .iter()
            .filter(|call| call.function == "tag")
            .count(),
        2
    );

    for binding in [
        "let b: tags::Tagged<8, 10> = a;",
        "let b: tags::Tagged<8> = tags::Tagged<8, 10> { x: a.x };",
    ] {
        let wrong = format!(
            "use tags; use facade::Again; block b(clk, a: Again) -> u8 {{ {binding} b.x }}"
        );
        assert!(
            check_module_with_options(&parse_module(&wrong)?, &options).is_err(),
            "{wrong}"
        );
    }
    root.close()?;
    let block = BlockEvaluator::lower(&checked, source)?;
    let value = IrValue::parse_typed("(bits[8]:7)")?;
    assert_eq!(
        block.evaluate(&[value.clone(), value])?,
        IrValue::parse_typed(
            "(bits[32]:9, bits[32]:9, bits[32]:10, bits[32]:9, bits[32]:10, bits[32]:8)"
        )?
    );
    Ok(())
}

#[test]
fn generic_block_register_resolves_a_direct_struct_type_and_zero_reset()
-> Result<(), Box<dyn Error>> {
    let root = struct_modules()?;
    let options = search_paths(&[root.path()]);
    let source = r#"
use fabric::wire;
use fabric::wire::Sized;
block store<T>(clk, x: T) -> T {
    reg q: Sized<8> = zero!<wire::Sized<u32:8>>();
    q := x;
    q
}
pub block sample(clk, x: u8) -> u8 {
    let saved = store(wire::sized(x));
    saved.data
}
"#;
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    let reset = checked
        .blocks()
        .iter()
        .flat_map(|b| &b.statements)
        .find_map(|stmt| match stmt {
            CheckedStmt::Reg { name, reset, .. } if name == "q" => reset.as_ref(),
            _ => None,
        });
    assert_eq!(reset, Some(&IrValue::parse_typed("(bits[8]:0)")?));
    let ir = lower_module(
        &checked,
        empty_package("direct_register"),
        ResetBehavior::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&ir.to_string(), None)?;
    Ok(())
}

#[test]
fn dslx_struct_and_field_names_preserve_apostrophes_and_bangs() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("wire.x"),
        r#"
enum Event { Idle, Data(u8), }
pub struct Header' { tag': u8, tag!: u8 }
pub fn read(h: Header') -> u8 { h.tag' - h.tag! }
"#,
    )?;
    let expr = struct_expression(
        root.path(),
        "use wire;",
        "import wire;",
        "x: u8",
        "u8",
        "let h: wire::Header' = wire::Header' { tag!: u8:1, tag': x }; wire::read(h) + h.tag!",
    )?;
    let x = IrValue::make_ubits(8, 19)?;
    assert_eq!(expr.compare(std::slice::from_ref(&x))?, x);
    Ok(())
}

#[test]
fn bang_field_followed_by_equality_matches_dslx() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(root.path().join("wire.x"), "pub struct Item { flag!: u8 }")?;
    let expr = struct_expression(
        root.path(),
        "use wire;",
        "import wire;",
        "s: wire::Item, y: u8",
        "bool",
        "s.flag!==y",
    )?;
    let s = IrValue::make_tuple(&[IrValue::make_ubits(8, 3)?]);
    for (y, equal) in [(3, true), (4, false)] {
        assert_eq!(
            expr.compare(&[s.clone(), IrValue::make_ubits(8, y)?])?,
            IrValue::bool(equal)
        );
    }
    Ok(())
}
