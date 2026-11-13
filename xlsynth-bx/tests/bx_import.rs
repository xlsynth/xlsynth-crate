// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs, io, path::Path};
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::ast::{Expr, Stmt};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module_with_options, check_source_with_options,
    compile_with_options, empty_package, lower_module, parse_module, run_tests,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

/// Write a BX or DSLX module in a test-specific import tree.
fn write_module(root: &Path, name: &str, source: &str) -> io::Result<()> {
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, source)
}

fn compile_design(source: &str, options: &ImportOptions<'_>) -> Result<Package, String> {
    compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        options,
    )
}

#[test]
fn imported_block_uses_stage_constants_from_its_defining_module() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "stages.bx",
        r#"
const START = u32:0;
pub const SAVED = u32:1;
pub block delay(clk, x: u8 @START) -> u8 @SAVED { reg q: u8; q := x; q }
"#,
    )?;
    let source = r#"
use stages;
const START = u32:4;
const SAVED = u32:6;
block top(clk, x: u8 @START) -> u8 @SAVED {
    let q = stages::delay(x);
    reg r: u8; r := q; r
}
block first(clk, x: u8) -> u8 @stages::SAVED { let q = stages::delay(x); q }
"#;
    check_module_with_options(&parse_module(source)?, &search_paths(&[dir.path()]))?;
    Ok(())
}

#[test]
fn explicit_reset_permission_belongs_to_the_defining_file() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    let permission = "#![allow(block_reset_signal)]";
    let child = "pub block hold<T>(clk, rst, x: T) -> T { reg q: T; if !rst { q := x; } q }";
    let parent = "use stages; block pipeline(clk, x: u8) -> u8 { let q = stages::hold(x); q }";
    write_module(dir.path(), "stages.bx", &format!("{permission}\n{child}"))?;
    let package = compile_design(parent, &options)?;
    assert!(package.get_block("pipeline").unwrap().reset.is_some());

    let explicit_parent = parent.replace("pipeline(clk,", "pipeline(clk, rst,");
    let diagnostic = "declaring an explicit reset signal requires #![allow(block_reset_signal)] in the defining module; omit `rst` to infer reset";
    let error = compile_design(&explicit_parent, &options).unwrap_err();
    assert_eq!(error, format!("block 'pipeline': {diagnostic}"));

    write_module(dir.path(), "stages.bx", child)?;
    let error = compile_design(&format!("{permission}\n{explicit_parent}"), &options).unwrap_err();
    assert_eq!(error, format!("block 'stages::hold': {diagnostic}"));
    Ok(())
}

#[test]
fn test_register_defaults_are_independent_of_imported_zero_functions() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    write_module(dir.path(), "values.bx", "pub fn zero!() -> u8 { u8:7 }")?;
    let options = search_paths(&[dir.path()]);
    let source = r#"
use values::zero!;
#[test] block drivers() {
    reg implicit: u8;
    reg explicit: u8 = zero!<u8>();
    assert_eq(implicit, u8:0);
    assert_eq(explicit, u8:7);
}
#[test] fn explicit_call() { assert_eq(zero!<u8>(), u8:7); }
"#;
    let report = run_tests(
        source,
        Path::new("driver_tests.bx"),
        empty_package("suite"),
        ResetBehavior::default(),
        &options,
        true,
    )?;
    assert_eq!(report.tests.len(), 2);
    for test in report.tests {
        assert_eq!(test.failure, None, "{}", test.name);
    }
    Ok(())
}

#[test]
fn imported_tuple_match_uses_the_library_constant() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "flags.bx",
        r#"
const TARGET = u8:73;
pub fn decode(x: u8) -> u8 {
    match (true, (x, ())) {
        (true, (TARGET, ())) => u8:1,
        _ => u8:0,
    }
}
"#,
    )?;
    let source = r#"
use flags;
const TARGET = u8:74;
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 { flags::decode(x) }
"#;
    let options = search_paths(&[dir.path()]);
    let package = compile_design(source, &options)?;
    assert_eq!(run(package.clone(), 73)?, IrValue::make_ubits(8, 1)?);
    assert_eq!(run(package, 74)?, IrValue::make_ubits(8, 0)?);
    Ok(())
}

/// Evaluate the selected block, including its imported child instances.
fn run(mut package: Package, x: u8) -> Result<IrValue, Box<dyn Error>> {
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    inline_all_blocks_in_package(&mut package)?;
    let top = package.get_top_block().ok_or("missing top block")?;
    let function = combinational_block_to_fn(top)?;
    let input = IrValue::make_ubits(8, u64::from(x))?;
    Ok(eval_fn_in_package_value(&package, &function, &[input]))
}

#[test]
fn imported_generic_block_checks_the_stage_of_a_destructured_register() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    let caller = "use pipe::saved; block top(clk, x: u8 @4) -> u8 @5 { let q = saved(x); q }";
    let pipe = r#"
pub block saved<T>(clk, x: T @2) -> T @3 {
    reg pair: (T, T);
    pair := (x, x);
    let (head, _) = pair;
    head
}
"#;
    write_module(dir.path(), "pipe.bx", pipe)?;
    check_source_with_options(caller, &options)?;
    write_module(dir.path(), "pipe.bx", &pipe.replace("@3", "@4"))?;
    assert_eq!(
        check_source_with_options(caller, &options).unwrap_err(),
        "block 'pipe::saved' (pipe::saved__0): expected cycle 4, got cycle 3"
    );
    Ok(())
}

#[test]
fn overwritten_literal_dimensions_resolve_in_library_source_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(dir.path(), "sizes.bx", "pub const W = u8:3;")?;
    let options = search_paths(&[dir.path()]);
    let root = "use lib; const ROOT_ONLY = u32:2; fn sample()->u8 { lib::pick() }";
    let pick = "pub fn pick()->u8 { u8:(uN[sizes::W]:129) }";
    write_module(dir.path(), "lib.bx", &format!("use sizes; {pick}"))?;
    check_source_with_options(root, &options)?;
    for library in [
        format!("{pick} use sizes;"),
        "pub fn pick()->u8 { u8:(uN[ROOT_ONLY]:129) }".to_owned(),
    ] {
        write_module(dir.path(), "lib.bx", &library)?;
        assert!(
            check_source_with_options(root, &options).is_err(),
            "{library}"
        );
    }
    Ok(())
}

#[test]
fn library_dimensions_keep_their_defining_scope() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "shape.bx",
        r#"
const WIDTH = u32:7;
pub const LANES = u32:2;
pub struct Packet { row: uN[WIDTH][LANES] }
pub fn sum(p: Packet) -> uN[WIDTH] {
    for (i, acc) in u32:0..LANES { acc + p.row[i] }(uN[WIDTH]:0)
}
"#,
    )?;
    let source = r#"
use shape;
const WIDTH = u32:5;
const LANES = u32:3;
#[allow(combinational_block)]
pub block circuit(clk, x:u8)->u8 {
    let row: u7[shape::LANES] = [x as u7, (x as uN[WIDTH]) as u7];
    shape::sum(shape::Packet { row: row }) as u8
}
"#;
    let options = search_paths(&[dir.path()]);
    assert_eq!(
        run(compile_design(source, &options)?, 228)?,
        IrValue::make_ubits(8, 104)?
    );
    let private = "use shape; fn bad(x:uN[shape::WIDTH])->u7 { x }";
    assert!(check_source_with_options(private, &options).is_err());
    Ok(())
}

#[test]
fn imported_array_fill_resolves_its_dimension_and_last_value_in_the_library()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "lanes.bx",
        r#"
const COUNT = u32:4;
fn step(x: u8) -> u8 { x + u8:1 }
pub fn spread(x: u8) -> u8[COUNT] { u8[COUNT]:[0, step(x), ...] }
"#,
    )?;
    let source = r#"
use lanes;
const COUNT = u32:2;
fn step(x: u8) -> u8 { x - u8:1 }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 { lanes::spread(x)[3] }
"#;
    let package = compile_design(source, &search_paths(&[dir.path()]))?;
    assert_eq!(run(package, 41)?, IrValue::make_ubits(8, 42)?);
    Ok(())
}

#[test]
fn library_struct_arguments_keep_their_defining_zero_binding() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(dir.path(), "widths.bx", "pub fn zero!() -> u32 { u32:7 }")?;
    let options = search_paths(&[dir.path()]);
    let source = r#"
use wire;
fn zero!() -> u32 { u32:5 }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 {
    let packet = wire::Packet { lane: wire::Lane<7> { value: x as u7 } };
    wire::apply<wire::read>(packet.lane) as u8
}
"#;
    for binding in ["fn zero!() -> u32 { u32:7 }", "use widths::zero!;"] {
        write_module(
            dir.path(),
            "wire.bx",
            &format!(
                r#"
{binding}
pub struct Lane<N: u32> {{ value: uN[N] }}
pub struct Packet {{ lane: Lane<{{zero!<u32>()}}> }}
pub fn read(x: Lane<{{zero!<u32>()}}>) -> u7 {{ x.value }}
pub fn apply<F: fn(Lane<{{zero!<u32>()}}>) -> u7>(x: Lane<7>) -> u7 {{ F(x) }}
extern fn external_read(x: Lane<{{zero!<u32>()}}>) -> u7;
extern block external_stage(clk, x: Lane<{{zero!<u32>()}}>) -> u7;
fn read_external(x: Lane<7>) -> u7 {{ external_read(x) }}
block stage_external(clk, x: Lane<7>) -> u7 {{ let y = external_stage(x); y }}
"#
            ),
        )?;
        assert_eq!(
            run(compile_design(source, &options)?, 228)?,
            IrValue::make_ubits(8, 100)?
        );
    }
    Ok(())
}

#[test]
fn public_bx_aliases_preserve_private_types_across_module_and_member_uses()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    write_module(
        dir.path(),
        "device.bx",
        r#"
const WIDTH = u32:7;
type Payload = uN[WIDTH];
enum State: u8 { Ready = 3, Busy = 5 }
struct Frame<T> { data: T, state: State }
pub type Packet = Frame<Payload>;
pub type Status = State;
"#,
    )?;
    write_module(
        dir.path(),
        "bus.bx",
        r#"
use device;
pub type Data = device::Packet;
pub type Phase = device::Status;
"#,
    )?;
    let root = r#"
use bus;
use bus::Phase;
type Payload = u16;
enum State: u8 { Ready = 20 }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 {
    let p: bus::Data = bus::Data { data: x as u7, state: Phase::Ready };
    match p.state {
        bus::Phase::Ready => (p.data as u8) + (p.state as u8),
        _ => Phase::Busy as u8,
    }
}
"#;
    assert_eq!(
        run(compile_design(root, &options)?, 250)?,
        IrValue::make_ubits(8, 125)?
    );
    for source in [
        "use device::Payload;",
        "use device; fn bad(x:device::State) {}",
        "use bus; fn bad(x:bus::Packet) {}",
        "use bus::Phase; type Phase = u8;",
        "use bus::Data; fn bad() -> u8 { Data::Ready as u8 }",
    ] {
        assert!(
            check_source_with_options(source, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn library_define_before_use_checks_alias_references_and_their_dimensions()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    let path = dir.path().join("lib.bx");
    let declarations = "pub block take(clk, x: Word) -> Word { x }\npub type Word = u8;";
    fs::write(&path, declarations)?;
    let source = "use lib; block top(clk, x: u8) -> u8 { let y = lib::take(x); y }";
    check_source_with_options(source, &options)?;
    for (declarations, owner, referenced) in [
        (declarations, "take", "Word"),
        (
            "type Pair = (Word, Word); pub type Word = u8;",
            "Pair",
            "Word",
        ),
        (
            "type Word = uN[WIDTH + u32:1]; const WIDTH = u32:7;",
            "Word",
            "WIDTH",
        ),
    ] {
        fs::write(
            &path,
            format!("#![require_define_before_use]\n{declarations}"),
        )?;
        assert_eq!(
            check_source_with_options("use lib;", &options).unwrap_err(),
            format!(
                "BX import 'lib' ({}): in '{owner}': '{referenced}' is declared later in the module",
                path.display()
            )
        );
    }
    Ok(())
}

#[test]
fn imported_bx_enum_members_resolve_through_module_and_member_uses() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "device/phase.bx",
        r#"
const BASE = u8:2;
fn step(x:u8) -> u8 { x + u8:1 }
pub enum Mode : u8 { First = BASE, Next = step(BASE) }
enum Hidden : u8 { A = 0 }
pub fn decode(x:Mode) -> u8 { x as u8 }
"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let bx = r#"
use device::phase;
use device::phase::Mode;
#[allow(combinational_block)]
block circuit(clk, x:u8) -> u8 {
    let m = if x == u8:0 { Mode::First } else { phase::Mode::Next };
    match m { phase::Mode::First => phase::decode(m), _ => Mode::Next as u8 }
}

"#;
    let package = compile_design(bx, &options)?;
    assert_eq!(run(package.clone(), 0)?, IrValue::make_ubits(8, 2)?);
    assert_eq!(run(package, 1)?, IrValue::make_ubits(8, 3)?);

    for source in [
        "use device::phase; fn bad() -> u8 { phase::Hidden::A as u8 }",
        "use device::phase; fn bad(phase:u8) -> u8 { phase::Mode::First as u8 }",
        "use device::phase; enum Mode:u8 { Next=3 } fn bad() -> u8 { phase::decode(Mode::Next) }",
    ] {
        assert!(
            check_source_with_options(source, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn a_library_enum_hides_map_starting_at_its_declaration() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "lib.bx",
        r#"
fn inc(x:u8)->u8 { x+u8:1 }
#[allow(combinational_block)]
pub block earlier(clk, x:u8)->u8 { map({ [x] }, inc)[u32:0] }
pub enum map:u8 { A=0 }
"#,
    )?;
    let source = r#"
use lib;
block circuit(clk, x:u8)->u8 { let y=lib::earlier(x); y }
"#;
    let options = search_paths(&[dir.path()]);
    assert_eq!(
        run(compile_design(source, &options)?, 12)?,
        IrValue::make_ubits(8, 13)?
    );
    let hidden =
        "fn inc(x:u8)->u8 { x } enum map:u8 { A=0 } fn f(xs:u8[1])->u8[1] { map(xs, inc) }";
    assert!(check_source_with_options(hidden, &options).is_err());
    Ok(())
}

#[test]
fn a_dslx_enum_use_shadows_map_in_block_declaration_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    write_module(dir.path(), "ops.x", "pub enum map:u8 { A=0 }")?;
    write_module(dir.path(), "math.x", "pub fn inc(x:u8)->u8 { x+u8:1 }")?;
    write_module(
        dir.path(),
        "types.x",
        "pub struct Box<N:u32> { x:uN[N] } pub fn two(x:Box<2>)->u8 { (x.x as u8)+u8:2 }",
    )?;
    let check_both = |source: &str, accepted: bool| -> Result<(), Box<dyn Error>> {
        write_module(dir.path(), "helper.bx", source)?;
        let root = "use helper; block run(clk,x:u8)->u8 { let y=helper::circuit(x); y }";
        for program in [source, root] {
            let result = compile_design(program, &options);
            if accepted {
                assert_eq!(run(result?, 1)?, IrValue::make_ubits(8, 2)?);
            } else {
                assert!(result.is_err(), "{source}");
            }
        }
        Ok(())
    };
    for (call, callback_use) in [
        ("map([x], |v| v+u8:1)", ""),
        ("map([x], math::inc)", "use math;"),
        ("map({ [x] }, math::inc)", "use math;"),
        // In a pure operand, (u8) groups the first field of this tuple
        // annotation.
        (
            "map({ let (y,_):((u8),u8)=(x,x); [y] }, math::inc)",
            "use math;",
        ),
        ("map((u8:1..u8:2), math::inc)", "use math;"),
        ("map([zero!<types::Box<2>>()], types::two)", "use types;"),
    ] {
        let body = format!(
            "#[allow(combinational_block)] pub block circuit(clk,x:u8)->u8 {{ {call}[u32:0] }}"
        );
        for (source, accepted) in [
            (format!("{body}\nuse ops::map; {callback_use}"), true),
            (format!("use ops::map;\n{body}\n{callback_use}"), false),
        ] {
            check_both(&source, accepted)?;
        }
    }
    let fold = "unroll_for! (_,a) in u32:0..u32:0 { a }";
    // In a pure map operand, postfix indexing of the whole fold requires
    // parentheses. Ordinary BX call arguments permit the index directly.
    check_both(
        &format!(
            "use math; #[allow(combinational_block)] pub block circuit(clk,x:u8)->u8 {{ map({fold}([[x]])[u32:0], math::inc)[u32:0] }} use ops::map;"
        ),
        false,
    )?;
    write_module(dir.path(), "ops.x", "pub fn map(a:u8,b:u8)->u8 { a+b }")?;
    let ordinary_fold = format!("map({fold}([x])[u32:0],x)");
    for (body, accepted) in [
        ("map(x,x)", true),
        (&ordinary_fold, true),
        // Ordinary block tuple patterns treat (u8) as a singleton tuple.
        ("map({ let (y,_):((u8),u8)=((x,),x); y.0 },x)", true),
        ("map({ x },x)", false),
    ] {
        let source = format!(
            "#[allow(combinational_block)] pub block circuit(clk,x:u8)->u8 {{ {body} }} use ops::map;"
        );
        check_both(&source, accepted)?;
    }
    Ok(())
}

#[test]
fn nested_module_and_grouped_members_share_structs_and_child_blocks() -> Result<(), Box<dyn Error>>
{
    let root = tempdir()?;
    write_module(
        root.path(),
        "device/lanes.bx",
        r#"
pub const BIAS = u8:4;
pub struct Sample { value: u8 }
fn bump(x: u8) -> u8 { let (BIAS, extra) = (x, BIAS); BIAS + extra }
pub fn pack(x: u8) -> Sample { Sample { value: bump(x) } }
#[allow(combinational_block)]
pub block stage(clk, x: u8) -> u8 { pack(x).value }
"#,
    )?;
    let source = r#"
use device::lanes;
use device::lanes::{Sample, pack, BIAS};
block circuit(clk, x: u8) -> (u8, u8, u8) {
    let first = lanes::stage(x);
    let second = lanes::stage(first);
    let p: Sample = pack(x);
    let q: lanes::Sample = Sample { value: BIAS + lanes::BIAS };
    (second, p.value, q.value)
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_design(source, &options)?;
    assert_eq!(
        package.get_top_block().ok_or("missing top")?.name,
        "circuit"
    );
    assert_eq!(
        run(package, 5)?,
        IrValue::parse_typed("(bits[8]:13, bits[8]:9, bits[8]:8)")?
    );
    Ok(())
}

#[test]
fn diamond_dependencies_keep_private_bindings_and_dslx_imports_in_their_own_modules()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    for (name, body) in [
        (
            "common/shared.bx",
            "fn helper(x: u8) -> u8 { x + u8:1 } pub fn base(x: u8) -> u8 { helper(x) }",
        ),
        ("west/arithmetic.x", "pub fn step(x: u8) -> u8 { x + u8:2 }"),
        ("east/arithmetic.x", "pub fn step(x: u8) -> u8 { x * u8:3 }"),
        (
            "left.bx",
            r#"
use common::shared;
use west::arithmetic;
fn helper(x: u8) -> u8 { arithmetic::step(x) }
pub fn apply(x: u8) -> u8 { helper(shared::base(x)) }"#,
        ),
        (
            "right.bx",
            r#"
use common::shared::base;
use east::arithmetic;
fn helper(x: u8) -> u8 { arithmetic::step(x) }
pub fn apply(x: u8) -> u8 { helper(base(x)) }"#,
        ),
    ] {
        write_module(root.path(), name, body)?;
    }
    let source = r#"
use left;
use right;
fn helper(x: u8) -> u8 { !x }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> (u8, u8, u8) {
    (left::apply(x), right::apply(x), helper(x))
}
"#;
    let options = search_paths(&[root.path()]);
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    assert_eq!(checked.functions().len(), 7);
    assert_eq!(checked.imported_calls().len(), 2);
    let fresh = compile_design(source, &options)?;
    drop(root);
    let saved = lower_module(&checked, empty_package("test"), ResetBehavior::default())?;
    assert_eq!(saved.to_string(), fresh.to_string());
    assert_eq!(
        run(saved, 5)?,
        IrValue::parse_typed("(bits[8]:8, bits[8]:18, bits[8]:250)")?
    );
    Ok(())
}

#[test]
fn imported_block_templates_distinguish_callbacks_with_the_same_signature()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "pipe.bx",
        r#"
pub block apply<In, Out, F: fn(In) -> Out>(clk, x: In) -> Out { F(x) }
pub fn widen(x: u8) -> u16 { (x as u16) + u16:1 }
"#,
    )?;
    let source = r#"
use pipe;
fn widen(x: u8) -> u16 { (x as u16) + u16:2 }
pub block circuit(clk, x: u8) -> (u16, u16) {
    let a = pipe::apply<pipe::widen>(x);
    let b = pipe::apply<widen>(x);
    (a, b)
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_design(source, &options)?;
    assert_eq!(
        run(package, 255)?,
        IrValue::parse_typed("(bits[16]:256, bits[16]:257)")?
    );
    Ok(())
}

#[test]
fn imported_value_parametrics_resolve_in_their_lexical_modules() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "math.bx",
        r#"
const STEP = u32:2;
const N = u32:7;
pub fn own<N: u32 = {N}>() -> u32 { N }
pub fn widen<N: u32, W: u32 = {N + STEP}>(x: uN[N]) -> uN[W] {
    (x as uN[W]) + (STEP as uN[W])
}
pub fn narrow<N: u32>(x: uN[N]) -> u5 {
    let N = u32:3;
    widen<N>(x as uN[N])
}
"#,
    )?;
    write_module(
        dir.path(),
        "mid.bx",
        r#"
use math::widen;
const STEP = u32:4;
pub fn apply(x: u8) -> u12 { widen<8, {8 + STEP}>(x) }
"#,
    )?;
    let source = r#"
use math;
use mid::apply;
const STEP = u32:6;
#[allow(combinational_block)]
pub block circuit(clk, x: u8) -> (u10, u12, u14, u5) {
    (math::widen(x), apply(x), math::widen<8, {8 + STEP}>(x), math::narrow(x))
}
"#;
    let options = search_paths(&[dir.path()]);
    assert_eq!(
        run(compile_design(source, &options)?, 19)?,
        IrValue::parse_typed("(bits[10]:21, bits[12]:21, bits[14]:21, bits[5]:5)")?
    );
    let defaulted = "use math; const N = u32:9; fn f() -> u32 { math::own() }";
    assert!(check_source_with_options(defaulted, &options).is_err());
    check_source_with_options(&defaulted.replace("own()", "own<4>()"), &options)?;
    Ok(())
}

#[test]
fn imported_templates_keep_callback_and_value_actuals_in_their_respective_scopes()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "library.bx",
        r#"
pub fn apply<N: u32, F: fn(uN[N]) -> uN[N]>(x: uN[N]) -> uN[N] { F(x) }
pub fn inc(x: u8) -> u8 { x + u8:1 }
pub const W = u32:8;
"#,
    )?;
    let source = r#"
use library::apply;
#[allow(combinational_block)]
pub block circuit(clk, x: u8) -> u8 { apply<8, inc>(x) }
use library::inc;
use library::W;
"#;
    let options = search_paths(&[dir.path()]);
    assert_eq!(
        run(compile_design(source, &options)?, 19)?,
        IrValue::make_ubits(8, 20)?
    );
    let later_value = source.replace("apply<8,", "apply<W,");
    let error = check_source_with_options(&later_value, &options).unwrap_err();
    assert!(error.contains("declared later"), "{error}");
    Ok(())
}

#[test]
fn bx_libraries_keep_their_extern_link_symbols() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "ops.bx",
        r#"
extern fn unused(x: u8) -> u8;
extern fn adjust(x: u8) -> u8 = "linked_adjust";
extern fn identity(x: u8) -> u8;
extern fn chip::invert(x: u8) -> u8;
extern block bridge(clk, x: u8) -> u8;
pub block apply(clk, x: u8) -> u8 {
    let y = bridge(identity(x));
    adjust(chip::invert(y))
}
"#,
    )?;
    write_module(
        root.path(),
        "other.bx",
        r#"
extern fn adjust(x: u8) -> u8 = "identity";
extern fn bump(x: u8) -> u8;
extern block bridge(clk, x: u8) -> u8;
pub block apply(clk, x: u8) -> u8 {
    let y = bridge(x);
    adjust(bump(y))
}
"#,
    )?;
    let source = r#"
use ops;
use other;
fn adjust(x: u8) -> u8 { x }
block circuit(clk, x: u8) -> (u8, u8, u8) {
    let y = ops::apply(x);
    let z = other::apply(x);
    (adjust(x), y, z)
}
"#;
    let supplied = Parser::new(
        r#"package test
fn linked_adjust(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = not(x, id=2) }
fn identity(x: bits[8] id=3) -> bits[8] { ret v: bits[8] = identity(x, id=4) }
fn __chip__invert(x: bits[8] id=5) -> bits[8] { ret n: bits[8] = not(x, id=6) }
block bridge(clk: clock, x: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=7)
  v: bits[8] = neg(x, id=8)
  out: () = output_port(v, name=out, id=9)
}
fn __test__bump(x: bits[8] id=10) -> bits[8] {
  one: bits[8] = literal(value=1, id=11)
  ret sum: bits[8] = add(x, one, id=12)
}
"#,
    )
    .parse_and_verify_package()?;
    let options = search_paths(&[root.path()]);
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    assert_eq!(
        checked
            .externs()
            .iter()
            .map(|f| (f.name.as_str(), f.symbol.as_deref()))
            .collect::<Vec<_>>(),
        [
            ("adjust", Some("linked_adjust")),
            ("identity", None),
            ("chip::invert", None),
            ("adjust", Some("identity")),
            ("bump", None),
        ]
    );
    assert_eq!(
        checked
            .extern_blocks()
            .iter()
            .map(|b| b.name.as_str())
            .collect::<Vec<_>>(),
        ["bridge", "bridge"]
    );
    let package = lower_module(&checked, supplied, ResetBehavior::default())?;
    assert_eq!(
        run(package, 3)?,
        IrValue::parse_typed("(bits[8]:3, bits[8]:253, bits[8]:254)")?
    );
    Ok(())
}

#[test]
fn private_members_missing_members_and_implicit_transitive_imports_are_rejected()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(root.path(), "hidden.bx", "pub fn marker(x: u8) -> u8 { x }")?;
    write_module(
        root.path(),
        "lib.bx",
        r#"
use hidden::marker;
const BIAS = u8:1;
struct Payload { value: u8 }
fn adjust(x: u8) -> u8 { marker(x) }
block child(clk, x: u8) -> u8 { adjust(x) }
pub fn expose(x: u8) -> u8 { adjust(x) + BIAS }
"#,
    )?;
    let options = search_paths(&[root.path()]);
    for member in ["BIAS", "Payload", "adjust", "child", "marker", "absent"] {
        let source = format!("use lib::{member}; block circuit(clk, x: u8) -> u8 {{ x }}");
        let error = check_module_with_options(&parse_module(&source)?, &options).unwrap_err();
        assert!(error.contains(member), "{source}\n{error}");
    }
    for callee in ["lib::adjust", "lib::marker", "marker"] {
        let source = format!("use lib; block circuit(clk, x: u8) -> u8 {{ {callee}(x) }}");
        assert!(
            check_module_with_options(&parse_module(&source)?, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn transitive_bx_use_cycles_report_the_modules_involved() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(root.path(), "first.bx", "use second;")?;
    write_module(root.path(), "second.bx", "use first;")?;
    let options = search_paths(&[root.path()]);
    let module = parse_module("use first; block circuit(clk, x: u8) -> u8 { x }")?;
    let error = check_module_with_options(&module, &options).unwrap_err();
    assert!(
        error.contains("cyclic BX import: first -> second -> first"),
        "{error}"
    );
    Ok(())
}

#[test]
fn an_independent_export_omits_unused_functions_blocks_and_prelinked_externs()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "arithmetic.x",
        r#"
pub fn increment(x: u8) -> u8 { x + u8:1 }
pub fn subtract(x: u8) -> u8 { x - u8:1 }
"#,
    )?;
    write_module(
        root.path(),
        "lanes.bx",
        r#"
use arithmetic;
extern fn external_score(x: u8) -> u8;
extern block external_stage(clk, x: u8) -> u8;
fn score(x: u8) -> u8 { external_score(arithmetic::subtract(x)) }
pub block scoring(clk, x: u8) -> u8 { let y = external_stage(x); score(y) }
fn step(x: u8) -> u8 { arithmetic::increment(x) }
pub fn adjust(xs: u8[1]) -> u8[1] {
    map(xs, |x| for (i, carry) in u8:0..u8:1 {
        match x {
            u8:0..u8:0 => external_score(x),
            _ => step(carry + i),
            u8:9 => external_score(x),
        }
    }(x))
}
"#,
    )?;
    let options = search_paths(&[root.path()]);
    let source = "use lanes; block circuit(clk, x: u8) -> u8 { lanes::adjust([x])[u32:0] }";
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    assert_eq!(checked.functions().len(), 2);
    assert_eq!(checked.blocks().len(), 1);
    assert!(checked.extern_blocks().is_empty() && checked.externs().is_empty());
    let package = lower_module(&checked, empty_package("test"), ResetBehavior::default())?;
    assert!(package.get_fn("__arithmetic__subtract").is_none());
    assert_eq!(run(package, 73)?, IrValue::make_ubits(8, 74)?);

    let source = source.replace("lanes::adjust([x])[u32:0]", "let y = lanes::scoring(x); y");
    assert!(
        compile_design(&source, &options)
            .unwrap_err()
            .contains("external_score")
    );
    let root_extern =
        "use lanes; extern fn root_score(x: u8) -> u8; block circuit(clk, x: u8) -> u8 { x }";
    assert!(
        compile_design(root_extern, &options)
            .unwrap_err()
            .contains("root_score")
    );
    Ok(())
}

#[test]
fn a_member_use_ignores_an_unrelated_bx_file_named_after_the_member() -> Result<(), Box<dyn Error>>
{
    let root = tempdir()?;
    let caller = "use arithmetic::bump; #[allow(combinational_block)] block circuit(clk, x: u8) -> u8 { bump(x) }";
    write_module(root.path(), "bump.bx", caller)?;
    let options = search_paths(&[root.path()]);
    for ext in ["bx", "x"] {
        write_module(
            root.path(),
            &format!("arithmetic.{ext}"),
            "pub fn bump(x: u8) -> u8 { x + u8:1 }",
        )?;
        assert_eq!(
            run(compile_design(caller, &options)?, 3)?,
            IrValue::make_ubits(8, 4)?
        );
    }
    Ok(())
}

#[test]
fn search_order_prefers_full_module_paths_and_existing_dslx_modules() -> Result<(), Box<dyn Error>>
{
    let first = tempdir()?;
    let second = tempdir()?;
    let source = "use device::probe; #[allow(combinational_block)] block circuit(clk, x: u8) -> u8 { probe::tag(x) }";
    for (dir, offset) in [(first.path(), 1), (second.path(), 7)] {
        write_module(
            dir,
            "device/probe.bx",
            &format!("pub fn tag(x: u8) -> u8 {{ x + u8:{offset} }}"),
        )?;
    }
    // A full module path wins over the same member name in a parent module.
    write_module(first.path(), "device.x", "pub fn probe(x: u8) -> u8 { x }")?;
    for (paths, expected) in [
        ([first.path(), second.path()], 4),
        ([second.path(), first.path()], 10),
    ] {
        let package = compile_design(source, &search_paths(&paths))?;
        assert_eq!(run(package, 3)?, IrValue::make_ubits(8, expected)?);
    }
    write_module(
        second.path(),
        "device/probe.x",
        "pub fn tag(x: u8) -> u8 { !x }",
    )?;
    let package = compile_design(source, &search_paths(&[first.path(), second.path()]))?;
    assert_eq!(run(package, 3)?, IrValue::make_ubits(8, 252)?);
    Ok(())
}

#[test]
fn matching_struct_layouts_in_different_libraries_remain_distinct_types()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let library = r#"
pub struct Tag { value: u8 }
pub fn make(x: u8) -> Tag { Tag { value: x } }
pub fn read(x: Tag) -> u8 { x.value + u8:1 }
"#;
    write_module(root.path(), "a/b.bx", library)?;
    write_module(root.path(), "a_b.bx", &library.replace("u8:1", "u8:7"))?;
    let options = search_paths(&[root.path()]);
    let source = "use a::b; use a_b; block circuit(clk, x: u8) -> u8 { b::read(a_b::make(x)) }";
    assert!(
        check_module_with_options(&parse_module(source)?, &options).is_err(),
        "{source}"
    );
    let source = r#"
use a::b;
use a_b;
#[allow(combinational_block)]
block circuit(clk, x: u8) -> (u8, u8) {
    (b::read(b::make(x)), a_b::read(a_b::make(x)))
}
"#;
    assert_eq!(
        run(compile_design(source, &options)?, 47)?,
        IrValue::parse_typed("(bits[8]:48, bits[8]:54)")?
    );
    Ok(())
}

#[test]
fn builtin_calls_follow_their_defining_module_and_use_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "ops.bx",
        r#"
pub fn map() -> u8 { u8:4 }
pub fn update() -> u8 { u8:5 }
pub fn zero!() -> u8 { u8:6 }
"#,
    )?;
    write_module(
        dir.path(),
        "lib.bx",
        r#"
pub fn original(x: u8) -> u8 {
    let xs = map([x], |v| v + u8:1);
    let ys = update([zero!<u8>()], u32:0, xs[u32:0]);
    ys[u32:0]
}
use ops::{map, update, zero!};
pub fn overridden() -> u8 { map() + update() + zero!<u8>() }
"#,
    )?;
    let source = r#"
use lib;
fn map() -> u8 { u8:7 }
fn update() -> u8 { u8:8 }
fn zero!() -> u8 { u8:9 }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> (u8, u8) { (lib::original(x), lib::overridden()) }
"#;
    let package = compile_design(source, &search_paths(&[dir.path()]))?;
    assert_eq!(
        run(package, 10)?,
        IrValue::parse_typed("(bits[8]:11,bits[8]:15)")?
    );
    Ok(())
}

#[test]
fn enums_shadow_zero_and_update_in_declaration_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    for (name, call) in [
        ("zero!", "zero!<u8>()"),
        ("update", "update([u8:0], u32:0, u8:7)[u32:0]"),
    ] {
        let declaration = format!("pub enum {name}:u8 {{ A=0 }}");
        write_module(dir.path(), "ops_bx.bx", &declaration)?;
        write_module(dir.path(), "ops_dslx.x", &declaration)?;
        for binding in [
            declaration,
            format!("use ops_bx::{name};"),
            format!("use ops_dslx::{name};"),
        ] {
            for callable in ["fn run()", "block run(clk)"] {
                let body = format!("{callable}->u8 {{ {call} }}");
                for (source, accepted) in [
                    (format!("{body}\n{binding}"), true),
                    (format!("{binding}\n{body}"), false),
                ] {
                    write_module(dir.path(), "helper.bx", &source)?;
                    for program in [source.as_str(), "use helper;"] {
                        let module = parse_module(program)?;
                        let result = check_module_with_options(&module, &options);
                        if accepted {
                            result?;
                        } else {
                            let error = result.unwrap_err();
                            assert!(
                                error.contains("cannot be called as a function"),
                                "{source}: {error}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn skipped_loops_distinguish_enum_annotations_from_struct_constructors()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    write_module(
        dir.path(),
        "wire.x",
        "enum Hidden:u8 { Idle=0 } pub type State = Hidden;",
    )?;
    for (declarations, name, permits_unused_read) in [
        ("enum State:u8 { Idle=0 }", "State", false),
        ("use wire;", "wire::State", false),
        ("use wire::State;", "State", true),
    ] {
        let loop_body = |result: &str| {
            format!(
                "{declarations}\nfn f()->u8 {{ unroll_for! (_i, v) in u32:0..u32:0 {{ {result} }}(u8:0) }}"
            )
        };
        let valid = loop_body(&format!("let state:{name}={name}::Idle; state as u8"));
        check_module_with_options(&parse_module(&valid)?, &options)?;
        let invalid = loop_body(&format!("{name} {{ value:v }}"));
        assert!(
            check_module_with_options(&parse_module(&invalid)?, &options).is_err(),
            "{invalid}"
        );
        let unused_call = loop_body(&format!("{name}()"));
        check_module_with_options(&parse_module(&unused_call)?, &options)?;
        // Native DSLX resolves a direct use binding in an unelaborated body.
        // A local type name and a module-qualified type still require a member.
        let bare_type = loop_body(name);
        assert_eq!(
            check_module_with_options(&parse_module(&bare_type)?, &options).is_ok(),
            permits_unused_read,
            "{bare_type}"
        );
    }
    Ok(())
}

#[test]
fn named_maps_in_library_blocks_resolve_later_callback_imports() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "math.bx",
        "pub fn bump(x: u8) -> u8 { x + u8:1 }",
    )?;
    write_module(
        dir.path(),
        "stage.bx",
        r#"
#[allow(combinational_block)]
pub block apply(clk, x: u8) -> u8 {
    let ys = map({ [x] }, math::bump);
    let zs = map((u8:2..u8:4), math::bump);
    ys[u32:0] + zs[u32:0]
}
use math;
"#,
    )?;
    let source = r#"
use stage;
fn map(x: u8, y: u8) -> u8 { x ^ y }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 { let y = stage::apply(x); y }
"#;
    let package = compile_design(source, &search_paths(&[dir.path()]))?;
    assert_eq!(run(package, 10)?, IrValue::make_ubits(8, 14)?);

    let root_block = r#"
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 { map([x], bump)[u32:0] }
use math::bump;
"#;
    let package = compile_design(root_block, &search_paths(&[dir.path()]))?;
    assert_eq!(run(package, 10)?, IrValue::make_ubits(8, 11)?);
    Ok(())
}

#[test]
fn signatures_bind_each_parameter_before_qualifying_its_type() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    for module in ["helper", "clk", "rst"] {
        write_module(
            root.path(),
            &format!("{module}.bx"),
            "pub struct Tag { value: u8 }",
        )?;
    }
    let options = search_paths(&[root.path()]);
    for (module, declaration) in [
        (
            "helper",
            "fn read(helper: u8, x: helper::Tag) -> u8 { x.value }",
        ),
        (
            "helper",
            "fn read(helper: helper::Tag) -> u8 { helper.value }",
        ),
        ("helper", "extern fn read(helper: u8) -> helper::Tag;"),
        (
            "helper",
            "block read(clk, helper: u8, x: helper::Tag) -> u8 { x.value }",
        ),
        (
            "helper",
            "fn read<helper: fn(u8) -> u8>(x: helper::Tag) -> u8 { x.value }",
        ),
        ("clk", "block read(clk, x: clk::Tag) -> u8 { x.value }"),
        ("rst", "extern block read(clk, rst, x: rst::Tag) -> u8;"),
    ] {
        let source = format!("use {module}; {declaration}");
        let error = check_module_with_options(&parse_module(&source)?, &options).unwrap_err();
        assert!(
            error.contains(&format!("'{module}' names a local value here")),
            "{source}\n{error}"
        );
    }
    let source = "use helper::Tag; fn read(Tag: Tag) -> u8 { Tag.value }";
    let error = check_module_with_options(&parse_module(source)?, &options).unwrap_err();
    assert!(error.contains("'Tag' names a local value here"), "{error}");
    write_module(
        root.path(),
        "hidden.bx",
        "struct Tag { value: u8 } pub fn read(Tag: Tag) -> u8 { Tag.value }",
    )?;
    let error = check_module_with_options(&parse_module("use hidden;")?, &options).unwrap_err();
    assert!(error.contains("'Tag' names a local value here"), "{error}");

    // A field named helper leaves the module visible to the next field, and a
    // type parameter remains a type inside a generic block signature.
    let source = r#"
use helper;
struct Payload { helper: u8, tag: helper::Tag }
block identity<T>(clk, data: T) -> T { data }
block circuit(clk, x: u8) -> u8 {
    let p = Payload { helper: x, tag: helper::Tag { value: x } };
    let result = identity(p);
    result.tag.value
}
"#;
    assert_eq!(
        run(compile_design(source, &options)?, 47)?,
        IrValue::make_ubits(8, 47)?
    );
    Ok(())
}

#[test]
fn qualified_extern_calls_and_underscore_hardware_ports_keep_their_source_bindings()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "device.bx",
        r#"
extern fn chip::invert(x: u8) -> u8;
fn invert(chip: u8) -> u8 { chip::invert(chip) }
#[allow(combinational_block)]
pub block drive(clk, _: u8) -> u8 { invert(_) }
"#,
    )?;
    let source = r#"
use device;
extern fn chip::invert(x: u8) -> u8;
pub block circuit(clk, chip: u8) -> (u8, u8) {
    let y = device::drive(chip);
    (y, chip::invert(chip))
}
"#;
    let linked = Parser::new(
        r#"package test
fn __chip__invert(x: bits[8] id=1) -> bits[8] { ret v: bits[8] = not(x, id=2) }
"#,
    )
    .parse_and_verify_package()?;
    let package = compile_with_options(
        source,
        linked,
        ResetBehavior::default(),
        &search_paths(&[dir.path()]),
    )?;
    assert_eq!(
        run(package, 5)?,
        IrValue::parse_typed("(bits[8]:250, bits[8]:250)")?
    );
    Ok(())
}

#[test]
fn calls_on_local_values_are_checked_when_a_loop_iteration_elaborates() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    let library = r#"
#[allow(combinational_block)]
pub block stage(clk, x: u8) -> u8 {
    unroll_for! (i, a) in u8:0..u8:0 { let y = x(a); y }(x)
}
"#;
    // The root has a block named x. A skipped call only needs its local x to
    // be in scope, and an elaborated call cannot instantiate the root block.
    let root = r#"
use device;
#[allow(combinational_block)]
block x(clk, data: u8) -> u8 { !data }
pub block circuit(clk, data: u8) -> u8 { let y = device::stage(data); y }
"#;
    write_module(dir.path(), "device.bx", library)?;
    assert_eq!(
        run(compile_design(root, &options)?, 5)?,
        IrValue::make_ubits(8, 5)?
    );
    write_module(
        dir.path(),
        "device.bx",
        &library.replace("..u8:0", "..u8:1"),
    )?;
    let error = compile_design(root, &options).unwrap_err();
    assert!(error.contains("'x' names a local value here"), "{error}");

    let loop_binder = library.replace("x(a)", "i(a)");
    write_module(dir.path(), "device.bx", &loop_binder)?;
    assert_eq!(
        run(compile_design(root, &options)?, 5)?,
        IrValue::make_ubits(8, 5)?
    );
    write_module(
        dir.path(),
        "device.bx",
        &loop_binder.replace("..u8:0", "..u8:1"),
    )?;
    let error = compile_design(root, &options).unwrap_err();
    assert!(error.contains("'i' names a loop-local value"), "{error}");

    // A name absent from the library is still invalid in a skipped loop,
    // even if the root declares that name.
    write_module(
        dir.path(),
        "device.bx",
        &library
            .replace("clk, x:", "clk, v:")
            .replace("}(x)", "}(v)"),
    )?;
    let error = compile_design(root, &options).unwrap_err();
    assert!(
        error.contains("unknown name 'x' in BX module 'device'"),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_simple_block_let_rebinding_a_tuple_pattern_still_shadows_an_extern()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(
        dir.path(),
        "device.bx",
        r#"
extern fn chip(x: u8) -> u8;
pub block stage(clk, x: u8) -> u8 {
    let (chip, _) = (x, x);
    let chip = x;
    chip(chip)
}
"#,
    )?;
    let module = parse_module("use device;")?;
    let error = check_module_with_options(&module, &search_paths(&[dir.path()])).unwrap_err();
    assert!(error.contains("'chip' names a local value here"), "{error}");
    Ok(())
}

#[test]
fn rebinding_a_block_formal_keeps_module_callees_hidden() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = search_paths(&[dir.path()]);
    for (formal, ty, call, type_formal) in [
        ("child: fn(u8) -> u8", "u8", "stage<id>(x)", false),
        ("child", "child", "stage(x)", true),
    ] {
        let caller = format!(
            "fn id(x: u8) -> u8 {{ x }} block circuit(clk, x: u8) -> u8 {{ let y = {call}; y }}"
        );
        for (body, context, callback_error) in [
            (
                "let y = child(x); y",
                "let 'y': ",
                "'child' names a local value here",
            ),
            (
                "child::op(x)",
                "result: ",
                "callable parameter 'child' may only be used as a callee",
            ),
        ] {
            let declarations = format!(
                r#"
extern fn child::op(x: u8) -> u8;
block child(clk, x: u8) -> u8 {{ reg q: u8; q := x; q }}
pub block stage<{formal}>(clk, x: {ty}) -> {ty} {{ let child = x; {body} }}
"#
            );
            write_module(dir.path(), "device.bx", &declarations)?;
            let reason = if type_formal {
                "'child' names a type parameter here"
            } else {
                callback_error
            };
            for (source, name) in [
                (declarations.as_str(), "stage"),
                ("use device::stage;", "device::stage"),
            ] {
                let error = check_source_with_options(&format!("{source}\n{caller}"), &options)
                    .unwrap_err();
                assert_eq!(
                    error,
                    format!("block '{name}' ({name}__0): {context}{reason}")
                );
            }
        }
    }
    Ok(())
}

#[test]
fn a_scoped_callee_keeps_its_binding_when_prepared_for_a_loop() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(dir.path(), "unused.bx", "")?;
    let options = search_paths(&[dir.path()]);
    let source = r#"
fn ident(x: u8) -> u8 { x }
#[allow(combinational_block)]
block circuit(clk, x: u8) -> u8 {
    ident({ let w = x; unroll_for! (i, a) in u8:0..u8:0 { let y = w(a); y }(x) })
}
"#;
    for uses in ["", "use unused;"] {
        let source = format!("{uses}\n{source}");
        assert_eq!(
            run(compile_design(&source, &options)?, 9)?,
            IrValue::make_ubits(8, 9)?
        );
        let error = compile_design(&source.replace("..u8:0", "..u8:1"), &options).unwrap_err();
        assert!(error.contains("'w' names a local value"), "{error}");
    }
    Ok(())
}

#[test]
fn import_qualification_bounds_guard_nesting_independently_of_expression_height()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    write_module(dir.path(), "unused.bx", "")?;
    let options = search_paths(&[dir.path()]);
    let sum = ["x"; 63].join("+");
    for guards in [3, 55] {
        let source = format!(
            "block circuit(clk, x: u8) -> u8 {{ reg r: u8; {} r := {sum}; {} r }}",
            "if true { ".repeat(guards),
            "} ".repeat(guards),
        );
        let standalone = compile_design(&source, &options)?;
        let imported = compile_design(&format!("use unused; {source}"), &options)?;
        assert_eq!(imported.to_string(), standalone.to_string());
    }

    let mut module = parse_module("use unused; block circuit(clk) -> () { () }")?;
    let statements = &mut module.blocks[0].statements;
    for _ in 0..128 {
        let body = std::mem::take(statements);
        statements.push(Stmt::If {
            condition: Expr::Bool(true),
            body,
        });
    }
    let error = check_module_with_options(&module, &options).unwrap_err();
    assert_eq!(error, "BX import guard exceeds 64 levels");
    Ok(())
}
