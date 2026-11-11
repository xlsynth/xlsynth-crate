// SPDX-License-Identifier: Apache-2.0

use super::support::search_paths;
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{check_module_with_options, parse_module};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};

const LIBRARY: &str = r#"
struct Hidden { x:u8 }
enum State:u2 { A=0, B=1 }
pub type Pair = (Hidden, u8);
pub type Modes = State[2];
pub fn make(x:u8)->Pair { (Hidden { x }, u8:5) }
pub fn modes(b:bool)->Modes { [if b { State::B } else { State::A }, State::A] }
pub fn both(x:u8,b:bool)->(Pair,Modes) { (make(x), modes(b)) }
pub fn first(p:Pair)->u8 { p.0.x + p.1 }
pub fn active(m:Modes)->bool { m[0] == State::B }
"#;

#[test]
fn public_native_aggregate_spelling_survives_bx_aliases_and_inferred_calls()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(directory.path().join("left.x"), LIBRARY)?;
    // The producer writes the structural types directly. Its consumer
    // is the first source declaration to name a usable aggregate alias.
    fs::write(
        directory.path().join("right.x"),
        LIBRARY
            .replace("->Pair", "->(Hidden,u8)")
            .replace("->Modes", "->State[2]")
            .replace("->(Pair,Modes)", "->((Hidden,u8),State[2])"),
    )?;
    let options = search_paths(&[directory.path()]);
    let declarations = r#"
type Named = left::Pair;
type States = left::Modes;
fn typed(p:Named,m:States)->(u8,bool) { (left::first(p), left::active(m)) }
"#;
    let body = r#"
let (p,m) = right::both(x, b);
(typed(left::make(x), left::modes(b)), right::first(p), right::active(m))
"#;
    let params = "x:u8,b:bool";
    let result = "((u8,bool),u8,bool)";
    let bx = format!(
        "use left; use right; {declarations}{}",
        bx_source(params, result, body)
    );
    let native = format!(
        "import left; import right; {declarations}{}",
        dslx_source(params, result, body)
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &options,
        compile_dslx_module(&native, &[directory.path()]),
    )?;
    for (x, b) in [(0, false), (8, true)] {
        let outputs = [IrValue::make_ubits(8, x + 5)?, IrValue::bool(b)];
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, x)?, IrValue::bool(b)])?,
            IrValue::make_tuple(&[
                IrValue::make_tuple(&outputs),
                outputs[0].clone(),
                outputs[1].clone()
            ])
        );
    }
    for (ty, body) in [
        ("u8", "right::first(left::make(x))"),
        ("bool", "right::active(left::modes(b))"),
    ] {
        let native = format!(
            "import left; import right; {}",
            dslx_source(params, ty, body)
        );
        assert!(
            compile_dslx_module(&native, &[directory.path()]).is_err(),
            "{native}"
        );
        let bx = format!("use left; use right; {}", bx_source(params, ty, body));
        assert!(
            check_module_with_options(&parse_module(&bx)?, &options).is_err(),
            "{bx}"
        );
    }
    Ok(())
}

#[test]
fn native_value_parametrics_reconstruct_private_members_of_public_aggregates()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("lib.x"),
        r#"
struct Hidden { x:u8 }
struct Inner { x:u8 }
pub struct Wrapper { inner:Inner }
enum Mode:u2 { Off=0, On=1, Full=3 }
pub type BXConstexprType0 = u3;
pub type Pair = (Hidden,u8);
pub type Modes = Mode[2];
pub type Wrapped = (Wrapper,u8);
type Binding = (Hidden,u8);
pub const PAIR = (Hidden { x:u8:19 },u8:7);
pub const MODES = [Mode::On, Mode::Full];
pub const WRAPPED = (Wrapper { inner:Inner { x:u8:19 } }, u8:11);
pub fn first<P:Pair>(x:u8)->u8 { P.0.x + P.1 + x }
pub fn other<P:Binding>(x:u8)->u8 { P.0.x + P.1 + x }
pub fn modes<M:Modes>(x:u8)->u8 { (M[0] as u8) + (M[1] as u8) + x }
pub fn nested<P:Wrapped>(x:u8)->u8 { P.0.inner.x + P.1 + x }
"#,
    )?;
    let options = search_paths(&[directory.path()]);
    let bx = format!(
        "use lib; type UserAlias = lib::BXConstexprType0; fn record<T>(_:T)->T {{ T {{ x:u8:19 }} }} fn mode<T>(_:T)->T {{ u2:1 as T }} {}",
        bx_source(
            "x:u8",
            "(u8,u8,u8,u8)",
            r#"
let empty = zero!<lib::Pair>();
let full = all_ones!<lib::Modes>();
let wrapper = zero!<lib::Wrapped>();
(lib::first<{empty}>(x), lib::other<{(record(empty.0),u8:7)}>(x),
 lib::modes<{[mode(full[0]),full[1]]}>(x),
 lib::nested<{(lib::Wrapper { inner:record(wrapper.0.inner) },u8:11)}>(x))
"#
        )
    );
    let native = format!(
        "import lib; {}",
        dslx_source(
            "x:u8",
            "(u8,u8,u8,u8)",
            r#"
(lib::first<{zero!<lib::Pair>()}>(x), lib::other<{lib::PAIR}>(x), lib::modes<{lib::MODES}>(x), lib::nested<{lib::WRAPPED}>(x))
"#
        )
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &options,
        compile_dslx_module(&native, &[directory.path()]),
    )?;
    for x in [0, 42, 255] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, x)?])?,
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, x)?,
                IrValue::make_ubits(8, (x + 26) & 255)?,
                IrValue::make_ubits(8, (x + 4) & 255)?,
                IrValue::make_ubits(8, (x + 30) & 255)?,
            ])
        );
    }
    for private in [
        "Hidden",
        "Inner",
        "Mode",
        "BXConstexprType1",
        "BXConstexprType2",
        "BXConstexprType3",
    ] {
        let source = format!("use lib; type Invalid = lib::{private};");
        assert!(
            check_module_with_options(&parse_module(&source)?, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn inline_native_operands_preserve_private_types_and_literal_constraints()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let library = r#"
struct Hidden { n:u8 }
enum Mode:u2 { Idle=0, Busy=3 }
pub type Pair = (Hidden, u8);
pub type Modes = Mode[2];
pub type Nested = ((Hidden, bool), u8);
pub type Empty = (Hidden[0], u8);
pub fn pair(x:u8)->Pair { (Hidden { n:x }, u8:9) }
pub fn modes(b:bool)->Modes { [Mode::Idle, if b { Mode::Busy } else { Mode::Idle }] }
pub fn read<N:u32>(p:Pair, m:Modes)->u8 { p.0.n + p.1 + (m[0] as u8) + (N as u8) }
pub fn nested(x:u8,b:bool)->Nested { ((Hidden { n:x }, b), u8:0) }
pub fn read_nested<N:u32>(p:Nested)->u8 { p.0.0.n + (p.0.1 as u8) + p.1 + (N as u8) }
pub fn read_empty<N:u32>(p:Empty)->u8 { p.1 + (N as u8) }
pub fn read_hidden(p:Hidden,y:u8)->u8 { p.n + y }
pub fn read_hidden_parametric<N:u32>(p:Hidden,y:u8)->u8 { p.n + y + (N as u8) }
"#;
    for name in ["wire.x", "peer.x"] {
        fs::write(directory.path().join(name), library)?;
    }
    let options = search_paths(&[directory.path()]);
    let params = "x:u8,b:bool";
    let values = "let p=wire::pair(x); let m=wire::modes(b);";
    let body = format!(
        r#"{values}
        (wire::read<1>((p.0, 200), [m[1], m[0]]),
         wire::read<3>((p.0, u8:2), [m[0], m[1]]),
         wire::read_nested<4>((wire::nested(x, b).0, 13)))"#
    );
    let result = "(u8,u8,u8)";
    let bx = format!("use wire; {}", bx_source(params, result, &body));
    let native = format!("import wire; {}", dslx_source(params, result, &body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &options,
        compile_dslx_module(&native, &[directory.path()]),
    )?;
    for (x, b) in [(0, false), (8, true)] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, x)?, IrValue::bool(b)])?,
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, x + 201 + if b { 3 } else { 0 })?,
                IrValue::make_ubits(8, x + 5)?,
                IrValue::make_ubits(8, x + 17 + u64::from(b))?,
            ])
        );
    }
    for (pair, modes) in [
        ("(p.0, 256)", "m"),
        ("(p.0, s8:2)", "m"),
        ("(peer::pair(x).0, 2)", "m"),
        ("p", "[peer::modes(b)[0], m[1]]"),
    ] {
        let body = format!("{values} wire::read<1>({pair}, {modes})");
        let native = format!(
            "import wire; import peer; {}",
            dslx_source(params, "u8", &body)
        );
        assert!(
            compile_dslx_module(&native, &[directory.path()]).is_err(),
            "{native}"
        );
        let bx = format!("use wire; use peer; {}", bx_source(params, "u8", &body));
        assert!(
            check_module_with_options(&parse_module(&bx)?, &options).is_err(),
            "{bx}"
        );
    }
    // An empty subarray has a usable projection; its element has none.
    let empty = "fn sample(p:wire::Empty)->u8 { wire::read_empty<3>((p.0, 8)) }";
    compile_dslx_module(&format!("import wire; {empty}"), &[directory.path()])?;
    check_module_with_options(&parse_module(&format!("use wire; {empty}"))?, &options)?;

    // Passing the private leaf itself still needs a public spelling for its
    // signature in the IR conversion probe.
    for (call, literal) in [("read_hidden", "u8:2"), ("read_hidden_parametric<3>", "2")] {
        let source =
            format!("use wire; fn sample(x:u8)->u8 {{ wire::{call}(wire::pair(x).0,{literal}) }}");
        let error = check_module_with_options(&parse_module(&source)?, &options).unwrap_err();
        assert!(error.contains("without a public type name"), "{error}");
    }
    Ok(())
}

#[test]
fn inline_tuple_discovers_its_alias_from_the_specialized_native_formal()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("wire.x"),
        r#"
struct Hidden { x:u8 }
pub struct Wrap { h:Hidden }
pub type Pair = (Hidden,u8);
pub fn consume<N:u32>(p:Pair)->u8 { p.0.x + p.1 + (N as u8) }
pub fn nested<N:u32>(p:(Pair,bool))->u8 { p.0.0.x + p.0.1 + (p.1 as u8) + (N as u8) }
"#,
    )?;
    let options = search_paths(&[directory.path()]);
    let params = "w:wire::Wrap";
    for body in [
        "wire::consume<2>((w.h, 3))",
        "wire::nested<2>(((w.h, 2), true))",
    ] {
        let bx = format!("use wire; {}", bx_source(params, "u8", body));
        let native = format!("import wire; {}", dslx_source(params, "u8", body));
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &options,
            compile_dslx_module(&native, &[directory.path()]),
        )?;
        for x in [0, 41, 255] {
            let hidden = IrValue::make_tuple(&[IrValue::make_ubits(8, x)?]);
            let wrapper = IrValue::make_tuple(&[hidden]);
            assert_eq!(
                evaluator.compare(&[wrapper])?,
                IrValue::make_ubits(8, (x + 5) & 255)?
            );
        }
    }
    Ok(())
}

#[test]
fn inline_array_projects_a_private_member_beyond_the_outline_depth() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    // Each name and array occupies an outline level. The resolved native
    // layout contains only the arrays and remains within the type-depth limit.
    let mut library = String::from("struct Hidden { x:u8 } type Level0 = Hidden;\n");
    for i in 1..35 {
        library.push_str(&format!("type Level{i} = Level{}[1];\n", i - 1));
    }
    library
        .push_str("pub type Packet = Level34[1]; pub fn consume<N:u32>(p:Packet)->u8 { N as u8 }");
    fs::write(directory.path().join("wire.x"), library)?;
    let options = search_paths(&[directory.path()]);
    let sample = "fn sample(p:wire::Packet)->u8 { wire::consume<2>([p[0]]) }";
    compile_dslx_module(&format!("import wire; {sample}"), &[directory.path()])?;
    check_module_with_options(&parse_module(&format!("use wire; {sample}"))?, &options)?;
    Ok(())
}
