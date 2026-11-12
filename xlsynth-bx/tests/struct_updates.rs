// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs};
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{
    ResetBehavior, check_module, compile, compile_with_options, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn inferred_struct_update_shares_a_computed_base() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Payload<N: u32> { words: uN[N][2] }
struct Packet<N: u32> { payload: Payload<N>, tag: u16, valid: bool }

fn choose(a: Packet<8>, b: Packet<8>, choose_a: bool) -> Packet<8> {
    if choose_a { a } else { b }
}

fn patched(a: Packet<8>, b: Packet<8>, choose_a: bool, valid: bool) -> Packet<8> {
    let packet = Packet { valid, ..choose(a, b, choose_a) };
    Packet<8> { ..packet }
}
"#;
    let package = compile(source, empty_package("packets"), ResetBehavior::default())?;
    let patched = package
        .get_fn("patched")
        .ok_or("missing patched function")?;
    let invokes = patched
        .nodes
        .iter()
        .filter(|node| matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "choose"))
        .count();
    assert_eq!(invokes, 1);
    let left = IrValue::parse_typed("(([bits[8]:1, bits[8]:2]), bits[16]:10, bits[1]:0)")?;
    let right = IrValue::parse_typed("(([bits[8]:3, bits[8]:4]), bits[16]:20, bits[1]:1)")?;
    for (choose_a, valid, expected) in [
        (
            true,
            true,
            "(([bits[8]:1, bits[8]:2]), bits[16]:10, bits[1]:1)",
        ),
        (
            false,
            false,
            "(([bits[8]:3, bits[8]:4]), bits[16]:20, bits[1]:0)",
        ),
    ] {
        let args = [
            left.clone(),
            right.clone(),
            IrValue::bool(choose_a),
            IrValue::bool(valid),
        ];
        assert_eq!(
            eval_fn_in_package_value(&package, patched, &args),
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn splat_base_inherits_context_and_explicit_value_bindings() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Box<N: u32> { a: uN[N], b: bool }
struct Pair<N: u32, L: u32> { a: uN[N], b: uN[L] }
struct Holder<T> { value: T }
fn context() -> Box<8> { Box { b: false, ..Box { a: 0, b: true } } }
fn explicit() -> u8 {
    let x = Box<8> { ..Box { a: 0, b: false } };
    x.a
}
fn partial(b: u11, take: bool) -> (u8, u11) {
    let x = Pair<8> { ..if take { Pair { a: 0, b } } else { Pair<8, 11> { a: 1, b } } };
    (x.a, x.b)
}
fn nested(b: u11) -> (u8, u11) {
    let x = Pair<8> { ..Pair { ..Pair { a: 0, b } } };
    (x.a, x.b)
}
fn partial_siblings(b: u11, take: bool) -> u11 {
    let x = Pair<8> { ..if take { Pair { a: 0, b: 0 } } else { Pair { a: 0, b } } };
    x.b
}
fn type_applied() -> u8 {
    let x = Holder::<u8> { ..Holder { value: 0 } };
    x.value
}
"#;
    let package = compile(source, empty_package("splat"), ResetBehavior::default())?;
    for (name, args, expected) in [
        ("context", vec![], "(bits[8]:0, bits[1]:0)"),
        ("explicit", vec![], "bits[8]:0"),
        (
            "partial",
            vec![IrValue::parse_typed("bits[11]:17")?, IrValue::bool(false)],
            "(bits[8]:1, bits[11]:17)",
        ),
        (
            "partial",
            vec![IrValue::parse_typed("bits[11]:19")?, IrValue::bool(true)],
            "(bits[8]:0, bits[11]:19)",
        ),
        (
            "nested",
            vec![IrValue::parse_typed("bits[11]:21")?],
            "(bits[8]:0, bits[11]:21)",
        ),
        (
            "partial_siblings",
            vec![IrValue::parse_typed("bits[11]:21")?, IrValue::bool(true)],
            "bits[11]:0",
        ),
        ("type_applied", vec![], "bits[8]:0"),
    ] {
        let function = package.get_fn(name).ok_or(name)?;
        assert_eq!(
            eval_fn_in_package_value(&package, function, &args),
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn splat_and_ordinary_constructors_respect_callback_shadowing() -> Result<(), String> {
    for constructor in ["S { ..base }", "S { value: u8:1 }"] {
        let source = format!(
            r#"
struct S<N: u32> {{ value: uN[N] }}
const base = S<8> {{ value: u8:0 }};
fn apply<S: fn(u8) -> u8>(x: u8) -> u8 {{ ({constructor}).value }}
fn identity(x: u8) -> u8 {{ x }}
fn run(x: u8) -> u8 {{ apply<identity>(x) }}
"#
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains("'S' names a local value here"), "{error}");
    }
    Ok(())
}

#[test]
fn partial_splat_bindings_reach_counted_and_unrolled_seeds() -> Result<(), Box<dyn Error>> {
    for keyword in ["for", "unroll_for!"] {
        let source = format!(
            r#"
struct Pair<N:u32, L:u32> {{ a:uN[N], b:uN[L] }}
fn seed_b(b:u11) -> u11 {{ b }}
fn run(b:u11) -> (u8, u11) {{
    let x = Pair<8> {{ ..{keyword} (i,p) in u32:0..u32:2 {{
        Pair {{ a:p.a + u8:1, ..p }}
    }}(Pair {{ a:0, b:seed_b(b) }}) }};
    (x.a, x.b)
}}
"#
        );
        let package = compile(&source, empty_package("splat"), ResetBehavior::default())?;
        let run = package.get_fn("run").ok_or("missing run")?;
        let calls = run.nodes.iter().filter(|node| matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "seed_b")).count();
        assert_eq!(calls, 1, "{keyword}");
        assert_eq!(
            eval_fn_in_package_value(&package, run, &[IrValue::parse_typed("bits[11]:42")?]),
            IrValue::parse_typed("(bits[8]:2, bits[11]:42)")?
        );

        let conflicting = format!(
            r#"
struct Pair<N:u32, L:u32> {{ a:uN[N], b:uN[L] }}
fn run(b:u11) -> u8 {{
    let x = Pair<8> {{ ..{keyword} (i,p): (u32, Pair<7,11>) in u32:0..u32:1 {{ p }}(Pair {{ a:0,b }}) }};
    x.a
}}
"#
        );
        let error = check_module(&parse_module(&conflicting)?).unwrap_err();
        assert!(error.contains("struct splat base type"), "{error}");
    }
    Ok(())
}

#[test]
fn struct_update_preserves_unmodified_register_fields() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Snapshot { held: u8, updated: u8, valid: bool }
const START = Snapshot { held: u8:90, updated: u8:0, valid: false };

pub block capture(clk, take: bool, next_value: u8) -> Snapshot {
    reg saved: Snapshot = Snapshot { updated: u8:1, ..START };
    if take {
        saved := Snapshot { valid: true, updated: next_value, ..saved };
    }
    saved
}
"#;
    let package = compile(source, empty_package("capture"), ResetBehavior::default())?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let rst = ports.input("rst").unwrap();
    let take = ports.input("take").unwrap();
    let next_value = ports.input("next_value").unwrap();
    for (reset, enable, value, expected) in [
        (true, false, 0, [90, 1, 0]),
        (false, true, 17, [90, 17, 1]),
        (false, false, 29, [90, 17, 1]),
        (false, true, 31, [90, 31, 1]),
        (true, true, 49, [90, 1, 0]),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&rst, reset)?;
                drive.set_bool(&take, enable)?;
                drive.set_u64(&next_value, value)
            })
            .unwrap();
        let actual = ["held", "updated", "valid"].map(|port| edge.outputs[port].to_u64_if_known());
        assert_eq!(actual, expected.map(Some));
    }
    Ok(())
}

#[test]
fn imported_struct_update_can_capture_a_base_in_map() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("records.bx"),
        r#"
pub struct Record<T> { valid: bool, payload: T }

pub fn with_flags<T>(x: Record<T>, flags: bool[2]) -> Record<T>[2] {
    map(flags, |valid| Record::<T> { valid, ..x })
}
"#,
    )?;
    let source = r#"
use records;

#[allow(combinational_block)]
pub block sample(clk, x: records::Record<u8>, flags: bool[2]) -> records::Record<u8>[2] {
    records::with_flags(x, flags)
}
"#;
    let options = search_paths(&[dir.path()]);
    let package = compile_with_options(
        source,
        empty_package("sample"),
        ResetBehavior::default(),
        &options,
    )?;
    let top = package.get_top_block().ok_or("missing sample block")?;
    let function = combinational_block_to_fn(top)?;
    let args = [
        IrValue::parse_typed("(bits[1]:0, bits[8]:42)")?,
        IrValue::parse_typed("[bits[1]:1, bits[1]:0]")?,
    ];
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &args),
        IrValue::parse_typed("[(bits[1]:1, bits[8]:42), (bits[1]:0, bits[8]:42)]")?
    );
    Ok(())
}

#[test]
fn struct_update_checks_timing_per_field() -> Result<(), String> {
    let source = r#"
struct Ports { valid: bool, data: u8, tag: u16 }

#[allow(combinational_block)]
pub block sample(clk, choose: bool, a: Ports, b: Ports, valid_in: bool @1)
    -> Ports @(valid: 1, data: 0, tag: 0) {
    Ports { valid: valid_in, ..if choose { a } else { b } }
}
"#;
    compile(source, empty_package("sample"), ResetBehavior::default())?;
    Ok(())
}

#[test]
fn imported_struct_splat_accepts_template_and_alias_context() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("records.x"),
        "pub struct Box<N: u32> { a: uN[N] }\npub type Byte = Box<8>;\n",
    )?;
    let source = r#"
use records;
use records::Box;
fn contextual() -> records::Byte {
    records::Box { ..records::Box { a: 0 } }
}
fn alias() -> u8 { (records::Byte { ..records::Box { a: 0 } }).a }
fn qualified_base() -> u8 { (Box<8> { ..records::Box { a: 1 } }).a }
fn direct_base() -> u8 { (records::Box<8> { ..Box { a: 2 } }).a }
fn from_sibling(b: records::Byte, q: bool) -> u8 {
    let x = if q { Box { ..records::Box { a: 0 } } } else { b };
    x.a
}
"#;
    let options = search_paths(&[dir.path()]);
    let package = compile_with_options(
        source,
        empty_package("splat"),
        ResetBehavior::default(),
        &options,
    )?;
    for (name, expected) in [
        ("contextual", "(bits[8]:0)"),
        ("alias", "bits[8]:0"),
        ("qualified_base", "bits[8]:1"),
        ("direct_base", "bits[8]:2"),
    ] {
        let function = package.get_fn(name).ok_or(name)?;
        assert_eq!(
            eval_fn_in_package_value(&package, function, &[]),
            IrValue::parse_typed(expected)?
        );
    }
    let sibling = package
        .get_fn("from_sibling")
        .ok_or("missing from_sibling")?;
    for (q, expected) in [(true, "bits[8]:0"), (false, "bits[8]:42")] {
        assert_eq!(
            eval_fn_in_package_value(
                &package,
                sibling,
                &[IrValue::parse_typed("(bits[8]:42)")?, IrValue::bool(q)]
            ),
            IrValue::parse_typed(expected)?
        );
    }
    fs::write(
        dir.path().join("other.x"),
        "pub struct Box<N: u32> { a: uN[N] }\n",
    )?;
    let distinct = r#"
use records::Box;
use other;
fn different() -> u8 { (Box<8> { ..other::Box<8> { a: 0 } }).a }
"#;
    let error = compile_with_options(
        distinct,
        empty_package("splat"),
        ResetBehavior::default(),
        &options,
    )
    .unwrap_err();
    assert!(error.contains("struct splat base type"), "{error}");
    Ok(())
}

#[test]
fn struct_update_rejects_invalid_fields_and_base_types() -> Result<(), String> {
    let declarations = r#"
struct S { data: u8, tag: u16 }
struct T { data: u8, tag: u16 }
struct Tagged<N: u32> { data: u8, tag: u16 }
struct Box<N: u32> { data: uN[N] }
struct Other<N: u32> { data: uN[N] }
"#;
    for function in [
        "fn patch(x: S) -> S { S { data: u16:0, ..x } }",
        "fn patch(x: S) -> S { S { other: u8:0, ..x } }",
        "fn patch(x: S, data: u8) -> S { S { data, data, ..x } }",
        "fn patch(x: T) -> S { S { data: u8:0, ..x } }",
        "fn patch(x: (u8, u16)) -> S { S { ..x } }",
        "fn patch(x: Tagged<1>) -> Tagged<2> { Tagged<2> { data: u8:0, ..x } }",
        "fn patch() -> Box<8> { Box { ..Other { data: 0 } } }",
        "fn patch() -> Box<8> { Box { ..Box<7> { data: 0 } } }",
        "fn patch() -> Box<8> { Box { ..Box { data: u7:0 } } }",
    ] {
        let source = format!("{declarations}\n{function}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{function}");
    }
    Ok(())
}

#[test]
fn struct_updates_share_a_siblings_nominal_type() -> Result<(), Box<dyn Error>> {
    for (parameter, field_type, argument) in [("N: u32", "uN[N]", "8"), ("T", "T", "u8")] {
        let source = format!(
            r#"
struct Box<{parameter}> {{ v: {field_type}, tag: bool }}
fn choose(b: Box<{argument}>, q: bool) -> (u8, bool, u8, bool) {{
    let left = if q {{ Box {{ tag: false, ..Box {{ v: 0, tag: true }} }} }} else {{ b }};
    let right = if q {{ b }} else {{ Box {{ ..Box {{ v: 1, tag: false }} }} }};
    (left.v, left.tag, right.v, right.tag)
}}
fn arrays(b: Box<{argument}>) -> (u8, u8, bool, u8, u8) {{
    let left = [Box {{ ..Box {{ v: 2, tag: false }} }}, Box {{ ..b }}];
    let right = [b, Box {{ tag: true, ..Box {{ v: 3, tag: false }} }}];
    let nested = [[Box {{ ..Box {{ v: 4, tag: false }} }}], [b]];
    let both = [Box {{ ..Box {{ v: 5, tag: false }} }}, Box {{ ..Box {{ v: b.v, tag: true }} }}];
    (left[u32:0].v, right[u32:1].v, right[u32:1].tag, nested[u32:0][u32:0].v, both[u32:0].v)
}}
"#
        );
        let package = compile(&source, empty_package("siblings"), ResetBehavior::default())?;
        let b = IrValue::parse_typed("(bits[8]:42, bits[1]:1)")?;
        let choose = package.get_fn("choose").ok_or("missing choose")?;
        for (q, expected) in [
            (true, "(bits[8]:0, bits[1]:0, bits[8]:42, bits[1]:1)"),
            (false, "(bits[8]:42, bits[1]:1, bits[8]:1, bits[1]:0)"),
        ] {
            assert_eq!(
                eval_fn_in_package_value(&package, choose, &[b.clone(), IrValue::bool(q)]),
                IrValue::parse_typed(expected)?
            );
        }
        let arrays = package.get_fn("arrays").ok_or("missing arrays")?;
        assert_eq!(
            eval_fn_in_package_value(&package, arrays, &[b]),
            IrValue::parse_typed("(bits[8]:2, bits[8]:3, bits[1]:1, bits[8]:4, bits[8]:5)")?
        );
        let invalid = format!(
            r#"
struct Box<{parameter}> {{ v: {field_type} }}
fn choose(b: Box<{argument}>, q: bool) -> u8 {{
    let x = if q {{ Box {{ ..Box {{ v: u7:0 }} }} }} else {{ b }};
    x.v
}}
"#
        );
        assert!(check_module(&parse_module(&invalid)?).is_err(), "{invalid}");
    }
    Ok(())
}

#[test]
fn partial_struct_splat_unifies_nested_branch_results() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Pair<N: u32, L: u32> { a: uN[N], b: uN[L] }
fn nested(b: u11, q: bool) -> (u8, u11, u11) {
    let left = Pair<8> { ..if q {
        if q { Pair { a: 1, b: 0 } } else { Pair { a: 2, b: 0 } }
    } else { Pair { a: 3, b } } };
    let right = Pair<8> { ..if q { Pair { a: 0, b: 4 } } else {
        match q { true => Pair { a: 0, b: 5 }, _ => { let y = b; Pair { a: 0, b: y } }, }
    } };
    (left.a, left.b, right.b)
}
"#;
    let package = compile(source, empty_package("siblings"), ResetBehavior::default())?;
    let nested = package.get_fn("nested").ok_or("missing nested")?;
    for (q, expected) in [
        (true, "(bits[8]:1, bits[11]:0, bits[11]:4)"),
        (false, "(bits[8]:3, bits[11]:42, bits[11]:42)"),
    ] {
        assert_eq!(
            eval_fn_in_package_value(
                &package,
                nested,
                &[IrValue::parse_typed("bits[11]:42")?, IrValue::bool(q)]
            ),
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn sibling_type_reaches_an_unannotated_splat_loop_seed() -> Result<(), Box<dyn Error>> {
    for keyword in ["for", "unroll_for!"] {
        for (parameter, field, argument) in [("N: u32", "uN[N]", "8"), ("T", "T", "u8")] {
            let source = format!(
                r#"
struct Box<{parameter}> {{ v: {field} }}
fn run(b: Box<{argument}>, q: bool) -> u8 {{
    let x = if q {{ Box {{ ..{keyword} (i,p) in u32:0..u32:1 {{ p }}(Box {{ v: 0 }}) }} }} else {{ b }};
    x.v
}}
"#
            );
            let package = compile(&source, empty_package("siblings"), ResetBehavior::default())?;
            let run = package.get_fn("run").ok_or("missing run")?;
            for (q, expected) in [(true, "bits[8]:0"), (false, "bits[8]:42")] {
                assert_eq!(
                    eval_fn_in_package_value(
                        &package,
                        run,
                        &[IrValue::parse_typed("(bits[8]:42)")?, IrValue::bool(q)]
                    ),
                    IrValue::parse_typed(expected)?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn deferred_splat_witness_emits_its_scoped_base_once() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Holder<T> { data: T, flag: bool }
fn echo(flag: bool) -> bool { flag }
fn run(t: bool) -> (u8, bool) {
    let x = if t { Holder { ..{ let q = echo(t); Holder { data: 0, flag: q } } } }
        else { Holder { ..Holder { data: u8:2, flag: t } } };
    (x.data, x.flag)
}
"#;
    let package = compile(source, empty_package("siblings"), ResetBehavior::default())?;
    let run = package.get_fn("run").ok_or("missing run")?;
    let invokes = run.nodes.iter().filter(|node| matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "echo")).count();
    assert_eq!(invokes, 1);
    for (t, expected) in [
        (true, "(bits[8]:0, bits[1]:1)"),
        (false, "(bits[8]:2, bits[1]:0)"),
    ] {
        assert_eq!(
            eval_fn_in_package_value(&package, run, &[IrValue::bool(t)]),
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn imported_partial_splat_receives_its_siblings_remaining_arguments() -> Result<(), Box<dyn Error>>
{
    let source = r#"
use apfloat;
const EXP = u32:7;
fn from_constructors(q: bool) -> u23 {
    let x = if q {
        apfloat::APFloat<8> { ..apfloat::APFloat { sign: 0, bexp: 0, fraction: 1 } }
    } else {
        apfloat::APFloat<8, 23> { ..apfloat::APFloat { sign: 0, bexp: 0, fraction: 2 } }
    };
    x.fraction
}
fn from_argument(b: apfloat::APFloat<8, 23>) -> u23 {
    let xs = [apfloat::APFloat<{EXP + u32:1}> { ..apfloat::APFloat { sign: 0, bexp: 0, fraction: 3 } }, b];
    xs[u32:0].fraction
}
"#;
    let package = compile(source, empty_package("partial"), ResetBehavior::default())?;
    let f = package
        .get_fn("from_constructors")
        .ok_or("missing from_constructors")?;
    for (q, expected) in [(true, "bits[23]:1"), (false, "bits[23]:2")] {
        assert_eq!(
            eval_fn_in_package_value(&package, f, &[IrValue::bool(q)]),
            IrValue::parse_typed(expected)?
        );
    }
    let f = package
        .get_fn("from_argument")
        .ok_or("missing from_argument")?;
    let b = IrValue::parse_typed("(bits[1]:1, bits[8]:127, bits[23]:42)")?;
    assert_eq!(
        eval_fn_in_package_value(&package, f, &[b]),
        IrValue::parse_typed("bits[23]:3")?
    );

    for prefix in ["7", "{u32:8 - u32:1}"] {
        let source = format!(
            r#"
use apfloat;
fn invalid(b: apfloat::APFloat<8, 23>, q: bool) -> u23 {{
    let x = if q {{
        apfloat::APFloat<{prefix}> {{ ..apfloat::APFloat {{ sign: 0, bexp: 0, fraction: 1 }} }}
    }} else {{ b }};
    x.fraction
}}
"#
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains("struct splat base type"), "{error}");
    }
    Ok(())
}
