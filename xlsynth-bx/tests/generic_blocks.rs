// SPDX-License-Identifier: Apache-2.0

mod support;

#[path = "generic_blocks/callbacks.rs"]
mod callbacks;

use std::collections::BTreeSet;
use std::error::Error;
use std::thread;
use support::search_paths;

use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, parse_module};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

const PIPE_REG: &str = r#"
block pipe_reg<T>(clk, valid_in: bool, data_in: T) -> (bool, T) {
    reg valid_q: bool = false;
    reg data_q: T;
    valid_q := valid_in;
    if valid_in { data_q := data_in; }
    (valid_q, data_q)
}
"#;

#[test]
fn generic_register_preserves_an_imported_apfloat_instance() -> xlsynth_vastly::Result<()> {
    let source = r#"
use float32::F32;
block hold<T>(clk, en: bool, x: T) -> T {
    reg q: T = zero!<T>();
    if en { q := x; }
    q
}
pub block fp_register(clk, en: bool, s: u1, e: u8, f: u23) -> F32 {
    let input = F32 { fraction: f, sign: s, bexp: e };
    let saved = hold(en, input);
    saved
}
"#;
    let mut package = compile(source, empty_package("fp"), ResetBehavior::default()).unwrap();
    inline_all_blocks_in_package(&mut package).unwrap();
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let module = compile_pipeline_module(&rtl.system_verilog)?;
    let mut runner = FixtureRunner::new(&module)?;
    let ports = runner.port_bindings();
    let rst = ports.input("rst")?;
    let en = ports.input("en")?;
    let s = ports.input("s")?;
    let e = ports.input("e")?;
    let f = ports.input("f")?;
    for (reset, enable, [sign, exponent, fraction], expected) in [
        (true, true, [1, 127, 3], [0, 0, 0]),
        (false, true, [1, 130, 0x400042], [1, 130, 0x400042]),
        (false, false, [0, 42, 1], [1, 130, 0x400042]),
        (true, false, [1, 42, 1], [0, 0, 0]),
    ] {
        let edge = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_bool(&rst, reset)?;
            drive.set_bool(&en, enable)?;
            drive.set_u64(&s, sign)?;
            drive.set_u64(&e, exponent)?;
            drive.set_u64(&f, fraction)
        })?;
        let actual = ["sign", "bexp", "fraction"].map(|port| edge.outputs[port].to_u64_if_known());
        assert_eq!(actual, expected.map(Some));
    }
    Ok(())
}

#[test]
fn generic_boolean_and_byte_literals_preserve_their_intrinsic_width() -> Result<(), Box<dyn Error>>
{
    let source = |ty: &str, body: &str| {
        format!(
            "block fill<T>(clk, x: T) -> T {{ {body} }} \
         block wrapper(clk, x: {ty}) -> {ty} {{ let value = fill(x); value }}"
        )
    };
    for (ty, literal, expected) in [
        ("bool", "T:false", "bits[1]:0"),
        ("s1", "T:true", "bits[1]:1"),
        ("u8", "T:'a'", "bits[8]:97"),
        ("s8", r"T:'\xff'", "bits[8]:255"),
    ] {
        let checked = check_module(&parse_module(&source(ty, literal))?)?;
        let xlsynth_bx::check::TypedExprKind::Literal(value) = &checked.blocks()[0].output.kind
        else {
            panic!("expected {literal} to lower to a literal");
        };
        assert_eq!(value, &xlsynth_pir::IrValue::parse_typed(expected)?);
        let body = format!("match x {{ {literal} => x, _ => x }}");
        check_module(&parse_module(&source(ty, &body))?)?;
    }
    for (ty, literal, diagnostic) in [
        (
            "uN[0]",
            "T:false",
            "a boolean literal requires at least one bit",
        ),
        (
            "u7",
            r"T:'\0'",
            "a character literal requires at least eight bits",
        ),
    ] {
        for body in [
            literal.to_owned(),
            format!("match x {{ {literal} => x, _ => x }}"),
        ] {
            let module = parse_module(&source(ty, &body))?;
            assert_eq!(
                check_module(&module).unwrap_err(),
                format!("block 'fill' (fill__0): result: {diagnostic}"),
            );
        }
    }
    Ok(())
}

#[test]
fn shared_pipe_reg_specialization_has_independent_state() -> xlsynth_vastly::Result<()> {
    let lanes = r#"
struct Lanes { left_valid: bool, left_data: u8, right_valid: bool, right_data: u8 }
block lanes(clk, en_l: bool, x: u8, en_r: bool, y: u8) -> Lanes {
    let left = pipe_reg(en_l, x);
    let right = pipe_reg(en_r, y);
    Lanes { left_valid: left.0, left_data: left.1, right_valid: right.0, right_data: right.1 }
}
"#;
    let mut package = compile(
        &format!("{lanes}{PIPE_REG}"),
        empty_package("design"),
        ResetBehavior::default(),
    )
    .unwrap();
    let top = package.get_top_block().unwrap();
    assert_eq!(top.name, "lanes");
    let [left, right] = top.instantiations.as_slice() else {
        panic!("expected two pipe_reg instances");
    };
    assert_ne!(left.name, right.name);
    assert_eq!(left.block, right.block);
    assert_eq!(package.members.len(), 2);
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    inline_all_blocks_in_package(&mut package).unwrap();
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let module = compile_pipeline_module(&rtl.system_verilog)?;
    let mut runner = FixtureRunner::new(&module)?;
    let ports = runner.port_bindings();
    let rst = ports.input("rst")?;
    let en_l = ports.input("en_l")?;
    let en_r = ports.input("en_r")?;
    let x = ports.input("x")?;
    let y = ports.input("y")?;
    for (reset, left_en, right_en, left_in, right_in, expected) in [
        (true, true, true, 10, 20, [0, 10, 0, 20]),
        (false, true, false, 11, 21, [1, 11, 0, 20]),
        (false, false, true, 12, 22, [0, 11, 1, 22]),
        (false, false, false, 13, 23, [0, 11, 0, 22]),
        (false, true, true, 14, 24, [1, 14, 1, 24]),
        (true, false, true, 15, 25, [0, 14, 0, 25]),
    ] {
        let edge = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_bool(&rst, reset)?;
            drive.set_bool(&en_l, left_en)?;
            drive.set_bool(&en_r, right_en)?;
            drive.set_u64(&x, left_in)?;
            drive.set_u64(&y, right_in)
        })?;
        let actual = ["left_valid", "left_data", "right_valid", "right_data"]
            .map(|name| edge.outputs[name].to_u64_if_known());
        assert_eq!(actual, expected.map(Some));
    }
    Ok(())
}

#[test]
fn pipe_payloads_retain_width_signedness_and_nominal_identity() -> Result<(), Box<dyn Error>> {
    let lanes = r#"
struct Request { id: u8 }
struct Reply { id: u8 }
block lanes(clk, v: bool, one: u1, x: u8, y: s8, w: uN[97], p: (u2, bool),
            a: u4[3], req: Request, rep: Reply) -> (bool, u8, s8, uN[97], (u2, bool), u4[3], Request, Reply) {
    let flag = pipe_reg(v, v);
    let single_bit = pipe_reg(v, one);
    let unsigned_byte = pipe_reg(v, x);
    let signed_byte = pipe_reg(v, y);
    let wide = pipe_reg(v, w);
    let pair = pipe_reg(v, p);
    let array = pipe_reg(v, a);
    let request = pipe_reg(v, req);
    let reply = pipe_reg(v, rep);
    (flag.1 && single_bit.1, unsigned_byte.1, signed_byte.1, wide.1, pair.1, array.1, request.1, reply.1)
}
"#;
    let package = compile(
        &format!("{lanes}{PIPE_REG}"),
        empty_package("design"),
        ResetBehavior::default(),
    )?;
    let top = package.get_top_block().unwrap();
    let definitions: BTreeSet<_> = top.instantiations.iter().map(|i| &i.block).collect();
    assert_eq!(top.instantiations[0].block, top.instantiations[1].block);
    assert_eq!(definitions.len(), 8);
    assert_eq!(package.members.len(), 9);
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    Ok(())
}

#[test]
fn unrolling_specializes_nested_generic_calls_and_register_resets() -> Result<(), Box<dyn Error>> {
    let blocks = r#"
block channels(clk, xs: (u8, u4[2])[2]) -> (u8, u4)[2] {
    unroll_for! (i, acc) in u32:0..u32:2 {
        let sample = capture(xs[i]);
        update(acc, i, sample)
    }(zero!<(u8, u4)[2]>())
}
block capture<A, B>(clk, x: (A, B[2])) -> (A, B) {
    let first = pipe_reg(true, x.0);
    reg last: B = zero!<B>();
    last := x.1[1];
    (first.1, last)
}
"#;
    let package = compile(
        &format!("{blocks}{PIPE_REG}"),
        empty_package("design"),
        ResetBehavior::default(),
    )?;
    let top = package.get_top_block().unwrap();
    let [first, second] = top.instantiations.as_slice() else {
        panic!("expected a capture per unrolled iteration");
    };
    assert_eq!(first.block, second.block);
    let capture = package.get_block(&first.block).unwrap();
    assert_eq!(capture.instantiations.len(), 1);
    assert_eq!(capture.registers.len(), 1);
    assert_eq!(
        capture.registers[0].reset_value,
        Some(xlsynth_pir::IrValue::parse_typed("bits[4]:0")?)
    );
    assert_eq!(package.members.len(), 3);
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn unused_generic_blocks_do_not_create_ir_or_compete_for_top() -> Result<(), String> {
    let source = r#"
pub block unused<T>(clk, x: T) -> T { x }
#[allow(combinational_block)]
block wrapper(clk, x: u8) -> u8 { x }
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    assert_eq!(package.members.len(), 1);
    assert_eq!(package.get_top_block().unwrap().name, "wrapper");
    Ok(())
}

#[test]
fn unused_generic_signatures_validate_nominal_types_and_fixed_dimensions() {
    for (source, expected) in [
        (
            "block unused<T>(clk, data: T) -> Missing { data }",
            "block 'unused': result: unknown struct 'Missing'",
        ),
        (
            "block unused<T>(clk, data: T[1048576]) -> T { data[0] }",
            "block 'unused': input 'data': type T[1048576] exceeds maximum concrete type size 1048576",
        ),
        (
            "block unused<T>(clk, out_0: T) -> (T, u8) { (out_0, 0) }",
            "block 'unused': duplicate name 'out_0'",
        ),
    ] {
        let module = parse_module(source).unwrap();
        assert_eq!(check_module(&module).unwrap_err(), expected);
    }
}

#[test]
fn block_dimensions_use_constants_from_the_definition_scope() -> Result<(), String> {
    let source = r#"
const LANES = u32:2;
const WIDTH = u32:7;
fn last(xs: u8[2]) -> u7 { xs[u32:1] as u7 }
block select<T, F: fn(T[LANES]) -> uN[WIDTH]>(clk, xs: T[LANES]) -> uN[WIDTH] {
    let saved: T[LANES] = xs;
    F(saved)
}
extern block device(clk, xs: u8[LANES]) -> uN[WIDTH];
block wrapper(clk, LANES: u32, WIDTH: u32, xs: u8[2]) -> u7 {
    let selected = select<last>(xs);
    selected
}
"#;
    let checked = check_module(&parse_module(source)?)?;
    let [selected, wrapper] = checked.blocks() else {
        panic!("expected the select specialization and its caller");
    };
    assert_eq!(selected.params[0].ty, wrapper.params[2].ty);
    assert_eq!(selected.result, wrapper.result);
    let [device] = checked.extern_blocks() else {
        panic!("expected the device declaration");
    };
    assert_eq!(device.params, selected.params);
    assert_eq!(device.result, selected.result);
    Ok(())
}

#[test]
fn unused_block_signatures_reject_unavailable_dimension_names() -> Result<(), String> {
    for source in [
        "block b<T>(clk, xs: T[N]) -> T { xs[0] } const N = u32:2;",
        "const N = u32:2; block b<N>(clk, xs: N[N]) -> N { xs[0] }",
        "const N = u32:2; block b<T, N: fn(T[N]) -> T>(clk, x: T) -> T { x }",
        "const N = u32:2; extern block device(clk, N: u32, xs: u8[N]) -> u8;",
    ] {
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(error.contains("'N'"), "{source}\n{error}");
    }
    Ok(())
}

#[test]
fn empty_block_loops_resolve_dimension_names_under_loop_binders() -> Result<(), String> {
    let source = r#"
const N = u32:8;
block fold(clk, x: u8) -> u8 {
    unroll_for! (i, acc) in u32:0..u32:0 {
        let y: uN[N] = acc;
        y
    }(x)
}
"#;
    check_module(&parse_module(source)?)?;
    let shadowed = source.replace("(i, acc)", "(N, acc)");
    check_module(&parse_module(&shadowed)?)?;
    let unknown = source.replace("uN[N]", "uN[missing]");
    assert!(
        check_module(&parse_module(&unknown)?)
            .unwrap_err()
            .contains("'missing'")
    );
    Ok(())
}

#[test]
fn linking_preserves_a_block_with_the_generated_specialization_name() -> Result<(), Box<dyn Error>>
{
    let source = r#"
block echo<T>(clk, x: T) -> T { x }
block wrapper(clk, x: u8) -> u8 { let child = echo(x); child }
"#;
    let standalone = compile(source, empty_package("design"), ResetBehavior::default())?;
    let child_name = &standalone.get_top_block().unwrap().instantiations[0].block;
    let linked = compile(
        &format!("#[allow(combinational_block)] block {child_name}(clk) -> bool {{ false }}"),
        empty_package("design"),
        ResetBehavior::default(),
    )?;
    let original = linked.get_block(child_name).unwrap().to_string();
    let combined = compile(source, linked, ResetBehavior::default())?;
    assert_eq!(
        combined.get_block(child_name).unwrap().to_string(),
        original
    );
    let top = combined.get_top_block().unwrap();
    assert_eq!(top.name, "wrapper");
    assert_ne!(&top.instantiations[0].block, child_name);
    xlsynth::IrPackage::parse_ir(&combined.to_string(), None)?;
    Ok(())
}

#[test]
fn concrete_arguments_supply_types_to_neighboring_literals() -> Result<(), Box<dyn Error>> {
    let source = r#"
block same<T>(clk, first: T, second: T) -> T { first }
block parts<T>(clk, pair: (u8, T[2])) -> T { pair.1[0] }
block wrapper(clk, x: u12) -> (u12, u12) {
    let first = same(1, x);
    let second = parts((3, [4, x]));
    (first, second)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn constructor_arguments_infer_type_parameters_from_corresponding_members()
-> Result<(), Box<dyn Error>> {
    let source = r#"
block same<T>(clk, a: T, b: T) -> T { a }
block pick<T>(clk, a: T) -> T { a }
block second<T>(clk, pair: (u8, T)) -> T { pair.1 }
block nested<T>(clk, a: (T, T[1])) -> T { a.0 }
block wrapper(clk, c: bool, x: u8, y: u12) -> ((u8, u12), (u8[2], u12)[1], (u8, u12)[2], u12, u12) {
    let pair = same((x, 0), (0, y));
    let array = nested(([([x, 0], 1)], [[([0, 0], y)]]));
    let only = pick([(x, 1), (0, y)]);
    let conditional = second(if c { (0, y) } else { (1, y) });
    let matched = second(match c { false => (0, y), true => (1, y) });
    (pair, array, only, conditional, matched)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let missing = source.replace("(0, y)", "(0, 1)");
    assert!(
        check_module(&parse_module(&missing)?)
            .unwrap_err()
            .contains("cannot infer type parameter 'T'")
    );
    Ok(())
}

#[test]
fn generic_calls_require_consistent_inferable_type_arguments() {
    let same = r#"
struct A { id: u8 }
struct B { id: u8 }
block same<T>(clk, x: T, y: T) -> T { x }
"#;
    for (template, statement, expected) in [
        (
            same,
            "let bad = same(u8:0, s8:0);",
            "block 'wrapper': let 'bad': input 'y', call to block 'same': conflicting types for type parameter 'T': inferred u8 and s8",
        ),
        (
            same,
            "let bad = same(A { id: u8:0 }, B { id: u8:0 });",
            "block 'wrapper': let 'bad': input 'y', call to block 'same': conflicting types for type parameter 'T': inferred A and B",
        ),
        (
            same,
            "let bad = same(1, 2);",
            "block 'wrapper': let 'bad': cannot infer type parameter 'T' in call to block 'same'; give a data argument an explicit type",
        ),
        (
            "block missing<T>(clk) -> T { zero!<T>() }",
            "let bad = missing();",
            "block 'missing': cannot infer type parameter 'T': it does not appear in any data input",
        ),
        (
            same,
            "inst bad: same; bad.x := u8:0; bad.y := u8:0;",
            "block 'wrapper': generic block 'same' needs data arguments to infer its type parameters; use 'let bad = same(...)'",
        ),
    ] {
        let source = format!("{template} block wrapper(clk) -> () {{ {statement} () }}");
        let parsed = parse_module(&source).unwrap();
        assert_eq!(check_module(&parsed).unwrap_err(), expected, "{statement}");
    }
}

#[test]
fn scoped_match_arguments_supply_types_to_aggregate_ports() -> Result<(), Box<dyn Error>> {
    let source = r#"
block part<T>(clk, x: (u8, T)) -> T { x.1 }
block same<T>(clk, a: T, b: T) -> T { a }
block wrapper(clk, x: u12) -> (u12, (u8, u12), s3) {
    let first = part(match true { _ => { let y = x; (0, y) } });
    let pair = same(match true { _ => { let y = u8:1; (y, 0) } },
                    match true { _ => { let y = x; (0, y) } });
    let signed = part(match true { _ => { let y = s3:-5; (0, y) } });
    (first, pair, signed)
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn complementary_arrays_at_the_expression_depth_limit_compile_on_a_small_stack()
-> Result<(), String> {
    // The call, tuple and leaf occupy three of BX's 64 expression levels.
    thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let depth = 61;
            let suffix = "[1]".repeat(depth);
            let open = "[".repeat(depth);
            let close = "]".repeat(depth);
            for formal in ["T".to_owned(), format!("T{suffix}")] {
                let source = format!(
                    r#"block same<T>(clk, a: {formal}, b: {formal}) -> {formal} {{ a }}
                    block wrapper(clk, x: u8, y: u12) -> (u8, u12){suffix} {{
                        let value = same({open}(x, 0){close}, {open}(0, y){close});
                        value
                    }}"#
                );
                compile(&source, empty_package("design"), ResetBehavior::default())?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?
        .join()
        .expect("BX compiler thread panicked")
}

#[test]
fn recursive_generic_hierarchies_are_rejected_even_if_payload_types_change() {
    for source in [
        r#"
block grow<T>(clk, x: T) -> bool {
    let nested = grow((x, false));
    nested
}
block wrapper(clk) -> bool { let child = grow(u8:0); child }
"#,
        r#"
block child<T>(clk, x: T) -> bool { let feedback = wrapper(); feedback }
block wrapper(clk) -> bool { let nested = child(u8:0); nested }
"#,
    ] {
        let error = check_module(&parse_module(source).unwrap()).unwrap_err();
        assert!(error.contains("recursive block dependency"), "{error}");
    }
}

#[test]
fn bound_payload_keeps_its_nominal_type_in_the_callees_scope() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::create_dir(dir.path().join("fabric"))?;
    std::fs::write(
        dir.path().join("fabric/wire.x"),
        "pub struct Header { value: u8 }",
    )?;
    std::fs::write(
        dir.path().join("payload_types.x"),
        "import fabric.wire; pub type Packet = wire::Header;",
    )?;
    let source = r#"
use payload_types::Packet;
struct Local { value: s8 }
block held<T>(clk, data: T) -> T {
    let Local = true;
    reg q: T = zero!<T>();
    if Local { q := data; }
    q
}
block capture(clk, wire: Packet, local: Local) -> (u8, s8) {
    let saved_wire = held(wire);
    let saved_local = held(local);
    (saved_wire.value, saved_local.value)
}
"#;
    let options = search_paths(&[dir.path()]);
    let package = xlsynth_bx::compile_with_options(
        source,
        empty_package("generic_import"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(
        package
            .members
            .iter()
            .filter(|m| matches!(m, xlsynth_pir::ir::PackageMember::Block(_)))
            .count(),
        3
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn type_parameters_resolve_in_empty_unrolled_bodies_and_typed_expressions()
-> Result<(), Box<dyn Error>> {
    let source = r#"
block offset<T>(clk, x: T) -> T {
    let empty = unroll_for! (i, acc): (u32, T) in u32:0..u32:0 {
        let item: T = i as T;
        acc + item
    }(x);
    match empty { T:0 => T:1, _ => empty }
}
block sampler(clk, x: u8) -> u8 {
    let out_data = offset(x);
    out_data
}
"#;
    let package = compile(
        source,
        empty_package("generic_expr"),
        ResetBehavior::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn match_pattern_annotations_capture_the_lexical_scope_of_inferred_lets() {
    for pattern in ["T:0", "T:0..=1", "0..=T:1"] {
        let source = format!(
            r#"
block capture<T>(clk, x: T) -> T {{
    let p = match x {{ _ => {{
        let T = x;
        match x {{ {pattern} => x, _ => x }}
    }} }};
    p
}}
block wrapper(clk, x: u8) -> u8 {{ let p = capture(x); p }}
"#
        );
        let module = parse_module(&source).unwrap();
        let error = check_module(&module).unwrap_err();
        assert!(
            error.contains("names a local value here"),
            "{pattern}: {error}"
        );
    }
}

#[test]
fn generated_blocks_share_the_unroll_work_budget() {
    let helper = r#"
block fold<T>(clk, x: T) -> T {
    let y = unroll_for! (i, acc): (uN[1048576], T) in uN[1048576]:0..uN[1048576]:33 { acc }(x);
    y
}
"#;
    let same =
        "block wrapper(clk, x: u8, y: u8) -> (u8, u8) { let a = fold(x); let b = fold(y); (a,b) }";
    let module = parse_module(&format!("{helper}{same}")).unwrap();
    check_module(&module).unwrap();
    let different = same
        .replace("y: u8", "y: s8")
        .replace("(u8, u8)", "(u8, s8)");
    let module = parse_module(&format!("{helper}{different}")).unwrap();
    let error = check_module(&module).unwrap_err();
    assert!(
        error.contains("generic block specialization expansion limit"),
        "{error}"
    );
}

/// Build a compact nominal type whose expanded IR shape has thousands of nodes.
fn tree_payload_declarations(max_depth: usize) -> String {
    let mut source = String::from("struct Tree0 { bit: u1 }\n");
    for depth in 1..=max_depth {
        let child = depth - 1;
        source.push_str(&format!(
            "struct Tree{depth} {{ left: Tree{child}, right: Tree{child} }}\n"
        ));
    }
    source
}

#[test]
fn each_instance_materializes_ports_even_when_its_callback_specialization_is_cached()
-> Result<(), String> {
    let mut source = tree_payload_declarations(14);
    source.push_str(
        r#"
extern fn expand(x: u8) -> (Tree14, Tree14);
block emit<Out, F: fn(u8) -> Out>(clk, x: u8) -> Out { F(x) }
block sample(clk, x: u8) -> u8 {
    unroll_for! (_, acc) in u32:0..u32:64 {
        let wires = emit<expand>(acc);
        acc
    }(x)
}
"#,
    );
    let error = check_module(&parse_module(&source)?).unwrap_err();
    assert!(
        error.contains("child block wiring exceeds the block expansion limit"),
        "{error}"
    );
    Ok(())
}

#[test]
fn checking_many_nominal_inputs_to_one_callback_block_is_bounded() -> Result<(), String> {
    let ports = (0..64)
        .map(|i| format!("p{i}: T"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = ["x"; 64].join(", ");
    let source = format!(
        r#"
{}
extern fn collapse(x: Tree14) -> u8;
block many<T, O, F: fn(T) -> O>(clk, {ports}) -> O {{ F(p0) }}
block sample(clk, x: Tree14) -> u8 {{ let result = many<collapse>({arguments}); result }}
"#,
        tree_payload_declarations(14)
    );
    let error = check_module(&parse_module(&source)?).unwrap_err();
    assert!(
        error.contains("callback block input exceeds the block expansion limit"),
        "{error}"
    );
    Ok(())
}

#[test]
fn both_child_instance_forms_charge_wiring_in_each_callback_specialization() -> Result<(), String> {
    let types = tree_payload_declarations(17);
    for instance in ["inst child: wide;", "let child = wide();"] {
        let source = format!(
            r#"
{types}
extern block wide(clk) -> Tree17;
extern fn a(x:u8) -> u8;
extern fn b(x:u8) -> u8;
extern fn c(x:u8) -> u8;
block keep<T,F:fn(T)->T>(clk, x:T) -> T {{ {instance} F(x) }}
block sample(clk, x:u8) -> u8 {{
    let first = keep<a>(x);
    let second = keep<b>(first);
    let third = keep<a>(second);
    third
}}
"#,
        );
        check_module(&parse_module(&source)?)?;
        let distinct = source.replace("keep<a>(second)", "keep<c>(second)");
        let error = check_module(&parse_module(&distinct)?).unwrap_err();
        assert!(
            error.contains(
                "child block wiring exceeds the generic block specialization expansion limit"
            ),
            "{instance}: {error}"
        );
    }
    Ok(())
}
