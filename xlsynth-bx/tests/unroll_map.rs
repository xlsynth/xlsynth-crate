// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::thread;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module, run_tests,
};

#[test]
fn nested_map_inputs_and_bodies_compile_on_a_small_stack() -> Result<(), String> {
    thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let mut input = "xs".to_owned();
            let mut body = "x".to_owned();
            for _ in 0..60 {
                input = format!("unroll_map! lane in {input} {{ lane }}");
                body = format!("unroll_map! _ in u1:0..u1:1 {{ {body} }}");
            }
            let output = format!("u8{}", "[1]".repeat(60));
            let source = format!(
                "fn inputs(xs:u8[1])->u8[1] {{ {input} }} fn bodies(x:u8)->{output} {{ {body} }}"
            );
            compile(&source, empty_package("depth"), ResetBehavior::default()).map(|_| ())
        })
        .map_err(|e| e.to_string())?
        .join()
        .expect("BX compiler thread panicked")
}

#[test]
fn pure_maps_preserve_lexical_scope_and_infer_element_types() -> Result<(), String> {
    let source = r#"
fn transform(xs: (u8, u8)[3], x: u8) -> (u8[3], u16[3], u8) {
    let sums = unroll_map! (a, b) in xs {
        let x = x + a;
        x + b
    };
    let wide: u16[3] = unroll_map! _ in xs { zero!() };
    (sums, wide, x)
}
const RESULT = unroll_map! i in u8:2..=u8:4 { (i, i * i) };
const_assert!(RESULT[1] == (u8:3, u8:9));
fn identity<T: type>(v: T) -> T { v }
fn first<T: type>(a: T, b: T) -> T { a }
fn infer_siblings(xs: u8[1], witness: u16[1], en: bool) -> (u16[1], u16[1], u8[1], u8[1]) {
    let a = first(unroll_map! _ in xs { zero!() }, witness);
    let b = first(witness, unroll_map! _ in xs { zero!() });
    let c = if en { unroll_map! _ in xs { zero!() } } else { unroll_map! x in xs { x } };
    let d = if en { unroll_map! x in xs { x } } else { unroll_map! _ in xs { zero!() } };
    (a, b, c, d)
}
fn annotated_fold(xs: u8[2]) -> u16 {
    for (x, acc): (u16, u16) in (unroll_map! _ in xs { zero!() }) { acc + x }(u16:7)
}
fn width<N: u32, W: u32>(a: uN[W][N], b: uN[W][N]) -> u32 { W }
fn infer_tuple(xs: u8[1], witness: (u16, u8)[1]) -> (u16, u8)[1] {
    let answer = first(unroll_map! x in xs { let y = x + u8:1; (zero!(), y) }, witness);
    answer
}
fn infer_conditional(xs: u8[1], witness: u16[1], en: bool) -> u16[1] {
    let result = first(
        if en { unroll_map! _ in xs { zero!() } } else { unroll_map! _ in xs { zero!() } },
        witness);
    result
}
#[test] fn check_values() {
    let pairs = [(u8:2, u8:7), (u8:1, u8:3), (u8:9, u8:5)];
    assert_eq(transform(pairs, u8:10), ([u8:19, u8:14, u8:24], u16[3]:[0, ...], u8:10));
    assert_eq((unroll_map! pair in pairs { pair.1 })[2], u8:5);
    assert_eq(identity(unroll_map! (x, _) in pairs { x }), [u8:2, u8:1, u8:9]);
    assert_eq(infer_siblings([u8:9], [u16:300], true), ([u16:0], [u16:300], [u8:0], [u8:9]));
    assert_eq(infer_siblings([u8:9], [u16:300], false), ([u16:0], [u16:300], [u8:9], [u8:0]));
    assert_eq(annotated_fold([u8:1, u8:2]), u16:7);
    assert_eq(width(unroll_map! _ in pairs { zero!() }, [u13:1, u13:2, u13:3]), u32:13);
    assert_eq(infer_tuple([u8:99], [(u16:99, u8:0)]), [(u16:0, u8:100)]);
    assert_eq(infer_conditional([u8:2], [u16:300], true), [u16:0]);
}
"#;
    let report = run_tests(
        source,
        Path::new("maps.bx"),
        empty_package("maps"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn contextual_maps_use_typed_siblings_in_either_order() -> Result<(), String> {
    let source = r#"
fn first<T: type>(a: T, b: T) -> T { a }
fn identity<T: type>(a: T) -> T { a }
fn width<W: u32>(a: uN[W][1][1]) -> u32 { W }
fn combine(xs: u8[1], en: bool) -> (u16[1], u16[1], u16[1], u16[1]) {
    let a = if en { unroll_map! x in xs { const if true { zero!() } else { x } } }
            else { unroll_map! _ in xs { u16:2 } };
    let b = if en { unroll_map! _ in xs { u16:2 } }
            else { unroll_map! x in xs { const if true { zero!() } else { x } } };
    let c = first(unroll_map! i in u1:0..u1:1 { const if i == u1:0 { zero!() } else { i } }, [u16:3]);
    let d = first([u16:3], unroll_map! i in u1:0..u1:1 { const if i == u1:0 { zero!() } else { i } });
    (a, b, c, d)
}
fn nested(xs: u8[0][1], witness: u16[0][1], en: bool) -> (u16[0][1], u16[0][1], u16[0][1]) {
    let a = first(unroll_map! row in xs { unroll_map! x in row { x } }, witness);
    let b = first(witness, unroll_map! row in xs { unroll_map! x in row { x } });
    let c = if en { unroll_map! row in xs { unroll_map! x in row { x } } }
            else { unroll_map! _ in xs { u16[0]:[] } };
    (a, b, c)
}
fn independent(xs: u8[1][1]) -> (u8[1][1], u32) {
    let a = identity(unroll_map! row in xs { unroll_map! x in row { x } });
    let b = width(unroll_map! row in xs { unroll_map! x in row { x } });
    (a, b)
}
struct S<T: type> { field: T }
fn records(xs: u8[1], witness: S<u16>[1]) -> (S<u16>[1], S<u16>[1], S<u8>[1]) {
    let a = first(unroll_map! _ in xs { S { field: zero!() } }, witness);
    let b = first(witness, unroll_map! _ in xs { S { field: zero!() } });
    let c = identity(unroll_map! x in xs { S { field: x } });
    (a, b, c)
}
fn nested_records(xs: u8[1][1], witness: (u8[1], S<u16>)[1]) -> (u8[1], S<u16>)[1] {
    let a = first(unroll_map! row in xs {
        (unroll_map! x in row { x }, S { field: zero!() })
    }, witness);
    a
}
"#;
    compile(
        source,
        empty_package("map_witnesses"),
        ResetBehavior::default(),
    )?;
    Ok(())
}

#[test]
fn compile_time_selection_in_a_map_only_instantiates_selected_lanes() -> Result<(), String> {
    let source = r#"
block delay(clk, x: u8) -> u8 { reg q: u8 = 0; q := x; q }
pub block lanes(clk, xs: u8[4]) -> u8[4] {
    unroll_map! i in u32:0..u32:4 {
        const if (i & u32:1) == u32:0 { delay(xs[i]) } else { xs[i] }
    }
}
"#;
    let pkg = compile(
        source,
        empty_package("conditional_lanes"),
        ResetBehavior::default(),
    )?;
    let children = &pkg.get_block("lanes").unwrap().instantiations;
    assert_eq!(children.len(), 2);
    assert!(children.iter().all(|child| child.block == "delay"));
    Ok(())
}

#[test]
fn typed_empty_maps_create_no_children_and_preserve_the_element_type() -> Result<(), String> {
    let source = r#"
block delay(clk, x: u8) -> u8 { reg q: u8 = 0; q := x; q }
#[allow(combinational_block)]
pub block empty(clk, pairs: (u8, u8)[0]) -> u8[0] {
    unroll_map! (x, _) in pairs { delay(x) }
}
fn promote(xs: u8[0], pick: bool) -> (u16[0], u16[0]) {
    let a: u16[0] = unroll_map! x in xs { x as u16 };
    let b = if pick { unroll_map! x in xs { x as u16 } } else { a };
    (a, b)
}
fn choose<T: type>(a: T, b: T) -> T { a }
fn from_sibling(xs: u8[0], witness: u16[0]) -> u16[0] {
    let selected = choose(unroll_map! x in xs { x as u16 }, witness);
    selected
}
fn from_conditional(xs: u8[0], witness: u16[0], pick: bool) -> u16[0] {
    let selected = choose(
        if pick { unroll_map! x in xs { x as u16 } } else { unroll_map! _ in xs { zero!() } },
        witness);
    selected
}
fn length<N: u32>(xs: u16[N]) -> u32 { N }
fn from_length(xs: u8[0]) -> u32 {
    length(unroll_map! x in xs { x as u16 })
}
"#;
    let pkg = compile(
        source,
        empty_package("empty_maps"),
        ResetBehavior::default(),
    )?;
    assert!(pkg.get_block("empty").unwrap().instantiations.is_empty());
    Ok(())
}

#[test]
fn contextual_map_dimensions_follow_branches_and_nested_results() -> Result<(), String> {
    let source = r#"
fn length<N:u32>(xs:u16[N]) -> u32 { N }
fn inner_length<N:u32>(xs:u16[N][1]) -> u32 { N }
fn dimensions<N:u32, W:u32>(xs:uN[W][N], witness:uN[W]) -> (u32, u32) { (N, W) }
fn measure<N:u32>(xs:u8[N], pick:bool) -> (u32, u32, (u32, u32)) {
    let rows = [xs];
    let a = length(if pick { unroll_map! _ in xs { zero!() } }
                   else { unroll_map! _ in xs { zero!() } });
    let b = inner_length(unroll_map! row in rows { unroll_map! _ in row { zero!() } });
    let c = dimensions(match pick {
        true => unroll_map! _ in xs { zero!() },
        _ => unroll_map! _ in xs { zero!() },
    }, u13:7);
    (a, b, c)
}
#[test] fn verify() {
    assert_eq(measure(u8[0]:[], true), (u32:0, u32:0, (u32:0, u32:13)));
    assert_eq(measure([u8:1, u8:2, u8:3], false), (u32:3, u32:3, (u32:3, u32:13)));
}
"#;
    let report = run_tests(
        source,
        Path::new("map_dimensions.bx"),
        empty_package("map_dimensions"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);

    let mismatch = r#"
fn length<N:u32>(xs:u16[N]) -> u32 { N }
fn different(xs:u8[1], ys:u8[2], pick:bool) -> u32 {
    length(if pick { unroll_map! _ in xs { zero!() } }
           else { unroll_map! _ in ys { zero!() } })
}
"#;
    let error = check_module(&parse_module(mismatch)?).unwrap_err();
    assert!(
        error.contains("conflicting values inferred for parametric 'N'"),
        "{error}"
    );
    Ok(())
}

#[test]
fn maps_report_invalid_result_shapes_and_child_call_sites() -> Result<(), String> {
    for (source, reason) in [
        (
            "fn sample(xs:u8[0]) -> u8 { let y = unroll_map! x in xs { x }; u8:0 }",
            "element type",
        ),
        (
            "fn sample(xs:u8[0]) -> u8[0] { unroll_map! x in xs { missing } }",
            "missing",
        ),
        (
            "fn sample(xs:u8[0]) -> u8[0] { unroll_map! (x,y) in xs { x } }",
            "tuple",
        ),
        (
            "fn sample(xs:u8[2]) -> u8[3] { unroll_map! x in xs { x } }",
            "expected array length 3, got 2",
        ),
        (
            "fn sample() -> u8 { let a = unroll_map! i in u32:0..u32:2 { const if i == u32:0 { u4:2 } else { u8:2 } }; a[0] as u8 }",
            "u4",
        ),
        (
            "block d(clk,x:u8)->u8 { reg q:u8 = 0; q := x; q } fn sample(xs:u8[2])->u8[2] { unroll_map! x in xs { d(x) } }",
            "pure function",
        ),
        (
            "block d(clk,x:u8)->u8 { reg q:u8 = 0; q := x; q } block sample(clk,xs:u8[2],en:bool)->u8[2] { if en { unroll_map! x in xs { d(x) } } else { xs } }",
            "unconditional",
        ),
    ] {
        let err = check_module(&parse_module(source)?).unwrap_err();
        assert!(err.contains(reason), "{source}: {err}");
    }
    Ok(())
}

#[test]
fn selected_map_lanes_supply_sibling_types_in_either_operand_order() -> Result<(), String> {
    let body = "{ const if i == u32:5 { u13:77 } else { zero!() } }";
    let contextual = format!("unroll_map! i in u32:2..u32:4 {body}");
    let witness = format!("unroll_map! i in u32:5..u32:7 {body}");
    let source = format!(
        r#"
fn first<T: type>(a: T, b: T) -> T {{ a }}
fn branches(pick: bool) -> (u13[2], u13[2]) {{
    let a = if pick {{ {contextual} }} else {{ {witness} }};
    let b = if pick {{ {witness} }} else {{ {contextual} }};
    (a, b)
}}
#[test] fn check_functions() {{
    let a = first({contextual}, {witness});
    let b = first({witness}, {contextual});
    assert_eq(a, [u13:0, u13:0]);
    assert_eq(b, [u13:77, u13:0]);
    assert_eq(branches(true), (a, b));
    assert_eq(branches(false), (b, a));
}}
#[allow(combinational_block)]
block first_port<T>(a: T, b: T) -> T {{ a }}
#[test] block check_ports() {{
    let a = first_port({contextual}, {witness});
    let b = first_port({witness}, {contextual});
    tick!();
    assert_eq(a, [u13:0, u13:0]);
    assert_eq(b, [u13:77, u13:0]);
}}
"#
    );
    let report = run_tests(
        &source,
        Path::new("lane_witness.bx"),
        empty_package("lane_witness"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 2);
    for test in report.tests {
        assert_eq!(test.failure, None, "{}", test.name);
    }
    Ok(())
}

#[test]
fn selected_map_arguments_take_their_element_types_from_peers() -> Result<(), String> {
    let source = r#"
fn first<T: type>(a: T, b: T) -> T { a }
fn field<T: type, U: type>(pair: (T, U), witness: U) -> T { pair.0 }
fn select(xs: u8[1], witness: u16[1]) -> (u16[1], u16[1], (u16[1], u8), (u16[1], u8)) {
    let a = first(const if true { unroll_map! _ in xs { zero!() } } else { xs }, witness);
    let b = first(witness, const if true { unroll_map! _ in xs { zero!() } } else { xs });
    let c = first(const if true { (unroll_map! _ in xs { zero!() }, u8:1) } else { (xs, u8:2) },
                  (witness, u8:3));
    let d = first((witness, u8:3),
                  const if true { (unroll_map! _ in xs { zero!() }, u8:1) } else { (xs, u8:2) });
    (a, b, c, d)
}
#[test] fn verify() {
    assert_eq(select([u8:1], [u16:37]), ([u16:0], [u16:37], ([u16:0], u8:1), ([u16:37], u8:3)));
    let chosen = field(const if true { (99, unroll_map! _ in [u8:1] { zero!() }) }
                       else { (u8:2, [u8:3]) }, [u16:2]);
    assert_eq(chosen, u7:99);
}
"#;
    let report = run_tests(
        source,
        Path::new("selected_maps.bx"),
        empty_package("selected_maps"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}
