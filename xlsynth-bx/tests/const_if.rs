// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs};
use xlsynth_bx::{ResetBehavior, compile, compile_with_options, empty_package};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_eval::eval_fn_in_package_value;

fn compile_design(source: &str) -> Result<Package, String> {
    compile(source, empty_package("design"), ResetBehavior::default())
}

#[test]
fn function_specializations_select_different_result_widths() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn is_byte(width: u32) -> bool { width == u32:8 }
fn repeat_byte<N: u32>(x: u8) -> uN[N] {
    const if is_byte(N) { x } else if N == u32:16 { x ++ x } else { x ++ x ++ x }
}
fn sample(x: u8) -> (u8, u16, u24) {
    (repeat_byte<8>(x), repeat_byte<16>(x), repeat_byte<24>(x))
}
"#;
    let package = compile_design(source)?;
    let sample = package.get_fn("sample").ok_or("missing sample")?;
    assert_eq!(
        eval_fn_in_package_value(&package, sample, &[IrValue::make_ubits(8, 0xab)?]),
        IrValue::parse_typed("(bits[8]:0xab, bits[16]:0xabab, bits[24]:0xababab)")?
    );
    for node in &sample.nodes {
        if let NodePayload::Invoke { to_apply, .. } = &node.payload {
            let specialized = package.get_fn(to_apply).ok_or("missing specialization")?;
            assert!(!specialized.nodes.iter().any(|node| matches!(
                node.payload,
                NodePayload::Sel { .. } | NodePayload::Invoke { .. }
            )));
        }
    }
    Ok(())
}

#[test]
fn unselected_branches_resolve_names_and_skip_typechecking() -> Result<(), String> {
    compile_design(
        "enum Mode: u1 { A = 0 } fn skipped(x:u8)->u8 { const if true { x } else { match x { Mode::Missing => x, _ => x } } }",
    )?;
    for (kind, header) in [
        ("fn", "fn sample(x: u8) -> u8"),
        (
            "block",
            "#[allow(combinational_block)] block sample(x: u8) -> u8",
        ),
    ] {
        for expression in [
            "const if true { x } else { let v = x; v + false }",
            "const if false { false } else { x }",
            "const if true { x } else { x..=false }",
        ] {
            compile_design(&format!("{header} {{ {expression} }}"))?;
        }
        for (expression, diagnostic) in [
            (
                "const if true { x } else { missing }",
                "unknown value 'missing'",
            ),
            (
                "const if false { let y = missing; x } else { x }",
                "unknown value 'missing'",
            ),
            (
                "const if true { let y = x; y } else { y }",
                "unknown value 'y'",
            ),
            (
                "const if true { x } else { missing..x }",
                "unknown value 'missing'",
            ),
            (
                "const if true { x } else { x..=missing }",
                "unknown value 'missing'",
            ),
            (
                "const if x == u8:0 { x } else { x }",
                "const if: predicate must evaluate to a compile-time constant",
            ),
            (
                "const if u8:1 { x } else { x }",
                "const if: expected bool, got u8",
            ),
            (
                "const if false { x } else if x == u8:0 { x } else { x }",
                "const if: predicate must evaluate to a compile-time constant",
            ),
            (
                "const if false { x } else { x + false }",
                "expected u8, got bool",
            ),
        ] {
            let error = compile_design(&format!("{header} {{ {expression} }}")).unwrap_err();
            assert_eq!(
                error,
                format!("{kind} 'sample': result: {diagnostic}"),
                "{expression}"
            );
        }
    }
    compile_design(
        "fn sample(x: u8) -> u8 { const if false { const_assert!(false); x } else { x } }",
    )?;
    Ok(())
}

#[test]
fn skipped_matches_resolve_tuple_bindings_only_inside_their_arm() -> Result<(), String> {
    for body in [
        "const if true { x } else { match (true, x) { (true, byte) => byte + false, _ => x } }",
        "unroll_for! (_, acc) in u32:0..u32:0 { match (true, acc) { (true, byte) => byte + false, _ => x } }(x)",
    ] {
        let source = format!("fn sample(x: u8) -> u8 {{ {body} }}");
        compile_design(&source)?;
        let unbound = source.replace("_ => x", "_ => byte");
        assert_eq!(
            compile_design(&unbound).unwrap_err(),
            "fn 'sample': result: unknown value 'byte'"
        );
    }
    Ok(())
}

#[test]
fn inactive_type_members_resolve_the_qualifier() -> Result<(), String> {
    for source in [
        "struct P { v: u8 } type Alias = P; fn run(x: u8) -> u8 { const if true { x } else { Alias::Missing } }",
        "fn run(x: u8) -> u8 { type Byte = u8; const if true { x } else { Byte::Missing } }",
        "fn run(x: u8) -> u8 { const if true { x } else { type Byte = u8; Byte::Missing } }",
        "fn f<T: type>(x: T) -> T { const if true { x } else { T::MIN } } fn run(x: u8) -> u8 { f(x) }",
        "fn f<T: type>(x: T) -> T { const if true { x } else { match x { T::Missing => x, _ => x } } } type T = u32; fn run(x: u8) -> u8 { f(x) }",
        "fn run(x: u8) -> u8 { const if true { x } else { u8::Missing + s64::MAX } }",
    ] {
        compile_design(source).map_err(|e| format!("{source}: {e}"))?;
    }
    for source in [
        "fn run(x: u8) -> u8 { const if true { x } else { Unknown::Missing } }",
        "fn run(x: u8) -> u8 { const if true { x } else { Later::Missing } } type Later = u8;",
        "type Alias = u8; fn run(x: u8) -> u8 { let Alias = x; const if true { x } else { Alias::Missing } }",
        "fn f<T: type>(x: T) -> T { const if true { x } else { let T = x; T::MIN } } fn run(x: u8) -> u8 { f(x) }",
        "type Alias = u8; fn run(x: u8) -> u8 { const if false { x } else { Alias::Missing } }",
        "fn f<T: type>(x: T) -> T { const if false { x } else { T::Missing } } fn run(x: u8) -> u8 { f(x) }",
        "fn run(x: u8) -> u8 { const if false { x } else { u8::Missing } }",
        "fn run(x: u8) -> u8 { const if true { x } else { u65::Missing } }",
    ] {
        assert!(compile_design(source).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn const_if_with_an_omitted_else_produces_unit_on_the_false_path() -> Result<(), String> {
    let source = "fn sample() { const if false { u8:42 } }";
    compile_design(source)?;
    assert!(compile_design(&source.replace("false", "true")).is_err());
    Ok(())
}

#[test]
fn imported_bx_function_resolves_each_branch_in_its_defining_module() -> Result<(), Box<dyn Error>>
{
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("transform.bx"),
        r#"
        const MASK = u8:0xf0;
        type Byte = u8;
        fn invert(x: u8) -> u8 { !x }
        pub fn keep<T: type>(x: T) -> T {
            const if true { x } else { T::MIN + Byte::Missing + u8::Missing }
        }
        pub fn apply<MASKED: bool>(x: u8) -> u8 {
            const if MASKED { x & MASK } else { invert(x) }
        }
    "#,
    )?;
    let source = r#"
        use transform;
        const MASK = u8:0;
        fn invert(x: u8) -> u8 { x }
        fn sample(x: u8) -> (u8, u8) {
            (transform::keep(transform::apply<true>(x)), transform::apply<false>(x))
        }
    "#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &support::search_paths(&[dir.path()]),
    )?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[IrValue::make_ubits(8, 0x5a)?]
        ),
        IrValue::parse_typed("(bits[8]:0x50, bits[8]:0xa5)")?
    );
    Ok(())
}

#[test]
fn inactive_struct_members_resolve_across_bx_imports() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("packets.bx"),
        r#"
        pub struct Packet { x: u8 }
        struct Private { x: u8 }
        pub fn keep(x: u8) -> u8 { const if true { x } else { Packet::Missing } }
    "#,
    )?;
    let options = support::search_paths(&[dir.path()]);
    let compile_source = |source: &str| {
        compile_with_options(
            source,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        )
    };
    compile_source("use packets; fn run(x: u8) -> u8 { packets::keep(x) }")?;
    for (import, qualifier) in [
        ("use packets;", "packets::Packet"),
        ("use packets::Packet;", "Packet"),
    ] {
        let source = format!(
            "{import} fn run(x: u8) -> u8 {{ const if true {{ x }} else {{ {qualifier}::Missing }} }}"
        );
        compile_source(&source)?;
        assert!(compile_source(&source.replace("true", "false")).is_err());
    }
    assert!(compile_source(
        "use packets; fn run(x: u8) -> u8 { const if true { x } else { packets::Private::Missing } }",
    ).is_err());
    Ok(())
}

#[test]
fn callback_predicate_uses_the_concrete_specialization() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn enable() -> bool { true }
fn disable() -> bool { false }
fn choose<F: fn() -> bool>(x: u8) -> u8 {
    const if F() { x } else { !x }
}
fn sample(x: u8) -> (u8, u8) { (choose<enable>(x), choose<disable>(x)) }
"#;
    let package = compile_design(source)?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[IrValue::make_ubits(8, 0xf0)?],
        ),
        IrValue::parse_typed("(bits[8]:0xf0, bits[8]:0x0f)")?
    );
    Ok(())
}

#[test]
fn selected_index_arithmetic_retains_the_default_index_width() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn sample(xs: bool[257]) -> (bool, bool) {
    (xs[255 + 1], xs[const if true { 255 + 1 } else { false }])
}
"#;
    // Narrowing the addition to eight bits would read xs[0].
    let mut values = vec![IrValue::bool(false); 257];
    values[256] = IrValue::bool(true);
    let package = compile_design(source)?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[IrValue::make_array(&values)?],
        ),
        IrValue::make_tuple(&[IrValue::bool(true), IrValue::bool(true)])
    );
    Ok(())
}

#[test]
fn selected_untyped_literal_takes_signed_width_from_its_sibling() -> Result<(), Box<dyn Error>> {
    let source = "fn sample(x: s19) -> s19 { (const if true { -1 } else { false }) + x }";
    let package = compile_design(source)?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[IrValue::make_ubits(19, 12345)?],
        ),
        IrValue::make_ubits(19, 12344)?
    );
    Ok(())
}

#[test]
fn selected_function_arguments_supply_or_receive_parametric_widths() -> Result<(), Box<dyn Error>> {
    for (formal, ty) in [("N:u32", "uN[N]"), ("T:type", "T")] {
        let definitions = format!(
            "fn id<{formal}>(a:{ty})->{ty} {{ a }} fn last<{formal}>(a:{ty}, b:{ty})->{ty} {{ b }}"
        );
        let source = format!(
            r#"
{definitions}
fn sample() -> (u16, u16, u16, u16, u16) {{
    let narrow = id(const if true {{ 255 + 1 }} else {{ false }});
    let nested = id(1 + (const if false {{ false }} else {{ 255 }}));
    let bound = last(u16:0, const if true {{ 255 + 1 }} else {{ false }});
    let wide: u16 = id(const if true {{ 255 + 1 }} else {{ false }});
    let ordered = last(const if true {{ 255 }} else {{ false }}, const if true {{ 127 + 1 }} else {{ false }});
    (narrow as u16, nested as u16, bound, wide, ordered as u16)
}}
"#
        );
        let pkg = compile_design(&source)?;
        assert_eq!(
            eval_fn_in_package_value(&pkg, pkg.get_fn("sample").ok_or("missing sample")?, &[]),
            IrValue::parse_typed(
                "(bits[16]:0, bits[16]:0, bits[16]:256, bits[16]:256, bits[16]:128)"
            )?
        );
        for call in [
            "id(1)",
            "id(1 + 255)",
            "last(const if true { 1 } else { false }, u16:2)",
        ] {
            assert!(
                compile_design(&format!(
                    "{definitions} fn sample()->u16 {{ let v={call}; v as u16 }}"
                ))
                .is_err(),
                "{call}"
            );
        }
    }
    Ok(())
}

#[test]
fn selected_function_argument_infers_an_imported_width() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("words.x"),
        r#"
pub fn id<N:u32>(x:uN[N])->uN[N] { x }
pub fn last<U:u32, N:u32>(x:uN[U], a:uN[N], b:uN[N])->uN[N] { b }
pub fn same<N:u32>(a:uN[N], b:uN[N])->uN[N] { b }
pub fn word(a:u16)->u16 { a }
"#,
    )?;
    let source = r#"
use words;
fn sample() -> (u16, u16, u16, u16, u16) {
    let lone = words::id(const if true { 255 + 1 } else { false });
    let narrow = words::last<3>(0, const if true { 255 + 1 } else { false }, 3);
    let wide = words::last<3>(0, u16:3, const if true { 255 + 1 } else { false });
    let implicit = words::same(u16:0, const if true { 255 + 1 } else { false });
    let concrete = words::word(1 + (const if true { 255 } else { false }));
    (lone as u16, narrow as u16, wide as u16, implicit, concrete)
}
"#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &support::search_paths(&[dir.path()]),
    )?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[]
        ),
        IrValue::parse_typed("(bits[16]:0, bits[16]:3, bits[16]:256, bits[16]:256, bits[16]:256)")?
    );
    let bare = "use words; fn sample()->u16 { let x=words::id(1); x as u16 }";
    assert!(
        compile_with_options(
            bare,
            empty_package("design"),
            ResetBehavior::default(),
            &support::search_paths(&[dir.path()])
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn selected_aggregate_argument_supplies_a_generic_type() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn id<T:type>(x:T)->T { x }
fn unpack<T:type>(p:(T,))->T { p.0 }
fn first<T:type>(p:T[2])->T { p[u32:0] }
fn sample()->(u8,u8,u8,u8,u8,u8,u8,u8) {
    let a = id(const if true { (1, 255 + 1) } else { false });
    let b = id([1, const if true { 255 + 1 } else { false }]);
    let c = unpack((const if true { 255+1 } else { false },));
    let d = first([const if true { 255+1 } else { false }, 3]);
    let e = first([3, const if true { 255+1 } else { false }]);
    let f = first([const if true { 3 } else { false }, const if true { 255+1 } else { false }]);
    (a.0 as u8, a.1, b[u32:0], b[u32:1], c as u8, d as u8, e as u8, f as u8)
}
"#;
    let pkg = compile_design(source)?;
    assert_eq!(
        eval_fn_in_package_value(&pkg, pkg.get_fn("sample").ok_or("missing sample")?, &[]),
        IrValue::parse_typed(
            "(bits[8]:1,bits[8]:0,bits[8]:1,bits[8]:0,bits[8]:0,bits[8]:0,bits[8]:3,bits[8]:3)"
        )?
    );
    for (formal, body, arg) in [("(T,)", "p.0", "(1,)"), ("T[2]", "p[u32:0]", "[255+1,3]")] {
        let bare = format!(
            "fn unwrap<T:type>(p:{formal})->T {{ {body} }} fn sample()->u16 {{ let x=unwrap({arg}); x as u16 }}"
        );
        assert!(compile_design(&bare).is_err(), "{arg}");
    }
    Ok(())
}

const BYPASS_STAGE: &str = r#"
block stage<T, BYPASS: bool>(clk, valid: bool, data: T) -> T {
    reg held: T = zero!<T>();
    held := data;
    const if BYPASS { if valid { data } else { held } } else { held }
}
"#;

#[test]
fn generic_feedback_through_registered_output_has_no_combinational_cycle() -> Result<(), String> {
    let wrapper = r#"pub block accumulate(clk, valid: bool, increment: u8) -> u8 {
    let feedback: u8;
    let current = stage<false>(valid, feedback);
    feedback = current + increment;
    current
}
"#;
    let source = format!("{BYPASS_STAGE}{wrapper}");
    let package = compile_design(&source)?;
    let top = package.get_top_block().ok_or("missing accumulate")?;
    let stage = package
        .get_block(&top.instantiations[0].block)
        .ok_or("missing stage")?;
    let output = stage.output_value(stage.get_output_port("out").ok_or("missing out")?);
    assert!(matches!(&stage.get_node(output).payload,
        NodePayload::RegisterRead { register } if register == "held"
    ));
    assert!(
        !stage
            .nodes
            .iter()
            .any(|node| matches!(node.payload, NodePayload::Sel { .. }))
    );

    // The same stage can enable its runtime bypass when its input is
    // independent of its output. The emitted mux must still read the runtime
    // valid port.
    let independent = source.replace(
        "stage<false>(valid, feedback)",
        "stage<true>(valid, increment)",
    );
    let bypass = compile_design(&independent)?;
    let top = bypass.get_top_block().ok_or("missing accumulate")?;
    let stage = bypass
        .get_block(&top.instantiations[0].block)
        .ok_or("missing stage")?;
    let output = stage.output_value(stage.get_output_port("out").ok_or("missing out")?);
    assert!(
        matches!(&stage.get_node(output).payload, NodePayload::Sel { selector, .. }
        if Some(*selector) == stage.get_input_port("valid"))
    );

    for (cyclic_source, flag, underline) in [
        (
            source.replace("stage<false>", "stage<true>"),
            "true",
            "-------------------------------------------",
        ),
        (
            source.replace("const if BYPASS", "if BYPASS"),
            "false",
            "--------------------------------------------",
        ),
    ] {
        let expected = format!(
            r#"error: combinational cycle in block 'accumulate'
  --> <bx>:10:5
   |
 9 |     let current = stage<{flag}>(valid, feedback);
   |     {underline} `current.out` depends on `current.data`
10 |     feedback = current + increment;
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ driver of `feedback`
   |
   = note: dependency cycle: current.out -> current.data -> feedback -> current.out"#
        );
        assert_eq!(compile_design(&cyclic_source).unwrap_err(), expected);
    }
    Ok(())
}

#[test]
fn selected_register_write_and_output_have_no_runtime_select() -> Result<(), String> {
    let source = r#"
block stage<BYPASS: bool>(clk, a: u8, b: u8) -> u8 {
    reg held: u8;
    held := const if BYPASS { a } else { b };
    const if BYPASS { b } else { held }
}
pub block pair(clk, a: u8, b: u8) -> (u8, u8) {
    let stored = stage<false>(a, b);
    let direct = stage<true>(a, b);
    (stored, direct)
}
"#;
    let package = compile_design(source)?;
    let pair = package.get_top_block().ok_or("missing pair")?;
    assert_eq!(pair.instantiations.len(), 2);
    for (instance, input) in pair.instantiations.iter().zip(["b", "a"]) {
        let stage = package.get_block(&instance.block).ok_or("missing stage")?;
        let data = stage.get_input_port(input).ok_or("missing input")?;
        assert!(stage.nodes.iter().any(|node| matches!(&node.payload,
            NodePayload::RegisterWrite { register, arg, .. } if register == "held" && *arg == data
        )));
        assert!(
            !stage
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Sel { .. }))
        );
    }
    Ok(())
}

#[test]
fn structural_const_if_selects_one_child_or_forwards_its_input() -> Result<(), String> {
    let source = r#"
block pipe_reg(clk, x: u8) -> u8 { reg q: u8; q := x; q }
block invert_reg(clk, x: u8) -> u8 { reg q: u8; q := !x; q }
#[allow(combinational_block)]
block stage<PIPE: bool, INVERT: bool>(clk, x: u8) -> u8 {
    let y: u8 = const if PIPE { pipe_reg(x) } else if INVERT { invert_reg(x) } else { x };
    y
}
pub block lanes(clk, x: u8) -> (u8, u8, u8) {
    let piped = stage<true, true>(x);
    let inverted = stage<false, true>(x);
    let direct = stage<false, false>(x);
    (piped, inverted, direct)
}
"#;
    let package = compile_design(source)?;
    let lanes = package.get_top_block().ok_or("missing lanes")?;
    assert_eq!(lanes.instantiations.len(), 3);
    for (instance, expected_child) in
        lanes
            .instantiations
            .iter()
            .zip([Some("pipe_reg"), Some("invert_reg"), None])
    {
        let stage = package.get_block(&instance.block).ok_or("missing stage")?;
        let x = stage.get_input_port("x").ok_or("missing input")?;
        let output = stage.output_value(stage.get_output_port("out").ok_or("missing output")?);
        let Some(expected_child) = expected_child else {
            assert!(stage.instantiations.is_empty());
            assert_eq!(output, x);
            continue;
        };
        let [child] = stage.instantiations.as_slice() else {
            return Err(format!("expected one child in '{}'", instance.name));
        };
        assert_eq!(
            (child.name.as_str(), child.block.as_str()),
            ("y", expected_child)
        );
        assert!(stage.nodes.iter().any(|node| matches!(
            &node.payload, NodePayload::InstantiationInput { instantiation, port_name, arg }
            if instantiation == "y" && port_name == "x" && *arg == x)));
        assert!(matches!(
            &stage.get_node(output).payload, NodePayload::InstantiationOutput { instantiation, port_name }
            if instantiation == "y" && port_name == "out"));
    }
    Ok(())
}

#[test]
fn unrolled_lets_instantiate_only_the_selected_iteration() -> Result<(), String> {
    for (binding, result) in [("(next, _)", "next"), ("next", "next.0")] {
        let source = format!(
            r#"
block sample(clk, x: u8) -> (u8, u8) {{ reg q: u8; q := x; (q, x) }}
pub block pipeline(clk, x: u8) -> u8 {{
    unroll_for! (i, a) in u32:0..u32:3 {{
        let {binding}: (u8, u8) = const if i == u32:1 {{ sample(a) }} else {{ (a, a) }};
        {result}
    }}(x)
}}
"#
        );
        let package = compile_design(&source)?;
        let pipeline = package.get_top_block().ok_or("missing pipeline")?;
        let [child] = pipeline.instantiations.as_slice() else {
            return Err(format!("expected one selected iteration for '{binding}'"));
        };
        assert_eq!(child.block, "sample");
        if binding == "next" {
            assert_eq!(child.name, "next_1");
        }
    }
    Ok(())
}

#[test]
fn unrolled_bypass_literal_retains_its_inferred_width() -> Result<(), String> {
    let source = r#"
block pipe_reg(clk, x: u8) -> u8 { reg q: u8; q := x; q }
#[allow(combinational_block)]
pub block sum(clk, x: u8) -> u8 {
    unroll_for! (_, a) in u32:0..u32:1 {
        let z = BYPASS;
        a + RESULT
    }(x)
}
"#;
    for (bypass, direct, result) in [
        ("1", "1", "z"),
        ("let t = 1; t", "1", "z"),
        ("let t = 1; (t, a)", "(1, a)", "z.0"),
        ("(const if true { 1 } else { a }, a)", "(1, a)", "z.0"),
    ] {
        let source = source.replace("RESULT", result);
        let conditional = format!("const if false {{ pipe_reg(a) }} else {{ {bypass} }}");
        assert_eq!(
            compile_design(&source.replace("BYPASS", &conditional))?.to_string(),
            compile_design(&source.replace("BYPASS", direct))?.to_string(),
            "{bypass}"
        );
    }
    Ok(())
}

#[test]
fn child_in_const_if_requires_a_structural_let_initializer() {
    for body in [
        "if enable { let y = const if true { pipe_reg(x) } else { x }; } x",
        "let y = if enable { const if true { pipe_reg(x) } else { x } } else { x }; y",
        "let y = { let a = x; const if true { pipe_reg(a) } else { a } }; y",
        "let y = const if true { let a = x; pipe_reg(a) } else { x }; y",
        "let y = const if true { pipe_reg(x) + x } else { x }; y",
        "let y = const if true { unroll_for! (_, a) in u32:0..u32:1 { let q = pipe_reg(a); q }(x) } else { x }; y",
        "const if true { pipe_reg(x) } else { x }",
    ] {
        let source = format!(
            "block pipe_reg(clk, x: u8) -> u8 {{ reg q: u8; q := x; q }}
             pub block parent(clk, x: u8, enable: bool) -> u8 {{ {body} }}"
        );
        let error = compile_design(&source).unwrap_err();
        assert!(
            error.contains("call to block 'pipe_reg'"),
            "{body}: {error}"
        );
    }
}

#[test]
fn conditional_loop_seeds_infer_a_default_and_accept_a_contextual_width()
-> Result<(), Box<dyn Error>> {
    for keyword in ["for", "unroll_for!"] {
        for seed in [
            "const if true { 255 + 1 } else { false }",
            "1 + (const if false { false } else { 255 })",
        ] {
            let fold = format!("{keyword} (i, a) in u32:0..u32:2 {{ a }}({seed})");
            let package = compile_design(&format!(
                r#"
fn sample() -> (u16, u16, u16, u16) {{
    let narrow = {fold};
    let wide: u16 = {fold};
    let from_sibling = ({fold}) + u16:1;
    let elements = [({fold}), u16:2];
    (narrow as u16, wide, from_sibling, elements[u32:0])
}}
"#
            ))
            .map_err(|error| format!("{fold}: {error}"))?;
            assert_eq!(
                eval_fn_in_package_value(
                    &package,
                    package.get_fn("sample").ok_or("missing sample")?,
                    &[]
                ),
                IrValue::parse_typed("(bits[16]:0, bits[16]:256, bits[16]:257, bits[16]:256)")?,
                "{fold}"
            );
        }
        for seed in ["1", "if true { 1 } else { 1 }"] {
            let source = format!(
                "fn sample() -> u8 {{ let y = {keyword} (i, a) in u32:0..u32:2 {{ a }}({seed}); y as u8 }}"
            );
            assert!(compile_design(&source).is_err(), "{source}");
        }
    }
    Ok(())
}

#[test]
fn skipped_child_calls_check_names_and_defer_the_hardware_signature() -> Result<(), String> {
    let source = |selected: bool, child: &str| {
        format!(
            r#"
fn identity(x: u8) -> u8 {{ x }}
block clocked(clk, x: u8) -> u8 {{ reg q: u8; q := x; q }}
#[allow(combinational_block)]
block apply_child<F: fn(u8) -> u8>(x: u8) -> u8 {{ F(x) }}
#[allow(combinational_block)]
block parent<G: fn(u8) -> u8>(x: u8) -> u8 {{
    let y = const if {selected} {{ {child} }} else {{ x }};
    y
}}
pub block wrapper(x: u8) -> u8 {{ let p = parent<identity>(x); p }}
"#
        )
    };
    let instance_count = |selected, child: &str| -> Result<usize, String> {
        let package = compile_design(&source(selected, child))?;
        let wrapper = package.get_top_block().ok_or("missing wrapper")?;
        let [call] = wrapper.instantiations.as_slice() else {
            return Err("expected wrapper to instantiate parent".into());
        };
        let parent = package.get_block(&call.block).ok_or("missing parent")?;
        Ok(parent.instantiations.len())
    };
    assert_eq!(instance_count(true, "apply_child<identity>(x)")?, 1);
    for (child, diagnostic) in [
        ("clocked(x)", "clockless block cannot call clocked block"),
        ("apply_child(x)", "expects 1 callable arguments, got 0"),
        (
            "apply_child<identity, true>(x)",
            "expects 1 callable arguments, got 2",
        ),
        (
            "apply_child<u8:7>(x)",
            "callable argument must name a function",
        ),
        ("apply_child<x>(x)", "'x' names a local value here"),
        ("apply_child<G>(x)", "cannot forward callable parameter 'G'"),
    ] {
        assert_eq!(instance_count(false, child)?, 0, "{child}");
        let error = compile_design(&source(true, child)).unwrap_err();
        assert!(error.contains(diagnostic), "{child}: {error}");
    }
    for (child, name) in [
        ("clocked(missing_data)", "missing_data"),
        ("apply_child<missing_callback, true>(x)", "missing_callback"),
    ] {
        let error = compile_design(&source(false, child)).unwrap_err();
        assert!(error.contains(name), "{child}: {error}");
    }
    Ok(())
}
