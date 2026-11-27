// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs, path::Path};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, compile, compile_with_options, empty_package, run_tests,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_eval::eval_fn_in_package_value;

fn compile_design(source: &str) -> Result<Package, String> {
    compile(source, empty_package("design"), ResetBehavior::default())
}

/// Run the single embedded BX test and surface any failed assertion.
fn check_block_test(source: &str) -> Result<(), String> {
    let report = run_tests(
        source,
        Path::new("const_match.bx"),
        empty_package("const_match"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn specializations_lower_only_the_selected_arm() -> Result<(), Box<dyn Error>> {
    let pkg = compile_design(
        r#"
        fn extend<N:u32>(x:u8) -> uN[N] {
            const match N { 8 => x, 16 => x ++ x, 24 => x ++ x ++ x }
        }
        fn sample(x:u8) -> (u24, u8, u16) { (extend<24>(x), extend<8>(x), extend<16>(x)) }
        "#,
    )?;
    let sample = pkg.get_fn("sample").ok_or("missing sample")?;
    assert_eq!(
        eval_fn_in_package_value(&pkg, sample, &[IrValue::make_ubits(8, 0xc2)?]),
        IrValue::parse_typed("(bits[24]:0xc2c2c2, bits[8]:0xc2, bits[16]:0xc2c2)")?
    );
    for node in &sample.nodes {
        if let NodePayload::Invoke { to_apply, .. } = &node.payload {
            let function = pkg.get_fn(to_apply).ok_or("missing specialization")?;
            assert!(!function.nodes.iter().any(|n| matches!(
                n.payload,
                NodePayload::Sel { .. } | NodePayload::PrioritySel { .. }
            )));
        }
    }
    Ok(())
}

#[test]
fn selected_tuple_fields_are_constants_in_values_and_types() -> Result<(), Box<dyn Error>> {
    let pkg = compile_design(
        r#"
        const CONFIG = (true, u8:3, u32:9);
        const OFFSET = const match CONFIG { (.., n) => n + u32:1 };
        fn sample(x:u9) -> (u9, u32) {
            const match CONFIG {
                (true, shift, width) => {
                    type Word = uN[width];
                    let y: Word = x << shift;
                    (y, const match u32:9 { width => OFFSET, _ => u32:0 })
                },
                _ => { const_assert!(false); x },
            }
        }
        "#,
    )?;
    assert_eq!(
        eval_fn_in_package_value(
            &pkg,
            pkg.get_fn("sample").ok_or("missing sample")?,
            &[IrValue::make_ubits(9, 27)?]
        ),
        IrValue::parse_typed("(bits[9]:216, bits[32]:10)")?
    );
    Ok(())
}

#[test]
fn selected_literals_receive_context_from_function_formals_and_loop_seeds()
-> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("word.x"),
        "pub fn last<N:u32>(a:uN[N], b:uN[N])->uN[N] { b }",
    )?;
    let pkg = compile_with_options(
        r#"
        use word;
        fn id<T:type>(x:T) -> T { x }
        fn sample() -> (u16, u16, u16, u16) {
            let narrow = id(const match u32:3 { 3 => 255 + 1, _ => false });
            let wide:u16 = id(const match u32:3 { 3 => 255 + 1, _ => false });
            let imported = word::last(u16:0, const match u32:3 { 3 => 255 + 1, _ => false });
            let folded:u16 = for (_, acc) in u32:0..u32:1 { acc }(
                const match u32:3 { 3 => 255 + 1, _ => false });
            (narrow as u16, wide, imported, folded)
        }
        "#,
        empty_package("design"),
        ResetBehavior::default(),
        &support::search_paths(&[dir.path()]),
    )?;
    assert_eq!(
        eval_fn_in_package_value(&pkg, pkg.get_fn("sample").ok_or("missing sample")?, &[]),
        IrValue::parse_typed("(bits[16]:0, bits[16]:256, bits[16]:256, bits[16]:256)")?
    );
    Ok(())
}

#[test]
fn checked_arms_and_patterns_have_distinct_requirements() -> Result<(), String> {
    for header in [
        "fn sample(x:u8) -> u8",
        "#[allow(combinational_block)] block sample(x:u8) -> u8",
    ] {
        for (expr, expected) in [
            (
                "const match x { _ => x }",
                "subject must evaluate to a compile-time constant",
            ),
            (
                "const match u8:1 { 0 => x }",
                "no pattern matches the subject value",
            ),
            (
                "const match u8:0 { 0 => x, 1 => missing }",
                "unknown value 'missing'",
            ),
            ("const match u8:0 { 0 => x, u32:1 => x }", "u32"),
            ("const match u8:0 { 0 => x + false, _ => x }", "bool"),
            (
                "const match (true, x) { (true, _) => x }",
                "compile-time constant",
            ),
        ] {
            let error = compile_design(&format!("{header} {{ {expr} }}")).unwrap_err();
            assert!(error.contains(expected), "{expr}: {error}");
        }
        compile_design(&format!(
            "{header} {{ const match true {{ true => x, false => x + false }} }}"
        ))?;
    }
    Ok(())
}

#[test]
fn named_comparison_patterns_follow_bx_binding_rules() -> Result<(), String> {
    let src = r#"
        fn compare(x:u8) -> u8 {
            const key = u2:1;
            const match u2:2 { key => x, _ => !x }
        }
    "#;
    compile_design(src)?;
    assert!(compile_design(&src.replace("const key", "let key")).is_err());
    Ok(())
}

#[test]
fn scoped_subject_requires_its_own_runtime_dependencies() -> Result<(), String> {
    for header in [
        "fn apply(x: u8) -> u8",
        "#[allow(combinational_block)] block apply(x: u8) -> u8",
    ] {
        let arms = "{ true => x, _ => !x }";
        let error = compile_design(&format!(
            "{header} {{ const match {{ let _ = x; true }} {arms} }}"
        ))
        .unwrap_err();
        assert!(error.contains("compile-time constant"), "{error}");
        let tuple_subject = "{ let (_, b) = (if true { u8:3 } else { x }, true); b }";
        let error = compile_design(&format!(
            "{header} {{ const match {tuple_subject} {arms} }}"
        ))
        .unwrap_err();
        assert!(error.contains("compile-time constant"), "{error}");
        compile_design(&format!(
            "{header} {{ const match {} {arms} }}",
            tuple_subject.replace("else { x }", "else { u8:0 }")
        ))?;
        compile_design(&format!(
            "{header} {{ const match {{ let _ = u8:4; true }} {arms} }}"
        ))?;
        compile_design(&format!(
            "{header} {{ let b = {{ let _ = x; true }}; const match b {arms} }}"
        ))?;
        compile_design(&format!(
            "{header} {{ let (_, b) = (if true {{ u8:3 }} else {{ x }}, true); const match b {arms} }}"
        ))?;
        // A nested const if has already discarded its inactive arm when it
        // supplies the subject. Pinned DSLX loses the inner predicate's value.
        compile_design(&format!(
            "{header} {{ const match const if true {{ true }} else {{ x == u8:0 }} {arms} }}"
        ))?;
    }
    Ok(())
}

#[test]
fn block_modes_select_live_input_or_stored_value() -> Result<(), String> {
    let source = r#"
        block stage<MODE:u32>(clk, x:u8) -> u8 {
            reg q:u8 = const match MODE { 0 => 0, _ => 9 };
            q := x;
            const match MODE { 0 => x, _ => q }
        }
        #[test] block outputs() {
            reg x:u8;
            let bypass = stage<0>(x);
            let delayed = stage<1>(x);
            tick!(rst: true);
            x := u8:23;
            assert_eq(bypass, u8:23);
            assert_eq(delayed, u8:9);
            tick!();
            assert_eq(delayed, u8:23);
        }
    "#;
    check_block_test(source)
}

#[test]
fn selected_block_arm_keeps_dslx_literal_inference() -> Result<(), String> {
    let source = r#"
        #[allow(combinational_block)]
        block select_bits<NEG:bool>(clk) -> (s8, s3, u16, (s8, u16), (u8, u8)) {
            let mask = 0xff;
            let selected:s8 = const match NEG { true => s8:-129, false => mask };
            let widened = u16:1 + (const match true { _ => 255 + 1 });
            let pair: (s8, u16) = const match NEG { true => (mask, 255 + 1), _ => (0xfe, 257) };
            let rows = [const match NEG { true => (1, u8:2), _ => (4, u8:5) }, (u8:3, 1)];
            (selected, const match true { _ => s3:-5 }, widened, pair, rows[u32:0])
        }
        #[test] block values() {
            let a = select_bits<true>();
            let b = select_bits<false>();
            assert_eq(a, (s8:127, s3:3, u16:257, (s8:-1, u16:256), (u8:1, u8:2)));
            assert_eq(b, (s8:-1, s3:3, u16:257, (s8:-2, u16:257), (u8:4, u8:5)));
        }
    "#;
    check_block_test(source)?;
    let outside = r#"
        #[allow(combinational_block)] block pair() -> (s3, s3) {
            (const match true { _ => s3:-5 }, s3:-5)
        }
    "#;
    assert!(compile_design(outside).is_err());
    Ok(())
}

#[test]
fn selected_fill_row_infers_its_shape_from_a_sibling() -> Result<(), String> {
    let source = r#"
        #[allow(combinational_block)] block sample() -> u2[2][2] {
            let rows = [const match true { _ => [1, ...] }, [2, 3]];
            rows
        }
        #[allow(combinational_block)] block conditional(x: bool) -> u2[2][2] {
            let rows = [const match true { _ => if x { [1, ...] } else { [0, ...] } }, [2, 3]];
            rows
        }
        #[allow(combinational_block)] block matched(x: bool) -> u2[2][2] {
            let rows = [const match true { _ => match x { true => [1, ...], _ => [0, ...] } }, [2, 3]];
            rows
        }
        #[allow(combinational_block)] block folded() -> u2[2][2] {
            let rows = [const match true { _ => unroll_for! (_, acc) in u32:0..u32:2 { acc }([1, ...]) }, [2, 3]];
            rows
        }
        #[test] block check_rows() {
            let rows = sample();
            let ones = conditional(true);
            let zeros = conditional(false);
            let matched_zeros = matched(false);
            let folded_ones = folded();
            assert_eq(rows, [[u2:1, u2:1], [u2:2, u2:3]]);
            assert_eq(ones, rows);
            assert_eq(zeros, [[u2:0, u2:0], [u2:2, u2:3]]);
            assert_eq(matched_zeros, zeros);
            assert_eq(folded_ones, ones);
        }
    "#;
    check_block_test(source)?;
    // The block test harness can simulate the unrolled loop. Check the
    // corresponding rolled loop's inferred port shape at compilation.
    compile_design(&source.replace("unroll_for!", "for"))?;
    Ok(())
}

#[test]
fn selected_struct_infers_its_parametric_width_from_a_sibling() -> Result<(), String> {
    let source = r#"
        struct Word<N:u32> { data:uN[N] }
        #[allow(combinational_block)] block sample(x:u8) -> u8[3] {
            let words = [const match true { _ => Word { data:1 } }, Word { data:x }];
            let choices = [const match true {
                _ => if x == u8:0 { Word { data:2 } } else { Word { data:3 } }
            }, Word { data:x }];
            [words[u32:0].data, words[u32:1].data, choices[u32:0].data]
        }
        #[test] block check_words() {
            let data = sample(u8:23);
            assert_eq(data, [u8:1, u8:23, u8:3]);
        }
    "#;
    check_block_test(source)
}

#[test]
fn selected_tuple_alias_infers_signed_members_in_either_order() -> Result<(), String> {
    let source = r#"
        #[allow(combinational_block)] block sample() -> ((s8,)[2], (s8,)[2]) {
            let t = (0xff,);
            let first = [const match true { _ => t }, (s8:0,)];
            let last = [(s8:0,), const match true { _ => t }];
            (first, last)
        }
        #[test] block check_alias() {
            let rows = sample();
            assert_eq(rows, ([(s8:-1,), (s8:0,)], [(s8:0,), (s8:-1,)]));
        }
    "#;
    check_block_test(source)
}

#[test]
fn structural_match_wires_the_first_matching_child_or_bypasses_it() -> Result<(), String> {
    let source = r#"
        block delay(clk, x:u8) -> u8 { reg q:u8; q := x; q }
        block invert_delay(clk, x:u8) -> u8 { reg q:u8; q := !x; q }
        #[allow(combinational_block)] block stage<MODE:u32>(clk, x:u8) -> u8 {
            let result:u8 = const match (MODE, MODE < u32:2) {
                (0, _) => delay(x),
                (_, true) => invert_delay(x),
                _ => x,
            };
            result
        }
        pub block lanes(clk, x:u8) -> (u8, u8, u8) {
            let plain = stage<0>(x);
            let inverted = stage<1>(x);
            let bypassed = stage<2>(x);
            (plain, inverted, bypassed)
        }
    "#;
    let package = compile_design(source)?;
    let lanes = package.get_block("lanes").ok_or("missing lanes")?;
    assert_eq!(lanes.instantiations.len(), 3);
    for (lane, expected) in
        lanes
            .instantiations
            .iter()
            .zip([Some("delay"), Some("invert_delay"), None])
    {
        let stage = package
            .get_block(&lane.block)
            .ok_or("missing stage specialization")?;
        let input = stage.get_input_port("x").ok_or("missing input")?;
        let output = stage.output_value(stage.get_output_port("out").ok_or("missing output")?);
        assert!(stage.nodes.iter().all(|node| !matches!(
            node.payload,
            NodePayload::Sel { .. } | NodePayload::PrioritySel { .. }
        )));
        let Some(expected) = expected else {
            assert!(stage.instantiations.is_empty());
            assert_eq!(output, input);
            continue;
        };
        let [child] = stage.instantiations.as_slice() else {
            return Err(format!("expected one child in {}", lane.name));
        };
        assert_eq!(
            (child.name.as_str(), child.block.as_str()),
            ("result", expected)
        );
        assert!(stage.nodes.iter().any(|node| matches!(
            &node.payload, NodePayload::InstantiationInput { instantiation, port_name, arg }
            if instantiation == "result" && port_name == "x" && *arg == input
        )));
        assert!(matches!(
            &stage.get_node(output).payload, NodePayload::InstantiationOutput { instantiation, port_name }
            if instantiation == "result" && port_name == "out"
        ));
    }
    Ok(())
}

#[test]
fn selected_child_ports_keep_match_arm_literal_inference() -> Result<(), String> {
    let source = r#"
        #[allow(combinational_block)] block child(x:s3, y:s8) -> (s3, s8) { (x, y) }
        #[allow(combinational_block)] block sized<N:u32>(x:sN[N], y:s8) -> (sN[N], s8) { (x, y) }
        block parent(clk) -> ((s3, s8), (s3, s8)) {
            let mask = 0xff;
            let a = const match true { _ => child(s3:-5, mask) };
            let b = const match (true, 0xff) {
                (true, m) => sized<3>(s3:-5, m),
                _ => (s3:0, s8:0),
            };
            (a, b)
        }
        #[test] block values() {
            let both = parent();
            assert_eq(both, ((s3:3, s8:-1), (s3:3, s8:-1)));
        }
    "#;
    check_block_test(source)
}

#[test]
fn selected_child_ports_check_the_signed_width_of_literal_groups() -> Result<(), String> {
    for (params, element, args) in [("", "s8", ""), ("<N:u32>", "sN[N]", "<8>")] {
        for (shape, too_wide, fits) in [
            ("", "0xff + 0xfe", "0x7f + 0x7e"),
            ("[2]", "[0xff, 0xfe]", "[0x7f, 0x7e]"),
        ] {
            let source = format!(
                r#"
                #[allow(combinational_block)]
                block child{params}(x:{element}{shape}) -> {element}{shape} {{ x }}
                block parent(clk) -> s8{shape} {{
                    let y = const match true {{ _ => child{args}({too_wide}) }};
                    y
                }}
                "#
            );
            let error = compile_design(&source).unwrap_err();
            assert!(
                error.contains("require 9 signed bits, got s8"),
                "child{args}({too_wide}): {error}"
            );
            compile_design(&source.replace(too_wide, fits))?;
        }
    }
    Ok(())
}

#[test]
fn selected_child_ports_infer_range_bounds_before_applying_the_port_type() -> Result<(), String> {
    for (params, len, args) in [("", "3", ""), ("<N:u32>", "N", "<3>")] {
        let row = format!("u8[{len}]");
        for (port, argument, output) in [
            (row.clone(), "2..(2 + 3)", "x"),
            (format!("(bool, {row})"), "(true, 2..(2 + 3))", "x.1"),
        ] {
            let source = format!(
                r#"
                #[allow(combinational_block)]
                block child{params}(x:{port}) -> {row} {{ {output} }}
                block parent(clk) -> u8[3] {{
                    let y = const match true {{ _ => child{args}({argument}) }};
                    y
                }}
                #[test] block values() {{
                    let row = parent();
                    assert_eq(row, [u8:2, u8:3, u8:4]);
                }}
                "#
            );
            let error = compile_design(&source).unwrap_err();
            assert!(
                error.contains("array range limit precedes its start"),
                "child{args}({argument}): {error}"
            );
            check_block_test(&source.replace("2 + 3", "2 + u8:3"))?;
        }
    }
    Ok(())
}

#[test]
fn nested_selectors_supply_pattern_constants_to_a_destructured_child() -> Result<(), String> {
    let source = r#"
        block stage<BIAS:u32>(clk, x:u8) -> (u8, u8) {
            reg held:u8 = 42;
            held := x + (BIAS as u8);
            (held, x)
        }
        block parent(clk, x:u8) -> (u8, u8) {
            let (previous, live): (u8, u8) = const match (true, u32:5) {
                (enabled, bias) => const if enabled {
                    const match (bias, u8:3) {
                        (5, offset) => stage<bias>(x + offset),
                        _ => (x, x),
                    }
                } else { (x, x) },
            };
            (previous, live)
        }
        #[test] block outputs() {
            reg x:u8;
            let pair = parent(x);
            tick!(rst: true);
            x := u8:10;
            assert_eq(pair, (u8:42, u8:13));
            tick!();
            x := u8:20;
            assert_eq(pair, (u8:18, u8:23));
        }
    "#;
    let package = compile_design(source)?;
    assert_eq!(
        package
            .get_block("parent")
            .ok_or("missing parent")?
            .instantiations
            .len(),
        1
    );
    check_block_test(source)
}

#[test]
fn structural_unrolls_instantiate_only_matching_iterations() -> Result<(), String> {
    let source = r#"
        block delay(clk, x:u8) -> u8 { reg q:u8; q := x; q }
        block split(clk, x:u8) -> (u8, u8) { reg q:u8; q := x; (q, x) }
        pub block lanes(clk, x:u8) -> (u8, u8[3]) {
            let held = unroll_for! (i, acc) in u32:0..u32:3 {
                let lane = const match i { 1 => delay(acc), _ => acc };
                let (first, _) = const match i { 2 => split(lane), _ => (lane, lane) };
                first
            }(x);
            let mapped = unroll_map! i in u32:0..u32:3 {
                const match i { 0 | 2 => delay(x), _ => x }
            };
            (held, mapped)
        }
    "#;
    let package = compile_design(source)?;
    let children = &package
        .get_block("lanes")
        .ok_or("missing lanes")?
        .instantiations;
    assert_eq!(
        children
            .iter()
            .map(|c| c.block.as_str())
            .collect::<Vec<_>>(),
        ["delay", "split", "delay", "delay"]
    );
    assert_eq!(children[0].name, "lane_1");
    Ok(())
}

#[test]
fn skipped_child_calls_resolve_names_and_defer_signature_checks() -> Result<(), String> {
    let source = r#"
        block delay<INIT:u32>(clk, x:u8) -> u8 { reg q:u8 = INIT as u8; q := x; q }
        #[allow(combinational_block)] block parent(x:u8) -> u8 {
            let result = const match false { true => delay<9>(x), false => x };
            result
        }
    "#;
    compile_design(source)?;
    let selected = source.replace("const match false", "const match true");
    let error = compile_design(&selected).unwrap_err();
    assert!(
        error.contains("clockless block cannot call clocked block"),
        "{error}"
    );
    compile_design(&source.replace("delay<9>(x)", "delay<9>(x, false)"))?;
    for child in ["missing(x)", "delay<missing>(x)", "delay<9>(missing)"] {
        let error = compile_design(&source.replace("delay<9>(x)", child)).unwrap_err();
        assert!(error.contains("missing"), "{child}: {error}");
    }
    Ok(())
}

#[test]
fn map_result_preserves_match_arm_literal_inference() -> Result<(), String> {
    let source = r#"
        #[allow(combinational_block)] block sample() -> s8[2][1] {
            let y = unroll_map! _ in u32:0..u32:1 {
                const match true { _ => [{ let m = 0xff; m }, s8:0] }
            };
            y
        }
        #[test] block values() {
            let y = sample();
            assert_eq(y, [[s8:-1, s8:0]]);
        }
    "#;
    check_block_test(source)?;
    let overflow = source.replace("{ let m = 0xff; m }", "0xff + 0xfe");
    let error = compile_design(&overflow).unwrap_err();
    assert!(error.contains("require 9 signed bits"), "{error}");
    Ok(())
}

#[test]
fn selected_child_calls_require_a_structural_site() {
    let delay = "block delay(clk, x:u8) -> u8 { reg q:u8; q := x; q }";
    for body in [
        "const match true { _ => delay(x) }",
        "if en { let y = const match true { _ => delay(x) }; } x",
        "let y = if en { const match true { _ => delay(x) } } else { x }; y",
        "let y = match en { true => delay(x), _ => x }; y",
        "let y = const match true { _ => delay(x) + x }; y",
        "let y = const match true { _ => delay(delay(x)) }; y",
        "let y = { let a = x; const match true { _ => delay(a) } }; y",
        "let y = const match true { _ => { let a = x; delay(a) } }; y",
        "let y = const match true { _ => unroll_map! a in [x] { delay(a) } }; y[0]",
        "unroll_for! (_, a) in u32:0..u32:1 { const match true { _ => delay(a) } }(x)",
    ] {
        let source = format!("{delay} pub block parent(clk, x:u8, en:bool) -> u8 {{ {body} }}");
        let error = compile_design(&source).unwrap_err();
        assert!(error.contains("call to block 'delay'"), "{body}: {error}");
    }
    let source =
        format!("{delay} fn f(x:u8) -> u8 {{ let y = const match true {{ _ => delay(x) }}; y }}");
    let error = compile_design(&source).unwrap_err();
    assert!(
        error.contains("pure function cannot call block 'delay'"),
        "{error}"
    );
}
