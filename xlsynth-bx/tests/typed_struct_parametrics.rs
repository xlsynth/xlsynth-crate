// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs};
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_source_with_options, empty_package,
    lower_module, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;

/// Evaluate the specialized sample block with the given hardware inputs.
fn sample(
    source: &str,
    options: &ImportOptions<'_>,
    inputs: &[IrValue],
) -> Result<IrValue, Box<dyn Error>> {
    let checked = check_source_with_options(source, options)?;
    let package = lower_module(&checked, empty_package("design"), ResetBehavior::default())?;
    let function = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    Ok(eval_fn_in_package_value(&package, &function, inputs))
}

#[test]
fn signedness_flows_from_fields_through_nested_struct_and_function_arguments()
-> Result<(), Box<dyn Error>> {
    let source = r#"
struct Word<S: bool> { x: xN[S][8] }
struct Wrap<S: bool> { word: Word<S> }
fn sign<S: bool>(w: Wrap<S>) -> bool { S }
block sample(clk, a: s8, b: u8) -> (bool, bool, bool) {
    let signed = Wrap { word: Word { x: a } };
    let unsigned = Wrap { word: Word::<false> { x: b } };
    (sign(signed), sign(unsigned), signed.word.x < s8:0)
}
"#;
    let input = IrValue::make_ubits(8, 0xff)?;
    let output = sample(source, &ImportOptions::default(), &[input.clone(), input])?;
    assert_eq!(
        output,
        IrValue::parse_typed("(bits[1]:1, bits[1]:0, bits[1]:1)")?
    );
    Ok(())
}

#[test]
fn unused_signed_wide_and_enum_bindings_reach_the_specialized_function()
-> Result<(), Box<dyn Error>> {
    let source = r#"
enum Phase: u2 { Cold = 0, Hot = 2 }
struct Stamp<D: s8, K: uN[128], P: Phase> { data: u1 }
fn read<D: s8, K: uN[128], P: Phase>(s: Stamp<D, K, P>) -> (s8, uN[128], u2) {
    (D, K, P as u2)
}
block sample(clk) -> (s8, uN[128], u2) {
    read(Stamp<{s8:-3}, {uN[128]:0x10000000000000001}, Phase::Hot> { data: u1:0 })
}
"#;
    assert_eq!(
        sample(source, &ImportOptions::default(), &[])?,
        IrValue::parse_typed("(bits[8]:253, bits[128]:0x10000000000000001, bits[2]:2)")?
    );
    Ok(())
}

#[test]
fn aggregate_bindings_and_a_record_default_determine_field_layouts() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Config { width: u8 }
struct Bundle<P: (bool, u32), A: u8[2], C: Config = {Config { width: A[u32:1] }}> {
    x: xN[P.0][P.1],
    tag: uN[C.width as u32],
}
const SPEC = (true, u32:8);
const WIDTHS = [u8:1, u8:3];
block sample(clk, x: s8, y: u3) -> (bool, u3) {
    let a = Bundle<SPEC, WIDTHS> { x: x, tag: y };
    let b: Bundle<{(true, u32:8)}, {[u8:1, u8:3]}, {Config { width: u8:3 }}> = a;
    (b.x < s8:0, b.tag)
}
"#;
    let result = sample(
        source,
        &ImportOptions::default(),
        &[IrValue::make_ubits(8, 0x80)?, IrValue::make_ubits(3, 5)?],
    )?;
    assert_eq!(result, IrValue::parse_typed("(bits[1]:1, bits[3]:5)")?);
    Ok(())
}

#[test]
fn equal_layouts_keep_distinct_phantom_bindings() -> Result<(), String> {
    let declarations = "enum Phase: u2 { A = 1, B = 2 }\nstruct Config { x: u8 }";
    for (ty, left, right) in [
        ("uN[128]", "{uN[128]:1}", "{uN[128]:0x10000000000000001}"),
        ("s8", "{s8:-1}", "s8:1"),
        ("Phase", "Phase::A", "Phase::B"),
        ("(u8, bool)", "{(u8:0, true)}", "{(u8:0, false)}"),
        ("u8[2]", "{[u8:0, u8:1]}", "{[u8:1, u8:0]}"),
        ("Config", "{Config { x: u8:1 }}", "{Config { x: u8:2 }}"),
    ] {
        let base = format!("{declarations}\nstruct Phantom<K: {ty}> {{ x: u8 }}");
        let source = |target: &str| {
            format!("{base}\nfn convert(v: Phantom<{left}>) -> Phantom<{target}> {{ v }}")
        };
        check_module(&parse_module(&source(left))?)?;
        assert!(
            check_module(&parse_module(&source(right))?).is_err(),
            "{ty}"
        );
    }
    Ok(())
}

#[test]
fn a_bx_library_resolves_dependent_formals_and_defaults_at_their_declaration()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("lanes.bx"),
        r#"const WIDTH = u32:4;
const SIGN = true;
pub struct Lane<N: u32 = {WIDTH}, K: uN[N] = {uN[N]:3}, S: bool = {SIGN}> {
    x: xN[S][K as u32],
}"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let source = r#"
use lanes::Lane;
const WIDTH = u32:6;
const SIGN = false;
block sample(clk, a: s3, b: u17) -> (s3, u17) {
    let from_library = Lane<> { x: a };
    let from_caller = Lane<WIDTH, {uN[WIDTH]:17}, SIGN> { x: b };
    (from_library.x, from_caller.x)
}
"#;
    assert_eq!(
        sample(
            source,
            &options,
            &[IrValue::make_ubits(3, 7)?, IrValue::make_ubits(17, 65537)?],
        )?,
        IrValue::parse_typed("(bits[3]:7, bits[17]:65537)")?
    );
    Ok(())
}

#[test]
fn partial_signedness_argument_reaches_a_struct_update_loop_seed() -> Result<(), Box<dyn Error>> {
    for keyword in ["for", "unroll_for!"] {
        let source = format!(
            r#"
struct Pair<S: bool, N: u8> {{ a: xN[S][8], b: uN[N] }}
block sample(clk, b: u11) -> (bool, u11) {{
    let x = Pair<true> {{ ..{keyword} (_, p) in u32:0..u32:2 {{
        Pair {{ a: p.a - s8:1, ..p }}
    }}(Pair {{ a: 0, b }}) }};
    (x.a < s8:0, x.b)
}}
"#
        );
        let result = sample(
            &source,
            &ImportOptions::default(),
            &[IrValue::make_ubits(11, 42)?],
        )?;
        assert_eq!(result, IrValue::parse_typed("(bits[1]:1, bits[11]:42)")?);
    }
    Ok(())
}

#[test]
fn integer_dimensions_respect_the_declared_range_and_signedness() -> Result<(), String> {
    let declarations = "struct Lane<S: u8, N: s4> { x: xN[S][N] }";
    let source = format!(
        "{declarations}\nfn build(x: s7) -> Lane<1, 7> {{ let lane = Lane {{ x }}; lane }}"
    );
    check_module(&parse_module(&source)?)?;
    for (definition, reason) in [
        (
            "fn invalid(x: s8) -> s8 { let a = Lane { x }; a.x }",
            "does not fit s4",
        ),
        ("fn invalid(a: Lane<1, s4:-1>) -> u1 { u1:0 }", "negative"),
        ("fn invalid(a: Lane<u8:2, 1>) -> u1 { u1:0 }", "signedness"),
        (
            "fn invalid(a: Lane<u32:1, 1>) -> u1 { u1:0 }",
            "expected u8",
        ),
    ] {
        let source = format!("{declarations}\n{definition}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
    Ok(())
}

#[test]
fn recursive_struct_formal_types_report_the_cycle() -> Result<(), String> {
    let source = r#"
struct Cycle<V: Cycle<1>> { x: u8 }
block sample(clk, a: Cycle<1>) -> u8 { a.x }
"#;
    assert_eq!(
        check_module(&parse_module(source)?).unwrap_err(),
        "recursive struct dependency involving 'Cycle'"
    );
    Ok(())
}

#[test]
fn concrete_struct_value_literals_allow_a_forward_layout_dependency() -> Result<(), String> {
    for arg in ["8", "u32:8"] {
        let source = format!(
            r#"
struct Box<N: u32> {{ data: uN[N] }}
struct Frame {{ inner: Outer }}
const C = zero!<Frame>();
struct Outer {{ data: Box<{arg}> }}
fn frame() -> Frame {{ C }}
"#
        );
        check_module(&parse_module(&source)?)?;
    }
    // The literal's annotation introduces a dependency even though its
    // numeric value is known. Elaborate this field after WIDTH is available.
    let dependent = r#"
struct Box<N: u32> { data: uN[N] }
const WIDTH = u32:32;
struct Outer { data: Box<{uN[WIDTH]:8}> }
const C = zero!<Outer>();
"#;
    check_module(&parse_module(dependent)?)?;
    Ok(())
}

#[test]
fn tuple_array_lengths_constrain_the_type_of_inferred_bit_widths() -> Result<(), String> {
    for value in ["(a, b)", "pair"] {
        let source = format!(
            r#"
struct A<N: u8> {{ both: (uN[N], u8[N]) }}
fn make(a: u5, b: u8[5]) -> A<5> {{
    let pair = (a, b);
    let result = A {{ both: {value} }};
    result
}}
"#
        );
        assert!(check_module(&parse_module(&source)?).is_err());
        check_module(&parse_module(&source.replace("N: u8", "N: u32"))?)?;
        let nominal = format!("struct Inner<K: u8> {{ x: u8 }}\n{source}")
            .replace("both: (uN[N]", "both: (Inner<N>")
            .replace("a: u5", "a: Inner<5>");
        assert!(check_module(&parse_module(&nominal)?).is_err());
    }
    Ok(())
}

#[test]
fn record_formal_waits_for_the_declared_records_layout() -> Result<(), String> {
    let source = r#"
struct Config { inner: Inner }
struct Phantom<K: Config> { x: u8 }
struct Data<N: u32> { v: uN[N] }
struct Inner { data: Data<8> }
struct Wrapper { item: Phantom<{zero!<Config>()}> }
fn read(x: Wrapper) -> u8 { x.item.x }
"#;
    check_module(&parse_module(source)?)?;
    // A dimension or default can need the layout even when the formal itself
    // has a primitive type.
    for (formal, application) in [
        (
            "K: uN[(zero!<Config>().inner.data.v as u32) + u32:8]",
            "Phantom<1>",
        ),
        ("K: u8 = {zero!<Config>().inner.data.v}", "Phantom<>"),
    ] {
        let source = format!(
            r#"
struct Config {{ inner: Inner }}
struct Phantom<{formal}> {{ x: u8 }}
struct Wrapper {{ item: {application} }}
struct Data<N: u32> {{ v: uN[N] }}
struct Inner {{ data: Data<8> }}
fn read(x: Wrapper) -> u8 {{ x.item.x }}
"#
        );
        check_module(&parse_module(&source)?)?;
    }
    Ok(())
}

#[test]
fn only_selected_defaults_require_their_nested_layouts() -> Result<(), String> {
    let source = r#"
struct B<N: u8> { nested: X }
struct A<K: u8 = {zero!<B<1>>().nested.a.x}> { x: u8 }
struct UsesDefault { a: A<> }
struct Generic<B> { a: A<>, value: B }
struct Holder<V: A<1>> { x: u8 }
struct X { a: A<1>, h: Holder<{A { x: u8:1 }}> }
fn read(v: UsesDefault) -> u8 { v.a.x }
fn read_generic(v: Generic<u8>) -> u8 { v.a.x }
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn nested_explicit_record_parameters_reuse_their_checked_formal_type() -> Result<(), String> {
    let mut source = String::from("struct A0<K: u8> { x: u8 }\n");
    let mut ty = "A0<1>".to_owned();
    for depth in 1..=18 {
        let previous = depth - 1;
        source.push_str(&format!(
            "const C{previous} = zero!<{ty}>();\nstruct A{depth}<K: {ty}> {{ x: u8 }}\n"
        ));
        ty = format!("A{depth}<C{previous}>");
    }
    source.push_str(&format!("fn read(a: {ty}) -> u8 {{ a.x }}"));
    check_module(&parse_module(&source)?)?;
    Ok(())
}

#[test]
fn nested_record_defaults_reuse_the_specialization_for_each_explicit_prefix() -> Result<(), String>
{
    let mut source = String::from("struct L0<N: u8, V: u8 = {N}> { x: u1 }\n");
    for depth in 1..=16 {
        let previous = depth - 1;
        source.push_str(&format!(
            "struct L{depth}<N: u8, V: L{previous}<N> = {{zero!<L{previous}<N>>()}}> {{ x: u1 }}\n"
        ));
    }
    source.push_str("fn read(x: L16<1>, y: L16<2>) -> (u1, u1) { (x.x, y.x) }");
    check_module(&parse_module(&source)?)?;
    let invalid = format!("{source}\nfn mix(x: L16<1>) -> L16<2> {{ x }}");
    assert!(check_module(&parse_module(&invalid)?).is_err());
    Ok(())
}

#[test]
fn signed_struct_arguments_preserve_literal_radix() -> Result<(), String> {
    let source = r#"
struct Stamp<K: s8> { data: u8 }
fn normalize(x: Stamp<0xff>) -> Stamp<s8:-1> { x }
fn make(x: u8) -> Stamp<{0xff}> { Stamp<0b1111_1111> { data: x } }
"#;
    check_module(&parse_module(source)?)?;
    for literal in ["255", "0x1ff"] {
        let invalid = source.replace("Stamp<0xff>", &format!("Stamp<{literal}>"));
        assert!(check_module(&parse_module(&invalid)?).is_err(), "{literal}");
    }
    Ok(())
}

#[test]
fn bool_and_u1_nominal_witnesses_agree_for_the_same_value() -> Result<(), String> {
    let source = r#"
struct Flag<X: bool> { data: u8 }
struct Bit<X: u1> { data: u8 }
struct Pair<X: bool> { values: (Flag<X>, Bit<X>) }
fn combine(a: Flag<true>, b: Bit<u1:1>) -> Pair<true> {
    let pair = Pair { values: (a, b) };
    pair
}
"#;
    check_module(&parse_module(source)?)?;
    let different = source.replace("b: Bit<u1:1>", "b: Bit<u1:0>");
    assert!(check_module(&parse_module(&different)?).is_err());
    Ok(())
}

#[test]
fn nested_struct_formals_receive_representable_scalar_values() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Inner<K: u8> { data: u8 }
struct Outer<N: u32> { word: uN[N], child: Inner<N> }
struct Signed<K: sN[96]> { data: u8 }
struct Widen<S: s8> { child: Signed<S> }
fn key<K: u8>(child: Inner<K>) -> u8 { K }
fn signed_key<K: sN[96]>(child: Signed<K>) -> sN[96] { K }
block sample(clk, x: u3) -> (u8, sN[96]) {
    let child = Inner<u8:3> { data: u8:0 };
    let outer = Outer { word: x, child };
    let wide = Widen<s8:-3> { child: Signed { data: u8:0 } };
    (key(outer.child), signed_key(wide.child))
}
"#;
    assert_eq!(
        sample(
            source,
            &ImportOptions::default(),
            &[IrValue::make_ubits(3, 0)?]
        )?,
        IrValue::make_tuple(&[IrValue::make_ubits(8, 3)?, IrValue::make_sbits(96, -3)?])
    );
    for (source_ty, target_ty, value) in [
        ("s8", "u8", "-1"),
        ("u8", "s8", "128"),
        ("u64", "s3", "4"),
        ("uN[65]", "s3", "4"),
        ("u8", "bool", "2"),
        ("uN[128]", "u64", "0x10000000000000000"),
    ] {
        let source = format!(
            "struct Inner<K:{target_ty}> {{ data:u8 }}
             struct Outer<N:{source_ty}> {{ child:Inner<N> }}
             fn f(x:Outer<{source_ty}:{value}>) -> u8 {{ x.child.data }}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    let function = r#"
struct Inner<K: u8> { data: u8 }
fn make<N: u32>() -> Inner<N> { Inner<N> { data: u8:0 } }
block sample(clk) -> u8 { make<u32:3>().data }
"#;
    assert!(check_module(&parse_module(function)?).is_err());
    let nominal_witness = r#"
struct Inner<K: u8> { data: u8 }
struct Outer<N: u32> { child: Inner<N> }
fn wrap(child: Inner<3>) -> u8 { let x = Outer { child }; x.child.data }
"#;
    assert!(check_module(&parse_module(nominal_witness)?).is_err());
    Ok(())
}
