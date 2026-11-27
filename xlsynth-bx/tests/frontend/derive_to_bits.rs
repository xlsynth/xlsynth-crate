// SPDX-License-Identifier: Apache-2.0

use std::error::Error;

use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::{ImportOptions, check_module, parse_module};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};

#[test]
fn derived_to_bits_packs_nested_fields_msb_first_in_constants_and_blocks()
-> Result<(), Box<dyn Error>> {
    let declarations = r#"
enum Tag: u2 { Start = 1, Stop = 2 }
#[derive(ToBits)]
struct Lane<N: u32> { sample: sN[N], tag: Tag }
#[derive(ToBits)]
struct Packet<N: u32> { valid: bool, lanes: Lane<N>[2], tail: ((), u3) }
const A = Lane<4> { sample: s4:-2, tag: Tag::Start };
const B = Lane<4> { sample: s4:3, tag: Tag::Stop };
const P = Packet<4> { valid: true, lanes: [A, B], tail: ((), u3:5) };
const_assert!(P.to_bits() == u16:0xf275);
"#;
    let params = "a: s4, b: s4";
    let body = r#"
        let first = Lane<4> { sample: a, tag: Tag::Start };
        let last = Lane<4> { sample: b, tag: Tag::Stop };
        Packet<4> { valid: true, lanes: [first, last], tail: ((), u3:5) }.to_bits()
    "#;
    let bx = format!("{declarations}\n{}", bx_source(params, "u16", body));
    let dslx = format!("{declarations}\n{}", dslx_source(params, "u16", body));
    let eval = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (a, b, expected) in [(-2, 3, 0xf275), (0, -1, 0x83f5)] {
        assert_eq!(
            eval.compare(&[IrValue::make_sbits(4, a)?, IrValue::make_sbits(4, b)?])?,
            IrValue::make_ubits(16, expected)?
        );
    }
    Ok(())
}

#[test]
fn derives_to_bits_after_bx_type_specialization_including_empty_layouts() -> Result<(), String> {
    let source = r#"
        struct Plain { data: u8 }
        #[derive(ToBits)] struct Empty {}
        #[derive(ToBits)] struct Item<T> { entry: T }
        #[derive(ToBits)] struct Unused { items: Plain[0] }
        const_assert!(Empty {}.to_bits() == uN[0]:0);
        const_assert!(zero!<Unused>().to_bits() == uN[0]:0);
        const_assert!(Item { entry: (uN[0]:0, true) }.to_bits() == u1:1);
        const_assert!(Item { entry: sN[129]:-1 }.to_bits() == all_ones!<uN[129]>());
    "#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn to_bits_field_conflicts_only_when_the_method_is_used() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
#[derive(ToBits)] struct Leaf { to_bits: u8 }
#[derive(ToBits)] struct Record<N:u32> { leaves: Leaf[N] }
fn field(x: Leaf) -> u8 { x.to_bits }
fn empty(x: Record<0>) -> uN[0] { x.to_bits() }
"#;
    check_module(&parse_module(declarations)?)?;
    parse_and_typecheck(declarations, "pack.x", "pack", &mut ImportData::default())?;
    for receiver in ["Leaf", "Record<1>"] {
        let source = format!("{declarations}\nfn pack(x: {receiver}) -> u8 {{ x.to_bits() }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("conflicts with the derived to_bits()"),
            "{error}"
        );
        assert!(
            parse_and_typecheck(&source, "pack.x", "pack", &mut ImportData::default()).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn named_map_can_be_the_first_use_of_a_derived_method_in_bx() -> Result<(), String> {
    let source = r#"
#[derive(ToBits)] struct Header { hi: u4, lo: u4 }
const H = Header { hi: u4:10, lo: u4:3 };
const_assert!(map([H], Header::to_bits) == [u8:0xa3]);
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn derived_array_packing_charges_each_projection_to_the_function() -> Result<(), String> {
    let decl = "#[derive(ToBits)] struct Word { entries: bool[300000] }";
    let source = format!("{decl} fn pack(x: Word) -> uN[300000] {{ x.to_bits() }}");
    check_module(&parse_module(&source)?)?;
    let source =
        format!("{decl} fn pack(x: Word, y: Word) -> uN[300000] {{ x.to_bits() ^ y.to_bits() }}");
    let error = check_module(&parse_module(&source)?).unwrap_err();
    assert!(error.contains("to_bits() exceeds the function"), "{error}");
    Ok(())
}

#[test]
fn packing_empty_nested_records_counts_the_expanded_tuple_layout() -> Result<(), String> {
    let mut declarations = String::from("#[derive(ToBits)] struct Z0 {}\n");
    for i in 0..17 {
        declarations.push_str(&format!(
            "#[derive(ToBits)] struct Z{} {{ left: Z{i}, right: Z{i} }}\n",
            i + 1
        ));
    }
    let source = format!("{declarations} fn f(x: Z17) -> uN[0] {{ x.to_bits() }}");
    check_module(&parse_module(&source)?)?;
    let source = format!(
        "{declarations} fn f(x: Z17) -> uN[0] {{ x.to_bits() ^ x.to_bits() ^ x.to_bits() }}"
    );
    let error = check_module(&parse_module(&source)?).unwrap_err();
    assert!(error.contains("to_bits() exceeds the function"), "{error}");
    Ok(())
}

#[test]
fn only_derived_structs_supply_the_receiver_method() -> Result<(), String> {
    let declarations = r#"
        struct Plain { to_bits: u8 }
        #[derive(ToBits)] struct Wrapped { payload: (bool, Plain[2]) }
        fn to_bits(p: Plain) -> u8 { p.to_bits }
        const_assert!(to_bits(Plain { to_bits: u8:5 }) == u8:5);
    "#;
    check_module(&parse_module(declarations)?)?;
    for (ty, diagnostic) in [
        ("u8", "struct receiver"),
        ("(u8,)", "struct receiver"),
        ("u8[1]", "struct receiver"),
        ("Plain", "does not derive ToBits"),
        ("Wrapped", "does not derive ToBits"),
    ] {
        let source = format!("{declarations}\nfn pack(x: {ty}) -> u8 {{ x.to_bits() }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains(diagnostic), "{ty}: {error}");
    }
    Ok(())
}
