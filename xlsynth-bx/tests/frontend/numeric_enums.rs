// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::BlockEvaluator;
use std::error::Error;
use xlsynth_bx::ast::Type;
use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, parse_module};
use xlsynth_pir::IrValue;

#[test]
fn resolved_enum_annotations_must_match_the_declaration() -> Result<(), Box<dyn Error>> {
    let source = parse_module("enum E:u3 { A=0 } struct P { e:E } fn id(x:E)->E { x }")?;
    for (name, signed, width, valid) in [
        ("E", false, 3, true),
        ("E", false, 4, false),
        ("E", true, 3, false),
        ("Other", false, 3, false),
    ] {
        let resolved = Type::Enum {
            name: name.to_owned(),
            signed,
            width,
        };
        let mut function = source.clone();
        function.functions[0].params[0].ty = resolved.clone();
        let mut structure = source.clone();
        structure.structs[0].fields[0].ty = resolved;
        for module in [function, structure] {
            let result = check_module(&module);
            if valid {
                result?;
            } else {
                assert!(result.unwrap_err().contains("no matching declaration"));
            }
        }
    }
    Ok(())
}

#[test]
fn enum_discriminants_share_inference_and_earlier_constexpr_functions() -> Result<(), Box<dyn Error>>
{
    let declarations = r#"
const BASE = s8:-3;
fn next(x:s8)->s8 { x + s8:1 }
enum State { Idle=0, Start=BASE, Run=next(BASE), Again=next(BASE) }
struct Packet { state:State, valid:bool }
const INITIAL = Packet { state: State::Run, valid: true };
fn packed(x: State)->(s16,bool) {
    let pkt = Packet { state:x, valid:INITIAL.valid };
    (pkt.state as s16, pkt.state == INITIAL.state)
}
"#;
    let bx = format!("{declarations}\nblock sample(clk, x:State)->(s16,bool) {{ packed(x) }}");
    let native = format!("{declarations}\nfn sample(x:State)->(s16,bool) {{ packed(x) }}");
    let checked = check_module(&parse_module(&bx)?)?;
    let state = &checked.enums()["State"];
    assert!(state.signed);
    assert_eq!(state.width, 8);
    assert_eq!(state.members["Run"], state.members["Again"]);
    let reference = compile_dslx_module(&native, &[])?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    for (input, expected) in [
        (0, "(bits[16]:0,bits[1]:0)"),
        (253, "(bits[16]:65533,bits[1]:0)"),
        (254, "(bits[16]:65534,bits[1]:1)"),
    ] {
        assert_eq!(
            evaluator.compare_ir(&reference, &[IrValue::make_ubits(8, input)?])?,
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn enum_match_selects_a_constant_width_slice_dimension() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
enum Size:s2 { Small=-1, Large=1 }
fn narrow(x:u8)->u2 {
    x[0+:uN[match Size::Small { Size::Small => u32:2, Size::Large => u32:4 }]]
}
"#;
    let bx = format!("{declarations}\nblock sample(clk, x:u8)->u2 {{ narrow(x) }}");
    let native = format!("{declarations}\nfn sample(x:u8)->u2 {{ narrow(x) }}");
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let reference = compile_dslx_module(&native, &[])?;
    assert_eq!(
        evaluator.compare_ir(&reference, &[IrValue::make_ubits(8, 0xad)?])?,
        IrValue::make_ubits(2, 1)?
    );
    Ok(())
}

#[test]
fn enum_register_ports_and_generic_block_reset_use_scalar_layouts() -> Result<(), Box<dyn Error>> {
    let bx = r#"
enum State:u3 { Idle=0, Busy=5 }
block delay<T>(clk, rst, x:T)->T {
    reg q:T = zero!<T>();
    q := x;
    q
}
block sample(clk, rst, run:bool)->State {
    let next = if run { State::Busy } else { State::Idle };
    let old = delay(next);
    old
}
"#;
    let package = compile(bx, empty_package("state_machine"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let sample = package.get_block("sample").ok_or("missing sample")?;
    assert_eq!(sample.output_ports().count(), 1);
    let state = package
        .members
        .iter()
        .find_map(|m| match m {
            xlsynth_pir::ir::PackageMember::Block(b) => b.registers.first(),
            _ => None,
        })
        .ok_or("missing state register")?;
    assert_eq!(state.reset_value, Some(IrValue::make_ubits(3, 0)?));
    Ok(())
}

#[test]
fn constant_enum_casts_require_a_declared_bit_pattern_at_the_declared_width()
-> Result<(), Box<dyn Error>> {
    let declarations = "enum State:u3 { Idle=1, Busy=5 } fn decode(x:u3)->State { x as State }";
    for init in [
        "decode(u3:2)",
        "State::Idle as State",
        "u2:1 as State",
        "zero!<State>()",
        "zero!<State[2]>()",
    ] {
        let source = format!("{declarations} const BAD = {init};");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    let good = format!("{declarations} const GOOD = decode(u3:5);");
    let checked = check_module(&parse_module(&good)?)?;
    assert_eq!(checked.constants()[0].value, IrValue::make_ubits(3, 5)?);
    Ok(())
}

#[test]
fn enum_layouts_respect_declaration_order_and_storage_bounds() -> Result<(), Box<dyn Error>> {
    let later = "struct Packet { state:State } enum State:u3 { Idle=0 }";
    let error = check_module(&parse_module(later)?).unwrap_err();
    assert!(error.contains("declared after struct 'Packet'"), "{error}");

    // Each uN[1048576] member owns 16,384 limbs. Reject the full enum before
    // materializing its discriminants.
    let members = (0..65)
        .map(|i| format!("V{i}={i}"))
        .collect::<Vec<_>>()
        .join(",");
    let error = check_module(&parse_module(&format!(
        "enum Wide:uN[1048576] {{ {members} }}"
    ))?)
    .unwrap_err();
    assert!(
        error.contains("enum representation exceeds the size limit"),
        "{error}"
    );
    Ok(())
}
