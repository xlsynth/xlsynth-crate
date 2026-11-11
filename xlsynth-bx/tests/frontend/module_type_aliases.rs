// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_source_with_options, compile, empty_package,
    parse_module,
};

use super::directed_expr_eval::compile_dslx_module;

#[test]
fn aliased_registers_share_specializations_and_keep_named_output_ports()
-> Result<(), Box<dyn Error>> {
    let source = r#"
const WIDTH = u32:8;
type Lane = uN[WIDTH];
struct Packet { valid: bool, data: Lane }
type Payload = Packet;
block hold<T>(clk, rst, x: T) -> T {
    reg q: T = zero!<T>();
    q := x;
    q
}
pub block packet_pipe(clk, rst, a: Payload, b: Packet) -> Output {
    let a_q = hold(a);
    let b_q = hold(b);
    Output { valid: a_q.valid, data: b_q.data }
}
type Output = Payload;
"#;
    let package = compile(source, empty_package("aliases"), ResetBehavior::default())?;
    let block = package.get_top_block().expect("top block");
    let [a, b] = block.instantiations.as_slice() else {
        panic!("expected two register instances");
    };
    assert_eq!(a.block, b.block);
    let ports: Vec<_> = block
        .output_ports()
        .map(|port| (block.port_name(port), block.port_type(port).to_string()))
        .collect();
    assert_eq!(
        ports,
        [
            ("valid", "bits[1]".to_owned()),
            ("data", "bits[8]".to_owned())
        ]
    );
    Ok(())
}

#[test]
fn value_formals_shadow_type_aliases_in_their_own_annotations() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("wire.x"), "pub type Byte = u8;")?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    for (formal, accepted) in [
        ("Word: Word", false),
        ("Word: Word[2]", false),
        ("wire: wire::Byte", false),
        ("N: Word", true),
        ("N: wire::Byte", true),
    ] {
        for declaration in [
            format!("fn hold<{formal}>(x:u8)->u8 {{x}}"),
            format!("struct Hold<{formal}> {{x:u8}}"),
        ] {
            let body = format!("type Word = u8; {declaration} fn sample(x:u8)->u8 {{x}}");
            let native = format!("import wire; {body}");
            assert_eq!(
                compile_dslx_module(&native, &[dir.path()]).is_ok(),
                accepted,
                "{native}"
            );
            let bx = format!("use wire; {body}");
            fs::write(dir.path().join("library.bx"), &bx)?;
            for source in [bx.as_str(), "use library;"] {
                assert_eq!(
                    check_source_with_options(source, &options).is_ok(),
                    accepted,
                    "{bx}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn aliases_resolve_once_in_definition_scope() -> Result<(), Box<dyn Error>> {
    let source = r#"
const WIDTH = u32:8;
type Word = uN[WIDTH];
fn use_word<WIDTH: u32>(x: Word, y: uN[WIDTH]) -> Word { x + y as Word }
fn identity<Word>(x: Word) -> Word { x }
const RESULT = use_word<u32:16>(u8:1, identity(u16:254));
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants().last().unwrap().value,
        xlsynth_pir::IrValue::make_ubits(8, 255)?
    );
    Ok(())
}

#[test]
fn unordered_module_ast_indexes_struct_after_its_field_alias() -> Result<(), Box<dyn Error>> {
    let mut module = parse_module(
        r#"
type Word = u8;
struct Packet { data: Word }
type Payload = Packet;
fn read(x: Payload) -> Word { x.data }
"#,
    )?;
    module.declaration_order.clear();
    check_module(&module)?;
    Ok(())
}

#[test]
fn repeated_tuple_aliases_are_bounded_before_expansion() -> Result<(), Box<dyn Error>> {
    let mut source = String::from("type Level0 = u1;\n");
    for i in 1..32 {
        let prev = i - 1;
        source.push_str(&format!("type Level{i} = (Level{prev}, Level{prev});\n"));
    }
    let error = check_module(&parse_module(&source)?).unwrap_err();
    assert!(error.contains("type expansion limit"), "{error}");
    Ok(())
}
