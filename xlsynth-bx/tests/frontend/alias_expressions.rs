// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, check_module, check_module_with_options, check_source_with_options, parse_module,
};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};

#[test]
fn aliases_in_literals_constructors_and_slices_match_dslx() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
type Word = u8;
type Octet = Word;
type Wide = sN[80];
type Lanes = Octet[3];
enum Code:u2 { Idle=0, Busy=3 }
type Status = Code;
struct Packet<N:u32> { lane:uN[N], code:Code }
type Header = Packet<8>;
fn update(x:u16, c:Status) -> (Lanes, Wide, Header, bool) {
    let base = Header { lane:Octet:'z', code:Status::Busy };
    let byte = x[Octet:4+:Word];
    let changed = Header { code:c, ..base };
    let accepted = match byte { u8:1..Octet:5 => true, _ => false };
    let idle = match c { Status::Idle => true, Status::Busy => false };
    (Lanes:[byte, Word:true, ...], Wide:-0x10000000000000001, changed, idle && accepted)
}
"#;
    let result = "(Lanes, Wide, Header, bool)";
    let bx = format!(
        "{declarations}{}",
        bx_source("x:u16, c:Status", result, "update(x, c)")
    );
    let native = format!(
        "{declarations}{}",
        dslx_source("x:u16, c:Status", result, "update(x, c)")
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    for (input, code, byte, accepted) in [(0x30, 0, 3, true), (0x2ff0, 3, 255, false)] {
        assert_eq!(
            evaluator.compare(&[
                IrValue::make_ubits(16, input)?,
                IrValue::make_ubits(2, code)?
            ])?,
            IrValue::parse_typed(&format!(
                "([bits[8]:{byte},bits[8]:1,bits[8]:1], bits[80]:0xfffeffffffffffffffff, (bits[8]:122,bits[2]:{code}), bits[1]:{})",
                u8::from(accepted)
            ))?,
        );
    }
    Ok(())
}

#[test]
fn alias_to_imported_enum_preserves_members_and_identity() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("status.x"),
        "pub enum Code:u2 { Idle=0, Busy=3 }",
    )?;
    let alias = "type Alias = status::Code;";
    let body = "let busy:status::Code = Alias::Busy; match busy { Alias::Busy => true, Alias::Idle => false }";
    let bx = format!("use status; {alias} {}", bx_source("", "bool", body));
    let native = format!("import status; {alias} {}", dslx_source("", "bool", body));
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &options,
        compile_dslx_module(&native, &[directory.path()]),
    )?;
    assert_eq!(evaluator.compare(&[])?, IrValue::bool(true));
    Ok(())
}

#[test]
fn aliases_preserve_literal_widths_nominal_identity_and_pattern_syntax() {
    let declarations = r#"
type Word = u7;
type Signed = s3;
type Row = Word[2];
enum Code:u2 { Idle=0, Busy=3 }
type Status = Code;
enum Other:u2 { Idle=0, Busy=3 }
struct Packet { data:Word }
type Header = Packet;
"#;
    for (result, body) in [
        ("Word", "Word:128"),
        ("Word", "Word:'a'"),
        ("Signed", "Signed:4"),
        ("Row", "Row:0"),
        ("Word", "Word:[0]"),
        ("Status", "Status:3"),
        ("Status", "Status::Missing"),
        ("Other", "Status::Idle"),
        ("Header", "Header { data:u8:0 }"),
        ("Word", "match x { Word:1 => Word:2, _ => Word:3 }"),
        ("Word", "let Word = x; Word:2"),
        ("Word", "let Word = x; u7:Word:2"),
    ] {
        let native = format!("{declarations}{}", dslx_source("x:Word", result, body));
        assert!(compile_dslx_module(&native, &[]).is_err(), "{native}");
        assert!(
            parse_module(&native)
                .and_then(|module| check_module(&module))
                .is_err(),
            "{native}"
        );
    }
    let forward = "fn sample()->u8 { u8:Later:1 } type Later = u8;";
    assert!(compile_dslx_module(forward, &[]).is_err());
    assert!(
        parse_module(forward)
            .and_then(|module| check_module(&module))
            .is_err()
    );
}

#[test]
fn overwritten_literal_annotations_require_a_type_for_bare_imports() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let library =
        "pub type Word = u8; type Hidden = u8; pub const N = u32:8; pub fn id(x:u8)->u8{x}";
    fs::write(dir.path().join("wire.x"), library)?;
    fs::write(dir.path().join("bus.bx"), library)?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    for (annotation, accepted) in [
        ("wire", false),
        ("wire::Hidden", true),
        ("wire::Missing", true),
        ("wire[1]", false),
        ("(u8,wire)[1]", false),
        ("(u8,wire::Missing)[1]", true),
    ] {
        let native = format!("import wire; fn sample()->u8 {{ u8:{annotation}:2 }}");
        assert_eq!(
            compile_dslx_module(&native, &[dir.path()]).is_ok(),
            accepted,
            "{native}"
        );
    }
    for subject in ["wire", "bus"] {
        for (use_path, annotation, accepted) in [
            (subject.to_owned(), subject.to_owned(), false),
            (format!("{subject}::N"), "N".into(), false),
            (format!("{subject}::id"), "id".into(), false),
            (format!("{subject}::Word"), "Word".into(), true),
            (subject.to_owned(), format!("{subject}::Hidden"), true),
            (subject.to_owned(), format!("{subject}::Missing"), true),
            (subject.to_owned(), format!("{subject}[1]"), false),
            (subject.to_owned(), format!("(u8,{subject})[1]"), false),
            (
                subject.to_owned(),
                format!("(u8,{subject}::Missing)[1]"),
                true,
            ),
            (format!("{subject}::N"), "N[1]".into(), false),
            (format!("{subject}::id"), "(u8,id)[1]".into(), false),
            (format!("{subject}::Word"), "(u8,Word)[1]".into(), true),
        ] {
            let source = format!("use {use_path}; pub fn get()->u8 {{ u8:{annotation}:2 }}");
            fs::write(dir.path().join("wrapper.bx"), &source)?;
            for root in [source.as_str(), "use wrapper; type N=u8; type id=u8;"] {
                assert_eq!(
                    check_source_with_options(root, &options).is_ok(),
                    accepted,
                    "{source}"
                );
                assert_eq!(
                    check_module_with_options(&parse_module(root)?, &options).is_ok(),
                    accepted,
                    "{source}"
                );
            }
        }
        let late = format!("block b(clk)->u8 {{ u8:{subject}::Missing:2 }} use {subject};");
        check_source_with_options(&late, &options)?;
        let strict = format!("#![require_define_before_use]\n{late}");
        assert_eq!(
            check_source_with_options(&strict, &options).unwrap_err(),
            format!("in 'b': '{subject}::Missing' is declared later in the module")
        );
    }
    for annotation in ["Word", "Word[1]", "(u8,Word)[1]"] {
        let forward = format!("block b(clk)->u8 {{ u8:{annotation}:2 }} type Word=u8;");
        check_source_with_options(&forward, &options)?;
        assert!(
            check_source_with_options(
                &format!("#![require_define_before_use]\n{forward}"),
                &options
            )
            .is_err()
        );
    }
    for (declarations, annotation, accepted) in [
        ("type Word=u8;", "(u8,Word)[1]", true),
        ("struct S { x:u8 }", "S[1]", true),
        ("enum E:u1 { A=0 }", "E[1]", true),
        ("const N=u32:1;", "N[1]", false),
        ("", "(u8,Missing)[1]", false),
        ("type x=u8;", "x[1]", false),
    ] {
        let source = format!("{declarations} fn sample(x:u8)->u8 {{ u8:{annotation}:2 }}");
        assert_eq!(
            compile_dslx_module(&source, &[]).is_ok(),
            accepted,
            "{source}"
        );
        assert_eq!(
            check_source_with_options(&source, &options).is_ok(),
            accepted,
            "{source}"
        );
    }
    for head in ["wire", "local", "rev"] {
        let source = format!("fn sample(local:u8)->u8[1] {{ u8[1]:{head}::Missing[1]:[2] }}");
        compile_dslx_module(&format!("import wire; {source}"), &[dir.path()])?;
        check_source_with_options(&format!("use wire; {source}"), &options)?;
    }
    Ok(())
}

#[test]
fn aliases_shadow_builtin_calls_and_callbacks_in_declaration_order() -> Result<(), Box<dyn Error>> {
    let helper = "fn id(x:u8)->u8 { x }";
    for (name, result, body) in [
        ("rev", "u8", "rev(x)"),
        ("rev", "u8[1]", "map([x], rev)"),
        ("map", "u8[1]", "map([x], id)"),
        ("zero!", "u8", "zero!<u8>()"),
        ("bit_count", "u32", "bit_count<u8>()"),
    ] {
        let alias = format!("type {name} = u8;");
        let block = bx_source("x:u8", result, body);
        let function = dslx_source("x:u8", result, body);
        let late = format!("{helper}\n{block}\n{alias}");
        let reference = format!("{helper}\n{function}\n{alias}");
        ExpressionEvaluator::compile_block(
            &late,
            &ImportOptions::default(),
            compile_dslx_module(&reference, &[]),
        )?
        .compare(&[IrValue::make_ubits(8, 1)?])?;

        let shadowed = format!("{helper}\n{alias}\n{block}");
        let reference = format!("{helper}\n{alias}\n{function}");
        assert!(
            parse_module(&shadowed)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{shadowed}"
        );
        assert!(compile_dslx_module(&reference, &[]).is_err(), "{reference}");
    }
    // DSLX resolves the callee name in an unused template and checks that
    // the declaration is callable only when a specialization is requested.
    let source = "type Word=u8; fn f<N:u32>(x:u8)->u8 { Word(x) } fn sample()->u8 { 1 }";
    check_module(&parse_module(source)?)?;
    compile_dslx_module(source, &[])?;
    Ok(())
}

#[test]
fn wildcard_and_self_alias_spellings_match_dslx() -> Result<(), Box<dyn Error>> {
    for valid in [
        "type self=u8; type _=u8; fn sample()->_ { _:42 }",
        "struct S{x:u8} type _=S; fn sample()->S { _{x:1} }",
        "type _=u8; fn sample()->_ { let (a,_)=(u8:1,u8:2); _:2+a }",
        "enum E:u1{A=1} type self=E; fn sample()->E { self::A }",
    ] {
        check_module(&parse_module(valid)?)?;
        compile_dslx_module(valid, &[])?;
    }
    for source in [
        "type _=u8; fn sample()->_ { _ }",
        "type _=u8; fn sample()->_ { let _=u8:0; _:2 }",
        "type _=u8; fn sample(x:u8)->u8 { let _=x; x as _ }",
        "type self=u8; fn sample(x:self)->u8 { x }",
        "type self=u8; fn sample()->u8 { self:1 }",
        "struct S{x:u8} type self=S; fn sample()->S { self{x:1} }",
    ] {
        assert!(compile_dslx_module(source, &[]).is_err(), "{source}");
        assert!(
            parse_module(source).and_then(|m| check_module(&m)).is_err(),
            "{source}"
        );
    }
    let directory = tempdir()?;
    fs::write(directory.path().join("names.bx"), "pub type self=u8;")?;
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    assert!(
        check_source_with_options("use names; fn sample(x:names::self)->u8{x}", &options).is_err()
    );
    Ok(())
}

#[test]
fn native_map_and_zero_aliases_shadow_builtins_after_the_use() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("aliases.x"),
        "pub type map=u8; pub type zero! = u8;",
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    let helper = "fn id(x:u8)->u8{x}";
    for (name, result, expression) in [
        ("map", "u8[1]", "map([x], id)"),
        ("zero!", "u8", "zero!<u8>()"),
        ("zero!", "u8[1]", "map([x], |v| v + zero!<u8>())"),
    ] {
        let import = format!("use aliases::{name};");
        let block = bx_source("x:u8", result, expression);
        for policy in ["", "#![require_define_before_use]"] {
            check_source_with_options(&format!("{policy}\n{helper}\n{block}\n{import}"), &options)?;
            assert!(
                check_source_with_options(
                    &format!("{policy}\n{helper}\n{import}\n{block}"),
                    &options
                )
                .is_err(),
                "{name}"
            );
        }
    }
    Ok(())
}
