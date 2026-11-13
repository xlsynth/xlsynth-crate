// SPDX-License-Identifier: Apache-2.0

//! Compare public constants read from native DSLX modules by BX and DSLX.

use std::{fmt::Write, fs};

use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

/// Two native libraries with the same public constant names and distinct
/// values.
struct Libraries {
    samples: String,
    derived: String,
    word_type: String,
    row_type: String,
}

/// Build the leaf values with private constants and constexpr calls. The
/// dependent library computes new values while retaining the leaf nominal
/// types.
fn libraries(width: usize, count: usize, flags: u8, seed: u64, family: u8) -> Libraries {
    let signed = flags & 1 != 0;
    let word_type = format!("{}N[{width}]", if signed { 's' } else { 'u' });
    let row_type = format!("{word_type}[{count}]");
    let pattern = IrBits::from_lsb_fn(width, |i| seed.rotate_right((i % 64) as u32) & 1 != 0)
        .to_string_fmt(IrFormatPreference::Hex, false);
    let visibility = if flags & 8 == 0 { "pub " } else { "" };
    let choice = if flags & 16 == 0 { "Off" } else { "On" };
    let mut samples = format!(
        r#"pub const SIGNED = {signed};
pub const WIDTH = u32:{width};
pub const COUNT = u32:{count};
type Word = {word_type};
const HIDDEN = Word:{pattern};
fn flip(x: Word) -> Word {{ !x }}
pub const WORD = flip(HIDDEN);
pub const ROW: Word[COUNT] = [WORD, HIDDEN, ...];
{visibility}enum Mode : u2 {{ Off = 0, On = 3 }}
{visibility}struct Packet {{ mode: Mode, row: Word[COUNT] }}
pub const MODE = Mode::{choice};
pub const PACKET = Packet {{ mode: MODE, row: ROW }};
pub const NESTED = ([PACKET, PACKET], (WORD, MODE));
pub fn flip_mode(m: Mode) -> Mode {{ if m == Mode::Off {{ Mode::On }} else {{ Mode::Off }} }}
pub fn make(m: Mode, r: Word[COUNT]) -> Packet {{ Packet {{ mode: m, row: r }} }}"#
    );
    let mut derived = r#"import samples;
pub const SIGNED = samples::SIGNED;
pub const WIDTH = samples::WIDTH;
pub const COUNT = samples::COUNT;
const HIDDEN = samples::WORD;
pub const WORD = !HIDDEN;
pub const ROW = array_rev(samples::ROW);
pub const MODE = samples::flip_mode(samples::MODE);
pub const PACKET = samples::make(MODE, ROW);
pub const NESTED = ([PACKET, samples::PACKET], (WORD, MODE));"#
        .to_owned();
    if matches!(family, 12 | 13) {
        // Both labels have the same fields. Only the value parameter
        // distinguishes the type accepted by observe.
        samples.push_str(
            r#"
pub struct Label<K: u32> { word: Word }
const NEXT = COUNT + u32:1;
pub const LABEL = Label<COUNT> { word: WORD };
pub const OTHER_LABEL = Label<NEXT> { word: HIDDEN };
pub fn observe(p: Label<COUNT>) -> Word { p.word }"#,
        );
        derived.push_str(
            r#"
pub const LABEL = samples::LABEL;
pub const OTHER_LABEL = samples::OTHER_LABEL;
pub fn observe(p: samples::Label<COUNT>) -> xN[SIGNED][WIDTH] { samples::observe(p) }"#,
        );
    }
    if family == 14 {
        samples.push_str(
            r#"
pub const EMPTY = Word[0]:[];
struct Hollow { word: Word, row: Word[0] }
const FIRST = Hollow { word: WORD, row: EMPTY };
const SECOND = Hollow { word: HIDDEN, row: EMPTY };
pub const HOLLOWS = ([FIRST, SECOND], EMPTY);"#,
        );
        derived.push_str(
            r#"
pub const EMPTY = samples::EMPTY;
pub const HOLLOWS = (array_rev(samples::HOLLOWS.0), EMPTY);"#,
        );
    }
    Libraries {
        samples,
        derived,
        word_type,
        row_type,
    }
}

/// Declarations and expressions shared by the BX block and DSLX function.
struct Consumer {
    members: &'static [&'static str],
    constants: String,
    types: String,
    result: String,
    body: String,
    valid: bool,
}

/// Select a public value shape and the operation that observes its type and
/// value. Library prefixes are substituted after choosing the import spelling.
fn consumer(family: u8, libs: &Libraries) -> Consumer {
    let word = &libs.word_type;
    let row = &libs.row_type;
    let (members, result, body): (&[&str], String, &str) = match family {
        0 => (
            &["WORD"],
            word.clone(),
            "if flag { lib::WORD } else { other::WORD }",
        ),
        1 => (
            &["ROW"],
            row.clone(),
            "if flag { COPIED } else { other::ROW }",
        ),
        2 => (
            &["MODE"],
            "u2".into(),
            "(if flag { COPIED } else { other::MODE }) as u2",
        ),
        3 => (
            &["PACKET"],
            format!("(u2, {row})"),
            "let p = if flag { lib::PACKET } else { other::PACKET }; (p.mode as u2, p.row)",
        ),
        4 => (
            &["NESTED"],
            format!("({row}, {word}, u2)"),
            "let t = if flag { COPIED } else { other::NESTED }; (t.0[flag as u32].row, t.1.0, t.1.1 as u2)",
        ),
        5 => (
            &["ROW", "WIDTH", "SIGNED", "COUNT"],
            word.clone(),
            "let row: xN[lib::SIGNED][lib::WIDTH][lib::COUNT] = lib::ROW; row[flag as u32]",
        ),
        6 => (
            &["WORD"],
            "bool".into(),
            "let x = if flag { COPIED } else { !COPIED }; match x { lib::WORD => true, _ => false }",
        ),
        7 => (
            &["WORD"],
            word.clone(),
            "let WORD = other::WORD; if flag { WORD } else { COPIED }",
        ),
        8 => (&["HIDDEN"], word.clone(), "COPIED"),
        9 => (&["MISSING"], word.clone(), "COPIED"),
        10 => (&["MODE"], "Mode".into(), "COPIED"),
        11 => (&["PACKET"], "Packet".into(), "COPIED"),
        12 => (&["LABEL"], word.clone(), "other::observe(COPIED)"),
        13 => (&["OTHER_LABEL"], word.clone(), "other::observe(COPIED)"),
        _ => (
            &["EMPTY", "HOLLOWS"],
            word.clone(),
            "if flag { FIRST } else { SECOND }",
        ),
    };
    let constants = if family == 14 {
        // Compare observations in the declared DSLX types. Its IR Value
        // representation cannot materialize a constant containing an empty
        // array.
        format!(
            r#"const SIZE = array_size(lib::EMPTY) + array_size(lib::HOLLOWS.1);
const FIRST = lib::HOLLOWS.0[u32:0].word ^ (SIZE as {word});
const SECOND = lib::HOLLOWS.0[u32:1].word ^ (array_size(lib::HOLLOWS.0[u32:1].row) as {word});"#
        )
    } else {
        format!("const COPIED = lib::{};", members[0])
    };
    let mut types = String::new();
    if matches!(family, 10 | 11) {
        types.push_str("enum Mode : u2 { Off = 0, On = 3 }");
        if family == 11 {
            types.push_str(&format!("\nstruct Packet {{ mode: Mode, row: {row} }}"));
        }
    }
    Consumer {
        members,
        constants,
        types,
        result,
        body: body.into(),
        valid: family < 8 || matches!(family, 12 | 14),
    }
}

/// Compare qualified and member imports through native source checking,
/// constant evaluation, and compiled IR on the same concrete inputs.
pub fn check_imported_constants(data: &[u8]) -> Result<(), String> {
    let [family, flags, width, length] = std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let family = family % 15;
    let seed = input_seed(data);
    let libs = libraries(
        usize::from(width % 130),
        2 + usize::from(length % 4),
        flags,
        seed,
        family,
    );
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    fs::write(dir.path().join("samples.x"), &libs.samples).map_err(|e| e.to_string())?;
    fs::write(dir.path().join("derived.x"), &libs.derived).map_err(|e| e.to_string())?;
    let c = consumer(family, &libs);
    let (module, other) = if flags & 2 == 0 {
        ("samples", "derived")
    } else {
        ("derived", "samples")
    };
    let module_prefix = format!("{module}::");
    let (bx_imports, bx_prefix) = if flags & 4 == 0 {
        (
            format!("use {module}; use {other};"),
            module_prefix.as_str(),
        )
    } else {
        let items = if c.members.len() == 1 {
            c.members[0].to_owned()
        } else {
            format!("{{{}}}", c.members.join(", "))
        };
        (format!("use {module}::{items}; use {other};"), "")
    };
    let mut relay = String::new();
    let (bx_imports, bx_prefix) = if flags & 128 != 0 {
        relay = bx_imports;
        let mut exported = Vec::with_capacity(c.members.len());
        for name in c.members {
            writeln!(relay, "\npub const EXPORTED_{name} = {bx_prefix}{name};").unwrap();
            exported.push(format!("EXPORTED_{name}"));
        }
        fs::write(dir.path().join("relay.bx"), &relay).map_err(|e| e.to_string())?;
        if flags & 32 == 0 {
            (format!("use relay; use {other};"), "relay::EXPORTED_")
        } else {
            (
                format!("use relay::{{{}}}; use {other};", exported.join(", ")),
                "EXPORTED_",
            )
        }
    } else {
        (bx_imports, bx_prefix)
    };
    let render = |s: &str, prefix: &str| {
        s.replace("lib::", prefix)
            .replace("other::", &format!("{other}::"))
    };
    let mut bx_decls = format!("{bx_imports}\n{}", c.types);
    // Ordinary DSLX imports enforce public visibility. The pinned experimental
    // use syntax accepts even a direct import of a private constant.
    let mut native_decls = format!(
        "import {module}; import {other};\n{}\n{}",
        c.types,
        render(&c.constants, &module_prefix)
    );
    let params = "flag: bool";
    let mut bx_body = render(&c.body, bx_prefix);
    let bindings = render(&c.constants, bx_prefix);
    if relay.is_empty() {
        writeln!(bx_decls, "\n{bindings}").unwrap();
    } else {
        // Observe the re-exported constants from their consuming BX body.
        bx_body = format!("{}\n{bx_body}", bindings.replace("const ", "let "));
    }
    let mut native_body = render(&c.body, &module_prefix);
    if flags & 32 != 0 {
        bx_decls.push_str(&format!(
            "\nfn read({params}) -> {} {{ {bx_body} }}",
            c.result
        ));
        native_decls.push_str(&format!(
            "\nfn read({params}) -> {} {{ {native_body} }}",
            c.result
        ));
        bx_body = "read(flag)".into();
        native_body = "read(flag)".into();
    }
    let bx = if flags & 64 != 0 {
        bx_decls.push_str(&format!(
            "\npub fn read_value({params}) -> {} {{ {bx_body} }}",
            c.result
        ));
        fs::write(dir.path().join("reader.bx"), &bx_decls).map_err(|e| e.to_string())?;
        // The wrong nominal result belongs to reader.bx. Observe its call
        // without introducing another unresolved type name in the root.
        let (result, body) = if matches!(family, 10 | 11) {
            ("bool", "let attempted = read_value(flag); true")
        } else {
            (c.result.as_str(), "read_value(flag)")
        };
        format!(
            "use reader::read_value;\n{}",
            bx_source(params, result, body)
        )
    } else {
        format!("{bx_decls}\n{}", bx_source(params, &c.result, &bx_body))
    };
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: native_decls,
            params: params.into(),
            result: c.result,
            body: native_body,
            valid: Some(c.valid),
            dslx_ir: family != 14,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        vec![vec![IrValue::bool(false)], vec![IrValue::bool(true)]],
        &[dir.path()],
    )
    .map_err(|e| {
        let reader = if flags & 64 != 0 {
            format!("\nreader.bx:\n{bx_decls}")
        } else {
            String::new()
        };
        let relay = if relay.is_empty() {
            relay
        } else {
            format!("\nrelay.bx:\n{relay}")
        };
        format!(
            "{e}{reader}{relay}\nsamples.x:\n{}\nderived.x:\n{}",
            libs.samples, libs.derived
        )
    })
}

#[cfg(test)]
mod tests {
    use super::check_imported_constants;

    #[test]
    fn compare_native_constant_imports() -> Result<(), String> {
        for family in 0..15 {
            for (flags, width) in [
                (0, 0),
                (7, 129),
                (105, 97),
                (126, 64),
                (128, 0),
                (161, 129),
                (196, 17),
                (255, 64),
            ] {
                check_imported_constants(&[family, flags, width, 3, 0xc3, 0xb7])?;
            }
        }
        Ok(())
    }
}
