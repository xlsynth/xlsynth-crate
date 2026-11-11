// SPDX-License-Identifier: Apache-2.0

//! Compare concrete module type aliases and their defining scopes with DSLX.

use std::{fmt::Write, fs};

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;
use xlsynth_pir::{IrBits, IrFormatPreference};

/// Emit a chain of aliases ending in the name used by the generated program.
fn alias_chain(source: &mut String, name: &str, target: &str, depth: u8) {
    let mut previous = target.to_owned();
    for i in 0..depth {
        let next = format!("{name}{i}");
        writeln!(source, "type {next} = {previous};").unwrap();
        previous = next;
    }
    writeln!(source, "pub type {name} = {previous};").unwrap();
}

/// Build values through aliases and pass them between aliased and underlying
/// types. Invalid variants substitute a different shape or nominal identity.
fn local_program(data: &[u8]) -> FunctionProgram {
    let [family, flags, width, length, salt] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let family = family % 8;
    let boolean = flags & 64 != 0;
    let width = if boolean { 1 } else { usize::from(width % 130) };
    let count = usize::from(length % 5) + 2;
    let sign = if !boolean && flags & 1 != 0 { 's' } else { 'u' };
    let concrete = if boolean {
        "bool".to_owned()
    } else {
        format!("{sign}N[{width}]")
    };
    let depth = (flags >> 1) % 4;
    let invalid = flags & 32 != 0;
    let mut declarations = format!("const WIDTH = u32:{width};\nconst COUNT = u32:{count};\n");
    let target = if !boolean && flags & 16 == 0 {
        format!("{sign}N[WIDTH]")
    } else {
        concrete.clone()
    };
    alias_chain(&mut declarations, "Word", &target, depth);
    let key = IrBits::from_lsb_fn(width, |i| salt.rotate_right((i % 8) as u32) & 1 != 0)
        .to_string_fmt(IrFormatPreference::Hex, false);
    writeln!(declarations, "const KEY: Word = Word:{key};").unwrap();
    let params = "x: Word, choose: bool";
    let (result, body) = match family {
        0 => {
            let other = if sign == 's' { 'u' } else { 's' };
            let cast = if invalid {
                format!("{other}N[{width}]")
            } else {
                "Word".to_owned()
            };
            writeln!(
                declarations,
                "fn convert(a: {concrete}) -> Word {{ a as {cast} }}"
            )
            .unwrap();
            (
                "Word",
                "if choose { convert(x) ^ KEY } else { zero!<Word>() }",
            )
        }
        1 | 2 => {
            alias_chain(&mut declarations, "Row", "Word[COUNT]", depth);
            declarations.push_str("const FILL: Row = Row:[KEY, ...];\n");
            if family == 1 {
                let fill = if invalid {
                    format!("Word[{}]", count + 1)
                } else {
                    "Row".to_owned()
                };
                writeln!(
                    declarations,
                    "fn row(a: {concrete}) -> Row {{ {fill}:[a, a ^ KEY, ...] }}"
                )
                .unwrap();
                (
                    "Row[2]",
                    "let r: Row = row(x); if choose { [r, FILL] } else { [FILL, r] }",
                )
            } else {
                alias_chain(&mut declarations, "Pair", "(Word, Row)", depth);
                declarations.push_str("const SAVED: Pair = (KEY, FILL);\n");
                let fields = if invalid { "(r, a)" } else { "(a, r)" };
                writeln!(
                    declarations,
                    "fn pair(a: Word, r: Word[COUNT]) -> Pair {{ {fields} }}"
                )
                .unwrap();
                (
                    "Pair",
                    "let r: Row = Row:[x, SAVED.0, ...]; if choose { pair(x, r) } else { SAVED }",
                )
            }
        }
        3 | 4 => {
            let parametric = family == 4;
            let formals = if parametric {
                declarations.push_str("type Dim = u32;\n");
                "<N: Dim>"
            } else {
                ""
            };
            writeln!(
                declarations,
                "struct Packet{formals} {{ payload: Word, flag: bool }}"
            )
            .unwrap();
            let target = if parametric {
                "Packet<COUNT>"
            } else {
                "Packet"
            };
            alias_chain(&mut declarations, "Record", target, depth);
            let constructor = if invalid && parametric {
                // The phantom argument changes identity while preserving every
                // field.
                format!("Packet<u32:{}>", count + 1)
            } else if invalid {
                declarations.push_str("struct Other { payload: Word, flag: bool }\n");
                "Other".to_owned()
            } else {
                "Record".to_owned()
            };
            declarations.push_str("const SAVED: Record = Record { payload: KEY, flag: false };\n");
            writeln!(declarations, "fn build(a: Word, flag: bool) -> {target} {{ {constructor} {{ payload: a ^ KEY, flag: flag }} }}").unwrap();
            (
                "Record",
                "let p: Record = build(x, choose); if p.flag { p } else { SAVED }",
            )
        }
        5 => {
            alias_chain(&mut declarations, "Repr", "u2", depth);
            declarations.push_str("enum Mode : Repr { Off = 0, On = 2 }\n");
            alias_chain(&mut declarations, "Choice", "Mode", depth);
            declarations.push_str("const ON: Choice = Choice::On;\n");
            let on = if invalid {
                declarations.push_str("enum Other : u2 { Off = 0, On = 2 }\n");
                "Other::On"
            } else {
                "ON"
            };
            writeln!(declarations, "fn flip(c: Mode) -> Choice {{ match c {{ Choice::Off => {on}, _ => Choice::Off }} }}").unwrap();
            (
                "Choice",
                "let c: Choice = if choose { Choice::On } else { Choice::Off }; flip(c)",
            )
        }
        _ => {
            declarations.push_str(match salt % 8 {
                0 => "type Broken = Later; type Later = Word;",
                1 => "type Broken = Broken;",
                2 => "type Word = u32;",
                3 => "struct Word { x: bool }",
                4 => "type Broken = Word[UNKNOWN];",
                5 => "type Broken = KEY;",
                6 => "fn broken<Word: Word>(x: u8) -> u8 { x }",
                _ => "struct Broken<Word: Word> { x: u8 }",
            });
            ("Word", "x ^ KEY")
        }
    };
    FunctionProgram {
        functions: declarations,
        params: params.into(),
        result: result.into(),
        body: body.into(),
        valid: Some(family != 6 && !invalid),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Import a public alias of a private declaration and use it from a module
/// with its own identically named types.
fn compare_imports(data: &[u8]) -> Result<(), String> {
    let [_, flags, width, variant] = std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let width = usize::from(width % 128) + 2;
    let sign = if flags & 1 == 0 { 'u' } else { 's' };
    // A public aggregate may be the only name available for values that
    // contain the library's private nominal types.
    let opaque = flags & 64 != 0;
    let packet_api = if !opaque {
        r#"struct Hidden { value: Word }
pub type Packet = Hidden;
pub fn make(x: Lane) -> Packet { Hidden { value: x } }
pub fn read(p: Packet) -> Lane { p.value }"#
    } else if flags & 128 == 0 {
        r#"struct Hidden { value: Word }
pub type Packet = (Hidden, bool);
pub fn make(x: Lane) -> Packet { (Hidden { value: x }, x != Lane:0) }
pub fn read(p: Packet) -> Lane { if p.1 { p.0.value } else { Lane:1 } }"#
    } else {
        r#"enum Hidden : u2 { Off = 0, On = 1 }
pub type Packet = Hidden[2];
pub fn make(x: Lane) -> Packet {
    [if x == Lane:0 { Hidden::Off } else { Hidden::On }, Hidden::Off]
}
pub fn read(p: Packet) -> Lane { (p[0] == p[1]) as Lane }"#
    };
    let packet_api = if opaque && data.get(4).is_some_and(|byte| byte & 2 != 0) {
        let shape = if flags & 128 == 0 {
            "(Hidden, bool)"
        } else {
            "Hidden[2]"
        };
        packet_api.replace("-> Packet", &format!("-> {shape}"))
    } else {
        packet_api.to_owned()
    };
    let library = format!(
        r#"const WIDTH = u32:{width};
type Word = {sign}N[WIDTH];
pub type Lane = Word;
{packet_api}
pub type Batch = (Packet[2], bool);
pub fn read_batch(b: Batch) -> Lane {{
    if b.1 {{ read(b.0[0]) - read(b.0[1]) }} else {{ read(b.0[1]) - read(b.0[0]) }}
}}"#
    );
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    fs::write(dir.path().join("reference.x"), &library).map_err(|e| e.to_string())?;
    fs::write(dir.path().join("native.bx"), &library).map_err(|e| e.to_string())?;
    let (bx_library, dslx_library) = if flags & 8 == 0 {
        ("native", "reference")
    } else {
        // Export the private leaf types through a second module. The helper
        // passes the nominal value across both module boundaries.
        let facade = r#"type Word = leaf::Lane;
type Hidden = leaf::Packet;
pub type Lane = Word;
pub type Packet = Hidden;
pub fn make(x: Lane) -> Packet { leaf::make(x) }
pub fn read(p: Packet) -> Lane { leaf::read(p) }
pub type Batch = leaf::Batch;
pub fn read_batch(b: Batch) -> Lane { leaf::read_batch(b) }"#;
        fs::write(
            dir.path().join("facade.x"),
            format!(
                "import reference;\n{}",
                facade.replace("leaf::", "reference::")
            ),
        )
        .map_err(|e| e.to_string())?;
        let leaf = if flags & 16 == 0 {
            "native"
        } else {
            "reference"
        };
        fs::write(
            dir.path().join("bx_facade.bx"),
            format!(
                "use {leaf};\n{}",
                facade.replace("leaf::", &format!("{leaf}::"))
            ),
        )
        .map_err(|e| e.to_string())?;
        ("bx_facade", "facade")
    };
    let bx_module = if flags & 2 == 0 {
        bx_library
    } else {
        dslx_library
    };
    let variant = variant % 5;
    let packet = if variant == 2 { "Hidden" } else { "Packet" };
    let lane = if variant == 3 { "Word" } else { "Lane" };
    let inferred = opaque && variant == 0 && data.get(4).is_some_and(|byte| byte & 1 != 0);
    let mut declarations = format!(
        "type Word = u32; type Field = reference::{lane};\nstruct Hidden {{ value: Field }}"
    );
    if !inferred {
        writeln!(declarations, "\ntype Record = reference::{packet};").unwrap();
    }
    let batch = flags & 32 != 0;
    let batch_imports = if batch {
        if !inferred {
            declarations.push_str("\ntype Samples = reference::Batch;");
        }
        ", Batch, read_batch"
    } else {
        ""
    };
    // Private names use the qualifier to avoid colliding with the caller's
    // declarations. The pinned DSLX typechecker cannot resolve a use-bound
    // scalar alias on another alias's RHS, so its reference stays qualified.
    let direct = flags & 4 != 0 && variant < 2;
    let mut bx_imports = if direct {
        format!("use {bx_module}::{{{packet}, {lane}, make, read{batch_imports}}};")
    } else {
        format!("use {bx_module};")
    };
    let mut dslx_imports = format!("import {dslx_library};");
    if variant == 4 {
        // An outer annotation replaces the inner type. Both frontends still
        // require a bare inner name to refer to a type, even across imports.
        writeln!(bx_imports, "\nconst PROBE = u32:{bx_module}:2;").unwrap();
        writeln!(dslx_imports, "\nconst PROBE = u32:{dslx_library}:2;").unwrap();
    }
    let bx_qualifier = if direct {
        String::new()
    } else {
        format!("{bx_module}::")
    };
    let make = |value: &str| {
        if opaque {
            format!("reference::make({value})")
        } else {
            format!("Record {{ value: {value} }}")
        }
    };
    let record_annotation = if inferred { "" } else { ": Record" };
    let samples_annotation = if inferred { "" } else { ": Samples" };
    let body = if variant == 1 {
        "let p = Hidden { value: x as Field }; reference::read(p) as Word".to_owned()
    } else if batch {
        format!(
            "let p{record_annotation} = {}; let samples{samples_annotation} = ([p, {}], (x & u32:1) == u32:0); reference::read_batch(samples) as Word",
            make("x as Field"),
            make("Field:1")
        )
    } else {
        format!(
            "let y: Field = x as Field; let p{record_annotation} = {}; reference::read(p) as Word",
            make("y ^ Field:reference::Lane:1")
        )
    };
    let params = "x: u32";
    let result = "Word";
    let bx = format!(
        "{bx_imports}\n{}\n{}",
        declarations.replace("reference::", &bx_qualifier),
        bx_source(params, result, &body.replace("reference::", &bx_qualifier))
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: format!(
                "{dslx_imports}\n{}",
                declarations.replace("reference::", &format!("{dslx_library}::"))
            ),
            params: params.into(),
            result: result.into(),
            body: body.replace("reference::", &format!("{dslx_library}::")),
            valid: Some(variant == 0),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[dir.path()],
    )
    .map_err(|e| format!("{e}\nimported module:\n{library}"))
}

/// Compare typechecking, constant declarations and evaluated block outputs.
pub fn check_type_aliases(data: &[u8]) -> Result<(), String> {
    if data.first().copied().unwrap_or(0) % 8 == 7 {
        return compare_imports(data);
    }
    compare_program(
        local_program(data),
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_type_aliases;

    #[test]
    fn compare_local_alias_values() -> Result<(), String> {
        for family in 0..6 {
            for flags in [0, 7, 16, 32, 39, 64] {
                check_type_aliases(&[family, flags, 95, 3, 0xaa])?;
            }
            check_type_aliases(&[family, 1, 0, 0, 0])?;
        }
        for error in 0..8 {
            check_type_aliases(&[6, 0, 12, 1, error])?;
        }
        Ok(())
    }

    #[test]
    fn compare_bx_aliases_across_modules() -> Result<(), String> {
        for flags in [0, 1, 4, 5, 8, 9, 12, 13, 32, 40, 64, 192] {
            for variant in 0..5 {
                check_type_aliases(&[7, flags, 63, variant])?;
            }
        }
        Ok(())
    }

    #[test]
    fn compare_dslx_aliases_across_modules() -> Result<(), String> {
        for flags in [2, 3, 6, 7, 10, 11, 14, 15, 34, 38, 42, 66, 194, 230] {
            for variant in 0..5 {
                check_type_aliases(&[7, flags, 63, variant])?;
            }
        }
        // A BX facade can re-export types from a DSLX leaf.
        for flags in [28, 56, 124, 252] {
            for variant in 0..5 {
                check_type_aliases(&[7, flags, 30, variant])?;
            }
        }
        // The caller may pass an imported result straight to the next call,
        // without first naming the private elements or their public alias.
        for flags in [66, 194, 230, 124, 252] {
            check_type_aliases(&[7, flags, 17, 0, 1])?;
            check_type_aliases(&[7, flags, 17, 0, 3])?;
        }
        Ok(())
    }
}
