// SPDX-License-Identifier: Apache-2.0

//! Compare packed struct methods with native DSLX derivation and execution.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;
use std::fs;

/// Generate the same struct declarations in both languages, then check calls
/// on runtime records and the packed value of a concrete module constant.
pub fn check_derive_to_bits(data: &[u8]) -> Result<(), String> {
    let [family, flags, width_byte, lanes_byte] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let family = family % 15;
    let width = usize::from(width_byte % 130);
    let lanes = usize::from(lanes_byte % 4) + 1;
    let sign = if flags & 1 == 0 { 'u' } else { 's' };
    let generic = flags & 2 != 0;
    let formal = if generic { "<N: u32>" } else { "" };
    let dimension = if generic {
        "N".into()
    } else {
        width.to_string()
    };
    let field_word = format!("{sign}N[{dimension}]");
    let input_word = format!("{sign}N[{width}]");
    let record = if generic {
        format!("Record<{width}>")
    } else {
        "Record".into()
    };

    // DSLX never invokes the element method when an array is empty, even
    // when a nonempty array of the same unmarked struct would be rejected.
    let empty_leaf_array = family == 7 && flags & 64 != 0;
    let leaf_derive = if family == 10 || empty_leaf_array {
        ""
    } else {
        "#[derive(ToBits)]"
    };
    let record_derive = if family == 9 { "" } else { "#[derive(ToBits)]" };
    let mut library = format!(
        r#"enum Tag : s3 {{ Low = -2, High = 3 }}
{leaf_derive}
struct Leaf {{ tagged: Tag, bit: bool }}
type LeafAlias = Leaf;"#
    );
    let (payload_ty, payload_expr, payload_width) = match family {
        1 => (format!("({field_word}, bool)"), "(x, on)".into(), width + 1),
        2 => (format!("{field_word}[{lanes}]"), "xs".into(), width * lanes),
        3 => ("Tag".into(), "tag".into(), 3),
        4 | 10 => ("LeafAlias".into(), "leaf".into(), 4),
        5 => (
            "((Leaf, bool), (bool,))[2]".into(),
            "[((leaf, on), (!on,)), ((other, !on), (on,))]".into(),
            12,
        ),
        6 => ("()".into(), "()".into(), 0),
        7 if empty_leaf_array => ("Leaf[0]".into(), "Leaf[0]:[]".into(), 0),
        7 => (
            format!("{field_word}[0]"),
            format!("zero!<{input_word}[0]>()"),
            0,
        ),
        _ => (field_word.clone(), "x".into(), width),
    };
    let (fields, members, packed_width) = if family == 8 {
        (String::new(), String::new(), 0)
    } else {
        let flag_field = if family == 14 { "to_bits" } else { "flag" };
        let fields = if flags & 4 == 0 {
            format!("{flag_field}: bool, item: {payload_ty}, tail: {field_word}")
        } else {
            format!("tail: {field_word}, item: {payload_ty}, {flag_field}: bool")
        };
        let members = format!("item: {payload_expr}, tail: !x, {flag_field}: on");
        (fields, members, 1 + payload_width + width)
    };
    let params = format!("x: {input_word}, xs: {input_word}[{lanes}], on: bool");
    let leaf_values = match family {
        3 => "let tag = if on { Tag::Low } else { Tag::High };",
        4 | 10 => {
            r#"let tag = if on { Tag::Low } else { Tag::High };
            let leaf = Leaf { bit: on, tagged: tag };"#
        }
        5 => {
            r#"let tag = if on { Tag::Low } else { Tag::High };
            let leaf = Leaf { bit: on, tagged: tag };
            let other = Leaf { bit: !on, tagged: if on { Tag::High } else { Tag::Low } };"#
        }
        _ => "",
    };
    library.push_str(&format!(
        r#"
{record_derive}
pub struct Record{formal} {{ {fields} }}
pub fn build({params}) -> {record} {{ {leaf_values} {record} {{ {members} }} }}"#
    ));

    let valid = family < 9;
    let result_bits = format!("uN[{}]", packed_width + usize::from(family == 12));
    // Use a concrete receiver call to materialize DSLX's lazy derived method
    // before referring to it as a named map callback.
    let map = flags & 8 != 0 && valid;
    let method_type = if map {
        format!("{result_bits}[2]")
    } else {
        result_bits.clone()
    };
    let result = if valid {
        format!("({method_type}, {result_bits})")
    } else {
        method_type
    };
    let concrete_lanes = (0..lanes)
        .map(|i| {
            if i % 2 == 0 {
                format!("!{input_word}:0")
            } else {
                format!("{input_word}:0")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let consumer = |declarations: &str, prefix: &str| {
        let mut functions = declarations.to_owned();
        if valid {
            functions.push_str(&format!(
                r#"
fn pack(p: {prefix}{record}) -> {result_bits} {{ p.to_bits() }}
const PACKED = {prefix}build({input_word}:0, [{concrete_lanes}], true).to_bits();"#
            ));
        }
        let method = match family {
            11 => "p.to_bits(on)".into(),
            13 => "x.to_bits()".into(),
            _ if map => format!("map([p, {prefix}build(!x, xs, !on)], {prefix}{record}::to_bits)"),
            _ if valid && flags & 16 != 0 => "pack(p)".into(),
            _ => "p.to_bits()".into(),
        };
        let returned = if valid {
            format!("({method}, PACKED)")
        } else {
            method
        };
        (
            functions,
            format!("let p = {prefix}build(x, xs, on); {returned}"),
        )
    };
    let dir = if flags & 32 != 0 {
        let imports = tempfile::tempdir().map_err(|e| e.to_string())?;
        for filename in ["native.bx", "records.x"] {
            fs::write(imports.path().join(filename), &library).map_err(|e| e.to_string())?;
        }
        Some(imports)
    } else {
        None
    };
    let ((bx_decls, bx_body), (functions, body)) = match &dir {
        Some(_) => (
            consumer("use native;", "native::"),
            consumer("import records;", "records::"),
        ),
        None => {
            let shared = consumer(&library, "");
            (shared.clone(), shared)
        }
    };
    let bx = format!("{bx_decls}\n{}", bx_source(&params, &result, &bx_body));
    let search_paths = dir.iter().map(|d| d.path()).collect::<Vec<_>>();
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions,
            params,
            body,
            result,
            valid: Some(valid),
            // DSLX IR cannot construct the empty array used in this field.
            dslx_ir: family != 7,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        vec![],
        &search_paths,
    )
}

#[cfg(test)]
mod tests {
    use super::check_derive_to_bits;

    #[test]
    fn compare_derive_to_bits_families() -> Result<(), String> {
        for family in 0..15 {
            for (flags, width) in [
                (0, 0),
                (7, 67),
                (8, 5),
                (11, 8),
                (18, 129),
                (32, 4),
                (39, 66),
                (40, 0),
                (43, 9),
                (103, 4),
            ] {
                check_derive_to_bits(&[family, flags, width, 2])?;
            }
        }
        Ok(())
    }
}
