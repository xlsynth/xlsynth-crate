// SPDX-License-Identifier: Apache-2.0

//! Compare specialization-dependent branch selection with pinned DSLX.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

/// Generate a conditional body and its expected acceptance at these bindings.
fn conditional_body(
    family: u8,
    flags: u8,
    first: bool,
    guard: &str,
    cast_width: usize,
) -> (String, bool) {
    match family {
        0 => (
            format!(
                r#"let v = const if {guard} {{ x as sN[{cast_width}] }}
                   else if B {{ !x as uN[{cast_width}] }} else {{ x }};
                   v as uN[N]"#
            ),
            true,
        ),
        1 => (
            format!(
                r#"let v = const if {guard} {{ (x, !x) }}
                   else if B {{ [x, !x] }} else {{ x }};
                   const if {guard} {{ v.0 }} else if B {{ v[u32:1] }}
                   else {{ if sel {{ v }} else {{ !v }} }}"#
            ),
            true,
        ),
        2 => (
            "const if N == u32:1 { x ^ u1:1 } else { x + uN[N]:1 }".into(),
            true,
        ),
        // A reached bad guard fails before either arm can be selected.
        3 => ("const if sel { x } else { !x }".into(), false),
        4 => ("const if u32:1 { x } else { !x }".into(), false),
        5 => (
            format!("const if {guard} {{ x }} else if sel {{ !x }} else {{ x }}"),
            first,
        ),
        6 => (
            format!("const if {guard} {{ x }} else if u2:0 {{ !x }} else {{ x }}"),
            first,
        ),
        // Only the selected arm needs a well-typed value and valid assertions.
        7 => (
            format!("const if {guard} {{ x }} else {{ (x, x) + sel }}"),
            first,
        ),
        8 => (
            format!("const if {guard} {{ x }} else {{ const_assert!(false); !x }}"),
            first,
        ),
        // DSLX resolves identifiers and parses the entire conditional,
        // including a guard or body discarded at a particular binding.
        9 => (
            format!("const if {guard} {{ x }} else {{ missing }}"),
            false,
        ),
        10 => (
            format!("const if {guard} {{ x }} else if missing {{ x }} else {{ !x }}"),
            false,
        ),
        11 => (
            format!("const if {guard} {{ x }} else const if B {{ x }} else {{ !x }}"),
            false,
        ),
        // A visible type can qualify the name in a skipped arm, including
        // a module type alias. Member lookup waits for branch selection.
        12 => (
            format!("const if {guard} {{ x }} else {{ Mode::Unknown as uN[N] }}"),
            first,
        ),
        13 => {
            // DSLX validates the constructor's field names even in an
            // inactive arm. It checks the field value types only if reached.
            let (members, valid) = match (flags >> 4) % 7 {
                0 => ("left:u8:0", false),
                1 => ("left:u8:0, right:u8:0, extra:u8:0", false),
                2 => ("left:u8:0, right:u8:0, left:u8:1", false),
                3 => ("extra:u8:0, ..base", false),
                4 => ("left:u8:0, left:u8:1, ..base", false),
                5 => ("left:(x, x), right:u8:0", first),
                _ => ("left:(x, x), ..base", first),
            };
            (
                format!(
                    r#"const if {guard} {{ x }} else {{
                       let base = Pair {{ left:u8:0, right:u8:0 }};
                       let pair = Pair {{ {members} }};
                       pair.left as uN[N] }}"#
                ),
                valid,
            )
        }
        14 => (
            format!("let v = const if {guard} {{ !x }}; const if {guard} {{ v }} else {{ x }}"),
            true,
        ),
        15 => (
            format!("[x, !x][const if {guard} {{ 0 }} else {{ 1 }}]"),
            true,
        ),
        16 => (
            format!("if (const if {guard} {{ 1 }} else {{ 0 }}) == x {{ !x }} else {{ x }}"),
            true,
        ),
        17 => {
            let keyword = if flags & 16 == 0 {
                "for"
            } else {
                "unroll_for!"
            };
            (
                format!(
                    r#"let offset = {keyword} (i, sum) in (const if {guard} {{ 1 }} else {{ 2 }})..u8:3 {{
                       sum + i
                    }}(u8:0); x + (offset as uN[N])"#
                ),
                true,
            )
        }
        18 => {
            let keyword = if flags & 16 == 0 {
                "for"
            } else {
                "unroll_for!"
            };
            let annotation = if flags & 32 == 0 { "" } else { ": u16" };
            let sibling = if flags & 64 == 0 { "" } else { " + u16:3" };
            (
                format!(
                    r#"let offset{annotation} = ({keyword} (_, acc) in u32:0..u32:2 {{ acc }}(
                       const if {guard} {{ 255 + 1 }} else {{ 7 }}
                    )){sibling}; x ^ (offset as uN[N])"#
                ),
                true,
            )
        }
        _ => (
            format!(
                "let v = sized(const if {guard} {{ 1 }} else {{ 255 + 1 }}); x ^ (v.0 as uN[N]) ^ (v.1 as uN[N])"
            ),
            true,
        ),
    }
}

/// Instantiate a const conditional in a BX function or combinational block
/// and compare it with a DSLX function at the same value bindings.
pub fn check_const_if(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 20;
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let cast_width = usize::from(bytes.next().unwrap() % 129) + 1;
    let first = flags & 1 != 0;
    let second = flags & 2 != 0;
    let block = flags & 4 != 0;
    let (binding, guard) = if flags & 8 == 0 {
        ("", "A")
    } else {
        ("let choose_first = A;", "choose_first")
    };
    let (expression, valid) = conditional_body(family, flags, first, guard, cast_width);
    let signature = "choose<N:u32, A:bool, B:bool>";
    let params = "x:uN[N], sel:bool";
    let template_body = format!("{binding}\n{expression}");
    let declarations = match family {
        12 if flags & 16 != 0 => "struct Record { v:u8 } type Mode = Record;",
        12 => "enum Mode:u1 { Keep=u1:0, }",
        13 => "struct Pair { left:u8, right:u8 }",
        19 => "fn sized<W:u32>(word:uN[W]) -> (uN[W], u32) { (word, W) }",
        _ => "",
    };
    let function =
        format!("{declarations}\nfn {signature}({params}) -> uN[N] {{ {template_body} }}");
    let bx_decl = if block {
        format!(
            "{declarations}\n#[allow(combinational_block)]\nblock {signature}(clk, {params}) -> uN[N] {{ {template_body} }}"
        )
    } else {
        function.clone()
    };
    let word = format!("uN[{width}]");
    let call = format!("choose<u32:{width}, {first}, {second}>(x, sel)");
    let (params, result, body) = if matches!(family, 0..=2 | 14..=19) {
        // Exercise all three outcomes with the same data port signature, and
        // instantiate a different width in the same module.
        (
            format!("x:{word}, y:u1, sel:bool"),
            format!("({word}, {word}, {word}, u1, {word})"),
            format!(
                r#"let a = choose<u32:{width}, true, true>(x, sel);
                   let b = choose<u32:{width}, false, true>(x, sel);
                   let c = choose<u32:{width}, false, false>(x, sel);
                   let d = choose<u32:1, {first}, {second}>(y, sel);
                   let e = {call};
                   (a, b, c, d, e)"#
            ),
        )
    } else {
        (
            format!("x:{word}, sel:bool"),
            word,
            format!("let r = {call}; r"),
        )
    };
    let bx = format!("{bx_decl}\n{}", bx_source(&params, &result, &body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: function,
            params,
            result,
            body,
            valid: Some(valid),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::check_const_if;

    #[test]
    fn selected_branch_types_and_values() -> Result<(), String> {
        for family in [0, 1, 2, 14, 15, 16, 17, 18, 19] {
            for flags in [0, 3, 6, 9, 12, 31, 33, 52, 65, 84] {
                for (width, cast_width) in [(0, 6), (12, 4), (70, 128), (128, 70)] {
                    check_const_if(&[family, flags, width, cast_width])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn guard_and_arm_errors() -> Result<(), String> {
        for family in 3..12 {
            for flags in [0, 1, 4, 5, 8, 9, 12, 13] {
                check_const_if(&[family, flags, 7])?;
            }
        }
        Ok(())
    }

    #[test]
    fn qualified_members_and_constructor_fields_in_skipped_arms() -> Result<(), String> {
        for flags in [0, 1, 4, 5] {
            check_const_if(&[12, flags, 7])?;
            check_const_if(&[12, flags | 16, 7])?;
            for fields in 0..7 {
                check_const_if(&[13, flags | (fields << 4), 7])?;
            }
        }
        Ok(())
    }
}
