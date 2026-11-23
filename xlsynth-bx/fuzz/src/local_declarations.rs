// SPDX-License-Identifier: Apache-2.0

//! Compare lexical constants and aliases with DSLX and with hoisted BX
//! declarations.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;

/// One concrete input type and the result type of its specialization.
struct Specialization {
    input: String,
    result: String,
    hoisted: String,
}

/// Compare both BX forms on identical inputs with DSLX's evaluation of the
/// local function.
fn compare_forms(
    local: &str,
    native: &str,
    a: Specialization,
    b: Specialization,
    seed: u64,
) -> Result<(), String> {
    let params = format!("a:{}, b:{}, sel:bool", a.input, b.input);
    let result = format!("({0},{0},{1},{1},{0},{0})", a.result, b.result);
    let bx_body = "(apply(a,sel), hoisted_a(a,sel), apply(b,sel), hoisted_b(b,sel), apply(a,!sel), hoisted_a(a,!sel))";
    let declarations = format!("{}\n{}", a.hoisted, b.hoisted);
    let bx = format!(
        "{local}\n{declarations}\n{}",
        bx_source(&params, &result, bx_body)
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            // Expose the hoisted module constants to the declaration-value
            // oracle. Every DSLX output still comes from the original local
            // function.
            functions: format!("{native}\n{declarations}"),
            params,
            result,
            body: r#"let p = apply(a,sel);
let q = apply(b,sel);
let r = apply(a,!sel);
(p,p,q,q,r,r)"#
                .into(),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Bind the width and signedness twice, shadow dependent declarations in a
/// block, and read the outer constant again after leaving the block.
fn value_parameters(
    width: usize,
    signed: bool,
    pad: u8,
    bias: u8,
    seed: u64,
) -> Result<(), String> {
    let local = format!(
        r#"
fn apply<S:bool,N:u32>(x:xN[S][N], sel:bool)->(xN[S][N],u32,u32) {{
    let pad = u32:{pad};
    const W = N + pad;
    type Word = xN[S][W];
    const MASK:Word = N as Word;
    let (inner, inner_width) = {{
        const W = W + u32:1;
        type Word = xN[S][W];
        const MASK:Word = (MASK as Word) + (u32:{bias} as Word);
        let value = if sel {{ ((x as Word) ^ MASK) as xN[S][N] }} else {{ x }};
        (value, bit_count<Word>())
    }};
    (inner + (MASK as xN[S][N]), bit_count<Word>(), inner_width)
}}"#
    );
    let specialize = |label: &str, width: usize, signed: bool| {
        let suffix = label.to_ascii_uppercase();
        let ty = format!("{}N[{width}]", if signed { 's' } else { 'u' });
        let result = format!("({ty},u32,u32)");
        let hoisted = format!(
            r#"
const W_{suffix} = u32:{width} + u32:{pad};
type Word{suffix} = xN[{signed}][W_{suffix}];
const MASK_{suffix}:Word{suffix} = u32:{width} as Word{suffix};
const INNER_W_{suffix} = W_{suffix} + u32:1;
type InnerWord{suffix} = xN[{signed}][INNER_W_{suffix}];
const INNER_MASK_{suffix}:InnerWord{suffix} = (MASK_{suffix} as InnerWord{suffix}) + (u32:{bias} as InnerWord{suffix});
fn hoisted_{label}(x:{ty}, sel:bool)->{result} {{
    let inner = if sel {{ ((x as InnerWord{suffix}) ^ INNER_MASK_{suffix}) as {ty} }} else {{ x }};
    (inner + (MASK_{suffix} as {ty}), bit_count<Word{suffix}>(), bit_count<InnerWord{suffix}>())
}}"#
        );
        Specialization {
            input: ty,
            result,
            hoisted,
        }
    };
    compare_forms(
        &local,
        &local,
        specialize("a", width, signed),
        specialize("b", width + 1, !signed),
        seed,
    )
}

/// Instantiate the same alias chain with distinct aggregate element types and
/// lengths.
fn type_parameters(width: usize, len: usize, array: bool, seed: u64) -> Result<(), String> {
    let body = r#"(xs:T[N], sel:bool)->T {
    type Word = T;
    type Row = (Word,Word);
    const LAST = N - u32:1;
    const EMPTY:Row = zero!<Row>();
    let p:Row = (xs[LAST], EMPTY.0);
    let picked = {
        type Word = Row;
        const EMPTY:Word = zero!<Word>();
        if sel { p.0 } else { EMPTY.1 }
    };
    let restored:Word = picked;
    restored
}"#;
    let local = format!("fn apply<T,N:u32>{body}");
    let native = format!("#![feature(generics)]\nfn apply<T:type,N:u32>{body}");
    let specialize = |label: &str, elem: String, len: usize| {
        let suffix = label.to_ascii_uppercase();
        let input = format!("{elem}[{len}]");
        let hoisted = format!(
            r#"
type Word{suffix} = {elem};
type Row{suffix} = (Word{suffix},Word{suffix});
const LAST_{suffix} = u32:{len} - u32:1;
const EMPTY_{suffix}:Row{suffix} = zero!<Row{suffix}>();
type InnerWord{suffix} = Row{suffix};
const INNER_EMPTY_{suffix}:InnerWord{suffix} = zero!<InnerWord{suffix}>();
fn hoisted_{label}(xs:{input}, sel:bool)->{elem} {{
    let p:Row{suffix} = (xs[LAST_{suffix}], EMPTY_{suffix}.0);
    let restored:Word{suffix} = if sel {{ p.0 }} else {{ INNER_EMPTY_{suffix}.1 }};
    restored
}}"#
        );
        Specialization {
            input,
            result: elem,
            hoisted,
        }
    };
    let agg = if array {
        format!("sN[{width}][2]")
    } else {
        format!("(sN[{width}],bool)")
    };
    compare_forms(
        &local,
        &native,
        specialize("a", format!("uN[{width}]"), len),
        specialize("b", agg, len + 1),
        seed,
    )
}

/// Read same-named module and function constants in successive initializers,
/// then independently shadow the outer scope in each arm of a runtime
/// conditional.
fn branch_scopes(width: usize, bias: u8, seed: u64) -> Result<(), String> {
    let ty = format!("uN[{width}]");
    let module = format!("const MASK = u32:{bias} as {ty};\ntype Word = {ty};");
    let body = |then_value: &str, else_value: &str, outer: &str| {
        format!(
            "let v = if sel {{ {then_value} }} else {{ {else_value} }};\n\
             match if sel {{ {outer} }} else {{ x }} {{ {outer} => v - {outer}, _ => v + {outer} }}"
        )
    };
    let local_body = body(
        &format!("type Word=u32; const MASK:Word=(MASK as Word) + u32:1; x ^ (MASK as {ty})"),
        &format!("type Word=u16; const MASK:Word=(MASK as Word) + u16:2; x + (MASK as {ty})"),
        "MASK",
    );
    let delta = if seed & 1 == 0 { "one()" } else { "Word:1" };
    let local = format!(
        r#"{module}
fn one() -> Word {{ Word:1 }}
fn apply(x:{ty}, sel:bool)->{ty} {{
    let step = {delta};
    const MASK:Word=MASK + step;
    {local_body}
}}"#
    );
    let specialize = |label: &str| {
        let suffix = label.to_ascii_uppercase();
        let hoisted_body = body(
            &format!("x ^ (THEN_{suffix} as {ty})"),
            &format!("x + (ELSE_{suffix} as {ty})"),
            &format!("OUTER_{suffix}"),
        );
        let hoisted = format!(
            r#"
const OUTER_{suffix}:Word = MASK + Word:1;
type ThenWord{suffix} = u32;
const THEN_{suffix}:ThenWord{suffix} = (OUTER_{suffix} as ThenWord{suffix}) + u32:1;
type ElseWord{suffix} = u16;
const ELSE_{suffix}:ElseWord{suffix} = (OUTER_{suffix} as ElseWord{suffix}) + u16:2;
fn hoisted_{label}(x:{ty}, sel:bool)->{ty} {{ {hoisted_body} }}"#
        );
        Specialization {
            input: ty.clone(),
            result: ty.clone(),
            hoisted,
        }
    };
    compare_forms(&local, &local, specialize("a"), specialize("b"), seed)
}

/// Check failures at the declaration and at references crossing a lexical
/// boundary.
fn rejected(mutation: u8, seed: u64) -> Result<(), String> {
    let body = match mutation % 14 {
        0 => "const BAD=x; x",
        1 => "let runtime=x; const BAD=runtime; BAD",
        2 => "type Word=uN[x as u32]; (x as Word) as u8",
        3 => "const EARLY=LATE; const LATE=u8:1; EARLY",
        4 => "type Early=Late; type Late=u8; x as Early",
        5 => "let y={const K=u8:2; x^K}; y^K",
        6 => "let y={type T=u8; x as T}; y as T",
        7 => "type Word=u16; const K:Word=u8:1; x + (K as u8)",
        8 => "const K=u8:1; let y={type K=u8; x + K}; y",
        9 => "type T=u8; let y={const T=u8:1; let z:T=x; z}; y",
        10 => "const W=true; type T=uN[W]; x ^ (T:1 as u8)",
        11 => "if true { x } else { const BAD=x; BAD }",
        12 => "const BAD={ x; u8:1 }; BAD",
        _ => "let (a,_)=(u8:4,x); const BAD=a; BAD",
    };
    compare_program(
        FunctionProgram {
            functions: format!("fn apply(x:u8)->u8 {{ {body} }}"),
            params: "x:u8".into(),
            result: "u8".into(),
            body: "apply(x)".into(),
            valid: Some(false),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

/// Generate declarations, specializing the hoisted reference separately for
/// each call type.
pub fn check_local_declarations(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 4;
    let width = 1 + usize::from(bytes.next().unwrap() % 128);
    let flags = bytes.next().unwrap();
    let size = 1 + bytes.next().unwrap() % 4;
    let bias = bytes.next().unwrap();
    let seed = input_seed(data);
    match family {
        0 => value_parameters(width, flags & 1 != 0, size, bias, seed),
        1 => type_parameters(width, usize::from(size), flags & 1 != 0, seed),
        2 => branch_scopes(width, bias, seed),
        _ => rejected(flags, seed),
    }
}

#[cfg(test)]
mod tests {
    use super::check_local_declarations;

    fn compare_family(family: u8) -> Result<(), String> {
        for flags in 0..8 {
            for width in [0, 9, 69, 127] {
                check_local_declarations(&[family, width, flags, flags % 4, 137])?;
            }
        }
        Ok(())
    }

    #[test]
    fn value_parameter_declarations() -> Result<(), String> {
        compare_family(0)
    }

    #[test]
    fn type_parameter_declarations() -> Result<(), String> {
        compare_family(1)
    }

    #[test]
    fn shadowed_branch_declarations() -> Result<(), String> {
        compare_family(2)
    }

    #[test]
    fn rejected_declarations() -> Result<(), String> {
        for mutation in 0..14 {
            check_local_declarations(&[3, 0, mutation])?;
        }
        Ok(())
    }
}
