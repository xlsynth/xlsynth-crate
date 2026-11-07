// SPDX-License-Identifier: Apache-2.0

//! Compare the inferred types and runtime results of DSLX-style counted folds.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;
use xlsynth_pir::{IrBits, IrFormatPreference};

/// One generated range or array and the expression that consumes its element.
struct Iterable {
    bindings: String,
    ty: String,
    binder: &'static str,
    value: String,
    term: String,
    array_params: String,
    // The pinned IR converter narrows nonzero range bases to u64. Its bytecode
    // evaluator supports negative and wider bases without that restriction.
    native_ir: bool,
}

/// Choose a range or runtime array, preserving the actual element type.
fn iterable(kind: u8, ty: &str, start: u8, count: u8, inclusive: bool) -> Iterable {
    match kind {
        0..=2 => {
            let (index_ty, base, native_ir) = match kind {
                0 => ("u32", i128::from(start), true),
                1 => ("s16", -i128::from(start), start == 0),
                _ => ("uN[96]", (1i128 << 64) + i128::from(start), false),
            };
            let limit = base + i128::from(count);
            let dots = if inclusive { "..=" } else { ".." };
            let term = if kind == 2 {
                format!("((item ^ (item >> u32:64)) as {ty})")
            } else {
                format!("(item as {ty})")
            };
            Iterable {
                bindings: String::new(),
                ty: index_ty.into(),
                binder: "item",
                value: format!("{index_ty}:{base}{dots}{index_ty}:{limit}"),
                term,
                array_params: String::new(),
                native_ir,
            }
        }
        _ => {
            let (element, binder, term) = if kind == 3 {
                (ty.to_owned(), "item", "item".to_owned())
            } else {
                (
                    format!("({ty}, u8)"),
                    "(item, tag)",
                    format!("(item ^ (tag as {ty}))"),
                )
            };
            // Native IR values require an element for array inputs. Empty
            // iteration is sampled by the range cases.
            let length = count.max(1);
            Iterable {
                bindings: String::new(),
                ty: element.clone(),
                binder,
                value: "if pick { left } else { right }".into(),
                term,
                array_params: format!(
                    ", left: {element}[{length}], right: {element}[{length}], pick: bool"
                ),
                native_ir: true,
            }
        }
    }
}

/// Generate a named or scoped array binding with a mixture of implicit and
/// explicitly typed literals. A lone untyped bit pattern occupies the whole
/// annotated width; a multi-element array has an explicitly typed sibling.
fn literal_iterable(
    scoped: bool,
    ty: &str,
    width: usize,
    count: u8,
    spelling: u8,
    payload: &[u8],
) -> Result<Iterable, String> {
    let length = usize::from(count.max(1));
    let anchor = (length > 1 || spelling & 1 != 0).then(|| usize::from(spelling) % length);
    let mut bytes = payload.iter().copied().cycle().chain(std::iter::repeat(0));
    let mut elements = Vec::with_capacity(length);
    for i in 0..length {
        let mut value = (&mut bytes).take(width.div_ceil(8)).collect::<Vec<_>>();
        let high_bit = (width - 1) % 8;
        let high = &mut value[(width - 1) / 8];
        *high &= u8::MAX >> (7 - high_bit);
        if anchor.is_none() {
            *high |= 1 << high_bit;
        } else if ty.starts_with('s') && Some(i) != anchor {
            // Multiple implicit prefixed literals can acquire a combined
            // positive minimum before the signed sibling supplies its type.
            *high &= !(1 << high_bit);
        }
        let bits = IrBits::from_le_bytes(width, &value).map_err(|e| e.to_string())?;
        let format = match (usize::from(spelling >> 1) + i) % 3 {
            0 => IrFormatPreference::Binary,
            1 if anchor.is_some() && ty.starts_with('s') => IrFormatPreference::SignedDecimal,
            1 if anchor.is_some() => IrFormatPreference::UnsignedDecimal,
            _ => IrFormatPreference::Hex,
        };
        let digits = bits.to_string_fmt(format, false);
        elements.push(if anchor == Some(i) {
            format!("{ty}:{digits}")
        } else {
            digits
        });
    }
    let definition = format!("let items = [{}];", elements.join(", "));
    let (bindings, value) = if scoped {
        (String::new(), format!("{{ {definition} items }}"))
    } else {
        (definition, "items".into())
    };
    // For an implicit prefixed literal, pinned bytecode keeps an unsigned tag
    // on the bound array item even after the loop annotation selects signed
    // bits. The identity cast preserves its bit pattern and source type.
    let term = if anchor.is_none() && ty.starts_with('s') {
        format!("(item as {ty})")
    } else {
        "item".into()
    };
    Ok(Iterable {
        bindings,
        ty: ty.to_owned(),
        binder: "item",
        value,
        term,
        array_params: String::new(),
        native_ir: true,
    })
}

/// Generate independent iterable, carry, binder and capture choices; compare
/// typecheck acceptance, then evaluate accepted programs on concrete inputs.
pub fn check_counted_for(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let kind = bytes.next().unwrap() % 7;
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap()) % 129 + 1;
    let sign = if flags & 1 == 0 { "u" } else { "s" };
    let ty = format!("{sign}N[{width}]");
    let start = bytes.next().unwrap() % 8;
    let count = bytes.next().unwrap() % 6;
    let mut input = if kind < 5 {
        iterable(kind, &ty, start, count, flags & 16 != 0)
    } else {
        let spelling = data.get(7).copied().unwrap_or(0);
        literal_iterable(
            kind == 6,
            &ty,
            width,
            count,
            spelling,
            data.get(8..).unwrap_or_default(),
        )?
    };
    let params = format!("seed: {ty}, bias: {ty}{}", input.array_params);
    let (carry, binder, seed) = if flags & 2 == 0 {
        (ty.clone(), "acc", "seed")
    } else {
        (format!("({ty}, {ty})"), "(acc, prev)", "(seed, bias)")
    };
    let mut annotation = if kind >= 5 || flags & 4 != 0 {
        format!(": ({}, {carry})", input.ty)
    } else {
        String::new()
    };
    let term = input.term;
    let step = match bytes.next().unwrap() % 3 {
        0 => format!("acc + {term}"),
        1 => format!("(acc - {term}) ^ bias"),
        _ => format!(
            "for (k, partial) in u8:0..u8:2 {{ partial ^ ((k as {ty}) + {term} + bias) }}(acc)"
        ),
    };
    let mut body = if flags & 2 == 0 {
        step
    } else {
        format!("({step}, prev + acc)")
    };
    let mutation = bytes.next().unwrap() % 12;
    match mutation {
        8 => annotation = format!(": ({}, uN[{}])", input.ty, width + 1),
        9 => body = "missing".into(),
        10 => input.value = "bias".into(),
        11 => input.value = "u32:0..(bias as u32)".into(),
        _ => { /* This sample has a well-typed body, annotation and iterable. */ }
    }
    let expression = format!(
        "for ({}, {binder}){annotation} in {} {{ {body} }}({seed})",
        input.binder, input.value
    );
    let expression = match (flags >> 5) & 3 {
        0 => expression,
        1 => {
            format!("if seed as bool {{ {expression} }} else {{ {seed} }}")
        }
        2 => format!(
            "match seed as bool {{ true => {{ let bias = bias ^ seed; {expression} }}, _ => {seed} }}"
        ),
        _ => format!("let answer: {carry} = {expression}; answer"),
    };
    let expression = format!("{}\n{expression}", input.bindings);
    let (functions, body) = if flags & 8 == 0 {
        (String::new(), expression)
    } else {
        let args = if input.array_params.is_empty() {
            "seed, bias"
        } else {
            "seed, bias, left, right, pick"
        };
        (
            format!("fn fold({params}) -> {carry} {{ {expression} }}"),
            format!("fold({args})"),
        )
    };
    compare_program(
        FunctionProgram {
            functions,
            params,
            result: carry,
            body,
            valid: Some(mutation < 8),
            dslx_ir: input.native_ir,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_counted_for;

    #[test]
    fn compare_generated_counted_folds() -> Result<(), String> {
        for kind in 0..7 {
            for flags in 0..32 {
                check_counted_for(&[
                    kind,
                    flags | ((flags & 3) << 5),
                    flags * 4,
                    3,
                    flags % 6,
                    flags % 3,
                    0,
                    flags / 3,
                    flags.wrapping_mul(37),
                ])?;
            }
            for mutation in 8..12 {
                check_counted_for(&[kind, 3, 7, 2, 0, 1, mutation])?;
            }
        }
        Ok(())
    }
}
