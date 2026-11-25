// SPDX-License-Identifier: Apache-2.0

//! Compare structural array folds with DSLX and the BX counted-loop lowering.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

const BX_PRODUCT: &str = include_str!("../../examples/grid.bx");
const DSLX_PRODUCT: &str = r#"
fn product<X: u32, Y: u32>() -> (u32, u32)[X * Y] {
    for (x, rows) in u32:0..X {
        for (y, pairs) in u32:0..Y {
            update(pairs, x * Y + y, (x, y))
        }(rows)
    }(zero!<(u32, u32)[X * Y]>())
}"#;
const COORD: &str = "fn coord<X: u32, Y: u32>() -> u32 { X * u32:17 + Y }";

/// An iterable expression and the contribution of each bound element to the
/// carry.
struct FoldInput {
    params: String,
    element: String,
    binder: &'static str,
    iterable: String,
    term: String,
    counted_term: String,
}

/// Sample runtime scalars, destructured nested tuples, and empty arrays.
fn runtime_array(word: &str, length: u8, tuple: bool, pair_name: bool) -> FoldInput {
    let (element, binder, term) = if tuple {
        let element = format!("(({word}, u8), {word})");
        if pair_name {
            (
                element,
                "item",
                format!("item.0.0 + (item.0.1 as {word}) - item.1"),
            )
        } else {
            (element, "((a, tag), b)", format!("a + (tag as {word}) - b"))
        }
    } else {
        (word.to_owned(), "item", "item".into())
    };
    // The native IR value API needs an element to construct a concrete array
    // argument. Generate the typed empty array in the function instead.
    let (params, iterable) = if length == 0 {
        (String::new(), format!("zero!<{element}[0]>()"))
    } else {
        (
            format!(", left: {element}[{length}], right: {element}[{length}], pick: bool"),
            "if pick { left } else { right }".into(),
        )
    };
    FoldInput {
        params,
        element,
        binder,
        iterable,
        counted_term: term.clone(),
        term,
    }
}

/// Read runtime data using coordinates produced by a pure parametric helper.
fn product(word: &str, rows: u8, columns: u8, parametric: bool) -> FoldInput {
    let length = (rows * columns).max(1);
    let data = format!("words[x * u32:{columns} + y]");
    let ordinary = "x * u32:17 + y";
    let coordinate = if parametric {
        "coord<x, y>()"
    } else {
        ordinary
    };
    FoldInput {
        params: format!(", words: {word}[{length}]"),
        element: "(u32, u32)".into(),
        binder: "(x, y)",
        iterable: format!("product<u32:{rows}, u32:{columns}>()"),
        term: format!("{data} ^ (({coordinate}) as {word})"),
        // A counted loop binds runtime coordinates, so evaluate the same
        // coordinate expression directly in that loop.
        counted_term: format!("{data} ^ (({ordinary}) as {word})"),
    }
}

/// Compare BX unrolled and counted results against the same DSLX structural
/// fold.
pub fn check_unroll_arrays(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let kind = bytes.next().unwrap() % 3;
    let flags = bytes.next().unwrap();
    let width = u16::from(bytes.next().unwrap() % 129) + 1;
    let sign = if flags & 1 == 0 { "u" } else { "s" };
    let word = format!("{sign}N[{width}]");
    let rows = bytes.next().unwrap() % 5;
    let columns = bytes.next().unwrap() % 5;
    let input = if kind == 2 {
        product(&word, rows, columns, flags & 2 != 0)
    } else {
        runtime_array(&word, rows, kind == 1, flags & 2 != 0)
    };
    let (carry, acc, seed) = if flags & 4 == 0 {
        (word.clone(), "acc", "seed")
    } else {
        (
            format!("({word}, {word})"),
            "(acc, previous)",
            "(seed, bias)",
        )
    };
    let annotation = if flags & 8 == 0 {
        String::new()
    } else {
        format!(": ({}, {carry})", input.element)
    };
    // With an explicit loop annotation, pinned DSLX needs the nonempty
    // product's synthesized array value to have its own type annotation.
    // Direct iteration also lets its converter elide empty array values.
    let (bindings, iterable) = if kind == 2 && rows * columns != 0 && !annotation.is_empty() {
        (
            format!(
                "let pairs: (u32, u32)[{}] = {};",
                rows * columns,
                input.iterable
            ),
            "pairs",
        )
    } else {
        (String::new(), input.iterable.as_str())
    };
    let fold = |keyword: &str, term: &str| {
        let step = format!("(acc * (u32:3 as {word})) - ({term}) + bias");
        let body = if flags & 4 == 0 {
            step
        } else {
            format!("({step}, acc ^ previous)")
        };
        format!(
            "{keyword} ({}, {acc}){annotation} in {iterable} {{ {body} }}({seed})",
            input.binder
        )
    };
    let unrolled = fold("unroll_for!", &input.term);
    let counted = fold("for", &input.counted_term);
    let params = format!("seed: {word}, bias: {word}{}", input.params);
    let result = format!("({carry}, {carry})");
    let bx_body = format!("{bindings} let u = {unrolled}; let c = {counted}; (u, c)");
    // Test the example's flattened grid against an independent nested
    // traversal.
    let (bx_helpers, dslx_helpers) = if kind == 2 {
        (
            format!("{BX_PRODUCT}\n{COORD}"),
            format!("{DSLX_PRODUCT}\n{COORD}"),
        )
    } else {
        (String::new(), String::new())
    };
    let bx = format!("{bx_helpers}\n{}", bx_source(&params, &result, &bx_body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: dslx_helpers,
            params,
            result,
            body: format!("{bindings} let u = {unrolled}; (u, u)"),
            valid: Some(true),
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
    use super::check_unroll_arrays;

    #[test]
    fn compare_array_folds() -> Result<(), String> {
        for kind in 0..3 {
            for (rows, cols) in [(0, 3), (2, 0), (1, 1), (2, 3), (3, 2)] {
                for flags in [0, 2, 5, 13, 15] {
                    check_unroll_arrays(&[kind, flags, flags * 8, rows, cols, 0x5b])?;
                }
            }
        }
        Ok(())
    }
}
