// SPDX-License-Identifier: Apache-2.0

//! Compare array-valued range expressions with the pinned DSLX frontend.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;
use num_bigint::BigInt;

/// A small ordered interval in the mathematical domain of its endpoint type.
struct Bounds {
    word: String,
    start: BigInt,
    end: BigInt,
    length: usize,
    dots: &'static str,
}

impl Bounds {
    /// Sample at both limits, across zero, and around the host integer boundary
    /// without ever generating an enormous valid array.
    fn new(width: usize, signed: bool, inclusive: bool, anchor: u8, distance: u8) -> Self {
        let one = BigInt::from(1);
        let extent = &one << width;
        let sign_bit = if signed && width > 0 {
            &extent >> 1
        } else {
            BigInt::from(0)
        };
        let min = -&sign_bit;
        let max: BigInt = &extent - &sign_bit - 1;
        let distance = usize::from(distance % 9).min(if width < 4 { (1 << width) - 1 } else { 8 });
        let last_start = &max - distance;
        let start = match anchor % 6 {
            0 => min.clone(),
            1 => last_start.clone(),
            2 => -BigInt::from(distance / 2),
            3 => BigInt::from(anchor / 6),
            4 => (&one << 64) - usize::from(anchor / 6),
            _ => -(&one << 64) - usize::from(anchor / 6),
        }
        .clamp(min, last_start);
        Self {
            word: format!("{}N[{width}]", if signed { 's' } else { 'u' }),
            end: &start + distance,
            start,
            length: distance + usize::from(inclusive),
            dots: if inclusive { "..=" } else { ".." },
        }
    }
}

/// Compare acceptance, the full BX result, and the independent DSLX executions.
pub fn check_array_ranges(data: &[u8]) -> Result<(), String> {
    let [family, flags, width, anchor, distance, variant] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let width = usize::from(width % 130);
    let bounds = Bounds::new(width, flags & 1 != 0, flags & 2 != 0, anchor, distance);
    let word = &bounds.word;
    let dots = bounds.dots;
    let length = bounds.length;
    let start = format!("{word}:{}", bounds.start);
    let end = format!("{word}:{}", bounds.end);
    let range = format!("{start}{dots}{end}");
    let array = format!("{word}[{length}]");
    let family = family % 12;
    let mut functions = String::new();
    let mut result = array.clone();
    let mut valid = true;
    let body = match family {
        0 => range,
        1 => format!("let lanes = {range}; lanes"),
        2 => {
            functions = format!("fn take(lanes: {array}) -> {array} {{ lanes }}");
            format!("take({range})")
        }
        3 if length > 0 => {
            functions = format!("const LANES = {range};");
            "LANES".into()
        }
        3 => {
            // The native API cannot convert an empty constexpr array into IR
            // for declaration comparison. Observe it through its DSLX size.
            functions = format!("const SIZE = array_size({range});");
            result = format!("(u32, {array})");
            format!("(SIZE, {range})")
        }
        4 => {
            functions = format!(
                "const FIRST = {start};\nfn finish() -> {word} {{ {end} }}\nfn lanes() -> {array} {{ (FIRST ^ {word}:0){dots}finish() }}"
            );
            "lanes()".into()
        }
        5 if length > 0 => {
            result = word.clone();
            format!("({range})[index % u32:{length}]")
        }
        5 => {
            result = "u32".into();
            format!("array_size({range})")
        }
        6 => {
            // The specializations share their function body and can have
            // different bounds even when their result arrays have the same
            // type.
            functions =
                format!("fn lanes<A: {word}, B: {word}, N: u32>() -> {word}[N] {{ A{dots}B }}");
            let singleton_length = usize::from(flags & 2 != 0);
            result = format!("({array}, {word}[{singleton_length}])");
            format!(
                "(lanes<{start}, {end}, u32:{length}>(), lanes<{end}, {end}, u32:{singleton_length}>())"
            )
        }
        7 => {
            result = format!("{array}[2]");
            format!("[{range}, {range}]")
        }
        8 => {
            valid = false;
            match variant % 5 {
                0 => format!("(index as {word}){dots}{end}"),
                1 => format!("{start}{dots}(index as {word})"),
                2 => format!("{start}{dots}uN[{}]:0", width + 1),
                3 => format!("({start}, true){dots}({end}, false)"),
                _ => {
                    result = format!("{word}[{}]", length + 1);
                    range
                }
            }
        }
        10 => {
            // The declared result sizes a direct range's unannotated bounds.
            // Keep other families explicitly typed for independent IR
            // execution.
            // An unannotated decimal zero needs a sign bit when inferred as
            // signed. Explicit sN[0]:0 still denotes its sole value.
            valid = width != 0 || flags & 1 == 0 || variant & 3 == 3;
            let first = if variant & 1 == 0 {
                bounds.start.to_string()
            } else {
                start
            };
            let last = if variant & 2 == 0 {
                bounds.end.to_string()
            } else {
                end
            };
            format!("{first}{dots}{last}")
        }
        11 => {
            // A signed result can type each hex bound directly. At a call
            // site, distinct bounds unify first and lose their radix. Two
            // references to the same bound retain its literal constraint.
            let width = width.max(1);
            let sign_bit = BigInt::from(1) << (width - 1);
            let literal = format!("{:#x}", &sign_bit + BigInt::from(anchor) % &sign_bit);
            result = format!("sN[{width}][1]");
            functions = format!("fn take(xs: {result}) -> {result} {{ xs }}");
            match variant % 3 {
                0 => format!("{literal}..={literal}"),
                1 => format!("let h = {literal}; take(h..=h)"),
                _ => {
                    valid = false;
                    format!("take({literal}..={literal})")
                }
            }
        }
        _ => {
            valid = false;
            result = "u32".into();
            let invalid = match variant % 2 {
                0 => format!("uN[129]:0x10000000000000000{dots}uN[129]:0xffffffffffffffff"),
                _ => format!("s8:0{dots}s8:-1"),
            };
            format!("array_size({invalid})")
        }
    };
    // The pinned converter cannot construct a literal for any empty range.
    let empty = family != 11 && (length == 0 || family == 6 && flags & 2 == 0);
    compare_program(
        FunctionProgram {
            functions,
            params: "index: u32".into(),
            result,
            body,
            valid: Some(valid),
            dslx_ir: !empty,
            dynamic_index: None,
        },
        input_seed(data),
        if empty {
            DslxExecution::BytecodeTests
        } else {
            DslxExecution::ConcreteConstants
        },
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_array_ranges;

    #[test]
    fn compare_array_range_families() -> Result<(), String> {
        for family in (0..8).chain([10]) {
            for (flags, width, anchor, distance) in [
                (0, 0, 0, 0),   // Empty at zero width.
                (3, 1, 0, 1),   // Entire signed one-bit domain.
                (2, 3, 1, 8),   // Inclusive maximum.
                (1, 8, 2, 6),   // Signed, crossing zero.
                (2, 96, 4, 5),  // Carries across bit 63.
                (1, 129, 0, 4), // Below i128::MIN.
            ] {
                check_array_ranges(&[family, flags, width, anchor, distance, 0])?;
            }
        }
        for width in [1, 8, 129] {
            for variant in 0..3 {
                check_array_ranges(&[11, 0, width, 37, 1, variant])?;
            }
        }
        for variant in 0..4 {
            check_array_ranges(&[10, 3, 0, 0, 0, variant])?;
        }
        for family in [8, 9] {
            for variant in 0..5 {
                check_array_ranges(&[family, 1, 96, 3, 4, variant])?;
            }
        }
        Ok(())
    }
}
