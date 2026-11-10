// SPDX-License-Identifier: Apache-2.0

//! Compare bits/array cast checking, constant values and lowered hardware with
//! DSLX.

use crate::eval::expr_eval::{ExpressionEvaluator, bx_source, compile_bundled_dslx, dslx_source};
use crate::function_oracle::{
    DslxExecution, FunctionProgram, check_sources, compare_program, dslx_literal,
};
use crate::input_seed;
use xlsynth_bx::ast::Type;
use xlsynth_bx::{check_module, parse_module};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::{IrBits, IrValue};

/// Generate a flat array and a scalar with the same packed width, or mutate
/// the width or element shape to exercise the cast's acceptance boundary.
pub fn check_bits_array_casts(data: &[u8]) -> Result<(), String> {
    let [mode, flags, width_byte, count_byte] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let mode = mode % 16;
    let pack = flags & 1 != 0;
    let width = match (mode, flags & 8 != 0) {
        (6, _) => 0,
        (_, true) => 1,
        _ => usize::from(width_byte % 96) + 1,
    };
    let count = if mode == 7 {
        0
    } else {
        usize::from(count_byte % 7) + 1
    };
    let total = width * count;
    let element = if width == 1 && flags & 8 != 0 {
        Type::Bool
    } else {
        Type::Bits {
            signed: flags & 2 != 0,
            width,
        }
    };
    let scalar = if total == 1 && flags & 16 != 0 {
        Type::Bool
    } else {
        Type::Bits {
            signed: flags & 4 != 0,
            width: total,
        }
    };
    if mode >= 9 {
        return check_invalid_cast(mode, pack, width, count, &element, &scalar);
    }
    let array = Type::Array {
        element: Box::new(element.clone()),
        size: count,
    };
    let seed = input_seed(data);
    if count == 0 {
        // The native IR interpreter cannot construct empty arrays. DSLX
        // constexpr execution can, so compare both casts through its result.
        return compare_program(
            FunctionProgram {
                functions: format!("const PACKED = ({scalar}:0 as {array}) as {scalar};"),
                params: format!("x: {scalar}"),
                body: format!("(x as {array}) as {scalar}"),
                result: scalar.to_string(),
                valid: Some(true),
                dslx_ir: false,
                dynamic_index: None,
            },
            seed,
            DslxExecution::ConcreteConstants,
            Vec::new(),
        );
    }

    let (source, target) = if pack {
        (&array, &scalar)
    } else {
        (&scalar, &array)
    };
    let input = asymmetric_input(pack, width, count, data.get(4..).unwrap_or_default())?;
    let literal = dslx_literal(source, &input)?;
    let constants = format!(
        "const SOURCE: {source} = {literal};\nconst RESULT: {target} = SOURCE as {target};"
    );
    let params = format!("x: {source}");
    let body = format!("x as {target}");
    if matches!(element, Type::Bits { signed: true, .. })
        || matches!(scalar, Type::Bits { signed: true, .. })
    {
        // DSLX bytecode cannot flatten signed array values, and an array
        // cast tags its output chunks or packed scalar unsigned. Compare
        // the signed result and its use against compiled DSLX on the exact
        // same input, including for BX's constant expression.
        let signed_probe = if pack && matches!(scalar, Type::Bits { signed: true, .. }) {
            Some(("", &scalar))
        } else if !pack && matches!(element, Type::Bits { signed: true, .. }) {
            Some(("[u32:0]", &element))
        } else {
            None
        };
        let (result, body, declarations) = match signed_probe {
            Some((index, ty)) => (
                format!("({target}, bool)"),
                format!("let v = {body}; (v, v{index} < {ty}:0)"),
                format!("{constants}\nconst OBSERVED = (RESULT, RESULT{index} < {ty}:0);"),
            ),
            None => (target.to_string(), body, constants),
        };
        let evaluator =
            ExpressionEvaluator::compile(&params, &result, &body, compile_bundled_dslx)?;
        let expected = evaluator.compare(&[input])?;
        let checked = check_module(&parse_module(&declarations)?)?;
        let actual = &checked
            .constants()
            .last()
            .ok_or("missing BX cast result")?
            .value;
        if *actual != expected {
            return Err(format!(
                "BX constant returned {actual}, DSLX IR returned {expected}\n{declarations}"
            ));
        }
        for args in generate_argument_sets_from_seed(&evaluator.bx.function, seed, 4) {
            evaluator.compare(&args)?;
        }
        Ok(())
    } else {
        compare_program(
            FunctionProgram {
                functions: constants,
                params,
                body,
                result: target.to_string(),
                valid: Some(true),
                dslx_ir: true,
                dynamic_index: None,
            },
            seed,
            DslxExecution::ConcreteConstants,
            vec![vec![input]],
        )
    }
}

/// Generate a reproducible value whose outer elements differ even when the
/// payload is all zeroes, so reversal remains observable in either direction.
fn asymmetric_input(
    pack: bool,
    width: usize,
    count: usize,
    entropy: &[u8],
) -> Result<IrValue, String> {
    let bit = |lane: usize, offset: usize| {
        if offset + 1 == width && (lane == 0 || lane + 1 == count) {
            return lane == 0;
        }
        let i = lane * width + offset;
        !entropy.is_empty() && entropy[i / 8 % entropy.len()] & (1 << (i % 8)) != 0
    };
    if pack {
        let words = (0..count)
            .map(|lane| IrValue::Bits(IrBits::from_lsb_fn(width, |i| bit(lane, i))))
            .collect::<Vec<_>>();
        IrValue::make_array(&words).map_err(|e| e.to_string())
    } else {
        Ok(IrValue::Bits(IrBits::from_lsb_fn(width * count, |i| {
            bit(i / width, i % width)
        })))
    }
}

/// Compare rejection by both typecheckers. The nested and nominal shapes
/// have matching flat widths to distinguish shape errors from width errors.
fn check_invalid_cast(
    mode: u8,
    pack: bool,
    width: usize,
    count: usize,
    element: &Type,
    scalar: &Type,
) -> Result<(), String> {
    let total = width * count;
    let array = format!("{element}[{count}]");
    let mut declarations = String::new();
    let (array, scalar) = match mode {
        9 => (array, format!("uN[{}]", total + 1)),
        10 => (array, format!("uN[{}]", total - 1)),
        11 => (format!("{element}[1][{count}]"), scalar.to_string()),
        12 => (format!("({element},)[{count}]"), scalar.to_string()),
        13 => {
            declarations = format!("struct Packet {{ value: {element} }}");
            (format!("Packet[{count}]"), scalar.to_string())
        }
        14 => {
            declarations = format!("enum Word : uN[{width}] {{ ZERO = 0 }}");
            (format!("Word[{count}]"), scalar.to_string())
        }
        _ => {
            declarations = format!("enum Packed : uN[{total}] {{ ZERO = 0 }}");
            (array, "Packed".into())
        }
    };
    let (source, target) = if pack {
        (array, scalar)
    } else {
        (scalar, array)
    };
    let params = format!("x: {source}");
    let body = format!("x as {target}");
    let bx = bx_source(&params, &target, &body);
    let native = dslx_source(&params, &target, &body);
    check_sources(
        &format!("{declarations}\n{bx}"),
        &format!("{declarations}\n{native}"),
        Some(false),
        &[],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_bits_array_casts;

    #[test]
    fn compare_bits_array_cast_families() -> Result<(), String> {
        for mode in [0, 6, 7, 9, 10, 11, 12, 13, 14, 15] {
            for flags in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 24, 25] {
                check_bits_array_casts(&[mode, flags, 67, 3, 0xb9, 0x0e])?;
            }
        }
        for flags in [24, 25] {
            check_bits_array_casts(&[0, flags, 0])?;
        }
        Ok(())
    }
}
