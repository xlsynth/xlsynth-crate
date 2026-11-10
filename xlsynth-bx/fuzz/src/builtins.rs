// SPDX-License-Identifier: Apache-2.0

//! Generate calls to pure builtins and compare their source and IR semantics.

use std::fmt::Write;

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, dslx_literal};
use xlsynth_bx::ast::Type::{self, Array, Bits, Bool, Tuple};
use xlsynth_pir::{IrBits, IrValue};

const ARGUMENT_NAMES: [&str; 4] = ["a", "b", "c", "d"];

const BUILTINS: &[&str] = &[
    "rev",
    "clz",
    "ctz",
    "and_reduce",
    "or_reduce",
    "xor_reduce",
    "encode",
    "decode",
    "one_hot",
    "one_hot_sel",
    "priority_sel",
    "gate!",
    "bit_slice_update",
    "signex",
    "smulp",
    "umulp",
    "array_rev",
    "array_size",
    "array_slice",
    "zip",
    "update",
    "all_ones!",
    "bit_count",
    "element_count",
    "checked_cast",
    "widening_cast",
    "ceillog2",
    "configured_value_or",
];

/// A concrete builtin signature and the inputs that exercise its boundaries.
struct BuiltinCase {
    params: Vec<Type>,
    result: Type,
    expression: String,
    reference_expression: Option<String>,
    inputs: Vec<Vec<IrValue>>,
    dslx_ir: bool,
    constant_only: bool,
}

impl BuiltinCase {
    /// Build a call and occasionally add a spurious value argument.
    fn new(
        name: &str,
        parametrics: &str,
        args: &str,
        params: Vec<Type>,
        result: Type,
        invalid: bool,
    ) -> Self {
        let extra = match (invalid, args.is_empty()) {
            (false, _) => "",
            (true, true) => "()",
            (true, false) => ", ()",
        };
        Self {
            params,
            result,
            expression: format!("{name}{parametrics}({args}{extra})"),
            reference_expression: None,
            inputs: Vec::new(),
            dslx_ir: true,
            constant_only: false,
        }
    }

    /// Pin one argument to boundary patterns while varying the other inputs.
    fn sample_argument(&mut self, arg: usize, values: Vec<IrBits>, seed: u64) {
        for (i, value) in values.into_iter().enumerate() {
            let mut args = self.patterned_args(seed.rotate_left(i as u32));
            args[arg] = IrValue::Bits(value);
            self.inputs.push(args);
        }
    }

    fn patterned_args(&self, seed: u64) -> Vec<IrValue> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, ty)| patterned_value(ty, seed.wrapping_add(i as u64)))
            .collect()
    }

    /// Check a concrete constant and the runtime call on the same input domain.
    fn compare(self, invalid: bool, helper: bool, seed: u64) -> Result<(), String> {
        let params = ARGUMENT_NAMES
            .iter()
            .zip(&self.params)
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut bindings = String::new();
        if !invalid {
            let args = self
                .inputs
                .first()
                .cloned()
                .unwrap_or_else(|| self.patterned_args(seed));
            for ((name, ty), value) in ARGUMENT_NAMES.iter().zip(&self.params).zip(args) {
                let value = dslx_literal(ty, &value)?;
                writeln!(bindings, "let {name}: {ty} = {value};").unwrap();
            }
        }
        let result = self.result.to_string();
        let source_parts = |expr: &str| {
            let mut declarations = String::new();
            if !invalid {
                writeln!(declarations, "const PROBE = {{\n{bindings}{expr}\n}};").unwrap();
            }
            let body = if self.constant_only && !invalid {
                "PROBE".to_owned()
            } else if helper {
                let args = ARGUMENT_NAMES[..self.params.len()].join(", ");
                writeln!(declarations, "fn probe({params}) -> {result} {{ {expr} }}").unwrap();
                format!("probe({args})")
            } else {
                expr.to_owned()
            };
            (declarations, body)
        };
        let (declarations, body) = source_parts(&self.expression);
        let bx = format!("{declarations}\n{}", bx_source(&params, &result, &body));
        let (functions, body) = self
            .reference_expression
            .as_deref()
            .map_or((declarations, body), source_parts);
        compare_bx_program(
            &bx,
            FunctionProgram {
                functions,
                params,
                result,
                body,
                valid: Some(!invalid),
                dslx_ir: self.dslx_ir,
                dynamic_index: None,
            },
            seed,
            DslxExecution::ConcreteConstants,
            self.inputs,
            &[],
        )
    }
}

/// Sample concrete builtin calls, checked constants and rejection of extra
/// operands.
pub fn check_builtins(data: &[u8]) -> Result<(), String> {
    let [op, flags, width_byte, other_byte, size_byte, pattern] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let name = BUILTINS[usize::from(op) % BUILTINS.len()];
    let bad_arity = flags & 0xe0 == 0xe0;
    let seed = crate::input_seed(data);
    let width = usize::from(width_byte % 129) + 1;
    let other = usize::from(other_byte % 129) + 1;
    let count = usize::from(size_byte % 7) + 1;
    let unsigned = bits(false, width);
    let signed = flags & 1 != 0;
    let word = bits(signed, width);
    let other_word = bits(flags & 4 != 0, other);
    let payload = match (flags >> 3) & 3 {
        0 => word.clone(),
        1 => Tuple(vec![word.clone(), Bool]),
        2 => array(word.clone(), 2),
        _ => Tuple(vec![array(word.clone(), 2), other_word.clone()]),
    };
    let array_ty = array(payload.clone(), count);
    let query_type = bits(false, 32);
    let make = |parametrics: &str, args: &str, params: Vec<Type>, result: Type| {
        BuiltinCase::new(name, parametrics, args, params, result, bad_arity)
    };
    let mut case = match name {
        "rev" | "clz" | "ctz" | "and_reduce" | "or_reduce" | "xor_reduce" | "encode" => {
            let width = if flags & 4 != 0 { 0 } else { width };
            let operand = bits(false, width);
            let result = match name {
                "and_reduce" | "or_reduce" | "xor_reduce" => Bool,
                "encode" => bits(false, width.next_power_of_two().trailing_zeros() as usize),
                _ => operand.clone(),
            };
            if flags & 8 != 0 {
                BuiltinCase::new(
                    "map",
                    "",
                    &format!("a, {name}"),
                    vec![array(operand, count)],
                    array(result, count),
                    bad_arity,
                )
            } else {
                let mut case = make("", "a", vec![operand], result);
                case.sample_argument(0, bit_patterns(width, pattern), seed);
                case
            }
        }
        "decode" => {
            let in_width = usize::from(width_byte % 8);
            let max_ir_width = 1 << in_width;
            let out_width = if flags & 4 == 0 {
                (other - 1) % (max_ir_width + 1)
            } else {
                max_ir_width + other
            };
            let result = bits(false, out_width);
            let mut case = make(
                &format!("<{result}>"),
                "a",
                vec![bits(false, in_width)],
                result,
            );
            // The pinned DSLX converter aborts on outputs wider than the
            // input can address. Its constexpr evaluator supports them.
            case.dslx_ir = out_width <= max_ir_width;
            case.sample_argument(0, index_patterns(in_width, out_width), seed);
            case
        }
        "one_hot" => {
            let args = if signed { "a, true" } else { "a, false" };
            let mut case = make("", args, vec![unsigned], bits(false, width + 1));
            case.sample_argument(0, bit_patterns(width, pattern), seed);
            case
        }
        "one_hot_sel" | "priority_sel" => {
            let mut params = vec![bits(false, count), array(word.clone(), count)];
            let args = if name == "priority_sel" {
                params.push(word.clone());
                "a, b, c"
            } else {
                "a, b"
            };
            let mut case = make("", args, params, word);
            case.sample_argument(0, bit_patterns(count, pattern), seed);
            case
        }
        "gate!" => {
            let mut case = make("", "a, b", vec![Bool, payload.clone()], payload);
            case.sample_argument(0, vec![IrBits::zero(1), IrBits::all_ones(1)], seed);
            case
        }
        "bit_slice_update" => {
            let index_width = usize::from(pattern % 129) + 1;
            let mut case = make(
                "",
                "a, b, c",
                vec![
                    unsigned.clone(),
                    bits(false, index_width),
                    bits(false, other),
                ],
                unsigned,
            );
            case.sample_argument(1, index_patterns(index_width, width), seed);
            case
        }
        "signex" => {
            let mut case = make("", "a, b", vec![word, other_word.clone()], other_word);
            case.sample_argument(0, bit_patterns(width, pattern), seed);
            case
        }
        "smulp" | "umulp" => {
            let operand = bits(name == "smulp", width);
            let mut case = make("", "a, b", vec![operand.clone(), operand], unsigned);
            // The native interpreters and PIR may select different valid
            // partial products.
            case.expression = format!("let (p0, p1) = {}; p0 + p1", case.expression);
            case
        }
        "array_rev" | "array_size" => {
            let result = if name == "array_rev" {
                array_ty.clone()
            } else {
                query_type
            };
            if flags & 8 == 0 {
                make("", "a", vec![array_ty], result)
            } else {
                let outer_size = other % 4 + 1;
                let mut case = BuiltinCase::new(
                    "map",
                    "",
                    &format!("a, {name}"),
                    vec![array(array_ty, outer_size)],
                    array(result, outer_size),
                    bad_arity,
                );
                // Pinned DSLX can interpret these map callbacks, but its
                // function IR converter cannot specialize them.
                case.dslx_ir = false;
                case
            }
        }
        "array_slice" => {
            let literal_source = flags & 4 != 0 && matches!(&payload, Bits { signed: false, .. });
            let result = array(payload, other % 7 + 1);
            let start_type = bits(false, other);
            let (mut case, start_arg) = if literal_source {
                // The typed witness supplies the element type of these
                // otherwise unsized source literals.
                let elements = (0..count)
                    .map(|i| ((pattern >> i) & 1).to_string())
                    .collect::<Vec<_>>();
                let mut case = make(
                    "",
                    &format!("[{}], a, b", elements.join(", ")),
                    vec![start_type, result.clone()],
                    result,
                );
                if !bad_arity {
                    // Pinned DSLX infers the shared type but can emit IR
                    // with the source literal's narrower element width.
                    let typed = elements
                        .iter()
                        .map(|value| format!("{word}:{value}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    case.reference_expression = Some(format!("array_slice([{typed}], a, b)"));
                }
                (case, 0)
            } else {
                (
                    make(
                        "",
                        "a, b, c",
                        vec![array_ty, start_type, result.clone()],
                        result,
                    ),
                    1,
                )
            };
            case.sample_argument(start_arg, index_patterns(other, count), seed);
            case
        }
        "zip" => make(
            "",
            "a, b",
            vec![array_ty, array(other_word.clone(), count)],
            array(Tuple(vec![payload, other_word]), count),
        ),
        "update" => {
            let inner = if flags & 4 == 0 { 2 } else { 4 };
            let outer = if signed { 2 } else { 4 };
            let matrix = array(array(payload.clone(), inner), outer);
            let outer_index = bits(false, usize::ilog2(outer) as usize);
            let inner_index = bits(false, usize::ilog2(inner) as usize);
            make(
                "",
                "a, (b, c), d",
                vec![matrix.clone(), outer_index, inner_index, payload],
                matrix,
            )
        }
        "all_ones!" => make(&format!("<{payload}>"), "", vec![], payload),
        "bit_count" | "element_count" => {
            let ty = if flags & 4 == 0 { array_ty } else { payload };
            make(&format!("<{ty}>"), "", vec![], query_type)
        }
        "checked_cast" => {
            let other_sign = flags & 4 != 0;
            let (input, wide) = match (signed, other_sign) {
                (true, true) => (bits(false, width - 1), bits(false, width + other)),
                _ => (word.clone(), bits(signed || other_sign, width + other)),
            };
            let type_args = cast_type_args(flags, &word, &wide);
            make(&type_args, &format!("a as {wide}"), vec![input], word)
        }
        "widening_cast" => {
            let out_signed = signed || flags & 4 != 0;
            let result = bits(out_signed, width + other);
            let type_args = cast_type_args(flags, &result, &word);
            make(&type_args, "a", vec![word], result)
        }
        "ceillog2" if flags & 8 != 0 => {
            // Pinned DSLX stores constexpr ceillog2 values as u32. Mapping
            // u32 operands keeps that representation consistent with the type.
            let input = bits(false, 32);
            let candidates = bit_patterns(32, pattern);
            let elements = (0..count)
                .map(|i| {
                    let value = &candidates[(i + usize::from(flags >> 3)) % candidates.len()];
                    dslx_literal(&input, &IrValue::Bits(value.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut case = BuiltinCase::new(
                "map",
                "",
                &format!("[{}], ceillog2", elements.join(", ")),
                vec![],
                array(input, count),
                bad_arity,
            );
            case.constant_only = true;
            case
        }
        "ceillog2" => {
            let candidates = bit_patterns(width, pattern);
            let input = candidates[usize::from(flags >> 2) % candidates.len()].clone();
            let args = dslx_literal(&unsigned, &IrValue::Bits(input))?;
            let mut case = make("", &args, vec![], unsigned);
            case.constant_only = true;
            // The pinned constexpr evaluator stores the result as u32 even
            // when its declared type has a different width.
            if !bad_arity {
                case.reference_expression = Some(format!("ceillog2({args}) as {}", case.result));
            }
            case
        }
        "configured_value_or" => {
            let fallback = dslx_literal(&word, &patterned_value(&word, seed))?;
            if flags & 4 != 0 {
                let label = array(bits(false, 8), count);
                make("", &format!("a, {fallback}"), vec![label], word)
            } else {
                make(
                    "",
                    &format!(r#""FuzzOption{pattern}", {fallback}"#),
                    vec![],
                    word,
                )
            }
        }
        _ => unreachable!("unhandled builtin {name}"),
    };
    let bad_type = flags & 0xe0 == 0xc0 && !case.params.is_empty();
    if bad_type {
        match name {
            "one_hot_sel" | "priority_sel" => {
                case.params[1] = array(case.result.clone(), count + 1);
            }
            "zip" => {
                let Array { size, .. } = &mut case.params[1] else {
                    unreachable!("zip's second operand is an array");
                };
                *size += 1;
            }
            "widening_cast" => {
                case.params[0] = bits(signed, width + other + 1);
            }
            "rev" | "clz" | "ctz" | "and_reduce" | "or_reduce" | "xor_reduce" | "encode"
            | "decode" | "one_hot" => {
                case.params[0] = bits(true, width);
            }
            _ => case.params[0] = Tuple(Vec::new()),
        }
    }
    let invalid = bad_arity || bad_type;
    if invalid {
        case.inputs.clear();
    }
    case.compare(invalid, flags & 2 != 0, seed)
}

fn bits(signed: bool, width: usize) -> Type {
    Bits { signed, width }
}

fn array(element: Type, size: usize) -> Type {
    Array {
        element: Box::new(element),
        size,
    }
}

/// Casts accept an optional explicit source type after the destination.
fn cast_type_args(flags: u8, dest: &Type, source: &Type) -> String {
    if flags & 16 == 0 {
        format!("<{dest}>")
    } else {
        format!("<{dest}, {source}>")
    }
}

/// Distinguish aggregate lanes even when fuzz entropy is all zeroes.
fn patterned_value(ty: &Type, seed: u64) -> IrValue {
    let element_seed = |i: usize| seed.wrapping_mul(11).wrapping_add((i + 1) as u64);
    match ty {
        Bool => IrValue::Bits(IrBits::from_lsb_fn(1, |_| seed & 1 != 0)),
        Bits { width, .. } => IrValue::Bits(IrBits::from_lsb_fn(*width, |i| {
            seed.rotate_left((i / 64) as u32) & (1 << (i % 64)) != 0
        })),
        Tuple(types) => IrValue::make_tuple(
            &types
                .iter()
                .enumerate()
                .map(|(i, ty)| patterned_value(ty, element_seed(i)))
                .collect::<Vec<_>>(),
        ),
        Array { element, size } => IrValue::make_array(
            &(0..*size)
                .map(|i| patterned_value(element, element_seed(i)))
                .collect::<Vec<_>>(),
        )
        .expect("builtin test arrays have a common element type and are nonempty"),
        _ => unreachable!("generator only builds structural types"),
    }
}

/// Give one-hot operations a zero, a one-hot value and multiple set bits.
fn bit_patterns(width: usize, pattern: u8) -> Vec<IrBits> {
    let selected = usize::from(pattern) % width.max(1);
    vec![
        IrBits::zero(width),
        IrBits::all_ones(width),
        IrBits::from_lsb_fn(width, |i| i == selected),
        IrBits::from_lsb_fn(width, |i| i <= selected),
    ]
}

/// Exercise the first and last valid offsets, clipping, and oversized indices.
fn index_patterns(width: usize, bound: usize) -> Vec<IrBits> {
    [0, bound.saturating_sub(1), bound]
        .map(|index| {
            IrBits::from_lsb_fn(width, |i| index.checked_shr(i as u32).unwrap_or(0) & 1 != 0)
        })
        .into_iter()
        .chain([IrBits::all_ones(width)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{BUILTINS, check_builtins};
    use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};

    #[test]
    fn compare_each_generated_builtin() -> Result<(), String> {
        for op in 0..BUILTINS.len() as u8 {
            for [flags, width, other, count, pattern] in [
                [0, 0, 0, 0, 0],
                [4, 0, 5, 4, 0],
                [7, 96, 32, 4, 128],
                [9, 8, 42, 2, 5],
                [18, 63, 64, 5, 73],
                [31, 128, 129, 7, 15],
                [192, 11, 3, 3, 10],
                [224, 11, 3, 3, 10],
            ] {
                let data = [op, flags, width, other, count, pattern];
                check_builtins(&data).map_err(|err| format!("input {data:?}: {err}"))?;
            }
        }
        Ok(())
    }

    #[test]
    fn reject_literals_that_cannot_infer_builtin_parameters() -> Result<(), String> {
        for (body, result) in [
            ("rev(1)", "u8"),
            ("xor_reduce(1)", "bool"),
            ("gate!(true, 1)", "u8"),
            ("signex(u4:1, 0)", "u8"),
            ("ceillog2(7)", "u8"),
            (r#"configured_value_or("flag", 7)"#, "u8"),
        ] {
            compare_program(
                FunctionProgram {
                    functions: String::new(),
                    params: String::new(),
                    result: result.to_owned(),
                    body: body.to_owned(),
                    valid: Some(false),
                    dslx_ir: false,
                    dynamic_index: None,
                },
                0,
                DslxExecution::ConcreteConstants,
                Vec::new(),
            )?;
        }
        Ok(())
    }
}
