// SPDX-License-Identifier: Apache-2.0

//! Compare the scope of a type operand across signatures and body annotations.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;
use std::fmt::Write;
use xlsynth_pir::IrValue;

const OBSERVATION: &str = "(bool, u32, uN[256])";
const OBSERVE: &str = r#"fn observe<S:bool, W:u32>(y:xN[S][W]) -> (bool, u32, uN[256]) {
    (S, W, y as uN[256])
}"#;

/// Each position receives the identical type expression. For a BX block port
/// the DSLX reference is a function with the corresponding value signature.
#[derive(Clone, Copy)]
enum Position {
    FunctionResult,
    FunctionInput,
    TypedLet,
    LocalAlias,
    LetInAlias,
    BlockResult,
    BlockInput,
}

const POSITIONS: [Position; 7] = [
    Position::FunctionResult,
    Position::FunctionInput,
    Position::TypedLet,
    Position::LocalAlias,
    Position::LetInAlias,
    Position::BlockResult,
    Position::BlockInput,
];

/// Build a conditional width or signedness operand in the template's lexical
/// scope. The module constant `x` has the same name as the data parameter.
fn type_annotation(family: u8, sign_operand: bool, pivot: u32) -> Annotation {
    let (good, data_operand, local, wrong_type) = if sign_operand {
        ("S", "x != u32:0", "x == N", "u32:5")
    } else {
        ("N + u32:1", "x", "x + u32:2", "true")
    };
    let (taken, alternate) = match family {
        0 => (good.into(), data_operand),
        1 => (good.into(), if sign_operand { "!S" } else { "N" }),
        2 => (format!("let x = N; {local}"), data_operand),
        3 => (good.into(), "MISSING"),
        4 => (good.into(), "LATER"),
        5 => (good.into(), wrong_type),
        // Unsized integer arithmetic in a width has the operand's u32 context.
        _ => (if sign_operand { "!S" } else { "127 + 2" }.into(), good),
    };
    let operand = format!("if N == u32:{pivot} {{ {taken} }} else {{ {alternate} }}");
    Annotation {
        operand,
        sign_operand,
    }
}

/// A generated operand can be written inline or bound by a preceding let.
struct Annotation {
    operand: String,
    sign_operand: bool,
}

impl Annotation {
    fn ty(&self, operand: &str) -> String {
        if self.sign_operand {
            format!("xN[{operand}][N]")
        } else {
            format!("xN[S][{operand}]")
        }
    }
}

/// Compare a template at the selected bindings. Observing the inferred
/// signedness and width catches same-layout errors even when the value is zero.
fn compare_position(
    position: Position,
    annotation: &Annotation,
    bindings: &[(u32, bool)],
    seed: u64,
    valid: Option<bool>,
) -> Result<(), String> {
    let value = "if sel { !0 } else { 0 }";
    let ty = annotation.ty(&annotation.operand);
    let (args, result, expression) = match position {
        Position::FunctionResult | Position::BlockResult => {
            ("x:u32, sel:bool".to_owned(), ty.as_str(), value.to_owned())
        }
        Position::TypedLet => (
            "x:u32, sel:bool".into(),
            OBSERVATION,
            format!("let y:{ty} = {value}; observe(y)"),
        ),
        Position::LocalAlias => (
            "x:u32, sel:bool".into(),
            OBSERVATION,
            format!("type Word = {ty}; let y:Word = {value}; observe(y)"),
        ),
        Position::LetInAlias => (
            "x:u32, sel:bool".into(),
            OBSERVATION,
            format!(
                "let k = {}; type Word = {}; let y:Word = {value}; observe(y)",
                annotation.operand,
                annotation.ty("k")
            ),
        ),
        Position::BlockInput | Position::FunctionInput => (
            format!("x:u32, sel:bool, y:{ty}"),
            OBSERVATION,
            "observe(y)".into(),
        ),
    };
    let template = format!("probe<N:u32, S:bool>({args}) -> {result} {{ {expression} }}");
    let declarations = format!("const x = u32:9;\n{OBSERVE}");
    let native = format!("{declarations}\nfn {template}\nconst LATER = u32:3;");
    let bx_template = if matches!(position, Position::BlockResult | Position::BlockInput) {
        format!("{declarations}\nblock {template}\nconst LATER = u32:3;")
    } else {
        native.clone()
    };
    let params = "x:u32, sel:bool";
    let mut body = String::new();
    let mut results = Vec::new();
    for (i, (n, signed)) in bindings.iter().enumerate() {
        let extra = if matches!(position, Position::BlockInput | Position::FunctionInput) {
            ", if sel { !0 } else { 0 }"
        } else {
            ""
        };
        writeln!(body, "let v{i} = probe<u32:{n}, {signed}>(x, sel{extra});").unwrap();
        let observed = match position {
            Position::FunctionResult | Position::BlockResult => format!("observe(v{i})"),
            _ => format!("v{i}"),
        };
        results.push(observed);
    }
    let result = match results.as_slice() {
        [] => {
            body.push('x');
            "u32".into()
        }
        [one] => {
            body.push_str(one);
            OBSERVATION.into()
        }
        _ => {
            write!(body, "({})", results.join(", ")).unwrap();
            format!("({})", vec![OBSERVATION; bindings.len()].join(", "))
        }
    };
    let bx = format!("{bx_template}\n{}", bx_source(params, &result, &body));
    let x = IrValue::make_ubits(32, 37).map_err(|e| e.to_string())?;
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: native,
            params: params.into(),
            result,
            body,
            valid,
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        vec![
            vec![x.clone(), IrValue::bool(false)],
            vec![x, IrValue::bool(true)],
        ],
        &[],
    )
}

/// Move a single type operand through all annotation positions. Calls can
/// leave the template unused, specialize once, or revisit bindings in order.
pub fn check_annotation_scope(data: &[u8]) -> Result<(), String> {
    let [family, flags, width, delta] = std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let pivot = u32::from(width % 129) + 1;
    let other = (pivot + u32::from(delta % 128)) % 129 + 1;
    let a = (pivot, flags & 8 != 0);
    let b = (other, !a.1);
    let bindings: &[_] = match flags % 8 {
        0 => &[],
        1 => &[a],
        2 => &[b],
        3 => &[a, b, a],
        4 => &[b, a, b],
        5 => &[a, (pivot, !a.1), a],
        6 => &[a, a],
        _ => &[b, b],
    };
    let ty = type_annotation(family % 7, flags & 16 != 0, pivot);
    for position in POSITIONS {
        compare_position(position, &ty, bindings, input_seed(data), None)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{POSITIONS, compare_position, type_annotation};

    #[test]
    fn unused_and_selected_data_parameters() -> Result<(), String> {
        for position in POSITIONS {
            for sign_operand in [false, true] {
                let ty = type_annotation(0, sign_operand, 4);
                for (bindings, valid) in [
                    (&[][..], true),
                    (&[(4, false)][..], true),
                    (&[(7, true)][..], false),
                    (&[(4, false), (7, true)][..], false),
                    (&[(7, true), (4, false)][..], false),
                ] {
                    compare_position(position, &ty, bindings, 37, Some(valid))?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn lexical_names_and_reused_specializations() -> Result<(), String> {
        for position in POSITIONS {
            for sign_operand in [false, true] {
                for family in [1, 6] {
                    let ty = type_annotation(family, sign_operand, 65);
                    let a = (65, false);
                    let b = (3, true);
                    for bindings in [[a, b, a], [b, a, b]] {
                        compare_position(position, &ty, &bindings, 37, Some(true))?;
                    }
                }
                let shadow = type_annotation(2, sign_operand, 65);
                compare_position(position, &shadow, &[(65, true)], 37, Some(true))?;
                for family in [3, 4] {
                    let unknown = type_annotation(family, sign_operand, 65);
                    compare_position(position, &unknown, &[], 37, Some(false))?;
                }
                let wrong_type = type_annotation(5, sign_operand, 65);
                compare_position(position, &wrong_type, &[], 37, Some(true))?;
                compare_position(position, &wrong_type, &[(65, false)], 37, Some(false))?;
            }
        }
        Ok(())
    }
}
