// SPDX-License-Identifier: Apache-2.0

//! Focused comparisons for DSLX expressions with observable evaluation effects.

use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue, ir};

/// Decodes an expression observer and the independent choices within its
/// family.
pub fn generate(data: &[u8]) -> Result<AuditCase, String> {
    let (&family, options) = data.split_first().unwrap_or((&0, &[]));
    match family % 4 {
        0 => formatted_assertion(options),
        1 => array_size(options),
        kind => struct_update(options, kind == 3),
    }
    .map_err(|error| error.to_string())
}

/// Records the reference success condition independently from its returned
/// value.
fn require(builder: &mut FnBuilder, predicate: BValue) -> Result<(), BuilderError> {
    let token = builder.after_all(&[])?;
    builder.assert(token, predicate, "observer condition", "")?;
    Ok(())
}

/// Generates a formatted or ordinary assertion with the same unit-valued
/// contract.
fn formatted_assertion(data: &[u8]) -> Result<AuditCase, BuilderError> {
    let consumer = data.first().copied().unwrap_or_default() % 5;
    let flags = data.get(1).copied().unwrap_or_default();
    let ordinary = flags & 1 != 0;
    let negate_predicate = flags & 2 != 0;
    let negate_guard = flags & 4 != 0;
    let predicate = if negate_predicate { "!x" } else { "x" };
    let call = if ordinary {
        format!("assert_eq({predicate}, true)")
    } else {
        format!(r#"assert_fmt!({predicate}, "checked")"#)
    };
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    let condition = if negate_predicate { builder.not(x)? } else { x };
    let unit = builder.tuple(&[])?;
    let (family, ty, body, result) = match consumer {
        0 => ("assert_fmt_return", "()", call, unit),
        1 => (
            "assert_fmt_tuple",
            "((), bool)",
            format!("({call}, x)"),
            builder.tuple(&[unit, x])?,
        ),
        2 => {
            let guard = if negate_guard { "!x" } else { "x" };
            (
                "assert_fmt_conditional",
                "()",
                format!("if {guard} {{ {call} }} else {{ () }}"),
                unit,
            )
        }
        3 => (
            "assert_fmt_equality",
            "bool",
            format!("{call} == ()"),
            builder.literal(IrValue::bool(true))?,
        ),
        _ => ("assert_fmt_discard", "bool", format!("{call}; x"), x),
    };
    let allowed = if consumer == 2 {
        let skipped = if negate_guard { x } else { builder.not(x)? };
        builder.or(skipped, condition)?
    } else {
        condition
    };
    require(&mut builder, allowed)?;
    let source = format!("pub fn main(x: bool) -> {ty} {{ {body} }}\n");
    Ok(AuditCase::Semantic(ContractCase::single(
        if ordinary {
            "assert_eq_unit_control"
        } else {
            family
        },
        source,
        builder.build_package(result, "reference")?.to_string(),
    )))
}

/// Models the DSLX bytecode evaluation of an array length. An inline operand
/// contributes its type only. Hoisting an effectful operand evaluates it first.
fn array_size(data: &[u8]) -> Result<AuditCase, BuilderError> {
    let mode = data.first().copied().unwrap_or_default() % 3;
    let length = usize::from(data.get(1).copied().unwrap_or_default() % 8) + 1;
    let negated = data.get(2).copied().unwrap_or_default() & 1 != 0;
    let predicate = if negated { "!x" } else { "x" };
    let assertion = if mode == 2 {
        String::new()
    } else {
        format!("assert_eq({predicate}, true); ")
    };
    let elements = vec!["x"; length].join(", ");
    let body = if mode == 1 {
        "let a = make_array(x); array_size(a)"
    } else {
        "array_size(make_array(x))"
    };
    let source = format!(
        "fn make_array(x: bool) -> bool[{length}] {{ {assertion}[{elements}] }}\npub fn main(x: bool) -> u32 {{ {body} }}\n"
    );
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    if mode == 1 {
        let condition = if negated { builder.not(x)? } else { x };
        require(&mut builder, condition)?;
    }
    let result = builder.literal(IrValue::u32(length as u32))?;
    Ok(AuditCase::Semantic(ContractCase::single(
        "array_size_bytecode_effects",
        source,
        builder.build_package(result, "reference")?.to_string(),
    )))
}

/// Models bytecode's evaluation count for a struct update. A direct base runs
/// once per inherited field, while an explicitly bound base runs exactly once.
fn struct_update(data: &[u8], trace: bool) -> Result<AuditCase, BuilderError> {
    let total = 1 + usize::from(data.first().copied().unwrap_or_default() % 4);
    let updated = 1 + usize::from(data.get(1).copied().unwrap_or_default()) % total;
    let observed = usize::from(data.get(2).copied().unwrap_or_default()) % total;
    let hoisted = data.get(3).copied().unwrap_or_default() & 1 != 0;
    let start = usize::from(data.get(4).copied().unwrap_or_default()) % total;
    let fields = (0..total)
        .map(|i| format!("f{i}: bool"))
        .collect::<Vec<_>>()
        .join(", ");
    let values = (0..total)
        .map(|i| format!("f{i}: x"))
        .collect::<Vec<_>>()
        .join(", ");
    let updates = (0..updated)
        .map(|i| format!("f{}: !x", (start + i) % total))
        .collect::<Vec<_>>()
        .join(", ");
    let effect = if trace {
        r#"trace_fmt!("BASE_EVALUATED")"#
    } else {
        "assert_eq(x, true)"
    };
    let (binding, base) = if hoisted {
        ("let s = base(x); ", "s")
    } else {
        ("", "base(x)")
    };
    let source = format!(
        "struct S {{ {fields} }}\nfn base(x: bool) -> S {{ {effect}; S {{ {values} }} }}\npub fn main(x: bool) -> bool {{ {binding}S {{ {updates}, ..{base} }}.f{observed} }}\n"
    );
    let evaluations = if hoisted { 1 } else { total - updated };
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    if !trace && evaluations > 0 {
        require(&mut builder, x)?;
    }
    let overridden = (observed + total - start) % total < updated;
    let result = if overridden { builder.not(x)? } else { x };
    let family = if trace {
        "struct_splat_bytecode_traces"
    } else {
        "struct_splat_bytecode_assertions"
    };
    let case = ContractCase::single(
        family,
        source,
        builder.build_package(result, "reference")?.to_string(),
    );
    if trace {
        Ok(AuditCase::Traces {
            case,
            expected: evaluations,
        })
    } else {
        Ok(AuditCase::Semantic(case))
    }
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::audit_contracts::AuditCase;
    use xlsynth::{IrFunctionJit, IrPackage, XlsIrValue};

    /// Runs the reference at both Boolean inputs and checks the trace
    /// expectation.
    fn check_reference(
        bytes: &[u8],
        values: [XlsIrValue; 2],
        fails: [bool; 2],
        traces: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (case, count) = match generate(bytes)? {
            AuditCase::Semantic(case) => (case, None),
            AuditCase::Traces { case, expected } => (case, Some(expected)),
            AuditCase::ConstexprFailure { .. } => return Err("unexpected acceptance case".into()),
        };
        assert_eq!(count, traces);
        let reference = IrPackage::parse_ir(&case.expected_ir, None)?;
        let jit = IrFunctionJit::new(&reference.get_function("main")?)?;
        for (i, value) in values.into_iter().enumerate() {
            let result = jit.run(&[XlsIrValue::bool(i == 1)])?;
            assert_eq!(result.value, value);
            assert_eq!(!result.assert_messages.is_empty(), fails[i]);
        }
        Ok(())
    }

    #[test]
    fn guarded_assertions_preserve_unit_and_only_fail_in_the_selected_arm()
    -> Result<(), Box<dyn std::error::Error>> {
        let unit = XlsIrValue::make_tuple(&[]);
        check_reference(
            &[0, 2, 4],
            [unit.clone(), unit.clone()],
            [true, false],
            None,
        )?;
        check_reference(
            &[0, 2, 2],
            [unit.clone(), unit.clone()],
            [false, true],
            None,
        )?;
        check_reference(
            &[0, 1],
            [
                XlsIrValue::make_tuple(&[unit.clone(), false.into()]),
                XlsIrValue::make_tuple(&[unit, true.into()]),
            ],
            [true, false],
            None,
        )
    }

    #[test]
    fn array_size_only_observes_hoisted_assertions() -> Result<(), Box<dyn std::error::Error>> {
        for (mode, fails) in [(0, [false, false]), (1, [false, true]), (2, [false, false])] {
            check_reference(&[1, mode, 5, 1], [6u32.into(), 6u32.into()], fails, None)?;
        }
        Ok(())
    }

    #[test]
    fn splat_checks_projected_values_and_base_evaluation_counts()
    -> Result<(), Box<dyn std::error::Error>> {
        // Four fields, two overrides starting at f3 (f3 and f0). The directly
        // inherited f1 returns x, whereas f0 returns !x.
        check_reference(
            &[3, 3, 1, 1, 0, 3],
            [false.into(), true.into()],
            [false, false],
            Some(2),
        )?;
        check_reference(
            &[3, 3, 1, 0, 1, 3],
            [true.into(), false.into()],
            [false, false],
            Some(1),
        )?;
        check_reference(
            &[2, 1, 1],
            [true.into(), false.into()],
            [false, false],
            None,
        )?;
        check_reference(
            &[2, 1, 1, 0, 1],
            [true.into(), false.into()],
            [true, false],
            None,
        )
    }
}
