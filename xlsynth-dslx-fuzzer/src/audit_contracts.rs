// SPDX-License-Identifier: Apache-2.0

//! Versioned converter audit grammars with semantic, event and compile-time
//! acceptance oracles.

use crate::converter_contracts::{self, ContractCase};
use crate::{
    SourceModule, aggregate_method_contracts, bound_parametric_contracts, builtin_contracts,
    dead_branch_contracts, enum_width_contracts, inferred_flow_contracts, literal_order_contracts,
    map_parametric_contracts, mixed_literal_contracts, nested_scope_contracts, nominal_contracts,
    observable_value_contracts, observer_contracts, pattern_acceptance, scope_contracts,
    signed_call_contracts, tuple_literal_contracts, unrolled_loop_contracts,
    zero_width_cast_contracts,
};

/// The observable property of a generated program under the standalone
/// converter.
#[derive(Debug)]
pub enum AuditCase {
    Semantic(ContractCase),
    /// Count trace events on the emitted main function at Boolean input true.
    Traces {
        case: ContractCase,
        expected: usize,
    },
    /// Exactly one trace message for each Boolean input, in false, true order.
    TraceMessages {
        case: ContractCase,
        expected: [String; 2],
    },
    /// A failing assertion in a module constant must prevent conversion.
    ConstexprFailure {
        modules: Vec<SourceModule>,
    },
    /// A typechecked tuple binding must convert to IR with the declared Boolean
    /// result.
    PatternAcceptance {
        modules: Vec<SourceModule>,
    },
    /// Compares the original binding against IR and an explicitly cast DSLX
    /// runtime control.
    TupleLiteral {
        case: ContractCase,
        runtime_module: SourceModule,
    },
    /// Checks inferred operand types against an independent value in IR and
    /// bytecode.
    MixedLiteral {
        case: ContractCase,
        raw_runtime_module: SourceModule,
        runtime_module: SourceModule,
    },
    /// Typecheck an integer literal used in a call, prove the emitted IR, and
    /// optionally evaluate the original binding with a DSLX bytecode test.
    BoundCall {
        case: ContractCase,
        runtime_module: Option<SourceModule>,
    },
}

impl AuditCase {
    /// Stable identifier used to retain replay bytes per property.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Semantic(case) | Self::Traces { case, .. } | Self::TraceMessages { case, .. } => {
                case.family
            }
            Self::ConstexprFailure { .. } => "constexpr_block_acceptance",
            Self::PatternAcceptance { .. } => "tuple_pattern_acceptance",
            Self::TupleLiteral { case, .. }
            | Self::MixedLiteral { case, .. }
            | Self::BoundCall { case, .. } => case.family,
        }
    }
}

/// Decodes audit-v1 inputs. Legacy contract bytes retain their original meaning
/// in the default focused grammar and can also be embedded under selector four.
pub fn generate(data: &[u8]) -> Result<AuditCase, String> {
    let (&family, options) = data.split_first().unwrap_or((&0, &[]));
    match family % 5 {
        0 => scope_contracts::generate(options).map(AuditCase::Semantic),
        1 => builtin_contracts::generate(options).map(AuditCase::Semantic),
        2 => observer_contracts::generate(options),
        3 => Ok(constexpr_block(options)),
        _ => converter_contracts::generate(options).map(AuditCase::Semantic),
    }
}

/// Decodes audit-v2 correctness cases. A leading zero embeds an audit-v1
/// sample and checks its semantics without imposing a graph-size limit.
pub fn generate_v2(data: &[u8]) -> Result<AuditCase, String> {
    let (&family, options) = data.split_first().unwrap_or((&0, &[]));
    match family % 7 {
        0 => {
            let mut case = generate(options)?;
            if let AuditCase::Semantic(contract) = &mut case {
                contract.max_and_nodes = None;
            }
            Ok(case)
        }
        1 => nominal_contracts::enumerate(options).map(AuditCase::Semantic),
        2 => nominal_contracts::projection(options).map(AuditCase::Semantic),
        3 => aggregate_method_contracts::derive_to_bits(options).map(AuditCase::Semantic),
        4 => aggregate_method_contracts::method_map(options).map(AuditCase::Semantic),
        5 => observable_value_contracts::trace_message(options),
        _ => observable_value_contracts::assertion_label(options),
    }
}

/// Decodes audit-v3 programs. A zero prefix replays an entire audit-v2 input.
pub fn generate_v3(data: &[u8]) -> Result<AuditCase, String> {
    let (&family, options) = data.split_first().unwrap_or((&0, &[]));
    match family % 8 {
        0 | 1 => generate_v2(options),
        2..=4 => unrolled_loop_contracts::generate(options).map(AuditCase::Semantic),
        5 | 6 => Ok(AuditCase::Semantic(enum_width_contracts::generate(options))),
        _ => aggregate_method_contracts::method_map_derived(options).map(AuditCase::Semantic),
    }
}

/// Decodes audit-v4 inputs. A zero prefix preserves any audit-v3 program and
/// oracle.
pub fn generate_v4(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    if selector % 4 == 3 {
        Ok(AuditCase::PatternAcceptance {
            modules: pattern_acceptance::generate(options),
        })
    } else {
        generate_v3(options)
    }
}

/// Decodes audit-v5 inputs. A zero prefix preserves the source and oracle of
/// audit-v4.
pub fn generate_v5(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    if selector % 4 == 3 {
        Ok(tuple_literal_contracts::generate(options))
    } else {
        generate_v4(options)
    }
}

/// Decodes audit-v6 inputs. A zero prefix preserves every audit-v5 case.
pub fn generate_v6(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    if selector % 4 == 3 {
        Ok(mixed_literal_contracts::generate(options))
    } else {
        generate_v5(options)
    }
}

/// Decodes nested scope, literal-order and inferred-flow cases. A zero prefix
/// embeds an audit-v6 input with the same source and oracle.
pub fn generate_v7(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    let case = match selector % 8 {
        5 => nested_scope_contracts::generate(options)?,
        6 => literal_order_contracts::generate(options),
        7 => inferred_flow_contracts::generate(options),
        _ => return generate_v6(options),
    };
    Ok(AuditCase::Semantic(case))
}

/// Exercises map specializations, same-width scalar casts and value-parametric
/// conditionals. Prefix an audit-v7 input with zero to preserve it.
pub fn generate_v8(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    let case = match selector % 8 {
        5 => map_parametric_contracts::generate(options)?,
        6 => zero_width_cast_contracts::generate(options),
        7 => dead_branch_contracts::generate(options),
        _ => return generate_v7(options),
    };
    Ok(AuditCase::Semantic(case))
}

/// Decodes calls using let-bound integer literals. Prefix an audit-v8 unit
/// with zero to retain its source and oracle.
pub fn generate_v9(data: &[u8]) -> Result<AuditCase, String> {
    let (&selector, options) = data.split_first().unwrap_or((&0, &[]));
    match selector % 8 {
        5 => Ok(bound_parametric_contracts::data_argument(options)),
        6 => Ok(bound_parametric_contracts::value_argument(options)),
        7 => Ok(signed_call_contracts::generate(options)),
        _ => generate_v8(options),
    }
}

/// Varies whether a checked call's result is bound or discarded. The call
/// must execute even when the final constant does not use its return value.
fn constexpr_block(data: &[u8]) -> AuditCase {
    let form = data.first().copied().unwrap_or_default() % 4;
    let flags = data.get(1).copied().unwrap_or_default();
    let succeeds = flags & 1 != 0;
    let tail = flags & 2 != 0;
    let call = format!("checked({succeeds})");
    let statement = match form {
        0 => format!("let _ = {call};"),
        1 => format!("let local = {call};"),
        2 => format!("let (v, _) = ({call}, false);"),
        _ => format!("{call};"),
    };
    let source = format!(
        r#"fn checked(p: bool) -> bool {{ assert!(p, "audit_const_assert"); !p }}
const K = {{ {statement} {tail} }};
pub fn main(x: bool) -> bool {{ x ^ K }}
"#
    );
    if succeeds {
        let value = u8::from(tail);
        let reference = format!(
            r#"package expected
top fn main(x: bits[1] id=1) -> bits[1] {{
  k: bits[1] = literal(value={value}, id=2)
  ret result: bits[1] = xor(x, k, id=3)
}}
"#
        );
        AuditCase::Semantic(ContractCase::single(
            "constexpr_block_control",
            source,
            reference,
        ))
    } else {
        AuditCase::ConstexprFailure {
            modules: vec![SourceModule {
                name: "sample".into(),
                text: source,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuditCase, generate, generate_v2, generate_v6, generate_v7, generate_v8, generate_v9,
    };

    #[test]
    fn zero_prefix_preserves_v7_and_v8_semantic_contracts() -> Result<(), String> {
        for selector in [5, 6, 7] {
            let bytes = [selector, 1, 2, 2, 5];
            let prefixed = [&[0][..], &bytes].concat();
            for (old, new) in [
                (generate_v7(&bytes)?, generate_v8(&prefixed)?),
                (generate_v8(&bytes)?, generate_v9(&prefixed)?),
            ] {
                let (AuditCase::Semantic(old), AuditCase::Semantic(new)) = (old, new) else {
                    return Err("expected an embedded semantic contract".into());
                };
                assert_eq!(old.family, new.family);
                assert_eq!(old.modules, new.modules);
                assert_eq!(old.expected_ir, new.expected_ir);
                assert_eq!(old.max_and_nodes, new.max_and_nodes);
            }
        }
        Ok(())
    }

    #[test]
    fn zero_prefix_preserves_mixed_literal_source_and_both_runtime_oracles() -> Result<(), String> {
        let bytes = [3, 1, 2, 2, 5];
        let prefixed = [&[0][..], &bytes].concat();
        let unpack = |case| match case {
            AuditCase::MixedLiteral {
                case,
                raw_runtime_module,
                runtime_module,
            } => Ok((
                case.family,
                case.modules,
                case.expected_ir,
                raw_runtime_module,
                runtime_module,
            )),
            _ => Err("expected a mixed-literal contract".to_string()),
        };
        assert_eq!(
            unpack(generate_v6(&bytes)?)?,
            unpack(generate_v7(&prefixed)?)?
        );
        Ok(())
    }

    #[test]
    fn embedded_guard_keeps_its_program_and_only_checks_correctness() -> Result<(), String> {
        let AuditCase::Semantic(old) = generate(&[4, 5, 1])? else {
            return Err("expected a semantic guard contract".into());
        };
        let AuditCase::Semantic(new) = generate_v2(&[0, 4, 5, 1])? else {
            return Err("expected an embedded semantic guard contract".into());
        };
        assert_eq!(old.modules, new.modules);
        assert_eq!(old.expected_ir, new.expected_ir);
        assert!(old.max_and_nodes.is_some());
        assert_eq!(new.max_and_nodes, None);
        Ok(())
    }
}
