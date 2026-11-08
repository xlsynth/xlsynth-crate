// SPDX-License-Identifier: Apache-2.0

//! Versioned converter audit grammars with semantic, event and compile-time
//! acceptance oracles.

use crate::converter_contracts::{self, ContractCase};
use crate::{
    SourceModule, aggregate_method_contracts, builtin_contracts, enum_width_contracts,
    nominal_contracts, observable_value_contracts, observer_contracts, scope_contracts,
    unrolled_loop_contracts,
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
}

impl AuditCase {
    /// Stable identifier used to retain replay bytes per property.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Semantic(case) | Self::Traces { case, .. } | Self::TraceMessages { case, .. } => {
                case.family
            }
            Self::ConstexprFailure { .. } => "constexpr_block_acceptance",
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
    use super::{AuditCase, generate, generate_v2};

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
