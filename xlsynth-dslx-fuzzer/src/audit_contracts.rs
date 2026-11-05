// SPDX-License-Identifier: Apache-2.0

//! Version-one converter audit grammar, with distinct semantic, event and
//! compile-time acceptance oracles.

use crate::converter_contracts::{self, ContractCase};
use crate::{SourceModule, builtin_contracts, observer_contracts, scope_contracts};

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
    /// A failing assertion in a module constant must prevent conversion.
    ConstexprFailure {
        modules: Vec<SourceModule>,
    },
}

impl AuditCase {
    /// Stable identifier used to retain replay bytes per property.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Semantic(case) | Self::Traces { case, .. } => case.family,
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
