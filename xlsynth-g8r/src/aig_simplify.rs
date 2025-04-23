// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigOperand, AigRef};
use crate::gate_builder::GateBuilder;

// Extracts the operands for an effective "or" pattern, i.e.
// a | b => ~((~a) & (~b))
fn extract_or_pattern(a: AigOperand, g8_builder: &GateBuilder) -> Option<(AigOperand, AigOperand)> {
    if !a.negated {
        return None;
    }
    let gate = &g8_builder.gates[a.node.id];
    match gate {
        AigNode::And2 { a, b, .. } => Some((a.negate(), b.negate())),
        _ => None,
    }
}

// If the gate is an and with one side being "true" and the other operand being
// negated, it's an effective "not" and we return the operand being negated.
#[allow(dead_code)]
fn extract_not_pattern(gate: &AigNode, g8_builder: &GateBuilder) -> Option<AigRef> {
    match gate {
        AigNode::And2 { a, b, .. } => {
            if g8_builder.is_known_true(*a) && b.negated {
                return Some(b.node);
            }
            if g8_builder.is_known_true(*b) && a.negated {
                return Some(a.node);
            }
            None
        }
        _ => None,
    }
}

#[allow(dead_code)]
fn extract_and_nn_pattern(
    aig_ref: AigRef,
    g8_builder: &GateBuilder,
) -> Option<(AigOperand, AigOperand)> {
    let gate = &g8_builder.gates[aig_ref.id];
    match gate {
        AigNode::And2 { a, b, .. } => {
            if a.negated && b.negated {
                return Some((a.negate(), b.negate()));
            }
            None
        }
        _ => None,
    }
}

/// Simplifies the case that goes: (a | b) & b => b
pub fn operand_simplify(aig_ref: AigRef, g8_builder: &mut GateBuilder) -> Option<AigOperand> {
    let gate = &g8_builder.gates[aig_ref.id];
    match gate {
        AigNode::And2 { a, b, .. } => {
            if let Some((or_lhs, or_rhs)) = extract_or_pattern(*a, g8_builder) {
                if or_lhs == *b {
                    return Some(*b);
                }
                if or_rhs == *b {
                    return Some(*b);
                }
            }
            if let Some((or_lhs, or_rhs)) = extract_or_pattern(*b, g8_builder) {
                if or_lhs == *a {
                    return Some(*a);
                }
                if or_rhs == *a {
                    return Some(*a);
                }
            }

            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::GateBuilderOptions;

    use super::*;

    /// Tests that we simplify:
    /// (a | b) & b => b
    #[test]
    fn test_aig_simplify_a_or_b_and_b() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut g8_builder = GateBuilder::new("test".to_string(), GateBuilderOptions::no_opt());
        let a = g8_builder.add_input("a".to_string(), 1);
        let b = g8_builder.add_input("b".to_string(), 1);
        let a0 = a.get_lsb(0);
        let b0 = b.get_lsb(0);
        let a_or_b = g8_builder.add_or_binary(*a0, *b0);
        let a_or_b_and_b = g8_builder.add_and_binary(a_or_b, *b0);
        let simplified = operand_simplify(a_or_b_and_b.non_negated().unwrap(), &mut g8_builder);
        assert!(simplified.is_some());
        assert_eq!(simplified.unwrap(), *b0);
    }
}
