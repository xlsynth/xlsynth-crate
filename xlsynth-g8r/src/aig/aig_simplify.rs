// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::{AigNode, AigOperand, AigRef};
use crate::gate_builder::GateBuilder;

/// Tests whether an effective OR has `operand` as a direct input.
pub(crate) fn or_contains_operand(gb: &GateBuilder, or: AigOperand, operand: AigOperand) -> bool {
    or.negated
        && matches!(gb.gates[or.node.id], AigNode::And2 { a, b, .. }
            if a.negate() == operand || b.negate() == operand)
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
            if or_contains_operand(g8_builder, *a, *b) {
                return Some(*b);
            }
            if or_contains_operand(g8_builder, *b, *a) {
                return Some(*a);
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
