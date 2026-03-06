// SPDX-License-Identifier: Apache-2.0

//! Reference-simulator metadata and conservative value-domain filtering.
//!
//! The language standard is the semantic source of truth. These helpers model
//! which external implementations are suitable for differential grounding over
//! a given subset of the language.

use crate::Env;
use crate::LogicBit;
use crate::Value4;
use crate::ast::BinaryOp;
use crate::ast::Expr;

/// Whether a reference implementation can credibly ground two-value-only or
/// full four-value semantics for a sample.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ValueDomain {
    TwoValue,
    FourValue,
}

/// Concrete external implementations used for differential checks.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ReferenceSimKind {
    Iverilog,
    YosysCxxrtl,
}

/// Describes the semantic/value-domain surface a reference implementation
/// should be used for.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ReferenceSimCapabilities {
    pub value_domain: ValueDomain,
    pub supports_expr_diff: bool,
    pub supports_module_diff: bool,
}

impl ReferenceSimKind {
    /// Returns the current differential-grounding capabilities for this
    /// reference implementation.
    pub fn capabilities(self) -> ReferenceSimCapabilities {
        match self {
            Self::Iverilog => ReferenceSimCapabilities {
                value_domain: ValueDomain::FourValue,
                supports_expr_diff: true,
                supports_module_diff: true,
            },
            Self::YosysCxxrtl => ReferenceSimCapabilities {
                value_domain: ValueDomain::TwoValue,
                supports_expr_diff: false,
                supports_module_diff: true,
            },
        }
    }
}

/// Returns whether every environment binding is strictly two-valued (`0/1`
/// only) and therefore suitable for grounding against a two-valued reference
/// implementation.
pub fn env_is_two_value_safe(env: &Env) -> bool {
    env.iter().all(|(_, value)| value_is_two_value(value))
}

/// Conservatively checks whether an expression stays within a two-valued subset
/// suitable for comparison against a two-valued reference implementation.
///
/// This intentionally rejects more than strictly necessary. False negatives are
/// acceptable here because the goal is to keep the initial two-valued grounding
/// path simple and semantically unsurprising.
pub fn expr_is_two_value_safe(expr: &Expr, env: &Env) -> bool {
    env_is_two_value_safe(env) && expr_is_two_value_safe_inner(expr)
}

fn expr_is_two_value_safe_inner(expr: &Expr) -> bool {
    match expr {
        Expr::Ident(_) => true,
        Expr::Literal(value) | Expr::UnsizedNumber(value) => value_is_two_value(value),
        Expr::UnbasedUnsized(bit) => matches!(bit, LogicBit::Zero | LogicBit::One),
        Expr::Call { .. } => false,
        Expr::Concat(parts) => parts.iter().all(expr_is_two_value_safe_inner),
        Expr::Replicate { count, expr } => {
            expr_is_two_value_safe_inner(count) && expr_is_two_value_safe_inner(expr)
        }
        Expr::Cast { width, expr } => {
            expr_is_two_value_safe_inner(width) && expr_is_two_value_safe_inner(expr)
        }
        Expr::Index { expr, index } => {
            expr_is_two_value_safe_inner(expr) && expr_is_two_value_safe_inner(index)
        }
        Expr::Slice { expr, msb, lsb } => {
            expr_is_two_value_safe_inner(expr)
                && expr_is_two_value_safe_inner(msb)
                && expr_is_two_value_safe_inner(lsb)
        }
        Expr::IndexedSlice {
            expr, base, width, ..
        } => {
            expr_is_two_value_safe_inner(expr)
                && expr_is_two_value_safe_inner(base)
                && expr_is_two_value_safe_inner(width)
        }
        Expr::Unary { expr, .. } => expr_is_two_value_safe_inner(expr),
        Expr::Binary { op, lhs, rhs } => {
            !matches!(op, BinaryOp::CaseEq | BinaryOp::CaseNeq)
                && expr_is_two_value_safe_inner(lhs)
                && expr_is_two_value_safe_inner(rhs)
        }
        Expr::Ternary { cond, t, f } => {
            expr_is_two_value_safe_inner(cond)
                && expr_is_two_value_safe_inner(t)
                && expr_is_two_value_safe_inner(f)
        }
    }
}

fn value_is_two_value(value: &Value4) -> bool {
    value
        .bits_lsb_first()
        .iter()
        .all(|bit| matches!(bit, LogicBit::Zero | LogicBit::One))
}

#[cfg(test)]
mod tests {
    use super::ReferenceSimKind;
    use super::ValueDomain;
    use super::env_is_two_value_safe;
    use super::expr_is_two_value_safe;
    use crate::Env;
    use crate::Value4;
    use crate::parser::parse_expr;

    fn parse(text: &str) -> crate::ast::Expr {
        parse_expr(text).unwrap_or_else(|e| panic!("parse failed for {text:?}: {e:?}"))
    }

    fn parse_value(text: &str) -> Value4 {
        match parse(text) {
            crate::ast::Expr::Literal(value) | crate::ast::Expr::UnsizedNumber(value) => value,
            other => panic!("expected literal-ish value, got {other:?}"),
        }
    }

    #[test]
    fn capabilities_distinguish_two_value_and_four_value_backends() {
        let four_value = ReferenceSimKind::Iverilog.capabilities();
        assert_eq!(four_value.value_domain, ValueDomain::FourValue);
        assert!(four_value.supports_expr_diff);
        assert!(four_value.supports_module_diff);

        let two_value = ReferenceSimKind::YosysCxxrtl.capabilities();
        assert_eq!(two_value.value_domain, ValueDomain::TwoValue);
        assert!(!two_value.supports_expr_diff);
        assert!(two_value.supports_module_diff);
    }

    #[test]
    fn two_value_filter_accepts_plain_zero_one_expressions() {
        let env = Env::new();
        let expr = parse("((4'b0011 + 4'b0001) == 4'd4) ? 1'b1 : 1'b0");
        assert!(expr_is_two_value_safe(&expr, &env));
        assert!(expr_is_two_value_safe(&parse("4'(2'b11)"), &env));
    }

    #[test]
    fn two_value_filter_rejects_unknown_sensitive_constructs() {
        let env = Env::new();
        assert!(!expr_is_two_value_safe(&parse("4'b10x1"), &env));
        assert!(!expr_is_two_value_safe(&parse("'z"), &env));
        assert!(!expr_is_two_value_safe(&parse("4'b0011 === 4'b0011"), &env));
        assert!(!expr_is_two_value_safe(&parse("$signed(4'b0011)"), &env));
    }

    #[test]
    fn env_filter_rejects_non_two_value_bindings() {
        let mut env = Env::new();
        env.insert("good", parse_value("4'b1010"));
        assert!(env_is_two_value_safe(&env));

        env.insert("bad", parse_value("4'b10x1"));
        assert!(!env_is_two_value_safe(&env));
        assert!(!expr_is_two_value_safe(&parse("bad == 4'b1010"), &env));
    }
}
