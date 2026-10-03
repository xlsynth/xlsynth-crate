// SPDX-License-Identifier: Apache-2.0

//! Backend-supplied cost models for deciding whether to apply PIR rewrites.

use crate::ir;

/// Ignores floating-point noise when comparing delays from the same model.
const DELAY_COMPARISON_TOLERANCE: f64 = 1e-9;

/// Estimated area and delay of a function in a backend's cost model.
///
/// Compare values from the same evaluator: evaluators define the units and
/// which operations contribute to the estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrCost {
    pub area: usize,
    /// Finite, nonnegative delay in the evaluator's model.
    pub delay: f64,
}

impl IrCost {
    /// Requires a strict improvement with a 1e-9 absolute delay tolerance.
    pub fn is_pareto_improvement_on(self, other: Self) -> bool {
        if !self.delay.is_finite()
            || !other.delay.is_finite()
            || self.delay < 0.0
            || other.delay < 0.0
        {
            return false;
        }
        let delay_change = self.delay - other.delay;
        self.area <= other.area
            && delay_change <= DELAY_COMPARISON_TOLERANCE
            && (self.area < other.area || delay_change < -DELAY_COMPARISON_TOLERANCE)
    }
}

/// Supplies a cost model without making PIR depend on a particular backend.
///
/// The input must be a well-typed, acyclic function graph in its containing
/// package; nodes need not be stored in topological order. Evaluators may treat
/// operations requiring package context as opaque boundaries.
/// `Ok(None)` declines an estimate (for example, after exhausting a work
/// budget); `Err` reports an unexpected evaluation failure. Rewrites compare
/// both alternatives using the same evaluator and preserve the input on either
/// outcome. Implementations may retain caches or budgets between calls.
pub trait IrCostEvaluator {
    fn estimate(&mut self, function: &ir::Fn) -> Result<Option<IrCost>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pareto_comparison_tolerates_delay_noise_without_counting_it_as_a_win() {
        let incumbent = IrCost {
            area: 10,
            delay: 5.0,
        };
        for delay_change in [0.0, -0.5e-9, 0.5e-9] {
            let candidate = IrCost {
                delay: incumbent.delay + delay_change,
                ..incumbent
            };
            assert!(!candidate.is_pareto_improvement_on(incumbent));
            assert!(
                IrCost {
                    area: 9,
                    ..candidate
                }
                .is_pareto_improvement_on(incumbent)
            );
        }
        let faster = IrCost {
            delay: incumbent.delay - 2e-9,
            ..incumbent
        };
        assert!(faster.is_pareto_improvement_on(incumbent));
        assert!(!IrCost { area: 11, ..faster }.is_pareto_improvement_on(incumbent));
        assert!(
            !IrCost {
                area: 9,
                delay: incumbent.delay + 2e-9,
            }
            .is_pareto_improvement_on(incumbent)
        );
    }

    #[test]
    fn pareto_comparison_rejects_invalid_delays_from_either_estimate() {
        let valid = IrCost {
            area: 10,
            delay: 5.0,
        };
        for delay in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-12] {
            assert!(!IrCost { area: 9, delay }.is_pareto_improvement_on(valid));
            assert!(!valid.is_pareto_improvement_on(IrCost { area: 11, delay }));
        }
    }
}
