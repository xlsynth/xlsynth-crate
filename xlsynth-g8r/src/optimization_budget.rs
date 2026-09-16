// SPDX-License-Identifier: Apache-2.0

//! Cooperative, opt-in deadline for speculative work with a saved incumbent.
//! A deadline may discard a private candidate, never a partially modified
//! caller-owned incumbent. Unbudgeted synthesis remains deterministic.

use anyhow::{Result, anyhow};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINES: RefCell<Vec<Option<Instant>>> = const { RefCell::new(Vec::new()) };
}

/// A same-thread, nested deadline guard that also restores state on errors.
pub(crate) struct ScopedOptimizationBudget(PhantomData<Rc<()>>);

impl ScopedOptimizationBudget {
    pub(crate) fn new(duration: Option<Duration>) -> Self {
        let deadline = duration.and_then(|duration| Instant::now().checked_add(duration));
        DEADLINES.with(|deadlines| {
            let mut deadlines = deadlines.borrow_mut();
            let outer = deadlines.last().copied().flatten();
            deadlines.push(match (outer, deadline) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            });
        });
        Self(PhantomData)
    }
}

impl Drop for ScopedOptimizationBudget {
    fn drop(&mut self) {
        DEADLINES.with(|deadlines| {
            deadlines.borrow_mut().pop();
        });
    }
}

/// Checks at transaction boundaries so cancellation never interrupts rollback.
pub(crate) fn check() -> Result<()> {
    let expired = DEADLINES.with(|deadlines| {
        deadlines
            .borrow()
            .last()
            .copied()
            .flatten()
            .is_some_and(|deadline| Instant::now() >= deadline)
    });
    if expired {
        Err(anyhow!(
            "alternative cover exceeded its optimization time budget"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_budgets_restore_and_cannot_extend_outer_deadline() {
        assert!(check().is_ok());
        {
            let _outer = ScopedOptimizationBudget::new(Some(Duration::ZERO));
            assert!(check().is_err());
            {
                let _inner = ScopedOptimizationBudget::new(None);
                assert!(check().is_err());
            }
            assert!(check().is_err());
        }
        assert!(check().is_ok());
    }
}
