// SPDX-License-Identifier: Apache-2.0

//! Pareto frontier utilities for (AND-count, depth).

use serde::{Deserialize, Serialize};

use crate::cut_db::fragment::GateFnFragment;
use crate::cut_db::tt16::TruthTable16;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub tt: TruthTable16,
    pub ands: u16,
    pub depth: u16,
    pub frag: GateFnFragment,
}

#[inline]
pub fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    (a.ands <= b.ands && a.depth <= b.depth) && (a.ands < b.ands || a.depth < b.depth)
}

#[inline]
pub fn same_cost(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    a.ands == b.ands && a.depth == b.depth
}
