// SPDX-License-Identifier: Apache-2.0

//! Internal precision and work limits shared by range-analysis transfers.

/// Maximum concrete values enumerated by small-domain transfers.
pub(super) const EXACT_VALUES: usize = 16;

/// Maximum intervals retained at existing result-coarsening points.
pub(super) const RESULT_INTERVALS: usize = 16;

/// Maximum unknown-bit positions expanded into interval alternatives.
pub(super) const MAX_KNOWN_BITS_SPLITS: usize = 4;

/// Maximum Cartesian product of input intervals before fragmentation reduction.
pub(super) const INTERVAL_COMBINATIONS: usize = 1_000_000;

/// Work limit for width-dependent packing and slice-update refinements.
pub(super) const BIT_WORK_BUDGET: usize = 1_000_000;

/// Work limit for conditioning normalization on candidate leading-one
/// positions.
pub(super) const NORMALIZE_BIT_BUDGET: usize = 65_536;

/// Intermediate interval count above which normalization coarsens its union.
pub(super) const NORMALIZE_INTERMEDIATE_INTERVALS: usize = 64;
