// SPDX-License-Identifier: Apache-2.0

//! 4-input cut database support.
//!
//! This module provides:
//! - A compact `u16` truth-table representation for 4-input single-output
//!   Boolean functions.
//! - NPN canonicalization (input permutation/negation + output negation).
//! - A compact AIG witness representation (`GateFnFragment`) that can be
//!   projected to `GateFn`.
//! - A generator and on-disk serialization format (bincode) for canonical
//!   entries.
//! - A runtime loader that expands canonical entries to a dense 65536-entry
//!   lookup table.

pub mod enumerate;
pub mod fragment;
pub mod loader;
pub mod npn;
pub mod pareto;
pub mod serdes;
pub mod tt16;
