// SPDX-License-Identifier: Apache-2.0

//! Helpers for optimizer fuzz targets and their focused semantic checks.

pub use xlsynth_pir_fuzz::fuzz_solver_limits;

#[cfg(feature = "has-bitwuzla")]
pub mod constant_shift_choices;
#[cfg(feature = "has-bitwuzla")]
pub mod constant_shift_choices_sample;
