// SPDX-License-Identifier: Apache-2.0

mod assert_valid_sv;
mod simulate_sv;

pub use assert_valid_sv::{assert_valid_sv, assert_valid_sv_flist, FlistEntry};
pub use simulate_sv::{
    simulate_pipeline_single_pulse, simulate_pipeline_single_pulse_custom, simulate_sv_flist,
};

pub mod ir_fuzz;
