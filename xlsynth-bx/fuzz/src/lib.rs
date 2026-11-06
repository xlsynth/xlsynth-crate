// SPDX-License-Identifier: Apache-2.0

mod eval;
mod generator;
mod module_graph;
mod unroll_types;

pub use module_graph::check_module_graph;
pub use unroll_types::check_unroll_types;

/// Generate a shared expression and compare both frontends on reproducible
/// inputs.
pub fn check(data: &[u8]) -> Result<(), String> {
    let sample = generator::generate(data);
    let tail = &data[data.len().saturating_sub(8)..];
    let mut seed = [0; 8];
    seed[..tail.len()].copy_from_slice(tail);
    eval::check_sample(&sample, u64::from_le_bytes(seed))
}
