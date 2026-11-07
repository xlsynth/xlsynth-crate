// SPDX-License-Identifier: Apache-2.0

mod eval;
mod function_calls;
mod function_oracle;
mod function_syntax;
mod generator;
mod module_graph;
mod scalar_match;
mod unroll_types;

pub use function_calls::check_function_calls;
pub use function_syntax::check_function_syntax;
pub use module_graph::check_module_graph;
pub use scalar_match::check_scalar_match;
pub use unroll_types::check_unroll_types;

/// Generate a shared expression and compare both frontends on reproducible
/// inputs.
pub fn check(data: &[u8]) -> Result<(), String> {
    let sample = generator::generate(data, generator::EvaluationDomain::Hardware);
    eval::check_sample(&sample, input_seed(data))
}

/// Seed the shared concrete input sampler from the end of the fuzz input.
fn input_seed(data: &[u8]) -> u64 {
    let tail = &data[data.len().saturating_sub(8)..];
    let mut seed = [0; 8];
    seed[..tail.len()].copy_from_slice(tail);
    u64::from_le_bytes(seed)
}
