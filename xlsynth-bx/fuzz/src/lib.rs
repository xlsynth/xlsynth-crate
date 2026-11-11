// SPDX-License-Identifier: Apache-2.0

mod array_concat;
mod array_literals;
mod bits_array_casts;
mod block_callbacks;
mod block_let_patterns;
mod block_value_parametrics;
mod builtins;
mod clocked_test;
mod clocked_test_oracle;
mod clocked_test_sample;
mod counted_for;
mod enums;
mod eval;
mod function_calls;
mod function_oracle;
mod function_parametrics;
mod function_syntax;
mod function_type_parametrics;
mod function_value_parametrics;
mod generator;
mod imported_struct_parametrics;
mod map_lambda;
mod module_constants;
mod module_graph;
mod scalar_match;
mod struct_parametrics;
mod struct_type_parameters;
mod struct_updates;
mod tuple_match;
mod type_aliases;
mod unroll_types;

pub use array_concat::check_array_concat;
pub use array_literals::check_array_literals;
pub use bits_array_casts::check_bits_array_casts;
pub use block_callbacks::check_block_callbacks;
pub use block_let_patterns::check_block_let_patterns;
pub use block_value_parametrics::check_block_value_parametrics;
pub use builtins::check_builtins;
pub use clocked_test::check_clocked_test;
pub use counted_for::check_counted_for;
pub use enums::check_enums;
pub use function_calls::check_function_calls;
pub use function_parametrics::check_function_parametrics;
pub use function_syntax::check_function_syntax;
pub use function_type_parametrics::check_function_type_parametrics;
pub use function_value_parametrics::check_function_value_parametrics;
pub use imported_struct_parametrics::check_imported_struct_parametrics;
pub use map_lambda::check_map_lambda;
pub use module_constants::check_module_constants;
pub use module_graph::check_module_graph;
pub use scalar_match::check_scalar_match;
pub use struct_parametrics::check_struct_parametrics;
pub use struct_type_parameters::check_struct_type_parameters;
pub use struct_updates::check_struct_updates;
pub use tuple_match::check_tuple_match;
pub use type_aliases::check_type_aliases;
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
