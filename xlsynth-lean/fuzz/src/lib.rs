// SPDX-License-Identifier: Apache-2.0

//! Generation options for the word-level Lean differential fuzz targets.
//!
//! Both targets draw from the same pure PIR operations and aggregate types.
//! Blocks use the subset that fits in a single block graph.

use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation,
};

/// Configures function generation for comparing pure return values in Lean and
/// PIR.
///
/// Generated packages can call helper functions and use arrays, tuples,
/// zero-width bits, and PIR extension operations. The generator excludes
/// events, counted loops, and partial products from these cases.
pub fn lean_function_options() -> RandomFnOptions {
    RandomFnOptions {
        max_nodes: 32,
        max_functions: 3,
        max_bit_width: 129,
        max_aggregate_leaves: 12,
        allow_gate: true,
        allow_map: true,
        allow_empty_case_sel: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_extension_ops: true,
        allow_events: false,
        max_counted_fors_per_function: 0,
        array_assumption_mode: ArrayAssumptionMode::ProvenSafe,
        enabled_operations: OperationSet::new(OperationSet::all_supported().iter().filter(|op| {
            !matches!(
                op,
                RandomOperation::Smulp
                    | RandomOperation::Umulp
                    | RandomOperation::CountedFor
                    | RandomOperation::AfterAll
                    | RandomOperation::Assert
                    | RandomOperation::Trace
                    | RandomOperation::Cover
            )
        })),
        ..RandomFnOptions::default()
    }
}

/// Configures a single clocked block with synchronous reset on every register.
///
/// Ports and registers may have zero width. The block reuses the function
/// generator's value operations except for `Invoke` and `Map`, which need a
/// separate callee.
pub fn lean_block_options() -> RandomBlockOptions {
    let function = lean_function_options();
    RandomBlockOptions {
        min_input_ports: 1,
        max_input_ports: 3,
        min_output_ports: 1,
        allow_zero_width_ports_and_registers: true,
        require_reset_on_all_registers: true,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_functions: 1,
            max_invokes_per_function: 0,
            allow_map: false,
            enabled_operations: OperationSet::new(
                function
                    .enabled_operations
                    .iter()
                    .filter(|op| !matches!(op, RandomOperation::Invoke | RandomOperation::Map)),
            ),
            ..function
        },
        ..RandomBlockOptions::default()
    }
}
