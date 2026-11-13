// SPDX-License-Identifier: Apache-2.0

//! Generation options for the word-level Lean differential fuzz targets.

use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation,
};

/// Generates pure values, including arrays, tuples, and PIR extensions.
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

/// Uses the same value operations inside a single clocked block without
/// callees.
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
