// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Compares every output and register of a PIR block against its RTL in Vastly.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::semantics::Trace;
use xlsynth_codegen_fuzz::vastly::check_rtl_trace;
use xlsynth_codegen_fuzz::{block_options, emit, generate};

fuzz_target!(|data: &[u8]| {
    // Allow libFuzzer to mutate inputs and initial state without changing the
    // graph.
    let (stimulus, structure) = data.split_at(data.len().min(8));
    let (output_count, graph) = structure.split_first().unwrap_or((&0, &[]));
    let mut options = block_options(true, true);
    // Bound per-cycle four-state work while exercising multiword signals and
    // nested aggregates.
    options.function_options.max_bit_width = 129;
    options.function_options.max_aggregate_leaves = 8;
    // Reserve visible outputs even if the body consumes its node budget.
    options.min_output_ports = 1 + usize::from(*output_count) % options.max_output_ports;
    // Codegen erases zero-width interfaces, which have no RTL signal to
    // compare.
    options.allow_zero_width_ports_and_registers = false;
    let package = generate(graph, &options);
    let seed = *blake3::hash(stimulus).as_bytes();
    let trace = Trace::with_seed(&package, seed);
    let rtl = emit(&package, &BlockCodegenOptions::default());
    if let Err(error) = check_rtl_trace(&rtl, &trace) {
        panic!(
            "PIR/Vastly block mismatch: {error}\nstimulus seed: {seed:?}\ninitial registers: {:?}\nIR:\n{package}\nRTL:\n{rtl}",
            trace.initial_state
        );
    }
});
