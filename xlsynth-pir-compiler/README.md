# Compiled PIR blocks

`block::CompiledBlock::compile(&package, top)` accepts a parsed or built PIR
package. It projects the block hierarchy and compiles the cycle function and
its callees once. Structural feedback is legalized in the projected cycle when
needed, preserving hierarchical register names and diagnostic source sites. Each
simulation can then advance independently.

Use `initial_state(RegisterInit::AllZeros)`, seeded `Random`, or `Supplied`
register values to start. Pass the state and one `IrValue` per data input to
`step`. The `inputs()`, `outputs()`, and `registers()` slices describe the
order and exact types. The result contains the outputs sampled before the edge
and the register state for the following edge. Clone a state to fork a run.

A reset pin must be supplied on each clock. An active synchronous reset takes
priority over enables for the registers wired to it. Otherwise a register
holds its old value or takes the value of its single enabled write. Two enabled
writes to the same register fail the clock. Assertion and bounds failures also
leave the previous state available for retry. Traces and covers are collected
by default. Use `step_with_options` or `prepare_instance_with_options` to change
optional observations. The PIR interpreter additionally supports node observers.

For repeated clocks, `prepare_instance(init)` allocates the input, register,
result, and scratch buffers once. Call `tick(&inputs)` and read selected
outputs using `output(index)`. To bind a caller's port order once, use
`bind_inputs(&names)` with `tick_bound` and `bind_output(name)` with
`read_output`. Bindings and snapshots belong to the compiled block that made
them. Output reads return owned values. A failed tick preserves the last
successful outputs, and restoring a snapshot clears them until the next edge.
Wide operations, materializing typed outputs, and event collection can allocate.

Behaviorally modeled external blocks use their models. Opaque external
instances, asynchronous reset, and unsupported clock or inout configurations
produce preparation errors. PIR types or operations outside the Cranelift
backend's supported subset produce compilation errors, including zero-length
native arrays. Zero-width bits and empty tuples are supported.

Run the [two-clock example](examples/clock_block.rs) or the block benchmarks
from the workspace root.

```sh
cargo run -p xlsynth-pir-compiler --example clock_block
cargo bench -p xlsynth-pir-compiler --bench block_performance
```

## Benchmark results

The benchmarks ran on a 64-bit x86 server using Cargo's optimized bench profile
and `ExecutionOptions::collect_all()`. The implementation was based on `ce12c004`.
The fixtures start with zero registers and repeat four prebuilt inputs.
Each execution measurement includes reading one owned output. Preparation starts
with parsed PIR. Compilation starts with the prepared cycle and excludes the
Cargo build. The benchmark checks outputs and register state against the
interpreter outside the timers.

The table reports medians of 15 samples. Brackets give the 25th and 75th
percentiles of the per-clock times. The allocation columns count Rust
allocation requests on the clocking thread, including the owned output read.

| Fixture | Prepare (µs) | JIT (µs) | Typed step (ns) | Prepared tick + output (ns) | Typed allocations | Prepared allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8-bit counter | 10.7 | 112 | 781 \[780, 782\] | 185 \[185, 186\] | 11 | 0 |
| 32-bit counter | 10.9 | 125 | 785 \[783, 789\] | 182 \[182, 182\] | 11 | 0 |
| 2-stage pipeline, 4 × 32-bit words | 13.4 | 128 | 1,992 \[1,990, 2,313\] | 619 \[619, 620\] | 27 | 3 |
| 4-stage pipeline, 16 × 64-bit words | 19.8 | 200 | 8,853 \[8,851, 8,855\] | 1,607 \[1,603, 1,613\] | 40 | 3 |
| 65-bit mask, shift, reduce | 22.2 | 401 | 1,780 \[1,779, 1,782\] | 498 \[496, 499\] | 34 | 7 |
| 129-bit mask, shift, reduce | 22.5 | 371 | 1,707 \[1,640, 1,751\] | 610 \[609, 610\] | 34 | 7 |
| 257-bit mask, shift, reduce | 22.3 | 457 | 1,701 \[1,672, 1,851\] | 576 \[573, 580\] | 34 | 7 |

After warm-up, the prepared tick itself recorded zero allocations on these
seven fixtures. Reading an array or tuple output accounts for the 3 or 7
allocations shown above. A separate counter with trace and cover sites
averaged 8.27 allocations per tick with observations enabled and zero with
them disabled. The 129-bit identity and mask/shift/reduction helpers both
recorded zero allocations inside the tick on this stimulus.
