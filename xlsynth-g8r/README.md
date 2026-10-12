## `xlsynth-g8r`: gate-level infrastructure

`xlsynth-g8r` hosts the gate-level side of the xlsynth stack:

- **`aig`**: core AIG/GateFn representation and structural transforms (fraig, balancing, etc.).
- **`aig_serdes`**: (de)serialization to/from AIGER and a textual gate format.
- **`aig_sim`**: scalar and SIMD gate-level simulators.
- **`liberty` / `liberty_proto`**: Liberty parsing, indexing, and proto bindings.
- **`netlist`**: Verilog-like gate-level netlist parsing, connectivity, cone traversal, and GV→IR.
- **`transforms`**: local gate-level rewrite passes used by optimization and MCMC logic.

Most functionality is exposed via the `xlsynth_g8r` library and thin binaries under `src/bin/`.

Augmented IR optimization is provided by
[`xlsynth-aug-opt`](../xlsynth-aug-opt/README.md), which depends on g8r for local
cost estimates. Use `xlsynth_aug_opt::ir2gates_from_ir_text` to optimize before
mapping; `xlsynth_g8r::ir2gates::Ir2GatesOptions` contains only mapping options.

## Explicit register controls and legalization

`SequentialGateFn` is the single sequential representation. Each
`RegisterBinding` has raw data (`d`), Q feedback, and optional `enable` and
`reset` fields. Controls name one-bit transition output ports and can be shared.
A synchronous reset has a width-matched `IrBits` value and polarity; an enable
has polarity. Designs use one rising-edge clock. Asynchronous reset is unsupported.

`validate()` accepts both explicit-control and legalized designs, including
mixed registers. `legalize_register_controls()` returns a design whose D
endpoints contain all control logic and whose control fields are cleared. Reset
overrides enable; otherwise enable selects data or Q (hold). Legalization
preserves every register and behavior from any common starting state, including
cycle zero. It does not clean up state and is a no-op on already-legalized graphs.

`aig_sim::sequential::simulate` evaluates either form directly. Explicit
controls are evaluated from the old state before simultaneous register updates.
Native text/binary serialization preserves both forms, including controls and
arbitrary-width reset values. Text serialization via `emit_g8r` returns
`Result<String, String>` and validates the complete metadata. Physical toggle
accounting, BLIF/RTL emission, and block-IR export require
`validate_legalized()`: they reject explicit controls rather than silently
ignoring them.

The existing block-lowering and pipeline-stitching APIs return legalized designs.
Their `_with_controls` variants retain explicit controls in the same
`SequentialGateFn` type:
`block_ir_to_sequential_gate_fn_with_controls`,
`block_package_to_sequential_gate_fn_with_controls`,
`lower_block_to_sequential_gate_fn_with_controls`, and
`stitch_gate_fns_into_pipeline_with_controls`.

Power-up initialization is not stored. A reset value only takes effect at an
asserted reset edge; simulation requires explicit initial state (including an
explicitly requested all-zero test state). Native formats are
`g8r_v4`/`g8rbin_v4` and preserve explicit controls. Older versions and explicit
BLIF latch initial values are rejected rather than silently discarded.

## Random direct G8R generation

`gate_random::generate_gate_design` accepts the same `EntropySource` interface
as the PIR generators (`DepletableBytes` for fuzzing or `RngEntropy` for seeded
sampling). It returns a validated `SequentialGateFn` and its builder policy.

`RandomGateOptions` controls port/register count and width ranges, the actual
AIG-node budget, AND/OR/XOR/XNOR/mux construction, folding and hashing, constants,
optional synchronous reset/enable metadata and polarity, endpoint reuse, and
output observability. Setting `registers: 0..=0` gives the combinational profile;
use `try_into_gate_fn()` when a consumer needs `GateFn`. Explicit controls remain
unlegalized. Width ranges starting at zero permit zero-width ports/state.

Structural stress options (chains, redundant pairs, and fanout) are opt-in and
off by default. The generic generator does not inject target-specific bug
patterns. It keeps the transition DAG topological, preserves Q/D feedback as
sequential boundaries, and reserves output-wiring entropy before consuming the
body stream. Invalid option combinations return errors rather than rejected
samples. Generated operation statistics report both emitted requests and newly
created operation roots that remain live. Fixed-seed RNG coverage sweeps check
operator liveness, interface bounds and widths, feedback, control combinations,
polarities, and nonempty endpoint aliases. Concrete input values and initial
state use the separate shared
`xlsynth_pir::random_inputs` library.

## Additional docs

- `docs/g8r_lib_timing_design.md`: why Liberty loading has `Library` (no timing) vs `LibraryWithTimingData` and the observed ASAP7 size/load tradeoffs.

## External Yosys APIs

`netlist::yosys::YosysToolchain` validates an executable and runs scripts with
explicit timeouts, captured diagnostics, and process-group cleanup. It needs no
Liberty files; `from_env()` uses `XLSYNTH_YOSYS_PATH` or `yosys` on `PATH`.

`YosysEnvironment` adds validated Liberty file paths for technology mapping.
`synthesize_to_gv` selects `YosysInputLanguage::{Verilog, SystemVerilog}` and
`YosysMappingKind::{Combinational, Sequential}` explicitly. The older convenience
methods retain their existing frontend and mapping behavior.

`YosysMappingContext` loads cell semantics from the same files and offers a
`preflight` mapping/import probe. Its environment setup requires
`XLSYNTH_YOSYS_PATH` and comma-separated `XLSYNTH_LIBERTY_FILES` pointing to
installed, compatible libraries. Sequential preflight requires flip-flop cells.
No library helper prints, caches global state, or decides whether a failure
should stop a fuzz campaign; callers own that policy. Execution errors preserve
`xlsynth::external_tool::ToolError` timeout/resource categories.

## Netlist parse benchmark

Run the synthetic netlist-parse microbenchmark with:

```shell
cargo bench -p xlsynth-g8r --bench netlist_parse_bench -- --verbose
```
