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

## Sequential transition boundaries and cleanup

`SequentialGateFn` represents a single-clock design as a combinational
`transition: GateFn` and bindings for external ports and register Q/D ports.
Synchronous enables and resets are already expressed in next-state logic; this
representation does not introduce asynchronous reset or multiple clocks.

`aig_serdes::sequential_transition::rebind_sequential_transition` restores the
packed sequential interface after an ordinary AIGER round trip or external
combinational optimization. Emit AIGER with symbols and retain every input and
output bit, in order. Rebinding validates scalar names and counts, restores
zero-width ports from the original schema, and rejects ambiguous flattened
names. It preserves register identities and shared output bindings, but does not
prove that replacement logic is equivalent. No external tool is invoked.

`aig::sequential_cleanup::cleanup_sequential_gate_fn` removes dead, constant,
and structurally equal/complementary register bits to a fixed point. The result
contains a new design and reduction statistics; the input is unchanged. It
preserves all external ports and the clock, compacts surviving packed register
bits in original LSB order, and removes dead combinational logic. It is not
retiming or general Boolean/sequential equivalence checking.

Constant and duplicate substitutions propagate through affected fanouts before
reconstruction, avoiding a full rebuild per stage of a duplicate register chain.
This does not add an unconditional combinational normalization pass; normal
structurally hashed lowering and external optimization retain their sharing.

All register startup values are unspecified. Cleanup uses synthesis-don't-care
semantics: it may replace constant-D state with a constant or merge equal and
complementary D expressions, restricting possible startup behavior. It must
preserve effective next-state behavior, including synchronous reset and enable
logic. Designs must not depend on independent power-up values. This is not
cycle-zero equivalence for independently sampled source and result state.
Cleanup is optional: retaining the original design retains all registers.

Cleanup and transition rebinding require `validate_legalized()` and reject
explicit reset/enable fields. Control-aware cleanup before legalization is a
separate future pass.

The cleanup fuzzer generates resettable block IR, lowers its reset logic to
G8R, and compares an independent block evaluator, the original G8R, and cleaned
G8R after a reset edge. Focused tests cover unreset data masked by resettable
valid state. Arbitrary partially reset feedback graphs are not yet covered by
that reset-based oracle.

```rust,ignore
use xlsynth_g8r::aig::sequential_cleanup::cleanup_sequential_gate_fn;
use xlsynth_g8r::aig_serdes::sequential_transition::rebind_sequential_transition;

let rebound = rebind_sequential_transition(&original, &optimized_transition)?;
let cleaned = cleanup_sequential_gate_fn(&rebound)?;
// Use cleaned.design as the source of both the next transition graph and
// its register metadata. The old register bindings are no longer applicable.
```

These APIs require no ABC, Yosys, or Liberty library.

## Liberty import contract

The Liberty proto stores normalized functional and characterization data, not
the complete source syntax. The current format requires regeneration from raw
Liberty for older protos: those artifacts may have silently omitted functional
attributes. Both binary and textproto loaders check the format discriminator.

Functional interpretation assumes normal powered-on operation. The importer
intentionally ignores `power_down_function` on pins and sequential groups
(`ff`, `latch`, and their bank forms): power-off output/state corruption is
outside this model. This also applies when characterization power data is
included; including power tables does not enable power-aware functional
simulation. Other unsupported normal-operation semantics remain rejected.
Protos previously imported with a `power_down_function` rejection marker must
be regenerated from source to make those cells usable; the wire format is
unchanged.

Supported source normalization includes scalar `ff`/`latch`, `ff_bank` and
`latch_bank`, bundles with explicit member order, shared scalar controls, and
multi-name pin groups. Banks become per-bit state equations within the same
physical cell; cell area is not duplicated. Bundle attributes and timing/power
groups are inherited by member pins, with scalar member overrides. Overriding
an inherited timing/power group is currently rejected instead of guessing its
precedence. Bank mapping/packing is a separate consumer capability.
Expansion is limited to 4,096 total bank state bits per physical cell, checked
before allocating the generated names/equations. This resource limit does not
constrain register widths in G8R designs. FF source descriptions require both
clock and next-state equations; latches require data and enable together unless
they are set/reset-only latches. Unsupported set/reset behavior is retained for
inspection and rejected by functional consumers, not mistaken for an empty cell.

`Cell::validate_functional_semantics()` rejects cells with recorded unsupported
source behavior, including tristate/X outputs, secondary clocks/enables,
simultaneous clear/preset behavior, state tables/internal-node descriptions,
unexpanded buses, and unknown sequential attributes/groups. Unsupported cells
remain available for inspection; Boolean netlist projection and sequential
interpretation reject them when used. Internal and bidirectional pin directions
are preserved but not accepted as ordinary Boolean cells. Malformed supported
constructs return import errors. This is an explicitly supported subset, not a
claim to implement all Liberty extensions.

Ordinary scan-flop `next_state` equations are retained. A functional mapper can
prove a constant-control assignment implements a DFF; test-only `test_cell`
metadata does not override that model. Scan-chain construction and test-pin
role metadata are not provided.

Pin capacitances, maximum capacitance/transition/fanout, and fanout loads retain
effective source-library defaults and explicit pin/bundle overrides before
libraries are merged. Values are optional, finite, nonnegative, and expressed
in the library's units; explicit zero is not absence. Fanout is the sum of sink
`fanout_load` values, not necessarily pin count. These fields preserve scalar
electrical constraints; timing/resizing consumers must enforce them separately.

## Liberty register mapping

`netlist::register_mapping::RegisterMappingLibrary` builds a reusable FF catalog
from the normalized Liberty `Library`. It requires no timing tables. Selection
uses the smallest physical FF area, then prefers no data inversion, followed by
cell name, output pin, data pin, and control tie-offs for deterministic ties.
`RegisterCellSelection::FixedCell` requires a named supported cell without
fallback; unsupported sequential cells have inspectable rejection diagnostics.
FF area does not include the cost of any required data inverter.

The initial subset is one-bit, positive-edge synchronous FFs. Latches, multi-state
cells, asynchronous clear/preset cells, gated or ambiguous clocks, `dont_use`
cells, invalid areas, and malformed pin/state metadata are excluded. Unsupported
functional semantics flagged by the Liberty importer are also rejected; they
cannot be enabled by a fixed-cell override. Extra
scan/enable inputs may be tied only when exhaustive evaluation proves that the
cell's next state reduces to D or !D for both previous-state values. Search is
bounded to eight tied inputs. Q and QN are recognized from their Boolean
functions, not pin names, including explicitly declared complementary state.

`map_registers` returns a `RegisterMappingPlan`: a shared cell binding plus one
physical FF assignment per logical register bit, in register/LSB order. Port IDs
refer to the supplied design; mapping does not mutate it or perform register
cleanup. The selected output connects directly to logical Q. When `invert_data`
is true, the physical data pin must receive !D; otherwise it receives D. Every
other input must receive its recorded tie-off, and unused outputs remain open.
The caller must materialize these connections during combinational mapping and
netlist assembly; the plan itself is not an emitted netlist.

Mapping requires `validate_legalized()`: call `legalize_register_controls()` to
fold explicit reset/enable fields into effective D logic first. Power-up state
remains unspecified. Zero-bit/state-free designs require no FF cell.
Register sizing, ABC invocation, full netlist reconstruction, and CLI integration
are intentionally outside this API.

```rust,ignore
use xlsynth_g8r::netlist::register_mapping::{RegisterCellSelection, RegisterMappingLibrary};

let cells = RegisterMappingLibrary::new(&library)?;
let legalized = design.legalize_register_controls()?;
let plan = cells.map_registers(&legalized, &RegisterCellSelection::SmallestArea)?;
// Or use RegisterCellSelection::FixedCell("DFF".to_string()).
```

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
