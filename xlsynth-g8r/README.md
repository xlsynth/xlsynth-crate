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
the complete source syntax. V4, V5, and V6 are versions of xlsynth-g8r's own
protobuf import contract, encoded in its `format_magic` field; they are not
versions of the Liberty file format. Binary and textproto loaders accept all
three payloads.

The discriminator originally separated a schema rewrite that reused field
numbers with different meanings: for example, the old library's units and the
new library's cells both use field 2. Some old payloads could otherwise decode
as plausible but incorrect data. The current value also identifies the import
contract. V4 importers could omit pin driver behavior, and V4/V5 importers could
omit `use_for_size_only`. An absent field cannot tell a reader whether the source
lacked that attribute or the importer discarded it. On import, cells carry
explicit markers for this missing coverage through merges and V6 reserialization.
V4 cells can be inspected and their timing payloads read, but cannot be used for
Boolean evaluation or behavior-dependent timing and power analysis. V5 cells
can be used for Boolean evaluation. Structural INV/NAND2 and register cell
selection require complete mapping-policy coverage, so V4 and V5 cells need
regeneration from source Liberty for those operations. V3 and older
incompatible schemas remain rejected. New files use V6 so older readers reject
restrictions they cannot enforce; older V6 readers also reject the new coverage
markers.

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
uses the smallest physical FF area, then prefers non-inverting D-to-output
polarity, followed by cell name, output pin, data pin, and control tie-offs for
deterministic ties.
`RegisterCellSelection::FixedCell` requires a named supported cell without
fallback; unsupported sequential cells have inspectable rejection diagnostics.
FF area does not include any polarity-compensation logic.

The initial subset is one-bit, positive-edge synchronous FFs. Latches, multi-state
cells, asynchronous clear/preset cells, gated or ambiguous clocks, `dont_use`
cells, invalid areas, and malformed pin/state metadata are excluded. Unsupported
functional semantics flagged by the Liberty importer are also rejected; they
cannot be enabled by a fixed-cell override. Extra
scan/enable inputs may be tied only when exhaustive evaluation proves that the
cell's next state reduces to D or !D for both previous-state values. Search is
bounded to eight tied inputs. Q and QN are recognized from their Boolean
functions, not pin names, including explicitly declared complementary state.

`map_registers` returns a `RegisterMappingPlan`: deduplicated cell bindings in
`cells` plus one physical FF assignment per logical register bit, in register/LSB
order. Each bit selects a binding with `cell_index`. Port IDs refer to the supplied
design; mapping does not mutate it or perform register cleanup.

The binding's `inverts_data_to_output` records the cell's intrinsic polarity:
whether its selected physical output captures !D rather than D. The bit's
`output_inverted` records its chosen state phase: physical Q equals logical Q XOR
that phase. Physical D must equal logical D XOR the state phase XOR the cell's
intrinsic polarity, after rephasing logical state uses. The clock pin receives
the design clock; remaining input pins receive their recorded tie-offs, and
unused outputs remain open. The caller materializes these connections during
combinational mapping and netlist assembly; the plan itself is not an emitted
netlist.

Selection preserves direct module-input-to-D and Q-to-module-output wires. A bit
on both boundaries therefore requires a non-inverting cell. Each bit uses the
smallest compatible cell; a fixed-cell request that cannot preserve both wires
is rejected rather than silently replaced.

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

## Direct ABC sequential mapping

`netlist::abc::AbcMappingContext` maps a `SequentialGateFn` to a complete flat
gate-level netlist without invoking Yosys. It legalizes explicit reset/enable
controls, optionally runs native register cleanup, selects physical FFs, and
sends only combinational transition logic to standalone ABC. The established
`abc-speed` script performs optimization, mapping, buffering, and resizing of
that combinational logic. Registers stay outside ABC; there is no retiming or
register sizing.

Supply **raw, uncompressed Liberty files** to ABC and a **separate normalized
Liberty model/proto** for native FF mapping and evaluation. No proto-to-Liberty
conversion occurs. For split libraries, the first one or two files must form a
usable combinational library (including inversion/buffering); additional files
are merged using `read_lib -m`. The returned diagnostics record the executable,
version, source paths, exact script, and captured output. Emitted cell/pin names
are checked against the normalized library, but matching names do not establish
functional/timing identity between the two inputs. Callers must use matching
libraries and filtering policies.

The mapper uses opaque scalar ABC port names, then restores the original packed
external interface and assembles FF clock/data/output connections and verified
tie-offs. Physical D inversions are separate mapping outputs, so shared logical
D/external-output bindings keep their original polarity. Zero-width ports are
retained in `AbcMappedDesign::design` and omitted from GV, which has no zero-width
port syntax. `emit_verilog()` returns deterministic text without ABC timestamps.

`AbcMappingOptions` requires explicit `io_constraints` and has no `Default`
implementation. `AbcMappingOptions::from_library(&library)` resolves and validates
the library's `reference_buffer_cell` and `default_output_load` at construction.
Alternatively, `AbcMappingOptions::new(constraints)` uses caller-supplied
`AbcIoConstraints`, without requiring library defaults. Both constructors select
the smallest-area FF, preserve all registers, and use a 180-second ABC timeout.
Callers can modify these options before mapping; the mapper validates the
constraints against its library rather than resolving defaults again.
The reference must be a usable ordinary non-inverting buffer. The library's
independent default load is expressed in its declared capacitance units and
converted to femtofarads for ABC; caller-supplied `output_load_ff` is already in
femtofarads. No multiplier is applied. Missing, nonpositive, or nonfinite default
loads and missing/invalid units are constructor errors, never an unconstrained
or buffer-derived fallback.

Set the optional library metadata when importing with
`lib2proto --reference-buffer-cell BUF --default-output-load CAPACITANCE`.
There is no cell-name heuristic or technology-specific default in code.
Older protos without these fields remain
readable and require explicit constraints for ABC mapping. The raw ABC library
must also contain the selected driver; a missing-driver warning is an error.
Resolved constraints are returned in the run diagnostics.

`AbcMappingOptions::script` contains the optimization/mapping passes and defaults
to the literal `ABC_SPEED_SCRIPT` string constant. To experiment with another
flow, replace this string, for example with `"&get -n; &dch; &nf; &put"`.
The wrapper always supplies raw-library loading, `read_aiger transition.aig`,
`strash`, and `read_constr timing.constr` before the passes, and
`write_verilog mapped.v` afterward. The default complete command sequence is
unchanged. Custom passes must leave a mapped network in ABC's main workspace
(use `&put` after GIA commands), preserve the combinational boundary, and omit
library/constraint/design IO commands. This is trusted ABC code, not a sandbox;
the wrapper does not parse or restrict the supplied commands. The full composed
script is captured in diagnostics. There is no CLI script flag yet.

ABC's legacy constraints are global: **the same output load also applies to
register-D boundaries**, rather than the actual selected flop pin capacitance.
This first version does not claim per-register launch/capture timing constraints
or a numeric PI-slew interface. Register-boundary STA remains a separate analysis.

```rust,ignore
use xlsynth_g8r::netlist::abc::{AbcMappingContext, AbcMappingOptions, AbcToolchain};

let toolchain = AbcToolchain::new("/path/to/abc")?;
let context = AbcMappingContext::new(toolchain, &["/path/to/cells.lib"], liberty)?;
let options = AbcMappingOptions::from_library(context.library())?;
let mapped = context.map_sequential(&design, &options)?;
let verilog = mapped.emit_verilog()?;
```

`AbcToolchain::from_env()` uses `XLSYNTH_ABC_PATH` or `abc` on PATH. The driver
exposes this API through `g8r2gv`. See `FUZZ.md` for the opt-in end-to-end mapping
campaign and its raw/proto library environment variables.

## Incremental netlist timing

`netlist::sta::incremental::IncrementalSta` owns a timing graph and shares the
prepared Liberty/output/setup kernels with full STA. `new` preserves independent
primary-input and register launch classes; `new_combinational` explicitly selects
combinational analysis. The instance and normalized-bit IDs remain stable.

`trial_edit(&TimingEdit, TrialOptions, callback)` is the common transaction API.
The callback observes candidate timing and affected instance IDs: `Ok(true)`
commits, while `Ok(false)`, an error, or an exhausted budget restores the
incumbent. A `Resize` preserves pin bindings and timing-condition constants.
One compatible `Resize` also supports sequential modules and updates launch and
capture timing. Combinational batches additionally support `Replace`, `Insert`
and `Rewire`; these can append bits and instances. The caller owns equivalence,
optimization policy and the matching source-netlist edit.

`TrialOptions::cell_budget` bounds combined timing and Boolean-coverage cell
visits in that transaction. Optional boundary limits allow exact early rejection.
Returned work separates timing, coverage, touched connectivity and exceptional
ordering work. Ordinary edits journal only touched state. Backward dependencies
may require a full ordering pass; accepted insertions maintain the traversal
list once. Registered feedback retains the bounded least-fixed-point coverage
fallback. Direct input-slew/output-load updates, deletions, module-port changes,
original-alias rewrites and registered topology changes are outside this API.

Resizing and fanout optimization are clients of this engine. Resize selection,
area/timing merit and electrical acceptance remain in the resize client; fanout
candidate construction and metrics remain in its client. The engine's read-only
graph/timing/electrical observations do not impose those policies.
`netlist::fanout_opt::optimize_fanout` accepts an owned `MappedNetlist`, including
one read from GV. `FanoutOptimizationOptions` exposes search bounds, timing/load
assumptions, full-STA evaluation, and per-candidate oracle verification; reports
retain profiling measurements. The CLI exposes the common path through
`aig-tech-map --strategy area`, which enables fanout optimization by default.
Use `--fanout-opt=false` for raw area mapping; other strategies retain their
existing defaults.

A shared edit plan supplies both netlist emission and timing-engine mutations.
Trial metrics contain only `area`, `cell_count`, and `worst_delay`; baseline,
accepted-decision, and final metrics retain electrical diagnostics. These
reporting changes do not change candidate eligibility or the delay/area objective.
The incremental profile no longer includes `metrics_seconds`.

## Native netlist resizing

`netlist::resize::resize_netlist` operates on a flat mapped module and its nets,
symbol interner, and timing-enabled Liberty library. It changes only cell types:
connections, port/register polarity, and the number of instances stay fixed.
Both combinational cells and compatible FFs can be resized. Families require
matching pin names/directions, output functions, and sequential semantics;
unsupported or `dont_use` alternatives are not selected. There is no buffer
insertion/removal, pin swapping, or logic remapping in this pass.

The bounded timing phase visits near-critical paths, evaluates nearby cell sizes,
and prices area-growing moves using an area-delay-squared merit. The subsequent
area phase tries smaller equivalents against the timing checkpoint reached by
the first phase. Both phases evaluate real pin capacitances and Liberty delay,
slew, clock-to-output, and setup tables. None of the four PI/register-to-PO/register
boundary classes may worsen; neither may existing capacitance, slew, or fanout
violations. Timing prioritizes register-to-register paths when present.

Incremental timing propagates changed arrival/slew frontiers in topological order
and stops at unchanged frontiers, without approximate cutoffs. Each trial has a
sparse undo log: rejected, erroneous, or budget-interrupted substitutions restore
the incumbent. Replacements that change arc-less output constants used to
evaluate timing conditions are skipped. Independent full STA using the `gv-stats`
model verifies the final result before replacing the caller's module. This is
max-delay/setup timing with ideal clocks, not hold, clock-tree, placement, or
routed timing signoff.

`ResizeOptions` sets deterministic round, candidate, and evaluation limits; these
are work bounds rather than a wall-clock deadline or a local-optimality guarantee.
Evaluation visits count timing propagation and, when a replacement may change
inferred Boolean constants, functional timing-coverage propagation. With `N`
input instances (including FFs), defaults are:

- `max(64, ceil(N / 512))` candidate sites per timing round.
- `max(50_000, 4 * N)` instance evaluation visits across all timing rounds.
- `max(100_000, 8 * N)` instance evaluation visits across all recovery rounds.
- Four timing rounds, two recovery rounds, and eight alternatives per site.

One substitution trial can evaluate many instances as timing changes propagate.
Phase allowances are independent, do not reset each round, and do not borrow
unused work from each other. There is no separate trial-count limit or combined
phase-work limit, and recovery has no candidate-site cap. Optional candidate and
evaluation overrides use these formulas when `None`; an explicit zero evaluation
budget permits no evaluation work. `ResizeOptions::work_limits(N)` resolves the
effective limits, which are also included in `ResizeReport`.

The default area allowance is 2% of the original input area and never compounds
across moves. `ResizeReport` records initial, upsized, and final area and boundary
delays, accepted moves (including FF changes), work, and budget exhaustion. The
`propagated_instances` count includes both kinds of evaluation visits.
Each phase reports `stop_reason`: `disabled`, `no_improvement`, `round_limit`, or
`propagation_limit`. `no_improvement` refers only to the bounded candidate search,
not proof of local optimality. `budget_exhausted` is true if either phase stops
because its evaluation budget is exhausted.
The existing `upsize`/`downsize` report fields and round-option names refer to the
timing-search/area-recovery phases respectively; timing search may move in either
size direction.
Set `resize_registers: false` to freeze FFs. Library callers supply `StaOptions`
explicitly when they want a particular IO model; the `gv-resize` CLI defaults its
output load from the library's `default_output_load` metadata.

```rust,ignore
use xlsynth_g8r::netlist::resize::{ResizeOptions, resize_netlist};

let options = ResizeOptions { timing, ..ResizeOptions::default() };
let report = resize_netlist(&mut module, &nets, &mut interner, &library, &options)?;
```

The library pass needs no external synthesis executable. The `gv-resize` driver
command exposes it separately so the same mapped input can be evaluated before
and after resizing. See the driver README for flags and `FUZZ.md` for the shared
G8R generation, sequential simulation, and timing-invariant campaign.

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

### Nonzero predicates on bounded logical shifts

Preparation replaces a nonzero/zero test of a logical shift or a contiguous
window of it with predicates on the input bits that survive. For example, for
`x: u23` and `k: u3`, `or_reduce(x << k)` becomes
`or_reduce(x[0:16]) | OR(i=16..22, x[i] & (k <= 22-i))`.
The common bits need one reduction; only the boundary bits need conditions.
Left and right logical shifts preserve zero-fill and overshift behavior.
Single-use slices and binary selects can carry the scalar test to the shift.
Shared numeric consumers block removal; shared boolean results are allowed.

The rule uses a bounded structural estimate, not speculative AIG construction:
source width at most 256, at most five effective amount bits, at most 32 traversed
plan nodes and 64 emitted input groups per predicate. Arithmetic right shifts,
constant data/amounts, and shapes that fail the estimated area benefit stay with
the existing lowering. For whole-word tests on data of at most 32 bits, wider
amounts use one upper-bits-zero guard and the low `ceil(log2(width))` bits. The
surviving-bit conditions still reject low-bit overshifts for non-power-of-two widths. The wide
path declines partial windows, wrapped one-hot data, and visible constant low
control bits to preserve native simplifications that the generic cost estimate
does not model. These admission checks share a 32-step bound; they leave the
existing narrow-amount rule unchanged. This is separate from the bounded affine
amount/window rewrite, which runs first and can reason about much wider shift amounts.
