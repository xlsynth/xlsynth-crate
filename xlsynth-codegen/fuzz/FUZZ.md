# xlsynth-codegen fuzz targets

All targets use only synthetic, public XLS block IR. Coverage-guided bytes select
block structure and independent stimulus entropy; legacy graph-only inputs use
a deterministic hash of the resulting IR for stimulus. Routine campaigns should pass
`--sanitizer none`.

## Generation profiles and input format

The two mixed semantic campaigns use `Profile::NativeSemantics` and
`Profile::StockXls`. Both mix scalar/aggregate combinational blocks with
synchronous register blocks. Limits are 48 nodes, 256-bit scalars, three levels
of aggregate nesting, 32 scalar aggregate leaves, eight array elements, six
tuple fields, four inputs, three outputs, and three registers. Boundary widths
around powers of two are favored; widths above 64 remain relatively sparse.
Signed/unsigned division and modulus have a separate 16-bit operand/result cap
in both profiles and the focused targets using their options. They remain enabled
for Icarus, stock-XLS differential, and Yosys mapping/formal checks. Other
operations retain their wider limits except for the multiply operand cap below.
This bounds divider synthesis cost and avoids Icarus multiword-division
pathologies without skipping generated cases.
Wide div/mod semantics are outside these random codegen campaigns; the backend
itself is not restricted to 16-bit division. The reusable PIR generator leaves
`max_div_mod_bit_width` unset by default. Both profiles cap signed/unsigned
multiply operands independently at 64 bits. Result widths remain independent,
up to the general 256-bit limit, so truncated, full-product, and extended
results remain covered. This caps multiplication complexity without narrowing
other operations or restricting the backend's accepted widths. The reusable
PIR generator leaves `max_multiply_operand_bit_width` unset by default.
The reusable PIR generator samples arithmetic data widths before operands, with
a 10% one-bit choice and the remaining probability shared
between eligible ranges 2–8, 9–16, 17–32, 33–64, and 65–maximum. Independent
choices mix boundary widths and uniform widths within each range. Exact typed
values are reused; missing widths use budgeted slices/extensions of existing
computations, preferring wider sources. A second compatible operand may be
constructed to avoid always reusing a lone new value on both pins. End-of-budget
operations can still reuse available widths. This is a sampling policy, not a
guarantee about all emitted node widths or input-value entropy.
Multiply generation mixes equal/independent operand widths and deliberately
samples truncated, full-product, and extended-result categories. Nonempty array
index/update shapes are favored when legal, while empty-index operations remain
possible. Dead nodes are allowed. With this policy, an incomplete body
operation is rolled back if finite entropy runs out; reserved wiring headers and
short interfaces retain deterministic padding. Indexed selects permit wide
selectors independently of their built-in 16-case limit; priority/one-hot
selects allow up to the built-in 256-case limit. Multiply input/result widths
are independent within those limits. These profiles select the proven-safe
array-assumption mode, so `assumed_in_bounds` is generated only with width-safe
or literal-safe indices and arbitrary simulation stimuli honor the promise.

The native profile includes all six public `ext_*` operations, zero-bit values,
zero-bit ports/registers, and aggregate gating. The stock profile excludes
extensions and zero-width generation. Known stock codegen limitations on
array-valued identity/gating and zero-bit comparisons/extensions/gating are counted
as skips, not successful checks. Unknown failures remain fatal. Calls, loops,
partial products, assertions, covers, traces, and event tokens are excluded
from both profiles. Existing hierarchy/extern targets remain separate
structural tests; the existing focused partial-product target is unchanged.

Version 1 mixed-profile inputs have a 48-byte prefix: `XBCF`, version byte `1`,
three reserved bytes, 32 stimulus-seed bytes, and eight generation/codegen-option
bytes. Short recognized headers are zero-padded. Unknown explicit versions are
rejected. Inputs without the magic use the legacy graph-only decoding. The
generator version/options still determine how a saved corpus decodes.

The graph region begins with `8 * (max_outputs + 2 * max_registers)` bytes reserved
exclusively for final connectivity. For these mixed profiles the header is
72 bytes: three little-endian u64 output choices, followed by three pairs of
u64 next-state/load-enable choices. All maximum slots are consumed even when
the generated block uses fewer outputs/registers. Missing header bytes are
zero-padded; inputs shorter than the header have an empty body stream.

The remaining bytes choose input/register types, reset metadata, body nodes,
and actual output/register counts. After body construction, each reserved
word selects an existing eligible node modulo the candidate count. Next-state
choices are type-compatible with their register; load-enable choices select
no enable or an existing `bits[1]` node. Output types follow their selected
nodes. Header-only mutations change connectivity but never body nodes. Dead
nodes are allowed and nothing forces operations to become observable.
Corpus bytes are reproducible with the same generator version/options; the
header format changes how older corpora decode.

The stimulus region can mutate independently of graph and wiring. Sixteen
stimulus slots mix zeros, ones, isolated/cleared bits, alternating bits, signed
extrema, small values near graph-derived widths/bounds, correlated equal
operands, extrema versus -1, and ones versus 1. Four slots use uniform values.
Patterns recurse through aggregate leaves and support widths above 64. Initial
state is independently patterned/random. Synchronous reset sequences include
deasserted startup, consecutive asserted cycles, and later pulses; asynchronous
reset events are not tested by these profiles.

Interface types are occasionally reused. Operand selection mixes uniform,
recent-node, and older-node windows, allowing chains, fanout, and reconvergence.
Option bytes choose combinational, general-sequential, or stage-aware
feed-forward topology. The feed-forward mode has two or three register
boundaries, uses stage-local value pools, and still takes final D/enable choices
from the wiring header. Both scalar and aggregate pipelines are checked under
`layout=none` and `layout=pipeline`, varied inlining depth, and
separate-assignment emission. Only feed-forward cases request pipeline layout.
Required bounds
checks remain enabled; disabling them would intentionally change semantics.

## Coverage reports and replay

The two mixed targets print cumulative `codegen-coverage` JSON on the first
case, every 4096 cases, and after 30 seconds at the next completed case.
This semantic-feature census complements libFuzzer
edge coverage. Counters distinguish generated operations, operations in
successfully checked graphs, and live operations feeding outputs/register
updates (including enables). Missing-operation lists are reported separately
for each category. Shape/width/attribute histograms describe **all generated
cases**, including skipped ones; they are not proof of semantic coverage.
Skipped cases have named reasons. Checking uses 16 input vectors for
combinational blocks or 24 cycles including reset stimuli for sequential
blocks, with independently evaluated outputs and next state.

The report also counts producer→consumer pairs, operand/result width
relationships, register dependency depths, and explicit feedback. Div/mod
opcode/width counts are reported in `div_mod_widths` for generated graphs. Observed
behavior counters distinguish in-/out-of-bounds array accesses, selected/default
arms, zero/nonzero priority inputs, zero divisors, and reset/enable combinations.
They are collected from live PIR nodes on vectors that passed **all required
oracles**; they are not simulator code coverage, and dead nodes do not earn
observed-behavior coverage. Generation-only reports never claim checked vectors.

Reports identify the generator revision and entropy engine. Scalar data-operand
width histograms exclude selector/index/reset/enable roles and distinguish
generated, checked, and checked-live nodes. Multiply category histograms do the
same for equal/mixed operands and truncated/full/extended results. Exact unique
graph and graph/stimulus-pair counts are reporter-local, not additive across
workers. The standalone corpus reporter counts unsupported input-format versions
as rejected inputs rather than treating them as checked graphs.

## Fresh-random evaluation

`cargo run --release --example random_eval -- --profile native --seed 1 --samples 1000 --artifact-dir /path/to/reproducer` generates graphs using
non-depleting `RngEntropy` with independently chosen body-node budgets from 4
through the configured maximum (48). It calls the same `FuzzCase` checker as the
guided targets, with the same two profiles, tool checks, stimuli and width caps.
`--duration SECONDS` bounds the run; `--case-seed N` replays a specific graph,
options and stimulus sequence. `--no-check` produces a generation-only census.
This is structured fresh random generation, not uniform sampling of all graphs.

Random workers do not mutate or reload a corpus. Run `random_eval` directly
for structured fresh-random testing; use OS-level process limits when running
many workers concurrently.

Guided workers with `-artifact_prefix` and random workers with `--artifact-dir`
preserve the in-flight IR, generator revision, codegen options, stimulus seed,
initial state and full trace before external checking. Guided bytes or the
fresh-random case seed accompany these files. They are overwritten on subsequent
cases, so crashes/timeouts leave the reproducer in place without saving every
passing case. A random worker also saves its seed before graph generation.

The semantic targets require `iverilog` and `vvp`, resolved through
`XLSYNTH_IVERILOG_PATH` / `XLSYNTH_VVP_PATH` or `PATH`. Selecting such a target
always requires both executables; missing tools fail instead of skipping the
semantic check. No extra Icarus feature is needed in this standalone fuzz crate.
Icarus compiles
the actual emitted RTL once per design, then a persistent simulation process
checks all vectors/cycles. Port widths and register probes come from block IR;
there is no Rust SystemVerilog parser/evaluator in this path. Arbitrary-width
four-state results are compared exactly; unexpected X/Z values are failures,
not converted to zero. Initial register values are set explicitly in both
oracles, including registers without reset. Sequential checks compare outputs
before each edge and every committed register value after nonblocking updates.

The stock-XLS oracle uses `--separate_lines=true` to materialize narrow
expressions before array-update comparisons with `genvar` values. Stock XLS's
inlining can widen those expressions and change their values. Native inlining
options remain randomized, and the IR and stimulus are unchanged. A directed
computed-index regression checks PIR, both emitted RTLs, and mapped native gates.
The mapped oracle prunes dead logic and consolidates repeated mux inputs before
constant folding and bit-level mapping. It shares only combinational cells and
collapses AND/OR/XOR chains into reductions before mapping repeated operands.
Shared reduction prefixes remain separate so remapping does not duplicate their
common inputs into each consumer.
Register cells are excluded from sharing, so arbitrary independent initial state
and every next-state bit remain checked.

External RTL compilation and each simulation response have a 60-second timeout;
Yosys/ABC mapping has a 120-second timeout. Mapping/timeout failures are fatal,
not successful checks or silent width-dependent skips.
timed-out child process groups are terminated. Set libFuzzer's `-rss_limit_mb`
and `-timeout` explicitly for campaigns. The RSS setting does not bound memory
used by external compilers/simulators; use OS-level job limits when running many
workers. The periodic counters are process-local; use corpus replay for a final
combined report (a killed worker cannot print a final snapshot).

From this directory, create reproducible seeds, run a campaign, and produce a
final report:

```sh
cargo run --release --example coverage -- \
  --profile native --samples 1000 --seed 1 --bytes 2048 \
  --write-corpus corpus/fuzz_codegen_native_semantics
cargo fuzz run --sanitizer none fuzz_codegen_native_semantics -- \
  -max_total_time=60 -max_len=4096 -timeout=180 -rss_limit_mb=2048
cargo run --release --example coverage -- \
  --profile native --corpus corpus/fuzz_codegen_native_semantics --check \
  > native-coverage.json
```

Use `--profile stock-xls` and `fuzz_codegen_xls_differential` for the other
campaign. `--corpus` accepts a directory tree or a single reproducer file.
Without `--check`, the example reports generation coverage only. Generated
samples can be saved with `--write-corpus`; bytes are saved before checking so
a failing sample can be replayed. Seed generation is generic, not a collection
of injected special cases. Missing stock tools are an error for an explicitly
requested stock check, not an all-skipped campaign.

## Liberty configuration

The mapping targets, stock-XLS profile, and fuzz library tests that exercise
them require `XLSYNTH_YOSYS_PATH` and `XLSYNTH_LIBERTY_FILES`. The latter is a
comma-separated list of explicit Liberty paths, consumed by both Yosys/ABC
and our netlist evaluator. Relative paths are resolved from the starting
working directory. The setup is independent of any particular PDK;
no LEF, GDS, or physical-design files are needed. Ordinary workspace
`yosys-tests` perform parsing/proofs without a cell library.

For the small CI harness library, run from this directory:

```sh
export XLSYNTH_YOSYS_PATH="$(command -v yosys)"
export XLSYNTH_LIBERTY_FILES="$PWD/testdata/public_cells.lib"
cargo fuzz run --sanitizer none --features external-yosys \
  fuzz_codegen_yosys_combo -- -max_total_time=30
```

Missing Yosys or Liberty configuration fails any selected Yosys check. Ordinary
workspace `yosys-tests` / `iverilog-tests` features do not control fuzz targets.
The fuzz crate's existing `external-yosys` feature selects its mapping/formal
targets; its stock-XLS profile always requires mapped-netlist evaluation.
Use a recent Yosys with SystemVerilog support (CI pins OSS CAD Suite 2026-08-29).

Absent, empty, nonexistent, or archive inputs produce a setup error with a
configuration example. Supply files from an existing PDK installation; the
harness does not install or extract PDKs, or silently fall back to the small
test library. For a stock-XLS campaign, validate configuration before
generating any samples (also requires Icarus and installed stock-XLS tools):

```sh
cargo run --example coverage -- --profile stock-xls --samples 0 --check
```

For real-library coverage, choose files from one compatible library family
and corner. Include the sequential library for blocks with registers. For
ASAP7, use `asap7sc7p5t_28`, RVT slow/SS NLDM: AO, INVBUF, OA, SEQ, and SIMPLE.
For example, in Bash, with all five files installed under `~/pdks/asap7`
(adjust `ASAP7_LIB_DIR` for another installation):

```bash
ASAP7_LIB_DIR="$HOME/pdks/asap7"
asap7_files=(
  "$ASAP7_LIB_DIR/asap7sc7p5t_AO_RVT_SS_nldm_211120.lib"
  "$ASAP7_LIB_DIR/asap7sc7p5t_INVBUF_RVT_SS_nldm_220122.lib"
  "$ASAP7_LIB_DIR/asap7sc7p5t_OA_RVT_SS_nldm_211120.lib"
  "$ASAP7_LIB_DIR/asap7sc7p5t_SEQ_RVT_SS_nldm_220123.lib"
  "$ASAP7_LIB_DIR/asap7sc7p5t_SIMPLE_RVT_SS_nldm_211120.lib"
)
export XLSYNTH_LIBERTY_FILES="$(IFS=,; echo "${asap7_files[*]}")"
cargo run --example coverage -- --profile stock-xls --samples 0 --check
cargo fuzz run --sanitizer none --features external-yosys \
  fuzz_codegen_yosys_sequential -- -max_total_time=30
```

Supply another library's compatible combinational/sequential files through
the same variable. Record the source revision, selected corner, file hashes,
and Yosys version with a campaign so results can be reproduced. These checks
compare semantics; they do not report PPA or select a timing target.

The same configuration supports `fuzz_codegen_yosys_sequential` and
`fuzz_codegen_yosys_formal`. These independent netlist simulation/SAT oracles
retain narrower scalar complexity limits to bound mapping cost. The broad
stock-XLS profile additionally requires mapped-netlist evaluation, with its
existing widths/shapes intact. These paths use our Liberty-backed gate evaluator,
not a Rust SystemVerilog parser/evaluator.

Use a real Liberty set such as ASAP7 for campaign coverage. The small public
test library is also supported for harness regressions; it is not an ASAP7 run.
Each accepted case compares four results using exactly the same inputs/state:

1. PIR block evaluation.
1. Native generated SystemVerilog in Icarus.
1. Stock-XLS generated RTL in Icarus.
1. Native generated SystemVerilog → Yosys/ABC → Liberty-mapped netlist → our
   G8R gate evaluator, including mapped flip-flops.

Yosys reads the unmodified emitted RTL, including generate loops and helper
functions. After elaboration, register Q nets are exposed as observation ports.
The mapped simulation oracle uses the explicit ABC sequence `strash; dretime; map`, avoiding SAT sweeping of wide arithmetic. This is a correctness oracle,
not a synthesis-QoR measurement; operand widths and comparisons are unchanged.
Register-eliminating/merging optimization is omitted so arbitrary independent
initial states, constant-data registers, and otherwise unobservable registers
remain comparable. Q correspondence comes from imported netlist connectivity,
including output inversion, not mapped instance-name guessing. Every committed
state is checked, including the final edge. This is a two-state, zero-delay
functional check, not timing/SDF or four-state mapped-cell simulation.

The stock-XLS differential target discovers `block_to_verilog_main` through
`XLSYNTH_BLOCK_TO_VERILOG_PATH`, `$XLSYNTH_TOOLS/block_to_verilog_main`, or
`PATH`, in that order. For example:

```sh
XLSYNTH_TOOLS=/path/to/xls/tools cargo fuzz run --sanitizer none \
  fuzz_codegen_xls_differential -- -max_total_time=30
```

## Direct fuzzing and parallel workers

Run targets directly with Cargo; no Python launcher is required. From
`xlsynth-codegen/fuzz`, seed a corpus if desired and start guided workers:

```sh
cargo run --release --example coverage -- \
  --profile native --samples 200 --seed 1 --bytes 2048 \
  --write-corpus corpus/fuzz_codegen_native_semantics
cargo fuzz run --sanitizer none fuzz_codegen_native_semantics -- \
  -max_total_time=10800 -timeout=180 -rss_limit_mb=4096 -jobs=4 -workers=4
```

Use `fuzz_codegen_xls_differential` and `--profile stock-xls` for stock
comparison and mapped-netlist evaluation. Mapping-only targets additionally
select `--features external-yosys`. Existing corpus contents are reused by
libFuzzer. Save a final generation census or explicitly check every input with:

```sh
cargo run --release --example coverage -- \
  --profile native --corpus corpus/fuzz_codegen_native_semantics
cargo run --release --example coverage -- \
  --profile native --corpus corpus/fuzz_codegen_native_semantics --check
```

Every target requiring external tools runs Rust preflight in libFuzzer's
initialization hook. Configuration errors print an actionable message and
exit with status 2 before processing samples, without creating a crash
artifact. Icarus tools are checked for native simulation; stock checks also
validate stock codegen and Yosys mapping. Mapping preflight loads the configured
Liberty files, requires flip-flops for sequential targets, and probes Yosys/ABC
mapping plus netlist import. These setup probes are separate from fuzz inputs.
The `coverage --check` and `random_eval` examples use the same preflight,
including zero-sample runs; generation-only commands do not require the tools.

Executable and Liberty paths are resolved before temporary working-directory
changes, so relative configuration paths are interpreted at process startup.
Tool execution captures diagnostics and uses bounded wall time. On Unix,
completion, failure, and timeout clean up the tool's process group, including
children that outlive its leader.

libFuzzer's RSS limit measures the fuzzer process, not Yosys/ABC/Icarus children.
Use OS-level resource limits for aggregate memory and disk limits on larger
runs. `-max_total_time` can overrun while an in-flight sample finishes.

## `fuzz_codegen_parser_robustness`

Parses arbitrary UTF-8 as block packages and passes accepted block tops through
the SystemVerilog emitter twice. Failures expose parser or backend panics,
nondeterministic diagnostics, and nondeterministic successful output.

## `fuzz_codegen_determinism`

Generates valid typed block packages with scalar and aggregate ports, arbitrary
multiply widths, registers, load enables, and synchronous or asynchronous
resets. It checks stable emission across repeated calls and PIR
parse/print roundtrips. Failures expose ordering instability, mutable emission
state, or metadata lost across serialization.

## `fuzz_codegen_combinational_semantics`

Generates scalar, register-free blocks and compares the emitted SystemVerilog
against the independent PIR interpreter on deterministic input samples using
Icarus simulation. Failures expose incorrect arithmetic,
signedness, comparisons, shifts, selection, gating, or port connectivity.

## `fuzz_codegen_native_semantics`

Uses the native mixed profile to compare Icarus simulation of generated
SystemVerilog with PIR evaluation. Scalar arithmetic, public extensions, nested
aggregates, and scalar/aggregate register feedback can occur in the same graph. Failures
expose interactions across operation families, widths, packing, out-of-bounds
handling, reset/load-enable behavior, and final wiring. Semantic-feature
coverage distinguishes generated, checked, and checked-live operations.

## `fuzz_codegen_sequential_semantics`

Generates synchronous blocks with registers, feedback, load enables, and resets.
It compares cycle-by-cycle output and committed register values against an
independent PIR block evaluator. Failures expose changed register state,
feedback, reset priority or polarity, and load-enable behavior.

## `fuzz_codegen_aggregate_semantics`

Generates register-free blocks with tuples, arrays, nested aggregates, and
aggregate operations. It compares every flattened output against the PIR
evaluator. Failures expose aggregate packing order, indexing, updates, slices,
selection, or out-of-bounds semantic differences.

## `fuzz_codegen_partial_products`

Generates signed and unsigned partial-product blocks with independently chosen
operand and result widths. It checks the modular sum of emitted product pairs
rather than requiring a particular internal pair representation. Failures
expose incorrect multiplication, sign extension, truncation, tuple layout, or
partial-product invariants.

## `fuzz_codegen_options`

Varies top selection, module naming, expression layout, inline depth, separate
assignments, and type-annotation settings on scalar/aggregate combinational,
feedback, and feedforward pipeline blocks. It checks deterministic emission,
outputs, and every next-state value. Required bounds checks stay enabled and
asynchronous reset is excluded. Failures expose option interactions or
formatting-induced functional changes.

## `fuzz_codegen_hierarchy`

Builds parameterized synthetic hierarchies with two instances of one child
block. It checks deterministic topological emission, dependency deduplication,
module selection, and independent instance connectivity. A required Icarus
simulation of a generated self-checking SystemVerilog testbench exercises
both child instances over deterministic stimuli and independently verifies
their combined arithmetic result.

## `fuzz_codegen_extern`

Builds parameterized synthetic foreign-function instantiations. It checks that
their external SystemVerilog templates preserve instance identity, data-input
bindings, and result bindings without emitting a fictitious external module.
An independently authored public behavior model is appended and a required
Icarus simulation of a self-checking testbench verifies external-instance
connectivity and observed values over deterministic stimuli.

## `fuzz_codegen_xls_differential`

Generates public scalar combinational, aggregate combinational, and synchronous
register blocks, then independently lowers each one through native codegen and
stock XLS `block_to_verilog_main`. Deterministic stimuli compare corresponding
output values and register behavior from both backends, plus the independently
Yosys/ABC-mapped native netlist, against PIR evaluation. Failures expose semantic differences in arithmetic,
signedness, shifts, comparisons, selections, aggregate packing, register/reset
behavior, width handling, and port connectivity. The required stock-XLS
executable is discovered through `XLSYNTH_BLOCK_TO_VERILOG_PATH`,
`XLSYNTH_TOOLS`, or `PATH`. Stock XLS emits Verilog for this reference oracle:
its unpacked-array assignment patterns in SystemVerilog mode are unsupported
by Icarus. Both outputs are simulated independently by Icarus without any Rust
RTL parsing or rewriting. The native DUT always emits SystemVerilog.
Known stock limitations are enumerated above and in `stock_xls_skip_reason`;
skipped and checked cases are reported separately. A checked case requires all
four results to agree; missing mapping tools/libraries are fatal.

## `fuzz_codegen_yosys_combo`

Emits random combinational blocks, maps the SystemVerilog through an external
Yosys/ABC/Liberty flow, and compares the mapped gate-netlist outputs against
independent PIR evaluation. This target requires the `external-yosys` feature,
`XLSYNTH_YOSYS_PATH`, and `XLSYNTH_LIBERTY_FILES`.

## `fuzz_codegen_yosys_sequential`

Emits synchronous blocks with a reset on every register, maps their
SystemVerilog through Yosys and Liberty flip-flop mapping, then compares
post-reset cycle outputs against independent PIR evaluation. This target
requires the `external-yosys` feature and the same external configuration as
the combinational Yosys target.

## `fuzz_codegen_yosys_formal`

Emits random combinational blocks, maps them through Yosys/Liberty, and proves
the resulting gate graph equivalent to independent PIR-to-G8R lowering with
CaDiCaL. This target requires the `external-yosys` feature and the same
external configuration as the other Yosys targets.
