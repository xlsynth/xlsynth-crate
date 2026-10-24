# xlsynth-g8r Fuzz Targets

Mapping targets selected with `external-yosys` require an explicit
`XLSYNTH_YOSYS_PATH` and comma-separated `XLSYNTH_LIBERTY_FILES`. Missing or
invalid setup exits with status 2 during libFuzzer initialization, before
generating IR or crash artifacts. Startup also checks mapping and netlist
import. Use compatible standard-cell Liberty files from one library/corner,
including flip-flop cells for sequential checks. A full PDK is unnecessary.
See [Liberty configuration](../../xlsynth-codegen/fuzz/FUZZ.md#liberty-configuration)
for the shared setup, including an ASAP7 configuration example.

## `fuzz_sequential_cleanup_equiv`

Generates synchronous block IR with reset values on every register and at least
one observable output, lowers it to
G8R, and compares an independent block evaluator, original G8R, and cleaned G8R.
Each sample uses eight independent startup assignments and 16 cycles. Reset is
asserted across the first clock edge and occasionally later; cleaned outputs are
compared only after the first edge. Original lowering is checked on every cycle,
including its register state. Also checks packed interfaces, reduction counts,
determinism, and idempotence. This catches reset/enable priority errors, unsound
state substitutions, and reconstruction bugs without requiring a state mapping.
No solver, external synthesis tool, or PDK is needed. The generator is generic
within the fully resettable domain; it does not inject special cleanup motifs.
The smoke sweep requires dead, constant, and duplicate removal to be exercised.
Reserving an output before constructing the body prevents vacuous output checks
when generation exhausts its node or entropy budget.

The oracle does not cover arbitrary partially reset feedback graphs. Focused
unit tests cover unreset data masked by resettable valid state and unreset
substitutions with corresponding startup assignments.

```sh
cargo fuzz run --sanitizer none fuzz_sequential_cleanup_equiv -- -max_total_time=300
```

## `fuzz_random_block_g8r_equiv`

Directly generates random block IR, lowers supported synchronous blocks to
sequential G8R, and compares cycle-by-cycle external outputs and register state
against an independent block-level reference evaluation. The stimulus sequence
and initial values are produced by a deterministic RNG seeded from the
generated IR, rather than from coverage-guided bytes. Registers without reset
values receive random initial state; reset-bearing registers start from their
declared reset value. A reset input is asserted with 50% probability on the
first cycle and 10% probability on later cycles. The target requests
synchronous-only reset generation and treats any asynchronous reset as a
generator-contract failure.

The essential property is that lowering a generated synchronous block to
sequential G8R preserves each cycle's visible outputs and committed register
state across random input sequences, including feedback, load-enables, mixed
reset coverage, aggregate ports/registers, and zero-output blocks with state.
Failures expose changed register feedback, load-enable, or reset semantics;
disagreement while flattening aggregate block ports or register values;
divergent visible outputs or committed register state on any simulated cycle;
or generated block metadata that cannot be lowered by the supported
synchronous block-to-G8R path.

## `fuzz_random_block_gv_eval_combo_equiv`

Generates bounded register-free block IR, lowers it to combinational G8R,
emits packed-port Verilog, maps it through the Yosys executable named by
`XLSYNTH_YOSYS_PATH` and ABC against the comma-separated Liberty files in
`XLSYNTH_LIBERTY_FILES`, then loads the mapped netlist through `gv-eval`.
The target is gated by the `external-yosys` feature because it requires
that external toolchain and Liberty data at runtime. Input samples use a
deterministic RNG seeded from the generated IR rather than coverage-guided
bytes.

The essential property is that combinational block IR evaluation and
Liberty-backed `gv-eval` agree on every flattened output for each random input
sample, including aggregate ports and zero-output blocks. Failures expose
block-to-G8R flattening disagreements, RTL emission or Yosys/ABC mapping
failures, unsupported mapped Liberty cell formulas, netlist input/output shape
changes, or visible output divergence after technology mapping.

## `fuzz_random_block_gv_eval_sequential_equiv`

Generates bounded block IR with at least one register and a synchronous reset
value on every register, lowers it to sequential G8R, emits packed-port
Verilog, maps FFs and combinational logic through the Yosys executable named by
`XLSYNTH_YOSYS_PATH` against the comma-separated Liberty files in
`XLSYNTH_LIBERTY_FILES`, then projects the mapped netlist directly into the
labeled sequential transition-AIG form used by `gv-eval`. The target is gated
by the `external-yosys` feature because it requires that external toolchain
and Liberty data at runtime. A deterministic RNG seeded from the generated IR
supplies arbitrary initial block state and random non-reset inputs. Reset is
asserted on cycle zero and deasserted for all later cycles; cycle-zero outputs
are intentionally ignored because the mapped state elements need not
correspond to the source registers. The mapped trace is also passed through
the labeled sequential toggle report path.

The essential property is that block IR evaluation and Liberty-backed
sequential `gv-eval` agree on every visible flattened output after the reset
edge, without matching mapped register names, count, or state layout. Failures
expose block-to-G8R reset/load-enable/feedback errors, sequential RTL emission
or Yosys `dfflibmap`/ABC mapping failures, unsupported Liberty FF metadata,
aggregate port flattening disagreements, netlist interface shape changes, or
visible output divergence on any post-reset simulated cycle. It also surfaces
crashes or shape inconsistencies in sequential labeled toggle accounting.
Zero-output samples still exercise the full lowering, mapping, loading,
simulation, and toggle-report path for crash detection.

## `fuzz_gatify`

Generates bounded gatify-supported PIR directly with
`xlsynth_pir::ir_random`, including PIR extension operations, lowers it with
gatify with and without folding, and proves that each generated gate function
remains equivalent to the original PIR function. Product-pair operations and
the opt-in XLS `gate` operation remain excluded from this target.

## `fuzz_bulk_replace`

Generates bounded AIG graphs and production-style acyclic cone/cut
substitutions, applies bulk replacement, checks graph invariants and output
shape, and compares evaluations before and after replacement. Failures expose
unsound substitutions, accidental cycles, or malformed rebuilt graphs.

## `fuzz_node_provenance`

Generates gatify-supported PIR directly, including PIR extension operations,
lowers it without gate folding, and checks
that every surviving gate node carries sorted, deduplicated provenance IDs
which originate in the source PIR graph. Failures expose lost, invented, or
misordered provenance during initial lowering.

## `fuzz_node_provenance_with_opts`

Generates gatify-supported PIR directly, including PIR extension operations,
gatifies it, applies FRAIG and cut-database
rewriting, then checks provenance IDs on surviving gates. Failures expose
provenance corruption introduced by gate-level optimization.

## `fuzz_sequential_legalization_equiv`

Generates generic synchronous stateful block IR with optional enables, optional
resets, feedback, and zero-width values. It lowers to explicit-control
`SequentialGateFn`, round-trips text or binary serialization, legalizes the
controls, and evaluates both forms against the independent block interpreter.
Eight arbitrary common startups and 16 cycles are checked per sample, including
cycle zero; no reset warm-up is assumed, so unreset state is covered. The target
checks external outputs and every register's next state, retained metadata,
serialization fidelity, and legalization idempotence. It catches reset/enable
priority or polarity errors, incorrect holds or simultaneous updates, bit-order
mistakes, state loss, and dropped serialized controls. Unlike register-cleanup
fuzzing, this is exact equivalence, not post-reset equivalence under startup
don't-care semantics. The deterministic smoke test requires enabled, resettable,
and unreset register coverage without injecting directed graph patterns.

Run from `xlsynth-g8r` with:

```shell
cargo fuzz run --sanitizer none fuzz_sequential_legalization_equiv -- -max_total_time=300
```

## Shared direct-G8R generation

All direct graph consumers use the library API
`xlsynth_g8r::gate_random::generate_gate_design` with `RandomGateOptions`.
The previous general `FuzzGraph` builder and bulk-replacement-only
`FuzzGateGraph` builder have been removed. Function/block-IR-derived targets
continue to use the IR generators; they provide complementary lowering coverage.

Profiles control interface counts and packed widths (including zero widths),
the actual AIG-node budget, Boolean construction operations, folding/hashing,
optional registers and control polarities, endpoint sharing, and output
observability. Optional chain, redundant-pair, and fanout distributions are
disabled by default and explicitly enabled only by the bulk-replacement target.
Seeded RNG sweeps (independent of libFuzzer guidance) require emitted and live
coverage of all enabled Boolean operations, varied interface shapes/widths,
feedback and feedforward state, all reset/enable combinations, both polarities,
and shared nonempty data/control endpoints. Data wiring samples both whole-port
reuse and independently selected bits.

Graph entropy is separate from the shared `xlsynth_pir::random_inputs` sampler
used for concrete inputs and startup state.

Text/AIGER round trips, both gate-transform targets, cut rewriting, dynamic
hashing/depth, cut replacement, and bulk substitution use the combinational
profile. Their input schema is now byte-stream graph entropy rather than the
older derived graph-description structs; old corpus bytes remain usable but
do not necessarily reconstruct their previous graph. Edit/substitution
sequences remain target-specific inputs.

## `fuzz_direct_g8r_sequential_equiv`

Directly generates bounded sequential AIGs without an IR lowering step, including
packed state, feedback, optional resets/enables, both control polarities, shared
endpoints, constants, dead logic, and zero-width values. It round-trips both
native formats and compares their metadata and cycle-by-cycle outputs/state,
then compares the legalized design from the same arbitrary initial values.
Each sample uses four startups and twelve cycles, including cycle zero, without
a reset warm-up. It detects lost controls, polarity/priority mistakes, incorrect
holds or feedback, serialization corruption, and non-idempotent legalization.
The smoke test checks that state and both polarities actually occur. The block-IR
legalization target remains separate to provide an independent block oracle.

```shell
cargo fuzz run --sanitizer none fuzz_direct_g8r_sequential_equiv -- -max_total_time=300
```

## `fuzz_blif_roundtrip`

Uses the shared direct-G8R generator for both combinational and sequential
samples, including packed ports/state, feedback, shared endpoints, and optional
reset/enable controls of either polarity. It legalizes controls into effective-D
logic before BLIF emission, parses the emitted file, and repeats the round trip.
Port names/order/widths, register names/order/widths, and the clock must survive.
Across four arbitrary startup states and twelve cycles, outputs and every
register's next state must match the explicit-control source and legalized graph
from cycle zero, with no reset warm-up. The profile excludes zero-width ports
and registers because plain BLIF cannot represent them, but permits empty
interfaces. Unexpected emission or parsing failures fail the sample. This
catches lost aliases, latch/bit ordering errors, polarity mistakes, and lost
control behavior. A seeded smoke sweep checks combinational, registered,
controlled, both-polarity, and packed-interface coverage.

```shell
cargo fuzz run --sanitizer none fuzz_blif_roundtrip -- -max_total_time=300
```

## `fuzz_gate_fn_aiger_binary_roundtrip`

Uses the same generator without registers to round-trip binary AIGER with and
without symbol tables. It checks the binary parser and automatic format
selection, restores packed ports with the original interface schema, and checks
structural equivalence, deterministic emission, repeated round trips, and
simulation on sixteen shared mixed-distribution input vectors per symbol mode. Zero-width ports,
empty interfaces, constants, and dead logic are included. This catches varint
and literal encoding errors, parser disagreement, lost port grouping, and bit
ordering mistakes. A seeded smoke sweep verifies that zero-width/packed ports
and nontrivial logic are exercised. These are valid-graph round trips, not
malformed-byte parser fuzzing.

```shell
cargo fuzz run --sanitizer none fuzz_gate_fn_aiger_binary_roundtrip -- -max_total_time=300
```

### `fuzz_arrival_arithmetic_rows`

Focused Bitwuzla equivalence coverage for common add/product arithmetic rows,
with input-arrival skew, signedness, modular widths, shared/sliced/truncated
products, and repeated contributions. Checks actual deferred-product hits and
resolved boundaries. Solver timeouts remain inconclusive.

## `fuzz_sentinel_lowering`

Generates live canonical `ext_clz` and both `ext_prio_encode` directions over
widths 1 through 161, plus offset, widened-output (including 80-bit), and
truncated-output CLZ controls. Shared inputs/results, input reversal, post-count
modular addition, and fold/hash on/off settings vary independently. It lowers
without FRAIG/cuts/reassociation, lifts the gates to PIR, and explicitly proves
them equivalent to independently desugared source PIR with Bitwuzla. Every
sample also compares concrete all-zero input evaluation. Failures expose wrong
sentinels, lost sharing, offset/wrap/truncation errors, or incorrect gate export.

Counters distinguish completed canonical lowerings, directly reused count bits
(sentinel-zero output positions), fallback controls, proofs, and inconclusives.
These lowerings have no profitability gate; canonical branch counts are not
aug-opt acceptance counts. The replay audit requires proved canonical paths and
nonzero direct-count-bit coverage for CLZ and each priority direction, every
input width, builder settings and sharing controls. It rejects inconclusives.
Zero-width inputs remain covered by the existing directed regression tests;
this mutation grammar starts at one bit. The generic g8r generators are unchanged.

From `xlsynth-g8r`, use the normal XLS environment and run:

```shell
cargo run --release --manifest-path fuzz/Cargo.toml \
  --features with-bitwuzla-built --example sentinel_lowering_coverage -- \
  --write-corpus /tmp/sentinel-corpus
XLSYNTH_FUZZ_REPORT_SAMPLES=1 cargo +nightly fuzz run fuzz_sentinel_lowering \
  /tmp/sentinel-corpus --features with-bitwuzla-built --sanitizer none -- \
  -seed=109601 -runs=3000 -max_total_time=300 -timeout=90 -max_len=128 \
  -print_final_stats=1
cargo run --release --manifest-path fuzz/Cargo.toml \
  --features with-bitwuzla-built --example sentinel_lowering_coverage -- \
  /tmp/sentinel-corpus
```

Omit replay arguments to audit the built-in matrix. Run this standalone audit
in addition to the mutation smoke target; CI wiring is pending. Per-sample reporting records exact input bytes
and lowering/proof counts; retain logs, seeds and libFuzzer artifacts for replay.
`with-bitwuzla-system` is also supported.
