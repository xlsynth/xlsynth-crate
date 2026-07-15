# xlsynth-g8r Fuzz Targets

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
