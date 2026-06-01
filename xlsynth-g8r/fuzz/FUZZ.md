# xlsynth-g8r Fuzz Targets

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
