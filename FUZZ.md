## Fuzz Targets Overview

This document lists the fuzz targets in the repository and summarizes what each one exercises at a high level. Each entry describes the essential property under test and the major failure modes it is intended to surface. Per-target early-return rationales are documented inline in the target source above the relevant condition, not here.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_roundtrip.rs

Generates a random XLS IR function via the C++ bindings, serializes the package/function to text, reparses with the Rust parser, and checks IR-level structural equivalence of the original vs reparsed functions. Panics if parsing/validation of our own pretty-printed IR fails, or if the top function is missing.

Primarily tests:

- C++-emitted IR text compatibility with the Rust parser
- Function/package pretty-printer roundtrip soundness
- Structural equivalence stability across roundtrip

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_opt_equiv.rs

Builds an XLS IR package from a random sample, runs IR optimization, and cross-checks equivalence between original and optimized IR using an external tool and available SMT backends. Flags disagreements between engines and unexpected failures.

Primarily tests:

- Optimization preserves semantics
- Cross-engine equivalence consistency (external tool vs SMT backends)

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gatify.rs

Parses C++-emitted IR into the Rust IR, then converts (gatifies) to `GateFn` with folding on and off, checking equivalence when requested. Skips uninteresting degenerate inputs.

Primarily tests:

- IR→gates conversion correctness with/without folding
- Equivalence of different conversion modes

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_fn_roundtrip.rs

Builds a random `GateFn`, serializes to text, parses it back, and checks structural equivalence of the original vs parsed `GateFn`.

Primarily tests:

- GateFn textual serdes roundtrip
- Structural equivalence stability in gate graphs

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_fn_aiger_roundtrip.rs

Builds a random `GateFn`, emits AIGER, reloads AIGER into a new `GateFn`, and checks structural equivalence.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_eval_interp_equiv.rs

Differentially compares our Rust IR function interpreter (`eval_fn`) with the xlsynth C++ interpreter on the same randomly generated function and arguments.

- Generates a random XLS IR function (via C++ builder), pretty-prints, and reparses to the Rust internal IR.
- Samples argument values that match the function parameter types using an `Arbitrary`-driven seed.
- Evaluates with both engines and asserts the results are equal.

See inline comments in the target source for early-return rationales.

Primarily tests:

- AIGER emitter/loader roundtrip
- Structural equivalence stability across AIGER boundary

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_transform_equiv.rs

Applies a randomly selected equivalence-preserving transform to a random `GateFn` and proves equivalence via IR-based checker; panics if a claimed always-equivalent transform breaks equivalence.

Primarily tests:

- Local graph transforms preserve semantics
- IR-based equivalence checker catches semantic breakages

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_transform_arbitrary.rs

Performs random sequences of transformations over a `GateFn` and cross-checks equivalence between SAT (Varisat) and Z3 on original/current/next pairs; panics on disagreements or when an always-equivalent transform yields inequivalence.

Primarily tests:

- Transform engine correctness under random application
- Cross-solver consistency for equivalence

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_bulk_replace.rs

Exercises bulk replacement algorithms on internal IR/gate structures (see source for specifics) under random inputs, checking invariants and/or equivalence as implemented in the target.

Primarily tests:

- Bulk replace operations and invariants
- Stability of large-scale rewrites

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_same_sig_pair.rs

Builds two XLS IR functions from a `FuzzSampleSameTypedPair` and asserts that both validate and share an identical `FnType`. Panics if either function fails to validate or if their types differ.

Primarily tests:

- `FuzzSample::gen_with_same_signature` produces paired samples with matching function signatures
- IR validation succeeds for both generated functions

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_rebase_equiv.rs

Generates two random XLS IR functions with the same input signature (orig and desired). Rebuilds desired on top of orig using `rebase_onto`, then proves semantic equivalence between `desired` and the rebased result using the external toolchain equivalence checker. Skips degenerate samples and generator-unsupported constructs.

Primarily tests:

- `rebase_onto` preserves semantics and IDs stability constraints across random graphs
- Integration of `rebase_onto` with parser/pretty printer and toolchain equivalence

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ir_outline_equiv.rs

Generates a random XLS IR function via the C++ builder, reparses into the Rust IR, then selects a random connected subgraph (via BFS from a seed) to outline into a new inner function. Rewrites the outer function to invoke the inner and proves semantic equivalence between the original function and the outlined outer using available SMT backends. The target also explores parameter/return ordering:

- Param ordering mode: Default or deterministically shuffled non-default.
- Return ordering mode: Default or deterministically shuffled non-default.

Non-default orderings are constructed by permuting the default `OutlineOrdering` while preserving validity (same coverage, no duplicates). The PRNG is seeded from a stable hash of the package text, ensuring reproducible behavior for a given sample.

See inline comments in the target source for early-return rationales.

Primarily tests:

- Outlining transformation preserves semantics for random connected regions
- Stability of parser/pretty-printer across transformation boundaries

______________________________________________________________________

Notes:

- Degenerate inputs (e.g., zero-width, empty ops) are typically skipped with a brief comment explaining why they are not informative for the property under test.
- Unexpected failures in normal API operations (e.g., pretty-printed text failing to parse) should be flagged (panic) to surface systemic issues early.
