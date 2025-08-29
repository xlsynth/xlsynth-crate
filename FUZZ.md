## Fuzz Targets Overview

This document lists the fuzz targets in the repository and summarizes what each one exercises at a high level. Each entry describes the essential property under test and the major failure modes it is intended to surface.

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

______________________________________________________________________

Notes:

- Degenerate inputs (e.g., zero-width, empty ops) are typically skipped with a brief comment explaining why they are not informative for the property under test.
- Unexpected failures in normal API operations (e.g., pretty-printed text failing to parse) should be flagged (panic) to surface systemic issues early.
