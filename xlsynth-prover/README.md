# `xlsynth-prover`

Formal proof helpers for XLS IR and DSLX.

This crate exposes both low-level proving primitives and higher-level request
APIs used by `xlsynth-driver`.

## Main APIs

### Pairwise IR equivalence

Use these when you already have parsed PIR functions or package text and want a
single equivalence proof:

- `xlsynth_prover::prover::prove_ir_fn_equiv()`
- `xlsynth_prover::prover::prove_ir_equiv()`
- `xlsynth_prover::prover::prove_ir_pkg_text_equiv()`

The `prove_ir_equiv()` entry point is the most configurable of the low-level
APIs. It takes `ProverFn` values plus options such as:

- `EquivParallelism`
- `AssertionSemantics`
- assertion label filtering
- aggregate flattening

### Structured IR-vs-IR proofs

Use `xlsynth_prover::ir_equiv` when the inputs are IR text plus proof options
and you want a structured request/report API instead of manually building
`ProverFn` values.

Main types:

- `IrModule`
- `IrEquivRequest`
- `run_ir_equiv()`

This API is a good fit for embedding the prover in higher-level tools that need
stable request/response structs.

### Equivalence-class membership

Use this when you have a candidate `C` and a corpus of known-equivalent members
`E`, and you want to prove whether `C` belongs to that class.

Main types:

- `EquivClassMember`
- `EquivClassRequest`
- `EquivClassShortlistOptions`
- `EquivClassReport`
- `EquivClassResult`
- `run_ir_equiv_class_membership()`

Behavior:

- ranks members with a structural-hash prepass plus edit distance
- runs bounded parallel pairwise proofs
- after the first proved match, stops scheduling new work and requests
  interruption of in-flight losing proofs
- reports an invariant violation if concurrently completed conclusive proofs
  disagree during stop propagation
- excludes interrupted losing proofs from that invariant check

This means the class-membership API usually finishes approximately as soon as
the first proved match completes, plus a small amount of worker unwind time.
That prompt stop behavior is currently meaningful with Bitwuzla, which supports
cooperative interruption inside the solver `check()` call. Other backends still
use the default no-op interrupt path, so they only stop taking on new work and
may let already-running solver calls finish naturally.

### QuickCheck-style proof APIs

Use these when proving that boolean-returning functions or DSLX quickchecks are
always true:

- `xlsynth_prover::prover::prove_ir_fn_quickcheck()`
- `xlsynth_prover::prover::prove_ir_quickcheck()`
- `xlsynth_prover::prover::prove_dslx_quickcheck()`

## Solver Selection

The low-level `prover::*` convenience functions use `SolverChoice::Auto`, which
selects the best available backend for the current build.

If you need explicit control over the backend, use the structured request APIs
and set:

- `IrEquivRequest::with_solver()`
- `EquivClassRequest::with_solver()`

Available backends depend on enabled cargo features and local tool
availability.

## Module Guide

- `ir_equiv`: structured IR proof request/report APIs
- `prover`: core proving traits, solver selection, and convenience entry points
- `solver`: SMT solver adapters
- `dslx_equiv`: DSLX-oriented equivalence helpers
- `dslx_utils` / `dslx_tactics`: DSLX support utilities
