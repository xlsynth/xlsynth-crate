# Arbitrary PIR Transform Fuzz Target

This fuzz target generates a random single-function PIR package from
`xlsynth_pir::ir_fuzz::FuzzSample`, then repeatedly picks random PIR transforms,
applies candidate rewrites, validates the resulting package, and formally proves
equivalence for any candidate marked `always_equivalent`.

Target name: `fuzz_pir_transform_arbitrary`

Run:

```bash
cargo fuzz run fuzz_pir_transform_arbitrary
```

Essential property under test:

- Candidate-level `always_equivalent` claims made by PIR transforms are sound,
  and applying random rewrite sequences preserves structural validity.

Early returns justification:

- Samples rejected by `generate_ir_fn` are skipped because IR construction failed
  before transform application could be exercised. These are generator-shape
  failures, not transform soundness failures.
- Packages that fail PIR parse/validation before any transform is applied are
  skipped because they do not provide a valid starting point for the rewrite
  harness.

Main failure modes surfaced:

- A candidate marked `always_equivalent` produces a disproved or solver-error
  result.
- A transform candidate applies successfully but yields an invalid PIR package.
- Transform discovery/application crashes on structurally valid PIR input.
