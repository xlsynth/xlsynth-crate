# Arbitrary PIR Transform Fuzz Target

This fuzz target generates a valid random single-function PIR package directly
with `xlsynth_pir::ir_random`, then repeatedly picks random PIR transforms,
applies candidate rewrites, validates the resulting package, and formally
proves equivalence for any candidate marked `always_equivalent`.
Product-pair multiply operations are excluded until prover translation
supports their tuple-valued result form. Solver work is bounded per query and
per sample; configured resource-limit exhaustion is treated as inconclusive so
one hard proof does not monopolize the campaign.

Target name: `fuzz_pir_transform_arbitrary`

Run:

```bash
cargo fuzz run fuzz_pir_transform_arbitrary
```

Essential property under test:

- Candidate-level `always_equivalent` claims made by PIR transforms are sound,
  and applying random rewrite sequences preserves structural validity.

Main failure modes surfaced:

- A candidate marked `always_equivalent` produces a disproved or solver-error
  result.
- A transform candidate applies successfully but yields an invalid PIR package.
- Transform discovery/application crashes on structurally valid PIR input.
- Direct random PIR construction or its initial validation unexpectedly fails.
