# Augmented optimizer fuzz targets

Run these targets from `xlsynth-aug-opt` with the normal XLS and system Bitwuzla
environment:

```shell
cargo +nightly fuzz run fuzz_aug_opt_equiv \
  --features with-bitwuzla-system --sanitizer none -- \
  -max_total_time=60 -timeout=90
cargo +nightly fuzz run fuzz_constant_shift_choices \
  --features with-bitwuzla-system --sanitizer none -- \
  -max_total_time=3600 -timeout=90 -print_final_stats=1
```

## `fuzz_aug_opt_equiv`

Generates upstream-standard random PIR with the shared `xlsynth-pir-fuzz`
generator, runs one PIR-only augmented optimizer round with the default g8r cost
evaluator, and proves equivalence whenever a rewrite fires. The input generator
includes `gate` and arbitrary-width multiply and excludes product-pair operations
pending formal support. It has no directed cases for particular rewrites.
Failures expose unexpected rejection of valid PIR, changes in semantics, or
rewritten IR that the in-process prover cannot check.

## `fuzz_constant_shift_choices`

Generates typed, shared shift-amount DAGs to exercise constant-shift-choice
fusion directly, with forced decisions and the real g8r cost evaluator. It
proves local alternatives and complete rewritten functions equivalent with
Bitwuzla, checks exact rollback on rejection/errors, and exhausts small
effectful cases including trace-only uses. This targets incorrect projections,
priority/default handling, lost shared users, and leaked speculative changes.
See the [repository fuzz overview](../../FUZZ.md) for the full target contract.
