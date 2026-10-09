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

## `fuzz_priority_result_fusion`

Generates typed priority encode/index/decode graphs from coverage-guided bytes.
Bounded index-operation sequences vary constant additions/subtractions, reflection,
zero extensions, and literal-zero prefixes. Cases include both priority directions,
predicate masks, direct decode and one-shift outputs, expanded concat/XOR decoder
leaves, wide inputs/constants, wrapping, overshifts, and retained users. Valid near
misses cover shared count/amount intermediates, truncation, incomplete overflow
checks, reversed leaves, non-one shifts, variable arithmetic, and sign extension.
The generic `fuzz_aug_opt_equiv` generator remains unchanged.

Forced acceptance proves the semantic rewrite independently of g8r profitability;
cost ties/errors must preserve the exact input. The real optimizer runs with fusion
off/on in one PIR-only round or one/three sandwich rounds. Direct PIR-to-Bitwuzla
proofs compare each result with the original, and small signatures also exhaust
concrete interpreter inputs. Failures surface wrong bit wiring, sentinel/wrapping/
overshift errors, unsafe recognition, lost shared values, leaked rejected changes,
and optimizer-composition failures. Solver limits count as inconclusive, not proofs.

Named case/feature counters distinguish generated samples, forced and real fusions,
completed proofs, rejection checks, and inconclusives. Replay validation requires
proved fusions across the positive case families and important feature axes,
checked/proved rejection cases, and a proved real-cost fusion in each pipeline mode.
It rejects a corpus that merely visits labels without exercising the rewrite.

From `xlsynth-aug-opt`, with the normal XLS and Bitwuzla environment:

```shell
# Validate the deterministic matrix, including its actual rewrite/proof coverage.
cargo test --manifest-path fuzz/Cargo.toml --release \
  --features with-bitwuzla-system --lib priority_result_fusion

# Create a focused seed corpus, then run coverage-guided mutation.
cargo run --manifest-path fuzz/Cargo.toml --release \
  --features with-bitwuzla-system --example priority_result_fusion_coverage -- \
  --write-corpus /tmp/priority-fusion-corpus
cargo +nightly fuzz run fuzz_priority_result_fusion /tmp/priority-fusion-corpus \
  --features with-bitwuzla-system --sanitizer none -- \
  -max_total_time=3600 -timeout=90 -max_len=256 -print_final_stats=1

# Replay the saved corpus; exit unsuccessfully if required coverage is missing.
cargo run --manifest-path fuzz/Cargo.toml --release \
  --features with-bitwuzla-system --example priority_result_fusion_coverage -- \
  /tmp/priority-fusion-corpus
```

Omit the replay example's arguments to audit its built-in deterministic matrix.
Use `with-bitwuzla-built` instead when a system Bitwuzla installation is unavailable.

Scalar consumers include OR/AND/XOR reductions (full and sliced), equality, and
all four unsigned comparisons, with literals on either side. The scalar grammar
adds exact-width wrap and widen-before/after controls, 80-bit constants, widths
1 through 161, explicit zero inputs, and shared count/amount/hot/input/result
cases. Truncation followed by arithmetic, masked/signed indices, and arbitrary
one-hot-shaped inputs are rejection controls. A final sliced reduction is
supported. All widths receive a concrete zero-input check in addition to SMT.

The replay audit requires forced and real-cost proved fusions for every scalar
family in both priority directions, specifically in PIR-only mode, so later XLS
canonicalization cannot fill a missing scalar hit. It also rejects any
inconclusive proof. CI's Fuzz Smoke Test job runs this standalone audit against
the deterministic matrix in a separate step before the short mutation smoke run
and retains its coverage counts in the job log. A passing smoke run alone does
not demonstrate these hits.

For bounded campaigns, set `XLSYNTH_FUZZ_REPORT_SAMPLES=1` to log exact input
bytes, family/feature labels, accepted rewrites, and completed/inconclusive proof
counts for every iteration. Keep the starting corpus, libFuzzer `-seed`, binary
revision, logs, and artifact directory. For example, append `-seed=109701 -runs=3000 -max_total_time=300 -max_len=128 -print_final_stats=1` to the fuzz
command above. The corpus replay example accepts any number of files or
directories and reports a failing input index and its bytes.

Grouped scalar cases retain two predicates of one count, with both priority
directions and all three optimizer modes. The audit requires proved real-cost
acceptance for live groups and groups with a dead count user. Retaining the count
as an output remains a rejection control, including when other users are eligible.
