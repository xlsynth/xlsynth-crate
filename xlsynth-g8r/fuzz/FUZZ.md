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

## `fuzz_ir_outline_equiv`

Generates upstream-standard PIR directly, including `gate` and
arbitrary-width multiply forms but excluding product-pair operations, outlines a deterministically selected
connected region, and proves that the outlined outer function remains
equivalent to the input function. Failures expose invalid outlining output or
unsound region extraction/order handling.

## `fuzz_aug_opt_equiv`

Generates upstream-standard PIR directly, including `gate` and
arbitrary-width multiply forms but excluding product-pair operations, executes
PIR-only augmented optimizer rewrites, and proves equivalence when a rewrite
fires. Failures expose unsound PIR rewrite patterns or invalid rewritten
output.

## `fuzz_ir_roundtrip`

Generates upstream-standard PIR directly, including `gate`, arbitrary-width
multiply, product-pair, token, and event forms (`after_all`, `assert`,
`trace`, and `cover`), along with `assumed_in_bounds` array attributes,
serializes it, parses and re-emits it through libxls, then parses and re-emits
the libxls output through PIR. This intentionally tests the `PIR printer -> libxls parser/printer -> PIR parser/printer` interoperability boundary and
structural stability, including event and assumption metadata, of the PIR
roundtrip.

## `fuzz_ir_opt_equiv`

Generates upstream-standard PIR directly, including `gate` and arbitrary-width
multiply forms but excluding product-pair operations until prover translation
supports them, loads it into libxls, optimizes it through XLS, parses the
optimized result into PIR, and proves original-versus-optimized equivalence.
Enabled in-process SMT backends perform the proofs; this target does not require
the external XLS equivalence binary. Failures expose optimizer semantic
regressions or incompatibility between generated PIR text and XLS-produced
optimized IR. The target currently skips division and modulus with a `shll`
divisor because libxls misoptimizes some valid overflow cases; other divisor
forms remain enabled.

## `fuzz_ir_eval_interp_equiv`

Generates upstream-standard PIR directly, including `gate` and
arbitrary-width multiply forms, obtains coverage-oriented argument values, and
compares the PIR evaluator with the libxls interpreter after libxls loads the
emitted PIR text. Product-pair operations remain excluded until their PIR
evaluator behavior is covered consistently. Failures expose interpreter
semantic disagreement or PIR-to-libxls loading incompatibility.

## `fuzz_ir_same_sig_pair`

Builds two direct random PIR functions across the full function-level PIR
surface, including token values, event operations, and `assumed_in_bounds`
array attributes, using
constrained-signature generation and asserts that their complete function
signatures match. Failures expose bugs in the paired-function generator used
by graph-comparison targets.

## `fuzz_ir_rebase_equiv`

Generates two same-signature upstream-standard PIR functions directly,
including `gate` and arbitrary-width multiply forms but excluding product-pair
operations pending formal-translation support, rebases one graph onto the
other, validates the rebased result, and checks equivalence through the XLS
toolchain. Failures expose invalid or semantically unsound rebasing.
