# PIR vs XLS Verifier Parity Fuzz Target

This fuzz target feeds arbitrary IR text into both our PIR verifier and the upstream xlsynth verifier and checks that they agree:

- Both succeed, or
- Both fail with the same error category (coarse-grained mapping such as UnknownCallee, NodeTypeMismatch, etc.).

Target name: `fuzz_ir_verify`

Run:

```bash
cargo fuzz run fuzz_ir_verify
```

Essential property under test:

- For any IR text, our PIR validation should maintain parity with the upstream xlsynth verifier at the level of success/failure and high-level error category.

Early returns justification:

- Non-UTF8 inputs are ignored because the IR parser expects textual input.
- Inputs larger than 64 KiB are ignored to bound resource usage; this does not hide interesting cases because smaller, reduced cases will still be explored by the fuzzer.

Main failure modes surfaced:

- Divergence in acceptance (one verifier accepts while the other rejects).
- Divergence in coarse error category (e.g., PIR flags NodeTypeMismatch while xlsynth flags a different class).

______________________________________________________________________

# Graph Edit Distance (GED) Fuzz Target

These fuzz targets generate two same-signature PIR functions directly with
`xlsynth_pir::ir_random` across its full function-level PIR surface, including
aggregate, extension-operation, and `assumed_in_bounds` array-attribute
forms, compute edits using a matcher, apply those edits to the first function,
and assert the result is isomorphic to the second.

Target names: `fuzz_greedy_matching_ged`, `fuzz_naive_matching_ged`

Run:

```bash
cargo fuzz run fuzz_greedy_matching_ged
```

Essential property under test:

- The greedy matcher’s produced edits, when applied, should transform the old function into one isomorphic to the new function (modulo ids/names).

Main failure modes surfaced:

- Incorrect edit planning or application that yields a non-isomorphic result.
- Crashes or panics during greedy selection or edit conversion.
- Constrained-signature direct PIR construction fails unexpectedly.

______________________________________________________________________

# IR -> DSLX Roundtrip Equivalence Fuzz Target

This fuzz target generates an IR function directly with
`xlsynth_pir::ir_random`, translates it to DSLX with the `xlsynth-pir`
IR->DSLX library routine, converts that DSLX back into IR, and then proves
equivalence between the original and round-tripped IR packages via toolchain
equivalence.

Target name: `fuzz_ir_fn_to_dslx_roundtrip_equiv`

Run:

```bash
cargo fuzz run fuzz_ir_fn_to_dslx_roundtrip_equiv
```

Essential property under test:

- The IR->DSLX translator preserves function semantics under roundtrip back to IR.

Early returns justification:

- Samples that hit explicit `UnsupportedType` / `UnsupportedNode` translation
  errors are skipped because this target is currently scoped to the MVP
  translator contract.
- External XLS equivalence checks that exceed the fuzz-only five-second
  watchdog are skipped because solver cost is not a translation-soundness
  failure for the sample.

Main failure modes surfaced:

- Translator failures on supported directly generated PIR forms.
- DSLX conversion failures on emitted translator output.
- Semantic drift where toolchain equivalence disproves the input vs round-tripped IR.

______________________________________________________________________

# Block IR Roundtrip Fuzz Target

This target generates standard PIR directly, invokes XLS codegen to produce
combinational or one-stage pipeline block IR, and checks that the PIR
parser/printer roundtrips the generated block text unchanged.

Target name: `fuzz_block_roundtrip`

Essential property under test:

- Block IR emitted by XLS codegen from valid generated PIR is parsed and
  reproduced deterministically by PIR.

Main failure modes surfaced:

- XLS codegen fails on standard generated PIR accepted at its input boundary.
- PIR cannot parse or faithfully re-emit codegen-produced block IR.

______________________________________________________________________

# Direct Random Block Roundtrip Fuzz Target

This target generates block IR directly with `xlsynth_pir::ir_random`,
including input/output ports, upfront registers, optional load-enables, and a
mix of reset and non-reset registers when reset is available. It validates the
generated package, parses its printed text, and checks that printing is stable.

Target name: `fuzz_random_block_roundtrip`

Essential property under test:

- Direct block generation should emit structurally valid PIR blocks whose
  metadata, register reads/writes, output ports, and parser/printer roundtrip
  remain consistent.

Main failure modes surfaced:

- Register metadata and `register_read` / `register_write` nodes disagree.
- Register write argument, reset, or load-enable operands have invalid types.
- Output port metadata drifts from the block return value shape.
- PIR block parser/printer roundtrip changes generated structure or metadata.

______________________________________________________________________

# `ext_nary_add` Eval vs Desugared Eval Fuzz Target

This fuzz target generates a single-function PIR package whose return value is
an `ext_nary_add` over zero to five randomly generated operands. Each term is
chosen with a 3/10 probability of being a literal, 1/10 probability of using an
existing parameter, and 6/10 probability of introducing a new parameter, so
parameter lists are driven by actual term usage instead of being generated
independently. Each term also carries random `signed` and `negated` flags. The
target constructs the PIR package directly in code, evaluates the native
extension-op package and a desugared clone over a
deterministic, non-coverage-guided corpus of argument tuples and asserts that
the results are identical. Operand, literal, and parameter widths are bounded
to 0..=8 bits in this target.

Target name: `fuzz_ext_nary_add_eval_equiv`

Run:

```bash
cargo fuzz run fuzz_ext_nary_add_eval_equiv
```

Essential property under test:

- `ExtNaryAdd` evaluation matches the semantics of its desugared basis-op form
  across random widths, operand mixes, signedness, negation, and architecture
  annotations.

Main failure modes surfaced:

- Native `ext_nary_add` evaluator behavior diverges from desugared evaluation.
- Programmatically generated `ext_nary_add` packages fail structural validation.
- Desugaring rejects a generated `ext_nary_add` shape that should be supported.

______________________________________________________________________

# IR Package Roundtrip Fuzz Target

This fuzz target generates upstream-standard acyclic random packages directly
as PIR, including helper functions, `invoke`, `counted_for`, `gate`,
arbitrary-width multiply, product-pair, token, and event forms (`after_all`,
`assert`, `trace`, and `cover`), parses and re-emits them through libxls, then
reparses with the Rust parser and checks IR-level structural equivalence across
the PIR roundtrip.

Target name: `fuzz_ir_roundtrip`

Essential property under test:

- PIR printer output accepted and re-emitted by libxls remains parseable and
  structurally stable through the Rust PIR parser/printer.

Main failure modes surfaced:

- libxls rejects generated PIR text.
- PIR rejects libxls-emitted IR.
- Function/package pretty-printer roundtrip changes structure or metadata.

______________________________________________________________________

# IR Eval vs Interpreter Fuzz Target

This fuzz target differentially compares our Rust IR evaluator with the libxls
interpreter on the same directly generated acyclic PIR package, using autocov
to grow a bounded corpus of interesting typed inputs instead of checking only
one arbitrary tuple.

Target name: `fuzz_ir_eval_interp_equiv`

Essential property under test:

- PIR evaluation agrees with libxls interpretation across autocov-selected
  inputs for generated upstream-standard packages, including helper functions,
  `invoke`, `counted_for`, `gate`, and arbitrary-width multiply forms.

Early returns justification:

- Nullary functions are skipped because autocov input exploration has no input
  space to search and direct evaluator tests cover those functions more
  directly.

Main failure modes surfaced:

- PIR evaluator semantics diverge from libxls interpretation.
- Generated PIR text is not parseable or executable through libxls.
- Autocov fails unexpectedly on a valid generated package.

______________________________________________________________________

# Same-Signature Pair Fuzz Target

This fuzz target builds two direct random PIR functions across the full
function-level random-generator surface, including token values, event
operations, and `assumed_in_bounds` array attributes, using constrained
signature generation and asserts that their complete function signatures
match.

Target name: `fuzz_ir_same_sig_pair`

Essential property under test:

- `generate_same_signature_pair` produces valid paired PIR samples with
  identical function signatures.

Main failure modes surfaced:

- The paired random generator emits mismatched function signatures.
- Directly generated functions violate PIR validation expectations.

______________________________________________________________________

# IR Rebase Equivalence Fuzz Target

This fuzz target generates two same-signature upstream-standard PIR functions,
rebases one graph onto the other with `rebase_onto`, validates the rebased
result, and checks equivalence through the XLS toolchain.

Target name: `fuzz_ir_rebase_equiv`

Essential property under test:

- `rebase_onto` preserves semantics and emits valid rebased PIR across random
  same-signature graphs.

Early returns justification:

- Tool interruption or infrastructure failure during toolchain equivalence is
  skipped because it is not a semantic property of the generated sample.

Main failure modes surfaced:

- Rebasing emits invalid PIR.
- Rebasing changes function semantics.
- Parser/printer/toolchain integration rejects the rebased form.

______________________________________________________________________

# IR Outline Equivalence Fuzz Target

This fuzz target generates an upstream-standard random function directly as
PIR, selects a deterministic connected subgraph to outline into a new inner
function, rewrites the outer function to invoke the inner, and proves semantic
equivalence between the original function and the outlined outer using enabled
SMT backends.

Target name: `fuzz_ir_outline_equiv`

Essential property under test:

- Outlining preserves semantics for random connected PIR regions and supported
  parameter/return orderings.

Early returns justification:

- Degenerate samples with too few nodes, no boundary output, or non-convex
  regions are skipped because they cannot form an informative valid outline.
- Configured solver resource-limit inconclusive results are skipped because
  they are fuzz infrastructure limits, not outlining failures.

Main failure modes surfaced:

- Outlining emits invalid package/function structure.
- Outlined outer function is not equivalent to the original.
- Non-default parameter/return ordering mishandles outlined interfaces.

______________________________________________________________________

# Augmented Optimizer Equivalence Fuzz Target

This fuzz target generates an upstream-standard random PIR function, runs the
PIR-only augmented optimizer rewrite loop, and proves equivalence when at least
one rewrite fires.

Target name: `fuzz_aug_opt_equiv`

Essential property under test:

- PIR aug-opt rewrites preserve semantics when they apply.

Early returns justification:

- Samples with no aug-opt rewrite are skipped because there is no transformed
  output to compare.
- Configured solver resource-limit inconclusive results are skipped because
  they are expected fuzzing noise, not rewrite failures.

Main failure modes surfaced:

- Aug-opt rejects valid generated PIR unexpectedly.
- A PIR rewrite changes semantics.
- Rewritten PIR cannot be checked by the configured in-process prover.

______________________________________________________________________

# XLS IR Optimizer Equivalence Fuzz Target

This fuzz target builds an XLS IR package from an upstream-standard random PIR
sample, runs XLS IR optimization through libxls, parses the optimized result
back into PIR, and proves original-versus-optimized equivalence with enabled
in-process SMT backends.

Target name: `fuzz_ir_opt_equiv`

Essential property under test:

- XLS optimization preserves semantics for generated upstream-standard PIR.

Early returns justification:

- `optimize_ir` errors and configured solver resource-limit inconclusive
  results are skipped because this target treats them as unsupported or
  infrastructure-limited samples rather than optimizer disproofs.

Main failure modes surfaced:

- XLS optimizer changes semantics.
- Generated PIR or XLS-optimized IR is incompatible with PIR parsing.
- Enabled prover backends disagree on optimizer preservation.
