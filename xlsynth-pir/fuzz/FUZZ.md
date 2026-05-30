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
