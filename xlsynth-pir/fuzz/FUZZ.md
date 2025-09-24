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

Thes fuzz targets generates two same-typed random IR functions, computes edits using a matcher, applies those edits to the first function, and asserts the result is isomorphic to the second.

Target names: `fuzz_greedy_matching_ged`, `fuzz_naive_matching_ged`

Run:

```bash
cargo fuzz run fuzz_greedy_matching_ged
```

Essential property under test:

- The greedy matcher’s produced edits, when applied, should transform the old function into one isomorphic to the new function (modulo ids/names).

Early returns justification:

- Degenerate generator outputs (empty op lists or zero input width) are skipped; these are not interesting samples for edit computation.
- Parser errors are skipped; infrastructure/transient parsing failures are not properties of the fuzz sample.

Main failure modes surfaced:

- Incorrect edit planning or application that yields a non-isomorphic result.
- Crashes or panics during greedy selection or edit conversion.
