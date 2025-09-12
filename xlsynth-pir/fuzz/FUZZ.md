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
