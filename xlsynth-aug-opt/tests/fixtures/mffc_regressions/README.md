# MFFC regression fixtures

Standalone XLS IR fixtures for regressions in shift lowering. Each file describes
the case it exercises; all parameter bits are unconstrained.

Include a `// DSLX expression:` comment above every fixture's top function.
Use the IR parameter names and preserve bit widths and signedness. Use a block
expression with `let` bindings for longer expressions.

[manifest.json](manifest.json) contains the mapping profile, reference
measurements, regression limits, and content hashes used by
[mffc_regressions_test.rs](../../mffc_regressions_test.rs).
The test compares the fixed IR with one PIR-only aug-opt round, using the same
g8r mapping profile for both.

Run from the workspace root with the solver/toolchain environment configured:

```sh
cargo test -p xlsynth-aug-opt --features with-bitwuzla-system \
  --test mffc_regressions_test -- --nocapture
```
