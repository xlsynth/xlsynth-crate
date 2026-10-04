# MFFC regression fixtures

Standalone XLS IR fixtures for aug-opt and gate-mapping regressions. Each file describes
the case it exercises; all parameter bits are unconstrained.

Include a `// DSLX expression:` comment above every fixture's top function.
Use the IR parameter names and preserve bit widths and signedness. Use a block
expression with `let` bindings for longer expressions.

Pair each `.ir` with a same-stem `.textproto` containing its content hash and
regression expectations. Tests discover these pairs automatically; the case name
comes from the filename and the top function from the IR. Historical measurements
belong in comments in the corresponding `.textproto`.

[profile.json](profile.json) pins the shared mapping options and Graph LE
tolerance used by [mffc_regressions_test.rs](../../mffc_regressions_test.rs).
The shift cases compare fixed IR with one PIR-only aug-opt round. The split-adder
cases compare recovery disabled and enabled in both PIR-only and sandwich modes.
Both use the pinned g8r profile without FRAIG or ABC. Split-adder cost bounds allow
area/delay tradeoffs; designated ordering variants must produce equivalent
structure and matching costs.

Run from the workspace root with the solver/toolchain environment configured:

```sh
cargo test -p xlsynth-aug-opt --features with-bitwuzla-system \
  --test mffc_regressions_test -- --nocapture
```
