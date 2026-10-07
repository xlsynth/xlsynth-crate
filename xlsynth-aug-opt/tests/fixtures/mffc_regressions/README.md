# MFFC regression fixtures

TODO: 2026-10-05 Rename this corpus when convenient; it also covers generic IR
regressions that are not strict MFFCs.

Standalone XLS IR fixtures for aug-opt and gate-mapping regressions. Each file describes
the case it exercises; all parameter bits are unconstrained.

Include a `// DSLX expression:` comment above every fixture's top function.
Use the IR parameter names and preserve bit widths and signedness. Use a block
expression with `let` bindings for longer expressions.

Pair each `.ir` with a same-stem `.textproto` containing its content hash and
regression expectations. Tests discover these pairs automatically; the case name
comes from the filename and the top function from the IR. Historical measurements
belong in comments in the corresponding `.textproto`.

[profile.textproto](profile.textproto) pins the shared mapping options and Graph LE
tolerance used by [mffc_regressions_test.rs](../../mffc_regressions_test.rs).
The shift cases compare fixed IR with one PIR-only aug-opt round. The split-adder
cases compare recovery disabled and enabled in both PIR-only and sandwich modes.
Both use the pinned g8r profile without FRAIG or ABC. Split-adder cost bounds allow
area/delay tradeoffs; designated ordering variants must produce equivalent
structure and matching costs.

Priority-result cases compare fusion disabled and enabled with one PIR-only
round and one/three complete sandwich rounds. They check expected fusion counts
(including rejection controls), prove both mapped outputs equivalent, and enforce
the same pinned profile's area, graph-LE, and depth bounds. Origin information
is a comment in the IR; there is no separate provenance format or validation.

Run from the workspace root with the solver/toolchain environment configured:

```sh
cargo test -p xlsynth-aug-opt --features with-bitwuzla-system \
  --test mffc_regressions_test -- --nocapture
```
