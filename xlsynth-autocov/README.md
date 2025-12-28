# xlsynth-autocov

`xlsynth-autocov` contains utilities for coverage-guided corpus growth and related analysis for XLS IR functions.

## `xlsynth-autocov-relevant`

`xlsynth-autocov-relevant` checks whether a boolean (`bits[1]`) IR value is **relevant** to a computation by comparing two variants of the function:

- **stuck-at-0**: the chosen node is replaced with the literal `0`
- **stuck-at-1**: the chosen node is replaced with the literal `1`

If the two variants are **equivalent** (under `AssertionSemantics::Same`), the node is reported as **irrelevant**. If they are **not** equivalent, the node is reported as **relevant** and the tool will print a counterexample when available.

This is useful for quickly identifying internal booleans that do not influence the function's observable behavior, which can help with:

- reducing the space of internal signals to track
- debugging and minimization (what signals actually matter)
- understanding which conditions drive behavior differences

### Example

Run the tool by pointing it at an IR package file, the function name, and the node text id:

```bash
cargo run -p xlsynth-autocov --bin relevant -- \
  --ir-file path/to/package.ir \
  --entry-fn f \
  --node-text-id 123
```

By default it uses `--solver auto`. You can also select a solver explicitly:

```bash
cargo run -p xlsynth-autocov --bin relevant -- \
  --ir-file path/to/package.ir \
  --entry-fn f \
  --node-text-id 123 \
  --solver toolchain \
  --tool-path /path/to/xls/tools
```

To use the native Bitwuzla backend, build with the corresponding feature and
select it via `--solver bitwuzla`:

```bash
cargo run -p xlsynth-autocov --features with-bitwuzla-built --bin relevant -- \
  --ir-file path/to/package.ir \
  --entry-fn f \
  --node-text-id 123 \
  --solver bitwuzla
```

Solver availability is feature-gated (mirroring `xlsynth-driver`); run
`xlsynth-autocov-relevant --help` to see the available `--solver` values for
your build.

Output is a single line like:

- `relevant_result node_text_id=123 relevant=false`
- `relevant_result node_text_id=123 relevant=true ...`
