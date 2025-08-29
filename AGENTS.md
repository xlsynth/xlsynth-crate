## Engineering conventions for xlsynth-crate

- Prefer placing reusable functionality in library crates (e.g., `xlsynth-g8r`) rather than inside `xlsynth-driver`.

  - Library code should expose typed APIs that can be invoked from other tools and tests.
  - Parsing, transformation, and conversion logic should live alongside related modules (e.g., netlist parsing and conversions under `xlsynth-g8r/src/netlist/`).

- The `xlsynth-driver` crate should remain a thin shim:

  - Command-line definition and argument parsing
  - Invoking library functions
  - Human-friendly error reporting and exit codes

This separation makes functionality testable without a CLI, keeps dependencies focused, and avoids duplication across binaries.

# AGENTS.md

All pull requests must be clean with respect to `pre-commit`.

To verify locally, run:

```bash
pre-commit run --all-files
```

PRs that fail this check will not be accepted.

## Agent Guidance: xlsynth-g8r and Fuzz Targets

If you are modifying code in the `xlsynth-g8r` crate, you **must** ensure that all related fuzz targets (such as those in `xlsynth-g8r/fuzz/fuzz_targets`) still build. CI will fail if any fuzz target does not build. Always check the build status of these fuzz targets after making changes to `xlsynth-g8r`.

### Fuzz Target Early-Return Documentation

When a fuzz target chooses to early-return on an error (instead of panicking), include a brief comment explaining why the error is not considered a sample failure. Examples: degenerate inputs, known unsupported generator outputs, or infrastructure issues that are not properties of the fuzz sample. This makes intent clear and prevents future regressions from reclassifying benign cases as failures.

### Prefer Flagging Unexpected Failures (Overtesting is OK)

It's acceptable (and encouraged) for fuzz targets to flag failures that are not strictly the primary focus of that target, when those failures indicate violations of normal API expectations or invariants (e.g., parse failures on our own pretty-printed text). Our goal is an overall system with strong, composable guarantees; surfacing unexpected failures early helps preserve those guarantees.

## License Compliance: SPDX Headers

All source files must carry an Apache-2.0 SPDX license header (for example, `// SPDX-License-Identifier: Apache-2.0`). This is enforced by automated Rust tests (see `xlsynth-test-helpers/tests/spdx_test.rs`). If any file is missing the header, CI will fail and the pull request will not be accepted.

To run only this check locally, you can execute:

```bash
cargo test -p xlsynth-test-helpers check_all_rust_files_for_spdx
```

## Deterministic Output

All tools, and especially the `xlsynth-driver` subcommands, are expected to produce deterministic output. This is critical for test stability and reproducibility. While hash maps (`HashMap`) can be used internally for performance, they must not cause observable run-to-run nondeterminism in any output (e.g., emitted Verilog, SystemVerilog, or other netlists). If the order of items in a map affects output, a stably ordered map (such as `BTreeMap`) or explicit sorting should be used before emitting output.

## Style

Prefer using raw string syntax (`r#"..."#`) for multi-line strings to avoid needless escaping.

Avoid `use` statements inside local function scopes; place all imports at the
module level (or at the top of a `mod tests` section) for clarity.

When using C-style inline comments to document a named argument style in function calls, prefer:

```text
foo(/*kwarg=*/false)
```

Note there is no space before the value: `/*kwarg=*/false` (not `/*kwarg=*/ false`).

## Documentation

When adding a **new** `xlsynth-driver` subcommand you **must** add a corresponding
section to `xlsynth-driver/README.md` that follows the style of the existing
entries (name, short description, flag list, example usage, etc.). Pull requests
that introduce a command without updating the README are subject to rejection.

When adding or changing flags for an existing `xlsynth-driver` subcommand, update the corresponding section in `xlsynth-driver/README.md` to document the new/changed flags and their defaults.

## Test

For changes related to boolector, you should test with

```bash
cargo test --features with-boolector-built --workspace
```
