# xlsynth-crate guidance for "agents"

## Engineering conventions for xlsynth-crate

- Prefer placing reusable functionality in library crates (e.g., `xlsynth-g8r`) rather than inside `xlsynth-driver`.

  - Library code should expose typed APIs that can be invoked from other tools and tests.
  - Parsing, transformation, and conversion logic should live alongside related modules (e.g., netlist parsing and conversions under `xlsynth-g8r/src/netlist/`).

- The `xlsynth-driver` crate should remain a thin shim:

  - Command-line definition and argument parsing
  - Invoking library functions
  - Human-friendly error reporting and exit codes

This separation makes functionality testable without a CLI, keeps dependencies focused, and avoids duplication across binaries.

## Keep the crate graph a DAG (including for tests)

Try to keep the workspace crate dependency graph **acyclic**, even in `dev-dependencies`.

- Prefer placing cross-crate integration / equivalence tests in a crate that already depends on both sides (often `xlsynth-prover`), rather than adding a `dev-dependency` from a lower-level crate back “up” the stack.
- This keeps build times and feature selection more predictable and avoids subtle CI-only cycles.

## `pre-commit`

All pull requests must be clean with respect to `pre-commit`.

To verify locally, run:

```bash
pre-commit run --all-files
```

PRs that fail this check will not be accepted.

## Tests

Completed changes must pass either `cargo test --workspace` or `cargo nextest run`.

Prefer `cargo nextest run` when it is available; for this workspace size it
usually gives better concurrency and faster feedback than plain `cargo test`.

For temporary files or directories in tests, prefer `tempfile` (or an existing
workspace helper built on it) over ad hoc paths under `std::env::temp_dir()`.
This avoids parallel-test collisions and makes cleanup automatic.

When working on serialization / deserialization or other roundtrip-heavy code
paths, prefer using the shared identity/signature corpus in
`xlsynth-g8r/src/test_utils.rs` (`interesting_ir_roundtrip_cases()`) instead of
adding only a one-off scalar case. That corpus exists specifically to cover
multi-bit and aggregate signatures such as `bits[2] -> bits[2]`,
`bits[4] -> bits[4]`, tuples, and arrays, which are easy to miss if a test only
uses `bits[1]`.

When adding characterization / QoR / equivalence sweeps that compare alternate
lowerings, name each preserved lowering by the structure or algorithm it emits
(for example, `RepeatedPrefix` or `RecursiveExplicitOrEq`) rather than by
chronology or API status (for example, `Old`, `Previous`, `Current`, or
`Public`). These reference implementations often outlive the change that
introduced them, so functional names keep later investigations understandable.

If network access is unavailable, it is acceptable to exclude the crates.io
version checks in `xlsynth-test-helpers/tests/version_test.rs` (which require
crates.io metadata). Document that exclusion in the change notes.

For non-trivial GitHub Actions logic, prefer a small reusable Python script
under `scripts/` over duplicating shell helpers inline in `.github/workflows`.
Those CI helper scripts should be compatible with the oldest runner/container in
the matrix rather than assuming modern tool versions; for example, do not assume
new `curl` flags or post-3.6 Python syntax are available everywhere.
In particular, the Rocky 8 CI lane currently installs Python 3.6.x, so CI
helper scripts under `scripts/` must remain compatible with Python 3.6 unless
the workflow is updated in the same change.

## Agent Guidance: xlsynth-g8r and Fuzz Targets

If you are modifying code in the `xlsynth-g8r` crate, you **must** ensure that all related fuzz targets (such as those in `xlsynth-g8r/fuzz/fuzz_targets`) still build. CI will fail if any fuzz target does not build. Always check the build status of these fuzz targets after making changes to `xlsynth-g8r`.

All fuzz targets must be documented in `FUZZ.md` with a short paragraph describing the essential property under test and the main failure modes being surfaced.

Do not bias a generic fuzz target by injecting a directed, hand-crafted case for
a specific transform or bug pattern into its normal input path. Generic fuzzing
targets should continue to find behavior through their generic input generation
mechanisms, so coverage remains unbiased and representative of the subsystem
being fuzzed. Put directed/stimulus-specific coverage in focused tests,
metamorphic/property tests, or a separately named target whose purpose is
explicitly documented, not inside an existing generic fuzz target.

### Fuzz Target Early-Return Documentation

When a fuzz target chooses to early-return on an error (instead of panicking), include a brief comment explaining why the error is not considered a sample failure. Examples: degenerate inputs, known unsupported generator outputs, or infrastructure issues that are not properties of the fuzz sample. This makes intent clear and prevents future regressions from reclassifying benign cases as failures.

Document these rationales as comments immediately above the early-returning block or condition in the fuzz target source, not in `FUZZ.md`. `FUZZ.md` should remain an overview map of targets and their aims.

### Prefer Flagging Unexpected Failures (Overtesting is OK)

It's acceptable (and encouraged) for fuzz targets to flag failures that are not strictly the primary focus of that target, when those failures indicate violations of normal API expectations or invariants (e.g., parse failures on our own pretty-printed text). Our goal is an overall system with strong, composable guarantees; surfacing unexpected failures early helps preserve those guarantees.

### Adding New Fuzz Targets

When adding a new fuzz target under `xlsynth-g8r/fuzz/fuzz_targets/`, you must also register it in `xlsynth-g8r/fuzz/Cargo.toml` by adding a `[[bin]]` entry with `name`, `path`, and `test/doc=false`. Otherwise `cargo fuzz run <target>` will fail with “no bin target named …”. Also add a short entry to `FUZZ.md` describing the target’s property and main failure modes.

## Library Routine Output Policy

- Library functions (especially under `xlsynth-g8r/src/**`) should not print to stdout/stderr.
- If diagnostics are useful, prefer logging at an appropriate level, or return structured
  error information to the caller.
- When invoking external tools from a library routine, capture stdout/stderr and propagate
  relevant information via return values; do not emit prints directly.

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

For non-trivial functions, prefer a one-line Rustdoc comment (`/// ...`) over no
comment at all. A short summary helps readers understand both the local code and
how it fits into the surrounding codebase.

Avoid `use` statements inside local function scopes; place all imports at the
module level (or at the top of a `mod tests` section) for clarity.

In general, when representing IR values with arbitrary bit widths, prefer `xlsynth::IrBits` / `xlsynth::IrValue` over fixed-width Rust integers (e.g. `u64`). Do not assume "64 bits is probably wide enough." In particular, avoid `IrValue::to_u64()` when matching literals in transforms; it errors for widths > 64. Prefer `IrValue::to_bits()` plus `IrBits::{to_bytes,is_zero,get_bit}` style helpers for width-agnostic inspection.

**Process guard**: in transform/matcher code, do not use `IrValue::to_u64()` / `IrBits::to_u64()` to recognize literals. Use width-agnostic helpers (`IrBits::equals_u64_value()` or `IrValue::bits_equals_u64_value()`), so patterns keep matching for widths > 64.

When using C-style inline comments to document a named argument style in function calls, prefer:

```text
foo(/*kwarg=*/false)
```

Note there is no space before the value: `/*kwarg=*/false` (not `/*kwarg=*/ false`).

- When adding a `TODO`, include the creation date in ISO format (for example
  `TODO(name): 2026-02-28 ...`) so readers can quickly judge how recent it is.

- When intentionally using a no-op match arm such as `_ => {}` (or otherwise ignoring a case), include a brief comment in that block explaining why ignoring it is correct/safe for that code path.

- When creating fake Liberty cell libraries or gate-level netlists in tests or benches, prefer simple, unsized cell
  names such as `INV`, `BUF`, `AND2`, `DFF`, `DFFN`, etc., unless drive strength or sizing is explicitly part of the
  behavior under test. This keeps examples clearer and avoids suggesting unnecessary realism.

## Documentation

When adding a **new** `xlsynth-driver` subcommand you **must** add a corresponding
section to `xlsynth-driver/README.md` that follows the style of the existing
entries (name, short description, flag list, example usage, etc.). Pull requests
that introduce a command without updating the README are subject to rejection.

When adding or changing flags for an existing `xlsynth-driver` subcommand, update the corresponding section in `xlsynth-driver/README.md` to document the new/changed flags and their defaults.

When adding or changing the `ir-query` pattern language (new matchers, syntax tweaks, etc.), update the `ir-query` documentation in `xlsynth-driver/README.md` to keep the DSL reference current.

For `xlsynth-driver` subcommands that operate on package-form IR and accept a
top-selection flag such as `--top`, prefer the following behavior:

- If the package already has a `top` function/member set, use that by default.
- If the package does not have a `top` set but there is exactly one suitable
  function/member for that command, use it by default.
- If the package does not have a `top` set and there are multiple suitable
  functions/members, require an explicit `--top` flag rather than silently
  picking one.
- If the user provides `--top`, it should override the package default.

## Augmented optimizer (`aug-opt`): how to exercise it

There are two primary ways to "play with" the augmented optimizer loop (libxls optimization + small PIR rewrites).

### `xlsynth-driver` (preferred CLI surface)

The driver exposes `--aug-opt=true` on several subcommands. The meaning is an "opt sandwich":
libxls opt -> PIR rewrites -> libxls opt.

- Only underlying libxls optimizations (no aug-opt):

```bash
cargo run -p xlsynth-driver -- ir2opt path/to/pkg.ir --top main
```

- libxls + aug-opt sandwich:

```bash
cargo run -p xlsynth-driver -- ir2opt path/to/pkg.ir --top main --aug-opt=true
```

Notes:

- `ir2opt --aug-opt=true` currently does not support `--toolchain` (external tool path). It uses the in-process APIs.
- For codegen flows, `--aug-opt=true` requires `--opt=true` (because it controls the optimization step before scheduling/codegen):

```bash
cargo run -p xlsynth-driver -- ir2combo path/to/pkg.ir --top main --delay_model=unit --opt=true --aug-opt=true
cargo run -p xlsynth-driver -- ir2pipeline path/to/pkg.ir --top main --delay_model=unit --opt=true --aug-opt=true
```

### Standalone binary: `xlsynth-pir-aug-opt`

The `xlsynth-pir` crate provides a dedicated debugging binary that runs the sandwich and allows multiple rounds:

```bash
cargo run -p xlsynth-pir --bin xlsynth-pir-aug-opt -- path/to/pkg.ir --top main --rounds 1
```

This binary also accepts `-` as input to read IR text from stdin.

### Aug-opt-only mode (debug binary only)

For isolating aug-opt rewrites (without any libxls optimization passes), use the debug binary's `--aug-opt-only` flag:

```bash
cargo run -p xlsynth-pir --bin xlsynth-pir-aug-opt -- path/to/pkg.ir --top main --rounds 1 --aug-opt-only
```

## Test

For changes related to boolector, you should test with

```bash
cargo test --features with-boolector-built --workspace
```

## Offline environments (web IDEs, sandboxed runners)

When building or running tools in an offline environment (e.g., Codex Web, remote sandboxes, `CARGO_NET_OFFLINE=1`):

- Ensure pre-fetched XLS artifacts are available via environment variables:
  - `XLS_DSO_PATH`: directory/path to the XLS dynamic library used for linking and rpath.
  - `DSLX_STDLIB_PATH`: path to the DSLX stdlib directory (ending in `.../xls/dslx/stdlib/`).
- Do not conflate the Rust crate version (`crate:vX.Y.Z`) with the XLS DSO/tools
  release tag (`dso:vA.B.C`). They are distinct version axes related by the
  compatibility map; CI and scripts that need the DSO/tools tag should resolve
  it through `scripts/get_required_libxls_dso_and_tools_release_tag.py` (or the
  underlying compatibility data), not derive it from the crate version.
- Our `xlsynth-sys/build.rs` will error early under `CARGO_NET_OFFLINE=1` if downloads would be required and these variables are not provided.
