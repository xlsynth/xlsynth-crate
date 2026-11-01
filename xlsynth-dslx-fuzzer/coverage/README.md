# C++ DSLX interpreter coverage

This workflow replays generated environments through the checkout's C++ parser,
typechecker, bytecode emitter, and bytecode interpreter. The native executable
depends directly on those libraries and does not use the IR converter, IR
interpreter, JIT, or the stock interpreter CLI's comparison framework.

Run from the Rust workspace root with Bazel, Cargo, Python 3.8+, and `c++filt`:

```bash
python3 xlsynth-dslx-fuzzer/coverage/run.py \
  --xls-root /path/to/xlsynth \
  --output /tmp/dslx-coverage \
  --samples 256 \
  --corpus /path/to/fuzz/corpus/fuzz_dslx_semantics
```

`--corpus` is optional. Samples from that directory are replayed in filename
order, followed by the requested number of generic generated samples. Use
`--samples 0` to measure only an existing corpus. Concrete argument vectors use
the shared mixed-input sampler, independently seeded from graph generation.
Each sample has 16 vectors, represented entirely with `IrBits`/`IrValue`.

The script stages its small Bazel package into a temporary package overlay;
it does not add source files to the XLS checkout. A separate Bazel output base
keeps coverage flags away from ordinary builds. The build uses `-c opt`, adds
`-fprofile-instr-generate -fcoverage-mapping` to DSLX compilation units, and links
the LLVM profiling runtime. Reporting uses `llvm-profdata` and `llvm-cov` from
the same hermetic LLVM toolchain that compiled the executable. See
[Clang's coverage documentation](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html)
for the profile format and reporting model.

Each exported `sample.x` has a nullary `coverage_entry` function returning a
tuple of all `main` evaluations. This forces concrete execution without an
extra argument parser or assertion wrapper. The wrapper contributes calls,
tuples, and literals to coverage; it does not add other language constructs.
The native harness creates fresh import/typechecking state per program, emits
that entry as bytecode, and interprets it. All imported helper modules retain
their original parametric functions.

Generated constants, computed dimensions, default parameters, and
`const_assert!` also reach bytecode interpretation during typechecking. Their
reference values are evaluated through PIR in Rust; the C++ coverage run
executes the original DSLX expressions. Slices, concatenations, and widening
casts exercise additional emitter/interpreter paths. Compound contracts also
exercise arrays, pattern matching, enums, loops, structs, strings, and explicit
builtins while retaining independent bits-valued reference semantics.

Pass `--constexpr-only` to export sources without `coverage_entry` and stop the
native harness after typechecking. Interpreter counters in this mode come from
constexpr evaluation, with no explicit replay calls. The same mutation corpus
and graph-generation seed allow comparison with the full interpreter run.
Summary reports are replaced on each run; preserve a prior report before
comparing modes. Raw profiles and exported sources have unique directories.

Outputs under `--output` include:

- `html/index.html`: annotated source with line and branch execution counts.
- `report.txt`: LLVM's per-file coverage summary for DSLX sources.
- `summary.json`: subsystem totals, per-file counts, and uncovered functions.
- `uncovered-functions.tsv`: demangled uncalled functions and source locations.
- `coverage.profdata` and `coverage.json`: machine-readable LLVM data.
- `samples-*`: DSLX environments, original mutation bytes, and a sample manifest.
- `profiles-*`: the raw profile from each run, kept separate to avoid merging
  stale coverage into a later measurement.

Use `--skip-build` to reuse an already built executable in the same output
directory. Keep the native source unchanged when selecting that option.

The summary excludes third-party sources and the measurement harness. Its
denominator is the DSLX code linked into this executable, not all source files
in the repository. Templates and macro expansions can affect function/branch
totals. Low error-path coverage is expected from a valid-program generator;
uncalled AST handlers and bytecode operations are more useful guides to missing
language features than a single overall percentage. Process execution is
outside this function-only harness.

This measures corpus replay coverage. It does not feed the native coverage
counters back into the Rust libFuzzer process. To grow a corpus using C++
coverage feedback, use the separate [native feedback workflow](../native/README.md),
then pass its corpus to this script. Edge counters and source line coverage are
different measurements; use these reports to compare line coverage before and
after a campaign. The native checkout can also be newer than the libxls release
used by the Rust equivalence target, so results describe the selected checkout.

See [the coverage expansion notes](RESULTS.md) for measured gains and the
remaining missing language families.
