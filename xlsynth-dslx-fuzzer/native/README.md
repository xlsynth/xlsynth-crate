# Native DSLX feedback

This workflow puts the checkout's C++ DSLX parser, typechecker, and constexpr
bytecode interpreter inside the Rust libFuzzer process. Native coverage counters
guide corpus retention and mutation alongside the Rust counters. Each generated
environment is checked with fresh C++ import state, converted by released
libxls, and proved equivalent to its independently constructed PIR reference.
Native rejection, conversion errors, and disproved equivalence fail the sample.
Solver resource limits retain the ordinary target's inconclusive-result policy.

The checker links no JIT, IR interpreter, or IR converter. Interpretation occurs
through the generator's constexpr contexts during typechecking. It does not
explicitly interpret runtime entry-point arguments; the formal oracle checks
the resulting function semantics for all inputs.

## Build and run

This setup currently requires Linux ELF, Bazel and the XLS checkout's Clang
toolchain, GNU `nm`, Python 3.8+, Cargo, cargo-fuzz, and Bitwuzla. Run the build
from the Rust workspace root:

```bash
python3 xlsynth-dslx-fuzzer/native/build.py \
  --xls-root /path/to/xlsynth \
  --output /tmp/dslx-native \
  --jobs 12
```

The script stages a Bazel package in an overlay and uses a separate output base,
leaving the XLS checkout's source tree unchanged. It builds with `-c opt` and
adds Clang inline 8-bit counters, PC tables, and comparison tracing to DSLX
compilation units. The small C ABI wrapper is excluded from instrumentation.
Build validation rejects dependencies on the JIT, IR interpreter, or IR converter
and requires exactly the two intended C exports.

The output `native.json` identifies the library and matching DSLX standard
library. Set its path during the Cargo build, including when starting cargo-fuzz:

```bash
cd xlsynth-dslx-fuzzer
DSLX_FUZZ_NATIVE_CONFIG=/tmp/dslx-native/native.json \
  cargo fuzz run --sanitizer none --features with-bitwuzla-system \
  fuzz_dslx_semantics /path/to/corpus -- \
  -max_total_time=120 -timeout=60 -max_len=1024
```

Use an existing corpus directory or create an empty one first. Startup prints
the native counter count, and libFuzzer reports both the native and Rust counter
modules. The configuration is optional: omit the environment variable at build
time to retain the ordinary target, which prints that native feedback is
disabled. Cargo tracks configuration and library changes. Keep the built
library and selected standard-library directory available while running the
binary; its library directory is recorded in the runtime search path. Stop
running fuzz processes before rebuilding the native library.

## Use the same checkout for conversion

Native feedback selects a parser/typechecker build; IR conversion still uses
the libxls linked by `xlsynth-sys`. To fuzz a top-of-tree checkout throughout,
build that checkout's libxls and select its matching standard library too:

```bash
cd /path/to/xlsynth
bazel build -c opt //xls/public:libxls.so
export XLS_DSO_PATH="$PWD/bazel-bin/xls/public/libxls.so"
export DSLX_STDLIB_PATH="$PWD/xls/dslx/stdlib"

cd /path/to/xlsynth-crate
python3 xlsynth-dslx-fuzzer/native/build.py \
  --xls-root /path/to/xlsynth --output /tmp/dslx-native

cd xlsynth-dslx-fuzzer
DSLX_FUZZ_NATIVE_CONFIG=/tmp/dslx-native/native.json \
  cargo fuzz run --sanitizer none --features with-bitwuzla-system \
  fuzz_dslx_semantics /path/to/corpus -- \
  -max_total_time=120 -timeout=60 -max_len=1024
```

Keep both exported variables set for the validation commands below. They form
a paired override; set both together. Alternatively, set
`XLSYNTH_ARTIFACT_CONFIG` to an absolute TOML path containing `dso_path` and
`dslx_stdlib_path`. That configuration takes precedence over the paired variables.
Record the checkout's commit when reporting a run. The checker stays isolated
and links no JIT; the standard libxls library provides conversion and the other
C API entry points used by Rust.

The tuple grammar includes sibling rest patterns that crash older released
converters. Use a checkout with the upstream tuple-destructuring fix; keep those
patterns enabled rather than silently skipping conversion failures.

## Verify the feedback and oracle

Run from the Rust workspace root:

```bash
DSLX_FUZZ_NATIVE_CONFIG=/tmp/dslx-native/native.json \
  cargo test --manifest-path xlsynth-dslx-fuzzer/fuzz/Cargo.toml \
  --features with-bitwuzla-system --tests

DSLX_FUZZ_NATIVE_CONFIG=/tmp/dslx-native/native.json \
  cargo run --manifest-path xlsynth-dslx-fuzzer/fuzz/Cargo.toml \
  --features with-bitwuzla-system --example native_feedback
```

The probe checks multiple generic samples discover new native edges and proves
every sample equivalent. It also checks native rejection of invalid DSLX and
deliberately substitutes subtraction for addition in a 65-bit reference to
verify that a wrong hypothesis is disproved. These directed checks run only in
the probe and regression test, outside the generic fuzz input path.

The native library exposes only a C checker and counter snapshot function. All
C++ symbols are local and bound within the library so that a newer checkout
can coexist with released libxls. The probe requires exact counter snapshots
before and after released-libxls/prover calls to verify that those calls do not
execute the instrumented checkout's code. Clang's module constructors register
the native sections with the executable's libFuzzer runtime; its ordinary
per-input counter reset and feature collection drive corpus selection.

## Measurements, 2026-10-25

Against checkout `9bc5938ab`, libFuzzer registered 94,994 native counters and
372,415 Rust counters. The probe observed 29 native locations before checking,
then about 16,300 across its 17 samples; every sample added native edges and passed
its formal proof. The deliberately wrong reference was disproved, and the
released-libxls/prover isolation checks passed.

A 121-second campaign completed 568 inputs without failure. Combined coverage
increased from 21,448 to 22,131, and retained feature count from 75,425 to
91,756. These combined figures include both counter modules and are not source
line percentages. The run used 32 generic random seed files, 1,024-byte maximum
inputs, and the ordinary 129-bit generator bound.

Appending 375 new corpus inputs to the 1,452-program baseline produced 1,827
programs and 29,232 concrete entry-point evaluations. Baseline program order and
argument-generation seeds were preserved. Full replay reached the same 17,547
of 42,351 DSLX lines: 41.43% overall and 52.55% within bytecode. The short native
campaign expanded counter features without adding source lines. New fragment
contracts and support for the deferred oracle operations remain necessary for
the missing language families.

For line coverage, replay the resulting corpus with the separate
[source coverage workflow](../coverage/README.md), selecting the same native
checkout and `--constexpr-only` when measuring the paths this checker executes.
Full replay additionally interprets concrete entry-point arguments. The
[line measurements](../coverage/RESULTS.md) record the baseline and this replay.
