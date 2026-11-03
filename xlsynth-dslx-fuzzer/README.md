# DSLX semantics fuzzer

This crate builds DSLX programs bottom up from AST fragments whose XLS IR
semantics are specified independently. Each fragment has two implementations:
format its DSLX syntax, and emit its hypothesized semantics with the checked PIR
builder. No DSLX conversion participates in reference construction.

`Body::push` checks operand availability, literal widths, constant types, and tuple indices.
`Sample::new` checks the translation environment: helpers have parametric bits
signatures, calls only reference earlier modules, and entry-point widths are
concrete, including dimensions computed by constant expressions. A call specializes a helper's reference body using the argument width
and a fresh value environment. The DSLX side retains real imported calls, with
either inferred or explicit width arguments. This exercises imports, lexical
binding, and parametric specialization as part of composition.

The generator consumes mutation bytes directly and produces at most three helper
modules, three output lanes, and six random fragments per ordinary body.
Constant-expression bodies contain at most three random fragments and do not
recursively generate constant definitions. Every concrete
lane calls a helper, and helpers can call earlier modules. Separate signed and
unsigned helpers can be instantiated at different widths in one environment.
In mission generation the top returns each lane's helper result followed by one
independently selected ordinary runtime intermediate from that lane. This lets
the oracle see mistakes in expressions that the helper result can discard.
Hand-built samples and the historical grammars return just the primary lane
results. Input exhaustion selects small valid forms rather than rejecting a
sample. Widths range from 1 through 129, with
additional probability at machine-word boundaries.

The fragments cover addition, subtraction, multiplication, division, remainder, bitwise operations,
negation, complement, shifts, all six signed/unsigned comparisons in conditionals,
tuple construction/projection, arbitrary-width literals, and cross-module calls.
Concatenation followed by either fixed or width slicing mixes two inputs;
widening, shifting, and narrowing exercises sign/zero extension and truncation.
Shift amounts are explicitly cast to unsigned bits; right shifts use the
left operand's signedness. All generated values and literals use `IrBits` or
`IrValue`; widths and graph indices use `usize`.
Division and remainder use an odd divisor to avoid division by zero.

Compound fragments keep aggregate values inside a typed scope and return bits
to their parent body. They cover dynamically indexed three-element arrays,
updates, slicing, concatenation, reversal, enumeration, zip, and repeated array
literals; exhaustive bits/tuple/enum matches; and parametric struct construction,
field access, and splat updates. Bounded range, array, and `const for` loops have
independent reference semantics constructed by unrolling zero through four
iterations. String lookup fragments carry three `IrBits` byte literals and
print escaped DSLX strings, including arbitrary high bytes.

Array maps apply builtins or imported functions to one through five elements.
Their references apply the scalar operation independently to each element.
Mission range expressions draw inclusive or exclusive endpoints within the
induction type's representable portion of `[-8, 8]`. The type can be signed or
unsigned and 1 through 129 bits wide, independently of the lane. Ordinary and
`const for` loops take zero through four trips. Indexed ranges contain one
through eight elements. The reference computes an element from the start and
index at the induction width, then applies the DSLX cast to the lane width.
Payload-bearing enums select between two runtime constructors
and match positional, named, or nested tuple payloads of independently chosen
widths. The reference computes the chosen payload result directly. The
experimental grammar additionally generates maps with captured lambdas and
`const match` expressions whose arm selection depends on the specialized lane
width and a generated two-bit constant.

Tuple binding fragments draw bounded trees with up to three levels and 40 nodes
from mutation bytes. They cover plain and annotated `let`, single-arm `match`,
nested bindings, wildcards, and leading/middle/trailing tuple-rest patterns,
including rest patterns that skip zero or all elements. Retained bindings feed
an ordered modular `3 * accumulator XOR value` fold. The independent PIR
reference folds the selected inputs directly; it does not evaluate DSLX patterns.

Builtin contracts include bit reversal, leading/trailing-zero counts,
reductions, one-hot operations, encode/decode, priority selection, gates,
checked/widening casts, explicit sign extension, bit-slice updates,
zero/all-ones macros, and type/array size queries.

`ConstExpr` binds literal inputs (or the enclosing helper's `N`) to a concrete
fragment body. These expressions appear in module and local `const` definitions,
computed type dimensions, and omitted parametric defaults. Constants and defaults
participate in returned values. Optional `const_assert!` statements compare DSLX
evaluation against values computed with the independent PIR interpreter. The
reference builder also instantiates defaults itself; neither reference values
nor reference dimensions come from the DSLX evaluator.
Ordinary loops and tuple patterns with bound names use concrete functions called
with constant arguments, because those bodies require an invocation boundary to
be evaluated as constexpr.

The AST is deliberately a small DSLX subset rather than the upstream compiler's
AST. To add a fragment, give it checked operands in `ast.rs`, implement its DSLX
printer and independent PIR semantics, add an observable primitive regression
case, then add it to the generic generator. Heterogeneous aggregate values at
body boundaries, first-class functions, processes, and effects are future
extensions. Partial-product multiplication awaits SMT translation support for
`smulp`/`umulp`. Builtin enumeration and
the old `range()` form differ across compiler releases; the generator uses
`std::enumerate` or its generated module-local counterpart and range syntax.

## Library use

```rust
let sample = xlsynth_dslx_fuzzer::generate(&[1, 2, 3, 4])?;
let sources = sample.sources()?;
let expected_package = sample.expected_ir()?;
let comparison = sample.prepare()?;
# Ok::<(), String>(())
```

`generate_with_options` accepts `GenerateOptions { max_width }` to bound entry
widths in `1..=129`. The ordinary generator and fuzz target retain the full
129-bit range and use the mission grammar. `Grammar::MissionLocal` draws the
same operations and widths but emits module-local implementations for named
maps and enumeration. `Grammar::MissionV1` replays bytes from the earlier
mission campaign. `Grammar::Experimental` replays the grammar with lambdas and
`const match`. The grammar choice is part of reproducing a saved fuzz input.
Historical formal regression sweeps use an eight-bit lane bound to keep
multiplication and division proofs within the routine solver budget. Wide
primitive tests and fuzzing exercise larger widths.

`prepare` writes modules to a `tempfile` directory, invokes the real DSLX
converter, and retains both IR programs and all sources after cleanup. Imported
source paths are remapped to relative paths in the parsed IR file table for
deterministic, portable reproducers. The reference side inlines calls; the
converted side preserves the converter's implementation. IR shape need not
match: the property is equality of the top's outputs for all inputs.

## Fuzzing and reproduction

Run from this crate directory, using the installed Bitwuzla shared library:

Use a current libxls and its matching standard library. The tuple grammar
exercises a historical converter crash involving sibling tuple-rest patterns;
the default released artifact can still contain that bug. See
[building against a native checkout](native/README.md#use-the-same-checkout-for-conversion)
to select a fixed top-of-tree converter for both tests and fuzzing.

```bash
cargo fuzz run --sanitizer none --features with-bitwuzla-system fuzz_dslx_semantics -- -max_total_time=60 -timeout=60 -max_len=1024
```

Set `DSLX_FUZZ_CONVERTER` to the standalone `ir_converter_main` to prove its
stdout against the PIR reference. After the proof attempt, the target converts
the same sources in-process so an instrumented libxls can guide mutations.
With the variable unset, the proof uses the in-process converter.
The target limits each Bitwuzla query to ten seconds and 1024 MiB.
`DSLX_PROOF_RESULTS` optionally appends one byte per completed check (`P` for
proved, `I` for inconclusive). `DSLX_INCONCLUSIVE_DIR` retains unresolved input
bytes by SHA-256 for longer replay. Generation, conversion, parsing, and prover
errors fail the input. Mismatches include solver counterexamples, the module
sources, and both IR programs.

For a four-worker standalone campaign, use three workers with local named
helpers and one with genuine standard library imports. Keeping their corpora
and artifacts separate records which grammar decodes each input. From this crate
directory, set `XLSYNTH_TOOLS` to the matching converter release directory and
`RUN_DIR` to an absolute output directory.

```bash
cargo fuzz build --sanitizer none --features with-bitwuzla-system fuzz_dslx_semantics
for worker in 0 1 2 3; do
  grammar=mission-local
  if [ "$worker" = 3 ]; then grammar=mission; fi
  out="$RUN_DIR/$worker"
  mkdir -p "$out/corpus" "$out/artifacts"
  printf '%s\n' "$grammar" > "$out/grammar"
  DSLX_FUZZ_GRAMMAR="$grammar" DSLX_FUZZ_CONVERTER="$XLSYNTH_TOOLS/ir_converter_main" \
    DSLX_INCONCLUSIVE_DIR="$out/inconclusive" DSLX_PROOF_RESULTS="$out/proofs.bin" \
    cargo fuzz run --sanitizer none --features with-bitwuzla-system fuzz_dslx_semantics \
    "$out/corpus" -- -fork=1 -ignore_crashes=1 -timeout=120 -max_len=1024 \
    -artifact_prefix="$out/artifacts/" > "$out/worker.log" 2>&1 &
done
wait
```

For a source-coverage snapshot, replay saved bytes using the same grammar and an
LLVM coverage build of the CLI. Set `DSLX_CLI_PROFILE` to an absolute
`cli-%m.profraw` path so the subprocess writes its own profile. Measure those
profiles against that CLI binary with `llvm-profdata` and `llvm-cov`.

Expand saved input bytes into reviewable files from the repository root:

```bash
cargo run -p xlsynth-dslx-fuzzer --example replay -- /path/to/crash-input /path/to/reproducer
```

Pass `--local`, `--mission-v1`, or `--experimental` before the input path for
those grammars. The output contains each DSLX module, `expected.ir`, and the
in-process `converted.ir`. Source files and expected IR are written before
conversion. To repeat a standalone proof, set the original
`DSLX_FUZZ_CONVERTER` and `DSLX_FUZZ_GRAMMAR` and pass the saved file to
`cargo fuzz run` in place of a corpus directory.

## Validation

```bash
cargo test -p xlsynth-dslx-fuzzer
cargo test --manifest-path xlsynth-dslx-fuzzer/fuzz/Cargo.toml --features with-bitwuzla-system --test formal
```

Primitive tests exhaust small widths and check mixed corner/random inputs at
wide boundaries using independent PIR and libxls interpreters. Composed sample
tests cover generated module environments and all explicit constexpr contexts,
including defaults with imported calls and signed/unsigned 129-bit constants.
Formal tests prove generated samples
and deliberately replace addition with subtraction in the reference to ensure
the oracle detects a wrong hypothesis. Source formatting uses golden files.

The oracle trusts each fragment's semantic hypothesis, the PIR builder, and the
prover. A counterexample can identify a bad hypothesis as well as a compiler
bug; keep primitive contracts small and reviewable.

## Native interpreter coverage

The [native feedback workflow](native/README.md) builds an instrumented C++
checker and links it into the Rust fuzz target. Parser, typechecker, and
constexpr interpreter counters then guide libFuzzer's mutations directly,
while each sample still runs through the independent equivalence oracle.
It needs no JIT. Native feedback is optional and configured at build time.

The [C++ coverage workflow](coverage/README.md) builds a separate native harness
for parsing, typechecking, and bytecode interpretation without linking the JIT.
It replays saved fuzz inputs and generic generated samples with mixed concrete
arguments, then emits source-annotated HTML, subsystem coverage totals, and a
list of uncalled native functions to guide fragment expansion. Use this replay
workflow to measure source line coverage of a corpus grown with native feedback.
