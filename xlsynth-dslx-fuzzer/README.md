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
The top returns a tuple of lane results. Input exhaustion selects small valid
forms rather than rejecting a sample. Widths range from 1 through 129, with
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
extensions. `map` awaits PIR support for XLS map nodes, and partial-product multiplication
awaits SMT translation support for `smulp`/`umulp`. Builtin enumeration and
the old `range()` form differ across compiler releases; the generator uses
`std::enumerate` and range syntax supported by the equivalence toolchain.

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
129-bit range. Formal regression sweeps use an eight-bit bound so multiplication
and division proofs stay within the routine solver budget; wide primitive tests
and fuzzing exercise larger widths.

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
to select a fixed top-of-tree converter for both tests and fuzzing. Generation
retains the full grammar and conversion failures remain sample failures.

```bash
cargo fuzz run --sanitizer none --features with-bitwuzla-system fuzz_dslx_semantics -- -max_total_time=60 -timeout=60 -max_len=1024
```

The target proves equivalence with Bitwuzla, limiting each solver query to ten
seconds and 1024 MiB. Resource-limited inconclusive results are skipped;
generation, conversion, parsing, and prover errors fail the input. Mismatches
include solver counterexamples, the module sources, and both IR programs.
The repository smoke runner discovers the target automatically.

Expand saved input bytes into reviewable files from the repository root:

```bash
cargo run -p xlsynth-dslx-fuzzer --example replay -- /path/to/crash-input /path/to/reproducer
```

The output contains each DSLX module, `expected.ir`, and `converted.ir`. Source
files and expected IR are written before conversion, preserving converter-error
reproducers too. Replay the saved bytes directly through the fuzz target to
repeat the formal check.

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
