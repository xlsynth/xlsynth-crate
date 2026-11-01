# Coverage expansion, 2026-10-24

Measured against XLS checkout `9bc5938ab`, using the optimized, DSLX-instrumented
C++ harness described in [README.md](README.md). The comparisons below use the
same 428 saved mutation inputs plus 256 deterministic generated inputs, with 16
concrete argument vectors per program. Changing the generator changes the ASTs
selected by those bytes, so this compares generator coverage on a fixed byte
corpus rather than identical programs.

## Line coverage on 684 programs

| Linked DSLX code | Initial fragments | Constexpr/slices/casts | Compound fragments |
| --- | ---: | ---: | ---: |
| All linked DSLX | 24.32% | 27.48% | 41.43% |
| Bytecode | 20.68% | 25.97% | 52.55% |
| Frontend | 28.73% | 31.57% | 43.23% |
| Type system v2 | 38.94% | 43.70% | 61.15% |
| Exhaustiveness | 0.00% | 0.00% | 70.20% |

The compound round adds multiplication and nonzero division/remainder, 22
builtin forms, arrays and dynamic indexing/updates, enum and tuple patterns,
parametric structs and splat updates, range/array/const loops, and escaped
strings carrying arbitrary byte values. Aggregates remain internal to fragments
whose externally composed type is bits. Enumeration imports `std`, so measured
coverage includes the necessary standard-library parsing and typechecking.

Within bytecode, emitter line coverage is now 63.83%, interpreter coverage
58.07%, and builtin coverage 55.27%. Builtin coverage was zero before the
compound expansion. Total covered DSLX lines increased from 11,639 to 17,547
out of the same 42,351 linked lines.

Increasing the deterministic random portion from 256 to 1,024 produced 1,452
programs and 23,232 concrete `main` evaluations. It reached exactly the same
17,547 DSLX lines: all line percentages above were unchanged. Bytecode branch
coverage increased from 36.69% to 36.71%, and frontend branch coverage from
39.06% to 39.26%. This corpus-size check indicates that further line coverage
needs new fragment contracts or oracle support rather than more of these
generated samples.

## Native feedback follow-up, 2026-10-25

The [in-process native checker](../native/README.md) now registers 94,994 C++
counters with Rust libFuzzer. A 121-second campaign completed 568 inputs without
failure, retaining additional native/Rust counter features. Its 375 new inputs
were appended to the 1,452-program baseline, preserving baseline program order
and concrete argument seeds. Replaying all 1,827 programs with 16 argument
vectors each covered the same 17,547 DSLX lines. All line percentages in the
table remain unchanged. This short run validates native feedback but does not
show a source line coverage gain.

## Constexpr contexts

Module/local constants, computed dimensions, default parameters, and
`const_assert!` exercise the interpreter during typechecking. In the earlier
constexpr expansion, a run without explicit replay wrappers invoked
`ConstexprEvaluator::InterpretExpr` and `BytecodeInterpreter::Interpret` 23,841
times each, up from 649 `InterpretExpr` calls with the initial fragments.
`--constexpr-only` makes this measurement directly repeatable.

Ordinary loops and tuple matches with bound names need a function-call boundary
to become constexpr. Their constant-expression templates therefore emit
concrete functions called with constant arguments. Their reference values still
come from independent PIR construction and evaluation; loops are unrolled on
the reference side.

## Validation and limits

The expanded generator passes exhaustive small-width primitive tests,
mixed-input boundary tests through 129 bits, explicit constant-context tests for
every primitive, and generated composition tests. Formal regression samples use
an eight-bit entry-width bound to keep multiplication/division proofs within the
routine budget. The fuzz target retains 129-bit entries and skips only solver
resource-limit results, as before. Final formal regressions pass, and a
121-second fuzz smoke run completed 563 inputs without a sample failure.
The final workspace sweep passed 4,671 tests, with 12 skipped and the same 16
pre-existing failures: seven driver golden mismatches and nine published-version
checks. Crate Clippy checks and the SPDX check pass.

Further language coverage currently meets these oracle/toolchain limits:

- `map` produces XLS map nodes that PIR cannot yet parse. Direct builtin map
  conversion is itself limited to particular builtin functions.
- `smulp`/`umulp` parse and interpret, but the prover lacks SMT translation.
  These were detected by formal validation and are excluded from generation.
- The supported libxls release requires `std::enumerate`; a direct enumerate
  builtin exists in the newer native checkout. The callable `range()` form is
  rejected by the supported typechecker, so the generator uses range syntax.

Wider/more varied pattern trees, heterogeneous aggregate values at body
boundaries, imported nominal types, and first-class functions remain useful
extensions. Proc/channel/effect handlers are outside this function-only
harness. Diagnostic paths and linked compatibility code also remain in the
coverage denominator, so the percentage alone does not describe missing valid
function semantics.
