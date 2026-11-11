# DSLX semantics

`fuzz_dslx_semantics` composes checked DSLX AST fragments into a module DAG,
independently instantiates their PIR semantics, converts the formatted DSLX, and
proves top-function equivalence for every input with Bitwuzla. It surfaces
semantic disagreements, specialization/import errors, and unexpected generation,
conversion, parsing, or proof failures. See the [crate guide](../README.md) for
the initial fragment set, bounds, commands, and reproducer workflow.

Setting `DSLX_FUZZ_NATIVE_CONFIG` during the Cargo build also runs an isolated
C++ checker before conversion. Its parser, typechecker, and constexpr bytecode
coverage counters feed the same libFuzzer mutation loop. Native rejection of a
generated program fails the sample. See the [native setup](../native/README.md)
for building the checker, verifying feedback and oracle isolation, and running
with it enabled.

Standalone converter campaigns can set `DSLX_FINDINGS_DIR` to keep fuzzing
after a solver counterexample, an unsuccessful converter exit, or a mismatch
between the emitted top function's signature and the DSLX reference. Each
category retains the exact input by SHA-256 and a first diagnostic. The optional
`DSLX_PROOF_RESULTS` ledger records `D` for disproof, `R` for converter
rejection, `X` for invalid output including a signature mismatch, `P` for
proof, and `I` for a solver limit. Checker errors still stop the worker. To
reproduce a finding with its full diagnostic, replay the saved file with the
same grammar and converter and unset `DSLX_FINDINGS_DIR`.

`fuzz_dslx_converter_contracts` exercises focused properties of the standalone
C++ converter. Mutations vary builtin map names and module paths, one-hot
priority expressions, named payload patterns, array casts, enum membership,
nested assertion guards, constexpr array packing and stdlib enumeration.
For each generated program it checks converter acceptance, parses the emitted
IR with the C++ parser, and compares both the return value and the assertion
failure condition against independent IR with Bitwuzla. For nested guards
it also bounds the emitted AND count after equivalence succeeds. Set
`DSLX_FUZZ_CONVERTER` to the standalone executable. This target uses the
same retention and replay settings as the semantics target, groups findings
by property, and records `X` for emitted IR rejected by the C++ parser and
`G` for excessive guard growth.

Set `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v1` on
`fuzz_dslx_converter_contracts` to use a separate byte grammar for lexical
shadowing in specialization arguments and type dimensions, explicit builtin
parameters, decode width boundaries, assertion result types, effectful struct
updates and array lengths, and discarded assertions in module constants.
The grammar also embeds the existing focused families. Keep its corpus in a
separate directory and set the same variable when replaying a retained input.
Leaving it unset selects the original eight-family encoding.

Set `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v2` to fuzz correctness across these
additional DSLX constructions:

- Independently declared types in two imported modules. The grammar varies
  type names, field layouts, and call order. Standard-library enumeration
  uses default DSLX. The shared projection and method-dispatch cases enable
  `#![feature(generics)]` in the module that defines the generic helper.
- Derived `ToBits` on nested aggregate fields, and instance methods supplied
  to `map` in different expression positions. The expected IR packs the
  source values or computes the mapped result independently.
- Ordinary and formatted assertion labels, and signed trace operands with
  default or explicit decimal formatting. The trace oracle compares the JIT
  messages with independently formatted expected text as well as proving the
  returned value.

Keep an audit-v2 corpus separate and select the same grammar for replay.
Prefix an audit-v1 input with one zero byte to reuse its program and semantic
checks in audit-v2. The v2 grammar checks correctness without a graph-size
limit.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v3` for the following checks.

- Unrolled loops must preserve the value returned by their body, including
  when the carry is bound to `_`. The checker also compares assertion failures
  when the iterable or initializer is a checked function call. It checks folds
  over enum arrays with both constant and runtime elements. These checks use
  independent reference IR because DSLX bytecode and IR conversion share the
  loop expansion.
- Casting an integer that represents a declared enum member must preserve its
  value, including when the source and enum widths differ and the expression is
  evaluated as a constant.
- An instance method derived by `ToBits` must be usable as a mapper. Controls
  invoke the method once before the map or implement it explicitly.

Use a separate audit-v3 corpus. Prefixing any audit-v2 input with one zero byte
retains its original source and oracle. Replay saved bytes with their original
grammar setting.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v4` to vary tuple binding patterns
independently of the subject type, including singleton parentheses with and
without a comma. Install the matching `typecheck_main` next to the standalone
`ir_converter_main` selected by `DSLX_FUZZ_CONVERTER`. For these samples the
worker runs the typechecker first. A clean DSLX type error satisfies the check.
An accepted program must convert and emit parseable IR with a Boolean top result.
A crash in either executable or a converter rejection after successful typechecking
is retained as a finding. For this property `P` means the acceptance check passed.
Prefix an audit-v3 input with one zero byte to replay it in a separate audit-v4 corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v5` to exercise tuple-bound integer
literals used directly as results or passed to typed helpers. The worker
typechecks and converts the original source, compares the exact IR signature,
and proves the value against an independently constructed constant. In a
separate module, DSLX bytecode executes the same program with an explicit cast
at the binding's use and checks the expected typed result. Provide the matching
`dslx_interpreter_main` and `typecheck_main` next to `ir_converter_main`.
A zero prefix embeds any audit-v4 input in the separate audit-v5 corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v6` to compare two integer literals
bound in a tuple or in separate lets. The generator varies literal spelling
(hexadecimal, binary, or decimal), magnitude, comparison operator, operand
order, and evaluation in a function body, module constant, or parametric
argument. For example, `0xff` can represent `-1` when compared at signed
width eight, while decimal `255` needs nine signed bits.

The worker typechecks the original source. Once it is accepted, the worker
checks the emitted IR against an independently computed Boolean and executes
the original expression and an explicitly cast control in separate DSLX
bytecode tests. This catches wrong bytecode even when IR is correct, and wrong
constant folding even when the top signature is correct. The same standalone
tools are required as for audit-v5. Prefix an audit-v5 input with one zero
byte to preserve it in an audit-v6 corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v7` to exercise three more wrong-value
properties with the standalone converter and independent reference IR.

- An expression that reads an inner binding inside its block and an outer
  binding outside it must respect both lexical scopes. Generate module
  constants, value arguments, type dimensions and runtime controls.
- Permuting three or more literals in an array while remapping the observed
  index, or reordering disjoint match arms while preserving the fallback,
  must preserve the widened value. Vary the literal spellings and include
  explicitly typed controls.
- A value selected by an ordinary `if` or `match` and used at a wider
  declared type must compute with that width. Compare inferred and explicitly
  typed producers in runtime and folded contexts. Signed match patterns and
  signed helper returns also check whether widening preserves the sign, even
  when the original IR has the expected width.

Use a separate audit-v7 corpus. Prefix an audit-v6 input with one zero byte to
preserve its source and oracle. Replay archived audit-v6 inputs directly with
their original grammar setting.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v8` to exercise three additional
families using ordinary free functions.

- Specialize a function passed to `map` with inline arithmetic, a let-bound
  expression, or its literal result. The reference adds the folded parameter
  to each array element independently.
- Cast between signed and unsigned scalar types of equal width, directly and
  through a value-parametric helper. Every cast must preserve the input bits,
  including at width zero where there is no sign bit to extend.
- Specialize a free function on a constexpr `if` with a runtime argument in
  the inactive arm. Compare inline addition, a bare conditional and a
  let-bound conditional against the live arm's value plus the addend. Vary
  branch polarity and include controls with two constant arms.

The one-byte inputs `05`, `06`, and `07` (hex) replay the baseline inline
`map` expression, signed zero-width cast, and wrapped constexpr `if`.
The converter runs in a child process, so a signal or nonzero exit can be
retained with its source while the campaign continues. Successful conversions
are checked against the same independent oracles. Prefix an audit-v7 unit
with one zero byte to preserve its source and oracle in an audit-v8 corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v9` to exercise the widths and
signedness of let-bound integer literals at calls. Each case runs the standalone
typechecker on its original source before conversion.

- `05` (hex) passes `let x = 3` to `identity<8>(x)`. The argument and result
  of the instantiated identity function must both be eight bits wide.
- `06` passes `let n = 4` as the value parameter of a function that returns its
  `u32` parameter. The emitted public return type must be `bits[32]`, even
  though three bits suffice to hold the value four.
- `07` passes `let x = 0x15` to a helper specialized with an `s5` parameter
  and a `u32` result. The expected value is `0xfffffff5`. The checker proves
  the emitted IR and executes the original call in a separate DSLX bytecode
  test. `07 04` exercises the same conversion with a concrete `s5` parameter.

Mutations vary widths, literal values and spellings, and how the operand gets
its type. Inline literals and explicitly typed bindings provide passing
controls. Prefix an audit-v8 unit with one zero byte to preserve its source
and oracle in an audit-v9 corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v10` for two properties involving
module constants in type dimensions.

- `06` (hex) generates `const N = u32:8` and a function with the parameter
  `N: uN[N]`. The standalone typechecker runs in a child process. A normal
  source type error is allowed for the self-shadowing form. If it typechecks,
  the emitted function must preserve the input at the module constant's width.
  Renaming the parameter, using a literal width and using a module alias each
  provide an unambiguous control that must typecheck and prove equivalent.
- `07` maps `|x| x + X` over `[X]`, where `X` has type `xN[SIGNED][WIDTH]`.
  The C++ IR parser must accept every emitted function name, including a lambda
  whose result has already been folded. A parser rejection is retained as
  invalid output (`X`). `07 01` generates the separate explicitly typed
  `|x: xN[SIGNED][WIDTH]|` control. Both forms use the same independent
  element-wise reference. Variants exercise signed and unsigned elements,
  runtime arrays, arithmetic and shifts.

Prefix an audit-v9 input with one zero byte to preserve it in an audit-v10
corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v11` for array cast contracts. The
one-byte inputs below are in hex.

- `04` packs unsigned elements into a signed word and widens it in a module
  constant. The baseline must produce `s16:-125` (`0xff83`).
- `05` packs signed elements in a module constant. Concatenating their bit
  patterns independently gives `u8:0x83` in the baseline. Both packing cases
  vary element widths, array length and bit patterns. Append `01` for a
  runtime control or `02` for an unsigned constexpr control.
- `06` casts zero bits into an empty array and keeps the array live through
  equality. The DSLX target runs `opt_main` in a child process and proves the
  optimized output against the expected scalar Boolean. The optimizer
  defaults to the `opt_main` alongside the converter. Set
  `DSLX_FUZZ_OPTIMIZER` to use a matching binary in another directory.
  `06 01` uses a nonempty array as a control.
- `07` casts a runtime `u8[2]` value to its own type. The emitted interface
  and every element must be preserved. `07 01` returns the array directly.

Prefix an audit-v10 input with one zero byte to preserve it in an audit-v11
corpus.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v12` for builtin boundaries.
The one-byte inputs below are in hex.

- `04` returns the module constant `ceillog2(u8:3)` from a function
  declared to return `u8`. The reference returns the eight-bit value 2.
  `04 01` observes the constant's width through concatenation, and
  `04 02` explicitly casts it to the expected width first.
- `05` passes a tuple-valued formal parameter directly to `update` on a
  two-dimensional array. The reference updates the selected element.
  `05 01` spells the same index as `(i.0, i.1)`.
- `06` calls `one_hot_sel` with a zero-bit selector and an empty case
  array. The reference ORs the selected elements, giving zero for no cases.
  `06 01` checks a one-element control.
- `07` calls `array_slice` with a zero-length witness. The emitted
  result must have the witness's empty array type. `07 01` checks a
  one-element result against an independent indexed read.

Prefix any audit-v11 input with one zero byte to preserve it in audit-v12.
For example, eleven zero bytes followed by `01 00 00 03 00 07` replay the
existing one-bit-input, eight-bit-output decode contract.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v13` for cross-module value parametrics
and array concatenation widths.
The inputs below are in hex.

- `04` returns `[2] ++ [3]` from a function declared to return `u8[2]`.
  The reference returns `[2, 3]` with eight bits per element. `04 01` types
  both literals as `u8`, `04 02` declares the inferred `u2[2]` return type,
  and `04 04` returns the flat literal `[2, 3]`.
  `04 08` packs the first and last elements into a fixed-width result. It
  expects `0x0203`, and `04 09` types both input literals for comparison.
  `04 10` widens the first element from an annotated signed array, where
  it must remain positive. `04 11` types both input literals explicitly.
- `05` calls equally specialized `apply` functions in `lane.ops` and
  `lane_ops`. They must both convert and keep their distinct results.
  `05 01` renames the flat module. `05 04` exercises a deeper import path.
  `05 02` uses a regular function named `apply__8` alongside `apply<N>` at
  width 8, and `05 03` renames the regular function.
- `06` forwards `Packet<A, B>` and `Packet<B, A>` through the same call site to
  a function that declares `B` before `A`. Both instances have the same IR
  layout, but one field is signed and the other unsigned. The reference
  widens the fields independently and mixes in distinct derived constants.
  `06 01` renames the callee binders, `06 02` infers them from the packet, and
  `06 04` swaps which instance is called first.
- `07` checks that an expression uses the nearest binding when specializing
  an imported function. `07 02` uses the expression to instantiate an imported
  struct and observes the resulting field width. `07 01` and `07 03` rename
  the local binding for comparison. Setting bit 2 of the second byte puts
  the expression in a separate `let` before passing it as a parametric.
  `07 08` observes a runtime word narrowed using the shadowed dimension,
  and `07 09` renames the local.

Prefix any audit-v12 input with one zero byte to preserve it in audit-v13.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v14` to check the width of a bare
integer field against a typed tuple match pattern. `03` matches `(1, x)`
against `(u8:1, false)`, and `03 01` explicitly types the scrutinee field as
`u8`. The independent reference returns `!x`. `03 08` changes the integer
pattern so the arm cannot match, with `03 09` as the typed control. The
generator also varies the field position, nesting, literal, and pattern width.
`03 10` evaluates the match as a module constant with a `false` Boolean
field, so the integer comparison must select the first arm. `03 11` types
the integer field explicitly.
`03 20` and `03 40` produce the subject tuple with an `if` or inner `match`,
respectively. Their independent runtime reference returns `p ^ x`. Set bit 0
of the second byte to type the integer fields in both arms. Set bit 7 to
check the same source in the bytecode interpreter with IR lowering disabled.
For example, `03 80`, `03 a0`, and `03 c0` check the direct, `if`, and inner
`match` bytecode results.
Prefix any audit-v13 input with one zero byte to preserve it in audit-v14.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v15` to check computed bases of
parametric struct splats. `03` inherits the width-bearing field through
`Pair { a: u8:1, ..make(x) }.b`. `03 01` binds the same call to a `let`
before splatting it. The second byte also selects explicit call and struct
parametrics and can observe both the replaced and inherited fields. The
third byte varies the inherited field width. The standalone converter gets a
two-second deadline for these small programs.
Rejections, crashes, and deadline misses record `R` with replay bytes, and
successful conversions proceed to the proof against independent IR.
Prefix any audit-v14 input with one zero byte to preserve it in audit-v15.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v16` to check values forwarded into a
nested struct's `s3` parametric. `06` forwards `uN[65]:4` and `06 01` forwards
`u64:4`. Both values are out of range and require a clean source type error.
`06 00 01` and `06 01 01` forward the representable value `3` and prove
that reading and sign-extending the nested parametric returns `3`. The
generator also varies source width, value, literal radix, and whether the
nested struct is imported. Unexpected acceptance records `X` and retains the
source and emitted IR. Crashes and unrelated errors record `R`.
Prefix any audit-v15 input with one zero byte to preserve it in audit-v16.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v17` to exercise aggregate-valued
function parametrics and struct width inference. `04` specializes a function
on two distinct tuple values with the same IR signature and calls the first
specialization again on a different input. It checks each field in order
against independent IR. The next byte varies tuple, array and record shapes,
signedness, constant names, forwarding, and imports. An aggregate type alias
lets the native typechecker reach function specialization. Aggregate-valued
struct parametrics in the pinned compiler still fail before IR emission.
`05` and `05 01` infer a struct's `u5` width parameter from `u5` and
`uN[5]` witnesses. They submit both spellings directly to the converter and
check the resulting field and inferred width against the same IR. `05 05`
uses a `u32` width parameter and `05 11` binds the narrow parameter explicitly. `05 40`
requires a clean rejection when inference would put `5` into `u2`.
Crashes and unexpected rejections retain the original source and replay bytes.
Prefix any audit-v16 input with one zero byte to preserve it in audit-v17.

Select `DSLX_FUZZ_CONTRACT_GRAMMAR=audit-v18` for array literal and ellipsis
coverage.

- `04` submits `[255, 255][i] == u16:255` to the converter. `04 02` checks
  the original bytecode result for both valid indices. `04 20` folds the
  comparison with a constant index in a module constant. Set the low bit of
  the option byte for explicitly typed controls with the same expected value.
- `05` converts `u8[0]:[]` and checks the exact array return type in
  unoptimized IR. `05 08` checks its bytecode result. `05 02` supplies a
  nonempty control. The options also vary element types and type annotations.
- `06` probes raw `()[2]:[x, ...]` syntax. `06 01` uses the declared return
  type and `06 02` uses an explicit unit type alias. All three have the same
  IR reference.
- `07` uses the length of an inline `make(y)` call to infer a sibling
  `[x, ...]`. `07 01` binds the call first and `07 02` annotates the result.
  The reference checks the inferred array length and selected element. This
  probes how the frontend propagates context to the unannotated literal.

Prefix any audit-v17 input with one zero byte to preserve it in audit-v18.

Replay a retained finding with the same grammar and the retained seed file.
The target recreates the import tree from the seed, including nested modules.

The target proves values and assertion conditions against independent IR and
compares the exact top signature. Trace-count cases run the standalone CLI's
emitted IR through the JIT at Boolean input true. The audit-v2 trace-format
cases compare the exact message at both Boolean inputs.
The effectful struct-base and array-size families use DSLX bytecode's observed
evaluation behavior as the reference. A discrepancy in those families
identifies a difference between execution paths. For failing module
constants, normal constexpr rejection satisfies the property, unexpected
acceptance records `X`, and a crash or unrelated diagnostic records `R`.

Standalone generic workers collect in-process C++ coverage after a successful
CLI conversion, including on a disproof or a public signature mismatch.
The primary CLI finding takes precedence over a secondary feedback error.

## Native BX function conversion

The `fuzz_bx_converter_contracts` target defaults to audit-v7. Set
`DSLX_FUZZ_CONTRACT_GRAMMAR` to `audit-v8`, `audit-v9`, `audit-v10` or
`audit-v11` to replay the corresponding focused worker's units. It runs the BX
frontend on single-module source,
lowers accepted functions with the BX Rust compiler, checks the signature of
`main`, and compares its IR with the generator's independent IR for all inputs
using Bitwuzla. On a discrepancy it compiles the original source with the
standalone DSLX converter and checks the two functions' public signatures,
return values and assertion behavior over all inputs. Imports, externs, and
contracts requiring optimizer, event, or acceptance oracles are outside the
scope of this target. Clocked block behavior requires a state-transition oracle.

Replay a saved audit-v7 unit from the repository root.

```sh
export DSLX_FUZZ_CONVERTER=/path/to/ir_converter_main
export DSLX_STDLIB_PATH=/path/to/matching/xls/dslx/stdlib
cargo fuzz run --fuzz-dir xlsynth-dslx-fuzzer/fuzz --features with-bitwuzla-system \
  --sanitizer none fuzz_bx_converter_contracts /path/to/audit-v7-unit -- -runs=1
```

Set `DSLX_PROOF_RESULTS` to a ledger file and `DSLX_FINDINGS_DIR` to a
findings directory for a continuous campaign. The BX ledger records `P` for a
proof against the independent reference. `A` means BX disagreed with that
reference but proved equivalent to DSLX. These inputs are counted without
saving a finding or stopping the fuzzer. `D` records a value or assertion
counterexample between BX and DSLX after BX failed the reference check. The
diagnostic includes both comparisons, which can have different witnesses.
`X` records a BX lowering, emitted IR or signature failure confirmed by
comparison with DSLX. `I` means a solver limit or unusable DSLX output left
the comparison unresolved. Set `DSLX_INCONCLUSIVE_DIR` to retain these
inputs for replay. `R` means BX rejected the source during parsing or
typechecking. `S` means the contract or its dependencies are outside this
target's scope.
