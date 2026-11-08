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
