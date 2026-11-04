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
