# PIR Function JIT vs Evaluator Fuzz Target

Target name: `fuzz_pir_function_jit_eval_equiv`

Run:

```bash
cargo fuzz run fuzz_pir_function_jit_eval_equiv
```

This target uses `xlsynth_pir::ir_random` to construct typed scalar PIR
functions directly from the fuzzer byte stream, with widths limited to the
current native-value execution boundary. It evaluates the same generated
arguments through `xlsynth_pir::ir_eval` and through `xlsynth-pir-compiler`,
then requires identical returned values.

Essential property under test:

- Any generated PIR function within the configured JIT subset has the
  same semantics under generated native code and the PIR evaluator.

Main failure modes surfaced:

- Incorrect Cranelift lowering for supported arithmetic, bitwise, comparison,
  extension, or slice operations.
- Incorrect masking of padded native carrier bits.
- Divergence in native argument/result access versus PIR evaluation.

Target name: `fuzz_pir_function_jit_aggregate_eval_equiv`

Run:

```bash
cargo fuzz run fuzz_pir_function_jit_aggregate_eval_equiv
```

This target expands the same differential property to native aggregates and
observable function events,
including multidimensional arrays, nested tuples, aggregate construction and
indexing, aggregate equality/inequality, aggregate-valued gate or selection
results, and default-only `sel` nodes. It also exercises tuple-valued
extension results such as `ext_normalize_left` and XLS partial-product
multiply tuples, along with `after_all`, `cover`, `assert`, and `trace`.
Observable event results are compared as unordered multisets because independent
event nodes are not semantically ordered unless their token dependencies require it.

Main additional failure modes surfaced:

- Incorrect native array stride or `#[repr(C)]`-compatible tuple field layout.
- Incorrect scratch materialization or recursive copying of aggregate values.
- Incorrect out-of-bounds semantics for native array slice/update/index nodes.
- Divergent assertion, trace, or cover callback behavior.

Target name: `fuzz_pir_function_compiler_wide_eval_equiv`

Run:

```bash
cargo fuzz run fuzz_pir_function_compiler_wide_eval_equiv
```

This target compares the PIR evaluator with native compilation for functions
whose bitvector leaves may be as wide as `bits[1024]`, including arrays and
tuples. Width generation is biased toward scalar-sized values while
periodically sampling the full wide range, so it exercises mixed narrow/wide
graphs as well as genuinely large values. It also generates token-based
`after_all`, `cover`, `assert`, and `trace` nodes and compares their observable
runtime results. The target enables every operation provided by the random PIR
function generator.

Main additional failure modes surfaced:

- Incorrect little-endian `u64` limb storage or high-limb masking.
- Incorrect direct limb lowering for wide bitwise, comparison, slice, concat,
  selection, or add-family operations.
- Incorrect runtime-helper semantics for wide multiply, divide, shifts,
  dynamic bit updates, encoding operations, or extension operations.
- Incorrect recursive layout or copying of aggregates containing wide leaves.
- Divergent assertion, trace, or cover callback behavior involving wide data.
