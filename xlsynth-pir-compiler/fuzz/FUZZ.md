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

This target expands the same differential property to native aggregates,
including multidimensional arrays, nested tuples, aggregate construction and
indexing, aggregate equality/inequality, aggregate-valued gate or selection
results, and default-only `sel` nodes. It also exercises tuple-valued
extension results such as `ext_normalize_left` and XLS partial-product
multiply tuples.

Main additional failure modes surfaced:

- Incorrect native array stride or `#[repr(C)]`-compatible tuple field layout.
- Incorrect scratch materialization or recursive copying of aggregate values.
- Incorrect out-of-bounds semantics for native array slice/update/index nodes.
