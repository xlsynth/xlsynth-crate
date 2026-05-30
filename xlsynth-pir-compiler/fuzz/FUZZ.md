# PIR Function JIT vs Evaluator Fuzz Target

Target name: `fuzz_pir_function_jit_eval_equiv`

Run:

```bash
cargo fuzz run fuzz_pir_function_jit_eval_equiv
```

This target uses `xlsynth_pir::ir_random` to construct typed PIR functions
directly from the fuzzer byte stream, with scalar operations and widths limited
to the current native-value execution boundary. It evaluates the same generated
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
