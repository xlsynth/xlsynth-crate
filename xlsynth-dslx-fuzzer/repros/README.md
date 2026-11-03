# Reduced DSLX converter failures

These programs reproduced on stock XLS v0.61.0. Run from this directory with
the matching tools on PATH and DSLX_STDLIB_PATH set to the release's stdlib.

- `signed_loop_wide.x` adds the sole induction value −1 to its argument. At
  input zero the DSLX evaluator returns −1, but the emitted 65-bit IR has its
  top bit clear and returns 2⁶⁴−1. The same signed start in
  `signed_loop_narrow.x` makes the converter abort at width two.
- `signed_range_length.x` indexes a two-element signed range. Both indices
  pass in the DSLX evaluator, but the converter rejects the range as empty.
- `signed_one_bit_range.x` evaluates the signed inclusive singleton
  `0..=0` at compile time. Both the typechecker and converter abort trying
  to represent +1 as a signed one-bit value. The same singleton with two-bit
  endpoints evaluates to zero.
- `signed_loop_observer.input` is a saved `mission-local-v2` fuzz mutation. The
  second lane loops over `sN[65]:-5..=sN[65]:-4` with a 99-bit accumulator.
  Bitwuzla found inputs for which the standalone converter sets the top bit of
  the observed loop result incorrectly. Projecting both IR programs onto their
  two primary lane outputs proves equivalent for every input. Replay the raw
  bytes from the crate directory with the same
  grammar and converter.
- `enumerate_width.x` typechecks and converts successfully. The specialized
  function name in the emitted IR contains a colon, and the C++ IR parser
  rejects it.

```sh
dslx_interpreter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" --evaluator=dslx-interpreter signed_loop_wide.x
dslx_interpreter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" --evaluator=ir-interpreter signed_loop_wide.x
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" signed_loop_narrow.x
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" signed_range_length.x
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" signed_one_bit_range.x
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" enumerate_width.x > enumerate_width.ir
opt_main --top=__enumerate_width__main enumerate_width.ir
```

```sh
cd ..
DSLX_FUZZ_GRAMMAR=mission-local-v2 DSLX_FUZZ_CONVERTER="$XLSYNTH_TOOLS/ir_converter_main" \
  cargo fuzz run --sanitizer none --features with-bitwuzla-system \
  fuzz_dslx_semantics repros/signed_loop_observer.input
```
