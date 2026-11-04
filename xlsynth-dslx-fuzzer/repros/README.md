# Reduced DSLX converter failures

Run from this directory with the matching tools on PATH and DSLX_STDLIB_PATH
set to their stdlib. The signed-range, signed-loop and enumerate failures
below reproduced on stock XLS v0.61.0.

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
- `bound_tuple_update_index.x` binds the index of a two-dimensional array
  update to a local tuple. The DSLX interpreter agrees with the scalar result
  on all 32 inputs. The standalone converter at XLS revision `81133e348`
  aborts in `FunctionConverter::HandleBuiltinUpdate`, passing the tuple as
  one IR index. Inlining `(row, col)` at the update call converts correctly.
  With [google/xls#5260](https://github.com/google/xls/pull/5260) applied,
  both spellings convert to IR equivalent to the scalar specification.
- `map_builtin_name_collision.x` maps the built-in bit reversal over a
  two-bit word and XORs it with the result of the user identity `rev__2`.
  At input 1, DSLX returns 3. The converter at XLS revision `77b64c916`
  reuses the identity as the map body and returns 0. Renaming the identity
  restores the correct result on all four inputs.

```sh
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" --top=main bound_tuple_update_index.x
ir_converter_main --dslx_stdlib_path="$DSLX_STDLIB_PATH" --top=main map_builtin_name_collision.x
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
