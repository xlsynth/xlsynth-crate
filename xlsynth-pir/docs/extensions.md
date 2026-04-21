# PIR Extension Ops

PIR extension ops are typed `NodePayload` variants that carry intent not present
as a single upstream XLS IR op. They are legal in xlsynth PIR, but they must be
lowered before handing text to tools that only understand upstream XLS IR.

These ops are largely a staging area for experimenting with IR concepts before
proposing them for inclusion in upstream XLS IR. They let xlsynth preserve and
exploit higher-level intent in local passes, especially g8r lowering and
prep-for-gatify rewrites, without first requiring an upstream IR change.

The main tradeoff is optimizer compatibility. The upstream XLS optimizer cannot
operate directly on xlsynth-only extension ops. Any flow that hands IR to
upstream XLS must first desugar the extension ops, or emit upstream-compatible
FFI wrappers for flows that intentionally preserve an external-call boundary.

The central implementation points are:

- [Node payload definitions](../src/ir.rs)
- [Parser and FFI-wrapper lift-back](../src/ir_parser.rs)
- [Evaluator](../src/ir_eval.rs)
- [Desugaring and FFI-wrapper export](../src/desugar_extensions.rs)
- [Prover translation](../../xlsynth-prover/src/prover/translate.rs)
- [g8r lowering](../../xlsynth-g8r/src/gatify/ir2gate.rs)
- [prep-for-gatify idiom rewrites](../../xlsynth-g8r/src/gatify/prep_for_gatify.rs)

## Common Rules

- Extension op names use the `ext_` prefix in textual PIR.
- Extension ops are reported by `NodePayload::is_extension_op`.
- `NodePayload::get_operator` returns the textual opcode name.
- Native text round-trips keep the extension op.
- Export to upstream-compatible XLS IR either desugars extension ops inline or
  emits FFI wrappers with `xlsynth_pir_ext=...` metadata.
- FFI-wrapped text is parsed back into native extension ops when the helper
  signature and metadata match.

## Op Index

| Op | Text form | Result type | Main purpose |
| --- | --- | --- | --- |
| `ext_carry_out` | `ext_carry_out(lhs, rhs, c_in, id=...)` | `bits[1]` | Carry-out bit from `lhs + rhs + c_in`. |
| `ext_prio_encode` | `ext_prio_encode(arg, lsb_prio=<bool>, id=...)` | `bits[ceil_log2(N + 1)]` | Priority index of the first set bit, with `N` sentinel for none. |
| `ext_clz` | `ext_clz(arg, id=...)` | `bits[ceil_log2(N + 1)]` | Count leading zeros. |
| `ext_mask_low` | `ext_mask_low(count, id=...)` | `bits[N]` | Low-bit mask with dynamic count and saturation. |
| `ext_nary_add` | `ext_nary_add(..., signed=[...], negated=[...], arch=<optional>, id=...)` | `bits[N]` | Multi-operand add/sub with per-term signedness and optional adder architecture. |

## `ext_carry_out`

Text form:

```text
ret carry: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)
```

Typing:

- `lhs: bits[N]`
- `rhs: bits[N]`
- `c_in: bits[1]`
- result: `bits[1]`

Semantics:

```text
bit_slice(zero_ext(lhs, N + 1) + zero_ext(rhs, N + 1) + zero_ext(c_in, N + 1),
          start=N,
          width=1)
```

Why this extension exists:

The desugared XLS IR form describes a full `N + 1`-bit adder and then keeps
only the carry bit. `ext_carry_out` preserves the fact that the low `N` sum bits
are not observable, so g8r can lower the result as carry-propagation logic,
which has a cheaper representation as an OR-chain-style carry network than
emitting a full adder and discarding the sum bits. It also gives
prep-for-gatify a stable target for recognizing carry-bit idioms that may
otherwise be obscured by slices and extensions.

FFI wrapper:

- Helper name: `__pir_ext__ext_carry_out__w<N>`
- Metadata: `xlsynth_pir_ext=ext_carry_out;width=N`
- Template module name: `pir_ext_carry_out`

Relevant tests:

- [PIR round-trip and simulation](../tests/ext_carry_out_roundtrip_and_sim.rs)
- [Desugar/prover equivalence](../../xlsynth-prover/tests/ext_carry_out_desugar_equiv.rs)
- [prep-for-gatify rewrite coverage](../../xlsynth-g8r/tests/ext_carry_out_prep_for_gatify_equiv.rs)

## `ext_prio_encode`

Text form:

```text
ret index: bits[3] = ext_prio_encode(arg, lsb_prio=true, id=2)
```

Typing:

- `arg: bits[N]`
- result: `bits[ceil_log2(N + 1)]`
- `lsb_prio=true` means the least-significant set bit wins.
- `lsb_prio=false` means the most-significant set bit wins.

Semantics:

- If any bit in `arg` is set, return the winning bit index.
- If no bit in `arg` is set, return the sentinel value `N`.

Inline desugaring uses `one_hot(arg, lsb_prio=...)` followed by `encode(...)`.

Why this extension exists:

The desugared XLS IR form reifies `one_hot(arg)` and then feeds that vector to
`encode(...)`. That is a useful semantic basis, but reifying the one-hot result
builds its own prefix-scan structure before the encode step consumes it. When
the two operations are known to be fused into one priority encoder, g8r has an
opportunity to avoid materializing the intermediate one-hot vector and emit a
more direct encoded-priority structure. The explicit op also preserves the
priority direction as an attribute.

FFI wrapper:

- Helper name: `__pir_ext__ext_prio_encode__w<N>__lsb<0|1>`
- Metadata: `xlsynth_pir_ext=ext_prio_encode;width=N;lsb_prio=<true|false>`
- Template module name: `pir_ext_prio_encode`

Relevant tests:

- [g8r prep rewrite coverage](../../xlsynth-g8r/tests/ext_prio_encode_prep_for_gatify_equiv.rs)
- [g8r QoR/lowering coverage](../../xlsynth-g8r/tests/ext_prio_encode_qor_ir2gates_test.rs)

## `ext_clz`

Text form:

```text
ret count: bits[4] = ext_clz(arg, id=2)
```

Typing:

- `arg: bits[N]`
- result: `bits[ceil_log2(N + 1)]`

Semantics:

- Return the number of leading zero bits in `arg`.
- If `arg` is all zero, return `N`.

Inline desugaring uses `reverse(arg)`, then `one_hot(..., lsb_prio=true)`, then
`encode(...)`.

Why this extension exists:

CLZ is a common arithmetic-normalization primitive. The equivalent XLS IR form
is effectively `encode(one_hot(reverse(arg), lsb_prio=true))`, which is
semantically clear but hides that only a leading-zero count is needed. The main
benefit of `ext_clz` is giving g8r a chance to emit a CLZ-specific gatification
pattern instead of first building the generic reverse/one-hot/encode structure.
It also lets prep-for-gatify preserve the CLZ-specific intent instead of only
seeing a generic priority encode.

FFI wrapper:

- Helper name: `__pir_ext__ext_clz__w<N>`
- Metadata: `xlsynth_pir_ext=ext_clz;width=N`
- Template module name: `pir_ext_clz`

Relevant tests:

- [PIR round-trip and simulation](../tests/ext_clz_roundtrip_and_sim.rs)
- [Desugar/prover equivalence](../../xlsynth-prover/tests/ext_clz_desugar_equiv.rs)
- [g8r lowering and prep rewrite coverage](../../xlsynth-g8r/tests/ext_clz_ir2gates_equiv.rs)

## `ext_mask_low`

Text form:

```text
ret mask: bits[8] = ext_mask_low(count, id=2)
```

Typing:

- `count: bits[M]`
- result: `bits[N]`
- `M` and `N` may be zero.
- The result width determines the mask width; there is no separate size
  attribute in the node payload.

Semantics:

- Output bit `i` is `1` iff `count > i`.
- `count == 0` returns all zeros.
- `count >= N` saturates to all ones.
- For `N == 0`, return `bits[0]`.

For `N > 0`, inline desugaring is:

```text
(bits[N]:1 << count) - bits[N]:1
```

g8r lowers this with a recursive boundary-decoder structure instead of a
general payload barrel shifter. The helper is
[`gatify_mask_low`](../../xlsynth-g8r/src/ir2gate_utils.rs).

Why this extension exists:

The desugared XLS IR form looks like shifting a literal payload and subtracting
one. A generic shift/sub lowering can build a barrel-shifter-like network even
though there is no arbitrary payload to route. What the mask needs is a postfix
OR-scan/boundary structure: once the decoded count boundary has passed a bit,
that bit is true, and all lower bits are true as well. `ext_mask_low` keeps that
intent explicit so g8r can avoid the barrel-shifter shape and emit a cheaper
recursive mask structure. It also gives prep-for-gatify a stable replacement
target for common low-mask idioms before they are obscured by later rewrites.

FFI wrapper:

- Helper name: `__pir_ext__ext_mask_low__outw<N>__countw<M>`
- Metadata: `xlsynth_pir_ext=ext_mask_low;out_width=N;count_width=M`
- Template module name: `pir_ext_mask_low`

Recognized prep-for-gatify idioms:

```text
sub(shll(bits[N]:1, count), bits[N]:1)
add(shll(bits[N]:1, count), bits[N]:all_ones)
not(shll(bits[N]:all_ones, count))
shll(shrl(bits[N]:all_ones, count), count)
shrl(bits[N]:all_ones, count)
concat(0..., ext_mask_low(count)) when count is proven <= low mask width
```

Relevant tests:

- [PIR round-trip and simulation](../tests/ext_mask_low_roundtrip_and_sim.rs)
- [Desugar/prover equivalence](../../xlsynth-prover/tests/ext_mask_low_desugar_equiv.rs)
- [g8r lowering equivalence](../../xlsynth-g8r/tests/ext_mask_low_ir2gates_equiv.rs)
- [prep-for-gatify rewrite coverage](../../xlsynth-g8r/tests/ext_mask_low_prep_for_gatify_equiv.rs)
- [g8r gate stats and QoR sweep](../../xlsynth-g8r/tests/mask_low_gate_stats_sweep_test.rs)
- [prep-for-gatify fuzz target](../../xlsynth-g8r/fuzz/fuzz_targets/fuzz_prep_for_gatify_mask_low.rs)

## `ext_nary_add`

Text form:

```text
ret sum: bits[16] = ext_nary_add(
  a,
  b,
  c,
  signed=[false, true, false],
  negated=[false, false, true],
  arch=brent_kung,
  id=4)
```

The formatted PIR text is normally emitted on one line; this example is wrapped
for readability.

Typing:

- Every operand must be bits-typed.
- Result must be `bits[N]`.
- `signed` and `negated` lists are required and must have one entry per
  operand.
- `arch` is optional.

Term semantics:

- If `signed[i]` is true, operand `i` is sign-resized to `N`; otherwise it is
  zero-resized to `N`.
- If the operand is wider than `N`, it is truncated to the low `N` bits.
- If `negated[i]` is true, the resized term is two's-complement negated.
- All terms are summed modulo `2^N`.
- With zero terms, or with `N == 0`, the result is zero.

Supported architecture tags:

- `ripple_carry`
- `brent_kung`
- `kogge_stone`

Inline desugaring emits the resize/truncate/negate operations followed by a
chain of binary `add` operations. The `arch` tag guides g8r lowering but does
not change mathematical semantics.

Why this extension exists:

The equivalent XLS IR form is usually a chain or tree of binary add/sub nodes
with surrounding sign/zero extensions, truncations, and negations. Once written
that way, it is harder for g8r to recover the whole multi-operand addition
problem and choose an implementation strategy globally. `ext_nary_add` keeps
all terms, signedness, negation, and optional architecture in one node, allowing
g8r to build a single multi-operand adder structure, combine literal
contributions, account for negation corrections, and honor an explicit adder
architecture choice.

FFI wrapper:

- Base helper name: `__pir_ext__ext_nary_add__outw<N>__ops<W0>_<W1>...`
- Optional helper suffixes:
  - `__sgn<0|1>_<0|1>...` when any signed bit is true
  - `__neg<0|1>_<0|1>...` when any negated bit is true
  - `__arch<arch>` when an architecture is present
- Metadata includes:
  - `xlsynth_pir_ext=ext_nary_add`
  - `out_width=N`
  - `operand_signed=<csv bool list>`
  - `operand_negated=<csv bool list>`
  - optional `arch=<arch>`
- Current emitted metadata omits operand widths because the parser recovers
  them from the helper signature. The parser also accepts optional
  `operand_widths=<csv usize list>` and checks it against the helper signature.
- Template module name: `pir_ext_nary_add`

Relevant tests and fuzzing:

- [PIR round-trip and simulation](../tests/ext_nary_add_roundtrip_and_sim.rs)
- [Desugar/prover equivalence](../../xlsynth-prover/tests/ext_nary_add_desugar_equiv.rs)
- [g8r lowering coverage](../../xlsynth-g8r/tests/ext_nary_add_ir2gates_test.rs)
- [g8r gate stats and QoR sweeps](../../xlsynth-g8r/tests/ext_nary_add_gate_stats_sweep_test.rs)
- [g8r equivalence fuzz target](../../xlsynth-g8r/fuzz/fuzz_targets/fuzz_ext_nary_add_gatify_equiv.rs)
- [PIR eval/desugar fuzz target](../fuzz/fuzz_targets/fuzz_ext_nary_add_eval_equiv.rs)

## Adding A New Extension Op

At minimum, update these places:

- [NodePayload definition and text rendering](../src/ir.rs)
- [Operand traversal/remapping](../src/ir_utils.rs)
- [Reference verification](../src/ir_verify.rs)
- [Type deduction](../src/ir_deduce.rs)
- [Validation](../src/ir_validate.rs)
- [Parser and FFI-wrapper lift-back](../src/ir_parser.rs)
- [Evaluator](../src/ir_eval.rs)
- [Desugaring and FFI-wrapper export](../src/desugar_extensions.rs)
- [Prover translation](../../xlsynth-prover/src/prover/translate.rs)
- [g8r lowering, when the op reaches gates](../../xlsynth-g8r/src/gatify/ir2gate.rs)

Also add tests in the same style as the existing extension tests:

- Native text round-trip and simulation under `xlsynth-pir/tests/`
- Desugar/prover equivalence under `xlsynth-prover/tests/`
- g8r lowering and prep-for-gatify coverage under `xlsynth-g8r/tests/`
- Fuzz target coverage when the op affects `xlsynth-g8r` behavior
