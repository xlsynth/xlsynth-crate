# Bit-Blasted Output Ordering

This note describes the output-bit ordering we want all bit-blasted carriers to
follow:

- `GateFn`
- ASCII AIGER
- binary AIGER

The source of truth is the Verilog emitted today by:

- `xlsynth-driver ir2combo`
- `xlsynth-driver ir2pipeline`

When we say "flat output order" below, we mean the LSB-first view of the final
packed output bus:

- `out[0]` is the least-significant bit
- `out[1]` is the next bit
- and so on

For AIGER, this same order corresponds to the scalar output stream `o0`, `o1`,
`o2`, ...

## Verilog Ordering Today

The current libxls Verilog codegen behavior is:

- Bits: preserve ordinary bit index order.
- Tuples: later tuple elements occupy lower bits.
- Arrays: lower array indices occupy lower bits.
- Nested arrays: apply the array rule recursively at each array boundary.

That means Verilog flattening treats tuples and arrays differently.

### Bits

For `bits[4]`, the flat output order is:

```text
[x0, x1, x2, x3]
```

### Tuples

For `(bits[1], bits[2])` returned as `tuple(a, b)`, Verilog packs as:

```verilog
assign out = {a, b};
```

So the flat LSB-first order is:

```text
[b0, b1, a]
```

More generally, tuple element `N-1` occupies the least-significant chunk, and
tuple element `0` occupies the most-significant chunk.

### Arrays

For `bits[2][3]` returned as `array(a, b, c)`, Verilog packs as:

```verilog
assign out = {c, b, a};
```

So the flat LSB-first order is:

```text
[a0, a1, b0, b1, c0, c1]
```

More generally, array element `0` occupies the least-significant chunk, array
element `1` the next chunk, and so on.

### Nested Arrays

Apply the array rule recursively.

For `bits[1][2][2]` returned as:

```text
array(array(a, b), array(c, d))
```

Verilog packs as:

```verilog
assign out = {{d, c}, {b, a}};
```

So the flat LSB-first order is:

```text
[a, b, c, d]
```

## Bit-Blasted Contract

Our bit-blasted representations should match that same Verilog-derived flat
order exactly.

In practice, that means:

- tuple tails go in the low bits
- array element `0` goes in the low bits
- nested arrays keep element `0` lowest at every array boundary

The canonical test helper for this contract is:

- `xlsynth-pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type()`

Its inverse helper is:

- `xlsynth-pir::ir_value_utils::ir_value_from_lsb0_bits_with_layout()`

Those helpers intentionally model the current Verilog layout:

- tuple elements are flattened in reverse source order
- array elements are flattened in forward index order

## Example Matrix

The following examples are the expected flat LSB-first order:

| IR return shape | Example construction | Flat output order |
| --- | --- | --- |
| `bits[1]` | `identity(x)` | `[x0]` |
| `bits[2]` | `identity(x)` | `[x0, x1]` |
| `bits[4]` | `identity(x)` | `[x0, x1, x2, x3]` |
| `(bits[1], bits[1])` | `tuple(a, b)` | `[b, a]` |
| `(bits[1], bits[2])` | `tuple(a, b)` | `[b0, b1, a]` |
| `(bits[2], bits[2])` | `tuple(a, b)` | `[b0, b1, a0, a1]` |
| `bits[2][3]` | `array(a, b, c)` | `[a0, a1, b0, b1, c0, c1]` |
| `bits[1][2][2]` | `array(array(a, b), array(c, d))` | `[a, b, c, d]` |
| `bits[2][5][3]` | `array(row0, row1, row2)` | `row0` chunk first, then `row1`, then `row2`; within each row, element `0` chunk first |

## How We Implement The Same Order

There are three relevant layers.

### 1. PIR/IR Value Flattening

`xlsynth-pir/src/ir_value_utils.rs` defines the logical aggregate layout used by
the tests and by width-preserving reconstruction helpers.

- `flatten_ir_value_to_lsb0_bits_for_type()` is the reference flattening rule.
- `ir_value_from_lsb0_bits_with_layout()` is the matching reconstruction rule.

### 2. IR-to-Gate Lowering

`xlsynth-g8r/src/gatify/ir2gate.rs` now follows the same convention when
lowering structured values into flat `AigBitVector`s.

Key places:

- array construction (`NodePayload::Array`)
- literal flattening (`flatten_literal_to_bits`)
- array indexing helpers
- array update helpers

These paths now keep array element `0` in the least-significant chunk, matching
Verilog rather than reversing array order.

### 3. Gate/AIGER Serialization And Lift

Once a `GateFn` already has the right flat output order, the final AIGER
serializers preserve it.

- `xlsynth-g8r/src/aig_serdes/emit_aiger.rs`
- `xlsynth-g8r/src/aig_serdes/emit_aiger_binary.rs`

They emit:

- outputs in `gate_fn.outputs` order
- bits within each output in LSB-to-MSB order

On the lift side, the structured reconstruction code mirrors the same
tuple-vs-array convention:

- `xlsynth-g8r/src/aig_serdes/gate2ir.rs`
- `xlsynth-g8r/src/ir_aig_sharing.rs`

## Regression Coverage

We now check this contract in two complementary ways.

### Shared Ordering Matrix

`xlsynth-g8r/tests/test_bit_blast_output_order.rs` uses a shared corpus from
`xlsynth-g8r/src/test_utils.rs`:

- `interesting_ir_output_ordering_cases()`

These cases are built from scalar leaf inputs and structured outputs so the flat
bit positions stay observable. The test compares:

- expected flat bits from `flatten_ir_value_to_lsb0_bits_for_type()`
- actual flat `GateFn` outputs from `gate_sim::eval()`

This catches tuple-vs-array ordering mismatches in the bit-blasted lowering.

### Verilog Tooling Smoke Tests

`xlsynth-driver/tests/invoke_test.rs` also contains focused `ir2pipeline` /
`run-verilog-pipeline` tests for:

- tuple low-bit placement
- array element-0 low-bit placement
- nested-array low-bit placement

These tests keep the document anchored to the current user-visible Verilog
tooling behavior rather than only to internal helpers.
