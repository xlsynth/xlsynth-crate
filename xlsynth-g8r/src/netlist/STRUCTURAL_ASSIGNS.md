<!-- SPDX-License-Identifier: Apache-2.0 -->

# Liberty-free structural-assign semantics

This document describes the narrow Verilog subset accepted by the Liberty-free
`gv2aig` structural path.

## Intent

This mode targets effectively structural netlists, not general behavioral
Verilog. The accepted constructs are intentionally small so validation and AIG
projection stay deterministic and easy to reason about.

## Supported syntax

- Continuous `assign` statements only.
- Combinational bitwise operators `~`, `&`, `|`, and `^`.
- Simple nets, bit-selects, part-selects, parentheses, and literals.

## Rejected syntax

- Cell instances, `inout` ports, concatenation, ternaries, logical operators,
  arithmetic, shifts, reductions, and procedural statements.

## Width semantics

- Bitwise `~`, `&`, `|`, and `^` use exact-width structural semantics.
- We do **not** apply implicit Verilog operand sizing to bitwise expressions in
  this mode. Inputs to `&`, `|`, and `^` must already have equal widths.
- Plain assignments are also exact-width by default when the RHS is a net or a
  composed expression.
- A bare literal RHS is the one intentional exception:
  - `assign y = 0;`
  - `assign y[3:0] = 1'b0;`
  - `assign y[0] = 2'b10;`
- For a bare literal RHS, structural mode zero-extends or truncates the literal
  to the destination width. This is meant for common tie-off patterns in
  netlists, not as a general behavioral sizing rule.

## Dependency semantics

- Dependency resolution is performed at bit granularity, so overlapping slices
  are accepted when the induced per-bit graph is acyclic.
- Packed ranges such as `[3:0]` and `[0:3]` are both supported; bit numbering
  follows the declared range rather than assuming descending declarations.
