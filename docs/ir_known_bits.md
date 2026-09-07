<!-- SPDX-License-Identifier: Apache-2.0 -->

# Known-bits analysis

The `xlsynth_pir::known_bits` module provides standalone, forward three-state
analysis for PIR functions and blocks. It computes bits that are always zero or
always one, without invoking XLS, a solver, or an external executable. It does
not rewrite the graph or change the source of facts used by existing production
consumers.

## API and fact contract

The entry points are `analyze_fn(&ir::Fn)` and `analyze_block(&ir::Block)`.
Both return `Result<KnownBitsAnalysis<'_>, AnalysisError>`.

The result borrows its immutable graph. Its queries are:

- `graph()`: the analyzed graph.
- `get(node)`: type-shaped facts for one node.
- `bits(node)`: facts for a scalar bits node.
- `leaf(node, path)`: a bits leaf in a tuple or array.
- `iter()`: facts in deterministic graph storage order.

All non-`Nil` nodes are analyzed, including dead computations. Invalid
references and reserved/deleted `Nil` slots return `None`; valid bits nodes
with no known bits still return `Some`. Facts must not be detached and applied
to a mutated graph merely because text IDs still match.

`KnownBits` stores private, equal-width `IrBits` mask/value fields. A set mask
bit means that the corresponding value bit is known. Unknown positions are
normalized to zero in the value field. Every compatible concrete value `v`
satisfies:

```text
(v & mask) == value
```

Constructors support unknown values, constants, and checked mask/value pairs.
Accessors expose the masks, bit counts, fully-known status, concrete-value
containment, and MSB-first ternary formatting with `X` for unknown bits.

`KnownValue` preserves tuple and array structure rather than flattening
aggregates. Tokens and empty aggregates have no bits; zero-width bits are
supported. The standalone types remain separate from existing XLS-backed
adapter types, so adopting this module does not change those APIs.

## Evaluation and coverage

The evaluator validates local graph contracts, then processes nodes in
topological order. Non-topological storage is permitted. Validation checks
operand bounds, cycles, local types and node semantics; function analysis also
checks the signature and return. Package-level callee, register and instance
validity remain the package verifier's responsibility.

Each transfer conservatively includes every concrete result compatible with
its input facts. Joins keep only known bits shared by every feasible
alternative.

| Operations | Forward transfer |
| --- | --- |
| Parameters, input ports, register reads | Unconstrained inputs/current state; resets are not assumptions |
| Literals, identity, tuples, arrays | Exact constants and type-shaped routing |
| Bitwise operations and reductions | Three-state Boolean evaluation |
| Add, subtract, negate | Three-state sum/carry propagation |
| Multiply | Partial-product reduction following the XLS Dadda schedule |
| Divide and remainder | Restoring division, including signed and zero-divisor semantics |
| Equality and ordered comparisons | Leaf equality and known-bit extremal bounds |
| Concat, slices, extension, reverse, shifts | Bit routing and joins across feasible dynamic positions |
| Decode, encode, one-hot | Three-state encoding, including priority |
| Selects and gate | Feasible-arm joins, priority, and multi-hot OR semantics |
| Array reads, slices, updates | Feasible-index joins; reads clamp and OOB writes preserve the array |
| Outputs, register writes, instance inputs, events | Unit/token results; no constraints propagated backward |
| Instance outputs, calls, loops, partial-product pairs | Conservative unknowns |
| `ext_carry_out`, `ext_nary_add` | Carry recurrence and signed/negated modular addition |
| `ext_prio_encode`, `ext_clz`, `ext_mask_low` | Feasible first-set/count facts and saturated mask thresholds |
| `ext_normalize_left` | Conditional CLZ/shift facts with bounded work and a linear fallback |

Block analysis describes a single cycle for arbitrary current register values.
Reset values, load enables and assertions do not constrain those values.
Hierarchical and extern outputs remain opaque; downstream local logic is still
analyzed. No inter-instance propagation or multi-cycle invariant inference is
performed.

Extension transfers operate directly on facts without constructing a lowered
graph. Priority/CLZ transfers visit feasible first-set positions rather than
enumerating concrete inputs. CLZ offsets retain their overflow bit before
truncation to the result width; normalization offsets follow the interpreter's
saturating shift semantics.

## Work limits and representation

Simple Boolean operations, reductions and static routing work directly on
packed `IrBits` masks/values. Routing visits individual bits without expanding
ternary vectors. More complex arithmetic retains per-bit algorithms but packs
results directly into limbs. Aggregate joins preallocate their result vectors
and move owned first alternatives into accumulators.

Expensive dynamic transfers retain conservative work limits: shifts and
bit-slice updates wider than 256 bits, compound results above 65,536 flattened
bits, and enumeration with at least ten unknown array-index bits.

Cheap cases run before these fallbacks: constants and direct aggregate routing,
fixed aggregate indices/slice starts, fixed or provably saturated shifts,
zero/all-ones shift identities, and fixed/OOB/empty bit-slice updates. A
one-dimensional array access with a wide uncertain index can instead scan at
most 512 actual positions when
`array_length * (index_width + result_width)` is at most 65,536. Reads and slices
include clamped last-element outcomes; uncertain writes retain possible no-op
outcomes. Values are not truncated to host integer widths.

Conditional normalization uses a budget of 65,536 for the number of feasible
CLZ counts times the larger input/output width. One fixed count is always
allowed. Beyond that budget, the linear fallback still computes raw CLZ facts
and retains guaranteed normalized zeros and surviving leading-one facts.

## Validation

In-tree tests cover:

- Exhaustive small ternary domains, soundness and precision regressions.
- Arbitrary widths, word boundaries, zero-width bits and aggregate shapes.
- Generic random graphs evaluated on random concrete inputs, checking every
  node and aggregate leaf, including parameters and dead nodes.
- All six extensions, including signed/negated sums and CLZ/normalization
  offsets.
- Agreement between function analysis and the same graph represented as a block,
  plus arbitrary register-state checks.
- Work-limit boundaries, fixed routing and conservative dynamic fallbacks.
- Malformed local graphs and graph-borrowing lifetime constraints.

Run the focused tests with:

```bash
cargo test -p xlsynth-pir --lib known_bits
cargo test -p xlsynth-pir --test known_bits_test --test known_bits_extensions_test --test known_bits_work_limits_test
cargo test -p xlsynth-pir --doc
```

The in-tree [libFuzzer target](../xlsynth-pir/fuzz/fuzz_targets/fuzz_known_bits_soundness.rs)
extends these checks with coverage-guided graph generation and independently
mutable concrete-input seeds. It checks every node, enables all six extensions,
and compares function facts with the equivalent combinational block. A shared
input helper randomly divides the bounded input budget between uniform and
corner-biased sampling, then shuffles the resulting vectors. See
[FUZZ.md](../FUZZ.md) for its invocation. Generation, analysis and interpreter
failures are sample failures, not silently discarded inputs.

A local, unpublished differential harness uses the generic typed graph generator
and a separate C++ executable running only the eager `TernaryQueryEngine`,
pinned to XLS `v0.54.7`, revision
`78446462c07943896edd7d18720b205e3c0e0601`. It compares every node ID and typed
aggregate leaf, requiring every XLS-known bit to be present with the same value.
Failures preserve the input bytes, IR and oracle output for replay.
Extension-enabled graphs use concrete PIR interpretation because upstream XLS
cannot parse those operations.

Validation on 2026-09-07 included 6,000 strict differential samples: 2,000 each
for small functions, wide functions and sequential blocks, with seeds 12,000
through 13,999. The small profile also passed 8,000 concrete all-node
evaluations. A separate extension-enabled sweep passed 2,000 graphs and 16,000
concrete evaluations at widths through 257, with function/block fact agreement.
The in-tree mixed-input libFuzzer target additionally passed 124,884 executions
in a 61-second sanitizer-free campaign (including corpus initialization and
replays, not distinct graph counts). The workspace run with
`--features with-bitwuzla-system` passed 4,409 tests, with 12 skipped; nextest
marked one passing test as leaky. All four PIR doctests passed separately. The
solver feature is needed by existing workspace tests, not by this analysis.

These checks do not constitute a formal soundness proof or a claim of universal
XLS precision parity. No formal known-bit claim checker is implemented.
