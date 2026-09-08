<!-- SPDX-License-Identifier: Apache-2.0 -->

# Known-bits analysis

The `xlsynth_pir::known_bits` module provides standalone, forward three-state
analysis for PIR functions and blocks. It computes bits that are always zero or
always one, together with bounds on the number of set bits, without invoking
XLS, a solver, or an external executable. It does
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

`KnownBits` stores private, equal-width `IrBits` mask/value fields and inclusive
`min_ones`/`max_ones` population bounds. A set mask bit means that the corresponding
value bit is known. Unknown positions are normalized to zero in the value field.
Every compatible concrete value `v` satisfies both conditions:

```text
(v & mask) == value
min_ones <= popcount(v) <= max_ones
```

Constructors support unknown values, constants, and checked mask/value pairs.
Unknown values start with population bounds `[0,width]`; constants have their
exact population. Mask/value construction derives the bounds already implied
by the known bits. `with_popcount_bounds(min, max)` intersects additional bounds
with the existing facts, rejects reversed, out-of-width, or contradictory bounds,
and fills remaining unknown bits when their population is forced to zero or all
ones. It does not represent an impossible value as an ordinary fact.

Accessors expose the masks, bit counts, `min_ones()`, `max_ones()`, fully-known
status, and concrete-value containment. Zero counts are derived rather than
stored separately: `min_zeros() = width - max_ones()` and
`max_zeros() = width - min_ones()`. Zero-width bits have population `[0,0]`.
`contains` checks both masks and population bounds. MSB-first ternary formatting
still displays only the per-bit mask, using `X` for unknown positions; it does
not display the additional population information.

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
its input facts. Joins keep only known bits shared by every feasible alternative
and widen population bounds to include every alternative.

Population facts are node-local: transfers depend on the operation and its
immediate operand facts, not a recognized graph neighborhood or a contextual
guard. For example, a one-hot result can have no known bit positions but still
have population `[1,1]`; its OR and XOR reductions are therefore known one.
NOT swaps the one/zero bounds, concatenation adds bounds, and static routing
retains bounds for the surviving bits. Bitwise operations and feasible-arm joins
also propagate conservative bounds. Other transfers may retain only the bounds
implied by their output masks. No new analysis configuration or backward pass is
required.

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
- Generic random graphs evaluated on random concrete inputs, checking masks
  and population bounds at every node and aggregate leaf, including parameters
  and dead nodes.
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
and compares function facts with the equivalent combinational block. Concrete
membership checks cover population bounds as well as per-bit masks. It uses the
[shared mixed-vector input policy](../FUZZ.md#shared-concrete-input-sampling),
choosing structured patterns or a fresh subset of special-valued arguments for
each evaluation, with optional sparse perturbation of special values. See
[FUZZ.md](../FUZZ.md) for its invocation. Generation, analysis and interpreter
failures are sample failures, not silently discarded inputs.

Optional formal checks use the existing Bitwuzla configuration:

```bash
cargo test -p xlsynth-prover --features with-bitwuzla-system --test known_bits_test
```

The checker asks whether any node or aggregate leaf can violate either its mask
or population bounds. Population counters use enough bits to represent the full
value width, including widths above one machine word. Each of 256 deterministic
small generated graphs receives a 250 ms bounded query. A satisfiable result must
replay through concrete PIR evaluation; timeout or unknown is inconclusive.
Directed negative controls exercise false mask/count claims, dead nodes, aggregate
layout, and counts above 64. General zero-width arithmetic and partial-product
pairs remain in concrete tests because the SMT translator does not support them;
directed formal cases still cover zero-width values. No solver is used at runtime.

A local, unpublished differential harness uses the generic typed graph generator
and a separate C++ executable running only the eager `TernaryQueryEngine`,
pinned to XLS `v0.54.7`, revision
`78446462c07943896edd7d18720b205e3c0e0601`. It compares every node ID and typed
aggregate leaf, requiring every XLS-known bit to be present with the same value.
This comparison measures ternary-mask precision; the additional population
claims are checked by concrete and formal soundness tests.
Failures preserve the input bytes, IR and oracle output for replay.
Extension-enabled graphs use concrete PIR interpretation because upstream XLS
cannot parse those operations.

Finite formal checks and fuzzing do not establish universal soundness or XLS
precision parity.
