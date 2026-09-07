<!-- SPDX-License-Identifier: Apache-2.0 -->

# Range analysis

`xlsynth_pir::range_analysis` provides standalone forward interval-set analysis
for functions and blocks. It does not invoke XLS, a solver, or an external
executable, and does not rewrite IR. Existing XLS-backed production consumers
are unchanged.

## API and fact contract

`analyze_fn(&ir::Fn)` and `analyze_block(&ir::Block)` return
`Result<RangeAnalysis<'_>, AnalysisError>`. The result borrows its immutable
graph. `get(node)`, `bits(node)`, `leaf(node, path)`, and `iter()` follow the
same conventions as [known-bits analysis](ir_known_bits.md).

Every non-`Nil` node is analyzed, including dead computations. `None` denotes
an invalid reference or reserved/deleted slot, not an unconstrained value.
Tuple/array paths preserve IR element order; tokens and empty aggregates retain
their shape. Facts must not be applied to another or mutated graph merely
because node text IDs match.

Each `IntervalSet` owns arbitrary-width `IrBits` endpoints. Private fields
enforce a canonical union of sorted, disjoint, nonadjacent, unsigned inclusive
intervals. Constructors split wrapping intervals at zero. Signed operations
interpret the same bit patterns in two's-complement order.

- `full(width)` includes every bit pattern; `empty(width)` includes none.
- `bits[0]` has one possible value. Its full set is not empty.
- `singleton`, `from_intervals`, and borrowed endpoint accessors preserve widths.
- `union`, `intersect`, and `is_subset_of` operate on complete sets, not hulls.
- Signed/unsigned extrema, membership, bounded cardinality/enumeration, and
  conversion to/from known bits are available.
- `intersects_known_bits` tests whether any represented value satisfies a
  known-bit pattern; it does not require every matching value to be included.
- `minimize(limit)` explicitly merges gaps to bound fragmentation. Exact set
  algebra never silently coarsens its result.

`RangeValue::contains` checks a concrete value against every leaf, including
its aggregate shape. `RangeValue::join` takes exact leafwise unions.

For a node's concrete reachable values `C`, its reported set `R` must satisfy
`C ⊆ R`. Aggregates do not encode relationships between different leaves.

## Evaluation and scope

Local validation checks operand bounds once, cycles, types, and operation
semantics. Function entry points also check signatures/returns; blocks require
the reserved sentinel. Traversal is topological, independent of storage order.
Validation is shared internally with known bits without changing that public
API. Package-level resource, callee, and instance validation remains the package
verifier's responsibility.

Transfers cover modular arithmetic, signed/unsigned division and comparisons,
bitwise operations/reductions, shifts, concatenation/slicing/extensions,
encoding/decoding, selects, gates, and aggregates. Array reads and slices clamp
out-of-bounds positions; uncertain writes preserve possible no-op outcomes.
One-hot selects OR the enabled values, including multi-hot inputs.

Blocks are analyzed for one cycle with arbitrary ports and current register
values. Reset constants, load enables, and assertions do not constrain current
state. Output ports, register writes, and instance inputs have unit results;
effects have their declared unit/token shape. Instance outputs, calls, loops,
and partial-product pairs are conservative opaque values. There is no
inter-instance propagation, state-invariant inference, or backward/contextual
analysis.

All six extension operations have direct transfers:

- `ext_carry_out`: widened addition and carry extraction.
- `ext_nary_add`: signed/unsigned resizing, negation, and modular summation;
  lowering architecture does not change arithmetic semantics.
- `ext_clz` and `ext_prio_encode`: feasible count/first-set ranges, including
  the zero-input sentinel and offsets before output truncation.
- `ext_mask_low`: saturated mask ranges and guaranteed low-one/high-zero facts.
- `ext_normalize_left`: conditional leading-zero/shift ranges, with separately
  retained CLZ results and a bounded packed-known-bits fallback.

No extension transfer constructs a lowered graph during analysis.

## Precision and work bounds

The differential baseline is a fresh standalone XLS `RangeQueryEngine`, with
no givens, BDD engine, contextual engine, or combined query engine. The initial
oracle uses XLS `v0.54.7`, revision
`78446462c07943896edd7d18720b205e3c0e0601`.

The required differential relation is `R_rust ⊆ R_xls` for each matching node
and bits leaf. Comparing only unsigned extrema, interval counts, or total
cardinality would miss precision regressions.

The implementation follows the baseline's precision-sensitive policies:
small exact domains, bounded interval combinations, and operation-specific
coarsening. Default scalar coarsening retains 16 intervals; exact enumeration
uses at most 16 concrete combinations; expensive variadic products are reduced
to at most 1,000,000 interval combinations. These are not universal caps on
all routing results. Array routing retains exact unions. Select results are
coarsened after joining alternatives.

Gap merging preserves the largest gaps, including XLS's later-gap tie-break.
Coarsening is not monotone: tighter input facts can change which gaps get
merged. Differential graph tests therefore remain necessary even for locally
more precise transfers. Exact singleton/identity/static cases precede expensive
fallbacks.

Concatenation enumerates the complete concrete operand product when it contains
at most 16 combinations and the width-dependent work estimate fits. This preserves
sparse results such as concatenating
an unknown bit with four zero bits: `{0,16}`, rather than the hull `[0,16]`.
The budget is checked before enumeration; constant-only concatenations pack
borrowed values directly. Larger products retain the endpoint fallback, which
pre-fills precise operands and prunes Cartesian suffix choices as soon as a
non-singleton prefix makes their endpoint intervals overlap. Generic high-fan-in
regressions compare that fallback with an unpruned reference.

Variable-position bit-slice updates enumerate at most 16 possible starts when
the additional width-dependent work estimate fits the existing work limit.
Each position uses the fixed-update transfer, including clipping and out-of-bounds
no-op behavior. For example, writing a one into a four-bit zero value at a start
in `[0,3]` produces `{1,2,4,8}`. The joined result is intersected with the existing
known-bits fallback so the refinement cannot discard its bit facts. If the work
limit or resulting fragmentation limit would be exceeded, the fallback remains
in use. These are local forward transfers, not context-sensitive select or guard
analysis.

Cheap precision improvements include exact interval bitwise NOT, constant
unsigned remainder, and a shift/truncate fallback for dynamic slices with large
start domains. For example, four-bit `~[3,5]` yields `[10,12]`, and
`[10,12] % 8` yields `[2,4]`.

## Validation

Focused deterministic tests:

```bash
cargo test -p xlsynth-pir --lib range_analysis
cargo test -p xlsynth-pir --test range_interval_set_test --test range_analysis_test
cargo test -p xlsynth-pir --doc
```

These cover exhaustive small intervals and concrete transfers, zero/limb-boundary
widths, interval holes, signed/modular edge cases, graph validation, extensions,
arbitrary-state blocks, all-node random concrete evaluation, and function/block
fact agreement.

Two coverage-guided targets extend concrete soundness testing:

- `fuzz_range_analysis_soundness` in PIR checks every node/leaf of a generated
  function and compares the equivalent block's facts.
- `fuzz_range_analysis_block_soundness` in g8r checks actual sequential blocks
  over arbitrary current state and short traces using observed PIR evaluation.

Both use independent graph/input entropy and the shared structured/special/
uniform sampler, with extensions enabled. See [FUZZ.md](../FUZZ.md) for commands
and the input policy. Routine campaigns use `--sanitizer none`.

Optional formal checks use the existing solver configuration:

```bash
cargo test -p xlsynth-prover --features with-bitwuzla-system --test range_analysis_test
```

They ask whether any reported node/leaf can lie outside its complete interval
union. SAT results are replayed concretely; timeout/unknown is inconclusive.
Negative controls test holes, dead nodes, and aggregate layout. The formal
generator explicitly excludes unsupported SMT-translator cases; these remain
in concrete testing. No solver is used by the analysis itself.

A locally built C++ graph oracle and differential runner compare all node/leaf facts
on identical standard-XLS graphs, retaining failure inputs, IR, and oracle
output. They reject missing records, mismatched identities, malformed output,
timeouts, and subprocess failures. Extension comparisons use standard-op
lowerings rather than passing unsupported extension syntax to XLS.

These development helpers are not shipped in the repository. An opt-in ignored
scalar differential test accepts a locally built interval-operations oracle via
`XLS_RANGE_TRANSFER_ORACLE`. The protocol and pinned version header are described
in `xlsynth-pir/src/range_analysis/ops_xls_test.rs`. This exercises
generated abstract input sets directly without exposing conditional facts as
unconditional public analysis results.

```bash
XLS_RANGE_TRANSFER_ORACLE=/path/to/range_transfer_oracle \
XLS_RANGE_TRANSFER_SAMPLES=100000 XLS_RANGE_TRANSFER_SEED=0 \
cargo test -p xlsynth-pir --lib generated_interval_givens_are_at_least_as_precise_as_xls -- --ignored --nocapture
```

Fuzzing is evidence of soundness and precision parity, not a universal proof.
Matching this standalone engine does not establish parity with libxls's
combined/contextual analysis API. Production consumer migration is separate.
