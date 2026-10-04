# Aug-opt optimizations

The [augmented optimizer](../src/optimizer.rs) applies local PIR rewrites to the
selected top function. In `Sandwich` mode, XLS optimization runs before the first
PIR round and after each round. `PirOnly` mode runs the PIR rounds directly.
Both modes rebuild range and known-bit facts each round.
The rewrites emit ordinary XLS operations; existing PIR extension operations
are preserved across XLS optimization through FFI wrappers.

The canonical entrypoints are
`xlsynth_aug_opt::run_aug_opt_over_ir_text` and
`xlsynth_aug_opt::run_aug_opt_over_ir_text_with_stats`. Both supply g8r costing
for profitability checks. See the [crate README](../README.md) for library use,
the standalone executable, and migration from the PIR and g8r entrypoints.

The following pseudocode uses `sel(p, [a, b])` for a one-bit selection that
returns `a` when `p` is zero and `b` when it is one. A slice `x[a:b]` contains
bits `a` through `b - 1`. Arithmetic on `N`-bit values is modulo `2^N`, and
shifts retain XLS overshift semantics. The linked source contains the exact
matchers and limits.

## Shift structure

### Affine shift amounts

A shift by a constant plus a zero-extended one-bit flag has only two possible
amounts:

```text
shift(x, zext(p) + K)
  -> sel(p, [shift(x, K), shift(x, (K + 1) mod 2^M)])
```

Here `M` is the amount width. In particular, when `K + 1` wraps, the second
amount is zero. The rewrite supports logical and arithmetic shifts, including
slices of their results, and exposes constant projections for later
simplification. The flag may also be represented by a concatenation with zero
prefix bits, or directly when the amount width is one.

### Constant shift choices

[Constant-choice fusion](../src/constant_shift_choices.rs) recognizes small
decision DAGs whose reachable shift amounts are all constant:

```text
shift(x, sel(p, [A, B]))
  -> sel(p, [shift(x, A), shift(x, B)])
```

It supports logical left and right shifts, including a slice of a logical
right shift. Constant shifts become source-bit projections with zero padding.
The recognizer handles nested indexed and priority selections, exact replicated
one-bit masks, and literal-prefix concatenations with a one-bit suffix. It
preserves selection defaults, priority, shared data, and oversized amounts.
A reachable variable amount rejects the candidate; structural limits bound
recognition and expansion.

Each site is compared independently through a
[`ShiftChoiceCostEvaluator`](../src/ir_cost.rs), using small graphs with the same
boundary inputs and any interior values still needed by outside users. Nearby
wiring and inversion preserve shared controls. Acceptance requires no increase
in either estimated area or delay and a strict improvement in at least one,
with a tolerance for delay roundoff. A tie or tradeoff keeps the original site.

Cost graphs borrow regions of the speculative working function. A proposal
appends replacement nodes before costing; acceptance redirects the original
users, while rejection discards the appended nodes. Sites are processed in
sequence so later proposals see the updated sharing. Evaluation errors discard
the working function and preserve the original input.

The [crate-root entrypoints](../src/lib.rs) supply an evaluator using g8r's
existing gate builder, reachable AND count, and Graph LE. External logic,
arrival times, and loads are outside this local model; downstream gate cleanup
can also change final costs. The optimizer's cost adapter uses g8r's neutral
region-lowering API, so neither g8r nor PIR depends on optimizer-specific
graphs or profitability policy.

### Predicates over shifted bits

- **Low bit of a left shift:** `shll(x, s)[0]` becomes
  `x[0] & (s == 0)`. Positive shifts force that output bit to zero.
- **Left-shifted slice compared with a literal:** equality becomes an OR of
  shift-distance tests and the source-bit constraints for each distance.
  Zero literals also account for shifts that force the whole slice to zero.
  The pass limits the number of generated terms.
- **Nonzero low slice of a right shift:** range and known-bit facts can prove
  that every positive in-range shift places a one in the observed slice. The
  rewrite then tests the shift range, with a separate test of the original low
  slice for shift distance zero. It uses known-one bits above the original low
  slice and requires an amount width of at most 64 bits. Overshifts still
  produce zero.

## Comparisons and selections

- **Guarded selection compared with one:**
  `nor(p, ne(sel(p, [x, y]), 1))` becomes `!p & (x == 1)`. This requires the
  same one-bit selector in both positions and a two-case selection without a
  default.
- **Comparison at the most significant bit's value:** for an unsigned `N`-bit
  value, `x > 2^(N-1) || (x == 2^(N-1) && hi)` becomes
  `x[N-1] && (or_reduce(x[0:N-1]) || hi)`. This requires `N >= 2` and a one-bit
  `hi`; the threshold must be the most significant bit's value.
- **Priority selection compared with a literal:** when analysis proves the
  default cannot equal the literal, equality becomes an OR of comparisons
  guarded by which case is actually selected. Each guard excludes earlier
  cases, preserving priority even when several selector bits are set.
- **Predicates over selections:** bit reductions and comparisons with constants
  move into the arms of a selection, leaving a selection of one-bit predicates.
  Ordinary selections must have a one-bit selector and two cases; priority
  selections must have a default. Comparisons require the selected value to
  have a single user to avoid replicating comparisons across several consumers.

## Arithmetic

### Split-adder recovery

[Split-adder recovery](../src/split_adder.rs) recognizes the low sum bit and
upper carry sum as one modular addition. For `x, y: bits[N]`, `N >= 2`, with
one-bit `xlo = x[0:1]` and `ylo = y[0:1]`:

```text
concat(x[1:N] + y[1:N] + zext<N-1>(xlo & ylo), xlo ^ ylo)
  -> x + y
```

The upper arithmetic must have exactly `N-1` bits. Matching allows commuted
operands and reassociated additions, and accepts a zero-prefix concatenation
for the carry extension. Upper arithmetic and the low XOR must have no users
outside the matched expression. Source operands may be shared.

This recovers an ordinary XLS `add` for downstream optimization and operator
mapping. It has no local gate-cost acceptance test: mapped area and delay can
trade off, including after gate cleanup. `AugOptOptions::recover_split_adders`
defaults to `true`; disabling it leaves the other rewrites enabled for
comparisons. `AugOptRewriteStats::split_adders_recovered` counts applications.

### Other arithmetic rewrites

- **Sum equal to zero:** `x + y == 0` becomes `y == 0 - x`, exposing a modular
  equation that XLS may simplify further.
- **Sum unequal to all ones:** `x + y != all_ones` becomes `!x != y`, where
  `!x` is the bitwise complement at the same width.
- **Unsigned modulo over selections:** `umod` distributes over selected operands,
  including nested selections on both sides. Constant arms can then fold in
  XLS. The pass limits the number of case combinations it creates.
- **Selected opposite subtracts:** `sel(p, [b - a, a - b])` becomes
  `sel(p, [b, a]) - sel(p, [a, b])`. This requires at least four result bits,
  a two-case one-bit selection without a default, and subtracts used only by
  that selection.

## Canonicalization before XLS optimization

Immediately before each post-rewrite XLS optimization in `Sandwich` mode,
exact Boolean masks can become selections:

```text
value & sign_ext(p)   -> sel(p, [0, value])
value & sign_ext(!p)  -> sel(p, [value, 0])
```

One-bit masks are handled directly. This rule requires a two-operand AND and a
value that depends on the same predicate, allowing XLS to simplify the value
under the selected condition. It leaves an AND that feeds another AND
unchanged. `PirOnly` mode does not run this canonicalization.

## Implementation and tests

The pass order and most rewrite tests are in [optimizer.rs](../src/optimizer.rs).
Constant-choice recognition and construction have their own
[module and unit tests](../src/constant_shift_choices.rs).
The [integration tests](../tests/) include `aug_opt_*` tests
for equivalence and mapped quality, including local cost decisions and their
composition with XLS optimization.
