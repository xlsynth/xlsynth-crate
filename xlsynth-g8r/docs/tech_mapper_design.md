# NAND2 Technology Mapping Design

This note describes the `nand2` AIG technology-mapping strategy in
`xlsynth-g8r`. For the `area` strategy, see the
[`aig-tech-map` documentation](../../xlsynth-driver/README.md#aig-tech-map-aiger-to-library-cell-netlist).

There are two separate decisions:

- the **mapping strategy**, which determines how an AIG is lowered into cells
- for `nand2`, the **cell-selection policy**, which chooses the concrete `INV`
  and `NAND2` variants used by that strategy

The strategy determines the produced structure; the policy chooses among
interchangeable implementations of the cell families used by `nand2`.

## NAND2 Strategy

The `nand2` strategy performs direct, deterministic structural lowering:

- each AIG `And2` becomes `NAND2` followed by `INV`
- negated operands become explicit edge `INV` instances
- negated outputs become explicit output `INV` instances
- the emitted netlist is purely structural and does not use continuous assigns

The purpose of this strategy is to preserve the AIG shape in a simple
standard-cell netlist. It is useful as a baseline, for inspection, and as a
predictable input to later netlist analyses such as STA.

This is not a general cell-covering algorithm. It does not search wider Liberty
cell sets, collapse Boolean cones, or optimize area or delay across alternative
decompositions.

## Cell-Selection Policies

The `nand2` strategy needs concrete `INV` and `NAND2` cells. The `aig-tech-map`
CLI exposes two `--cell_policy` values for choosing among matching cells in a
Liberty proto:

- `small-normal-vt`: prefer VT class `0`, then the unsuffixed canonical cell
  name, then an `x1` spelling
- `max-speed`: prefer the highest available VT class, then the highest parsed
  drive strength

This option applies only to `nand2`; specifying it with `area` is an error.

These policies do not change the mapped topology. They only choose which
concrete cell names appear in the output netlist. Drive strength is inferred
from cell names; neither policy compares timing data. Mapping does not require
timing data. Later STA requires a timing-enabled proto, but its current model may
not diagnose all missing or incomplete timing arcs. Verify timing coverage for
the selected cells before relying on its delay results; the `gv-sta` section of
the driver README describes the current checks.

If a library does not provide VT ordering metadata, all matching cells are
treated as unordered and selection still remains deterministic. If a candidate
set mixes ordered and unordered VT metadata, selection is rejected because the
relative ranking would be ambiguous.

## Cell Binding

The `nand2` strategy binds cell families by convention:

- inverter candidates have names beginning with `INV`, one input pin, and one
  output pin whose Boolean function is the negation of that input
- two-input NAND candidates have names beginning with `NAND2`, two input pins,
  and one output pin whose Boolean function is the negation of their
  conjunction

That is sufficient for `nand2`, but it is deliberately narrower than a general
Liberty-function matcher.

## Output Contract

The `nand2` strategy returns parsed netlist structures, not only text, so other
library flows can consume the result without reparsing emitted Verilog-like
netlists. Text emission is a separate step layered on top of the parsed
representation.

The `nand2` output is deterministic:

- generated instance names are stable
- generated net names are stable
- connections are emitted in sorted pin-name order

That keeps regression output reviewable and makes baseline comparisons useful.
