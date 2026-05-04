# Technology Mapper Design

This note describes the current AIG technology-mapping design in
`xlsynth-g8r`.

There are two separate decisions in the mapper:

- the **mapping mode**, which decides how an AIG is lowered into cell families
- the **cell-selection policy**, which decides which concrete cell variant to
  use after the mapper has chosen a family such as `INV` or `NAND2`

Keeping those separate matters. A mode changes the produced structure; a policy
only chooses among interchangeable implementations of a structure the mode has
already selected.

## Mapping Mode

### Structural Baseline

This mode performs direct, deterministic structural lowering:

- each AIG `And2` becomes `NAND2` followed by `INV`
- negated operands become explicit edge `INV` instances
- negated outputs become explicit output `INV` instances
- the emitted netlist is purely structural and does not use continuous assigns

The purpose of this mode is to preserve the AIG shape in a simple standard-cell
netlist. It is useful as a baseline, for inspection, and as a predictable input
to later netlist analyses such as STA.

This is not a general cell-covering algorithm. It does not search wider Liberty
cell sets, collapse Boolean cones, or optimize area or delay across alternative
decompositions.

## Cell-Selection Policies

The implemented structural baseline currently needs concrete `INV` and `NAND2`
cells. The `aig-tech-map` CLI exposes two policies for choosing among matching
cells in a timing-enabled Liberty proto:

- `small-normal-vt`: prefer VT class `0`, then the unsuffixed canonical cell
  name, then an `x1` spelling
- `max-speed`: prefer the highest available VT class, then the highest parsed
  drive strength

These policies do not change the mapped topology. They only choose which
concrete cell names appear in the output netlist.

If a library does not provide VT ordering metadata, all matching cells are
treated as unordered and selection still remains deterministic. If a candidate
set mixes ordered and unordered VT metadata, selection is rejected because the
relative ranking would be ambiguous.

## Current Cell Binding

The first implementation binds cell families by convention:

- inverter candidates have names beginning with `INV`, one input pin, and one
  timed output pin whose Boolean function is the negation of that input
- two-input NAND candidates have names beginning with `NAND2`, two input pins,
  and one timed output pin whose Boolean function is the negation of their
  conjunction

That is sufficient for the structural baseline, but it is deliberately narrower
than a general Liberty-function matcher.

## Output Contract

The mapper returns parsed netlist structures, not only text, so other library
flows can consume the result without reparsing emitted Verilog-like netlists.
Text emission is a separate step layered on top of the parsed representation.

The output is deterministic:

- generated instance names are stable
- generated net names are stable
- connections are emitted in sorted pin-name order

That keeps regression output reviewable and makes baseline comparisons useful.
