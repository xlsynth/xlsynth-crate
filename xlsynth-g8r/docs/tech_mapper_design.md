# Technology Mapper Design

This note describes the two technology-mapping paths in `xlsynth-g8r`.
They intentionally have separate implementations and purposes.

## Final Choice-AIG Mapper

The clean-sheet final mapper lives in the top-level `techmap` module. It is
the path intended for final mapping after ABC has finished Boolean
optimization:

```text
final ABC q-AIGER + Liberty proto + endpoint timing constraints
    -> choice-aware cut matching
    -> selected Liberty-cell cover
    -> parsed gate-level netlist
```

It is not an ABC-loop protocol. The mapper does not serialize a selected cover
back into AIGER, unmap cells into logic, or resume ABC optimization.

### Input Contract

The preferred input is ABC binary AIGER carrying its q-extension structural
choices. The loader preserves otherwise-dead choice cones and records ABC's
backwards sibling links in `ChoiceAig`. Ordinary ASCII or binary AIGER is
also accepted and is represented as a no-choice graph.

The mapper closes sibling links into complete deterministic choice classes.
Because ABC choices may be equivalent up to complement, it computes each AIG
node's all-zero phase and normalizes members to a canonical class polarity.

### Liberty Index

The mapper indexes cells by Boolean function rather than by family names. An
eligible first-pass cell must:

- be combinational and not marked `dont_use`
- have one output pin with a parseable Liberty Boolean function
- have no clocking input pins
- have at most the configured cut size in function inputs
- use every declared input pin in the function

For every eligible output, the mapper evaluates a compact truth table. Like
ABC NF's root-library preparation, it retains one minimum-area root cell per
native Boolean function before indexing deterministic input-pin permutations
plus per-input polarity transforms. In NF mode it also suppresses redundant
pin permutations with the same transformed truth and leaf-polarity mask, like
ABC's default `fPinPerm=0` matching database. Drive-strength selection is
intentionally left to a later sizing pass. The first implementation skips
multi-output, sequential, clock-gating, and partially used input cells.

### Cut Matching

Each AIG node gets bounded priority cuts, with `k=6` and 16 retained cuts per
node by default. Cut truth tables include complemented AIG edges and minimize
away unused support variables after composition. Sibling cuts are
phase-adjusted and propagated through parents during enumeration, so a choice
alternative can create mapping opportunities above the choice node, as in ABC
NF.

### Cover Selection

The selector follows ABC NF's shape:

- one delay-first pass establishes the achievable endpoint target
- three area-flow passes propagate required times backward and blend global
  mapping reference counts
- two exact-area passes dereference and rereference trial cones so shared logic
  is charged only when it becomes newly live

For an unconstrained run, the inner search uses one unit of delay per cell
input, matching the generated-genlib objective used by ABC's normal `&nf`
flow. Required-time propagation, area flow, and exact-area recovery stay in
that unit-delay domain. If the controller supplies any endpoint timing
constraint, the search instead uses compact scalar Liberty arc delays and
re-evaluates each selected live cover with the same rise/fall, slew,
capacitive-load, conditional-arc, and timing-table semantics used by
`gv-stats`, so the explicit constraints remain meaningful.

In either mode, the finished selected cover is re-evaluated with the shared
`gv-stats` timing semantics. The final reported delay is recomputed again from
the emitted netlist with the same STA path when all selected cells have
complete timing.

The outside controller may supply flattened primary-input arrival times and
primary-output required times. Without an explicit required time, the first
delay pass establishes a global target. If the compact NF root library cannot
meet an explicit endpoint requirement, mapping reports that failure rather
than silently changing the target. Buffer-tree insertion and drive-strength
sizing remain separate later refinements.

### Output Contract

The mapper returns parsed netlist structures plus statistics, not only text.
Text emission is layered on top of the parsed representation. Emission is
deterministic:

- generated instance and net names are stable
- cell connections are sorted by pin name
- output ports are driven structurally through selected cells, buffers, or
  paired inverters

## Structural Baseline

The older `netlist::techmap` path remains as a separate baseline. It maps
each AIG `And2` into `NAND2` followed by `INV`, materializes complemented
edges with inverters, and chooses concrete `INV` / `NAND2` variants by
name-oriented policy.

That path is useful for predictable structural lowering and regression
comparison, but it is not the implementation foundation for the final
choice-AIG mapper.
