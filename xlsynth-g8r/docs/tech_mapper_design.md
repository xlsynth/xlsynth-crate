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

For every eligible output, the mapper evaluates a compact truth table and
indexes deterministic input-pin permutations plus per-input polarity
transforms. The first implementation intentionally skips multi-output,
sequential, clock-gating, and partially used input cells.

### Cut Matching

Each AIG node gets bounded k-feasible cuts, with `k=6` by default because a
cut truth table fits in a `u64`. Cut truth tables include complemented AIG
edges. For a choice class, the matcher considers cuts from every class member
and adjusts each member's truth table into canonical polarity before Liberty
lookup.

### Cover Selection

The selector keeps a bounded non-dominated area/delay frontier per canonical
choice state and polarity. Cell area comes from Liberty. The current delay
estimate is conservative and scalar: for each input pin, it uses the maximum
characterized `cell_rise` / `cell_fall` table value on matching
combinational arcs.

The outside controller may supply flattened primary-input arrival times and
primary-output required times. If a required time is present, cells with
incomplete input timing are excluded and the mapper chooses the least-area
retained solution meeting the required time. Without required times, it
chooses least area with deterministic timing and identity tie-breaks.

The scalar delay model is deliberately an initial selection model. Full
load/slew-aware STA, exact shared-area recovery, buffering, and sizing are
later refinements that do not require changing the final-only handoff.

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
