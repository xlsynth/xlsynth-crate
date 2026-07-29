# Technology Mapper Design

This note describes the two technology-mapping paths in `xlsynth-g8r`.
They intentionally have separate implementations and purposes.

## Final Choice-AIG Mapper

The clean-sheet final mapper lives in the top-level `techmap` module. It is
the path intended for final mapping after ABC has finished Boolean
optimization:

```text
final ABC q-AIGER + Liberty proto + endpoint timing constraints
    -> NF-priority choice-aware structural cuts
    -> unit-delay or representative-pin-delay Liberty-cell matching
    -> four area-flow mapping rounds
    -> two exact-area recovery rounds
    -> parsed gate-level netlist and exact Liberty STA
```

It is not an ABC-loop protocol. The mapper does not serialize a selected cover
back into AIGER, unmap cells into logic, or resume ABC optimization.

### Input Contract

The preferred input is ABC binary AIGER carrying its q-extension structural
choices. The loader preserves otherwise-dead choice cones and records ABC's
backwards sibling links in `ChoiceAig`. Ordinary ASCII or binary AIGER is
also accepted and is represented as a no-choice graph.

The loader can close sibling links into complete deterministic choice classes
for diagnostics. Mapping itself deliberately keeps one state per concrete AIG
node and polarity, like ABC NF: a sibling contributes phase-adjusted cuts to
the later node, but referenced sibling roots are not merged into one shared
mapping state. Because ABC choices may be equivalent up to complement, the
mapper computes each AIG node's all-zero phase when it imports sibling cuts.

### Liberty Index

The mapper indexes cells by Boolean function rather than by family names. An
eligible first-pass cell must:

- be combinational and not marked `dont_use`
- have one output pin with a parseable Liberty Boolean function
- have no clocking input pins
- have at most the configured cut size in function inputs
- use every declared input pin in the function

For every eligible output, the mapper evaluates a compact truth table. Like
ABC NF's root-library preparation, it selects one minimum-area root cell per
native Boolean function before expanding deterministic input-pin permutations
and per-input polarity transforms. This avoids expanding and then discarding
every drive-strength variant. Unlike ABC's intermediate GENLIB conversion,
the root index retains the full-precision Liberty area rather than rounding
it to two decimal places. With ABC's generated unit-delay arcs, equally
priced roots are selected by deterministic cell identity; characterized
Liberty arc delays do not influence root or cut selection. In NF mode the
index also suppresses redundant
pin permutations with the same transformed truth and leaf-polarity mask, like
ABC's default `fPinPerm=0` matching database. Drive-strength selection is
intentionally left to a later sizing pass. The first implementation skips
multi-output, sequential, clock-gating, and partially used input cells.

### Cut Matching

Each AIG node gets bounded priority cuts, with `k=6` and 16 retained cuts per
node by default. Structural cut priority follows `giaNf.c`: useful function,
structural area flow within the NF epsilon, unit-delay depth, and leaf count.
The frontier applies ABC's early support-containment check, removes strict
supersets on insertion, and reserves a unit-cut slot when preparing fanins.
Cut flow uses structural leaf costs rather than Liberty areas or timing.

Cut truth tables include complemented AIG edges and minimize away unused
support variables after composition. Sibling cuts are phase-adjusted and
propagated through parents during enumeration, so a choice alternative can
create mapping opportunities above the choice node, as in ABC NF. Initial
flow references follow ABC's structural fanout counting, including its
multiplexer/XOR fanout discount. Later rounds blend in selected-cover
reference counts.

### Cover Selection

The selector follows ABC NF's shape:

- one delay-first pass establishes the achievable endpoint target
- three area-flow passes propagate required times backward and blend global
  mapping reference counts
- two exact-area passes dereference and rereference trial cones so shared logic
  is charged only when it becomes newly live

After exact-area recovery, an NF-like primary-output driver cleanup reorients
multiply referenced plain-inverter closures when possible: the direct root
implementation drives the shared internal phase, and the inverter stays on the
output-only phase. This avoids making an output polarity choice add a heavily
loaded inverter to an otherwise shared cone.

The default `NfLiberty` implementation is a separate single-objective engine in
`techmap::nf`. It visits retained cuts and native cell bindings directly,
keeps one fastest and one lowest-area-flow match per concrete object and
polarity, and selects an area child only when it meets the current required
time. Its explicit inverter closure follows NF's direct-phase behavior and
charges inverter area without dividing it by the root's flow references.

It uses the same 16-cut default frontier and the same flow and exact-area
rounds as ABC NF. Each native cell input receives a fixed representative
Liberty delay. Each rise/fall arc is
interpolated using the same NLDM evaluator as final `gv-stats` at the
configured primary-input slew and an output load equal to twice the median
indexed input-pin capacitance. The larger of the interpolated rise and fall
delays becomes the input-pin delay. Both forward arrivals and backward
required times, including explicit inverter closures and exact-area
recovery, use that same pin-specific delay. Incomplete synthetic Liberty
libraries retain their scalar delay estimates as a fallback.

Selecting `NfUnit` through a public mapper entry point panics. The unit-delay
objective remains available only for private characterization inside the
explicit experimental `Balanced` portfolio.

The exact-area rounds recursively dereference the old cone and reference each
trial cone, charging a shared cell only when its reference count changes
between zero and one. Required times and output phase cleanup are propagated
through the actual selected mapping.

The earlier `BufferedLiberty` and two-cover `Balanced` strategies remain
available as explicitly selected experimental modes. They are not evaluated
or selected by the default NF mapper. Native buffer insertion and resizing
are likewise independent, opt-in netlist passes rather than part of the
default mapping objective.

If the controller supplies an endpoint timing constraint, mapping instead runs
one scalar-Liberty search and re-evaluates the selected live cover with the
same rise/fall, slew, capacitive-load, conditional-arc, and timing-table
semantics used by `gv-stats`, so the explicit constraints remain meaningful.

In either mode, the finished selected cover is re-evaluated with the shared
`gv-stats` timing semantics. The final reported delay is recomputed again from
the emitted netlist with the same STA path when all selected cells have
complete timing.

The outside controller may supply flattened primary-input arrival times and
primary-output required times. Without an explicit required time, the first
delay pass establishes a global target. If the compact NF root library cannot
meet an explicit endpoint requirement, mapping reports that failure rather
than silently changing the target.

### Buffer Insertion

`netlist::buffer` classifies true noninverting buffers from Liberty Boolean
functions rather than relying on cell names. It counts each real scalar input
pin, accumulates separate rise and fall sink capacitance, and partitions
overloaded nets into deterministic, balanced buffer trees. Selection favors
the smallest buffer that retains characterized output-capacitance headroom,
avoiding the steep-delay edge of an NLDM table; it falls back to the smallest
legal buffer when that headroom is unavailable. The subsequent sizing pass
can select a stronger equivalent variant.

Primary-input buffering is optional, and clock and inout nets are protected.
When a high-fanout net is also a primary output, insertion moves its original
driver to a fresh internal net so the public output name remains unchanged.
Unachievable per-stage load constraints are reported without repeatedly
inserting non-improving tree levels.

### Cell Resizing and Area Recovery

`netlist::resize` groups combinational cells by exact Boolean function,
identical input-pin names, and identical output-pin names. A substitution is
therefore structurally and logically safe without name- or suffix-based drive
strength assumptions.

The resizer builds the scalar timing graph once. Each candidate updates both
the replacement cell and the rise/fall input capacitances seen by its upstream
drivers, propagates exact `gv-stats` Liberty timing through the affected
downstream cone, and rolls back every rejected trial. It first accepts
bounded, beneficial substitutions on the worst output paths. After buffer
insertion and upsizing have established the achievable delay, it downsizes
noncritical cells only when a trial preserves that delay.

`netlist::optimize::optimize_mapped_netlist` exposes the complete reusable
buffer-then-resize pipeline. It verifies both initial and final results using
independent full parsed-netlist area and timing analysis. Both passes are
disabled by default for NF mapping and can be explicitly enabled or bounded
individually.

### Output Contract

The mapper returns parsed netlist structures plus statistics, not only text.
Text emission is layered on top of the parsed representation. Emission is
deterministic:

- generated instance and net names are stable
- cell connections are sorted by pin name
- constant output bits are emitted as zero-area scalar Verilog assignments
- output ports are driven structurally through selected cells, buffers, or
  paired inverters when they are not constant

### Sequential Transition Mapping

A synchronous design is represented by `SequentialGateFn`: an ordinary
combinational transition `GateFn`, one clock, external interface bindings,
and register bindings. Each register's current `Q` value is a transition
input and its effective next-state `D` value is a transition output.
Synchronous reset and load-enable behavior are already included in the
next-state logic.

ABC optimizes only that combinational transition graph. The controller keeps
the original `.g8r` or `.g8rbin` sequential metadata and exports the
transition using `ir2g8r --transition-aiger-out`. ABC's final binary AIGER
retains structural choices as usual. No sequential AIGER extension, latch
encoding, or retiming is required.

Register cleanup can be inserted after ABC has optimized an ordinary
transition but before its final choice-generation pass. The reusable
`cleanup_sequential_transition` API first validates and repacks the optimized
scalar AIGER boundary against the original native transition. It then removes
unobservable register bits, propagates initialization-compatible constant
state, and merges equal or complemented next-state bits to a fixed point.
Both the transition AIG and native register bindings are rebuilt together.
`PreserveCycleZero` retains explicit initialization and the simulator's
all-zero startup; `UninitializedDontCare` additionally optimizes state bits
without an explicitly specified initial value. The cleaned ordinary
transition can then return to ABC for final choice generation, avoiding any
need to modify or discard structural-choice sibling links.

When register removal exposes constant external bus bits, sequential boundary
restoration preserves them as zero-area continuous assignments to those packed
bits. It does not insert a Liberty buffer or inverter for each constant output.

`map_sequential_choice_aig_to_netlist` validates that the optimized graph
preserves every scalar transition input and output. It then maps the
transition logic using the same choice-aware Liberty mapper and reconnects
each state boundary to a real Liberty flip-flop. Register `Q` and `D` remain
internal wires; the final module exposes only the original external
interface and clock.

Flip-flop eligibility follows Liberty `ff` metadata, clock expressions,
state-variable output functions, next-state functions, `dont_use`, and
deterministic cell area ordering. Complemented-output flip-flops are
eligible when the corresponding next-state polarity can preserve the
logical register. Cell-name suffixes and assumed drive-strength conventions
do not determine eligibility.

Register clock-to-output launch arrivals and setup-constrained next-state
endpoints seed the NF mapper. Final timing and area are recomputed from the
complete sequential netlist using the same register-boundary STA as
`gv-stats`, with controller-provided primary-input arrivals applied to their
corresponding scalar port bits. The final physical cover must satisfy both
the clock period and each primary-output required time. Statistics distinguish
flip-flop area, primary-input capture, register-to-register timing,
register-to-output timing, and optional clock slack.

The initial supported subset is one positive-edge clock and synchronous
flip-flops. Latches, negative-edge cells, asynchronous clear or preset,
explicit power-up state, and mapped-netlist buffering or resizing are
rejected rather than silently approximated. Register-aware buffering and
sizing are deliberately separate follow-up work.

## Structural Baseline

The older `netlist::techmap` path remains as a separate baseline. It maps
each AIG `And2` into `NAND2` followed by `INV`, materializes complemented
edges with inverters, and chooses concrete `INV` / `NAND2` variants by
name-oriented policy.

That path is useful for predictable structural lowering and regression
comparison, but it is not the implementation foundation for the final
choice-AIG mapper.
