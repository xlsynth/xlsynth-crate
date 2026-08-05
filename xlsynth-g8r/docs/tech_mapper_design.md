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
    -> representative-pin-delay Liberty-cell matching
    -> one delay-first and three area-flow mapping rounds
    -> two exact-area recovery rounds
    -> parsed gate-level netlist
    -> optional timing-aware buffering and cell resizing
    -> exact full-netlist Liberty STA
```

It is not an ABC-loop protocol. The mapper does not serialize a selected cover
back into AIGER, unmap cells into logic, or resume ABC optimization.

### Input Contract

The preferred input is ABC binary AIGER carrying its q-extension structural
choices. The loader preserves otherwise-dead choice cones and records ABC's
backwards sibling links in `ChoiceAig`. Ordinary ASCII or binary AIGER is
also accepted and is represented as a no-choice graph.

Every choice class must be a single nonbranching sibling chain whose links
point toward earlier AIG nodes. Only the canonical chain head may feed an
ordinary AIG node or primary output; every noncanonical alternative is
reachable only through its choice links. Construction and q-AIGER import
reject branching chains, forward links, noncanonical primary outputs, and
alternatives with ordinary fanout.

The loader can close sibling links into complete deterministic choice classes
for diagnostics. Mapping itself deliberately keeps one state per concrete AIG
node and polarity, like ABC NF: a noncanonical sibling contributes
phase-adjusted cuts to the canonical choice, but cannot become an independently
referenced sibling root. Because ABC choices may be equivalent up to
complement, the mapper computes each AIG node's all-zero phase when it imports
sibling cuts.

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
it to two decimal places. Equal-area roots prefer the lower worst characterized
nominal pin delay, then deterministic cell identity; a faster root never
displaces a strictly smaller-area one. Structural cut flow remains independent
of Liberty area and timing. In NF mode the index also suppresses redundant
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
Cut flow uses structural leaf costs rather than Liberty areas or timing. On a
full frontier, up to three useful low-depth cuts are protected from
structural-area eviction while at least one ordinary minimum-flow cut is
retained. This small deterministic timing reserve exposes faster covers without
expanding the configured 16-cut search frontier.

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

The default `NfLiberty` implementation is a separate compact two-match engine in
`techmap::nf`. It visits retained cuts and native cell bindings directly,
keeps one fastest and one lowest-area-flow match per concrete object and
polarity, and scores delay and area child choices independently. The delay
candidate always follows its fastest child; the area candidate may follow a
cheaper child only when that child meets the current required time. Exact
Boolean truth tables identify interchangeable Liberty input pins, allowing
late signals to move to faster functionally equivalent pins during both flow
selection and exact-area recovery. Its explicit inverter closure follows NF's
direct-phase behavior and charges inverter area without dividing it by the
root's flow references.

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

For an unusually large unconstrained combinational cover, the mapper can
evaluate a bounded deterministic recovery portfolio before postprocessing. A
selected cover containing more than 4,096 actual Liberty cells and driving at
most 256 primary outputs admits up to seven alternate remappings that vary
equal-area root tie-breaking, functionally interchangeable pin assignment,
low-depth cut protection, and feasible-area child selection. An alternate
cover is considered only when it has at least 15% fewer selected cells. The
original and eligible unoptimized covers are emitted and independently timed
with full rise/fall Liberty STA under the configured external loads; an
alternative must also have strictly lower exact delay. To avoid replacing
moderately large covers with candidates whose initial improvement disappears
after buffering and resizing, an alternative must improve that unoptimized
delay by at least 10% unless the original cover exceeds 5,000 cells. The
fastest qualifying cover wins, with deterministic area and cell-count
tie-breaking. Smaller designs, wide-output designs, endpoint-constrained
combinational mapping, and registered transition mapping keep the original
single-cover path. The chosen cover still proceeds through the same requested
buffering, resizing, and final full-STA verification.

When distinct primary outputs share a selected implementation, deterministic
netlist emission may need an explicit unary identity driver. That driver is
selected from every eligible Liberty identity cell, not just the compact NF
root index. Full rise/fall NLDM interpolation at the actual configured
primary-output load chooses the fastest legal output buffer, with area and
stable cell identity breaking timing ties.

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

`netlist::timing_buffer` is the shared production inserter for both
combinational and registered mapped modules. It normalizes the finished
netlist into per-bit Liberty connectivity, classifies true noninverting
buffers by their Boolean functions, and accumulates real rise and fall sink
capacitances. An initial full Liberty STA ranks overloaded roots and fanout
sinks by downstream timing criticality. The inserter keeps critical sinks
near the original driver, selects actual buffer strengths using ABC-style
electrical effort and characterized slew/load timing, and checks bounded
batches of proposed trees against independently recomputed worst-path
timing. Explicit fanout and target-load bounds, Liberty maximum output
capacitances, and independent rise/fall sink loads remain electrical
constraints rather than scalar fanout estimates.

For registered modules, register Q pins are timing launches, register D pins
are capture endpoints, and clock pins are never buffered. User clock periods
and external output requirements remain hard constraints. Primary-input
buffering is optional, and packed port bits, public primary-output names,
and zero-area constant assignments retain their original spelling.

After an initial combinational buffer-and-resize pass, a bounded speculative
round can revisit actual post-sizing electrical conditions. On designs with
at most 4,096 instances and 256 previously inserted buffers, an unresolved
load or fanout above six permits a stricter timing-first buffer tree followed
by another complete incremental resizing pass. The second tree is built on a
cloned netlist; delay-oriented buffer-strength selection must still reduce its
parent driver's load. The complete trial is retained only when independent
full Liberty STA reports strictly better global delay. Otherwise every
instance, net, and interned name is restored unchanged.

A narrowly bounded variant also recognizes a weak combinational driver that
simultaneously drives a visible primary output and an internal data sink. For
designs with at most 128 instances, cheap normalized connectivity first checks
whether such a shared output exists. Only matching topologies receive exact
slew and critical-path analysis: the output transition must exceed half the
global worst-path delay, and its root must be electrically stressed and
near-critical. A speculative buffer can isolate the external output load while
leaving the critical internal consumer directly connected to its original
driver. Clocks, sequential designs, and unshared primary outputs are excluded;
the same cloned-netlist and strictly improving full-STA acceptance rules
apply.

Once the best timing is fixed, bounded sibling-buffer consolidation can
replace two inserted buffers driven by the same parent with one legal stronger
identity buffer. Candidate mergers preserve maximum fanout, configured
per-stage load, Liberty output capacitance, and the original parent load on
both edges. A move is accepted only when exact full-netlist timing does not
regress and total cell area strictly decreases. Public outputs, clocks, and
sequential modules are protected, and the search is capped at four accepted
mergers and 16 full timing evaluations.

The retired capacitance-balanced `netlist::buffer::insert_buffers` entry point
panics if called. Its algorithm is retained only for explicitly named test
characterization and cannot run in a production build.

### Cell Resizing and Area Recovery

`netlist::resize` exposes the combinational entry point to the shared
`netlist::timing_resize` engine. It groups combinational cells by exact
Boolean function, identical input-pin names, and identical output-pin names;
physical flip-flops additionally require identical clock, data/control
interface, state transition, and state-output polarity. Substitutions are
therefore structurally and logically safe without name- or suffix-based drive
strength assumptions.

The resizer builds a normalized per-bit timing graph once. Each rise and fall
transition retains the Liberty input pin, transition, slew, load, and setup arc
that actually determine its arrival. An initially one-percent near-critical
window traces the exact winning paths and all other characterized input arcs
with sufficient slack. It also considers otherwise noncritical gates and
registers whose input capacitance loads a critical driver. The window expands
adaptively when no improving move is available. Candidates are ranked first by
independently meaningful endpoint delay, and non-overlapping changes are
applied in deterministic batches rather than limiting each sizing iteration to
a single replacement.

Move sorting uses exact floating-point total order plus deterministic instance,
cell, and pin tie-breakers. Numerical improvement tolerances apply only when
accepting or rejecting a move; they never define equality inside a sort
comparator.

Cell substitution and input-pin exchange are moves in the same incremental
optimizer. The Liberty catalog derives legal combinational input swaps by
checking the complete cell truth table; a pin exchange is never inferred from
cell or pin naming. Each trial updates both rise and fall input capacitances,
propagates exact `gv-stats` Liberty timing through the affected upstream and
downstream cones, and restores all timing, predecessor, known-pin, setup, and
clock-load state when rejected. The best complete solution includes both cell
types and physical pin connections. Equivalent cell alternatives are ordered
using their timing at the actual observed slew and load; fair evaluation
scheduling considers zero-area pin exchanges and larger, equal-area, and
smaller drive variants.

Timing optimization and timing-protected area recovery alternate for at most
three rounds by default, stopping early at a local fixed point. Area recovery
accepts only functionally equivalent smaller cells that preserve every
achieved endpoint timing class. A final bounded, zero-area pin-swap cleanup
can expose one last timing-protected recovery opportunity.

`netlist::optimize::optimize_mapped_netlist` exposes the complete reusable
combinational buffer-then-resize pipeline, including strictly improving
speculative rebuffering and fixed-delay sibling-buffer recovery. It verifies
both initial and final results using independent full parsed-netlist area and
timing analysis. Both primary passes are disabled by default for NF mapping
and can be explicitly enabled or bounded individually.

The same engine handles mapped sequential modules without changing the
objective or substituting unit delays. It tracks primary-input and register
launches separately, recomputes actual Liberty clock-to-Q and setup arcs, and
preserves register-to-register, input-to-register, register-to-output, and
input-to-output timing independently. Candidate trials preserve clock-period
and individual primary-output constraints, and physical flip-flop substitutions
continue to track total clock-pin load.

Sequential optimization runs in the order physical-register restoration,
optional timing-aware buffer insertion, ABC-style smallest-adequate electrical
sizing, and bounded alternating gate/buffer/flip-flop sizing, combinational
pin swapping, and timing-preserving area recovery. Physical clock, data,
reset, and enable pins are never exchanged. Both the existing `buffer_options`
and `resize_options` remain independently optional; final area, register
counts, setup slack, and endpoint timing are independently recomputed using
the production `gv-stats` engine.

### Exact Timing Engine and Runtime

Mapping characterization, buffer evaluation, incremental resizing, and final
reporting share the same exact Liberty rise/fall timing implementation. NLDM
interpolation preserves arbitrary characterized axis values, conservative
nonmonotonic-surface repair, conditional timing arcs, transition clamping, and
register setup semantics. Hot-path interpolation coordinates and one- or
two-edge timing candidates use bounded stack-backed storage, avoiding
repeated heap allocations without changing the calculated timing values.

During incremental resizing, a thread-local cache scoped to the borrowed
Liberty library lazily computes each queried delay or transition table's
complete coordinatewise monotone upper envelope. Later interpolation corners
read the cached exact prefix maximum instead of repeatedly scanning every
earlier characterized entry. Cache ownership ends before its library borrow,
nested library scopes remain isolated, and setup tables retain their original
unmodified semantics. Cached and uncached timing, interpolation diagnostics,
and signed floating-point results remain bit-identical.

Incremental registered timing indexes only real physical register capture
instances when evaluating setup endpoints. Purely combinational instances do
not enter the capture endpoint loop; input-to-output, input-to-register,
register-to-register, and register-to-output objectives remain independently
tracked.

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

The supported sequential subset is one positive-edge clock and synchronous
flip-flops, with synchronous reset and enable already represented in the
transition logic. Physical-register restoration is followed by optional
register-aware Liberty buffering and optional resizing of combinational cells,
buffers, and eligible flip-flops. Clock nets are never buffered, physical
control pins are never exchanged, and all timing classes and setup constraints
are verified again after optimization. Latches, negative-edge cells,
asynchronous clear or preset, and unsupported explicit power-up state are
rejected rather than silently approximated.

## Structural Baseline

The older `netlist::techmap` path remains as a separate baseline. It maps
each AIG `And2` into `NAND2` followed by `INV`, materializes complemented
edges with inverters, and chooses concrete `INV` / `NAND2` variants by
name-oriented policy.

That path is useful for predictable structural lowering and regression
comparison, but it is not the implementation foundation for the final
choice-AIG mapper.
