# DSLX Block Language MVP Architecture

## Status and intent

This document describes the experimental Rust implementation of an explicit
RTL-level `block` construct. It is a prototype adjacent to DSLX rather than a
fork of the DSLX expression language. The frontend owns only block structure;
ordinary expressions and functions remain DSLX and are processed by existing
XLS/xlsynth facilities.

The target is ordinary XLS Block IR. A source port, register, or instance must
have a direct Block IR counterpart. The frontend does not add a second
scheduler or redefine Block IR behavior.

## Component boundaries

```text
block source
  |
  v
xlsynth-block-lang
  - block lexer/parser and source spans
  - symbol/order checks and block elaboration
  - DSLX expression adapter
  - Block IR construction
  - optional proc adapter
  |
  +--> xlsynth: DSLX parse/typecheck/constexpr/IR conversion
  +--> xlsynth-pir: Block IR model, parser, verifier, emitter
  +--> official codegen_main: proc -> child Block IR (only when needed)
  |
  v
verified package-form Block IR
  |
  +--> block_to_verilog_main --generator=combinational
  v
SystemVerilog
```

Reusable parsing, elaboration, and lowering live in a library crate.
`xlsynth-driver` owns command-line parsing, file I/O, external-tool selection,
and human-readable error reporting. The workspace crate graph remains acyclic.

## Source model

### Reused DSLX

The following are not reimplemented: identifiers, fixed-width and aggregate
types, literals, unary/binary operators, conditionals and matches, indexing and
slicing, casts, arrays and tuples, `let`, constants, imports, parametrics,
constexpr functions, pure functions, and ordinary function calls.

The block parser retains source text for expressions and starting byte offsets
for structural declarations. Existing DSLX conversion provides typechecking,
parametric validation, constexpr evaluation, and expression IR. Structural
diagnostics use the retained offsets. Diagnostics produced while typechecking a
generated helper are not yet remapped to the original expression range.
The structural shell tracks ordinary delimiter depth while finding parametric
angle groups, so grouped comparison and shift operators retain DSLX meaning.

### New structural AST

The block-specific AST contains:

- `BlockDecl`: name, visibility, parametric bindings, ordered ports, body.
- `PortDecl`: direction, name, and type; clock/reset are distinguished types.
- `LetDecl` and `ConstDecl`: source expression and optional type.
- `RegisterForwardDecl` and `RegisterContract`.
- `OutputAssign`.
- `InstanceDecl`: target symbol, parametrics, and named input bindings.
- `Assert` and `Cover`.
- Constexpr structural selection.

Structural declarations carry a starting byte offset. `let` and `const` nodes
retain both the complete statement and the exact parser-delimited RHS, so an
`=` inside a type cannot be mistaken for the binding operator. Full source
ranges are follow-up work. Ordered vectors are used whenever order is
observable. Maps used for lookup must not determine emitted order.

Before semantic lowering, the frontend elaborates structural conditions to
find the selected top's reachable block closure. Unreachable blocks are not
typechecked or lowered and cannot spuriously require proc tools. Parametric
block declarations are materialized only for concrete reachable instances.

## Static semantics

### Namespace and ordering

Parameters, ports, constants, lets, registers, and instances occupy one value
namespace. A definition becomes visible only after its declaration, except
that a `reg` name becomes visible at the start of its own contract. Shadowing
and duplicate names are errors.

Source block and instance names are preserved. Before lowering, the frontend
rejects names reserved by either the active XLS IR parser, such as `top`, or
the SystemVerilog backend, such as `module` and `wire`. A requested
`--module_name` follows the same rule instead of relying on backend mangling.
Compiler-owned helper, specialization, node, property, and instance names use
deterministic allocators seeded with authored symbols; users do not reserve an
`__xlsynth_` prefix to avoid capture.

A single declaration-analysis pass owns source ordering, shadowing, and output
read checks and records the complete symbol set. Helper construction consumes
that result rather than repeating namespace analysis.

Clock ports are structural and are not ordinary expression values. Reset ports
retain their polarity/synchrony type at the interface, but inside the block a
reset is intentionally readable as its physical one-bit port value. This
supports reset-dependent combinational nets without discarding the static
reset contract. Compiler-generated reset-active values normalize polarity for
register and property lowering.

### Drivers and graph validity

Inputs cannot be assigned. Outputs cannot be read and must each have one
`assign`. Register contracts are unique. The combined combinational graph,
including instance edges, must be acyclic. Every instance input is connected
once; unconsumed outputs are legal.

### Registers

A register contract has required `next` and optional `en` and `init_value`
fields. Fields may appear in any order but not more than once. Omitted `en` is
the DSLX boolean constant `true`. Omitted `init_value` and `init_value: none`
are equivalent. `init_value` names the value loaded by synchronous reset; it
does not describe a SystemVerilog `initial` value or power-on state. When a
`reg` repeats the type from an earlier `declreg`, a synthesized binding asks the
ordinary DSLX typechecker to verify type identity. This accepts aliases and
equivalent aggregate spellings without maintaining a second type system in the
structural frontend.

For a register with an `init_value`, the frontend builds a reset-select data
expression and reset-inclusive load enable. It intentionally does not encode a
dynamic expression as Block IR's static register reset value. A non-resettable
register has no reset path and continues its normal enable/next behavior while
the block reset is active.

All writes are concurrent: generated helper functions take current register
reads as parameters and calculate every contract from the same state snapshot.

## Expression bridge and lowering

For each elaborated block, the frontend constructs one or more private DSLX
helper functions. Inputs include ordinary block inputs, current register values,
and child-instance outputs. Returns include named lets that must remain visible,
output drivers, register `init_value`/enable/next expressions, instance input
values, and assertion/cover predicates.

Block-local `const` values are substituted through later items in declaration
order before helper construction. This makes them usable in types and child
parametrics without making a later constant visible early. Compiler-generated
helper parameters use a fresh-name allocator seeded with every authored symbol.
Substituted expressions retain grouping, including complete braced DSLX const
arguments. Declaration-order analysis, local-constant substitution, and
instance-output rewriting share lexical `for` and `match` pattern scopes, so a
nested binder cannot capture a block parameter, local constant, or instance.
Structural `if` conditions are checked against authored declaration order
before constexpr substitution, preserving the exact offending-token offset.
Clock/reset type recognition also uses lexer tokens, treating comments and
whitespace identically.

The helpers are converted to function IR. Nodes are cloned topologically into
the block, replacing helper parameters with Block IR input ports, register
reads, or instantiation outputs. Block-specific sinks are then emitted:

| Source | Block IR |
|---|---|
| ordinary input | `input_port` |
| clock | block clock metadata/port |
| reset | reset input and reset metadata |
| `assign` | `output_port` |
| `reg` read | `register_read` |
| register contract | `register_write` |
| block/proc child | block instantiation |
| instance binding | `instantiation_input` |
| instance output use | `instantiation_output` |
| assertion/cover | Block IR assert/cover node |

No scheduling pass is used for an authored block. Function calls therefore
remain combinational. In every mode each synthesized helper is optimized while
it is still ordinary function IR, before its nodes are cloned into Block IR.
XLS 0.53 cannot emit an `invoke` from Block IR, so this step always inlines
ordinary calls. `free` also permits combinational DCE. The preserving modes
first append authored `let` values to the helper return tuple, then add an
identity anchor when an alias would otherwise have no IR node. This retains
used and unused source names without exposing helper returns as block ports.
Doing this before Block IR assertions and covers are attached also avoids the
pinned optimizer's inability to run its non-synthesizable-separation pass on a
block containing properties.

Direct Verilog FFI invokes are outlined before optimization: each result becomes
an opaque helper parameter, while each operand is added to the helper return
tuple. After ordinary calls are inlined, those opaque parameters and operand
results become extern instantiation outputs and inputs. This prevents XLS from
replacing an FFI result with the DSLX fallback body. FFI calls reached through
another function are rejected in the MVP rather than approximated.
FFI calls are also rejected in structural and parametric constexpr evaluation:
the DSLX fallback body is not the external hardware implementation and cannot
select elaborated structure.

In `free` mode, the frontend runs a second libxls package optimization only
when the package has neither runtime properties nor extern instantiations. XLS
0.53 cannot optimize either form after lowering. Their synthesized helpers have
already been optimized and ordinary calls inlined.
`preserve-names-and-functions` currently behaves like
`preserve-names` plus an explicit warning: function-to-combinational-child
materialization remains follow-up work, while calls are inlined for codegen
correctness.

## Port and package representation

`xlsynth-pir::BlockMetadata` gains an ordered vector describing every port and
its role. Existing name-to-node maps remain for lookup. `Instantiation` gains
an explicit block-versus-extern kind. These are intentional, versioned Rust
data-model changes. They preserve IR behavior but require downstream Rust
struct-literal users to add the new fields; constructors and `Default` support
the migrated forms.

Legacy PIR parsing still leaves the ordered vector empty, so existing
`Package::to_string()` output keeps its canonical clock/input/output order.
Block-language callers explicitly opt into order-preserving parsing. The PIR
verifier checks populated vectors against parameter IDs, types, reset, return
shape, compatibility maps, and package-wide output-port ID uniqueness. The
legacy parser retains its clock-first rule; the ordered parser permits mixed
order, and both reject duplicate clocks. XLS 0.53 codegen canonicalizes
clock/reset ports
first, so `dslx-block2sv` restores every generated block header to its
authoritative order. A requested `--module_name` is applied to a codegen-only
package clone before emission; the rename updates the selected top and every
retained block-instantiation target instead of patching generated text.

Structural `if` items are elaborated before namespace checking. Conditions may
use ordinary DSLX constexpr functions, resolved parametrics, and prior
block-local constants. The frontend asks the official DSLX converter and
interpreter to typecheck and evaluate them instead of maintaining a second
constexpr evaluator.
Direct parametric child-block instances are materialized under deterministic
`__xlsynth_spec_...` names and identical specializations are reused.

Blocks referenced by instances must precede their users in emitted package
order. Package merge and renaming use stable traversal and `BTreeMap` or
explicit sorting where observable.

## Proc adapter

A target resolved as a DSLX proc is not directly represented as a proc inside
the parent block. The adapter:

1. Invokes official `ir_converter_main` for the selected proc, writing its full
   IR output directly to a temporary artifact and retaining its reachable
   proc-network closure.
1. Runs official `codegen_main` with unit delay and a deterministic one-stage
   configuration, requesting Block IR output.
1. Parses and verifies the generated package.
1. Uses the converter's declared `top proc`, then finds the generated top block
   and its dependency closure.
1. Prefixes every imported closure member with a deterministic
   `__xlsynth_proc_<module>_<proc>_<reset-polarity>__...` identity.
1. Rewrites internal block- and extern-instantiation references consistently.
1. Imports dependencies before users and instantiates the renamed top block.
1. Uses generated non-clock/reset port names for explicit source bindings. The
   child shares the parent's structural clock. A collision-free generated reset
   port is connected to the parent reset after a polarity check.

Proc lowering is specialized by proc identity and reset polarity. It requires a
configured external toolchain. The MVP does not
flatten proc state, reinterpret channels, or promise a configurable schedule.
A scheduling failure is a compilation error with captured tool diagnostics.
External conversion/codegen has a configurable wall-time limit. Converter IR
artifacts are lossless up to a separate artifact-size limit; captured streams
are also bounded. Unix tools run in fresh process groups; timeout, artifact
overflow, and leader exit terminate remaining descendants so inherited pipes or
artifact writers cannot outlive supervision. Proc codegen artifacts are watched
while the tool runs, not only checked after exit.

## Assertions, coverage, and unsupported constructs

Assertions and covers are current-cycle predicates and are automatically gated
inactive during reset: assertions use `reset_active || predicate`, while covers
use `!reset_active && predicate`. The authored string remains the assertion
message. A separate, sanitized, occurrence-indexed
`__xlsynth_assert_<n>_...`/`__xlsynth_cover_<n>_...` label is used as the IR/HDL
identifier, so spaces, punctuation, duplicates, ports, and signals cannot
collide. This label problem was introduced by the prototype frontend's initial
conflation of message strings with IR identifiers; it is not an existing DSLX
restriction. The structural parser matches the pinned XLS 0.53 DSLX scanner's
string-token and escape acceptance for UTF-8-representable messages and rejects
unknown or malformed escapes before lowering. Byte escapes `\x80` through
`\xff` are explicitly rejected: XLS treats them as raw bytes, while PIR
assertion messages are Rust UTF-8 strings, so accepting them would silently
change the bytes. Plain Verilog output, temporal assertions, final assertions,
and four-state checks are not part of the MVP.

Combinational DSLX `#[extern_verilog(...)]` functions are supported. A reachable
FFI invoke becomes a Block IR `kind=extern` instantiation with explicit
`instantiation_input` and `instantiation_output` nodes, while the `ffi_proto`
function member remains in the package. `xlsynth-pir` parses, verifies, emits,
and reparses this representation.

XLS 0.53 has an asymmetry: `codegen_main` emits `foreign_function=` Block IR,
but `block_to_verilog_main` rejects that spelling when reading it back. The SV
driver therefore clones the verified package, binds each FFI operand to a
compiler-owned SV-safe identity anchor, replaces extern outputs with named
codegen placeholders, and invokes the official combinational generator. It then
replaces each placeholder assignment with the authored FFI template. The
authoritative package is never rewritten. Ports, hierarchy, registers, and
every other generated statement remain official Block IR codegen output.
Only blocks reachable from the selected top receive patches. Module and
assignment spans are located lexically in the original generated text before
replacements are applied, so strings, comments, and earlier templates cannot
redirect later patches. Template decoding preserves UTF-8 and rejects unknown
escapes.
Supported templates use ordinary `{fn}`, `{return}`, and exact function
parameter placeholders. Sequential `extern block` declarations remain
deferred.

Also deferred: asynchronous or multiple resets, negative-edge or multiple
clocks, generated arrays of instances, inferred memories, latches, tri-states,
arbitrary proc scheduling controls and parametric proc instances.

## Public library and CLI

The library exposes compile options, parametric bindings, combinational
optimization mode, warning controls, external-tool time, captured-output, and
artifact limits, structured diagnostics, and a result containing the parsed
PIR package and deterministic IR text.

`dslx-block2ir` emits verified Block IR. It requires an external toolchain only
if a reachable proc instance is present. `dslx-block2sv` additionally invokes
official `block_to_verilog_main` with `--generator=combinational`. Both commands
use the existing DSLX search path, stdlib, warning, and toolchain conventions.
`dslx-block2sv` accepts only module naming, array-bounds checking, and cosmetic
line-separation codegen controls. It forces SystemVerilog and does not expose
pipeline, reset, boundary-flop, or pre/postprocessing sidecar controls.

`dslx-block2sv` uses `--generator=combinational` because the authored block
already contains its register boundaries. In XLS 0.53, this path emits existing
state and hierarchy directly. The pipeline generator first tries to reconstruct
feed-forward stages and rejects register feedback. The selected official path
retains register reads, writes, reset/load-enable logic, hierarchy, assertions,
and covers; the frontend does not reinterpret registers.

## Validation architecture

- Parser and semantic tests cover valid syntax and focused negative cases.
- Golden tests cover deterministic Block IR and ordered ports.
- Roundtrip tests parse, verify, emit, and reparse generated packages.
- Proc tests compile a simple stateful streaming proc and a spawned child-proc
  closure, verify deterministic renaming, and run official Block IR codegen.
- Codegen smoke tests run official Block IR-to-SystemVerilog conversion.
- The composition integration test combines a function-calling custom block,
  a proc-wrapping block, a top block instantiating both, and one combinational
  Verilog FFI call. With the XLS tools, Icarus, and VVP available, it checks an
  exact Block IR parse/emit roundtrip, official codegen, compilation, mixed
  child-port order, top-module renaming, and cycle-level simulation. The main
  Ubuntu CI lane sets `XLSYNTH_REQUIRE_BLOCK_E2E=1`, so missing tools fail that
  lane instead of silently skipping the proof.
- Pinned-tool tests lint active-high and active-low reset-masked assertions and
  covers with Slang. Slang is mandatory when the E2E lane is required; Icarus
  with assertions disabled is only a local fallback. The test also inspects the
  Block IR property operands for both reset polarities. A separate simulation
  proves active-low proc reset and state holding under ready/valid backpressure.
- Bedrock proof scripts compile and roundtrip representative parameterizations
  and lint official SystemVerilog output without writing into `bedrock-rtl`.
  The independent Verilator campaign passes all 25 configurations across three
  deterministic 500-cycle seeds, and EQY proves temporal induction for the
  same matrix within its depth-12 SAT strategy. Unsupported SECDED/FIFO
  parameterizations are rejected at elaboration rather than compiled into an
  unimplemented fallback structure.
