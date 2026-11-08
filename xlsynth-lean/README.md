# xlsynth-lean

`xlsynth-lean` exports pure XLS IR functions, combinational `GateFn` graphs and
stateful XLS blocks to executable Lean. The function and gate exporters can
evaluate batches and return typed results to Rust. The gate exporter also
checks user-supplied equivalence proofs. The block exporter accepts block IR or
BX source. With BX source it also emits a typed interface and correspondence
proofs.

The companion Lean library is tested with Lean 4.34.1. Building the Rust crate
does not require Lean to be installed.

## Export a GateFn

```rust
use std::path::Path;
use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;
use xlsynth_lean::{
    LeanExportOptions, LeanModuleName, export_gate_fn, write_lean_support,
};

let graph = load_gate_fn_from_path(Path::new("design.g8rbin"))?;
let options = LeanExportOptions {
    module_name: LeanModuleName::new("Generated.Design")?,
};
let bundle = export_gate_fn(&graph, &options)?;
write_lean_support(Path::new("lean-output"))?;
let written = bundle.write_to(Path::new("lean-output"))?;
```

`export_gate_fn` is an in-memory operation. `LeanCircuitBundle::write_to` writes
the generated circuit source and an ASCII AIGER (`.aag`) sidecar. The separate
`write_lean_support` operation writes the companion Lean sources and a
`lean-toolchain` version pin into a caller-selected directory.

Module names use dot-separated ASCII identifiers. The first component cannot be
`Init`, `Lean`, `Std`, `Lake`, `XlsynthLean`, or `EvalMain` (case-insensitively),
because those names are reserved for the Lean toolchain and companion support
modules.

The generated module embeds the ASCII AIGER sidecar with `include_str` and
exposes a fixed-width circuit and function:

```lean
import XlsynthLean.Aag

def circuit : XlsynthLean.Aig.Circuit 3 1 :=
  (XlsynthLean.Aag.parse 3 1
    (include_str "Design.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec 3) : BitVec 1 :=
  circuit.eval inputs
```

The native file loader requires a clockless, register-free design for this
combinational API. Use the block IR API below for stateful models.

## Export XLS block IR

```rust
let block = xlsynth_lean::block::select_block(&package, None)?;
let name = xlsynth_lean::LeanModuleName::new("Generated.Counter")?;
let path = xlsynth_lean::block::write_to(&package, block, &name, project_root)?;
```

`write_to` takes a verified `xlsynth_pir::ir::Package` and a block borrowed
from it. If you already have the block, pass it directly. Otherwise, `select_block`
accepts a block name, or selects the package top when given `None`. A package
without a top must contain exactly one block for automatic selection.
This example writes `Generated/Counter.lean` with declarations such as
`Generated.Counter.eval`. Generated blocks and functions each write only
their model file. Build [XlsynthLean/IR.lean](lean/XlsynthLean/IR.lean) once
as part of your Lean project and put its build output on Lean's module search
path. Models and proofs can then use `import XlsynthLean.IR`. A proof can
use any definition from that module, including helpers absent from its model.
For callers that manage their own files, `block::emit(&package, block)`
returns source in the `XlsBlock` namespace with the same import.

The generated clock step preserves old-state outputs, simultaneous register
writes, reset precedence and source assertions. Blocks use the function
emitter's translation of PIR types, literals and local operations, including
one-hot selection over tuples and arrays. Calls to pure functions and fixed-trip-count loops remain explicit. Unsupported operations and asynchronous resets return contextual errors.

When starting from saved IR text, call PIR's `Parser::parse_and_validate_package`
before selecting the block, as in the `export_block` example.

Generated block files use `L.` for Lean library declarations. For example,
`L.Vector (L.BitVec 8) 2` is an array of two eight-bit words. Each file
defines its own short spellings. A proof importing the model can use Lean's
usual library names, and source fields such as `p.Vector` keep their spelling.

The generated `eval` takes arbitrary old registers and current inputs and
returns outputs, next registers, and assertion results. Outputs read old
registers; all register writes commit together. For example, if `a` is written
from `b` and `b` from `a`, an old state `(a = 1, b = 2)` produces next state
`(a = 2, b = 1)`. An output driven by `a` still observes `1` on that edge.
`reset` activates the declared reset input for one ordinary clock edge. It
does not invent initial values for registers without reset values.

Every assertion remains an observable result, including disconnected ones;
assertions never constrain the model's input or old-state domain.

### Compile a BX block

BX declares each register, its reset value, the data inputs and the output
ports. For example, this block exposes its old count on output and increments
it on each clock unless reset is asserted.

```
pub block counter(clk, take: bool) -> u8 {
    reg count: u8 = 0;
    if take { count := count + u8:1; }
    count
}
```

```rust
use xlsynth_lean::block::compile_bx;

let compiled = compile_bx("counter", &source, xlsynth_bx::empty_package("counter"),
                          Default::default(), &Default::default())?;
let bundle = compiled.emit_readable(None)?;
```

The readable interface preserves unsigned bits, Boolean fields, positional
tuples, fixed-size arrays and named BX structs. A struct result exposes its fields as output
ports in BX declaration order. One output becomes the value of `observe q x`;
several outputs become a Lean product in port order. The register and port
types come from checked declarations, and every equation comes from the linked
block. Internal bindings can be renamed without changing these public fields.

Public source declarations live under `XLS.<full logical module>`. For
example, importing `left.bus` and `right.bus` gives the proof author distinct
types `XLS.left.bus.Packet` and `XLS.right.bus.Packet`. Readable DSLX functions
use the same rule, including concrete specializations such as
`XLS.left.bus.«choose<5>»`. Compiler-generated loop bodies live under
`XlsReadable.Helpers`.

Supply the BX module name as the first argument to `compile_bx`. For block
`controller` in `common.bx`, `XLS.common.controller q x` evaluates the block
on state `q` and input `x` and returns the complete `XlsReadable.Evaluation`.
A DSLX module `common.x` can contribute other declarations under the same
namespace, such as `XLS.common.inc`. Readable emission reports an error if the
two modules claim the same declaration. Multiple imports of one DSLX module
within the BX import graph can share its declarations. If a separately
supplied checked DSLX module and the BX imports both include the same DSLX
module, call the separately supplied function via a BX
`extern`, or omit the separate DSLX input and use the BX import.

A DSLX or BX field such as `recOn` retains its proof-facing spelling, so
`p.recOn` reads the
source field. For a struct with such a collision, use positional construction
and pattern matching, for example `(⟨41, true⟩ : XLS.bus.Packet)` for two fields
declared in that order.

For a two-bit DSLX enum with a constructor such as `recOn`, use the source
spelling to construct a value (`XLS.bus.Tag.recOn`) or match it
(`| .recOn => ...`). Lean's `cases t` tactic also works. When naming its
tactic branches explicitly, use `| Constructors.recOn => ...` for enums
whose constructor names overlap Lean-generated declarations.
Lean reserves `_root_` for absolute names, so an enum that needs these
constructor abbreviations cannot also declare a constructor named `_root_`.

For a BX `use` of a DSLX function, readable emission takes parameter names
and types from the same checked DSLX compilation that supplied the function's
IR, including concrete parametric calls. The BX linker records the final
symbol of each imported function so a collision in the destination package
does not change which declaration supplies its Lean interface.

`LeanBlockBundle::files` contains these five files in dependency order:

1. `Generated.lean` - the exact raw emission of the lowered block.
1. `ReadableTypes.lean` - source types and total physical codecs.
1. `ReadableGenerated.lean` - executable equations with source names and types.
1. `ReadableRepresentation.lean` - state and result conversions.
1. `ReadableCorrespondence.lean` - proofs relating both models and their traces.

Callers choose where to write the files and must compile the generated Lean
sources to check their proofs. The readable evaluator executes its own graph
equations; it does not call the raw evaluator. This library does not run block
simulations or orchestrate application-specific property proofs.

## GateFn interface and graph contract

Inputs and outputs are flattened in port-list order, then least-significant bit
first within each port. The bundle retains every port name and width, including
zero-width ports, so Rust evaluation results can be regrouped into the original
`Vec<IrBits>` interface.

The exporter accepts arbitrary node storage order and literal nodes at arbitrary
indices. It requires:

- All references to be in bounds and the complete graph to be acyclic.
- Every input bit to bind a unique, non-negated `AigNode::Input`.
- Every stored input node to be bound by the `GateFn` interface.

Input bindings determine input identity; the name and bit-index metadata on an
input node do not affect evaluation. Invalid graphs return `LeanError` values.

## Evaluate through Lean

Use the actual `bin/lean` executable from a Lean distribution:

```rust
use xlsynth_ir_value::IrBits;
use xlsynth_lean::LeanEvaluator;

let evaluator = LeanEvaluator::new("/path/to/lean/bin/lean")?;
let input_cases = vec![
    vec![IrBits::make_ubits(8, 42)?],
    vec![IrBits::make_ubits(8, 255)?],
];
let output_cases = evaluator.eval_gate_fn_batch(&graph, &input_cases)?;
```

`LeanEvaluator` invokes the selected `lean` executable directly.
`LeanEvaluator::new` materializes and compiles the support modules once in a
temporary directory. Each batch compiles the generated circuit module, then
runs a small wrapper that imports the circuit and the shared evaluation helper.
The evaluator also offers `eval_gate_fn` for a single case and
`eval_circuit_batch` for an already exported bundle.

Input port counts and widths are checked before invoking Lean. The response is
checked for the expected number of cases, bit values, and output widths. Values
use arbitrary-width `IrBits`; they are not limited to 64 bits.

Calls wait for Lean to finish. Process failures return captured Lean diagnostics
without printing from the library.

### Evaluation protocol

The generated Lean entry point passes its imported circuit to the shared
evaluation helper and receives request/response paths. Requests use JSON
containing LSB-first bit strings:

```json
{"cases":["0101","1111"]}
```

Responses use the same bit ordering:

```json
{"outputs":["01","10"]}
```

An empty string represents a zero-width vector. Lean computes and returns these
values; it does not generate a theorem for each test vector.

## Check an equivalence proof

Prepare a target from two `GateFn` graphs, then run a supplied proof:

```rust
use std::path::{Path, PathBuf};
use xlsynth_lean::{LeanProofSources, prepare_equivalence_target, run_equivalence_proof};

prepare_equivalence_target(&lhs, &rhs)?.write_to(Path::new("prepared"))?;
let sources = LeanProofSources {
    root: PathBuf::from("proof"),
    body: PathBuf::from("proof/Body.lean"),
    modules: vec![PathBuf::from("proof/Helpers/Partition.lean")],
};
let output = run_equivalence_proof(
    Path::new("/path/to/lean/bin/lean"),
    Path::new("prepared"),
    &sources,
    Path::new("work"),
)?;
```

The input and output interfaces must have the same port names and widths in the
same order. Interface mismatches return structured `LeanError` variants before
Lean is invoked. The generated `Target.lean` embeds `Lhs.aag` and `Rhs.aag`
through the shared AAG loader, defines `Generated.Lhs.circuit` and
`Generated.Rhs.circuit`, and defines `Target.Equivalent` for that pair. Its meaning
is equality of the complete output vectors for every shared input vector:

```lean
def Circuit.Equivalent (lhs rhs : Circuit inputCount outputCount) : Prop :=
  ∀ inputs, lhs.denote inputs = rhs.denote inputs
```

The proof file contains a complete Lean term for `Target.Equivalent`, including
`by` when using tactics. For example, the small three-input AND test uses:

```lean
by
  unfold Target.Equivalent XlsynthLean.Aig.Circuit.Equivalent
  native_decide
```

The runner generates `UserProof.lean` with `import Target`, imports for every
supplied helper module, and the declaration
`theorem UserProof.equivalent : Target.Equivalent := (...)`. The body is placed
verbatim inside those parentheses, preserving its whitespace and literals. A
direct term such as `Helpers.Proof.equivalent` is also accepted. The body is a
term fragment rather than a standalone Lean module, so it does not contain
top-level imports or a theorem declaration.

This example's `native_decide` uses Lean's native execution trust and contributes
a generated axiom to the proof.

The prepared directory contains the companion support sources, `Lhs.aag`,
`Rhs.aag`, `Target.lean`, and `Check.lean`. It can be retained
as a build artifact or inspected while developing the proof. `Check.lean`
shows the expected root declaration for manual use.

The runner requires an empty work directory. It copies the fixed prepared
artifacts and generates `UserProof.lean` at the work root. Helper modules keep
their paths relative to `LeanProofSources.root`; every supplied file
must be under that root. The generated wrapper imports helpers in the supplied
order, and Lean's `--src-deps` mode determines their compilation order. The
selected installation must include its Lean sources.
Generated module roots and colliding paths are rejected. Before processing
the proof, the driver saves the compiled target and its generated dependencies.
A separate verifier loads those originals alongside the compiled proof modules,
replays the proof declarations through Lean's kernel, and looks up
`UserProof.equivalent` and `Target.Equivalent` by their internal names.
It checks that the theorem's type is definitionally equal to the generated
proposition. User syntax can therefore help write a proof without changing the
verifier's interpretation of the goal.

The runner invokes Lean directly and does not manage Lake projects. Lean's
default solver lookup finds the bundled CaDiCaL beside the Lean executable.
The caller chooses the proof tactics.

`check_equivalence_proof` remains a convenience API for a single proof body; it
prepares and checks the target in a temporary directory.

A successful `LeanProofOutput` means Lean accepted the named theorem at the
generated target type. It returns `lean_version` and the captured `stdout` and
`stderr` from successful compilation steps. A `LeanProofFailure` retains the
underlying `LeanError`, the Lean version if its query succeeded, and diagnostics
from completed compilation steps and the failed process. The checker performs
no axiom audit: Lean's normal acceptance of declarations using axioms or
`sorry` applies.

## Lean semantics and conversion trust

`XlsynthLean.Aag.parse` parses and validates the ASCII AIGER resource before
constructing a stock `Std.Sat.AIG`. `Circuit.eval` evaluates each node once in
topological order. The Lean theorem `Circuit.eval_eq_denote` connects this
evaluator to the stock AIG semantics.

The Rust g8r reader, `GateFn` exporter, and the parser's interpretation of
ASCII AIGER are part of the conversion trust boundary. Differential fuzzing
checks this conversion and evaluation path over generated graphs and inputs;
it is not a formal proof of Rust-to-Lean correspondence.

## Generated ASCII AIGER graph

The `.aag` sidecar uses the canonical combinational subset of ASCII AIGER. It
contains a header, input literals, output literals, and AND records. The
exporter assigns input nodes first, followed by AND nodes in topological order.
Each AND refers to earlier nodes, preserving graph sharing. Source literals map
to AIGER's constant references. The exporter includes all source AND nodes,
including unused nodes.

The Lean parser accepts this emitted subset: consecutive variable numbers,
single-space separators, LF line endings, and no latches, symbols, comments, or
extensions.

The generated Lean module embeds the sidecar and constructs its fixed-width
circuit with `XlsynthLean.Aag.parse`, using `native_decide` to establish that
parsing and validation succeeded. Compile the generated module after each bundle
update so `include_str` reads the current `.aag` contents. Proof modules can
then import the module and use its `circuit` directly. Native g8r files remain
the stored source artifacts.

## Word-level IR functions

`export_ir_function(&package, top, &module_name)` exports a PIR package entry
as a `LeanFunction`. Its `write_to` method writes one model file, which imports
the project's `XlsynthLean.IR` module. The generated `eval` accepts arguments
in XLS signature order. `bits[N]` becomes `BitVec N`, arrays become `Vector α n`, and tuples
become right-nested products ending in `Unit`. For example, a two-field
tuple value is `(a, b, ())`. `evalPacked` packs the first argument at the
low end. Within a tuple its last field is least significant, and within an
array its first element is least significant. Every array dimension in the
selected function and its callees must have positive length.

The exporter supports pure arithmetic (including mixed-width signed and
unsigned multiply), bitwise operations, comparisons, shifts, reductions,
slices and updates, extensions, encoding/decoding, ordinary, priority and
one-hot selects, gates, tuples, arrays, `invoke`, and fixed-array `map`.
PIR extension operations use the existing desugaring pass. Loop, token,
event, block, and partial-product (`smulp`, `umulp`) operations are outside
the supported function fragment. Array reads clamp, slices repeat the final
element, and out-of-range writes preserve the array. For operations marked
`assumed_in_bounds`, equivalence to a successful PIR evaluation assumes
the promised bounds.

Use `LeanEvaluator::eval_ir_function_batch(&function, &argument_lists)` to
compile the generated module using the evaluator's prepared support library. A runner
imports that module and evaluates all argument lists in one request, returning
one typed `IrValue` per case. `LeanEvaluator::eval_ir_functions` accepts a slice of
`LeanFunctionInputs { function, input_cases }` and runs all functions in one
Lean invocation. Each function has its own namespace and may have a different
signature. Results follow function order, then input order.

## Word-level equivalence proofs

`prepare_ir_function_equivalence_target` takes two PIR packages and their optional entrypoint
names. Its bundle contains independently generated `Generated.Lhs.eval` and
`Generated.Rhs.eval` definitions. `Target.Equivalent` quantifies over their
positional XLS arguments and equates the complete return values. Both parameter
and result types must match, including array lengths, tuple nesting and bit widths.
Parameter names may differ. For nullary functions the target is equality of the results.

Each proof export gives every SSA node its own declaration under the generated
function's namespace. Its name is the XLS node ID prefixed with `n`. A proof
can unfold one operation, use `congrArg` on an operand equality, or use a
callee's previously proved contract. The ordinary executable exporter uses
local lets to share intermediates during evaluation. Both layouts use the same
operation translations.

Supply a Lean module and its theorem name using `LeanIrProofSources`, then call
`run_ir_function_equivalence_proof`. For example, when the left graph computes
a four-bit sum through a callee and the right graph adds its operands in reverse
order, `Adder.lean` can prove the target directly:

```lean
import Target

theorem Adder.equivalent : Target.Equivalent := by
  intro a b
  change a + b = b + a
  exact BitVec.add_comm a b
```

Select `Adder.equivalent` with `--proof-theorem` in the driver or `proof_theorem`
in the Bazel rule. Supply any modules imported by `Adder.lean` as additional
sources. The driver generates a wrapper that imports `Adder` and references the
selected theorem by its qualified name. The final verifier uses the original
compiled target to check the theorem, its defining module, and the wrapper's
reference to it.

Import `XlsynthLean.IRProof` for selector and shift lemmas and the
`ir_bv_normalize` tactic. For bit-vector obligations,
`ir_bv_normalize <;> bv_decide` introduces hypotheses, rewrites common XLS
operations, and submits remaining goals to Lean's bit-vector solver.

### Factoring a proof at helper calls

Import `XlsynthLean.IRDecompose` to prove a contract for a helper once and
use it throughout an exported function. In this example the left implementation
calls a conditional tuple swap twice.

```lean
xls_cut (Generated.Lhs.eval p x y) at exchange using exchange_spec as routed expect 2
```

The tactic introduces `routed0` and `routed1` for the two tuple results.
`routed0_spec` specializes the theorem at the first call's actual arguments.
`routed1_spec` describes the second call with `routed0` as its incoming pair.
All uses of a shared result refer to the same variable. A premise in the
original theorem remains a premise in the specialized fact. The caller can
finish the resulting goal with ordinary Lean tactics or cut the other side
using a different helper and contract.

The stage must be fully applied. The contract takes the helper's operands in
order and states a proposition about its result, which can be a bit vector,
tuple, or array. The optional `expect` checks the number of reachable direct
calls. Without `as`, result names start with `cut`. To compose multiple levels
of the call graph, prove a stage contract
and use it in its caller. `xls_unfold helper` expands the helper's own
reachable SSA nodes while preserving calls to other functions.

Lean metaprograms can use `XlsynthLean.IR.sliceCalls` to inspect the actual
typed call applications, shared result nodes, and downstream definitions.
The routing example applies distinct contracts on the two sides of an
equivalence. [Routing proof](tests/fixtures/decomposition/RouteProof.lean)

The checker inspects the theorem's transitive axiom dependencies. By default it permits
Lean's `propext`, `Classical.choice` and `Quot.sound`, and replays closed Boolean
checks from native tactics with Lean's compiler, including the LRAT checks
used by `bv_decide`. A successful IR audit returns the number of distinct checks
replayed in `LeanProofOutput.native_checks_replayed`. Use
`run_ir_function_equivalence_proof_with_policy` with `LeanIrAxiomPolicy::RejectNativeChecks`
to allow only those three axioms. The verifier reports the first disallowed dependency
even if it comes from an imported helper theorem. SAT search can still supply evidence
for a kernel-checked proof term. Unproved assumptions and `sorry` fail either policy. This
uses the same independent root-type check as the AIG proof runner. Proof
elaboration and native computation run with the caller's OS permissions.
Running untrusted code requires a separate sandbox that protects the toolchain
and the verification process. The supplied IR, translator, and Lean toolchain
are trusted inputs.

## Validation

Run the Rust-only tests with

```sh
cargo test -p xlsynth-lean
```

To also run the AIG, XLS IR and block trace evaluation and proof tests, enable `lean-tests`
and set `XLSYNTH_LEAN` to the executable from the [pinned toolchain](lean/lean-toolchain).

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --features lean-tests
```

The block tests check BX field order across a BX constructor and a separately compiled DSLX selector.

Ubuntu 24.04 CI runs this suite with the official Linux x86-64 Lean release.
When updating `lean/lean-toolchain`, update the [release checksum](../scripts/lean_linux_x64.sha256) to match.

The differential fuzz target generates bounded graphs, permutes their node
storage, and compares Lean output batches with `PreparedGateSim`:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo fuzz run --fuzz-dir xlsynth-lean/fuzz \
    --features external-lean --sanitizer none fuzz_gate_fn_lean_eval \
    -- -max_total_time=300
```

For a stateful block, `LeanEvaluator::eval_block_trace` takes an initial register
state and one input vector per clock. Each result contains the outputs observed
on that clock and the register values carried into the next clock. Ports and
registers use declaration order. Each value has its declared XLS type, including
arrays, tuples, and bit vectors of any supported width (even `bits[0]`).

To compare generated block traces against the PIR interpreter, run:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo fuzz run --fuzz-dir xlsynth-lean/fuzz \
    --features external-lean --sanitizer none fuzz_block_lean_eval \
    -- -max_total_time=300
```

See the repository's `FUZZ.md` for fuzz target coverage and failure diagnostics.
