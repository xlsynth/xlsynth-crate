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
use xlsynth_lean::block;

let selected = block::select_block(&package, None)?;
let generated_lean = block::emit(&package, selected)?;
```

Pass a verified `xlsynth_pir::ir::Package` to `select_block`. Supply a block
name to select it explicitly, or omit the name to select the declared top
or the sole block. `emit` uses the package's source locations and the selected
block to produce a self-contained Lean module in namespace `XlsBlock`.
When starting from saved IR text, run native XLS verification and typed PIR
validation on that text before selecting the block, as in the `export_block`
example.

The generated `eval` takes arbitrary old registers and current inputs and
returns outputs, next registers, and assertion results. Outputs read old
registers; all register writes commit together. For example, if `a` is written
from `b` and `b` from `a`, an old state `(a = 1, b = 2)` produces next state
`(a = 2, b = 1)`. An output driven by `a` still observes `1` on that edge.
`reset` activates the declared reset input for one ordinary clock edge. It
does not invent initial values for registers without reset values.

The supported subset includes exact-width bits, nested tuples, fixed-size arrays, tokens,
registers, synchronous resets, assertions, bitwise operations, addition,
equality, concatenation, slicing, sign extension, one-hot encoding, and
ordinary, priority, and bits-valued one-hot selection. Calls to pure functions
and fixed-trip-count loops remain explicit in the model. Unsupported nodes,
zero-width bits, block instances, and asynchronous resets return
contextual errors. Parsing can succeed for operations that emission rejects.
Every assertion remains an observable result, including disconnected ones;
assertions never constrain the model's input or old-state domain.

### Compile a BX block

BX declares each register, its reset value, the data inputs and the output
ports. For example, this block exposes its old count on output and increments
it on each clock unless reset is asserted.

```
pub block counter(clk, rst, take: bool) -> u8 {
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
as a build artifact or inspected while developing the proof.

The runner requires an empty work directory. It copies the fixed prepared
artifacts and generates `UserProof.lean` at the work root. Helper modules keep
their paths relative to `LeanProofSources.root`; every supplied file
must be under that root. The generated wrapper imports helpers in the supplied
order, and Lean's `--src-deps` mode determines their compilation order. The
selected installation must include its Lean sources.
Generated module roots and colliding paths are rejected. A final generated
module checks `UserProof.equivalent` at `Target.Equivalent`.

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
as a `LeanFunction`. `write_to` writes its Lean source, companion IR module,
and toolchain pin. The generated `eval` accepts arguments in XLS signature
order. `bits[N]` becomes `BitVec N`, arrays become `Vector α n`, and tuples
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
compile the standalone module written by `LeanFunction::write_to`. A runner
imports that module and evaluates all argument lists in one request, returning
one typed `IrValue` per case. `LeanEvaluator::eval_ir_functions` accepts a slice of
`LeanFunctionInputs { function, input_cases }` and runs all functions in one
Lean invocation. Each function has its own namespace and may have a different
signature. Results follow function order, then input order.

## Validation

Rust export tests run normally:

```sh
cargo test -p xlsynth-lean
```

Run the gate evaluation, block trace and readable correspondence tests with the Lean executable.

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --test lean_eval --test block_native -- --ignored
```

The block tests check BX field order for every input, both in the readable
interface and across a BX constructor and a separately compiled DSLX selector.

Run the real Lean word-level evaluation tests with the same toolchain:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --test ir_fn -- --include-ignored
```

Proof integration tests use the same explicit toolchain configuration:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --test proof -- --ignored
```

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
registers use declaration order. This evaluator accepts bit-vector ports and
registers.

To compare generated block traces against the PIR interpreter, run:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo fuzz run --fuzz-dir xlsynth-lean/fuzz \
    --features external-lean --sanitizer none fuzz_block_lean_eval \
    -- -max_total_time=300
```

See the repository's `FUZZ.md` for fuzz target coverage and failure diagnostics.
