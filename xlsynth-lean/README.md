# xlsynth-lean

`xlsynth-lean` exports combinational `GateFn` graphs and stateful XLS block IR
to Lean. Gate graphs can also be evaluated through Lean, returning computed
output values to Rust. Native `.g8r` and `.g8rbin` files can be loaded with the
existing `xlsynth-g8r` reader before calling the gate-graph API.

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

The generated `eval` takes arbitrary old registers and current inputs and
returns outputs, next registers, and assertion results. Outputs read old
registers; all register writes commit together. For example, if `a` is written
from `b` and `b` from `a`, an old state `(a = 1, b = 2)` produces next state
`(a = 2, b = 1)`. An output driven by `a` still observes `1` on that edge.
`reset` activates the declared reset input for one ordinary clock edge. It
does not invent initial values for registers without reset values.

The supported subset includes exact-width bits, nested tuples, tokens,
registers, synchronous resets, assertions, bitwise operations, addition,
equality, concatenation, slicing, sign extension, one-hot encoding, and
ordinary, priority, and bits-valued one-hot selection. Unsupported nodes,
arrays, zero-width bits, block instances, and asynchronous resets return
contextual errors. Parsing can succeed for operations that emission rejects.
Every assertion remains an observable result, including disconnected ones;
assertions never constrain the model's input or old-state domain.

### Compile a stateful DSLX function

A parameter prefixed with `__reg_` represents the current register value. The
last return-tuple elements supply the next register values in parameter order.
Earlier elements are output ports named `out0`, `out1`, and so on.

```dslx
struct Outputs { seen: bool }

fn counter(take: bool, __reg_count: u8) -> (Outputs, u8) {
    (Outputs { seen: take }, __reg_count + u8:1)
}
```

The model exposes `count` as state. With one output port, `observe q x`
returns the `Outputs` value directly. A Lean proof can bind it with
`let out0 := observe q x` and use `out0.seen`. The original DSLX supplies
parameter names, aliases, struct fields and enum variants. The block IR
supplies the cycle transition. Internal bindings stay private.

```rust
use std::path::Path;
use xlsynth_lean::block::compile_stateful_function;

let compiled = compile_stateful_function(&source, Path::new("counter.x"), "counter", None, &[])?;
let ir = compiled.block_ir();
let bundle = compiled.emit_readable()?;
```

The readable interface supports unsigned bits of positive width, positional
tuples, nonempty local structs, aliases of supported local types and unsigned
two-bit enums with distinct encodings. For each unnamed two-bit encoding the
emitter adds a Lean constructor so every physical value can be decoded and
encoded again. Exposed struct fields and enum cases must avoid names Lean
generates for their type, such as `recOn`.

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

## Validation

Rust export tests run normally:

```sh
cargo test -p xlsynth-lean
```

Lean integration tests for gate evaluation and readable block correspondence
are explicitly enabled with a toolchain path:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --test lean_eval --test block_native -- --ignored
```

The differential fuzz target generates bounded graphs, permutes their node
storage, and compares Lean output batches with `PreparedGateSim`:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo fuzz run --fuzz-dir xlsynth-lean/fuzz \
    --features external-lean --sanitizer none fuzz_gate_fn_lean_eval \
    -- -max_total_time=300
```

See the repository's `FUZZ.md` for the target's coverage and failure diagnostics.
