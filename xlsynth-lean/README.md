# xlsynth-lean

`xlsynth-lean` exports combinational `GateFn` graphs to Lean, evaluates them
through Lean, and checks user-supplied Lean equivalence proofs. Evaluation
returns the computed output values to Rust. Native `.g8r` and `.g8rbin` files
can be loaded with the existing `xlsynth-g8r` reader before calling this crate.

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
combinational API. Stateful export is future work.

### Interface and graph contract

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

## Validation

Rust export tests run normally:

```sh
cargo test -p xlsynth-lean
```

Lean integration tests are explicitly enabled with a toolchain path:

```sh
XLSYNTH_LEAN=/path/to/lean/bin/lean \
  cargo test -p xlsynth-lean --test lean_eval -- --ignored
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

See the repository's `FUZZ.md` for the target's coverage and failure diagnostics.
