# `xlsynth-codegen`

`xlsynth-codegen` lowers existing XLS block IR into deterministic, synthesizable
SystemVerilog. The backend preserves the source block's ports, combinational
behavior, register boundaries, reset and load-enable behavior, block hierarchy,
and supported foreign-function instantiations.

## Rust API

Parse and verify a package with `xlsynth-pir`, then emit its selected block:

```rust
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir_parser::Parser;

let source = r#"package public_example

top block identity(data: bits[8], result: bits[8]) {
  data: bits[8] = input_port(name=data, id=1)
  result: () = output_port(data, name=result, id=2)
}
"#;

let package = Parser::new(source).parse_and_verify_package().unwrap();
let generated = emit_system_verilog(&package, &BlockCodegenOptions::default())
    .unwrap();
assert!(generated.system_verilog.contains("module identity("));
assert!(generated.system_verilog.contains("assign result = data;"));
```

The `block2sv` subcommand in `xlsynth-driver` exposes the same backend on the
command line:

```sh
cargo run -p xlsynth-driver -- block2sv /path/to/public_block.ir
cargo run -p xlsynth-driver -- block2sv --layout pipeline /path/to/public_block.ir
```

## Supported block features

- Scalar XLS arithmetic, comparisons, shifts, reductions, selectors, partial
  products, and bit-manipulation operations.
- Packed tuples, arrays, nested aggregates, indexing, updates, and array
  slices with XLS-compatible boundary behavior.
- Synchronous and asynchronous active-high or active-low resets, load enables,
  state feedback, and optional register-generation templates.
- Transitive block hierarchy and existing foreign-function instantiations.
- Width-specific automatic SystemVerilog helpers for mixed-width multiplication,
  partial products, and division/modulo, using original-width operands.
- Priority selects lowered to compact `unique casez` helpers with disjoint
  patterns; encoders built from one OR expression per output bit.
- One-hot priority encoders lowered to width- and direction-specialized
  `unique casez` functions, with an explicit all-zero sentinel arm. Repeated
  operations reuse the same helper; zero-width inputs fold to the sentinel.
  The fallback is unknown for unmatched four-state inputs; the lowering
  preserves two-state XLS behavior, not ternary-chain X/Z propagation.
- Leading-zero counts, priority encoders, and left normalization lowered into
  shared `casez` helper functions with explicit priority patterns. Normalization
  selects static slices and concatenations, with an optional leading-zero count.
- Reset-aware assertions, coverage, and fully formatted trace operations.
- Source port ordering, optional SystemVerilog port types, deterministic
  naming, module-name overrides, and configurable expression inlining.

`Layout::None`, the default, emits the source block in dependency order and
supports arbitrary legal register feedback. Ports, temporaries, registers,
and helper-function signals and return types use explicit `logic` declarations.
Helpers retain function-name assignments and `casez` for compatibility with
Yosys's built-in SystemVerilog frontend. Leading/trailing-bit classification,
normalization, and priority selection use `unique casez` with patterns that are
disjoint for two-state inputs and an explicit default arm.
Module signal declarations
precede statements in flat layout; pipeline layout groups declarations before
the statements within each stage, retaining the register sections.
`Layout::Pipeline` reconstructs
feed-forward pipeline stages from the existing registers and adds stage and
register comments without changing their timing. Designs containing register
feedback or inconsistent stage dependencies are rejected in pipeline layout;
they remain supported by `Layout::None`.
Literals and pure constant-only expressions can be shared across stages without
extra registers. They are emitted before their earliest consumer and are not
required to have the same stage number as every consuming register write.
Inputs, register reads, and instance outputs retain their cycle dependencies.

Expression materialization separates mandatory lowering constraints from
readability heuristics. Arithmetic width boundaries, packed arrays, and operand
roles requiring a named reference are mandatory. In particular, each nonzero-width
array-update index uses an explicitly sized `logic` signal, so comparison against
a `genvar` cannot widen the index's computation. Ports and register reads already
provide such signals; zero-width indices use their unique constant-zero value.
Explicit node names, multiple uses, and `max_inline_depth` provide additional
readability boundaries. `separate_lines` forces assignments throughout; increasing
the inline-depth limit never disables mandatory materialization.
Dynamic slices and slice updates use width-specialized functions with typed
arguments and results, explicit padding/truncation, and out-of-range guards.
Guards are omitted when the index width proves the start is always in range.
Carry results are assigned at their declared widths; static slices and tuple
projections use named sources for direct part-selects. This keeps shrinking casts
out of surrounding expressions where tool-specific width inference could alter
concatenation positions. Extended-adder terms also use named sources when
resizing requires part-selects.

Block instances are checked for combinational cycles across hierarchical
boundaries. Register-delimited feedback remains valid, and independently timed
child outputs are analyzed separately. Each declared register must have exactly
one `register_read` and one `register_write`; multiple write nodes are rejected
instead of silently dropping an update.

`invoke` and `counted_for` are unsupported, including unused and zero-width
results and zero-trip loops. Inline function calls and unroll loops before code
generation. This does not affect generated arithmetic helpers or external
instantiations. Name collisions use double-underscore suffixes such as `add_3__1`.
Emitted ports (including clocks), instance names, and assertion/coverage labels
are fixed module-scope names. They are validated and reserved before any generated
declarations; fixed-name collisions produce an error, while generated names are
uniquified around them. Names can be reused in different modules. Omitted
zero-bit ports and disabled assertions do not reserve names. Additional identifiers
introduced inside raw foreign-function or register templates remain the template
author's responsibility.

Dynamic array indexing clamps out-of-bounds indices by default. Set
`assumed_in_bounds=true` on an index node to omit the guard for a proven-safe
access. Guards are also omitted when the index bit width cannot represent an
out-of-bounds value, independently for each indexed dimension. Set
`array_index_bounds_checking` to `false` to request stock-XLS-compatible
unchecked indexing. Array slices retain their defined clamping semantics in
either configuration. Indexed array updates, including multidimensional arrays,
packed subarrays, and tuple elements, use compact SystemVerilog generate
loops. Multi-element array slices use the same generated-element structure.
Loop variables use short names such as `__i0` and `__i1`, reused across sibling
loops; labels use `gen__<signal>_<dimension>`. Collision suffixes prevent
shadowing module signals.
Ports, intermediates, and registers retain array dimensions as descending packed
ranges: `bits[8][3][2]` becomes `logic [1:0][2:0][7:0]`. Element zero occupies the
least-significant element-sized region, preserving the bit layout of flat
connections. Tuples remain flat packed vectors, including tuple leaves of arrays;
zero-bit values are omitted. Explicit `sv_type` port overrides are preserved,
with shaped internal views for custom-typed array inputs. Nested dynamic reads
use named subarrays between dimensions for simulator portability. Bounds
checking, reset/enable behavior, and pipeline timing are independent of this
representation.

## Validation

`emit_asserts` (default: `true`) controls emission of assert operations already
present in the input block. It does not infer new invariants or disable coverage
and trace statements.

Semantic tests selected with `iverilog-tests` require iverilog and vvp. Generated RTL
is compiled once per design and simulated against the independent PIR block
interpreter, with exact four-state output comparisons and cycle-by-cycle
register checks. Testbench interfaces come from IR, not an SV parser.
The separate `yosys-tests` suite checks consumption of generated SystemVerilog,
combinational outputs with Yosys `eval`, and focused synthesis equivalence
without Liberty files or a PDK. Random-block fuzzing also checks the
Yosys/ABC → mapped netlist → netlist evaluator path
against PIR expectations. These mapping checks require `XLSYNTH_YOSYS_PATH`
and `XLSYNTH_LIBERTY_FILES`, a comma-separated list of standard-cell Liberty
files. Include flip-flop cells for sequential mapping. The same files supply
cell definitions to Yosys/ABC and our netlist evaluator.
These codegen paths do not use C++ compilation or the `xlsynth-vastly`
evaluator.

Shared compilation, syntax checks, and finite simulations live in
`xlsynth-test-helpers::iverilog::IcarusToolchain`; the persistent vector/cycle
protocol lives in `xlsynth-test-helpers::rtl_sim`. Both use
`xlsynth::external_tool` for bounded execution and process cleanup.
Yosys execution, batched combinational evaluation, and Liberty-backed mapping live in
`xlsynth-g8r::netlist::yosys`. The codegen adapters retain only IR interfaces,
testbench construction, and semantic comparisons.

The golden corpus and its in-process runner live in this crate. Fixtures use
typed `OPTIONS` metadata and exact `EXPECT-SV` / `EXPECT-ERROR` comments; tests
call `emit_system_verilog` directly. No codegen or driver binary is needed.
See [the fixture format](tests/goldens/block2sv/README.md).

Run the backend/golden tests and the small driver CLI-boundary suite with:

```sh
cargo test -p xlsynth-codegen
cargo test -p xlsynth-driver --test block2sv_cli

# Explicitly select direct RTL simulation and Yosys consumption/proof checks.
cargo test -p xlsynth-codegen --features iverilog-tests,yosys-tests
```

Regenerate native fixture expectations without external EDA tools:

```sh
XLSYNTH_UPDATE_GOLDEN=1 cargo test -p xlsynth-codegen \
  --test block2sv_goldens block2sv_golden_fixtures -- --exact
```

The remaining standalone `.sv` snapshots use the same update variable in
the Icarus suite. To update both kinds and run their semantic checks:

```sh
XLSYNTH_UPDATE_GOLDEN=1 cargo test -p xlsynth-codegen --features iverilog-tests
```

Review the diff after regeneration.

Both features are off by default. Without them, external-tool tests are not
selected; when selected, missing or broken tools fail the tests. Icarus uses
`XLSYNTH_IVERILOG_PATH` / `XLSYNTH_VVP_PATH`, falling back to `PATH`. Yosys tests
use `XLSYNTH_YOSYS_PATH`, falling back to `PATH`; they require recent
SystemVerilog support (CI pins OSS CAD Suite 2026-08-29). Runtime library APIs
remain available without these test-selection features.

CI runs the library-free Yosys smoke/proof tests and builds the mapping fuzz
targets, but does not run technology mapping. For mapping campaigns, supply a
compatible Liberty set such as ASAP7 7.5-track RVT slow-corner NLDM. This is runtime configuration;
it adds no PDK-specific Cargo feature. See the
[Liberty configuration](fuzz/FUZZ.md#liberty-configuration) for setup and a
preflight command. No LEF, GDS, or other physical-design collateral is needed
for these checks.

The standalone coverage-guided test suite in `fuzz/` exercises determinism,
scalar and aggregate semantics, sequential behavior, hierarchy, foreign
instantiations, Yosys/ABC mapping, and bounded
formal equivalence. See [`fuzz/FUZZ.md`](fuzz/FUZZ.md) for required external
tool configuration and individual target descriptions.
