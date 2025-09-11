# `xlsynth-driver` command line interface

The `xlsynth-driver` binary is a "driver program" for various XLS/xlsynth tools and functionality behind a single unified command line interface. It is organized into subcommands.

## Example Usage

While developing you can invoke the driver with `cargo run`. The example below
assumes a toolchain configuration file at `$HOME/xlsynth-toolchain.toml`:

```shell
cargo run -p xlsynth-driver -- --toolchain=$HOME/xlsynth-toolchain.toml \
    dslx2ir ../sample-usage/src/sample.x
cargo run -p xlsynth-driver -- --toolchain=$HOME/xlsynth-toolchain.toml \
    dslx2pipeline ../sample-usage/src/sample.x add1 \
    --delay_model=asap7 --pipeline_stages=2
cargo run -p xlsynth-driver -- dslx2sv-types ../tests/structure_zoo.x
```

For a full list of options, run `xlsynth-driver <subcommand> --help`.

## Subcommands

### `ir-equiv`

Proves two IR functions to be equivalent or provides a counterexample to their equivalence.

Key flags:

- `--top <NAME>` or per-side `--lhs_ir_top <NAME>` / `--rhs_ir_top <NAME>` to select entry points.
- `--solver <toolchain|bitwuzla|boolector|z3-binary|bitwuzla-binary|boolector-binary>`
- `--flatten_aggregates=<BOOL>`
- `--drop_params <CSV>`
- `--parallelism-strategy <single-threaded|output-bits|input-bit-split>`
- `--assertion-semantics <ignore|never|same|assume|implies>`
- `--lhs_fixed_implicit_activation=<BOOL>` / `--rhs_fixed_implicit_activation=<BOOL>`
- `--output_json <PATH>` to write the JSON result.

### `ir-equiv-blocks`

Proves two IR blocks to be equivalent by selecting block members from package-form IR inputs (both operands must be packages) and checking function-level equivalence on the lifted blocks (as in `ir-equiv`).

Key flags:

- `--lhs_top <NAME>` / `--rhs_top <NAME>` or shared `--top <NAME>` to select block entry points (by block name in each package). If omitted, the package `top` block is used when present; otherwise the first block member is selected.
- `--solver <toolchain|bitwuzla|boolector|z3-binary|bitwuzla-binary|boolector-binary>`
- `--flatten_aggregates=<BOOL>`
- `--drop_params <CSV>`
- `--parallelism-strategy <single-threaded|output-bits|input-bit-split>`
- `--assertion-semantics <ignore|never|same|assume|implies>`
- `--lhs_fixed_implicit_activation=<BOOL>` / `--rhs_fixed_implicit_activation=<BOOL>`
- `--output_json <PATH>` to write the JSON result.

### `lib2proto`: liberty files to proto

Liberty files can be unwieldy and large in their textual form -- this command reformats the data
for streamlined querying, e.g. by the `gv2ir` subcommand.

```shell
xlsynth-driver lib2proto \
  --output ~/asap7.proto \
  ~/src/asap7/asap7sc7p5t_28/LIB/NLDM/*TT*.lib
```

### `gv2ir`: gate-level netlist to IR

```shell
xlsynth-driver gv2ir \
  --netlist ~/my_netlist.v \
  --liberty_proto ~/asap7.proto > ~/my_netlist.ir
```

- Optional flags:
  - `--dff_cells <CSV>` – comma-separated list of DFF cell names to treat as identity (D->Q).
  - `--dff_cell_formula <STR>` – auto-classify cells as DFFs for identity wiring when any output pin's Liberty function exactly matches this string (e.g., `IQ`). Identity wiring sets `Q = D`.
  - `--dff_cell_invert_formula <STR>` – auto-classify cells as DFFs with inverted output when any output pin's Liberty function exactly matches this string (e.g., `IQN`). Inverted wiring sets `QN = NOT(D)`.

Example (ASAP7):

```shell
xlsynth-driver gv2ir \
  --netlist add_mul.vg \
  --liberty_proto ~/asap7.proto \
  --dff_cell_formula IQ \
  --dff_cell_invert_formula IQN > add_mul.ir
```

### `gv-read-stats`: netlist statistics

Reads a gate-level netlist (optionally gzipped) and prints summary statistics such as
instance counts, net counts, memory usage, parse time, and per-cell instance histogram.

```shell
xlsynth-driver gv-read-stats my_module.gv.gz
```

This command has no flags.

### `ir2g8r`: IR to gate-level representation

Converts an XLS IR file to an `xlsynth_g8r::GateFn` (i.e. a gate-level netlist in AIG form).

- By default the pretty-printed GateFn is sent to **stdout**.
- Additional artifacts can be emitted with flags:
  - `--bin-out <PATH>` – write the GateFn as a binary **.g8rbin** file (bincode-encoded).
  - `--stats-out <PATH>` – write a JSON summary of structural statistics.
  - `--netlist-out <PATH>` – write a human-readable gate-level netlist to a file.
- The same optimization / analysis flags accepted by `ir2gates` are supported (`--fold`, `--hash`, `--fraig`, `--toggle-sample-count`, …).

Example:

```shell
xlsynth-driver ir2g8r my_module.opt.ir \
  --fraig=true \
  --bin-out my_module.g8rbin \
  --stats-out my_module.stats.json > my_module.g8r
```

The command above leaves three artifacts:

1. `my_module.g8r` – human-readable GateFn (stdout redirection).
1. `my_module.g8rbin` – compact bincode serialisation of the same GateFn.
1. `my_module.stats.json` – structural summary statistics as JSON.

### `g8r2v`: GateFn to gate-level netlist (Verilog-like)

Converts a `.g8r` (text) or `.g8rbin` (bincode) file containing a gate-level `GateFn` to a `.ugv` netlist (human-readable, Verilog-like) on **stdout**.

- By default, emits the netlist with the original GateFn inputs.
- The `--add-clk-port[=NAME]` flag inserts an (unused) clock port as the first input:
  - If omitted: no clock port is added.
  - If given as `--add-clk-port` (no value): adds a port named `clk`.
  - If given as `--add-clk-port=foo`: adds a port named `foo`.

Additional flags:

- `--flop-inputs` – add a layer of flops for all inputs.
- `--flop-outputs` – add a layer of flops for all outputs.
- `--use-system-verilog` – emit SystemVerilog instead of Verilog.
- `--module-name <NAME>` – override the generated module name.

Note: If `--flop-inputs` or `--flop-outputs` is used you must also provide `--add-clk-port=<NAME>` to name the clock.

Example usage:

```shell
# No clock port
xlsynth-driver g8r2v my_module.g8r > my_module.ugv

# Add a clock port named 'clk'
xlsynth-driver g8r2v my_module.g8r --add-clk-port > my_module.ugv

# Add a clock port named 'myclk'
xlsynth-driver g8r2v my_module.g8r --add-clk-port=myclk > my_module.ugv
```

The output is always written to stdout; redirect to a `.ugv` file as needed.

Example with flops and SystemVerilog output:

```shell
xlsynth-driver g8r2v my_module.g8r \
  --add-clk-port=clk \
  --flop-inputs --flop-outputs \
  --use-system-verilog \
  --module-name=my_module_g8r > my_module.ugv
```

### `ir-round-trip`

Parses an IR file and writes it back to stdout. Useful for validating round-trip stability and (optionally) removing position metadata.

- Positional arguments: `<ir_input_file>`
- Option:
  - `--strip-pos-attrs=<BOOL>` – when `true`, strip `file_number` lines and any `pos=[(fileno,line,col), ...]` attributes from the output.

Example:

```shell
xlsynth-driver ir-round-trip my_pkg.ir --strip-pos-attrs=true > my_pkg.nopos.ir
```

### `version`

Prints the driver version string to **stdout**.

### `dslx2pipeline`: DSLX to pipelined Verilog

Translates a DSLX entry point to a pipelined SystemVerilog module.
The resulting Verilog is printed on **stdout**.
Diagnostic messages and the path to temporary files (when
`--keep_temps=true`) are written to **stderr**.

- The `--type_inference_v2` flag enables the experimental type inference v2 algorithm.
  **Requires:** `--toolchain` (external tool path). If used without `--toolchain`, the driver will print an error and exit.

Additional outputs:

- `--output_unopt_ir <PATH>` – write the unoptimized IR package to a file.
- `--output_opt_ir <PATH>` – write the optimized IR package to a file.

### `dslx2ir`: DSLX to IR

Converts DSLX source code to the XLS IR. The IR text is emitted on **stdout**.
DSLX warnings and errors appear on **stderr**.

- The `--type_inference_v2` flag enables the experimental type inference v2 algorithm.
  **Requires:** `--toolchain` (external tool path). If used without `--toolchain`, the driver will print an error and exit.

Optional optimization:

- `--opt=true` – run the IR optimizer before emitting. When set, `--dslx_top` becomes required.

Additional flags:

- `--convert_tests=<BOOL>` – convert DSLX `#[test]` procs/functions to IR as regular IR functions (default `false`).

### `dslx2sv-types`: DSLX type definitions to SystemVerilog

Generates SystemVerilog type declarations for the definitions in a DSLX file.
The output is written to **stdout**.

### `dslx-show`: Show a DSLX symbol definition

Resolves and prints a DSLX symbol definition (enums, structs, type aliases, constants, functions, quickchecks).

- Positional: `SYMBOL` – either unqualified (`Name`) or qualified with a dotted module path plus `::member` (e.g., `foo.bar::Name`, `foo.bar.baz::Name`).
- Optional flags:
  - `--dslx_input_file <FILE>` – required when `SYMBOL` is unqualified; the file’s directory is added to the search path.
  - `--dslx_path <P1;P2;...>` – semicolon-separated list of additional DSLX search directories.
  - `--dslx_stdlib_path <PATH>` – path to the DSLX standard library root.

Note: In DSLX source files, imports use dot-separated module paths (e.g., `import foo.bar.baz;`). On the CLI, qualify symbols as `<dotted.module.path>::<Member>`, e.g., `foo.bar.baz::Name`.

Examples:

```shell
# Show a struct defined in a local file
xlsynth-driver dslx-show \
  --dslx_input_file sample-usage/src/sample_with_struct_def.x \
  Point

# Show an enum defined in another module by qualifying the symbol
xlsynth-driver dslx-show \
  --dslx_path=sample-usage/src \
  sample_with_enum_def::MyEnum

# Modules under nested directories (example)
xlsynth-driver dslx-show \
  --dslx_path=/path/to/dslx/libs \
  foo.bar.baz::Baz
```

The definition is printed to stdout; errors are written to stderr and a non-zero status is returned if the symbol cannot be resolved.

### `dslx-g8r-stats`: DSLX GateFn statistics

Converts a DSLX entry point all the way to a gate-level representation and
prints a JSON summary of structural statistics. It performs IR conversion,
optimization, and gatification using either the toolchain or the runtime APIs.

- The `--type_inference_v2` flag enables the experimental type inference v2 algorithm.
  **Requires:** `--toolchain` (external tool path). If used without `--toolchain`, the driver will print an error and exit.

### `ir2opt`: optimize IR

Runs the XLS optimizer on an IR file and prints the optimized IR to **stdout**.
Requires `--top <NAME>` to select the entry point.

### `ir2pipeline`: IR to pipelined Verilog

Produces a pipelined SystemVerilog design from an IR file. The generated code
is printed to **stdout**. When `--keep_temps=true` the location of temporary
files is reported on **stderr**.

Optional optimization:

- `--opt=true` – optimize the IR before scheduling/codegen.

### `ir2combo`: IR to *combinational* SystemVerilog

Similar to `ir2pipeline` but requests the *combinational* backend in `codegen_main`.
Generates a single‐cycle (no pipeline registers) SystemVerilog module on **stdout**.
All the usual code-gen flags (e.g., `--use_system_verilog`, `--add_invariant_assertions`,
`--flop_inputs`, `--flop_outputs`, etc.) are supported.

Optional optimization:

- `--opt=true` – optimize the IR before code generation.

Example:

```shell
xlsynth-driver --toolchain=$HOME/xlsynth-toolchain.toml \
  ir2combo my_design.opt.ir \
  --top my_main \
  --delay_model=unit \
  --use_system_verilog=true > my_design.sv
```

### `ir-fn-to-block`: IR function to Block IR (toolchain-only)

Emits the Block IR for a single IR function using the external toolchain.

_Implementation note:_ This is a thin wrapper over `codegen_main` with `--generator=combinational`, `--delay_model=unit`, and `--output_block_ir` directed to a temporary file that is then printed to **stdout**.

- Requires a `--toolchain` whose TOML points to a valid `tool_path` containing `codegen_main`.
- Positional arguments and flags:
  - `<ir_input_file>` – path to the package IR file.
  - `--top <NAME>` – name of the IR function to emit as a block.

Example:

```shell
xlsynth-driver --toolchain=$HOME/xlsynth-toolchain.toml \
  ir-fn-to-block my_pkg.ir --top my_main > my_main.block.ir
```

### `ir2delayinfo`

Runs the `delay_info_main` tool for a given IR entry point and delay model.
The produced delay-information proto text is written to **stdout**; any tool
diagnostics appear on **stderr**.

### `ir-ged`

Computes the Graph-Edit-Distance between two IR functions. Without further
flags a summary line like `Distance: N` is printed on **stdout**. With
`--json=true` the result is emitted as JSON.

### `ir-structural-similarity`

Computes a structural similarity summary between two IR functions by hashing node structure per depth and comparing multisets.

- Positional arguments: `<lhs.ir> <rhs.ir>`
- Entry-point selection (optional): `--lhs_ir_top <NAME>` and `--rhs_ir_top <NAME>`; if omitted, the package `top` or first function is used on each side.
- Output:
  - Always prints the return-node depth for each side, then one line per discrepant depth with the total discrepancy count, followed by concise opcode summaries on separate lines for LHS and RHS.
  - With `--show_discrepancies=true`, also prints detailed signature lines for items present only on one side.
  - Copies the original inputs to an output directory for convenience:
    - `lhs_orig.ir` and `rhs_orig.ir`.
    - Control the directory with `--output-dir=<DIR>`. If omitted, a temporary directory is created and its path is printed.

Example:

```shell
xlsynth-driver ir-structural-similarity lhs.opt.ir rhs.opt.ir
```

Sample output (truncated):

```text
LHS return depth: 53
RHS return depth: 53
depth 12: 2
  lhs: {}
  rhs: {nor: 1, or: 1}
depth 13: 5
  lhs: {and: 1, or: 1}
  rhs: {and: 1, or: 2}
```

Verbose details:

```shell
xlsynth-driver ir-structural-similarity lhs.opt.ir rhs.opt.ir --show_discrepancies=true
```

Notes:

- Structural hashing ignores position metadata, assertion/trace strings, and parameter text ids (params are keyed by ordinal position in the signature). It includes node kinds, types, selected attributes (e.g., widths), and child structure.
- Opcode summaries group discrepancies by operator per depth to make eyeballing easier; detailed signatures include operand/attribute types for precise diagnosis.

### `ir-localized-eco`

Computes a localized ECO diff (old → new) between two IR functions and emits a JSON edit list plus a brief summary. Optionally writes outputs to a directory and runs quick interpreter sanity checks.

- Positional arguments: `<old.ir> <new.ir>`
- Entry-point selection (optional): `--old_ir_top <NAME>` and `--new_ir_top <NAME>`; if omitted, the package `top` or first function is used on each side.
- Output controls:
  - `--json_out <PATH>` – write the JSON edit list to this file; if omitted, a temp file is created and its path printed.
  - `--output_dir <DIR>` – write outputs (JSON, patched old .ir) to this directory; if omitted, a temp dir is created and printed.
- Sanity checks:
  - `--sanity-samples <N>` – if > 0, run N randomized interpreter samples (in addition to all-zeros and all-ones) to sanity-check that patched(old) ≡ new.
  - `--sanity-seed <SEED>` – seed for randomized interpreter samples.
  - `--compute-text-diff=<BOOL>` – compute IR/RTL text diffs (expensive). Defaults to `false`.

Example:

```shell
xlsynth-driver ir-localized-eco old.opt.ir new.opt.ir \
  --old_ir_top=main --new_ir_top=main \
  --output_dir=eco_out --sanity-samples=10 --sanity-seed=0
```

### `ir2gates`: IR to GateFn statistics

Maps an IR function to a gate-level representation and prints a structural
statistics report. By default the report is human-readable text. With
`--quiet=true` the summary is emitted as JSON instead. The optional
`--output_json=<PATH>` flag writes the same JSON summary to a file regardless of
the quiet setting.

Supported flags include the common gate-optimization controls:

- `--fold` – fold the gate representation (default `true`).
- `--hash` – hash-cons the gate representation (default `true`).
- `--adder-mapping=<ripple-carry|brent-kung|kogge-stone>` – choose the adder
  topology.
- `--fraig` – run fraig optimization (default `true`).
- `--fraig-max-iterations=<N>` – maximum FRAIG iterations.
- `--fraig-sim-samples=<N>` – number of random samples for FRAIG.
- `--toggle-sample-count=<N>` – if non-zero, generate `N` random samples and
  report toggle statistics.
- `--toggle-seed=<SEED>` – seed for toggle sampling (default `0`).
- `--compute-graph-logical-effort` – compute graph logical effort statistics.
- `--graph-logical-effort-beta1=<BETA1>` / `--graph-logical-effort-beta2=<BETA2>`
  – parameters for graph logical effort analysis.

### `ir-fn-eval`

Interprets an IR function with a tuple of typed argument values and prints the
result. Example:

```shell
xlsynth-driver ir-fn-eval my_mod.ir add '(bits[32]:1, bits[32]:2)'
```

### `ir-strip-pos-data`

Reads an `.ir` file and emits the same IR with all position data removed. This drops:

- `file_number` lines (the file table)
- any `pos=[(fileno, line, col), ...]` attributes on nodes

Output is written to **stdout**.

Example:

```shell
xlsynth-driver ir-strip-pos-data input.ir > input.nopos.ir
```

### `g8r-equiv`

Checks two GateFns for functional equivalence using the available engines. A
JSON report is written to **stdout**. The command exits with a non-zero status
if any engine finds a counter-example. Errors are printed to **stderr**.

### `dslx-equiv`

Checks two DSLX functions for functional equivalence. By default it converts both to IR and uses the selected solver/toolchain to prove equivalence. Alternatively, you can provide a tactic script to drive a tactic-based prover flow.

- Positional arguments: `<lhs.x> <rhs.x>`
- Entry-point selection: either `--dslx_top <NAME>` or both `--lhs_dslx_top <NAME>` and `--rhs_dslx_top <NAME>`.
- Search paths: `--dslx_path <P1;P2;...>` and `--dslx_stdlib_path <PATH>`.
- Behavior flags:
  - `--solver <toolchain|bitwuzla|boolector|z3-binary|bitwuzla-binary|boolector-binary>`
  - `--flatten_aggregates=<BOOL>`
  - `--drop_params <CSV>`
  - `--parallelism-strategy <single-threaded|output-bits|input-bit-split>`
  - `--assertion-semantics <ignore|never|same|assume|implies>`
  - `--lhs_fixed_implicit_activation=<BOOL>` / `--rhs_fixed_implicit_activation=<BOOL>`
  - `--assume-enum-in-bound=<BOOL>`
  - `--type_inference_v2=<BOOL>` (requires `--toolchain`)
  - `--lhs_uf <func_name:uf_name>` (may be specified multiple times)
  - `--rhs_uf <func_name:uf_name>` (may be specified multiple times)
  - `--tactic-script <PATH>` Use tactic-based proving driven by a script (JSON array or JSONL of `ScriptStep`). When present, the driver builds a tactic obligation tree and executes it instead of direct equivalence.
  - `--output_json <PATH>`

UF semantics:

- Functions mapped to the same uf_name are treated as the same uninterpreted symbol and are assumed equivalent at call sites.
- Assertions inside uninterpreted functions are ignored during proving.

#### Tactic Scripts

Sometimes two DSLX top functions are equivalent, but **proving that directly** means throwing an SMT solver at the entire design.
That can be slow or even time out, even if only one small helper truly changed.
**Tactic scripts** let you describe a proof plan, and the driver turns the plan into a tree of proof obligations and solves the leaves.

##### Workflow at a glance

1. **Start at `root`**. This is the default "prove `LHS:top ≡ RHS:top`" obligation.
1. **Apply a tactic**. This decomposes an obligation into smaller child obligations. You may further apply tactics on the children to form an obligation tree.
1. **Mark leaves as `Solve`**. We directly prove those checks that are tractable with SMT solvers.
1. **All leaves succeed**. When all leaves succeed, the whole proof succeeds.

Provide the plan with `--tactic-script` as either a JSON array or JSONL (one step per line).

```bash
xlsynth-driver dslx-equiv lhs.x rhs.x \
  --dslx_top main \
  --tactic-script tactic.json \
  --output_json report.json
```

##### The Proof Script

Each step specifies **where** to act and **what** to do:

```json
{
  "selector": ["root", "..."],
  "command": "Solve"
}
```

- `selector`: a path like `["root", "pair_1", "skeleton"]`. Tactics create *named* children, and you refer to them with the names from `root`.
- `command`:
  - `"Solve"`: mark that leaf to be proved by the SMT solver.
  - `{ "Apply": <Tactic> }`: replace that leaf with children (new obligations).

You can supply steps as a JSON array, or stream them as JSONL (one JSON object per line). Blank lines and lines starting with `#` are ignored.

##### Quickstart Example

**Goal**: Prove that the `LHS:top ≡ RHS:top`.

LHS (`lhs.x`):

```dslx
pub fn f1(x: u32) -> u32 { x + u32:1 }

pub fn top(x: u32) -> u32 {
  let y = f1(x);
  y * u32:2 // some heavy computation
}
```

RHS (`rhs.x`):

```dslx
pub fn f1(x: u32) -> u32 { u32:1 + x }  // different body; same semantics

pub fn top(x: u32) -> u32 {
  let y = f1(x);
  y * u32:2 // some heavy computation
}
```

**Tactic**: Use `Focus` and (1) prove `LHS:f1 ≡ RHS:f1` directly, and (2) prove the original tops while treating calls to `f1` as the same **opaque uninterpreted function (UF)**.
This could be effective if the `top` is not changed, performs very heavy arithmetics, while the `f1`s are easy to prove.

**Script (JSONL)**:

```json
{ "selector": ["root"], "command": { "Apply": { "Focus": { "pairs": [ { "lhs": "f1", "rhs": "f1" } ] } } } }
{ "selector": ["root", "pair_1"], "command": "Solve" },
{ "selector": ["root", "skeleton"], "command": "Solve" }
```

This produces the following obligation tree and both leaves can be solved easily.

```text
root
├─ pair_1     (prove LHS:f1 ≡ RHS:f1)
└─ skeleton   (prove LHS:top ≡ RHS:top with calls to f1 treated as the same UF)
```

##### Tactic Reference

###### `Focus`: Prove a few helper pairs; treat them as UFs elsewhere

Input:

```json
{ "Focus": { "pairs": [ { "lhs": "foo", "rhs": "bar" }, { "lhs": "g", "rhs": "h" } ] } }
```

Creates:

- `pair_1`, `pair_2`: direct leaf proofs for each pair.
- `skeleton`: keeps the original top functions, maps each pair to a shared UF so callers don't blow up.

###### `Cosliced` -- Factor both sides into slices plus a composed function

Use when both designs can be expressed as the same composition of smaller pieces.

Left (`lhs.x`):

```dslx
pub fn top (x: u16, y: u16, z: u16) -> u16 {
  let p = x * y;
  specialized_adder(p, z)
}
```

Rhs (`rhs.x`):

```dslx
pub fn top (x: u16, y: u16, z: u16) -> u16 {
  let p = specialized_multiplier(x, y);
  p + z
}
```

Here, it could be the case that the adder and/or the multiplier is too complex so their composition cannot be proven. However, we can refactor the code:

Left refactored:

```dslx
pub fn top (x: u16, y: u16, z: u16) -> u16 {
  let p = x * y;
  specialized_adder(p, z)
}

pub fn slice1(x: u16, y: u16) -> u16 {
  x * y
}

pub fn slice2(p: u16, z: u16) -> u16 {
  specialized_adder(p, z)
}

pub fn composed(x: u16, y: u16, z: u16) -> u16 {
  let p = slice1(x, y);
  slice2(p, z)
}
```

Right can be refactored in a similar way.

To prove that `LHS:top` is equivalent to `RHS:top`, we may prove that

- `slice_1`: `LHS:slice1 ≡ RHS:slice1`
- `slice_2`: `LHS:slice2 ≡ RHS:slice2`
- `lhs_self`: `LHS:top ≡ LHS:composed`
- `rhs_self`: `RHS:top ≡ RHS:composed`
- `skeleton`: assuming `LHS:slice1 ≡ RHS:slice1` and `LHS:slice2 ≡ RHS:slice2` by replacing them as shared uninterpreted function, prove `LHS:composed ≡ RHS:composed`

Here, we can avoid proving the complex composition of adders and multiplers.

With `Cosliced` we can prove the pieces and stitch them. Here the code can be specified by the raw text or the path to the file containing it.

```json
[
  { "selector": ["root"], "command": { "Apply": { "Cosliced": {
    "lhs_slices": [
      { "func_name": "slice1", "code": { "Text": "pub fn slice1(x: u16, y: u16) -> u16 { x * y }" } }
      { "func_name": "slice2", "code": { "Text": "pub fn slice2(p: u16, z: u16) -> u16 { specialized_adder(p, z) }" } }
    ],
    "rhs_slices": [
      { "func_name": "slice1", "code": { "Path": "path_to_rhs_slice1.x" } }
      { "func_name": "slice2", "code": { "Path": "path_to_rhs_slice2.x" } }
    ],
    "lhs_composed": { "func_name": "lhs_comp", "code": { "Text": "pub fn composed(x:u16,y:u16,z:u16)->u16{ let p = slice1(x, y); slice2(p, z) }" } },
    "rhs_composed": { "func_name": "rhs_comp", "code": { "Path": "path_to_rhs_composed.x" } }
  } } } },

  { "selector": ["root","lhs_self"], "command": "Solve" },
  { "selector": ["root","rhs_self"], "command": "Solve" },
  { "selector": ["root","slice_1"],  "command": "Solve" },
  { "selector": ["root","slice_2"],  "command": "Solve" },
  { "selector": ["root","skeleton"], "command": "Solve" }
]
```

Tree shape:

```text
root
├─ slice_1
├─ slice_2
├─ lhs_self
├─ rhs_self
└─ skeleton
```

##### Troubleshooting / pitfalls

- Invalid selector path: ensure each `selector` matches a created node.
- Invalid identifiers: `func_name`, `lhs`, `rhs`, and composed names must be valid identifiers.
- Slice count mismatch: `lhs_slices.len()` must equal `rhs_slices.len()`.
- Forgot to solve the skeleton: many proofs hinge on the `skeleton` leaf.
- JSON vs JSONL: JSONL is one object per line.
- Inline `Text` fragments: ensure names in code match `func_name`.

### `prove-quickcheck`

Proves that DSLX `#[quickcheck]` functions always return true.

- Inputs: `--dslx_input_file <FILE>` plus optional DSLX search paths.
- Filters: `--test_filter <REGEX>` restricts which quickcheck functions are proved.
- Backend: `--solver <...>` selects the solver/toolchain.
- Semantics: `--assertion-semantics <ignore|never|assume>`.
- UF mapping: `--uf <func_name:uf_name>` may be specified multiple times to treat functions as uninterpreted.
- Output: `--output_json <PATH>` to write results as JSON.

UF semantics:

- Functions mapped to the same uf_name are treated as the same uninterpreted symbol and are assumed equivalent at call sites.
- Assertions inside uninterpreted functions are ignored during proving.

### `prover`

Runs a prover plan described by a JSON file with a process-based scheduler.

- Concurrency: `--cores <N>` controls maximum concurrent processes.
- Plan: `--plan_json_file <PATH_OR_->` path to a ProverPlan JSON file, or `-` for stdin.
- Output: `--output_json <PATH>` writes a full JSON report:
  - Top-level `{ "success": <bool>, "plan": <tree> }`.
  - Each task node includes `cmdline`, `outcome`, `stdout`, `stderr`, and `task_id` (when provided).
  - `outcome` is one of `Success`, `Failed`, or an indefinite reason such as `Timeout`, `IndefiniteChildren`, `GroupCriteriaAlreadyMet`, or `Cleanup`.

### Equivalence/proving flags: meanings

- flatten_aggregates: When `true`, tuples and arrays are flattened to plain bit-vectors during equivalence checking. This relaxes type matching so two functions can be considered equivalent even if their aggregate shapes differ, as long as the bit-level behavior matches.

- drop_params: Comma-separated list of parameter names to remove from the function(s) before proving equivalence. The check fails if any dropped parameter is referenced in the function body. Use this to align functions that differ by unused or environment-only parameters.

- assume-enum-in-bound: When `true`, constrains enum-typed parameters to their declared enumerators (domain restriction) during proofs. This is usually desirable because the underlying bit-width can represent more values than the defined enum members. Default is `true` for supported solvers. Supported by native SMT backends (e.g., z3-binary, bitwuzla, boolector) and not by the toolchain or legacy boolector paths; requesting it where unsupported results in an error.

- assertion-semantics: How to treat `assert` statements when proving equivalence. Let r_l/r_r be results and s_l/s_r indicate that no assertion failed on the left/right.

  - ignore: Ignore assertions.
  - never: Both sides must never fail; results must match.
  - same: Both must either fail (both) or succeed with the same result.
  - assume: Assume both sides succeed; only compare results if they do.
  - implies: If the left succeeds, the right must also succeed and match; if the left fails, the right is unconstrained.

## Prover configuration JSON (task-spec DSL)

The driver exposes a small, composable JSON DSL for describing prover tasks, used by programmatic callers and (optionally) config files. It mirrors the command-line flags and subcommands.

- A single task is one object tagged by `kind`.
- Collections of tasks can be composed into a tree using groups with `kind` equal to `all`, `any`, or `first`.

Top-level forms:

- Task: an object with `kind` ∈ {`ir-equiv`, `dslx-equiv`, `prove-quickcheck`} and fields below.
- Group: an object with `kind` ∈ {`all`, `any`, `first`} and `tasks` = array of the same top-level forms (recursive).

Example: single task

```json
{
  "kind": "ir-equiv",
  "lhs_ir_file": "lhs.ir",
  "rhs_ir_file": "rhs.ir",
  "top": "main",
  "solver": "toolchain",
  "parallelism_strategy": "output-bits",
  "assertion_semantics": "same",
  "flatten_aggregates": true,
  "drop_params": ["p0", "p1"],
  "json": true,
  "timeout_ms": 30000,
  "task_id": "my-task-1"
}
```

Example: group composition

```json
{
  "kind": "all",
  "keep_running_till_finish": false,
  "tasks": [
    { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir" },
    { "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "foo" },
    { "kind": "prove-quickcheck", "dslx_input_file": "qc.x" }
  ]
}
```

Groups: all / any / first

- `all`: overall success if and only if all children succeed.
- `any`: overall success if at least one child succeeds.
- `first`: the first finished children dominates the result.

Timeouts

- Any task may specify `"timeout_ms": <milliseconds>`.
- When the deadline elapses, the scheduler cancels the task’s process group and marks the task outcome as `"Timeout"` (an indefinite outcome).
- Group semantics with timeouts (indefinite outcomes):
  - `any`: resolves `Success` as soon as any child succeeds; if all children finish without a success and at least one is indefinite (e.g., `Timeout`), the group resolves to `IndefiniteChildren`.
  - `all`: resolves `Failed` if any child fails; if none failed but at least one is indefinite, resolves to `IndefiniteChildren`; otherwise `Success`.
  - `first`: only the first non-indefinite child determines the result; timeouts do not resolve the group. If all children finish and none produced a definite result, the group resolves to `IndefiniteChildren`.

Task identifiers

- Any task may specify `"task_id": <string>`.
- The `task_id` is echoed into the final report on the corresponding task node to make it easy to correlate results with the original task specification.

Optional group flag

- `keep_running_till_finish` (default `false`):
  By default, the scheduler prunes the sibling tasks when the group result can
  be resolved to accelerate the overall proof.
  This can be turned off by setting `keep_running_till_finish` to `true`.
  In this case, all child tasks continue to run to completion, and the group's outcome is only set after all of its children have finished. If this flag is set on the root group, the prover run will wait for all tasks in the plan to finish before exiting, while the overall success is still determined by the group's semantics.
  This is useful for debugging to diagnose all the tasks without proactively pruning
  them for overall proving speed.

Tree structure example

```json
{
  "kind": "first",
  "keep_running_till_finish": true,
  "tasks": [
    { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir", "top": "main" },
    {
      "kind": "any",
      "keep_running_till_finish": false,
      "tasks": [
        { "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "foo" },
        { "kind": "prove-quickcheck", "dslx_input_file": "qc.x", "test_filter": ".*prop" }
      ]
    }
  ]
}
```

Visual shape

```
first
├─ ir-equiv(lhs.ir, rhs.ir)
└─ any
   ├─ dslx-equiv(lhs.x, rhs.x)
   └─ prove-quickcheck(qc.x)
```

Schema details

- Common conventions

  - Unspecified fields use the same defaults as the CLI.
  - Paths are strings. Arrays of paths use JSON arrays. For DSLX search paths we join paths with `;` internally.
  - Enum fields are lowercase/kebab-case strings as shown below.

- `kind: "ir-equiv"` (IrEquivConfig)

  - Required: `lhs_ir_file` (path), `rhs_ir_file` (path)
  - Entry-point selection: either `top` (string) or both `lhs_ir_top` and `rhs_ir_top` (strings)
  - Optional:
    - `solver`: one of `toolchain`, `bitwuzla`, `boolector`, `z3-binary`, `bitwuzla-binary`, `boolector-binary` (availability gated by build features)
    - `flatten_aggregates`: bool
    - `drop_params`: array of strings (joined with commas for the CLI)
    - `parallelism_strategy`: one of `single-threaded`, `output-bits`, `input-bit-split`
    - `assertion_semantics`: one of `ignore`, `never`, `same`, `assume`, `implies`
    - `lhs_fixed_implicit_activation`: bool
    - `rhs_fixed_implicit_activation`: bool
    - `json`: bool
    - `timeout_ms`: integer (milliseconds) — optional per-task timeout

- `kind: "dslx-equiv"` (DslxEquivConfig)

  - Required: `lhs_dslx_file` (path), `rhs_dslx_file` (path)
  - Entry-point selection: either `dslx_top` (string) or both `lhs_dslx_top` and `rhs_dslx_top` (strings)
  - DSLX search paths:
    - `dslx_path`: array of paths (joined with `;`)
    - `dslx_stdlib_path`: path
  - Optional behavior flags:
    - `solver`: same values as above
    - `flatten_aggregates`: bool
    - `drop_params`: array of strings
    - `parallelism_strategy`: `single-threaded` | `output-bits` | `input-bit-split`
    - `assertion_semantics`: `ignore` | `never` | `same` | `assume` | `implies`
    - `lhs_fixed_implicit_activation`: bool
    - `rhs_fixed_implicit_activation`: bool
    - `assume_enum_in_bound`: bool
    - `type_inference_v2`: bool (requires external toolchain)
    - `lhs_uf`: array of strings, each "`<func_name>:<uf_name>`" (repeats map to repeated CLI flags). Functions sharing the same `uf_name` are assumed equivalent; assertions inside them are ignored.
    - `rhs_uf`: array of strings, each "`<func_name>:<uf_name>`". Same semantics as above.
    - `json`: bool
    - `timeout_ms`: integer (milliseconds) — optional per-task timeout

- `kind: "prove-quickcheck"` (ProveQuickcheckConfig)

  - Required: `dslx_input_file` (path)
  - Optional:
    - `test_filter`: string (regex)
    - `solver`: same values as above
    - `assertion_semantics`: `ignore` | `never` | `assume`
    - `uf`: array of strings, each "`<func_name>:<uf_name>`". Functions sharing the same `uf_name` are assumed equivalent; assertions inside them are ignored.
    - `json`: bool
    - `timeout_ms`: integer (milliseconds) — optional per-task timeout

Mapping to CLI

Each task translates 1:1 to an `xlsynth-driver` subcommand invocation. The JSON above for `ir-equiv` maps to:

```shell
xlsynth-driver ir-equiv lhs.ir rhs.ir \
  --top main \
  --solver toolchain \
  --flatten_aggregates true \
  --drop_params p0,p1 \
  --parallelism-strategy output-bits \
  --assertion-semantics same \
  --json true
```

Notes

- Enum values are case-insensitive on the CLI but serialized in lowercase/kebab-case in JSON.
- `type_inference_v2` is only honored when using the external toolchain (`--toolchain`).
- `dslx_path` is joined with `;` regardless of platform to match upstream tools.

### `dslx-stitch-pipeline`: Stitch DSLX pipeline stages

Takes a collection of `*_cycleN` pipeline‐stage functions in a DSLX file (e.g. `foo_cycle0`, `foo_cycle1`, …) and:

1. Generates Verilog/SystemVerilog for **each** stage function.
1. Emits a wrapper module named `<top>_pipeline` that instantiates the stages and wires them together to form the complete pipeline.

The generated text is written to **stdout**; diagnostic messages appear on **stderr**.

Supported flags:

- `--use_system_verilog=<BOOL>` – emit SystemVerilog when `true` *(default)* or plain Verilog when `false`.
- `--stages=<CSV>` – comma-separated list of stage function names that determines the pipeline order (overrides the default discovery of `<top>_cycleN` functions).

The usual DSLX-related options (`--dslx_input_file`, `--dslx_top`, `--dslx_path`, `--dslx_stdlib_path`, `--warnings_as_errors`) are also accepted.

Additional semantics:

- `--dslx_top=<NAME>` specifies the *logical* pipeline prefix. Stage
  functions are expected to be named `<NAME>_cycle0`, `<NAME>_cycle1`, … (or
  be provided explicitly via `--stages`). A DSLX function named exactly
  `<NAME>` is **ignored** by this command – only the `_cycleN` stage functions
  participate in stitching. When `--stages` is supplied the prefix is only
  used for the wrapper module name and **not** for stage discovery.

Example:

```shell
xlsynth-driver dslx-stitch-pipeline \
  --dslx_input_file my_design.x \
  --dslx_top foo \
  --stages=foo_cycle0,foo_cycle1,foo_cycle2 > foo_pipeline.sv
```

### `run-verilog-pipeline` *(experimental)*

Runs a synthesized *pipelined* SystemVerilog module through a throw-away, automatically-generated test-bench and prints the value(s) that appear on the data output port(s).

> **Experimental:** This command is a thin wrapper that glues together three separate external facilities – on-the-fly test-bench generation, [`slang`](https://github.com/MikePopoloski/slang) for Verilog/SV parsing, and the `iverilog` + `vvp` simulator pair. It exists purely to *kick the tires* on freshly generated pipelines. **Do not** rely on it for rigorous or long-running verification.
>
> Internally it expects:
>
> 1. One **or more** data input ports (plus optional handshake/reset/clock). When there are several, supply a tuple value on the CLI that matches the port order.
> 1. A free-running clock named `clk` – this port **must** be present in the top‐level module.
> 1. The pipeline source text provided either via **stdin** or as a positional file argument.

Basic usage (latency known a-priori):

```shell
# Create a 1-stage pipeline and immediately simulate it with x = 5
xlsynth-driver dslx2pipeline my_module.x main \
  --pipeline_stages=1 --delay_model=asap7 | \
  xlsynth-driver run-verilog-pipeline --latency=1 bits[32]:5
# Prints:  out: bits[32]:6
```

`run-verilog-pipeline` accepts the SystemVerilog text either **via `stdin`** (pass `-`) or by specifying a *file path* as a second positional argument.

If the pipeline uses *valid* handshake signals the latency can be discovered automatically:

```shell
# Reading Verilog from a file
xlsynth-driver run-verilog-pipeline \
  --input_valid_signal=in_valid \
  --output_valid_signal=out_valid \
  --reset=rst \
  --reset_active_low=false \
  bits[32]:5  pipeline.sv

# Equivalent stdin form
cat pipeline.sv | xlsynth-driver run-verilog-pipeline \
  --input_valid_signal=in_valid \
  --output_valid_signal=out_valid \
  --reset=rst \
  --reset_active_low=false \
  bits[32]:5 -
```

Key flags:

- `--input_valid_signal=<NAME>` Name of the *input-valid* handshake port.
- `--output_valid_signal=<NAME>` Name of the *output-valid* handshake port. If omitted you **must** specify `--latency`.
- `--latency=<CYCLES>` Pipeline latency in cycles when no output-valid handshake is present.
- `--reset=<NAME>` Optional reset signal name; defaults to none.
- `--reset_active_low` Treat the reset as active-low (default is active-high).
- `--waves=<PATH>` Write a VCD dump of the simulation to `PATH`.

Reset sequencing:

When a `--reset` signal is provided the generated test-bench:

1. Drives the reset **active** (respecting `--reset_active_low`) for two rising edges of `clk`.
1. De-asserts the reset and waits one negative edge before applying data inputs / `input_valid`.

This guarantees that the design observes at least one full cycle of reset before valid stimulus arrives.

The positional argument `<INPUT_VALUE>` is an *XLS IR* typed value. For modules with **multiple** data input ports supply a *tuple* whose order matches the port list.

Example with two data inputs (`a`, `b`) each 32-bits wide:

```shell
# Suppose `pipeline.sv` has ports:  clk, a, b, out
xlsynth-driver run-verilog-pipeline --latency=1 '(bits[32]:5, bits[32]:17)' pipeline.sv
# Prints lines like:
#  out: bits[32]:22
```

On success the command prints one line per data output:

```
<port_name>: bits[W]:<VALUE>
```

making it easy to splice into shell pipelines or test scripts.

## Toolchain configuration (`xlsynth-toolchain.toml`)

Several subcommands accept a `--toolchain` option that points at a
`xlsynth-toolchain.toml` file. The file *must* define a top-level
`[toolchain]` table and can contain **nested** tables for DSLX- and
code-generation-specific settings:

- `[toolchain]` | `tool_path` | Directory containing the XLS tools (`codegen_main`, `opt_main`, …). |
- `[toolchain.dslx]` | `type_inference_v2` | Enables the experimental type-inference-v2 algorithm globally unless overridden by a CLI flag. |
  | | `dslx_stdlib_path` | Path to the DSLX standard library. |
  | | `dslx_path` | *Array* of additional DSLX search paths. |
  | | `warnings_as_errors` | Treat DSLX warnings as hard errors. |
  | | `enable_warnings` / `disable_warnings`| Lists of DSLX warning names to enable / suppress. |
  | `[toolchain.codegen]` | `gate_format` | Template string used for `gate!` macro expansion. |
  | | `assert_format` | Template string used for `assert!` macro expansion. |
  | | `use_system_verilog` | Emit SystemVerilog instead of plain Verilog. |

Only the fields you need must be present. When invoked with
`--toolchain <FILE>` the driver uses these values as defaults for the
corresponding command-line flags.

Example:

```toml
[toolchain]
tool_path = "/path/to/xls/tools"

[toolchain.dslx]
type_inference_v2 = true
dslx_stdlib_path = "/path/to/dslx/stdlib"
dslx_path         = ["/path/to/extra1", "/path/to/extra2"]
warnings_as_errors = true
enable_warnings    = ["foo", "bar"]
disable_warnings   = ["baz"]

[toolchain.codegen]
gate_format        = "br_gate_buf gated_{output}(.in({input}), .out({output}))"
assert_format      = "`BR_ASSERT({label}, {condition})"
use_system_verilog = true
```

## Experimental `--type_inference_v2` Flag

Some subcommands support an experimental flag:

```
--type_inference_v2
```

This flag enables the experimental type inference v2 algorithm for DSLX-to-IR and related conversions.
**It is only supported when using the external toolchain (`--toolchain`).**
If you request this flag without `--toolchain`, the driver will print an error and exit.

### Supported Subcommands

| Subcommand | Supports `--type_inference_v2`? | Requires `--toolchain` for TIv2? | Runtime API allowed without TIv2? |
|--------------------|:-------------------------------:|:-------------------------------:|:---------------------------------:|
| `dslx2pipeline` | Yes | Yes | Yes |
| `dslx2ir` | Yes | Yes | Yes |
| `dslx-g8r-stats` | Yes | Yes | Yes |
| `dslx2sv-types` | No | N/A | Yes |

### Migration and Use

The main benefit of this flag is that it enables an attempt at migrating `.x` files with **no associated source text changes** (e.g., that would change the position metadata in the resulting IR file).

> **Note:**
> This flag may be short-lived, as it will likely become the default mode when TIv1 is deleted.
> However, it may assist with migration testing and validation during the transition period.

**How to use:**

```shell
xlsynth-driver --toolchain=path/to/xlsynth-toolchain.toml dslx2ir \
  --dslx_input_file my_module.x \
  --dslx_top main \
  --type_inference_v2=true
```
