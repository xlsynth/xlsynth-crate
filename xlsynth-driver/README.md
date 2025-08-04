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
- `--solver <toolchain|bitwuzla|boolector|boolector-legacy|z3-binary|bitwuzla-binary|boolector-binary>`
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

### `dslx2sv-types`: DSLX type definitions to SystemVerilog

Generates SystemVerilog type declarations for the definitions in a DSLX file.
The output is written to **stdout**.

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

### `ir2delayinfo`

Runs the `delay_info_main` tool for a given IR entry point and delay model.
The produced delay-information proto text is written to **stdout**; any tool
diagnostics appear on **stderr**.

### `ir-ged`

Computes the Graph-Edit-Distance between two IR functions. Without further
flags a summary line like `Distance: N` is printed on **stdout**. With
`--json=true` the result is emitted as JSON.

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

### `g8r-equiv`

Checks two GateFns for functional equivalence using the available engines. A
JSON report is written to **stdout**. The command exits with a non-zero status
if any engine finds a counter-example. Errors are printed to **stderr**.

### `dslx-equiv`

Checks two DSLX functions for functional equivalence. By default it converts both to IR and uses the selected solver/toolchain to prove equivalence.

- Positional arguments: `<lhs.x> <rhs.x>`
- Entry-point selection: either `--dslx_top <NAME>` or both `--lhs_dslx_top <NAME>` and `--rhs_dslx_top <NAME>`.
- Search paths: `--dslx_path <P1;P2;...>` and `--dslx_stdlib_path <PATH>`.
- Behavior flags:
  - `--solver <toolchain|bitwuzla|boolector|boolector-legacy|z3-binary|bitwuzla-binary|boolector-binary>`
  - `--flatten_aggregates=<BOOL>`
  - `--drop_params <CSV>`
  - `--parallelism-strategy <single-threaded|output-bits|input-bit-split>`
  - `--assertion-semantics <ignore|never|same|assume|implies>`
  - `--lhs_fixed_implicit_activation=<BOOL>` / `--rhs_fixed_implicit_activation=<BOOL>`
  - `--assume-enum-in-bound=<BOOL>`
  - `--type_inference_v2=<BOOL>` (requires `--toolchain`)
  - `--output_json <PATH>`

### `prove-quickcheck`

Proves that DSLX `#[quickcheck]` functions always return true.

- Inputs: `--dslx_input_file <FILE>` plus optional DSLX search paths.
- Filters: `--test_filter <REGEX>` restricts which quickcheck functions are proved.
- Backend: `--solver <...>` selects the solver/toolchain.
- Semantics: `--assertion-semantics <ignore|never|assume>`.
- Output: `--output_json <PATH>` to write results as JSON.

### `prover`

Runs a prover plan described by a JSON file with a process-based scheduler.

- Concurrency: `--cores <N>` controls maximum concurrent processes.
- Plan: `--plan_json_file <PATH_OR_->` path to a ProverPlan JSON file, or `-` for stdin.
- Output: `--output_json <PATH>` writes `{ "success": <bool> }`.

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
  "json": true
}
```

Example: group composition

```json
{
  "kind": "all",
  "tasks": [
    { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir" },
    { "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "foo" },
    { "kind": "prove-quickcheck", "dslx_input_file": "qc.x" }
  ]
}
```

Groups: all / any / first

- `all`: run every child task; overall success if and only if all children succeed.
- `any`: overall success if at least one child succeeds; schedulers may stop at the first success.
- `first`: evaluate children in order; stop at the first child that completes successfully; if none succeed, the overall result is failure.

Tree structure example

```json
{
  "kind": "first",
  "tasks": [
    { "kind": "ir-equiv", "lhs_ir_file": "lhs.ir", "rhs_ir_file": "rhs.ir", "top": "main" },
    {
      "kind": "any",
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
    - `solver`: one of `toolchain`, `bitwuzla`, `boolector`, `boolector-legacy`, `z3-binary`, `bitwuzla-binary`, `boolector-binary` (availability gated by build features)
    - `flatten_aggregates`: bool
    - `drop_params`: array of strings (joined with commas for the CLI)
    - `parallelism_strategy`: one of `single-threaded`, `output-bits`, `input-bit-split`
    - `assertion_semantics`: one of `ignore`, `never`, `same`, `assume`, `implies`
    - `lhs_fixed_implicit_activation`: bool
    - `rhs_fixed_implicit_activation`: bool
    - `json`: bool

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
    - `json`: bool

- `kind: "prove-quickcheck"` (ProveQuickcheckConfig)

  - Required: `dslx_input_file` (path)
  - Optional:
    - `test_filter`: string (regex)
    - `solver`: same values as above
    - `assertion_semantics`: `ignore` | `never` | `assume`
    - `json`: bool

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
