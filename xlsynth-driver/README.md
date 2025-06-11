# `xlsynth-driver` command line interface

The `xlsynth-driver` binary is a "driver program" for various XLS/xlsynth tools and functionality
behind a single unified command line interface. It is organized into subcommands.

## Subcommands

### `ir-equiv`

Proves two IR functions to be equivalent or provides a counterexample to their equivalence.

### `lib2proto`: liberty files to proto

Liberty files can be unweildy and large in their textual form -- this command reformats the data
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

### `ir2g8r`: IR to gate-level representation

Converts an XLS IR file to an `xlsynth_g8r::GateFn` (i.e. a gate-level netlist in AIG form).

- By default the pretty-printed GateFn is sent to **stdout**.
- Additional artifacts can be emitted with flags:
  - `--bin-out <PATH>` – write the GateFn as a binary **.g8rbin** file (bincode-encoded).
  - `--stats-out <PATH>` – write a JSON summary of structural statistics.
- The same optimization / analysis flags accepted by `ir2gates` are supported (`--fold`, `--hash`, `--fraig`, `--toggle-sample-count`, …).

Example:

```shell
xlsynth-driver ir2g8r my_module.opt.ir \
  --fraig=true \
  --bin-out my_module.g8rbin \
  --stats-out my_module.stats.json > my_module.g8r
```

The command above leaves three artefacts:

1. `my_module.g8r`   – human-readable GateFn (stdout redirection).
1. `my_module.g8rbin` – compact bincode serialisation of the same GateFn.
1. `my_module.stats.json` – structural summary statistics as JSON.

### `g8r2v`: GateFn to gate-level netlist (Verilog-like)

Converts a `.g8r` (text) or `.g8rbin` (bincode) file containing a gate-level `GateFn` to a `.ugv` netlist (human-readable, Verilog-like) on **stdout**.

- By default, emits the netlist with the original GateFn inputs.
- The `--add-clk-port[=NAME]` flag inserts an (unused) clock port as the first input:
  - If omitted: no clock port is added.
  - If given as `--add-clk-port` (no value): adds a port named `clk`.
  - If given as `--add-clk-port=foo`: adds a port named `foo`.

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

### `version`

Prints the driver version string to **stdout**.

### `dslx2pipeline`: DSLX to pipelined Verilog

Translates a DSLX entry point to a pipelined SystemVerilog module.
The resulting Verilog is printed on **stdout**.
Diagnostic messages and the path to temporary files (when
`--keep_temps=true`) are written to **stderr**.

- The `--type_inference_v2` flag enables the experimental type inference v2 algorithm.
  **Requires:** `--toolchain` (external tool path). If used without `--toolchain`, the driver will print an error and exit.

### `dslx2ir`: DSLX to IR

Converts DSLX source code to the XLS IR. The IR text is emitted on **stdout**.
DSLX warnings and errors appear on **stderr**.

- The `--type_inference_v2` flag enables the experimental type inference v2 algorithm.
  **Requires:** `--toolchain` (external tool path). If used without `--toolchain`, the driver will print an error and exit.

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

### `ir2pipeline`: IR to pipelined Verilog

Produces a pipelined SystemVerilog design from an IR file. The generated code
is printed to **stdout**. When `--keep_temps=true` the location of temporary
files is reported on **stderr**.

### `ir2combo`: IR to *combinational* SystemVerilog

Similar to `ir2pipeline` but requests the *combinational* backend in `codegen_main`.
Generates a single‐cycle (no pipeline registers) SystemVerilog module on **stdout**.
All the usual code-gen flags (e.g., `--use_system_verilog`, `--add_invariant_assertions`,
`--flop_inputs`, `--flop_outputs`, etc.) are supported.

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

Computes the Graph-Edit-Distance between two IR functions.  Without further
flags a summary line like `Distance: N` is printed on **stdout**.  With
`--json=true` the result is emitted as JSON.

### `ir2gates`: IR to GateFn statistics

Maps an IR function to a gate-level representation and prints structural
statistics.  With `--quiet=true` a compact JSON summary is produced; otherwise
the report is human-readable text.

### `ir-fn-eval`

Interprets an IR function with a tuple of typed argument values and prints the
result. Example:

```shell
xlsynth-driver ir-fn-eval my_mod.ir add '(bits[32]:1, bits[32]:2)'
```

### `g8r-equiv`

Checks two GateFns for functional equivalence using the available engines.  A
JSON report is written to **stdout**.  The command exits with a non-zero status
if any engine finds a counter-example.  Errors are printed to **stderr**.

## Toolchain configuration (`xlsynth-toolchain.toml`)

Several subcommands accept a `--toolchain` option that points at a
`xlsynth-toolchain.toml` file.  This TOML file houses a `[toolchain]` table that
specifies where the driver can find external XLS resources and supplies default
values for certain flags.

Supported fields:

| Key | Purpose |
|-----|---------|
| `dslx_stdlib_path` | Path to the DSLX standard library. |
| `dslx_path` | Array of extra search paths for DSLX modules. |
| `tool_path` | Directory containing the XLS tools (`codegen_main`, `opt_main`, …). |
| `warnings_as_errors` | Treat DSLX warnings as hard errors. |
| `enable_warnings` / `disable_warnings` | Lists of DSLX warning names to enable / suppress. |
| `gate_format` | Template string used for `gate!` macro expansion. |
| `assert_format` | Template string used for `assert!` macro expansion. |
| `use_system_verilog` | Emit SystemVerilog rather than plain Verilog. |

Only the fields you need must be present.  When invoked with
`--toolchain <FILE>` the driver uses these values as defaults for the
corresponding command-line flags.

## Experimental `--type_inference_v2` Flag

Some subcommands support an experimental flag:

```
--type_inference_v2
```

This flag enables the experimental type inference v2 algorithm for DSLX-to-IR and related conversions.
**It is only supported when using the external toolchain (`--toolchain`).**
If you request this flag without `--toolchain`, the driver will print an error and exit.

### Supported Subcommands

| Subcommand         | Supports `--type_inference_v2`? | Requires `--toolchain` for TIv2? | Runtime API allowed without TIv2? |
|--------------------|:-------------------------------:|:-------------------------------:|:---------------------------------:|
| `dslx2pipeline`    | Yes                             | Yes                             | Yes                              |
| `dslx2ir`          | Yes                             | Yes                             | Yes                              |
| `dslx-g8r-stats`   | Yes                             | Yes                             | Yes                              |
| `dslx2sv-types`    | No                              | N/A                             | Yes                              |

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
