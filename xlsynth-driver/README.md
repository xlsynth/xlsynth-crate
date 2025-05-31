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
- The same optimisation / analysis flags accepted by `ir2gates` are supported (`--fold`, `--hash`, `--fraig`, `--toggle-sample-count`, …).

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
