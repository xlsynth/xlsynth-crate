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
