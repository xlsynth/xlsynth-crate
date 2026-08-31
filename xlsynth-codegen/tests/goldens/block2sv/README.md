# Block-codegen golden fixtures

This intentionally small, independently authored suite contains valid block
IR, declarative codegen options, and exact expected library results. It keeps
textual contracts reviewable without trying to snapshot every supported
operation. Broad correctness belongs in focused tests, iverilog/Yosys checks,
and fuzzing. Tests run inside `xlsynth-codegen`, without a driver or codegen
binary.

```text
package example

top block identity(x: bits[8], y: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}

// OPTIONS: top = "identity"
// EXPECT-SV: module identity(
// EXPECT-SV:   input logic [7:0] x,
// EXPECT-SV:   output logic [7:0] y
// EXPECT-SV: );
// EXPECT-SV:   assign y = x;
// EXPECT-SV: endmodule
```

Successive `OPTIONS` lines form one TOML document deserialized into
`BlockCodegenOptions`. Omitted settings use the library defaults. Unknown
top-level fields, duplicate keys, invalid types, and invalid layouts fail
fixture parsing. There is no CLI parsing or auxiliary-file lookup.

Register configuration is embedded rather than loaded through a CLI path:

```text
// OPTIONS: [register_codegen_options]
// OPTIONS: reg_template = "always_ff @(posedge {{clock}}) {{reg}} <= {{next}};"
```

Use `EXPECT-ERROR` instead of `EXPECT-SV` for backend errors. Expected errors
contain the library message, without the CLI's `error: ` prefix. Mixing success
and error expectations is invalid. Exactly one optional space after the colon
is removed; output indentation and blank lines are significant. Updates
preserve ordinary IR comments and options.

Run and regenerate native fixtures from the repository root:

```sh
cargo test -p xlsynth-codegen --test block2sv_goldens
XLSYNTH_UPDATE_GOLDEN=1 cargo test -p xlsynth-codegen \
  --test block2sv_goldens block2sv_golden_fixtures -- --exact
```

Update mode still fails on unexpected codegen success/failure; it only updates
expected text. Review the diff. Standalone `.svtxt` snapshots in `tests/testdata`
also honor this variable, preserve SPDX headers, and run in the Icarus suite:

```sh
XLSYNTH_UPDATE_GOLDEN=1 cargo test -p xlsynth-codegen --features iverilog-tests
```

## External validation

```sh
cargo test -p xlsynth-codegen --features iverilog-tests --test block2sv_iverilog
cargo test -p xlsynth-codegen --features yosys-tests --test block2sv_yosys
```

Both features are off by default. Selected suites require their tools and fail
when tools are missing. iverilog checks compilation, outputs, hierarchy,
register templates, and reset/enable behavior. Yosys consumes generated
SystemVerilog.
Random-block fuzzing also checks Yosys/ABC → mapped
netlist → netlist evaluation; see the crate's fuzz guide.

Driver CLI tests stay in `xlsynth-driver/tests/block2sv_cli.rs` and cover flags,
aliases, configuration file loading, exit codes, and diagnostic formatting.
