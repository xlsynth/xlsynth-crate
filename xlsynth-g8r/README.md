## `xlsynth-g8r`: gate-level infrastructure

`xlsynth-g8r` hosts the gate-level side of the xlsynth stack:

- **`aig`**: core AIG/GateFn representation and structural transforms (fraig, balancing, etc.).
- **`aig_serdes`**: (de)serialization to/from AIGER and a textual gate format.
- **`aig_sim`**: scalar and SIMD gate-level simulators.
- **`liberty` / `liberty_proto`**: Liberty parsing, indexing, and proto bindings.
- **`netlist`**: Verilog-like gate-level netlist parsing, connectivity, cone traversal, and GV→IR.
- **`transforms`**: local gate-level rewrite passes used by optimization and MCMC logic.

Most functionality is exposed via the `xlsynth_g8r` library and thin binaries under `src/bin/`.

## Additional docs

- `docs/g8r_lib_timing_design.md`: why Liberty loading has `Library` (no timing) vs `LibraryWithTimingData` and the observed ASAP7 size/load tradeoffs.

## External Yosys APIs

`netlist::yosys::YosysToolchain` validates an executable and runs scripts with
explicit timeouts, captured diagnostics, and process-group cleanup. It needs no
Liberty files; `from_env()` uses `XLSYNTH_YOSYS_PATH` or `yosys` on `PATH`.

`YosysEnvironment` adds validated Liberty file paths for technology mapping.
`synthesize_to_gv` selects `YosysInputLanguage::{Verilog, SystemVerilog}` and
`YosysMappingKind::{Combinational, Sequential}` explicitly. The older convenience
methods retain their existing frontend and mapping behavior.

`YosysMappingContext` loads cell semantics from the same files and offers a
`preflight` mapping/import probe. Its environment setup requires
`XLSYNTH_YOSYS_PATH` and comma-separated `XLSYNTH_LIBERTY_FILES` pointing to
installed, compatible libraries. Sequential preflight requires flip-flop cells.
No library helper prints, caches global state, or decides whether a failure
should stop a fuzz campaign; callers own that policy. Execution errors preserve
`xlsynth::external_tool::ToolError` timeout/resource categories.

## Netlist parse benchmark

Run the synthetic netlist-parse microbenchmark with:

```shell
cargo bench -p xlsynth-g8r --bench netlist_parse_bench -- --verbose
```
