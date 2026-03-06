# xlsynth-vastly

`xlsynth-vastly` simulates Verilog/SystemVerilog that XLS codegen emits, with
support for both combinational (`combo`) and pipelined designs over the
constructs we target. We ground correctness with fuzzing over XLS-generated
artifacts in both forms, and with oracle-style comparisons such as semantic VCD
diffing against `iverilog`/`vvp`.

## Quickstart Commands

Run these from the workspace root.

- Run combinational simulation and emit a VCD (best when debugging pure combo logic):

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd
```

- Run pipeline/clocked simulation and emit a VCD (best when validating cycle-by-cycle behavior):

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-pipeline -- /path/to/foo.sv \
  --inputs "0xf00,0xba5@2;0x0000,0xf800@3" \
  --half-period 5 \
  --vcd-out ./pipeline.vcd
```

- Compare combo simulation results with Icarus Verilog (best as a semantic oracle check):

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd \
  --compare-to-iverilog
```

- Run baseline tests that do not require external simulators:

```bash
cargo test -p xlsynth-vastly
```

- Run extended oracle tests that use `iverilog`/`vvp`:

```bash
cargo test -p xlsynth-vastly --features iverilog-tests
```

## Tooling Requirements

- Core unit/integration tests do not require `iverilog`/`vvp`.
- Oracle/differential tests against Icarus Verilog require `iverilog` and `vvp`
  on `PATH`, and are enabled with `--features iverilog-tests`.
- `iverilog-tests` exists so default `cargo test` remains tool-independent,
  while still providing an explicit opt-in lane for oracle-backed comparisons.
- In `iverilog-tests` mode, `iverilog`/`vvp` are only invoked as subprocesses to
  produce oracle outputs; the oracle backend is intentionally pluggable so
  additional simulators can be slotted in later (e.g. for further SV support
  comparisons).

## Stimulus Format

CLI stimuli are expressed as XLS-style IR value text. In practice this means:

- `--inputs` accepts scalar literal text values (`0x...`, decimal, etc.) in
  positional order.
- `vastly-sim-pipeline` can also read cycle-by-cycle stimuli from an XLS
  `.irvals` text file via `--input-irvals`.

## Command Details

## `vastly-sim-combo`

Run a combinational `*.combo.v` with one-or-more input vectors and dump a VCD.

### Usage

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd
```

By default, `vastly-sim-combo` prints top-level output port values. To disable:

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd \
  --print-outputs=false
```

- `--inputs`: semicolon-separated vectors, each vector is comma-separated values.
- Values may be hex (`0x...`) or decimal.
- Values map positionally to module input ports in module-header order.

### Compare against Icarus Verilog

If `iverilog`/`vvp` are on `PATH`, generate an Icarus VCD and semantically diff:

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd \
  --compare-to-iverilog
```

## `vastly-sim-pipeline`

Run a clocked SV module with a standard clock toggle stimulus. Inputs are
scheduled vectors applied on a given cycle index; unspecified cycles drive
zeros on all non-clock inputs.

### Usage

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-pipeline -- /path/to/foo.sv \
  --inputs "0xf00,0xba5@2;0x0000,0xf800@3" \
  --half-period 5 \
  --vcd-out ./pipeline.vcd
```

- `--inputs`: semicolon-separated entries of the form
  `<comma-separated values>@<cycle_index>`.
- Values map positionally to non-clock input ports in module-header order.
- `@N` schedules the vector for cycle `N` (while `clk` is low, before posedge).
- If `--cycles` is omitted, run length is inferred as `max_scheduled_tick + 1`.

## Testing

- Baseline tests (no Icarus dependency):
  `cargo test -p xlsynth-vastly`
- Extended oracle tests (requires `iverilog`/`vvp`):
  `cargo test -p xlsynth-vastly --features iverilog-tests`
- Fuzz targets live under `xlsynth-vastly/fuzz`; see workspace `FUZZ.md`.
