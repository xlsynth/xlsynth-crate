# xlsynth-vastly

`xlsynth-vastly` simulates Verilog/SystemVerilog that XLS codegen emits, with
support for both combinational (`combo`) and pipelined designs over the
constructs we target. We ground correctness with fuzzing over XLS-generated
artifacts in both forms, and with differential checks against external
reference implementations such as semantic VCD diffing against an external
simulator backend.

The language standards are the semantic source of truth here. External
simulators are used as independent implementations to cross-check against the
spec, not as spec providers themselves.

## Scope and Trust Model

`xlsynth-vastly` is a deliberately limited simulator intended for testing and
debugging the subset of Verilog/SystemVerilog emitted by XLS and VAST.

It is not designed or intended to be a general-purpose Verilog simulator. The
implementation makes a best-effort attempt to reflect the Verilog/SystemVerilog
specification, but any trust in its behavior should be limited to the language
subset exercised by generated XLS/VAST outputs and the surrounding regression
and differential-test coverage.

In other words: if `xlsynth-vastly` accepts and correctly simulates broader
source-language inputs, that is useful, but it is not the primary contract of
the crate.

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

- Compare combo simulation results against a reference simulator:

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd \
  --compare-to-reference-sim=iverilog
```

- Run baseline tests that do not require external simulators:

```bash
cargo test -p xlsynth-vastly
```

- Run extended reference-simulator tests:

```bash
cargo test -p xlsynth-vastly --features reference-sim-tests
```

## Tooling Requirements

- Core unit/integration tests do not require external simulators.
- Reference-simulator tests require the current external simulator backend to
  be available on `PATH`; enable them with `--features reference-sim-tests`.
- `reference-sim-tests` exists so default `cargo test` remains
  tool-independent while still providing an explicit opt-in lane for
  differential checks against external implementations.
- The reference-simulator layer is intentionally pluggable so additional
  implementations can be slotted in later, including backends with only
  two-value semantics such as Yosys/CXXRTL.

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

### Compare against a reference simulator

If the current reference-simulator backend is on `PATH`, generate a reference
VCD and semantically diff:

```bash
cargo run -p xlsynth-vastly --bin vastly-sim-combo -- /path/to/foo.combo.v \
  --inputs "0xf00,0xba5;0x0000,0x3f80" \
  --vcd-out ./combo.vcd \
  --compare-to-reference-sim=iverilog
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

- Baseline tests (no external-simulator dependency):
  `cargo test -p xlsynth-vastly`
- Extended reference-simulator tests:
  `cargo test -p xlsynth-vastly --features reference-sim-tests`
- Fuzz targets live under `xlsynth-vastly/fuzz`; see workspace `FUZZ.md`.
