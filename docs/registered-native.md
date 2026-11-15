# Running the registered native mapper

Use `xlsynth-driver g8r2gv` to map a G8R design and write its final sized gates. The command runs the optimization in-process. The mapper selects library cells, attaches the physical registers, and optimizes the assembled stage. It then sizes the selected netlist.

## Map a circuit with boundary registers

In this example `input.g8rbin` contains the circuit to map, `cells.timing.proto` defines its timing library. Replace `clk` with the desired clock port name.

```sh
xlsynth-driver g8r2gv input.g8rbin \
  --add-clk-port clk --flop-inputs --flop-outputs \
  --module-name top --mapper native \
  --liberty-proto cells.timing.proto \
  --cleanup-registers --clock-gating=false \
  -o result.gv
```

The boundary flags create launch and capture registers. The example enables cleanup of redundant registers and disables clock gating. The command writes the mapped and sized gate-level netlist. Use `--adaptive-resize-report run.json` to record sizing decisions and intermediate gate hashes.

## Cell-library inputs

The timing proto supplies the legal cells, area, pin capacitance and timing arcs. Use `--resize-liberty-proto` when a separate compatible library supplies the sizing view.

## What runs

| Step | Search and stopping rule |
| --- | --- |
| Registered mapping | Compare native delay covers with physical registers attached. Use timing-cone candidates and capacitance-balanced buffer trees. Preserve electrical checks during optimization. During integrated area recovery, try the predicted fastest minimum-area cell first in the eight-cell shortlist. |
| Area recovery | Freeze registers. Try two downsizing rounds per iteration with no timing rounds and no area growth. Stop when the emitted netlist is unchanged or after seven iterations. |
| Timing and area optimization | Each iteration allows ten timing rounds and four area-recovery rounds, with at most 2% area growth in that iteration. Try timing candidates in delay bands, fastest first. Stop on an unchanged netlist. After iteration 23, continue up to iteration 46 only if the positive register-to-register delay has fallen by at least 1% since iteration 12. |

The two repeated sizing steps retain the tightest delay reached at each boundary when renewing their area-recovery search budget. Mapping and finishing use the same physical I/O constraints. The defaults are an input transition of 0.01 in the mapping library's time units, its default output load and zero additional wire capacitance. When a separate sizing library declares different units, the driver converts the input transition and output load before sizing. The final registers are fixed during both finishing steps.

The [driver reference](../xlsynth-driver/README.md#g8r2gv-g8r-to-mapped-gate-level-netlist) describes the mapping and sizing options.
