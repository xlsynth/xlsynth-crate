# Isolated frontend/backend QoR experiments

`scripts/run_frozen_backend_qor.py` compares frontend or backend revisions
without invoking Yosys. It requires Python 3.9 or newer. Keep experiment
artifacts and technology-specific configuration outside the repository.

## Boundary and controls

The frontend produces an ordinary AIG and a serialized sequential interface.
The harness wraps a combinational function with input/output registers and
runs one pinned register-cleanup implementation before freezing this boundary.
It records and verifies SHA-256 hashes of the frozen artifacts. Backend runs
must consume these artifacts rather than lowering the IR again.

Both backends invoke the same standalone ABC binary and structural optimization
recipe. The ABC reference continues through `&nf`, buffering, and sizing. The
native backend reads ABC's choice-preserving output and runs NF-Liberty mapping,
timing-aware buffering, and register-aware resizing. All final netlists are
measured by one pinned `gv-stats` binary, not by the individual mapper's estimate.

For a fixed-register audit, specify `--cell CELL`. Native mapping and resizing
then exclude other sequential cells. The ABC path adjusts data polarity before
mapping and restores exactly that cell afterward. This avoids crediting
combinational improvements to different register area. Both paths use the same
uninitialized-don't-care policy; explicit initialization is unsupported.

The ABC constraint file should use an appropriate representative launch driver
and the selected register's D-pin load. This is a proxy during ABC optimization,
not exact clock-to-Q modeling. Final `gv-stats` includes the restored cells'
clock-to-Q and setup timing. The harness uses an external input transition of
0.01 and external output load of 2.308, in the library's units. Ensure these
are appropriate for the selected library before interpreting results.

## Usage

The balanced restart configuration uses one five-round `&resyn3` candidate,
NF-Liberty mapping, timing-aware buffering, and register-aware resizing. It
retains the normal delay-oriented cover policy; an area-children/structural-cut
default was evaluated separately and is not selected. The harness's native
settings are three resizing rounds, 16 timing iterations, 32 area iterations,
64 candidate evaluations per iteration, and a fanout limit of 12. Fix the FF
cell only for the controlled audit; normal mapping may select and resize FFs.

Build each revision separately and retain immutable release binaries. The
adapter is an example over reusable library APIs:

```sh
cargo build --release -p xlsynth-driver --bin xlsynth-driver
cargo build --release -p xlsynth-g8r --example sequential_mapping_boundary
python3 scripts/run_frozen_backend_qor.py freeze \
  --driver /path/to/pinned-frontend \
  --utility-driver /path/to/pinned-cleanup-driver \
  --corpus sample=/path/to/optimized-ir \
  --output /path/to/frozen-inputs --workers 8 --timeout 600
```

Use the same `--inputs`, evaluator, adapter, standalone ABC, raw Liberty files,
timing proto, constraints, and register cell for each backend run:

```sh
python3 scripts/run_frozen_backend_qor.py run \
  --inputs /path/to/frozen-inputs/results.json \
  --backend native --name candidate --syn-command '&resyn3' \
  --driver /path/to/candidate-driver --evaluator /path/to/pinned-evaluator \
  --adapter /path/to/sequential_mapping_boundary --abc /path/to/abc \
  --proto /path/to/timing.proto --liberty /path/to/cells.lib \
  --constraints /path/to/timing.constr --cell DFF \
  --output /path/to/candidate-results --workers 8 --timeout 600 --verify 1000
```

Repeat with `--backend abc` and a different output directory for a same-script
comparison. Use `--syn-command '&syn2'` on the ABC reference for an end-to-end
comparison against conventional ABC-speed, and state that the restructuring
recipe also differs in that comparison. Multiple
`--liberty` arguments are supported. `--selection` accepts a JSON array of
objects with `corpus`, `design`, and (for freezing) `input` fields.
The default restructuring command is `&syn2`; `--syn-command '&resyn3'`
permits an explicit script ablation. Hold this setting fixed when comparing
only mappers, and label any end-to-end backend comparison that changes it.

`--verify 1000` compares the frozen sequential AIG and final mapped netlist on
1,000 uniformly distributed input vectors, using a deterministic seed. Two
warmup cycles flush arbitrary initial state in this input/output-flopped
combinational workload. This is not a general sequential-equivalence check,
nor does it independently check the IR-to-AIG frontend.

The timeout applies separately to each subprocess, not to the entire sample.
Partial results, errors, timeout diagnostics, commands, hashes, and mapped
netlists are retained. Output directories must not already exist.

```sh
python3 scripts/run_frozen_backend_qor.py summarize \
  --reference /path/to/abc-results/results.json \
  --candidate /path/to/candidate-results/results.json \
  --output /path/to/summary.json
python3 -m unittest discover -s scripts -p test_run_frozen_backend_qor.py
```

Summaries use only paired successes with meaningful register-to-register delay,
and include status counts, paired design names, total area, register area,
combinational area, and geometric-mean ratios. Negative deltas mean smaller or
faster. Always compare candidates on the same intersection when ranking them;
do not compare geometric means over different timeout sets. Group designs by
baseline area when a corpus contains many tiny functions.

To isolate frontend changes, freeze each frontend revision separately and run
the same fixed backend on both. To isolate backend changes, freeze once using
the unmodified frontend. Only after these comparisons should both extracted
branches be combined.
