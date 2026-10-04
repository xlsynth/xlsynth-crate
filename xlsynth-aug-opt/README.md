# `xlsynth-aug-opt`: augmented IR optimization

`xlsynth-aug-opt` combines upstream XLS optimization with local PIR rewrites.
It owns optimization rounds, rewrite options and results, candidate construction,
and profitability checks. The default cost model uses g8r's ordinary lowering,
reachable AND count, and Graph LE to evaluate bounded regions around candidate
rewrites. See the [optimizer guide](docs/aug_opt.md) for the individual passes.

## Dependencies

| Dependency | Role |
| --- | --- |
| `xlsynth` | Upstream XLS optimization and analysis APIs |
| `xlsynth-pir` | IR representation, parsing, matching, and manipulation |
| `xlsynth-g8r` | Gate lowering and cost measurement |

`xlsynth-driver` consumes this crate. PIR and g8r do not depend on or re-export
it. Optimizer integration tests live here so they do not add a dependency back
from either library.

## Library use

```rust
use xlsynth_aug_opt::{AugOptOptions, run_aug_opt_over_ir_text};

fn optimize(ir_text: &str) -> Result<String, String> {
    run_aug_opt_over_ir_text(
        ir_text,
        Some("main"),
        AugOptOptions {
            enable: true,
            ..AugOptOptions::default()
        },
    )
}
```

Options default to `enable: false`, one round, and `AugOptMode::Sandwich`.
When enabled, `Sandwich` runs XLS optimization before the first PIR round and
after each round. `AugOptMode::PirOnly` applies only the PIR rounds. Both modes
supply the default g8r cost evaluator. These optimizer
entrypoints require an explicit top function when optimization is enabled.

`run_aug_opt_over_ir_text_with_stats` also returns total and per-rewrite counts
in `AugOptRunResult`. For optimization followed by gate mapping, use
`xlsynth_aug_opt::ir2gates_from_ir_text(ir_text, top, aug_options, g8r_options)`.
It performs optimization before the mapper's array-alias preparation and range
analysis, and uses the supplied g8r folding and hashing settings for costing.
It uses the package's declared top function when `top` is `None`.

`run_aug_opt_over_ir_text_with_evaluator` accepts an
`ir_cost::ShiftChoiceCostEvaluator` for callers that need a custom cost policy.

## Command line

The debugging executable retains its name, `xlsynth-pir-aug-opt`:

```sh
cargo run -p xlsynth-aug-opt --bin xlsynth-pir-aug-opt -- input.ir --top main --rounds 1
```

Use `-` as the input path to read stdin, or add `--aug-opt-only` to select
`PirOnly`. The main driver exposes the same optimizer through `--aug-opt=true`
on its supported optimization and code-generation commands.

For comparisons, set `AugOptOptions::recover_split_adders` to `false` to disable
only split-adder recovery. The debugging executable accepts
`--recover-split-adders=false`; `xlsynth-driver ir2opt` accepts
`--aug-opt-recover-split-adders=false` alongside `--aug-opt=true`.

## API migration

| Previous API or location | Replacement |
| --- | --- |
| `AugOptMode`, `AugOptOptions`, `AugOptRewriteStats`, and `AugOptRunResult` from `xlsynth_pir::aug_opt` or `xlsynth_g8r::aug_opt` | The same types at `xlsynth_aug_opt`'s crate root |
| `xlsynth_g8r::aug_opt::run_aug_opt_over_ir_text[_with_stats]` | `xlsynth_aug_opt::run_aug_opt_over_ir_text[_with_stats]` |
| `xlsynth_pir::aug_opt::run_aug_opt_over_ir_text[_with_stats]` | The same crate-root entrypoints, now with g8r costing by default |
| `xlsynth_pir::aug_opt::run_aug_opt_over_ir_text_with_evaluator` | `xlsynth_aug_opt::run_aug_opt_over_ir_text_with_evaluator` |
| `xlsynth_pir::constant_shift_choices` and `xlsynth_pir::ir_cost` | `xlsynth_aug_opt::constant_shift_choices` and `xlsynth_aug_opt::ir_cost` |
| `xlsynth_g8r::gatify::ir2gate::GateBuilderCostEvaluator` | `xlsynth_aug_opt::cost::GateBuilderCostEvaluator` |
| `Ir2GatesOptions::aug_opt` | Pass `AugOptOptions` separately to `xlsynth_aug_opt::ir2gates_from_ir_text` |
| `cargo run -p xlsynth-g8r --bin xlsynth-pir-aug-opt` | `cargo run -p xlsynth-aug-opt --bin xlsynth-pir-aug-opt` |

Callers must update their dependencies and imports directly; upward
compatibility re-exports would introduce dependency cycles. Optimizer tests and
the `fuzz_aug_opt_equiv` and `fuzz_constant_shift_choices` targets live in this
crate's `tests/` and `fuzz/` directories.
