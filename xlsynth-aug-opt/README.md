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
`ir_cost::ShiftChoiceCostEvaluator` for callers that need a custom cost policy
for constant-shift choices. Priority-result fusion uses its whole-function
g8r reference model independently of this local-region evaluator.

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

## Priority-result fusion

`AugOptOptions::fuse_priority_results` defaults to true within aug-opt. It rewires
`decode(F(encode(one_hot(x))))` or `1 << F(encode(one_hot(x)))` directly from
the one-hot result. Each affine operation in `F` retains its original modular
width; the zero-input sentinel and overshifts are included. An optional final
`F(...) & sign_ext(p)` selects the wired result when the one-bit predicate is
true and bit zero otherwise.

The pass runs before initial XLS optimization and in each requested PIR round.
This preserves semantic index structure before XLS narrows arithmetic or
distributes predicate masks, while allowing later rounds to recognize newly
exposed forms. Zero rounds performs no fusion. All accepted candidates pass
through subsequent XLS optimization in sandwich mode.

The pass emits only ordinary XLS IR. Whole-function costing prepares private
clones with g8r, preserving external sharing and loads in the comparison.
The reference profile uses folded/hashed Brent-Kung mapping without range
information, gate DCE, and graph logical effort with beta1=1/beta2=0, before
FRAIG or ABC.
It requires a strict improvement in live AND count or graph logical effort
without worsening either. The gate-mapping library wrapper forwards its
fold/hash settings. Unsupported shapes, externally shared count/index
intermediates, exhausted node IDs, cost errors, and ties preserve the input.

This is a g8r cost policy, not a universal backend benefit or physical PPA
claim. Matched Yosys/ABC measurements include an area/delay tradeoff for a
256-bit affine decoder and a small whole-function regression for
`std::next_pow2(u32)`. Those backend comparisons are diagnostic; acceptance
uses raw g8r AND count and graph logical effort.

Aug-opt itself remains opt-in: set `AugOptOptions::enable` to true or pass
`--aug-opt=true` to `ir2opt`, `dslx2ir`, `ir2combo`, or `ir2pipeline`.
No separate fusion flag is needed. For ablation, set `fuse_priority_results`
to false or pass `--aug-opt-fuse-priority-results=false`; the debugging
executable accepts `--fuse-priority-results=false`. The default round count
remains one, configurable with `--aug-opt-rounds=N` or the debugger's
`--rounds N`. `AugOptRewriteStats::priority_results_fused` reports accepted sites.

The optimizing `xlsynth_aug_opt::ir2gates_from_ir_text` wrapper includes fusion
when aug-opt is enabled. Plain `xlsynth_g8r` mapping APIs and the driver's
`ir2gates`, `ir2g8r`, and `dslx-g8r-stats` paths do not invoke aug-opt.
To optimize before these lowering commands, run `ir2opt --aug-opt=true`
first. Semantic rewrites remain in aug-opt; prep performs synthesis shaping.

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
