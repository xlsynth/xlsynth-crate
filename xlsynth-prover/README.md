# xlsynth-prover

Reusable equivalence checking, DSLX proof tactics, and proof-plan execution.

For a file-based equivalence proof, build a `dslx_equiv::DslxEquivFiles`
request. Call `prove` to check it directly or `into_tactic_tree` to decompose
the same property. Supply DSLX function names for uninterpreted functions;
the library specializes the requested tops and translates names for the solver.

For QuickCheck, `dslx_quickcheck::DslxQuickCheckRequest::run` proves the
selected properties and returns the suite conclusion and each property's
result. Its callback can display progress. Use `tactic_tree` to select the
same properties for a script. A root `Solve` carries the assertion semantics,
label filter and uninterpreted function mappings through to the worker.

For assertion proofs, `dslx_assertions::AssertionUfMappings::LocalDslx` accepts
helper names in the entry module. Use `Ir` when you already have resolved IR
names, for example to abstract a helper in an imported module.

## Tactic engine

`dslx_tactics::script` owns the obligation tree, tactic application, solve
selection, task-result correlation, rollback, and whole-proof completion.
Construct an `OblTree` from a `ProverObligation` and `OblTreeConfig`, parse script
text with `parse_script_steps_from_json_str` or
`parse_script_steps_from_jsonl_str`, then call `execute_script_with_runner`.

The runner receives selected `(task_id, obligation)` pairs and configuration,
and returns a typed `proof_plan::report::ProverReport`. It does not receive the
mutable obligation tree. A runner is responsible for correctly proving each
submitted obligation; the engine is responsible for interpreting those results
against the final tree. Missing outcomes leave obligations unresolved.

Use `ScriptReport::is_complete()` to determine whether the proof is finished.
An empty script or successful subset of obligations is insufficient. Failed
children can reopen their parent, so `solved` records successful attempts while
`unresolved` describes unproved leaves in the surviving tree. Repeated calls
can continue working on the same tree. An accepted runner report consumes its
`Solve` requests, including those with missing or inconclusive results. An empty
follow-up script reports the existing proof state; explicitly select `Solve`
again to retry an obligation. If the runner returns an error or reports invalid
result IDs, requests remain queued for retry. Only a definite counterexample can
trigger a decomposition rollback; inconclusive results and execution errors
leave the selected obligations unresolved.

Persisted tactic reports do not have a backward-compatibility guarantee.
Deserializing a `ScriptReport` requires an explicit `unresolved` field; reports
from older versions without it are intentionally rejected. Missing completion
information cannot be interpreted as an empty list of unproved obligations:
even an old empty-script report could otherwise appear complete. Regenerate
legacy reports with the current engine instead of adding a default value.

`root_disproved` records a counterexample to the surviving root, including one
found in an earlier call. `ScriptReport::proof_status()` preserves that conclusion
across empty follow-up calls; failed decomposition children only justify rollback,
not a root disproof. Reports require this field as well as `unresolved`.

Nested `Focus` and `Cosliced` decompositions allocate fresh solver symbols
against both sides' inherited mappings. Distinct function pairs must retain
independent symbols; accidentally sharing one can produce a false proof.

`LecObligation::options` records the equivalence settings for each obligation;
fixed implicit activation is recorded on each `LecSide`. Runners must honor
these settings. A root-only `Solve` preserves assertion semantics, assertion
label filters, enum domains, parameter drops, aggregate flattening, activation
settings, and parallelism. The process backend preserves source basenames so
module-qualified assertion filters retain their meaning. `LecSide::uf_map`
uses DSLX function names, before IR name mangling.

`Focus` and `Cosliced` currently require ignored assertions, no parameter drops,
no aggregate flattening, no fixed implicit activation, and single-threaded
equivalence checking. Unsupported combinations return an error before applying
the tactic: UF abstraction discards assertion effects, and changing tops does
not preserve signature-dependent settings automatically.

Both tactics prove their children over all enum bit representations, even when
the parent assumes valid enum inputs. Internal calls can receive invalid enum
values, so assuming valid inputs for a helper could falsely prove its callers
equivalent. This stronger requirement means a decomposition can fail even when
a direct proof restricted to valid root inputs succeeds.

## Process backend

On Unix, `dslx_tactics::process::execute_script` runs the same engine using an
explicit path to an `xlsynth-driver` executable. Its `ScriptExecution` contains
the root report and an optional solver batch for displaying captured task
diagnostics. Its serialized status comes from the root: a successful batch
does not prove leaves the script did not select. The lower-level
`run_obligations` adapter builds tasks, retains temporary DSLX sources until
execution finishes, and returns captured output. Library entrypoints do not
print task diagnostics.

`proof_plan::config` provides serializable task and group configurations.
`proof_plan::scheduler::run_prover_plan_with_driver` executes plans using a
caller-supplied driver path and concurrency limit. Pass a `proof_plan::config::ProofDriver` to select a toolchain TOML for all workers. `Scheduler::new_with_driver`
also accepts a caller-owned cancellation flag. The convenience plan runner
temporarily installs Unix signal handlers; use `Scheduler` directly when the
embedding application owns signal handling.

`proof_outcome::ProofStatus` distinguishes `proved`, `disproved`, `inconclusive`,
and `execution-error`. Child commands must write JSON containing both `status`
and `success`, with the following values and normal process exit codes:

| `status` | `success` | Exit code |
| --- | --- | --- |
| `proved` | `true` | `0` |
| `disproved` | `false` | `1` |
| `inconclusive` | `false` | `1` |
| `execution-error` | `false` | `1` |

The result must be a JSON object with a string `status` and a Boolean
`success`, each present exactly once. Additional diagnostic fields are allowed.
Missing, malformed, or contradictory
results are protocol errors; signals and unexpected exit codes are process
failures, even if the child wrote successful JSON first. Neither establishes a
proof or counterexample. Legacy JSON containing only `success` is intentionally
rejected. Captured stdout/stderr remain available in task reports, including
after cancellation.

Inconclusive results and execution errors are indefinite scheduler outcomes:
they cannot win a `first` group or trigger tactic rollback. Native solver
counterexamples remain `disproved`. External-tool failures without structured
counterexample evidence are conservatively `execution-error`, because a
nonzero exit alone cannot distinguish disproof from a broken invocation.

The driver handles files/stdin, CLI arguments, diagnostic presentation, and
exit codes. Engine tests live in this crate; driver integration tests cover
command wiring and output compatibility. The subprocess backend invokes the
driver executable at runtime, without adding a Cargo dependency on the driver.
