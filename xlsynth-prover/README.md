# xlsynth-prover

Reusable equivalence checking, DSLX proof tactics, and proof-plan execution.

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
can continue working on the same tree.

Nested `Focus` and `Cosliced` decompositions allocate fresh solver symbols
against both sides' inherited mappings. Distinct function pairs must retain
independent symbols; accidentally sharing one can produce a false proof.

## Process backend

On Unix, `dslx_tactics::process::execute_script` runs the same engine using an
explicit path to an `xlsynth-driver` executable. The lower-level
`run_obligations` adapter builds tasks, retains temporary DSLX sources until
execution finishes, and returns captured output. Library entrypoints do not
print task diagnostics.

`proof_plan::config` provides serializable task and group configurations.
`proof_plan::scheduler::run_prover_plan_with_driver` executes plans using a
caller-supplied driver path and concurrency limit. `Scheduler::new_with_driver`
also accepts a caller-owned cancellation flag. The convenience plan runner
temporarily installs Unix signal handlers; use `Scheduler` directly when the
embedding application owns signal handling.

The driver handles files/stdin, CLI arguments, diagnostic presentation, and
exit codes. Engine tests live in this crate; driver integration tests cover
command wiring and output compatibility. The subprocess backend invokes the
driver executable at runtime, without adding a Cargo dependency on the driver.
