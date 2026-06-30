## Fuzz Targets Overview

This document lists the fuzz targets in the repository and summarizes what each one exercises at a high level. Each entry describes the essential property under test and the major failure modes it is intended to surface. Per-target early-return rationales are documented inline in the target source above the relevant condition, not here.

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_roundtrip.rs

Generates an upstream-standard acyclic random package directly as PIR, including
helper functions, `invoke`, `counted_for`, `gate`, arbitrary-width multiply,
product-pair, token, and event forms (`after_all`, `assert`, `trace`, and
`cover`), parses and re-emits it through libxls, then reparses with the Rust
parser and checks IR-level structural equivalence across the PIR roundtrip.
Panics if either implementation rejects the emitted IR or if the top function
is missing.

Primarily tests:

- libxls parse/re-emit IR text compatibility with the Rust parser
- Function/package pretty-printer roundtrip soundness
- Structural equivalence stability across roundtrip, including event metadata

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_opt_equiv.rs

Builds an XLS IR package from an upstream-standard random sample, including
`gate` and arbitrary-width multiply but excluding product-pair operations
pending prover support, runs IR optimization, and cross-checks equivalence
between original and optimized IR using an external tool and available SMT
backends. Flags disagreements between engines and unexpected failures.

Primarily tests:

- Optimization preserves semantics
- Cross-engine equivalence consistency (external tool vs SMT backends)

### xlsynth-pir/fuzz/fuzz_targets/fuzz_aug_opt_equiv.rs

Generates an upstream-standard random XLS IR function, including `gate` and
arbitrary-width multiply but excluding product-pair operations pending formal
support, runs the PIR aug-opt rewrite loop, and checks toolchain equivalence
between the original and rewritten IR when at least one rewrite fires. The
target flags unexpected aug-opt failures or inequivalent rewrites.

Primarily tests:

- PIR aug-opt rewrites preserve semantics when they apply
- End-to-end compatibility of PIR lowering and toolchain equivalence checks

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_dslx_stitch_pipeline_names.rs

Generates small two-stage DSLX stitch-pipeline samples with parameter names
drawn from wrapper/control names, generated internal-name patterns,
SystemVerilog keywords, and random legal identifiers. Unsafe samples must fail
through the controlled stitch-pipeline name validator; accepted samples are then
compiled and simulated with `xlsynth-vastly` using a single-module form that
mirrors the fixed two-stage wrapper shape, because the `xlsynth-vastly` pipeline
compiler currently operates on a single module. The arithmetic check is
intentionally non-commutative so swapped stage bindings produce a different
observed output.

Primarily tests:

- DSLX-derived names that would collide with wrapper/control/output/internal
  names are rejected before codegen
- SystemVerilog keywords do not leak into emitted stage or wrapper interfaces
- Accepted names preserve the arithmetic behavior of the stitched two-stage
  pipeline under reset/valid simulation

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_mul_by_const_csd_equiv.rs

Builds small `umul(x, literal)` PIR functions up to 12 bits wide (with the
literal on either side), then gatifies with built-in mul-by-const lowering. Each
run exports the gates back to IR and proves source-vs-gate equivalence with the
in-process Bitwuzla prover so the target surfaces any semantic mismatch
introduced by the Canonical Signed Digit (CSD) / Non-Adjacent Form (NAF)
gate-level lowering path.

Primarily tests:

- Built-in gatify mul-by-const lowering preserves semantics
- Canonical Signed Digit (CSD) / Non-Adjacent Form (NAF) decomposition plus
  shift/add/sub array-add accumulation remain equivalent

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_g8r_evaluation.rs

Generates random gatify-supported PIR functions with up to 64 generated nodes
and runs them through the canonical production g8r pipeline with all default
optimization stages enabled. It then
differentially evaluates the source PIR and optimized `GateFn` over 1,024
deterministic, corner-biased input samples. Input generation is seeded from the
generated IR text so coverage-guided entropy is dedicated to exploring program
structure instead of stimulus values.

Primarily tests:

- End-to-end prep, gatify, FRAIG, reassociation, and cut-DB rewrite correctness
- Aggregate input/output flattening stays aligned between PIR and g8r evaluation
- Broad concrete-value equivalence at higher throughput than formal proof

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_g8r_formal.rs

Generates random gatify-supported PIR functions with up to 64 generated nodes,
runs the same canonical production g8r pipeline, exports the source and
optimized gate forms to standard XLS IR, and
asks the in-process Bitwuzla prover to establish formal equivalence. Each solver
query has a 10-second time limit and a 512-MiB memory limit; resource-limit
inconclusive results are not treated as correctness failures.

Primarily tests:

- Exhaustive source-to-final-g8r equivalence for solver-proven samples
- End-to-end correctness across the complete default g8r optimization pipeline
- Gate-to-IR export remains compatible with formal equivalence checking

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_fn_roundtrip.rs

Builds a random `GateFn`, serializes to text, parses it back, and checks structural equivalence of the original vs parsed `GateFn`.

Primarily tests:

- GateFn textual serdes roundtrip
- Structural equivalence stability in gate graphs

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_fn_aiger_roundtrip.rs

Builds a random `GateFn`, emits AIGER, reloads AIGER into a flat raw `GateFn`,
then explicitly regroups that flat interface using the original `GateFn`
interface as the schema source. The target checks structural equivalence only
after that explicit-schema regroup step.

Primarily tests:

- Raw AIGER load preserves the flat bitstream interface semantics
- Explicit-schema regroup reconstructs the original grouped `GateFn` interface
- AIGER emission/load stay structurally aligned once the intended interface is
  imposed explicitly

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_eval_interp_equiv.rs

Differentially compares our Rust IR function interpreter with the xlsynth C++ interpreter on the same directly generated acyclic PIR package, but instead of checking a single arbitrary argument tuple it uses autocov to grow a bounded corpus of interesting typed inputs.

- Generates an upstream-standard random PIR package, including helper functions, `invoke`, `counted_for`, `gate`, and arbitrary-width multiply, via `xlsynth_pir::ir_random`, then parses its emitted IR through libxls for the reference interpreter.
- Runs autocov on the generated IR text to synthesize a small corpus of semantically interesting input tuples.
- Evaluates every corpus sample with both engines and asserts the results are equal, including division/modulus edge cases and composite-valued `one_hot_sel` cases.

See inline comments in the target source for early-return rationales.

Primarily tests:

- Autocov-selected boundary-ish inputs expose the same interpreter semantics as direct xlsynth interpretation
- Generated IR text remains parseable and executable across the C++ and PIR interpreters
- Arbitrary generated ops and value shapes behave identically in both interpreters across autocov-selected inputs

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_ext_nary_add_gatify_equiv.rs

Generates a one-function PIR package whose return value is an `ext_nary_add`
over zero to five operands with source mix (3/10 literals, 1/10 existing
parameters, 6/10 newly introduced parameters), `signed` flags, `negated`
flags, and optional adder architecture. This target restricts all generated
operand, literal, and parameter widths to 1..=8 bits to avoid zero-width
gate-to-IR cases while exercising the g8r lowering path. The input package is
constructed programmatically, then the target gatifies that function, exports
the original package to desugared XLS IR, converts the
resulting `GateFn` back to XLS IR, reparses both packages into PIR, and then
formally proves the two top functions equivalent with an in-process solver
backend (Bitwuzla or Boolector, depending on enabled features).

Primarily tests:

- `ExtNaryAdd` lowering through g8r preserves semantics for random operand
  counts, widths, signedness, negation, and architecture choices
- Export/desugaring of generated `ext_nary_add` packages remains compatible with
  PIR parsing and in-process formal equivalence
- Gate-to-IR conversion of the gatified result stays semantically aligned with
  the exported XLS IR form

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_prep_for_gatify_mask_low.rs

Builds small PIR functions containing either recognized mask-low idioms or
near-miss variants, runs `prep_for_gatify` with only the mask-low rewrite
enabled, and exhaustively evaluates all count values for the bounded generated
count width before and after prep. The target also checks that only recognized
low-mask and high-mask forms are rewritten through `ext_mask_low`: shift-minus
one, add-all-ones, shifted all-ones, complement high-mask forms,
shifted-right all-ones masks, same-count `shll(shrl(all_ones, count), count)`
high masks, and zero-prefixed low masks where prep can prove the count does not
reach the prefixed bits.

Primarily tests:

- The prep-for-gatify mask-low rewrite preserves semantics for generated output
  widths, count widths, and count values
- Matching and near-miss idiom detection stays precise as rewrite guards evolve
- `ext_mask_low` evaluation remains aligned with the shift/sub basis expression
  introduced by the rewrite

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_node_provenance.rs

Builds a random gatify-supported PIR function, including PIR extension
operations, using the shared `xlsynth_pir::ir_random` generator, then gatifies with `fold=false` and
`hash=false`. For each resulting AIG node, the target checks the initial
provenance seeding invariant against the original parsed PIR function: the
builder's dedicated constant-false literal is the only literal node, and its
provenance (if any) must stay sorted, deduplicated, and tied to original PIR
nodes. Every lowered `Input` / `And2` must carry a non-empty, sorted,
deduplicated set of PIR `text_id`s, each of which must correspond to some node
in the original pre-prep PIR function.

Primarily tests:

- Initial PIR-to-g8r lowering seeds non-empty provenance onto every lowered
  `Input` / `And2`
- Seeded provenance ids refer to real original PIR nodes, so prep-for-gatify
  and initial lowering do not silently shift provenance onto fabricated ids

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_node_provenance_with_opts.rs

Builds a random gatify-supported PIR function, including PIR extension
operations, using the shared `xlsynth_pir::ir_random` generator, gatifies with folding and hashing enabled,
then runs one bounded FRAIG iteration and one bounded cut-db rewrite
iteration. For each surviving AIG node, the target checks the provenance
invariant against the original parsed PIR function: every surviving `Input` /
`And2` must carry a non-empty, sorted, deduplicated set of PIR `text_id`s, and
every stored id must correspond to some node in the original pre-prep PIR
function.

Primarily tests:

- PIR-to-g8r provenance survives the optimized default gate pipeline
  (fold/hash, FRAIG, cut-db rewrite)
- Surviving provenance ids remain non-empty, sorted, deduplicated, and tied to
  original PIR nodes rather than fabricated helper ids

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_transform_equiv.rs

Applies a randomly selected equivalence-preserving transform to a random `GateFn` and proves equivalence via IR-based checker; panics if a claimed always-equivalent transform breaks equivalence.

Primarily tests:

- Local graph transforms preserve semantics
- IR-based equivalence checker catches semantic breakages

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_transform_arbitrary.rs

Performs bounded random sequences of transformations over a `GateFn` and
runs Cadical gate-level equivalence proofs on original/current/next pairs;
panics when an always-equivalent transform yields inequivalence.

Primarily tests:

- Transform engine correctness under random application
- Cadical-backed gate formal validation remains stable under transform churn

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_cut_db_rewrite_equiv.rs

Builds a random `GateFn`, runs the cut-db rewrite pass (`rewrite_gatefn_with_cut_db`) using the vendored 4-input cut database, checks area-cost deltas and delay-gating decisions for accepted rewrites against independently DCE-cleaned graph copies, and proves the rewritten graph is SAT-equivalent (Cadical) to the original. Panics on any semantic mismatch or cost-accounting inconsistency.

Primarily tests:

- Cut-db rewrite pass preserves semantics on arbitrary AIGs
- Stability of cut enumeration + replacement instantiation under random graphs
- Area rewrite live-node cost accounting matches cleaned graph deltas
- Depth rewrite critical-path reduction and area rewrite no-regression checks
  match cleaned graph depths

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_dynamic_structural_hash.rs

Builds a random `GateFn`, wraps it in the dynamic structural hash edit state,
then applies a bounded sequence of random `add_and`, fanin/output edge moves,
and delete-node attempts. After every operation, it rebuilds fanouts,
output-use counts, and local strash buckets from scratch and asserts that the
incrementally maintained state is identical.

Primarily tests:

- Dynamic local-strash updates stay coherent across node addition, edge
  rewiring, replacement cascades, and inactive-node/MFFC deletion
- Illegal random edits such as cycle-creating moves or deleting still-used nodes
  leave the edit state unchanged and coherent

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_dynamic_depth.rs

Builds a random `GateFn`, wraps it in the dynamic forward/backward depth edit
state, then applies a bounded sequence of random `add_and`, fanin/output edge
moves, replacement attempts, and delete-node attempts. After every operation,
it rebuilds forward and backward depths from scratch and asserts that the
maintained state is identical.

Primarily tests:

- Dynamic depth bookkeeping stays coherent across node addition, edge rewiring,
  replacement, and inactive-node/MFFC deletion
- Illegal random edits such as cycle-creating moves or deleting still-used nodes
  leave the edit state unchanged and coherent

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_dynamic_depth_cut_replacement.rs

Builds a random `GateFn`, chooses random live AND roots, expands random cuts
toward inputs, builds random replacement AND fragments from the cut leaves using
the dynamic structural hash, and replaces the cut root with the fragment output.
After each cut-style replacement, it updates `DynamicDepthState` only from the
touched cut nodes and replacement nodes, then compares forward and backward
depths against a full recomputation.

Primarily tests:

- Incremental dynamic-depth updates stay coherent for cutdb-like replacements
  that add strashed fragment nodes, rewire root fanouts, and delete dangling
  MFFC nodes
- The dirty-node contract for replacement cuts covers cut leaves, old root
  fanouts, replacement fragment nodes, and reused strash representatives

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_bulk_replace.rs

Exercises bulk replacement algorithms on internal IR/gate structures (see source for specifics) under random inputs, checking invariants and/or equivalence as implemented in the target.

Primarily tests:

- Bulk replace operations and invariants
- Stability of large-scale rewrites

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_same_sig_pair.rs

Builds two PIR functions across the full function-level random-generator
surface, including token values and event operations, with
`generate_same_signature_pair` and asserts that both validate and share an
identical signature. Panics if either function fails to validate or if their
types differ.

Primarily tests:

- `generate_same_signature_pair` produces paired samples with matching function signatures
- IR validation succeeds for both generated functions

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_rebase_equiv.rs

Generates two upstream-standard PIR functions with the same signature,
including `gate` and arbitrary-width multiply but excluding product-pair
operations pending formal support. Rebuilds desired on top of orig using
`rebase_onto`, then proves semantic equivalence between `desired` and the
rebased result using the external toolchain equivalence checker.

Primarily tests:

- `rebase_onto` preserves semantics and IDs stability constraints across random graphs
- Integration of `rebase_onto` with parser/pretty printer and toolchain equivalence

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_outline_equiv.rs

Generates an upstream-standard random function directly as PIR, including
`gate` and arbitrary-width multiply but excluding product-pair operations,
then selects a random connected
subgraph (via BFS from a seed) to outline into a new inner function. Rewrites
the outer function to invoke the inner and proves semantic equivalence between
the original function and the outlined outer using available SMT backends. The
target also explores parameter/return ordering:

- Param ordering mode: Default or deterministically shuffled non-default.
- Return ordering mode: Default or deterministically shuffled non-default.

Non-default orderings are constructed by permuting the default `OutlineOrdering` while preserving validity (same coverage, no duplicates). The PRNG is seeded from a stable hash of the package text, ensuring reproducible behavior for a given sample.

See inline comments in the target source for early-return rationales.

Primarily tests:

- Outlining transformation preserves semantics for random connected regions
- Stability of parser/pretty-printer across transformation boundaries

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_netlist_scanner.rs

Materializes a randomized sequence of token-like payloads, emits a textual netlist fragment for those payloads, scans the text with `TokenScanner`, and checks that the resulting token payloads round-trip (shape and values) for the supported subset. Early-return is used for uninteresting or degenerate samples where whitespace coalescing or intentionally unsupported constructs would otherwise cause spurious mismatches; this target focuses on exercising tokenizer behavior rather than full parser structure.

Primarily tests:

- Tokenization stability for identifiers, keywords, punctuation, comments
- Verilog-style integer literal tokenization (with/without width), including
  cases that appear near structural punctuation so that implicit nets and
  concatenations are tokenized correctly.
- Top-level annotation tokenization with string-valued fields

### xlsynth-vastly/fuzz/fuzz_targets/diff_iverilog.rs

Builds random expression+environment pairs, normalizes expressions through the
local parser/renderer, and compares evaluation results against an
`iverilog`/`vvp` oracle for accepted samples. The target surfaces semantic
mismatches in value bits and inferred widths.

Primarily tests:

- Consistency between `xlsynth-vastly` expression evaluation and
  `iverilog`/`vvp` semantics on shared accepted inputs
- Stability of parser+pretty-printer normalization before differential
  evaluation

### xlsynth-vastly/fuzz/fuzz_targets/xls_ir_codegen_semantics.rs

Generates XLS IR functions and checks that code generated from those functions
behaves consistently across `xlsynth` interpretation and `xlsynth-vastly`
simulation paths (including combo and pipelined forms) when supported by the
sample/tooling.

Primarily tests:

- End-to-end semantic equivalence between XLS IR interpretation and generated
  Verilog/SystemVerilog simulation
- Codegen/simulation consistency across combo and pipelined lowering paths
  under random typed inputs

______________________________________________________________________

Notes:

- Degenerate inputs (e.g., zero-width, empty ops) are typically skipped with a brief comment explaining why they are not informative for the property under test.
- Unexpected failures in normal API operations (e.g., pretty-printed text failing to parse) should be flagged (panic) to surface systemic issues early.
