## Fuzz Targets Overview

This document lists the fuzz targets in the repository and summarizes what each one exercises at a high level. Each entry describes the essential property under test and the major failure modes it is intended to surface. Per-target early-return rationales are documented inline in the target source above the relevant condition, not here.

### Shared concrete input sampling

Random graph-evaluation tests use the typed or flat-vector adapters in
`xlsynth_pir::random_inputs`. Each evaluation independently draws one complete
vector of N function arguments or block input ports:

- With 10% probability, choose equally among six whole-vector patterns:
  all-zero arguments, all-one arguments, alternating all-one/all-zero arguments,
  alternating all-zero/all-one arguments, all arguments with alternating bits
  ending in zero (`...1010`), or all arguments with alternating bits ending in
  one (`...0101`). The middle two alternate across argument positions; the last
  two alternate within every argument, using the same least-significant-bit
  phase for all inputs and restarting that phase in each aggregate bits leaf.
- Otherwise, choose K uniformly from 0 through N and select K distinct positions
  uniformly. Those arguments use special values; the other N - K use uniform
  random bits. Aggregates count as one argument and special patterns recurse
  through their leaves.
- After constructing the vector, one 50% gate enables perturbation of all
  special-derived arguments, including whole-vector patterns. Each eligible
  bit then flips independently with probability 1/64. Uniform-derived arguments
  are never perturbed. No flip is forced, and there is no flip-count cap.

Special values include zero, numeric one, all ones, signed extrema, one-hot and
one-cold values, alternating bit patterns, and low/high runs of ones. Arbitrary
widths, tokens, and empty aggregate shapes are preserved. Existing seed/entropy
sources make replay deterministic; graph generation remains separate from input
sampling. Block reset protocols can override the sampled reset port afterward.
Initial register-state vectors use the same sampler where arbitrary state is
permitted.

Explicit directed/exhaustive cases and adaptive/autocov inputs remain additional
coverage, not replacements for this policy. Uniform-only statistical workloads
and graph-literal generation keep their existing distributions.

### Running the smoke suite

With Python 3.11+ and cargo-fuzz installed, run the smoke suite with:

```shell
python3 scripts/run_all_fuzz_tests.py --target-dir target/fuzz-ci
```

The runner builds each existing fuzz package once, sequentially, using a shared
`target/fuzz-ci` output directory. Cargo reuses compatible dependencies across
packages. There is no generated package or workspace; each package retains its
own manifest and lockfile, and individual `cargo fuzz` commands are unchanged.

`--features` selects common features (default: `with-bitwuzla-system`), forwarded
to each fuzz package that declares them. A package's prebuild and subsequent runs
use identical flags. Fuzz manifests align solver forwarding
where applicable, but different transitive feature sets may still require
separate compiled variants. Unknown features are errors. Targets whose
`required-features` are not enabled are reported as excluded, as are the existing
default-excluded external ABC and Yosys targets.

The optional `--target-dir` changes the shared build output directory. Relative
paths are resolved from the invoking directory. Target names must be unique
across packages; the runner checks this before any builds. Corpora and crash
artifacts stay in each original fuzz package's `corpus/` and `artifacts/`
directories, separate from the build cache.

Each CI run starts with an empty `target/fuzz-ci`. The smoke job pins its Rust
nightly and restores the most recent sccache archive available for that
toolchain and runner platform. A fresh checkout can reuse compilations whose
inputs match even when Cargo sees new source timestamps. Successful `main`
runs save the updated compiler cache for subsequent pushes and pull requests.

Fuzz runs use four workers by default. CI retains its `-max_total_time=5` override and
`--sanitizer none`. Without a `--fuzz-bin-args` override, the runner also sets a
60-second per-input watchdog. These smoke runs do not replace longer fuzz
campaigns. Optional external-tool targets still require explicit selection and
their runtime prerequisites.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_expr_eval_equiv.rs

Generates typed expressions in the common BX and DSLX syntax, including
arithmetic, shifts, casts, packed slices, conditionals, arrays, and tuples.
It also generates `unroll_for!` folds with constant ranges, optional binder
annotations, loop-local lets, and scalar or array accumulators. Seeds can use
the block inputs or `zero!`, and array folds read and update successive values.
Each expression becomes both a combinational BX block and a DSLX function.
The target evaluates the lowered BX block with PIR and the compiled DSLX
function with the native XLS IR interpreter on four shared argument vectors.
It also evaluates a BX register reset initializer with the first vector bound
to constants.
Unexpected rejection or differing values can expose errors in parsing, type
inference, lowering, or constant folding.

Run the target from `xlsynth-bx`.

```shell
cargo +nightly fuzz run fuzz_bx_dslx_expr_eval_equiv --sanitizer none
```

To replay a finding, pass the saved artifact path immediately after the target
name. The same bytes reproduce the expression and its argument sets.

Directed BX regression tests use the same expression evaluator. They use the
bundled DSLX frontend by default. To compare against a particular XLS
checkout, run the frontend tests with its converter and standard library.

```shell
BX_DSLX_IR_CONVERTER=/path/to/xls/bazel-bin/xls/dslx/ir_convert/ir_converter_main \
DSLX_STDLIB_PATH=/path/to/xls/xls/dslx/stdlib \
cargo test -p xlsynth-bx --test frontend packed_slice
```

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_function_calls.rs

Generates functions with identical definitions in BX and DSLX, including
names that overlap hardware keywords, local bindings, signed and arbitrary-width
arithmetic, and scalar or aggregate results. A BX block invokes the generated
function directly or through nested calls that compute it on different argument
expressions and select between the results. The target compares source
acceptance with native DSLX parsing and typechecking. On four shared vectors
it compares the BX block's PIR evaluation with both the compiled DSLX function
in the XLS IR interpreter and calls to that function in DSLX's own interpreter.
The pinned DSLX API exposes the latter through constexpr metadata. The harness
binds each invocation to a module constant with the exact concrete arguments.
Invalid calls exercise missing, extra, and incompatible arguments. Signed slice
starts stay within the source word for bytecode comparison. In the pinned DSLX
bytecode interpreter, an out-of-range signed slice produces unsigned zero.
For example, `x[i+:s8] + s8:1` with `x = u8:0` and `i = u8:8` fails with
`ubits vs. sbits`, while the hardware IR evaluates to 1.
See the [pinned DSLX interpreter implementation](https://github.com/xlsynth/xlsynth/blob/e50f6ad07a1433af08d0f38b7eca686f631e9413/xls/dslx/bytecode/bytecode_interpreter.cc#L1761-L1792).
The expression hardware
target continues to generate unrestricted slice starts.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_function_calls --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_function_syntax.rs

Varies function declaration order, nested singleton tuple signatures, implicit
argument widths across scalar, array, and tuple parameters, builtin shadowing,
tuple destructuring, branch-local bindings, and the spelling of identifiers
ending in `!` next to an operator. It also varies comment delimiters and
parentheses around conditionals followed by tuple or array indexing.
Generated `zero!<T>()` calls resolve to the builtin, to a local
`fn zero!()` returning a generated nonzero value, or to a value binding that
makes the call invalid. It compares native DSLX source acceptance and
interpreter results with BX. It also checks both hardware IR evaluations for
modules whose DSLX names survive the pinned converter's IR text roundtrip.
The generator projects unsized hex literals through tuples and arrays in
scalar and aggregate call arguments, return expressions, and bindings. It also
checks radix retention through aliases and tuple destructuring. Comparisons,
branches, and array elements place a bound literal before or after the operand
that supplies its signed type. For example, passing `(0xff,).0` to
an `s8` parameter preserves the eight input bits in both native evaluators.
When that callee widens its parameter to `s16`, pinned DSLX bytecode returns
255 and the native hardware IR sign-extends it to −1.
For a dynamic array read or update beyond the end, the target checks the
native DSLX bounds error. It compares both hardware evaluations, where reads
clamp to the final element, and checks that an out-of-bounds update returns
the original array.

The syntax target invokes `$XLSYNTH_TOOLS/dslx_interpreter_main` with
`--evaluator=dslx-interpreter --compare=none --warnings_as_errors=false` and the
bundled DSLX standard library path. It batches in-range assertions in one DSLX
unit test and verifies the native index error separately for each out-of-range
input. Point `XLSYNTH_TOOLS` at the tools from the release pinned by the
checkout. From `xlsynth-bx`, run:

```shell
XLSYNTH_TOOLS=/path/to/pinned/tools cargo +nightly fuzz run fuzz_bx_dslx_function_syntax --sanitizer none
```

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_array_concat.rs

Compares array concatenation in BX blocks, local functions, and module constants with pinned DSLX. Generated operands vary outer lengths, signedness, element widths, nested arrays, tuples, structs, enums, and independently inferred literals. The target checks acceptance, result shape, and element order through BX block IR, compiled DSLX IR, and DSLX constant evaluation. Typed empty operands use DSLX constant evaluation because the pinned converter cannot lower an empty array literal in a function. Rejected programs exercise mismatched element types, distinct nominal declarations with the same layout, incorrect result lengths, and mixed array and bit operands.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_array_concat`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_array_literals.rs

Compares BX and DSLX array literal acceptance and element values, including a trailing ellipsis that repeats a runtime expression. It varies prefix lengths, trailing commas, scalar and aggregate element types, module-constant and parametric dimensions, and the type supplied by a return or annotated let. Both frontends must reject missing seeds, misplaced ellipses, mismatched element types and incorrect sizes. Accepted programs run against DSLX's interpreter and lowered IR, except that empty literals use the interpreter because the pinned converter rejects them.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_array_literals`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_bits_array_casts.rs

The target compares BX bits-to-array and array-to-bits casts with pinned DSLX. It varies signed, unsigned and Boolean types, zero-width and wide elements, empty arrays, and asymmetric bit patterns to expose reversed elements or slices. It checks module constants and runtime values against DSLX IR and also checks unsigned casts against DSLX constexpr evaluation. Signed cases use compiled DSLX because the pinned bytecode interpreter cannot flatten signed elements and assigns unsigned tags to cast results. Unequal packed widths and casts involving nested arrays, tuples, structs or enums must fail typechecking.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_bits_array_casts`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_builtins.rs

Generates calls to pure DSLX builtins in BX module constants, local functions, and combinational blocks. It compares source acceptance, constant values, and runtime results against pinned DSLX. The target varies signedness, zero-width and wide operands, aggregate shapes, selector masks, and indices near clipping boundaries. It also generates calls with extra arguments and incompatible operand types, and uses unary builtins as named `map` callbacks. It compares the sum of multiply partial products because the interpreters may split the product differently. It checks `ceillog2` in constants and uses DSLX constexpr execution for decode widths the pinned IR converter cannot lower.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_builtins`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_enums.rs

Generates numeric enums and compares BX with pinned DSLX source checking, concrete evaluation and compiled IR. Varying discriminants, backing types, aliases and module boundaries exposes nominal type, extension and exhaustiveness errors. Cast probes compare hardware results on arbitrary bits and separately check whether DSLX can evaluate each cast as a constant. Ordering probes exercise direct calls, map callbacks and counted-loop helpers. BX and DSLX must diagnose ordering of enum operands. The target also compares the hardware results of explicitly ordering their backing bits.

Calls to one generic DSLX helper with same-named enums from two modules check specialization identity across equal and different backing widths. The native reference compares each module's members directly.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_enums`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_scalar_match.rs

Compares scalar `match` in BX blocks and local functions against pinned DSLX
source checking, native bytecode execution and hardware IR. The generator
varies signedness and widths up to 129 bits, literals, alternatives, wildcards,
and exclusive and inclusive ranges. It generates exhaustive partitions,
coverage gaps, partial overlaps, freely combined patterns and malformed
arms. Match subjects include names, function calls and unsized literals, so
both the subject and the patterns can supply the first concrete bit width.
Signed patterns also vary the spelling of negative literals whose magnitude
uses the full width. Result arms mix unsized literals with explicit signed
values and local calls, both directly and inside aggregate results.
For runtime subjects, the target exercises every pattern endpoint and its
immediate neighbors alongside random inputs. Distinct arm results expose priority inversions,
and an array read confined to the zero arm exposes eager evaluation of
unselected arms in bytecode.

From `xlsynth-bx`, run:

```shell
XLSYNTH_TOOLS=/path/to/pinned/tools cargo +nightly fuzz run fuzz_bx_dslx_scalar_match --sanitizer none
```

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_tuple_match.rs

Compares tuple `match` in BX blocks and local functions with pinned DSLX
source checking, concrete execution, and compiled IR. The generator varies
nested tuple shapes, singleton and unit patterns, alternatives, and leaves
with booleans, wildcards, constants, and signed or unsigned ranges. Generated
coverage partitions, gaps, and overlaps detect exhaustiveness and first-match
priority errors. Concrete inputs cross range endpoints with the Boolean
combinations to exercise overlapping arms and nested field projections.
Rejected programs exercise redundant patterns and incorrect tuple arity or
leaf signedness.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_tuple_match`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_unroll_types.rs

Compares the inferred type of an unannotated BX binding to native DSLX
constant metadata for scalar and array folds, including signedness and widths
above 64. Separate rejection cases vary the range, binder annotation and body
result to detect acceptance of runtime bounds, reversed ranges, or incompatible
types.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_unroll_types --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_counted_for.rs

Compares counted `for` expressions in BX blocks and local functions with pinned DSLX. Generated loops vary half-open and inclusive ranges, runtime arrays, tuple binders, accumulator types, captures, nesting, and invalid annotations or iterables. Named and scoped literal arrays exercise how an element annotation determines the signedness of implicit bit patterns. Accepted programs run on the same concrete inputs through BX Block IR and DSLX constant calls. The target also compares native DSLX IR when the range base fits its converter. It detects acceptance differences, incorrect base offsets or carry ordering, and lost captured values.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_counted_for --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_map_lambda.rs

Compares `map` in BX blocks and local functions with pinned DSLX typechecking, concrete calls, and compiled IR. Named callbacks exercise local functions, inferred width-parametric DSLX imports, and BX callback parameters. Both `map(xs, f)` and `map(xs, |x| f(x))` in BX must produce DSLX's named-map result for the same arrays or typed ranges, including aggregate elements and results. Invalid cases vary arity, element widths, shadowing, and non-array inputs. For untyped literal arrays and ranges, the target checks the actual DSLX element width when typechecking permits a wider named parameter.

Anonymous callbacks vary bit widths, signedness, nested maps, helper calls, result shapes, and invalid annotations. Runtime inputs exercise captures of parameters, shadowed lets, outer map elements, and counted-loop indices, carries, and local bindings. Module constants exercise compile-time capture evaluation. The target detects lost or misresolved captures, incorrect types, and element reordering. It checks BX rejection against DSLX's executable element width when a wider annotation would change a literal input, a constant callback result, or a captured loop index or literal. In those cases DSLX can accept the annotation while returning narrower values, including through scalar and tuple-destructured aliases. Captured callbacks returning tuple or array literals use result annotations because the pinned typechecker cannot infer their result. Capture-free callbacks with inferred aggregate literals use concrete execution because the converter emits unparsable helper names.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_map_lambda --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_function_parametrics.rs

Compares BX functions specialized on named callbacks with independently generated DSLX programs that call the corresponding helpers directly. Generated calls vary aggregate and scalar signatures, signedness, bit widths, callback identity and order, counted folds, and public DSLX imports with inferred parametrics. The target compares acceptance and evaluates accepted programs on the same inputs through BX block IR, native DSLX IR, and concrete DSLX calls. It detects incorrect signature checks, incorrect specialization reuse, and misresolved callback targets.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_function_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_function_type_parametrics.rs

Generates calls to a pure BX zip function over arrays with independently chosen element and result types. Inferred lengths, explicit lengths, named generic identity maps, and callbacks returning records, tuples, arrays, or bits exercise type binding and specialization reuse. The target checks acceptance against concrete DSLX signatures and compares BX IR with unrolled DSLX calls in compiled IR and concrete evaluation. It catches misbound nominal types, mismatched lengths, and incorrect callback results or element order.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_function_type_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_function_value_parametrics.rs

Generates direct calls and named maps over value-parametric BX functions and imported DSLX functions. It compares acceptance, BX block IR, native DSLX IR, and concrete DSLX evaluation. Calls vary positional prefixes, inferred widths and signedness, dependent defaults, and scalar or nominal values. Inputs include nested aggregates, inline literals, and let-bound values. Repeated calls with equal runtime signatures exercise specialization reuse, including bindings that differ above bit 63. Rejected cases check conflicting or missing dimensions, nonconstant actuals, and nominal values from the wrong declaration.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_function_value_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_block_callbacks.rs

Generates type-generic BX blocks with named callbacks, flattens their combinational child instances, and compares acceptance and computed outputs with ordinary DSLX calls using concrete inputs and compiled DSLX IR. It varies signedness, widths, aggregate shapes, callback identity in each position, and imported generic functions at different widths. It catches incorrect forward inference, stale specialization reuse, and mismatched argument or result shapes.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_block_callbacks --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_block_value_parametrics.rs

Compares explicit `u32` value parametrics on BX blocks with DSLX functions.
Calls vary literal, named and braced constant arguments, inferred payload
types, and forwarding through a second template. A structural unroll uses
the value as its bound. Repeated calls at equal and distinct bounds exercise
specialization reuse with unchanged port types. Invalid samples cover missing,
extra, mistyped, and runtime arguments. The target checks source acceptance
and compares flattened BX Block IR with both compiled and constexpr DSLX.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_block_value_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_block_let_patterns.rs

Compares destructuring lets in BX blocks with the same patterns in DSLX functions. Generated programs vary nested tuple shapes, wildcards, rest spans, type annotations, inferred integer fields, lexical shadowing, and direct or child-block initializers. Invalid patterns exercise duplicate binders, extra rest markers, and shape or type mismatches. Accepted programs return selected fields or compare inferred operands, including decimal and radix-prefixed values of different widths and signs inside structural unrolls. The target compares BX block IR with compiled DSLX IR and native DSLX evaluation on shared inputs. For implicit conversions that can abort the pinned converter, it first checks the original source with both frontends, then uses explicit casts or native array-element unification to give the DSLX execution probe the common operand type.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_block_let_patterns --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_module_constants.rs

Generates module constants with dependent scalar expressions, nested aggregates, and calls to earlier pure functions. BX blocks and local functions read them in expressions, named match patterns, map callbacks, loop bounds, and slice widths. Constants also set element signedness, bit widths, and array lengths in annotations. Counted traversals use the same length constant as the array they index. The target compares BX and pinned DSLX acceptance, including shadowing, forward references, duplicate definitions, builtin name collisions, and invalid dimension types. It checks every constant's name, declaration order, concrete type, and computed value against DSLX metadata, including signedness and width inside tuples and arrays. It compares block results on shared inputs against both compiled DSLX IR and concrete DSLX calls.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_module_constants --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_type_aliases.rs

Compares concrete module type aliases in BX and pinned DSLX. Generated chains refer to booleans, signed and unsigned bits from zero to 129 bits wide, nested arrays and tuples, enums, and structs including specializations with identical fields. The target checks source acceptance, constant values, and runtime outputs in BX block IR, compiled DSLX IR, and concrete DSLX calls. Paired library and consumer modules exercise qualified and directly imported public aliases, re-exported aggregates of private types, and same-named types in different scopes. Rejected programs expose lost nominal identity, invalid alias targets, declaration-order mistakes, and access to private types.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_type_aliases -- -max_len=13 -len_control=0`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_module_graph.rs

Generates small DSLX module graphs with two distinct, same-shaped structs,
public aliases, private declarations, enum values, and field dimensions computed
in another module. The target compares the Rust outline and the checked BX
layout with native DSLX metadata, including a field passed directly from a
function result to a concrete or generic consumer. It also evaluates projected
fields through both frontends. Passing an alias, passing the other module's
struct, and naming a private declaration test type acceptance independently of
value equality.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_module_graph --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_imported_struct_parametrics.rs

Generates DSLX structs with two value parameters and two specializations that share a field layout. One parameter can be absent from the fields, or both can contribute to the same field width. BX blocks and local functions use concrete specializations in signatures, let annotations, and struct constructions alongside public aliases and inferred call results. The generator varies bare and typed `u32` arguments, omitted trailing defaults, and module-qualified and directly imported struct names. The DSLX reference qualifies imported constructors. Both frontends evaluate a consumer that observes both parameters on shared inputs and reject passing or assigning a result of the other specialization. This exposes lost bindings, incorrect specialization reuse, and field layout errors.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_imported_struct_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_struct_parametrics.rs

Generates native BX and DSLX struct templates with scalar and array fields. Cases vary explicit and inferred arguments, derived defaults, nested symbolic applications, widths inferred from unsized fields, and `xN[S][N]` signedness with signed or unsigned width parameters. Signed arguments vary decimal, hexadecimal, and binary spellings at the sign bit and beyond the declared width. Phantom parameters distinguish instances with identical fields using enum members or bit patterns that can differ only above bit 63. Nested records forward values between signed, unsigned, and boolean formals and check whether each value fits the destination. Some BX cases use turbofish for the same applications spelled with plain angle brackets in DSLX. Rejected programs exercise incompatible specializations, wrong nominal values, conflicting field types, and extra arguments. The target compares source acceptance and checks projected data and inferred bindings against both compiled DSLX IR and concrete DSLX calls.

DSLX v0.62.0 forwards nonnegative values from sources wider than 64 bits as hexadecimal literals, allowing the destination's sign bit to turn a positive value negative. For that overflow case the reference requires a constexpr `checked_cast` to check BX's value-preserving rule. For tuple-, array-, and struct-valued parameters, which the pinned DSLX typechecker cannot specialize on structs, the target compares BX results to DSLX functions that take those same constants as ordinary arguments.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_struct_parametrics --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_struct_type_parameters.rs

Compares type-parametric BX block results with concrete DSLX struct definitions and pure functions. A generic BX block returns a named record with an inferred bits, tuple, array, or nominal payload type. Calls reuse and vary payload types, and constructors use implicit context or explicit type arguments. The target checks that a result from a distinct specialization is rejected when assigned to the wrong nominal type. On accepted programs it compares the named field values from flattened BX Block IR with compiled and constexpr DSLX results.

Run from `xlsynth-bx` with `cargo +nightly fuzz run fuzz_bx_dslx_struct_type_parameters --sanitizer none`.

### xlsynth-bx/fuzz/fuzz_targets/fuzz_bx_dslx_struct_updates.rs

Generates struct updates ranging from a base-only copy to replacement of every field. It varies field order, signed and unsigned widths, nested payloads, and inferred, partial, or fully specified value parametrics. Nested struct literals and loop carries exercise width inference from the update constructor, its result type, or a sibling in an array or conditional. For accepted programs it compares the number of `useless_struct_splat` warnings and every field from BX block IR, DSLX IR, and concrete DSLX calls on the same inputs. Rejection cases cover unknown and duplicate fields, incompatible base and field types, and a comma after the final splat. Calls and conditionals that produce a parametric base are bound with a `let` before the update for compatibility with the pinned DSLX typechecker.

Run from `xlsynth-bx` with `cargo +nightly fuzz run --sanitizer none fuzz_bx_dslx_struct_updates -- -max_len=14 -len_control=0` to vary all six program-generation bytes from the start.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_abc_sequential_mapping.rs

Uses the shared resettable block-IR generator, lowers to G8R with explicit reset
and enable controls, and maps through standalone ABC plus native FF selection
and complete flat netlist assembly. Mapping must legalize the controls; the
reference simulation evaluates the unlegalized source directly. Independently
sampled optional input/output register layers cover unreset pipeline boundaries.
Both register-cleanup settings are checked on every sample. The original and
mapped designs receive independently sampled startup states and shared concrete
input vectors; outputs must agree after reset has propagated through the optional
boundary layers, including later reset reassertions in a 24-cycle run.
The final emitted GV is loaded through the
gate-level evaluator, exercising physical FF polarity/tie-offs, packed ports,
constants, aliases, and reset/enable legalization. An independent connectivity
oracle follows wire aliases (not buffers) and checks that direct input-to-D and
Q-to-output connections survive mapping. Separate directed tests cover
zero-width logical ports, which are preserved natively and omitted from physical GV.
No special optimization motifs are injected into the generic generator.
The configured `abc_mapping::tests::configured_abc_mapping_smoke` unit test
also runs this equivalence oracle with a shorter custom mapping script; the
fuzz target itself uses the default speed script.

This optional target requires `XLSYNTH_ABC_PATH`, comma-separated uncompressed
`XLSYNTH_LIBERTY_FILES` for ABC, and `XLSYNTH_LIBERTY_PROTO` for native FF mapping
and evaluation. The libraries must describe matching cells at one corner.
The proto must select a usable `reference_buffer_cell` and a positive finite
`default_output_load` in declared library capacitance units (set with
`lib2proto --reference-buffer-cell CELL --default-output-load CAPACITANCE`).
Mapping uses that driver and the explicit load on mapped logic outputs; direct
connections bypass ABC. Startup preflight requires a non-inverting FF for a
direct one-register identity and verifies actual combinational mapping;
setup failures and unexpected sample
failures are errors, not successful/skipped samples. Run with
`cargo fuzz run --sanitizer none fuzz_abc_sequential_mapping -- -max_total_time=300`.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_sequential_cleanup_equiv.rs

Generates bounded synchronous block IR with a reset value on every register
and at least one observable output,
then lowers the reset/enable logic to G8R and runs register cleanup. For each
sample, eight independently randomized source/implementation startups are
simulated for 16 cycles, with reset asserted across the first clock edge and
occasionally reasserted later. An independent block evaluator checks the original
G8R on every cycle; cleaned G8R must match all external outputs after the first
reset edge. Pre-reset implementation outputs are deliberately not compared.
Registers and ports may be packed or aggregate, and state may contain feedback.
The target also checks metadata, interface preservation, reduction accounting,
determinism, and idempotence. It detects changed reset/enable behavior, incorrect
constant or duplicate substitutions, and broken packed-register reconstruction.
Graph generation contains no injected cleanup motifs. The first eight bytes seed
independent evaluation entropy; remaining bytes drive the generic block generator.
The deterministic smoke sweep checks that its samples exercise dead, constant,
and duplicate register removal. An output is reserved before body generation so
exhausting the node/entropy budget cannot make all output comparisons vacuous.

This oracle covers reset-reachable behavior, not arbitrary unreset startup.
Focused unit tests cover mixed-reset data/valid pipelines and corresponding-state
checks for unreset constant and duplicate substitutions. Arbitrary partially
reset feedback graphs remain outside the reset-based fuzz oracle.

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_roundtrip.rs

Generates an upstream-standard acyclic random package directly as PIR, including
helper functions, `invoke`, `map`, `counted_for`, `gate`, arbitrary-width multiply,
product-pair, token, and event forms (`after_all`, `assert`, `trace`, and
`cover`), parses and re-emits it through libxls, then reparses with the Rust
parser and checks IR-level structural equivalence across the PIR roundtrip.
Panics if either implementation rejects the emitted IR or if the top function
is missing.

Primarily tests:

- libxls parse/re-emit IR text compatibility with the Rust parser
- Function/package pretty-printer roundtrip soundness
- Structural equivalence stability across roundtrip, including event metadata

### xlsynth-pir/fuzz/fuzz_targets/fuzz_known_bits_soundness.rs

Generates pure typed PIR graphs directly from libFuzzer bytes, including all six
extension operations, aggregates, zero-width bits, and widths through 257.
The first eight bytes independently seed concrete-input generation; the
remaining bytes drive graph construction. Each graph is analyzed once and
evaluated on eight input sets (once for nullary functions). The shared
`generate_mixed_argument_sets_with_rng` helper uses the
[shared mixed-vector policy](#shared-concrete-input-sampling), independently
selecting structured vectors or a fresh subset of special-valued arguments for
each evaluation, with optional sparse perturbation of special values.
Every node and aggregate leaf must satisfy both its known-bit masks and its
minimum/maximum population bounds, including parameters and dead nodes. These
are node-local facts; the target checks their concrete membership without
recognizing graph neighborhoods. It also requires identical facts when the
function is represented as a combinational block. It flags unsound facts,
missing value callbacks, unexpected generation/analysis/interpreter failures,
and function/block disagreements. No XLS oracle or solver is invoked.

Run a bounded campaign from the repository root with:

```bash
cd xlsynth-pir
cargo fuzz run --sanitizer none fuzz_known_bits_soundness -- -max_total_time=60 -timeout=15 -max_len=4096
```

For new concrete-value fuzz targets, prefer this shared mixed-input policy over
uniform-only sampling so wide corner values receive meaningful coverage.

### xlsynth-pir/fuzz/fuzz_targets/fuzz_range_analysis_soundness.rs

Generates pure typed function graphs, including all six extension operations,
aggregates, zero-width bits, and widths through 257, and checks every node's
range against concrete interpretation. The first eight bytes seed the shared
mixed-input sampler independently of graph construction. Eight evaluations
(one for nullary functions) cover parameters and dead nodes as well as the
return value. It also requires identical facts after conversion to a
combinational block. Unexpected generation, analysis, or evaluation failures,
missing callbacks, unsound intervals, and function/block disagreements fail
the sample. This target invokes neither XLS nor a solver.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_range_analysis_block_soundness.rs

Generates general sequential blocks with registers, optional resets/enables,
aggregate ports, and extension operations. Independent input entropy drives
eight mixed-sampler arbitrary-Q initial states, each followed for two cycles
with new port inputs; Q is not restricted to reset values or reset-reachable
states. The observed cycle interpreter checks every node's value, including
register reads, dead nodes, and unit sinks, against the block's unconditional
intervals. Missing values, unexpected generation/analysis/evaluation errors,
and excluded concrete values are failures. The existing block observer lives
in the g8r fuzz crate, avoiding
a new dependency cycle. No external tool or solver is invoked.
The same observed values also validate known-bit masks and population bounds,
including arbitrary current register values and next-state transitions.

Run either target from its crate directory with
`cargo fuzz run --sanitizer none <target> -- -max_total_time=60 -timeout=15 -max_len=4096`.

### xlsynth-pir/fuzz/fuzz_targets/fuzz_ir_opt_equiv.rs

Builds an XLS IR package from an upstream-standard random sample, including
`gate` and arbitrary-width multiply but excluding product-pair operations
pending prover support, runs IR optimization, and cross-checks equivalence
between original and optimized IR using an external tool and available SMT
backends. Flags disagreements between engines and unexpected failures.

Primarily tests:

- Optimization preserves semantics
- Cross-engine equivalence consistency (external tool vs SMT backends)

### xlsynth-aug-opt/fuzz/fuzz_targets/fuzz_aug_opt_equiv.rs

Generates an upstream-standard random XLS IR function, including `gate` and
arbitrary-width multiply but excluding product-pair operations pending formal
support, runs one PIR-only aug-opt round with the default g8r cost evaluator,
and checks in-process Bitwuzla equivalence between the original and rewritten
IR when at least one rewrite fires. The generator stays generic, without
directed cases for particular rewrites. The target flags unexpected aug-opt
failures or inequivalent rewrites.

Primarily tests:

- PIR aug-opt rewrites preserve semantics when they apply
- End-to-end compatibility of PIR lowering and in-process equivalence checks

### xlsynth-aug-opt/fuzz/fuzz_targets/fuzz_constant_shift_choices.rs

Decodes small typed DAGs directly from fuzz bytes to exercise constant-shift-choice
fusion, including nested selects, priority/default choices, replicated Boolean
masks, literal-prefix concats, shared controls, wide shift amounts, and slices.
Explicitly eligible cases must reach the evaluator; valid near misses and limit
boundaries also exercise rejection. The target uses forced acceptance, real g8r
costing, mixed decisions, and injected evaluator failures, including failures
after an earlier accepted site.

Bitwuzla proves both the completed local incumbent/candidate pairs (including
rejected alternatives and all retained outputs) and complete rewritten functions
equivalent through direct PIR-to-SMT translation. Small effectful cases exhaust
inputs and compare full interpreter results. Structural checks cover local graph
closure, unchanged signatures, and exact rollback. Failures expose incorrect
bit projections, selection semantics, lost shared users/effects, or speculative
changes that survive rejection. Progress counters distinguish completed proofs
from inconclusives and report exercised decision paths.

From `xlsynth-aug-opt`, with the normal XLS and system Bitwuzla environment:

```shell
cargo +nightly fuzz run fuzz_constant_shift_choices \
  --features with-bitwuzla-system --sanitizer none -- \
  -max_total_time=3600 -timeout=90 -print_final_stats=1
```

### xlsynth-aug-opt/fuzz/fuzz_targets/fuzz_priority_result_fusion.rs

Generates focused, typed priority encode/index/decode and scalar-predicate graphs
with bounded affine
operation sequences, either priority, predicates, wide constants, direct/expanded
decoders, sharing, and valid rejection controls. It proves the forced semantic
candidate and the real off/on PIR-only or one/three-round sandwich results against
the original with direct Bitwuzla translation; small signatures also exhaust
interpreter inputs. Cost ties/errors must preserve the exact input. Failures expose
wrong wiring, zero-input/wrapping/overshift errors, incomplete overflow guards,
unsafe sharing, speculative changes, and composition regressions. Named counters
and a corpus replay audit require actual proved fusions and checked rejection paths;
solver inconclusives do not count as proofs. See the
[aug-opt fuzz guide](xlsynth-aug-opt/fuzz/FUZZ.md) for seed generation, campaign, and
coverage validation commands.
Scalar families cover reductions and literal equality/unsigned comparisons. The
audit requires proved real-cost acceptance in PIR-only mode for each family and
priority direction, and checks width/wrap/sentinel/sharing controls. CI's Fuzz Smoke
Test job requires this standalone audit to pass before the short mutation smoke
test and retains the coverage counts in its log.

The existing target also runs a bounded scalar-bit proof matrix during
initialization, before random mutation, with no additional binary or CI step.
It covers bare count bits, nested reductions/comparisons, independent and dependent
predicate groups, separately observed slices, retained count/amount users, and
unsupported multibit/truncated/variable/arbitrary-encode inputs. Exact forced
rewrite counts prevent a missed match from passing on equivalence alone.
Widths, both priorities, zero sentinels and wrapping/widening arithmetic vary in
the byte grammar; startup requires proved acceptance and rejection coverage.

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
deterministic input samples using the shared mixed-vector policy. Input generation
is seeded from the generated IR text so coverage-guided entropy is dedicated to exploring program
structure instead of stimulus values.

Primarily tests:

- End-to-end prep, gatify, FRAIG, reassociation, and cut-DB rewrite correctness
- Aggregate input/output flattening stays aligned between PIR and g8r evaluation
- Broad concrete-value equivalence at higher throughput than formal proof

### xlsynth-lean/fuzz/fuzz_targets/fuzz_ir_fn_lean_eval.rs

Generates pure, acyclic PIR packages, exports their selected function directly
to word-level Lean, and compares every typed output with `eval_fn_in_package`
on the original package. Each fuzz input checks one exported function with
32 argument sets. The fuzzer imports the generated module using a name
selected from `Generated.Function`, `BitVec`, `Vector`, and `Nat.Generated` by
hashing the input. This exercises Lean name resolution in different namespaces.

The first eight bytes seed the argument sampler (zero-padded for short inputs).
Additional bytes, up to a total of 3,072, drive graph generation directly so
libFuzzer can mutate and shrink the graph. For inputs of eight bytes or fewer,
the same seed drives `RngEntropy` with a budget of 1 to 32 body nodes.
Packages include nested arrays/tuples, calls, maps,
PIR extensions, zero-width and wide bits. The generator excludes loops,
events, and partial-product multiplies and only
generates proven-safe bounds promises. Unexpected export, elaboration,
execution, or value discrepancies fail the sample. The external Lean
executable is required at startup.

```bash
XLSYNTH_LEAN=/path/to/lean/bin/lean cargo fuzz run \
  --fuzz-dir xlsynth-lean/fuzz --features external-lean --sanitizer none \
  fuzz_ir_fn_lean_eval -- -timeout=45 -max_len=3072
```

### xlsynth-lean/fuzz/fuzz_targets/fuzz_gate_fn_lean_eval.rs

Differentially evaluates bounded, directly generated combinational `GateFn`s
through `PreparedGateSim` and the `xlsynth-lean` export/evaluation path. Each
sample generates up to 256 nodes and permits constants, complemented edges,
sharing, dead logic, and independently sized input/output ports, including
zero-width and greater-than-64-bit vectors. A generic node permutation exercises valid
graphs whose stored IDs are not topologically ordered. The target compares 32
input vectors from the shared mixed-vector sampler and surfaces graph export,
bit ordering, interface, or Lean evaluation mismatches. Failures report the
fuzz bytes, seed, textual native g8r artifact, inputs, and available outputs.

Set `XLSYNTH_LEAN` to the actual executable in a compatible Lean toolchain.
The evaluator initializes once at fuzzer startup. Each graph writes an ASCII
AIGER sidecar, compiles an exported Lean module, and evaluates its full input
batch through a generated wrapper. Missing or unusable Lean setup fails startup;
unexpected export or evaluation errors fail the fuzz sample. The non-default
`external-lean` feature keeps this external-tool target out of the ordinary
smoke suite unless explicitly requested.

```bash
XLSYNTH_LEAN=/path/to/lean cargo fuzz run \
  --fuzz-dir xlsynth-lean/fuzz --features external-lean --sanitizer none \
  fuzz_gate_fn_lean_eval -- -timeout=30 -max_len=2048
```

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

Both `fuzz_g8r_evaluation` and `fuzz_g8r_formal` require Bitwuzla because the
canonical production g8r pipeline enables formal array alias analysis. Run the
targets against a system Bitwuzla installation with:

```bash
cargo fuzz run --features=with-bitwuzla-system --sanitizer=none fuzz_g8r_evaluation
cargo fuzz run --features=with-bitwuzla-system --sanitizer=none fuzz_g8r_formal
```

Use `with-bitwuzla-built` instead of `with-bitwuzla-system` to build the solver
from source.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_sentinel_lowering.rs

Exercises canonical CLZ and both priority sentinel lowerings, with width, offset,
wrap, sharing and builder-fold controls. Explicit Bitwuzla checks compare the
lowered gates to desugared source semantics, including concrete zero inputs.
A deterministic coverage audit requires actual canonical lowerings and completed
proofs for all families; generic fuzz smoke success alone is insufficient. See
the [g8r fuzz guide](xlsynth-g8r/fuzz/FUZZ.md) for reproducible commands and limits.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_gate_fn_roundtrip.rs

Builds a random `GateFn`, serializes to text, parses it back, and checks structural equivalence of the original vs parsed `GateFn`.

The shared `gate_random::generate_gate_design` engine reserves a separate
fuzzed seed for output wiring. The combinational profile disables registers.
Its first output selects a generated operation when one exists; additional
outputs can select any input or operation. This also gives the gate transform
targets observable generated logic instead of systematically selecting inputs.
Builder optimizations can still simplify selected operations to inputs or constants.
All stored nodes, including dead logic, are serialized; parse failures are sample
failures rather than silently skipped.

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

Differentially compares our Rust IR function interpreter with the xlsynth C++ interpreter on the same directly generated acyclic PIR package, using the shared mixed-vector sampler plus a bounded autocov corpus of interesting typed inputs.

- Generates an upstream-standard random PIR package, including helper functions, `invoke`, `map`, `counted_for`, `gate`, and arbitrary-width multiply, via `xlsynth_pir::ir_random`, then parses its emitted IR through libxls for the reference interpreter.
- Draws 32 shared mixed input vectors, seeded from the generated IR, then runs autocov to add semantically interesting tuples. Nullary functions are evaluated on their single empty argument vector without autocov.
- Checks the return values from both the event-reporting PIR evaluator and the memoized value-only evaluator against libxls on every sample, including division/modulus edge cases and composite-valued `one_hot_sel` cases.

Primarily tests:

- Autocov-selected boundary-ish inputs expose the same interpreter semantics as direct xlsynth interpretation
- Generated IR text remains parseable and executable across the C++ and PIR interpreters
- Arbitrary generated ops and value shapes behave identically in both interpreters across mixed random and autocov-selected inputs

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

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_mapped_netlist_proto_roundtrip.rs

Generates small acyclic mapped GV designs with four input bits. It varies
the connectivity, cell types, pin order, literal tieoffs, port declarations,
implicit wires, comments, line endings, and directive layout. After converting
the GV to protobuf and back, Icarus Verilog elaborates both designs with the same
cell models and compares every gate output for all 16 binary inputs using
four-state equality. This catches lost modules, invalid emitted Verilog, and
wiring changes. The target requires `iverilog` and `vvp`.

### xlsynth-g8r/fuzz/fuzz_targets/fuzz_netlist_resize.rs

Uses the shared direct-G8R generator to construct combinational and sequential
graphs, lowers them structurally into a small slew/load-sensitive Liberty
library, randomizes initial cell strengths, and runs bounded resizing. It compares
cycle simulation of the original G8R graph, mapped netlist, and resized netlist
from the same randomized initial state. It also checks deterministic output,
all four boundary timing limits, recomputed area, area limits, unchanged
connectivity, and independent per-phase evaluation budgets.
Round counts, candidate limits, evaluation budgets, and register resizing are
varied, including graph-scaled defaults, explicit zero budgets, disabled phases,
and frozen register cells.
The library also varies optional capacitance, slew and fanout limits, including
clock-pin slew limits and input fanout weights, while retaining unconstrained
samples. This exercises electrical rejection and rollback as limits appear or
disappear across functionally equivalent cell sizes.
The resizer's independent full-STA checks also expose stale incremental timing
or rollback state. The generator uses nonconstant, unfolded expressions to
satisfy the structural mapper's literal-free contract. It generates ordinary
registers without explicit reset/enable controls because resizing consumes
already-legalized mapped netlists.

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
behaves consistently across native PIR interpretation and `xlsynth-vastly`
simulation paths (including combo and pipelined forms) when supported by the
sample/tooling.

Primarily tests:

- End-to-end semantic equivalence between native PIR interpretation and XLS-generated
  Verilog/SystemVerilog simulation
- Codegen/simulation consistency across combo and pipelined lowering paths
  under random typed inputs
- Pipeline-oracle coverage up to the configured multiplication-dependency-depth
  budget; deeper chains still exercise the combinational oracles without
  invoking resource-pathological upstream scheduling

______________________________________________________________________

Notes:

- Degenerate inputs (e.g., zero-width, empty ops) are typically skipped with a brief comment explaining why they are not informative for the property under test.
- Unexpected failures in normal API operations (e.g., pretty-printed text failing to parse) should be flagged (panic) to surface systemic issues early.

## `fuzz_sequential_legalization_equiv`

Generates generic synchronous stateful block IR with optional enables, optional
resets, feedback, and zero-width values. It lowers to explicit-control
`SequentialGateFn`, round-trips text or binary serialization, legalizes the
controls, and evaluates both forms against the independent block interpreter.
Eight arbitrary common startups and 16 cycles are checked per sample, including
cycle zero; no reset warm-up is assumed, so unreset state is covered. The target
checks external outputs and every register's next state, retained metadata,
serialization fidelity, and legalization idempotence. It catches reset/enable
priority or polarity errors, incorrect holds or simultaneous updates, bit-order
mistakes, state loss, and dropped serialized controls. Unlike register-cleanup
fuzzing, this is exact equivalence, not post-reset equivalence under startup
don't-care semantics. The deterministic smoke test requires enabled, resettable,
and unreset register coverage without injecting directed graph patterns.

Run from `xlsynth-g8r` with:

```shell
cargo fuzz run --sanitizer none fuzz_sequential_legalization_equiv -- -max_total_time=300
```

## Shared direct-G8R generation

All direct graph consumers use the library API
`xlsynth_g8r::gate_random::generate_gate_design` with `RandomGateOptions`.
The previous general `FuzzGraph` builder and bulk-replacement-only
`FuzzGateGraph` builder have been removed. Function/block-IR-derived targets
continue to use the IR generators; they provide complementary lowering coverage.

Profiles control interface counts and packed widths (including zero widths),
the actual AIG-node budget, Boolean construction operations, folding/hashing,
optional registers and control polarities, endpoint sharing, and output
observability. Optional chain, redundant-pair, and fanout distributions are
disabled by default and explicitly enabled only by the bulk-replacement target.
Seeded RNG sweeps (independent of libFuzzer guidance) require emitted and live
coverage of all enabled Boolean operations, varied interface shapes/widths,
feedback and feedforward state, all reset/enable combinations, both polarities,
and shared nonempty data/control endpoints. Data wiring samples both whole-port
reuse and independently selected bits.

Graph entropy is separate from the shared `xlsynth_pir::random_inputs` sampler
used for concrete inputs and startup state.

Text/AIGER round trips, both gate-transform targets, cut rewriting, dynamic
hashing/depth, cut replacement, and bulk substitution use the combinational
profile. Their input schema is now byte-stream graph entropy rather than the
older derived graph-description structs; old corpus bytes remain usable but
do not necessarily reconstruct their previous graph. Edit/substitution
sequences remain target-specific inputs.

## `fuzz_direct_g8r_sequential_equiv`

Directly generates bounded sequential AIGs without an IR lowering step, including
packed state, feedback, optional resets/enables, both control polarities, shared
endpoints, constants, dead logic, and zero-width values. It round-trips both
native formats and compares their metadata and cycle-by-cycle outputs/state,
then compares the legalized design from the same arbitrary initial values.
Each sample uses four startups and twelve cycles, including cycle zero, without
a reset warm-up. It detects lost controls, polarity/priority mistakes, incorrect
holds or feedback, serialization corruption, and non-idempotent legalization.
The smoke test checks that state and both polarities actually occur. The block-IR
legalization target remains separate to provide an independent block oracle.

```shell
cargo fuzz run --sanitizer none fuzz_direct_g8r_sequential_equiv -- -max_total_time=300
```

## `fuzz_blif_roundtrip`

Uses the shared direct-G8R generator for both combinational and sequential
samples, including packed ports/state, feedback, shared endpoints, and optional
reset/enable controls of either polarity. It legalizes controls into effective-D
logic before BLIF emission, parses the emitted file, and repeats the round trip.
Port names/order/widths, register names/order/widths, and the clock must survive.
Across four arbitrary startup states and twelve cycles, outputs and every
register's next state must match the explicit-control source and legalized graph
from cycle zero, with no reset warm-up. The profile excludes zero-width ports
and registers because plain BLIF cannot represent them, but permits empty
interfaces. Unexpected emission or parsing failures fail the sample. This
catches lost aliases, latch/bit ordering errors, polarity mistakes, and lost
control behavior. A seeded smoke sweep checks combinational, registered,
controlled, both-polarity, and packed-interface coverage.

```shell
cargo fuzz run --sanitizer none fuzz_blif_roundtrip -- -max_total_time=300
```

## `fuzz_gate_fn_aiger_binary_roundtrip`

Uses the same generator without registers to round-trip binary AIGER with and
without symbol tables. It checks the binary parser and automatic format
selection, restores packed ports with the original interface schema, and checks
structural equivalence, deterministic emission, repeated round trips, and
simulation on sixteen shared mixed-distribution input vectors per symbol mode. Zero-width ports,
empty interfaces, constants, and dead logic are included. This catches varint
and literal encoding errors, parser disagreement, lost port grouping, and bit
ordering mistakes. A seeded smoke sweep verifies that zero-width/packed ports
and nontrivial logic are exercised. These are valid-graph round trips, not
malformed-byte parser fuzzing.

```shell
cargo fuzz run --sanitizer none fuzz_gate_fn_aiger_binary_roundtrip -- -max_total_time=300
```

### `fuzz_arrival_arithmetic_rows`

Focused Bitwuzla equivalence coverage of the common arrival-aware arithmetic-row
mapper. Generates add-only sums and signed/unsigned or constant products, mixed
negation, delayed inputs, repeated contributions, and shared/sliced/truncated
product boundaries. Asserts deferred-product hits and resolved negative controls
using the actual lowering map. Solver timeouts are inconclusive, not proofs.

### Low-prefix nonzero predicates

`fuzz_low_bit_nonzero` checks direct and conditional negation, reversed arms,
contiguous decomposed prefixes, and rejection controls for retained numeric
consumers, offset/gapped slices, bitwise NOT, and dead predicates. It proves
both forced eligible rewrites and the ordinary PIR/sandwich output with
Bitwuzla, checks cost-tie/error rollback, and audits accepted rewrites in all
three pipeline modes. Retained arithmetic must reject before gate costing.

### Projected positive-mask preparation

`fuzz_projected_positive_mask` generates valid matching and near-matching PIR,
checks whether preparation rewrites the comparison, and proves original versus
prepared equivalence with Bitwuzla. It covers both AND operand orders, direct and
XOR input expressions, shared comparisons/negations/masks, safely shared final
ANDs, and constant/source/comparison/extension/slice/AND-shape mismatches.
The target's initialization runs a deterministic 632-proof audit once before
random exploration, including in the existing CI fuzz smoke. It covers every
category at widths 3, 4, 8, 16, 32, 64, 65 and 129, plus width-2 matching/sharing
and width-1 nonmatching cases. Proofs quantify all input values, including zero
and signed extrema. This is targeted coverage, not an exhaustive enumeration of
all IR graphs. Run with
`cargo fuzz run --sanitizer none fuzz_projected_positive_mask --features with-bitwuzla-built -- -max_total_time=30`
from `xlsynth-g8r`; use `-runs=0` instead of `-max_total_time=30` to run the audit and existing seed corpus without random exploration.
