# Plan: a Cranelift JIT for PIR functions

This document sketches a Rust-native JIT backend that consumes
`xlsynth_pir::ir::Fn` and generates executable code with Cranelift. It is an
initial design for implementation work, not a commitment to support the full
XLS execution surface.

The existing libxls-backed compilation inventory is documented in
[../../xlsynth/docs/jit_aot_libxls_architecture.md](../../xlsynth/docs/jit_aot_libxls_architecture.md).

## Initial goal

Build a library JIT for PIR **functions**:

```text
xlsynth_pir::ir::Fn
  -> validation and layout planning
  -> PIR node lowering with cranelift_frontend::FunctionBuilder
  -> cranelift_jit::JITModule
  -> callable Rust wrapper
```

The first version should:

- Compile and run pure functions.
- Support a native typed entrypoint which reads inputs and writes results
  directly through borrowed Rust storage without packing each call.
- Have an explicit, testable native value layout independent of libxls and
  LLVM.
- Use caller-allocated, aligned scratch storage for materialized temporary
  values rather than emitting potentially large stack allocations.
- Reject unsupported types or operations at compile time with structured
  errors.
- Leave an architecture that can later share its lowering with
  `cranelift_object::ObjectModule` for AOT object emission.

Not in the first version:

- Procs, blocks, channels, registers, or continuation points.
- Trace, assert, or cover event handling.
- Compatibility with existing libxls AOT object metadata.
- Full support for all PIR nodes before the basic execution model is proven.

## Important dependency boundary

Taking PIR as input does not by itself remove every libxls dependency today.
`xlsynth_pir::ir::NodePayload::Literal` stores `xlsynth::IrValue`, and the
existing evaluator API also accepts and returns `xlsynth::IrValue`.
`xlsynth::IrValue` is backed by the libxls C API.

There are therefore two separable objectives:

1. Replace libxls **JIT compilation and execution** with Cranelift.
1. Remove libxls from the PIR value, literal, and parsing boundary.

A native typed JIT entrypoint can avoid `IrValue` for runtime inputs and
outputs while proving the compiler backend. Literals in `ir::Fn` remain an
issue until they are copied into a native compile-time constant form or PIR
stores them independently of `xlsynth::IrValue`. A complete DSO-free route
needs Rust-native PIR literal/value representation.

## Crate placement

The backend should be reusable library functionality, not logic embedded in
`xlsynth-driver`.

A separate library crate, `xlsynth-pir-compiler`, is preferable to
putting Cranelift directly into `xlsynth-pir`:

```text
xlsynth-pir-compiler
  -> xlsynth-pir
  -> cranelift-codegen
  -> cranelift-frontend
  -> cranelift-module
  -> cranelift-jit
  -> cranelift-native
```

This keeps Cranelift dependencies out of users that only parse or transform
PIR. During the transitional `IrValue` period it indirectly retains the
existing `xlsynth-pir -> xlsynth` dependency; decoupling PIR values is separate
follow-up work.

Do not put a new native JIT implementation in `xlsynth`, because
`xlsynth-pir` already depends on `xlsynth`; making `xlsynth` consume PIR would
create a crate cycle.

## Native typed entrypoints

The primary performance-facing API should be zero-copy: generated code receives
pointers into native Rust input and result values. It should not marshal a
struct into a separate byte buffer merely to enter generated code.

This is possible with constraints:

- Native bridge types must have specified layout, normally `#[repr(C)]`
  generated structs and fixed-size arrays.
- Plain Rust tuples and ordinary Rust-layout structs are not an ABI contract
  and must not be accessed by generated code using assumed offsets.
- By-value C struct parameters are expressible in Cranelift via
  `ArgumentPurpose::StructArgument`, but pointer parameters are preferable
  here: they avoid target-specific register classification and avoid copying
  large argument values.
- Result storage should be supplied as `&mut Output`, allowing generated code
  to write the final result directly in place.
- The safe wrapper must enforce lifetimes, exclusive result/scratch access,
  valid output values, and exact correspondence between PIR types and Rust
  bridge layouts.

For example, a generated or explicitly bound native interface can look like:

```rust
#[repr(transparent)]
pub struct Bits1(u8); // Private storage; constructors accept only 0 or 1.

#[repr(transparent)]
pub struct Bits24(u32); // Private storage; constructors clear/check high bits.

#[repr(C)]
pub struct Bits257 {
    limbs: [u64; 5], // Least-significant limb first; high padding masked.
}

#[repr(C)]
pub struct Output {
    valid: Bits1,   // PIR bits[1], not Rust bool at the machine-code ABI.
    value: Bits24,
}

pub struct CompiledFunction {
    // Owns JITModule/code memory, entrypoint, scratch, and native layout metadata.
}

impl CompiledFunction {
    pub fn run_native(
        &self,
        lhs: &Bits24,
        rhs: &Bits257,
        output: &mut Output,
    ) -> Result<(), RunError>;
}
```

A dynamic convenience API can exist separately:

```rust
pub fn run_values(&self, args: &[PirValue]) -> Result<PirValue, RunError>;
```

That API may allocate or copy because its values are dynamically typed. It
should not define the JIT's native ABI.

The generated function should expose a small internal C-compatible ABI:

```rust
type NativeEntrypoint = unsafe extern "C" fn(
    inputs: *const *const u8, // Points directly into borrowed native values.
    output: *mut u8,          // Points directly at caller output storage.
    scratch: *mut u8,
    runtime: *mut RuntimeContext,
) -> i32;
```

For pure functions, there is one output and the initial `RuntimeContext` may
contain only error/status facilities or be unused. The input pointer array is
only an array of borrows; constructing it does not copy parameter values.
Keeping a context pointer in the ABI leaves a straightforward path for checked
runtime failures and future events without reproducing the full XLS C++
callback ABI.

The caller owns input and output storage. The wrapper owns scratch and runtime
storage. This gives each invocation independent scratch memory and removes
scratch-buffer aliasing as an obstacle to concurrent execution. Whether one
compiled handle can be shared across threads must still follow the `JITModule`
ownership and thread-safety guarantees exposed by Cranelift.

PIR compilation happens at runtime, while Rust struct definitions normally
exist at compile time. A typed binding therefore requires one of:

- A generated Rust bridge module, analogous to the existing AOT typed wrappers,
  that supplies `#[repr(C)]` types and native layout descriptors.
- An unsafe binding API or derive mechanism by which a caller asserts that its
  `#[repr(C)]` types match a PIR signature and supplies `size_of`, `align_of`,
  and field-offset metadata.

The JIT must validate a supplied native binding layout against the PIR
signature before it generates loads and stores.

### Rust layout contract

The native API can use annotated Rust data directly because Rust specifies the
relevant layouts:

| Rust type form | Suitability for direct JIT access | Contract |
| --- | --- | --- |
| `u8`, `u16`, `u32`, `u64`, `u128` | Suitable for scalar bit storage | Fixed byte sizes; alignment is target-specific. |
| `[T; N]` | Suitable when `T` is suitable | Contiguous elements at `n * size_of::<T>()`, with `T`'s alignment. |
| `#[repr(C)] struct S { ... }` | Suitable for tuples/records and wide wrappers | Fields are laid out in declaration order with C-style padding and target alignment. |
| `#[repr(transparent)] struct BitsN(u32);` | Suitable for validated scalar newtypes | Same layout and ABI as its single non-zero-sized field. |
| Default `struct` (`repr(Rust)`) | Not suitable | Field ordering and other layout details are not guaranteed. |
| Rust tuple `(A, B)` | Not suitable as an exposed ABI value | It uses the unspecified Rust representation. |
| `bool` as writable JIT output | Avoid | Only valid Rust boolean representations may be written. |
| Rust `enum` as arbitrary bits output | Avoid initially | Only declared discriminants are valid Rust values. |

For nested values, annotations are recursive in effect: `#[repr(C)]` on the
outer struct fixes placement of each field, but it does not change a nested
field's own layout. Consequently any nested user-defined bridge struct also
needs a specified representation.

The JIT should access fields by layout descriptors generated from the Rust type
definitions, for example:

```rust
NativeValueLayout {
    size: std::mem::size_of::<Output>(),
    alignment: std::mem::align_of::<Output>(),
    fields: vec![
        std::mem::offset_of!(Output, valid),
        std::mem::offset_of!(Output, value),
    ],
}
```

A derive such as `#[derive(NativePirType)]` over a required `#[repr(C)]` or
`#[repr(transparent)]` type could generate this descriptor and the mapping to
the PIR type. Its safe constructors must enforce width invariants for padded
bits values; generated output stores must mask high padding before returning.
The safe wrapper should pass an initialized valid output value to generated
code, or retain output as `MaybeUninit<T>` until successful execution has
written every field with a valid representation.

## Cranelift interface

The implementation uses the following Cranelift layers:

| Crate/API | Responsibility |
| --- | --- |
| `cranelift_frontend::FunctionBuilder` | Emit CLIF instructions and control flow for a PIR function. |
| `cranelift_codegen::ir::{Type, Signature, AbiParam}` | Define scalar machine values and the generated entrypoint ABI. |
| `cranelift_module::Module` | Declare and define named generated functions and future runtime imports. |
| `cranelift_jit::{JITBuilder, JITModule}` | Finalize host code and obtain an executable entrypoint pointer. |
| `cranelift_native` | Select the host target ISA for JIT compilation. |
| `cranelift_object::ObjectModule` | Later AOT backend using shared lowering, not needed for the first JIT milestone. |

Each PIR `Fn` can initially lower to one CLIF function. The partitioning and
resumption machinery in XLS's `xls/jit/function_base_jit.cc` exists primarily
for proc/block early exits and for its node-function compilation strategy; it
is not required for a pure-function MVP.

## Three representations, not one

The design should keep these concepts separate:

1. **Native interface layout**: the target-native Rust storage accessed
   directly through input/result pointers.
1. **Computation representation**: CLIF scalar `Value`s or memory references
   used while lowering operations.
1. **Scratch layout**: aligned slots allocated for materialized intermediate
   values.

LLVM currently makes these concepts look related because the existing JIT
uses LLVM native type layout for buffers. A new backend does not need to
inherit LLVM's layout, but a zero-copy API intentionally needs a
target-dependent native Rust layout. An optional portable/dynamic API can pack
values separately if required.

## Type mapping

### Bits

Cranelift does not support arbitrary-width scalar integer types. Its
documented integer scalar types are `I8`, `I16`, `I32`, `I64`, and `I128`.
The current XLS LLVM lowering rounds bits values up to an LLVM integer width
which is at least eight and then a power of two; LLVM can continue past
`i128` to `i256`, `i512`, and larger values.

Recommended native mapping:

| PIR type | CLIF computation representation | Native Rust bridge representation |
| --- | --- | --- |
| `bits[0]` | Special zero value; no meaningful scalar bits | Zero-sized bridge value or omitted leaf |
| `bits[1]` through `bits[8]` | `I8`, masked to PIR width | `u8`, including `bits[1]` |
| `bits[9]` through `bits[16]` | `I16`, masked to PIR width | `u16` |
| `bits[17]` through `bits[32]` | `I32`, masked to PIR width | `u32` |
| `bits[33]` through `bits[64]` | `I64`, masked to PIR width | `u64` |
| `bits[65]` through `bits[128]` | Candidate `I128`, after an operation/target validation spike | Candidate `u128` |
| `bits[129+]` | Memory-backed limbs, deferred from the first scalar milestone | `#[repr(C)]` fixed-size limb wrapper |

All operations returning `bits[N]` must clear bits above `N` unless the
lowering can prove they are already clear. Signed operations must sign-extend
from the PIR width, not from the padded CLIF width.

`bits[1]` should not use `bool` in the machine-code ABI: writing an invalid
Rust `bool` representation would be undefined behavior. A generated ergonomic
layer can validate or convert between a user-facing boolean and an ABI-facing
`u8`.

The first executable milestone should support `bits[1..=64]`. Before making
`I128` part of the supported contract, add characterization tests for every
intended `I128` arithmetic, shift, compare, and conversion operation on each
supported JIT target. A later wide-bits milestone should choose between:

- `I128` through width 128 followed by `I64` limbs for larger values.
- `I64` limbs for all widths above 64, reducing special backend cases at the
  cost of giving up possible native `I128` handling.

### Tuples and arrays

At the Rust boundary, tuples/struct-like PIR values should map to generated
`#[repr(C)]` named Rust structs so the JIT can access caller storage without a
packing copy. Plain Rust tuples are not suitable for this ABI because their
layout is not specified for external access.

Inside CLIF, these should not initially be modelled as first-class struct
values. Cranelift exposes C-ABI struct-argument support, but its ordinary CLIF
value types are scalar/vector/reference types rather than LLVM-style
first-class aggregate types.

Recommended baseline representation:

- Define `NativeValueLayout` recursively for `Type::Tuple` and `Type::Array`,
  matching the provided `#[repr(C)]` bridge representation.
- Represent an aggregate during lowering as a pointer to caller or scratch
  storage plus its layout.
- Implement `tuple`, `tuple_index`, `array`, and array indexing/update as
  stores, loads, and copies using computed offsets.
- Consider scalarizing small aggregates only after the baseline path is
  correct and measured.

The native layout is target-specific:

- Scalar bits bridge fields use native Rust integer storage and alignment.
- Struct-like values use `#[repr(C)]` field order and padding.
- Arrays use Rust fixed-size arrays of the native element bridge type.
- Wide bits use a defined fixed-size limb wrapper.

Layout descriptors should preferably be generated using Rust's own
`size_of`, `align_of`, and `offset_of!` for bridge types rather than having
the JIT independently guess their offsets. Scratch slots can use the same
native representation, making copies needed for PIR intermediate values
ordinary native memory copies rather than packing/unpacking conversions.

### Endianness

For the in-process JIT, native bridge storage should follow the execution
machine's native endian convention. A `u32` field is loaded as the target's
native `u32`; it is not converted to a canonical little-endian byte stream on
entry and exit.

For wide values, limb *ordering* still needs a semantic convention independent
of byte endian: for example, `limbs[0]` is the least-significant limb, while
each `u64` limb has native machine byte order in memory.

This native ABI is not a portable serialization or cross-target artifact
format. If a future AOT artifact needs portable metadata, or a dynamic API
needs canonical bytes, it should define a separate packed representation and
perform conversion explicitly.

### Tokens

Tokens contain no data, but they sequence side effects. Since the initial JIT
does not support `assert`, `trace`, or other effecting function nodes, reject
functions containing token-using operations in the first milestone. Token
layout can be zero bytes when event support is added.

## Internal lowering model

A lowering context needs both semantic and storage information:

```rust
enum ComputedValue {
    Scalar {
        value: cranelift_codegen::ir::Value,
        pir_width: usize,
    },
    Materialized {
        scratch_slot: ScratchSlot,
        layout: NativeValueLayout,
    },
}

struct ScratchSlot {
    offset: usize,
    size: usize,
    alignment: usize,
}
```

An intentionally conservative first implementation can materialize each
non-parameter, non-result node into its own scratch slot. Scalar operations
then load operands, compute in CLIF, mask to their PIR width, and store their
result. This is not the final performance strategy, but it makes aggregates,
wide values, and debugging use the same storage model.

Once the semantic path is established, retain `bits[1..=64]` intermediates as
`ComputedValue::Scalar` when they do not need an address. Materialize only
values that are:

- Aggregates or wide limb-based values.
- Used across a generated call boundary.
- Needed as addressable input or output storage.
- Required by a runtime helper.

## Scratch slab

The generated function should not use unbounded CLIF stack slots for PIR
temporary values. Large arrays, tuples, or future wide values can make those
allocations substantial. The existing XLS JIT uses a mixed allocator:
`AllocationKind::kTempBlock` assigns caller-owned scratch slots for values
that must persist across generated partitions, while `kAlloca` is available
for partition-local values and `kNone` for values needing no new buffer. The
baseline below deliberately starts with a simpler, more conservative slab
policy for materialized PIR values.

At compile time:

```text
PIR nodes requiring materialization
  -> NativeValueLayout for each node
  -> monotonically assign aligned ScratchSlot offsets
  -> record total scratch_size and maximum scratch_alignment
```

At run time:

```text
CompiledFunction::run
  -> allocate one scratch block with recorded size/alignment
  -> pass base pointer into NativeEntrypoint
  -> generated code calculates scratch + constant_offset for each slot
```

Baseline policy:

- Allocate a unique scratch slot per materialized intermediate.
- Allocate no scratch slot for native parameter storage or the final result
  buffer unless a copy is required.
- Zero-initialize scratch in debug/testing mode if useful for diagnosing
  missing writes; do not require zeroing for correct supported code.
- Do not reuse slots by liveness in the first implementation.

Later optimization:

- Calculate last use of each materialized node and reuse non-overlapping
  scratch slots.
- Keep the layout planner deterministic so diagnostics and tests remain stable.

Unlike XLS's common proc/block JIT, this function-only implementation does not
need scratch values to persist across continuation points. The caller-owned
slab remains useful to avoid large stack allocation and to support aggregate
materialization.

## First supported nodes

The compiler should expose an explicit support matrix and return an
`UnsupportedNode`/`UnsupportedType` error rather than silently interpreting
unsupported IR.

Suggested implementation stages:

| Stage | Supported PIR surface |
| --- | --- |
| Scalar bring-up | Parameters, literals, return value, `identity`, `not`, `neg`, `add`, `sub`, `and`, `or`, `xor`, `nand`, `nor`, `eq`, `ne`; `bits[1..=64]` only. |
| Scalar bit manipulation | Comparisons, zero/sign extension, static slice, concat, logical/arithmetic shifts, reductions, selects. |
| Aggregate storage | Tuple creation/indexing, arrays, and array index/update/slice over native bridge and scratch storage. |
| Wider values | Validated `I128` path and/or limb-based values above 64 bits. |
| PIR-specific computation | Suitable extension ops such as `ext_carry_out`, `ext_clz`, `ext_mask_low`, and `ext_nary_add`. |
| Calls/control constructs | `invoke` and `counted_for`, compiling reachable function dependencies or applying a defined inlining strategy. |
| Effects | Tokens, `assert`, `trace`, and a Rust runtime/event context. |

Instantiation and register nodes are block behavior and remain out of scope
for a function-only JIT.

## Compilation and execution flow

```text
compile(Fn)
  1. Check PIR layout/type invariants and reject non-function behavior.
  2. Validate supplied native bridge layouts against parameter/result types.
  3. Walk node types and build NativeValueLayout descriptions.
  4. Determine supported computation representations.
  5. Allocate deterministic scratch slots for materialized nodes.
  6. Build a CLIF Signature for NativeEntrypoint.
  7. Use FunctionBuilder to:
       - load native parameter fields through caller pointers,
       - evaluate nodes in dependency order,
       - store materialized results into scratch slots,
       - store the return value directly into caller output storage,
       - return a success status.
  8. Define and finalize the function in JITModule.
  9. Hold JITModule alive with the resolved entrypoint and FunctionLayout.

run_native(inputs, output)
  1. Borrow annotated native input and output storage.
  2. Construct pointer arguments referring directly to that storage.
  3. Allocate aligned scratch storage.
  4. Invoke NativeEntrypoint through an unsafe `extern "C"` function pointer.
  5. Validate status and return with `output` already populated.

run_values(args), if provided
  1. Convert dynamically represented values to native bridge storage.
  2. Call run_native.
  3. Convert the native result back to its dynamic representation.
```

If PIR permits nodes to appear before operands in the stored node list, the
backend should lower according to a deterministic topological traversal rather
than assuming node index is execution order. The existing interpreter already
has this distinction.

## Testing strategy

Implementation tests should be layered:

| Test layer | Purpose |
| --- | --- |
| Layout unit tests | Compare native layout descriptors with `size_of`, `align_of`, and `offset_of!`; specify scratch size/alignment and high-bit clearing. |
| CLIF compile smoke tests | Verify that each supported PIR operation compiles and finalizes on the host ISA. |
| Differential value tests | Compare JIT results with `xlsynth_pir::ir_eval` for supported functions and generated inputs. |
| Transitional libxls parity tests | Where useful, compare to current libxls JIT for overlapping upstream operations while it remains available. |
| Wide/aggregate characterization | Lock down edge cases at widths 0, 1, 7, 8, 9, 63, 64, 65, 127, 128, and above 128 once limbs are implemented. |
| Concurrency tests | Run one compiled function concurrently with separate scratch blocks. |

Initial tests should exercise both arithmetic and native layout; a value
comparison alone will not catch a mismatched struct offset or alignment that
would make zero-copy access unsound.

## Milestones

### M0: dependency and representation spike

- Create the backend library crate and pin a Cranelift release family.
- Compile a hand-built CLIF add function through `JITModule`.
- Confirm host JIT support and function-pointer invocation.
- Characterize scalar CLIF support for `I8`, `I16`, `I32`, `I64`, and `I128`.
- Define the annotated native bridge layout contract for the prototype.

Exit criterion: a Rust test calls generated code without calling libxls JIT.

### M1: scalar PIR functions

- Accept `ir::Fn`.
- Implement `NativeValueLayout` for bits and the scratch planner.
- Execute directly against native scalar input and output storage.
- Lower the scalar bring-up node subset for `bits[1..=64]`.
- Provide a wrapper capable of running supported functions.
- Differential-test results against `ir_eval`.

Exit criterion: a non-trivial set of pure scalar PIR functions executes through
Cranelift with deterministic unsupported-operation errors.

### M2: scalar completeness and layout contract

- Add remaining common scalar operations, including shifts/slices/selects.
- Specify and test native storage, high-bit masks, signed semantics, and
  `bits[0]`.
- Decide and implement the `I128` or limb cutoff.

Exit criterion: supported scalar types have a documented native bridge layout.

### M3: aggregates and slab-backed values

- Implement tuple and array layouts as memory-backed aggregates.
- Lower aggregate constructors, projections, indexing, and updates.
- Add large-aggregate tests demonstrating bounded generated stack use and
  caller-provided scratch usage.

Exit criterion: function signatures and intermediates involving tuples/arrays
execute without representing aggregates as CLIF struct values.

### M4: remove transitional libxls value use

- Introduce or adopt Rust-native PIR literal storage and any dynamic value API
  required alongside native entrypoints.
- Convert parser/evaluator/backend edges needed by the JIT away from
  `xlsynth::IrValue`.

Exit criterion: the PIR JIT's compile-and-run test path does not require
loading libxls.

### Later work

- Calls (`invoke`, `counted_for`) and reachable-function compilation.
- Runtime events and token sequencing.
- AOT emission using `ObjectModule`.
- Public API integration or migration from current `IrFunctionJit`.
- Performance comparison, scalarization, and scratch-slot reuse.

## Open questions

| Question | Initial direction | Resolution work |
| --- | --- | --- |
| How should bits wider than 128 be computed? | Native-endian `u64` limbs in least-significant-limb-first order. | Prototype common wide operations and compare complexity/performance with helper calls. |
| Should `I128` be used for widths 65 through 128? | Treat as optional until tested across operations and targets. | CLIF characterization tests on x86-64 and AArch64 where supported. |
| Should tuples map to structs? | Yes at the Rust boundary through `#[repr(C)]` bridge structs; no as first-class CLIF values initially. | Revisit internal scalarization only as an optimization. |
| How should native layouts be declared? | Generated bridge types or an unsafe/derived `NativePirType` binding using `size_of`, `align_of`, and `offset_of!`. | Prototype API ergonomics and compile-time checks. |
| Should the JIT retain a portable packed API? | Only as an optional dynamic adapter; native target-endian access is primary. | Decide based on migration and AOT artifact needs. |
| Should every temporary receive a scratch slot? | Yes for initial materialized baseline; later retain scalar intermediates in SSA. | Benchmark after correctness coverage exists. |
| How are runtime errors reported? | Reserve `RuntimeContext` and status return in the ABI. | Define behavior for assumed-in-bounds violations and future events. |
| Where should Rust-native PIR values live? | Prefer a lower-level PIR/value crate or refactor that does not depend on `xlsynth`. | Avoid introducing a crate cycle during API migration. |
| How should compiled callees use scratch? | Defer calls initially. | Allocate caller/callee scratch regions or use inlining when `invoke` is implemented. |

## Source references

Current local implementation points:

- `xlsynth-pir/src/ir.rs`: PIR `Type`, `Fn`, and `NodePayload` definitions.
- `xlsynth-pir/src/ir_eval.rs`: existing evaluator and differential-test
  reference behavior.
- `xlsynth/src/ir_value.rs`: current libxls-backed dynamic value representation
  and bytes adapter that a native entrypoint can eventually bypass.
- XLS `xls/jit/function_base_jit.{h,cc}`: entrypoint ABI, `BufferAllocator`,
  partitions, and scratch-buffer metadata.
- XLS `xls/jit/llvm_type_converter.{h,cc}`: current LLVM type mapping and
  native layout calculations.

Cranelift API references:

- [`cranelift_frontend::FunctionBuilder`](https://docs.rs/cranelift-frontend/latest/cranelift_frontend/struct.FunctionBuilder.html)
- [`cranelift_codegen::ir::types`](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/types/index.html)
- [`cranelift_jit::JITModule`](https://docs.rs/cranelift-jit/latest/cranelift_jit/struct.JITModule.html)
- [`cranelift_object::ObjectModule`](https://docs.rs/cranelift-object/latest/cranelift_object/struct.ObjectModule.html)
- [Rust Reference: type layout and `repr(C)`](https://doc.rust-lang.org/reference/type-layout.html)
