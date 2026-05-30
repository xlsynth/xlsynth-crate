# JIT and AOT execution through libxls

This note describes the JIT and AOT components currently used by the `xlsynth`
crate, with emphasis on the boundary that would need to change for a
Rust-native implementation. It is an inventory of the existing function
execution flow, not yet a replacement design.

## Scope and source baseline

The Rust APIs currently traced here compile and execute XLS **functions**:

- `IrFunctionJit` provides runtime JIT evaluation.
- `AotCompiled`, `AotRunner`, and `aot_builder` provide function AOT
  compilation and execution.
- Generated `standalone-aot` wrappers use `xlsynth-aot-runtime`.

Upstream XLS also supports JIT/AOT machinery for procs and blocks, including
channels, continuation points, and register state. Those capabilities are
visible in the shared XLS implementation, but no corresponding `xlsynth`
public API is part of the flows above.

The XLS source checkout inspected for this inventory contains the direct AOT C
API (`xls_aot_compile_function`, `xls_aot_exec_context_*`, and
`xls_aot_entrypoint_trampoline`) and the common LLVM implementation. It does
not contain the `xls_standalone_aot_*` symbols or
`standalone_runtime_feature_requirements` proto field consumed by the current
`xlsynth-crate` standalone path. The crate obtains those from its selected XLS
release artifacts (`xlsynth-sys` currently selects `v0.50.1`). Thus, the
standalone section below identifies the Rust-visible and artifact interfaces;
the implementation inside the release static archive was not available in the
source checkout examined here.

## Summary

There are three consumer-visible execution paths:

| Rust surface | Compilation happens | Execution dependency | Native XLS content involved |
| --- | --- | --- | --- |
| `IrFunctionJit` | At runtime inside `libxls` | `libxls` DSO | LLVM ORC JIT plus XLS runtime/callbacks |
| `AotCompiled` + `AotRunner` | Before execution inside `libxls` | `libxls` DSO plus linked AOT object | LLVM-emitted object and direct AOT trampoline/context |
| Generated `standalone-aot` runner | In a Cargo build script inside `libxls` | `libxls_aot_runtime.a` plus linked AOT object, not the `libxls` DSO | LLVM-emitted object and XLS standalone native runtime |

All three paths still depend on native XLS code. The standalone path removes
the dynamic `libxls` dependency from the final artifact consumer, but its
compiled object and static runtime are still produced or supplied by XLS.

The JIT and AOT compile paths share most of their implementation:

```text
XLS Function
  -> JittedFunctionBase / JitBuilderContext / IrBuilderVisitor
  -> LLVM IR with the JitFunctionType buffer-and-callback ABI
  -> LlvmCompiler
       -> OrcJit: executable machine code loaded in memory
       -> AotCompiler: relocatable object bytes plus entrypoint metadata
```

## Rust-side components

### FFI and artifact plumbing

`xlsynth-sys/src/lib.rs` declares the C symbols used by both flows:

- JIT: `xls_make_function_jit`, `xls_function_jit_run`, and
  `xls_function_jit_free`.
- AOT compilation: `xls_aot_compile_function` and buffer-free functions.
- Direct AOT execution: `xls_aot_exec_context_*` and
  `xls_aot_entrypoint_trampoline`.

`xlsynth-sys/build.rs` provides and links the `libxls` DSO. It either consumes
explicit artifact paths, builds against a development XLS workspace, or
downloads a selected release. This dependency exists for ordinary JIT use and
for build scripts performing AOT compilation.

### Runtime JIT API

`xlsynth/src/ir_package.rs` owns the public `IrFunctionJit`. Creating one calls
`xls_make_function_jit` while retaining the parent `IrPackage`, because the
native function and its XLS types remain package-owned. Running one passes
`IrValue` handles through `xls_function_jit_run`.

`xlsynth/src/lib_support.rs` converts the returned value and native
`InterpreterEvents` data into:

```rust
pub struct RunResult {
    pub value: IrValue,
    pub trace_messages: Vec<TraceMessage>,
    pub assert_messages: Vec<String>,
}
```

Consequently, a compatible replacement of this API includes more than value
evaluation: it must preserve trace and assertion behavior or change the public
contract.

One observable ABI asymmetry exists today: in the inspected XLS C wrapper,
`xls_function_jit_run` reports each JIT trace message with verbosity `0`,
whereas `xls_aot_exec_context_get_trace_message` preserves the XLS statement
verbosity. A replacement should make an intentional compatibility choice here.

### AOT compilation and direct execution

`xlsynth/src/aot_lib.rs` exposes `AotCompiled::compile_ir`. It parses IR through
libxls, chooses a function by name, and calls `xls_aot_compile_function`.
Compilation returns:

- Object-file bytes containing the compiled entrypoint.
- A serialized XLS entrypoint proto.
- Rust-parsed `AotEntrypointMetadata`.

`xlsynth/src/aot_entrypoint_metadata.rs` decodes the relevant parts of the
proto: symbol name, buffer sizes and alignments, temp-buffer geometry, function
signature, leaf layouts, and standalone feature requirements.

`xlsynth/src/aot_runner.rs` is the direct runtime API. It allocates aligned
input, output, and temporary buffers from the metadata; creates a native
`xls_aot_exec_context`; and invokes the linked object through
`xls_aot_entrypoint_trampoline`. This runner still requires `libxls` at
execution time for the trampoline, runtime helper object, and event capture.

### Generated standalone AOT execution

With the `standalone-aot` feature, `xlsynth/src/aot_builder.rs` is intended for
use in a build script. For each selected function it:

1. Adds a stable artifact-scoped forwarding function in IR.
1. Calls `AotCompiled::compile_ir`, so the compile step still uses `libxls`.
1. Writes the object bytes and entrypoint proto into `OUT_DIR`.
1. Generates a typed Rust wrapper from XLS type/layout metadata.
1. Uses `cc::Build` to link the emitted object into the consumer.

The generated wrapper uses `xlsynth-aot-runtime`, not `AotRunner`.
`xlsynth-aot-runtime/src/lib.rs` implements Rust-side typed values, aligned
buffers, leaf packing/unpacking, and `StandaloneRunner`, but calls native
`xls_standalone_aot_*` symbols to initialize callback state and invoke the
compiled entrypoint.

`xlsynth-aot-runtime/build.rs` provides those symbols by linking the
same-version `libxls_aot_runtime.a` archive and its link configuration, unless
an enclosing build graph supplies an equivalent native runtime dependency.
Standalone generated code currently permits assertions and rejects entrypoints
requiring trace or cover callbacks.

## XLS-side components

The native implementation behind the direct C API is organized as follows.
Paths in this section are relative to an XLS source tree.

| Component | Role in the flow |
| --- | --- |
| `xls/public/c_api.cc` | C ABI surface called by `xlsynth-sys`; wraps `FunctionJit`, emits AOT object/proto output, and owns direct AOT execution context/trampoline functions. |
| `xls/jit/function_jit.{h,cc}` | Function-level facade. Creates JIT instances, creates AOT object code, validates argument types, marshals `Value` inputs/outputs, and owns reusable runtime buffers. |
| `xls/jit/function_base_jit.{h,cc}` | Common compilation/execution core for functions, procs, and blocks. Defines the compiled entrypoint ABI, builds dependent functions and partitions, allocates scratch-layout metadata, and invokes compiled pointers. |
| `xls/jit/ir_builder_visitor.{h,cc}` | Lowers XLS nodes to LLVM IR, including arithmetic/aggregate operations, invokes/calls, and callback-bearing operations such as assertions and traces. |
| `xls/jit/llvm_type_converter.{h,cc}` and `xls/jit/type_layout.*` | Maps XLS bits/arrays/tuples/tokens to LLVM types and computes byte size, alignment, padding, and flattened leaf layout metadata. |
| `xls/jit/llvm_compiler.{h,cc}` | Shared LLVM target, data-layout, verification, and optimization (`O3` by default) abstraction. |
| `xls/jit/orc_jit.{h,cc}` | JIT backend: emits executable host code into LLVM ORC and resolves entrypoint addresses. |
| `xls/jit/aot_compiler.{h,cc}` | AOT backend: selects target/cpu policy, optimizes the LLVM module, and emits relocatable object bytes. |
| `xls/jit/aot_entrypoint.{h,cc}` and `aot_entrypoint.proto` | Serializes the ABI metadata needed to find and call an AOT object entrypoint. |
| `xls/jit/jit_runtime.{h,cc}` | Native value packing/unpacking and LLVM-data-layout support used while running compiled code. |
| `xls/jit/jit_callbacks.*` and `xls/ir/events.*` | Runtime callback table and event collection for assertions, traces, formatting, allocations, and the proc/block features in the common ABI. |

## LLVM-facing component breakdown

### Layout and function metadata

Several related objects describe how compiled code is called. They are not
interchangeable:

| Object | Lifetime and role | Important contents |
| --- | --- | --- |
| `llvm::DataLayout` | Selected by the LLVM target machine for both JIT and AOT. AOT serializes its string form once per object package. | Native type sizes and target ABI alignment rules used while lowering, allocating, packing, and unpacking. |
| `LlvmTypeConverter` | Compile-time helper instantiated against a `DataLayout`. A separate instance is used when emitting AOT metadata. | XLS type to LLVM type conversion, packed LLVM forms, padding masks, native byte size/alignment calculations, and `TypeLayout` creation. |
| `TypeBufferMetadata` | In-memory descriptor stored in `JittedFunctionBase`, one entry per top-level input/output buffer. | Native `size`, `preferred_alignment`, `abi_alignment`, and optional `packed_size`. |
| `TypeLayout` / `ElementLayout` | Structural layout for one XLS input or output type; serialized into AOT metadata. | Total byte size and a flattened leaf list of byte `offset`, meaningful `data_size`, and zero-padded `padded_size`. |
| `JittedFunctionBase` | Live in-process compiled-entrypoint descriptor shared by JIT and AOT construction. | Native and optional packed symbol/function pointers, input/output buffer metadata, temp-buffer size/alignment, and proc/block continuation or queue information. |
| `AotEntrypointProto` | Serialized equivalent of the entrypoint calling contract, needed after the compiler process is gone. | Entrypoint symbol names, input/output buffer geometry, packed symbols/sizes, scratch geometry, input/output layouts and names, and function/proc/block interface metadata. |
| `AotPackageEntrypointsProto` | Envelope for object-file entrypoints. | One or more `AotEntrypointProto` records and the LLVM `data_layout` string used to create runtime packing support. |

The normal entrypoint uses LLVM's **native** aggregate layout. XLS also builds
an optional **packed** wrapper for functions, in which each XLS value is
flattened to a packed LLVM integer form. The metadata records both forms when
the packed wrapper exists. The `xlsynth` AOT runner and generated wrapper
consume the ordinary unpacked function symbol and native-layout metadata.

The distinction between the native byte size and leaf content is important.
For example, `bits[42]` contains five meaningful bytes but is represented in
an eight-byte LLVM slot, and tuple/array elements can acquire target-dependent
offsets and padding. A replacement that exposes the same AOT artifacts either
has to generate exactly these layouts or preserve the metadata as the
authority for packing.

### XLS IR to LLVM IR lowering

The core compilation path is shared until the completed LLVM module is handed
to a backend:

```text
Function / Proc / Block
  -> LlvmCompiler creates target machine, DataLayout, and LLVM module
  -> JitBuilderContext owns module, symbol map, queue indices, and LlvmTypeConverter
  -> JittedFunctionBase::BuildInternal
       -> GetDependentFunctions(top)
       -> BuildFunctionInternal for each dependency and top
            -> buffer allocator assigns temporary native storage
            -> CreateNodeFunction / IrBuilderVisitor lowers XLS nodes
            -> partitions encode early exits for blocking operations
       -> BuildPackedWrapper for ordinary Function targets
       -> LlvmCompiler::CompileModule(module)
  -> JittedFunctionBase records the callable ABI and buffer metadata
```

`IrBuilderVisitor` is the operation semantics layer. It maps XLS nodes to LLVM
instructions for bits and aggregate operations, calls lowered dependent XLS
functions, emits tuple/array memory operations, clears padded bits where
needed, and emits runtime callback invocations for nodes such as `trace` and
`assert`.

`JitBuilderContext` is the per-compilation assembly object. It is where the
LLVM module, type converter, mangled symbol mapping, and proc queue-index
allocation meet; it does not execute code itself.

`JittedFunctionBase` is therefore somewhat misnamed: it is both the result
descriptor and the coordinator that causes the LLVM IR entrypoint to be
built. It exists in both paths:

- In JIT mode, it is populated with real function pointers resolved from ORC.
- During AOT object production, it contains symbol names and layouts while
  function pointer fields are intentionally unusable; those fields become
  meaningful when a separately linked AOT symbol is supplied later.

### Services invoked by compiled code

For ordinary arithmetic and aggregate operations, generated code can operate
directly on its input, output, and temporary buffers. Operations with runtime
effects require native support passed through the compiled ABI.

| Runtime object or service | How compiled code reaches it | Use for function execution |
| --- | --- | --- |
| `InterpreterEvents*` | Direct entrypoint argument, also passed to callbacks. | Collects assertion failures and trace messages. |
| `InstanceContext*` and `InstanceContextVTable` | Direct entrypoint argument; LLVM code performs indirect calls through fixed vtable offsets. | Creates/formats/records trace messages, records assertions, and allocates/deallocates large scratch buffers. |
| `JitRuntime*` | Direct entrypoint argument and argument to formatting callbacks. | Holds `DataLayout`/type conversion support; outside the entrypoint it packs XLS `Value` arguments and unpacks results, and callbacks can use it to format typed values. |
| Queue, active-next, register-write, and observer callbacks | Additional `InstanceContextVTable` entries emitted by shared lowering when needed. | Required for upstream proc/block/observer support; not central to the currently exposed `xlsynth` function JIT/AOT APIs. |

This is deliberately a manual callback ABI: generated code does not need to
directly link arbitrary C++ mangled helper symbols for these actions. It
dereferences known function-pointer slots at the start of `InstanceContext`.
For Rust replacement planning, trace/assert support is therefore a runtime
ABI problem in addition to an instruction-lowering problem.

### Where JIT and AOT diverge

| Stage | Runtime JIT | AOT object production |
| --- | --- | --- |
| Frontend source | `FunctionJit::Create` | `FunctionJit::CreateObjectCode` |
| LLVM compiler implementation | `OrcJit`, a `LlvmCompiler` implementation | `AotCompiler`, a `LlvmCompiler` implementation |
| Target and layout | Host target/DataLayout selected for in-process execution | Target/DataLayout selected for the object artifact; supports native, `x86_64`, and `aarch64` policies in the source inspected |
| Shared lowering | `JittedFunctionBase::Build` plus `IrBuilderVisitor` | The same `JittedFunctionBase::Build` plus `IrBuilderVisitor` |
| Terminal backend action | Optimize, compile, load into an ORC execution session, and resolve function pointers | Optimize and emit position-independent relocatable object bytes |
| How calling metadata survives | Held directly by the live `JittedFunctionBase` and `FunctionJit` objects | Converted from `JittedFunctionBase` to `AotEntrypointProto` plus package `data_layout` |
| Execution setup | `FunctionJit` already owns `JitRuntime`, callback context, and reusable buffers | A later runner parses the proto, creates a `JitRuntime` from `data_layout`, allocates buffers, and calls the externally linked symbol |

The AOT compilation path is therefore not a separate lowering implementation.
It is the same LLVM entrypoint construction with a different terminal compiler
backend and an additional metadata serialization step.

## Compiled ABI

XLS compiles an entrypoint with `JitFunctionType`, conceptually:

```c++
int64_t entrypoint(
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    void* temp_buffer,
    InterpreterEvents* events,
    InstanceContext* instance_context,
    JitRuntime* jit_runtime,
    int64_t continuation_point);
```

For an ordinary function, `outputs` has the return value and the continuation
point is normally zero. The broader ABI also supports proc suspension/resume
and block state behavior.

This ABI is why AOT is not just an object-file generation issue. The linked
entrypoint expects:

- Input and output buffers using the LLVM-selected native layout.
- Correct sizes, preferred alignments, padding, and scratch-buffer layout.
- Callback and event state for assertion or trace nodes.
- A runtime helper capable of formatting or unpacking values when generated
  code requests it.

Bits values are not represented simply by their XLS bit count. In the XLS LLVM
mapping, sub-byte values occupy at least eight bits and other odd widths are
padded to LLVM-friendly power-of-two widths. Aggregates introduce
target-data-layout offsets and padding. The entrypoint metadata and Rust
packing logic carry these requirements into AOT callers.

## Runtime JIT flow

The `IrFunctionJit` sequence is:

```text
Rust IrFunctionJit::new
  -> xls_make_function_jit
  -> FunctionJit::Create
  -> OrcJit::Create + host DataLayout
  -> JittedFunctionBase::Build
     -> discover transitively called functions
     -> lower XLS nodes/partitions to LLVM IR
     -> compute native buffer/temp metadata
  -> OrcJit compiles and loads symbols

Rust IrFunctionJit::run(args)
  -> xls_function_jit_run
  -> FunctionJit::Run
     -> type-check XLS Values
     -> JitRuntime::PackArgs into aligned native buffers
     -> execute JitFunctionType pointer with callbacks/events
     -> JitRuntime::UnpackBuffer to XLS Value
  -> copy value, trace messages, and assert messages back through C ABI
```

The JIT object's argument, result, and scratch buffers are reused between
calls; XLS documents `FunctionJit` as not thread-safe for concurrent runs.

## AOT compilation flow

The AOT compilation path uses the same lowering but swaps the terminal LLVM
backend:

```text
Rust AotCompiled::compile_ir
  -> xls_aot_compile_function
  -> FunctionJit::CreateObjectCode
  -> AotCompiler::Create + target DataLayout
  -> JittedFunctionBase::Build
     -> same XLS-to-LLVM lowering and JitFunctionType ABI
  -> AotCompiler emits object bytes
  -> GenerateAotEntrypointProto records symbol/layout/type metadata
  -> Rust receives object bytes and serialized proto
```

In the source examined, AOT supports a native target as well as explicit
`x86_64` and `aarch64` selections. It emits position-independent object code
and uses a baseline CPU policy rather than all host CPU features.

## Direct AOT runtime flow

When callers use `AotRunner`, execution remains inside the DSO-backed
environment:

```text
Rust AotRunner::new
  -> parse metadata and allocate aligned buffers
  -> xls_aot_exec_context_create(proto)
     -> parse LLVM DataLayout
     -> allocate InterpreterEvents, InstanceContext, JitRuntime

Rust AotRunner::run[_with_events]
  -> pack wrapper/user bytes into buffers
  -> xls_aot_entrypoint_trampoline(function pointer, buffers, context)
  -> linked AOT object executes using C++ callback/runtime state
  -> read trace/assert events from context through C ABI
```

This route avoids JIT compilation at invocation time, but it does not avoid
the `libxls` runtime dependency.

## Standalone AOT runtime flow

Generated standalone wrappers shift the final runtime dependency:

```text
build.rs:
  xlsynth + libxls
    -> object bytes + proto
    -> generated typed Rust wrapper + linked object archive

consumer binary:
  generated wrapper
    -> Rust StandaloneRunner buffers and pack/unpack code
    -> xls_standalone_aot_* native runtime from libxls_aot_runtime.a
    -> linked XLS-generated object entrypoint
```

This is a meaningful packaging improvement: tests assert that the standalone
consumer does not dynamically depend on `libxls`. It is not a Rust-native
compiler or runtime yet, because native XLS still generates the object and
supplies the static invocation/callback implementation.

## What a replacement must account for

A later plan should first decide which current contracts are retained. The
inventory identifies these independently replaceable concerns:

| Concern | Present owner | Replacement decision |
| --- | --- | --- |
| Function graph and op semantics | XLS IR objects and LLVM lowering | Reuse a Rust IR representation, adapt existing libxls-owned IR temporarily, or translate from text. |
| Runtime acceleration mechanism | LLVM ORC for JIT, LLVM object emission for AOT | Select how Rust code obtains fast executable behavior and whether JIT and AOT share a lowerer. |
| Arbitrary-width and aggregate value semantics | XLS `Value`, type converter, and layout metadata | Preserve exact XLS semantics while either retaining the native-layout ABI or replacing it with a Rust-defined ABI. |
| Trace/assert/callback behavior | `InterpreterEvents`, `InstanceContext`, and runtime callbacks | Preserve `RunResult` behavior and establish what standalone generated code supports. |
| AOT object/wrapper protocol | Entrypoint proto, `JitFunctionType`, generated wrapper, static runtime | Keep compatibility for existing artifacts or version a new Rust artifact format. |
| Build and distribution model | `libxls` DSO and standalone runtime release assets | State separately whether the goal removes libxls during execution, during AOT builds, or entirely. |
| Concurrency and reuse | Mutable native and Rust runner buffers | Retain per-runner non-thread-safe behavior or introduce a different runner contract. |

Two boundaries are especially important:

1. Replacing only `IrFunctionJit` execution does not remove libxls from the
   crate if IR construction, parsing, optimization, DSLX conversion, or AOT
   generation still use the DSO.
1. Replacing only the standalone runtime does not remove libxls from AOT build
   scripts if object-code production still calls `xls_aot_compile_function`.

## Existing parity anchors

The following existing tests and examples provide starting observations for a
future replacement effort:

- `xlsynth/tests/ir_interpret_test.rs` compares interpreted and JIT evaluation.
- `xlsynth/benches/f32_add.rs` exercises JIT reuse and performance.
- `xlsynth/tests/aot-test-crate` covers generated AOT wrappers, compound and
  wide values, assertions, and parity between direct and standalone runners.
- `xlsynth/tests/aot-standalone-test-crate` verifies that generated standalone
  consumers execute without a dynamic `libxls` dependency.

Additional replacement validation will need a broader function/operator corpus,
event semantics, aggregates and arbitrary widths, and performance comparisons.

## Source map

Rust-side files:

- `xlsynth-sys/src/lib.rs`
- `xlsynth-sys/build.rs`
- `xlsynth/src/ir_package.rs`
- `xlsynth/src/lib_support.rs`
- `xlsynth/src/aot_lib.rs`
- `xlsynth/src/aot_entrypoint_metadata.rs`
- `xlsynth/src/aot_runner.rs`
- `xlsynth/src/aot_builder.rs`
- `xlsynth-aot-runtime/src/lib.rs`
- `xlsynth-aot-runtime/build.rs`

XLS-side files examined:

- `docs_src/ir_jit.md`
- `xls/public/c_api.{h,cc}`
- `xls/jit/function_jit.{h,cc}`
- `xls/jit/function_base_jit.{h,cc}`
- `xls/jit/ir_builder_visitor.{h,cc}`
- `xls/jit/llvm_type_converter.{h,cc}`
- `xls/jit/type_layout.{h,cc}`
- `xls/jit/llvm_compiler.{h,cc}`
- `xls/jit/orc_jit.{h,cc}`
- `xls/jit/aot_compiler.{h,cc}`
- `xls/jit/aot_entrypoint.{h,cc}`
- `xls/jit/aot_entrypoint.proto`
- `xls/jit/jit_runtime.{h,cc}`
- `xls/jit/jit_callbacks.{h,cc}`
