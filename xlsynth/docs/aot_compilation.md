# Native AOT compilation

The supported AOT implementation lives in `xlsynth-pir-compiler`. It lowers
optimized XLS IR to portable IR (PIR), compiles PIR with Cranelift, and emits a
native object together with generated Rust wrappers. Executing the generated
code requires only `xlsynth-pir-compiler-runtime`; it does not link a separate
XLS runtime archive.

Two build-script entry points are available:

- `TypedDslxAotPackageBuilder` typechecks and lowers DSLX, runs XLS IR
  optimization, constructs nominal DSLX type metadata, and compiles one or more
  entrypoints into a shared native object.
- `TypedIrAotPackageBuilder` consumes checked-in XLS IR and explicit PIR AOT
  metadata before compiling the selected entrypoints.

Generated Rust definitions preserve DSLX structs, enums, aliases, arrays, and
tuples. Runtime scalar and wide-bit values come from
`xlsynth-pir-compiler-runtime`. Generated aggregate types provide
`all_zeros()` when callers need an explicit zero-valued instance.

The DSLX path is normally used from `build.rs`:

```rust,ignore
use xlsynth_pir_compiler::aot::{
    CargoDslxEnv, DslxAotEntrypoint, TypedDslxAotPackageBuilder,
};

let dslx = CargoDslxEnv::new()?;
TypedDslxAotPackageBuilder::new("example")
    .add_dslx_file(
        std::path::Path::new("src/example.x"),
        dslx.dslx_options(),
        [],
        [DslxAotEntrypoint::new("step", "step")],
    )
    .build_and_export_env("EXAMPLE_AOT_RS")?;
```

The generated source is included by the consuming crate, and Cargo links the
Cranelift-produced object emitted by the build script. See
`xlsynth-pir-compiler/tests/aot-dslx-test-crate` and
`xlsynth-pir-compiler/tests/aot-ir-test-crate` for complete integration
examples.
