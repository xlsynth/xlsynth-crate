# xlsynth-aot-runtime

`xlsynth-aot-runtime` supplies reusable runtime support for executing generated
standalone AOT artifacts. It can be consumed directly from Cargo or integrated
into a Bazel final link.

## Native Link Ownership

Direct Cargo consumers use the default native link mode:

```text
xlsynth-aot-runtime
  -> links libxls_aot_runtime.a
  -> receives the native runtime symbols from the released archive
```

Bazel consumers that declare the standalone runtime dependency use declared
link mode:

```text
XLS_AOT_RUNTIME_LINK_MODE=declared

Bazel target
  -> depends on @<name>_runtime//:xls_aot_runtime_source_dep
  -> depends on xlsynth-aot-runtime

xlsynth-aot-runtime
  -> emits no native runtime archive link request
```

Declared link mode changes only which build graph supplies the native runtime
symbols. It does not change the standalone AOT runtime ABI or execution
behavior.

Use `native` mode, the default, when Cargo must link the released
`libxls_aot_runtime.a` archive. Use `declared` mode only when the enclosing
build graph declares an equivalent runtime dependency, such as the source-backed
Bazel export from `rules_xlsynth`; otherwise the final link will have no
provider for the standalone runtime symbols.
