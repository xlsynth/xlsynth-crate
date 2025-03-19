# XLS (AKA XLSynth) Rust Crate

Rust bindings to the functionality in the "Accelerated Hardware Synthesis" library.

```rust
use xlsynth::{DslxToIrPackageResult, IrValue, IrPackage, IrFunction, XlsynthError};

fn sample() -> Result<IrValue, XlsynthError> {
    let converted: DslxToIrPackageResult = xlsynth::convert_dslx_to_ir(
        "fn id(x: u32) -> u32 { x }",
        std::path::Path::new("/memfile/sample.x"),
        &xlsynth::DslxConvertOptions::default())?;
    assert!(converted.warnings.is_empty());
    let package: IrPackage = converted.ir;
    let mangled = xlsynth::mangle_dslx_name("sample", "id")?;
    let f: IrFunction = package.get_function(&mangled)?;
    let mol: IrValue = IrValue::make_ubits(32, 42)?;

    // Use the IR interpreter.
    let interp_result: IrValue = f.interpret(&[mol.clone()])?;

    // Use the IR JIT.
    let jit = xlsynth::IrFunctionJit::new(&f)?;
    let jit_result: xlsynth::RunResult = jit.run(&[mol])?;
    assert_eq!(jit_result.value, interp_result);

    Ok(jit_result.value)
}

fn main() {
    assert_eq!(sample().unwrap(), IrValue::make_ubits(32, 42).unwrap());
}
```

## Project Structure

The `xlsynth` crate builds on top of the shared library `libxls.{so,dylib}` releases created in
<https://github.com/xlsynth/xlsynth/releases/> -- this is the underlying C/C++ core.

* `xlsynth-sys`: wraps the shared library with Rust FFI bindings
* `xlsynth`: provides Rust objects for interacting with core facilities; this includes:
  * IR building
  * JIT compilation
  * Creating Verilog AST directly for tools that need to do so
  * Building Rust and SystemVerilog bridges for interacting with XLS/DSLX artifacts
* `sample-usage`: demonstrates use of the APIs provided by the `xlsynth` crate
* `xlsynth-estimator`: Rust implementation of the XLS IR operation-level delay estimation
  methodology
* `xlsynth-g8r`: _experimental_ XLS IR to gate mapping library

## Development Notes

The `xlsynth` Rust crate leverages a dynamic library with XLS' core functionality (i.e. `libxls.so`
/ `libxls.dylib`).

The DSO is built and released for multiple platforms via GitHub actions at
[xlsynth/xlsynth/releases](https://github.com/xlsynth/xlsynth/releases/).

The version that this crate expects is described in `xlsynth-sys/build.rs` as
`RELEASE_LIB_VERSION_TAG`. By default, this crate pulls the dynamic library from the targeted
release.

To link against a local version of the public API, instead of a released version, supply the
`DEV_XLS_DSO_WORKSPACE` environment variable pointing at the workspace root where the built shared
library resides; e.g.

```shell
$ export DEV_XLS_DSO_WORKSPACE=$HOME/proj/xlsynth/
$ ls $DEV_XLS_DSO_WORKSPACE/bazel-bin/xls/public/libxls.so
/home/cdleary/proj/xlsynth//bazel-bin/xls/public/libxls.so
$ cargo clean  # Make sure we pick up the new env var.
$ cargo test -vv |& grep -i "DSO from workspace"
[xlsynth-sys ...] cargo:info=Using DSO from workspace: ...
```

Where in `~/proj/xlsynth/` (the root of the xlsynth workspace) we build the DSO with

```shell
bazel build -c opt //xls/public:libxls.so
```

Note that on OS X you additionally will have to set:

```shell
export DYLD_LIBRARY_PATH=$HOME/proj/xlsynth/bazel-bin/xls/public/:$DYLD_LIBRARY_PATH
```

### Pre-Commit

The `pre-commit` tool is used to help with local checks before PRs are created:

```shell
sudo apt-get install pre-commit
pre-commit install
pre-commit run --all-files
```

This `pre-commit` step is also run as part of continuous integration.
