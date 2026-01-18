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
    let mol: IrValue = IrValue::u32(42);

    // Use the IR interpreter.
    let interp_result: IrValue = f.interpret(&[mol.clone()])?;

    // Use the IR JIT.
    let jit = xlsynth::IrFunctionJit::new(&f)?;
    let jit_result: xlsynth::RunResult = jit.run(&[mol])?;
    assert_eq!(jit_result.value, interp_result);

    Ok(jit_result.value)
}

fn main() {
    assert_eq!(sample().unwrap(), IrValue::u32(42));
}
```

## Project Structure

The `xlsynth` crate builds on top of the shared library `libxls.{so,dylib}` releases created in
<https://github.com/xlsynth/xlsynth/releases/> -- this is the underlying C/C++ core.

- `xlsynth-sys`: wraps the shared library with Rust FFI bindings
- `xlsynth`: provides Rust objects for interacting with core facilities; this includes:
  - DSLX parsing/typechecking, conversion to XLS IR
  - IR building
  - JIT compilation and IR interpretation
  - Creating Verilog AST directly for tools that need to do so
  - Building Rust and SystemVerilog bridges for interacting with XLS/DSLX artifacts
- `sample-usage`: demonstrates use of the APIs provided by the `xlsynth` crate
- `xlsynth-estimator`: Rust implementation of the XLS IR operation-level delay estimation
  methodology
- `xlsynth-g8r`: _experimental_ XLS IR to gate mapping library

## Example Use

This shows sample use of the driver program which integrates XLS functionality for command line use:

```shell
echo 'fn f(x: u32, y: u32) -> u32 { x + y }' > /tmp/add.x
cargo run -p xlsynth-driver -- dslx2ir --dslx_input_file /tmp/add.x --dslx_top f > /tmp/add.ir
cargo run -p xlsynth-driver -- ir2gates /tmp/add.ir
```

## Installing In Custom Environments

By default the crate attempts to download the shared library and DSLX standard library that it needs
for out-of-the-box operation. However, this can also be specified manually at build time with the
following environment variables:

```shell
cargo clean
export XLS_DSO_PATH=$HOME/opt/xlsynth/lib/libxls-v0.0.173-ubuntu2004.so
export DSLX_STDLIB_PATH=$HOME/opt/xlsynth/latest/xls/dslx/stdlib/
cargo build -vv -p xlsynth-sys |& grep "Using XLS_DSO_PATH"
# Ensure host binaries (including Cargo build scripts) can locate the DSO at build/test time.
export LD_LIBRARY_PATH="$(dirname "$XLS_DSO_PATH")":$LD_LIBRARY_PATH
cargo test --workspace
```

Note: `XLS_DSO_PATH` and `DSLX_STDLIB_PATH` must be set together; setting only one is treated as a
misconfiguration.

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
$ ls $DEV_XLS_DSO_WORKSPACE/bazel-bin/xls/public/libxls.* | egrep '(.dylib|.so)$'
/home/cdleary/proj/xlsynth//bazel-bin/xls/public/libxls.so
$ cargo clean  # Make sure we pick up the new env var.
$ cargo test -vv |& grep -i "DSO from workspace"
[xlsynth-sys ...] cargo:info=Using DSO from workspace: ...
```

Where in `~/proj/xlsynth/` (the root of the xlsynth workspace) we build the DSO with

```shell
bazel build -c opt //xls/public:libxls.so
```

### Pre-Commit

The `pre-commit` tool is used to help with local checks before PRs are created:

```shell
sudo apt-get install pre-commit
pre-commit install
pre-commit run --all-files
```

This `pre-commit` step is also run as part of continuous integration.

### Repository scripts

Python helper scripts now live under `scripts/` at the repo root. Invoke them from the workspace root, e.g.:

```shell
python3 scripts/update_golden_files.py
python3 scripts/run_all_fuzz_tests.py --fuzz-bin-args=-max_total_time=5
```

### Developer note: xlsynth DSO/dylib release versioning

The following versioning convention applies to the underlying DSO/dylib artifacts (e.g., `libxls.so`, `libxls.dylib`) released by the [xlsynth/xlsynth](https://github.com/xlsynth/xlsynth) repository, not to the versioning of this Rust crate itself. Occasionally, we need to create a successor to a patch release without bumping the minor or major version. In these cases, we use a dash-suffixed version tag (e.g., `v0.0.219-1`, `v0.0.219-2`). The plain form (e.g., `v0.0.219`) is implicitly equivalent to `v0.0.219-0`. This allows us to cherry-pick fixes onto a patch release when necessary.

**Note:** We hope to eventually switch to bumping the `v0.X.0` field for such successors, so that these dash releases can instead become patch releases, using the patch field as intended by semantic versioning. Until then, please be aware of this convention when working with release artifacts and tooling.
