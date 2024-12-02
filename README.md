# XLS (AKA XLSynth) Rust Crate

Rust bindings to the functionality in the "Accelerated Hardware Synthesis" library.

```rust
extern crate xlsynth;

use xlsynth::{IrValue, IrPackage, IrFunction, XlsynthError};

fn sample() -> Result<IrValue, XlsynthError> {
    let package: IrPackage = xlsynth::convert_dslx_to_ir(
        "fn id(x: u32) -> u32 { x }",
        std::path::Path::new("/memfile/sample.x"))?;
    let mangled = xlsynth::mangle_dslx_name("sample", "id")?;
    let f: IrFunction = package.get_function(&mangled)?;
    let ft: IrValue = IrValue::parse_typed("bits[32]:42")?;
    f.interpret(&[ft])
}

fn main() {
    assert_eq!(sample().unwrap(), IrValue::parse_typed("bits[32]:42").unwrap());
}
```

### Development Notes

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
$ bazel build -c opt //xls/public:libxls.so
```

#### Pre-Commit

The `pre-commit` tool is used to help with local checks before PRs are created:

```shell
$ sudo apt-get install pre-commit
$ pre-commit install
$ pre-commit run --all-files
```

This `pre-commit` step is also run as part of continuous integration.
