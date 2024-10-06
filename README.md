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

To link against a local version of the public API, instead of a released version,
supply the `DEV_XLS_DSO_WORKSPACE` environment variable pointing at the workspace
root where the built shared library resides; e.g.

```shell
$ export DEV_XLS_DSO_WORKSPACE=$HOME/proj/xlsynth/
$ ls $DEV_XLS_DSO_WORKSPACE/bazel-bin/xls/public/libxls.so
/home/cdleary/proj/xlsynth//bazel-bin/xls/public/libxls.so
$ cargo test -vv |& grep -i workspace
[xlsynth-sys ...] cargo:info=Using DSO from workspace: ...
```