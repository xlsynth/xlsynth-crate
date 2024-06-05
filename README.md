# XLS (AKA XLSynth) Rust Crate

Rust bindings to the functionality in the "Accelerated Hardware Synthesis" library.

```rust
extern crate xlsynth;

use xlsynth::{IrValue, IrPackage, IrFunction, XlsynthError};

fn sample() -> Result<IrValue, XlsynthError> {
    let package: IrPackage = xlsynth::convert_dslx_to_ir("fn id(x: u32) -> u32 { x }")?;
    let mangled = xlsynth::mangle_dslx_name("test_mod", "id")?;
    let f: IrFunction = package.get_function(&mangled)?;
    let ft: IrValue = IrValue::parse_typed("bits[32]:42")?;
    f.interpret(&[ft])
}

fn main() {
    assert_eq!(sample().unwrap(), IrValue::parse_typed("bits[32]:42").unwrap());
}
```