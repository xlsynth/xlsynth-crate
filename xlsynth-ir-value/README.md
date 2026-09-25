# xlsynth-ir-value

Native Rust representations and parsers for XLS IR values. The crate provides:

- `IrBits` bitvectors of arbitrary width.
- Recursive `IrValue` values for bits, tokens, tuples, and homogeneous arrays.
- Structural `Type` values, including element types for empty arrays.
- Typed value and positional or named `.irvals` parsing and formatting.

The crate has no dependency on `xlsynth` or `xlsynth-sys` and does not require
`libxls`.

```rust
use xlsynth_ir_value::{IrFormatPreference, IrValue, Type};

let value = IrValue::parse_typed("bits[129]:0x1_0000_0000_0000_0000").unwrap();
assert_eq!(value.type_(), Type::Bits(129));
assert_eq!(
    value.to_string_fmt(IrFormatPreference::Hex),
    "bits[129]:0x1_0000_0000_0000_0000"
);
```

`xlsynth-pir` reexports these APIs. Libxls conversions are provided by
`xlsynth_pir::libxls_bridge`.
