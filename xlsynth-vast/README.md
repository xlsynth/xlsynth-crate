# xlsynth-vast

`xlsynth-vast` builds and emits Verilog and SystemVerilog entirely in Rust. It
does not link against `libxls` or depend on the `xlsynth` or `xlsynth-sys`
crates.

A `VastFile` owns its complete syntax tree. Modules, expressions, data types,
and statement blocks are lightweight, copyable handles into that file; mutation
and inspection happen through the file itself. Handles from a different file are
rejected, and construction requires neither shared ownership nor interior
mutability.

```rust
use xlsynth_vast::{VastFile, VastFileType};

let mut file = VastFile::new(VastFileType::SystemVerilog);
let module = file.add_module("passthrough");
let data_type = file.make_bit_vector_type(8, false);
let input = file.add_input(module, "input_data", &data_type);
let output = file.add_output(module, "output_data", &data_type);
let assignment = file.make_continuous_assignment(&output.to_expr(), &input.to_expr());
file.add_member_continuous_assignment(module, assignment);

assert_eq!(
    file.emit(),
    "module passthrough(\n  input wire [7:0] input_data,\n  output wire [7:0] output_data\n);\n  assign output_data = input_data;\nendmodule\n",
);
```

Reusable register-building and expression-reduction utilities are available
through `xlsynth_vast::helpers`. Configuration-specific register templates are
provided separately by `xlsynth::vast_helpers`.

## Integer literals

`VastFile::make_literal` parses typed values such as `bits[8]:42` and renders
them according to `LiteralFormat`. `Binary`, `Hex`, `SignedDecimal`, and
`UnsignedDecimal` preserve the source bit width; `UnsizedBinary`,
`UnsizedDecimal`, and `UnsizedHex` omit it. Unsized decimal values must fit in
`i32::MAX`, while unsized binary and hexadecimal values must fit in `u32::MAX`.
These bounds apply to the represented value, not the declared source width.

Use `VastFile::make_unsized_decimal_literal` for signed decimal values across
the complete `i32` range, including negative values.
