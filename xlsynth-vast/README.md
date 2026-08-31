# xlsynth-vast

`xlsynth-vast` builds and emits Verilog and SystemVerilog entirely in Rust. It
does not link against `libxls` or depend on the `xlsynth` or `xlsynth-sys`
crates.

A `VastFile` owns its complete syntax tree. Modules, expressions, data types,
and statement blocks are lightweight, copyable handles into that file; mutation
and inspection happen through the file itself. Handles from a different file are
rejected, and construction requires neither shared ownership nor interior
mutability.

Members retain their insertion order by default. Use `module_member_count` to
mark the start of a section and `hoist_module_declarations` to group its signal
declarations before statements without moving declarations across sections or
out of nested scopes.

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

## Automatic functions

Procedural bodies support both ordinary `case` (`block_add_case`) and wildcard
`casez` (`block_add_casez`) statements. `make_binary_pattern` constructs explicit
MSB-first patterns from `0`, `1`, and `?` digits, with validated widths and syntax.
`case_set_unique` adds SystemVerilog's uniqueness check for disjoint case labels.
Single-assignment `casez` arms omit `begin`/`end`. A function whose body consists
only of a `casez` also omits its outer `begin`/`end`; multi-statement blocks keep
their grouping.

Modules can contain typed, automatic helper functions. Function inputs and
local declarations keep their insertion order, and `function_result` identifies
the implicit result variable assigned by the function body. SystemVerilog scalar,
vector, and packed-array return types explicitly use `logic`; external typedefs
retain their declared type without an additional storage keyword.

```rust
use xlsynth_vast::{VastFile, VastFileType};

let mut file = VastFile::new(VastFileType::SystemVerilog);
let module = file.add_module("arithmetic");
let byte = file.make_bit_vector_type(8, false);
let helper = file.add_function(module, "add_bytes", &byte).unwrap();
let lhs = file.function_add_input(helper, "lhs", &byte).unwrap();
let rhs = file.function_add_input(helper, "rhs", &byte).unwrap();
let sum = file.make_add(&lhs.to_expr(), &rhs.to_expr());
file.block_add_blocking_assignment(
    file.function_body(helper),
    &file.function_result(helper).to_expr(),
    &sum,
);

let call = file.make_function_call("add_bytes", &[&lhs.to_expr(), &rhs.to_expr()]);
assert_eq!(file.emit_expression(&call), "add_bytes(lhs, rhs)");
```

`function_add_logic_input` selects `logic` rather than `reg` for a function
argument. `function_add_reg` and `function_add_logic` declare function-local
values. `make_function_call` also supports SystemVerilog system functions, such
as `$signed` and `$unsigned`. Use `make_indexable_expression` when indexing a
computed expression; emission inserts the required parentheses.

## Integer literals

`VastFile::make_literal` parses typed values such as `bits[8]:42` and renders
them according to `LiteralFormat`. `Binary`, `Hex`, `SignedDecimal`, and
`UnsignedDecimal` preserve the source bit width; `UnsizedBinary`,
`UnsizedDecimal`, and `UnsizedHex` omit it. Unsized decimal values must fit in
`i32::MAX`, while unsized binary and hexadecimal values must fit in `u32::MAX`.
These bounds apply to the represented value, not the declared source width.

Use `VastFile::make_unsized_decimal_literal` for signed decimal values across
the complete `i32` range, including negative values.
