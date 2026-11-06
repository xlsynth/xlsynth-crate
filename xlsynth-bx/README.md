# xlsynth-bx

BX describes clocked hardware using blocks, registers, and explicit wiring. Its value language aims to implement a subset of DSLX with the same syntax, type inference, and evaluation rules. For the shared language, see the [DSLX reference manual](https://google.github.io/xls/dslx_reference/).

The Rust frontend parses BX, type-checks it, and adds verified blocks to an XLS IR package. DSLX functions can supply combinational logic through a `use` declaration or a prelinked `extern fn` declaration.

```text
extern fn math::transform(x: u8) -> u16;

struct ResultPorts {
    data: u16,
    valid: bool,
}

pub block transform_pipe(clk, rst, valid_in: bool, x: u8) -> ResultPorts {
    reg data_q: u16;
    reg valid_q: bool = false;
    if valid_in {
        data_q := math::transform(x);
    }
    valid_q := valid_in;
    ResultPorts { data: data_q, valid: valid_q }
}
```

BX supports concrete bit types, booleans, tuples, named structs, and fixed-size arrays. Bit types of width 1 through 64 have shorthand names `u1` through `u64` and `s1` through `s64`. For width zero or widths above 64, use `uN[width]` or `sN[width]`, as in `uN[0]` or `sN[84]`.

The expression subset includes arithmetic, comparisons, bitwise and logical operations, shifts, concatenation, bit casts, field selection, array indexing, value-producing `if` expressions, and width slices such as `word[offset+:u8]`. See the manual’s [width slice examples](https://google.github.io/xls/dslx_reference/#width-slice).

## Compile and link

Use `parse_module` and `check_module` to inspect BX. The checker resolves DSLX module imports and accepts signatures for prelinked externs. Pass the checked result to `lower_module`, or use `compile` for all three stages. Supply a PIR package containing your externs, or `empty_package("design")` when all function calls come from imports.

```rust
use xlsynth_bx::{compile, ResetBehavior};
use xlsynth_pir::ir_parser::Parser;

let ir = r#"package design
fn __math__transform(x: bits[8] id=1) -> bits[16] {
  ret zero_ext.2: bits[16] = zero_ext(x, new_bit_count=16, id=2)
}"#;
let linked = Parser::new(ir).parse_and_validate_package().map_err(|e| e.to_string())?;
let bx = r#"
extern fn math::transform(x: u8) -> u16;
block transform_pipe(clk, x: u8) -> u16 {
    reg q: u16;
    q := math::transform(x);
    q
}"#;
let package = compile(bx, linked, ResetBehavior::default())?;
let block_ir = package.to_string();
# Ok::<(), String>(())
```

Declare and call a prelinked DSLX function by its module-qualified name, such as `math::transform`. The qualifier is a single module identifier. The linker looks for the corresponding DSLX symbol (`__math__transform`) in the supplied IR package, whatever the package is named. Two modules can export functions with the same name.

For an unqualified declaration such as `extern fn transform(x: u8) -> u16;`, call `transform(x)`. The linker tries the plain IR name first and then the DSLX symbol for the package's own module. To bind a specialization or choose a local alias, supply the exact IR symbol, as in `extern fn widen(x: u8) -> u16 = "__math__transform";`.

The linker checks argument order, widths, aggregate shapes, and result type before emitting `invoke`. Signedness and struct names are BX-level distinctions. Their IR representations are bits and tuples respectively. The linked function and its callees must have pure IR bodies.

### Using DSLX modules and members

`use arithmetic;` resolves `arithmetic.x` on DSLX's module search path. For a nested module, `use ip::arithmetic;` makes its public functions and struct types available as `arithmetic::name`. To bind individual members, write `use ip::arithmetic::clamp;` or group several under the same path, as in `use ip::arithmetic::{clamp, Limits};`. Each leaf binds its final name in the BX module. To migrate an aliased import, use the module’s final name or select the member directly. If two modules export the same member name, qualify it with distinct module bindings. If the module names also collide, bind one module and the other member directly. For example, `use a::math; use b::math::clamp;` gives you `math::clamp(...)` and `clamp(...)`. BX checks every `use` declaration, including unused ones.

```text
use std;
use ip::arithmetic::clamp;

block mul_pipe(clk, a: u8, b: u5, cap: u13) -> u13 {
    let product = std::umul(a, b);
    reg q: u13;
    q := clamp(product, cap);
    q
}
```

For this example, place a public `clamp(x: u13, cap: u13) -> u13` function in `dslx/ip/arithmetic.x` and pass `dslx` as a search directory.

```shell
xlsynth-driver bx2ir mul_pipe.bx --dslx-path dslx > mul_pipe.ir
xlsynth-driver block2sv mul_pipe.ir > mul_pipe.sv
```

To map the same block to library cells, supply a normalized Liberty library.

```shell
xlsynth-driver bx2gv mul_pipe.bx --dslx-path dslx --liberty-proto cells.proto -o mul_pipe.gv
```

```rust,no_run
use std::path::Path;
use xlsynth_bx::{compile_with_options, empty_package, ImportOptions, ResetBehavior};

let source = std::fs::read_to_string("mul_pipe.bx").map_err(|e| e.to_string())?;
let options = ImportOptions {
    additional_search_paths: vec![Path::new("dslx")],
    ..ImportOptions::default()
};
let package = compile_with_options(&source, empty_package("design"), ResetBehavior::default(), &options)?;
# Ok::<(), String>(())
```

The importer consults the additional directories in the supplied order. `dslx_stdlib_path` can override the bundled library location. The Rust frontend records imports, type declarations, aliases, and field order from one source snapshot. Its `parse_dslx_outline` API exposes the written types and enum values. BX gives DSLX a private copy of that snapshot with relocated import paths, so DSLX evaluates dimensions and specializes functions against the selected modules. Emitted IR uses the original module names and source positions. For separate checking and lowering, pass the same options to `check_module_with_options`. The checked module retains the imported type definitions and function implementations, so `lower_module` uses the sources resolved at check time.

BX checks the arguments, then asks the DSLX typechecker to infer the concrete function signature. Here `std::umul` returns `u13` because its result width is the sum of the operand widths. A call with `u8` and `u12` in the same block gets a separate `u20` specialization. Write a standalone integer argument with its type, as in `std::round_up_to_nearest(x, u32:4)`.

BX can name public, non-parametric DSLX structs in ports, registers, and expressions. For example, given `pub struct Header { tag: u8, valid: bool }` in `packet.x`:

```text
use packet::Header;

block tag_header(clk, h: Header) -> Header {
    Header { tag: h.tag + u8:1, valid: h.valid }
}
```

BX can read fields through a private struct nested inside a public one. To pass the nested value itself as a DSLX function argument, expose its type through a public name or type alias. A public DSLX type alias preserves the identity of its underlying struct. Imported calls accept these structs as arguments and results, alongside bits, booleans, tuples, and fixed-size arrays. Functions must be public and pure. Linked functions and their module path components use letters, digits, and underscores so their generated names are valid XLS IR identifiers. Parametric values must be inferable from the arguments and DSLX defaults. Functions such as `std::lsbs` require an explicit result-width parameter and are outside this syntax.

Lowering emits an ordinary IR `invoke` and includes the specialized function and its callees. Some functions, including `std::popcount`, contain a loop in their original IR. Consumers such as `xlsynth-codegen` that require unrolled loops need an optimization step for those bodies before RTL emission.

## Clocked blocks

- The first port is `clk`, followed by optional `rst` and typed data inputs. Pipelines conventionally put `valid_in` first among the data inputs. The final expression drives the outputs continuously. A struct result exposes its fields as output ports in declaration order, so those field names must be valid XLS IR port names. A tuple exposes `out_0`, `out_1`, and so on. Unit exposes zero outputs, and a scalar or array exposes one port named `out`. Nest a tuple or struct inside a named field to expose it as one aggregate port.
- A `reg` name in an expression reads its current value. Each register has exactly one `:=` update site. A surrounding `if` supplies the load enable. An ordinary value expression such as `q := if en { x } else { q };` supplies a mux on the data input. All registers update together at the active edge.
- A register initializer is a constant reset value and requires `rst`. Reset takes priority over load enable for that register. Registers declared without an initializer have unspecified power-up values. In a value expression, `rst` means reset is asserted. `ResetBehavior` configures polarity and timing for the hierarchy, defaulting to active-high synchronous reset.
- A module with one block selects it as top. With multiple blocks, a single `pub block` selects the top. For other configurations, select the desired block on the returned package.

### Calling a block

An unconditional `let` in a block body creates one persistent child instance. Pass the data inputs in declaration order. The child inherits its parent's clock and, if it declares `rst`, the parent's reset. Each call binding has its own registers. Reading the binding multiple times reads the same child's outputs.

```text
struct LaneOut { data: u8, valid: bool }
extern block lane(clk, rst, x: u8) -> LaneOut;

block pair(clk, rst, x: u8, y: u8) -> (LaneOut, LaneOut) {
    let left = lane(x);
    let right = lane(y);
    (left, right)
}
```

Use the same call syntax for a BX block defined anywhere in the module. An `extern block` declaration supplies the signature of an existing block in the linked IR package. BX checks data-port names, directions and types. It identifies the physical clock and reset by their roles and checks that the reset policy matches. Inside the parent, the result has the child's declared type, so `left.valid` selects one output.

For feedback wiring, `inst cell: delay;` gives the child a name before its inputs are connected. For example, `cell.d := cell.out + step;` is valid when `cell.out` reads a register. Connect each data input once. Both forms lower to Block IR instantiations, and a same-cycle combinational loop is a lowering error.

The [pipeline fixture](tests/fixtures/mul_add.bx) exercises two stages, a separately compiled DSLX multiplier, and load enables.

Hierarchy checks use the dependencies in the emitted IR. An `invoke` depends on its arguments, for example, so feedback that relies on a callee ignoring one of its arguments can be rejected. Put the register boundary on the feedback path explicitly.

To bound the size of IR types and register reset values, the checker limits each concrete type to 1,048,576 units. A bit type costs its width, with a minimum of one unit. An aggregate costs one unit plus the costs of its members. An empty array still counts one element because IR retains its element type. Expanded aggregate nesting and block hierarchies are limited to 64 levels.
