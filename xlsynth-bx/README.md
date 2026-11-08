# xlsynth-bx

BX describes clocked hardware using blocks, registers, and explicit wiring. Its value language aims to implement a subset of DSLX with the same syntax, type inference, and evaluation rules. For the shared language, see the [DSLX reference manual](https://google.github.io/xls/dslx_reference/).

The Rust frontend parses BX, type-checks it, and adds verified functions and blocks to an XLS IR package. Define a combinational function in the same `.bx` file as its caller, import a DSLX function with `use`, or link an existing IR function with `extern fn`.

```text
fn transform(x: u8) -> u16 {
    let wide = x as u16;
    wide * wide
}

struct ResultPorts {
    data: u16,
    valid: bool,
}

pub block transform_pipe(clk, rst, valid_in: bool, x: u8) -> ResultPorts {
    reg data_q: u16;
    reg valid_q: bool = false;
    if valid_in {
        data_q := transform(x);
    }
    valid_q := valid_in;
    ResultPorts { data: data_q, valid: valid_q }
}
```

BX supports concrete bit types, booleans, tuples, named structs, and fixed-size arrays. Bit types of width 1 through 64 have shorthand names `u1` through `u64` and `s1` through `s64`. For width zero or widths above 64, use `uN[width]` or `sN[width]`, as in `uN[0]` or `sN[84]`.

The expression subset includes arithmetic, comparisons, bitwise and logical operations, shifts, concatenation, bit casts, field selection, array indexing, value-producing `if` expressions, and width slices such as `word[offset+:u8]`. See the manual’s [width slice examples](https://google.github.io/xls/dslx_reference/#width-slice).

### Scalar match expressions

Use `match` on a boolean or a signed or unsigned bit value in a function or
block. The first matching arm supplies the result. All arms share a result
type, which can also be an array, tuple, or struct.

```text
fn classify(op: u4) -> (u8, bool) {
    match op {
        0 | 2 => (u8:1, true),
        2..=5 => (u8:2, true),
        _ => (u8:0, false),
    }
}
```

Here `op == 2` selects the first arm. Use `a..b` to exclude the upper endpoint
and `a..=b` to include it. A single scalar pattern can also name a visible
module constant. Patterns support literal endpoints, `|`
alternatives, and `_` for any remaining value. The arms must cover the whole
subject type. For example, `true` and `false` cover a boolean without a
wildcard.

## Module constants

Declare a value with `const NAME = expression;` or `pub const NAME: Type = expression;`.
Constants are checked and evaluated in declaration order. Later functions, blocks,
and constant initializers can read their concrete values. Parameters and local
bindings can shadow a module constant.

```text
const STEP: u8 = 3;
fn advance(x: u8) -> u8 { x + STEP }
const START = advance(u8:4);

pub block counter(clk, rst) -> u8 {
    reg q: u8 = START;
    q := advance(q);
    q
}
```

Give an otherwise unsized integer initializer a type, as in `const STEP = u8:3;`.
Aggregate elements keep their inferred widths at every use. Write
`const WORDS: u8[2] = [1, 2];` when a consumer expects an array of bytes.
BX evaluates pure value expressions and calls to earlier concrete local functions
within its compile-time work limit. Evaluation follows the dependencies of the
result. For example, an out-of-bounds access in an unused helper binding can
go unevaluated in BX even when DSLX reports a constexpr error. Calls to imported
functions, extern functions, and callback-parameter specializations require
runtime evaluation, including when reached through a local helper. Constants can
supply loop bounds and width-slice expressions. Array sizes and bit widths in
type declarations still use literal dimensions.

## Defining functions

Write `fn name(parameters) -> ResultType { ... }` or `pub fn` at module scope. Use concrete types in the signature, including structs, tuples, and arrays. In function type annotations, both `(u8)` and `(u8,)` mean a one-element tuple. In a value expression, write `(x,)` to construct that tuple.

Inside a function, `let` can bind a name or destructure a tuple, as in `let (data, valid) = packet;`. Braced expressions and `if` arms can introduce their own local bindings. The final expression supplies the result. A function that returns `()` may omit the result annotation and end with a semicolon or an empty body.

Give a bound integer literal its intended width before passing it to a function.
For a parameter of type `u8`, write `let n: u8 = 1; take(n)` or `take(u8:1)`.
With `let n = 1`, BX infers `u1`. The pinned DSLX typechecker can accept the
wider call, but its interpreter can retain the narrower value and IR conversion
can fail on the mismatched signature.

Define a helper before the functions that call it, as in DSLX. A BX block can call a local function declared anywhere in the file. Calls in value expressions become XLS IR `invoke` nodes. Functions can also call imported DSLX functions and declared externs. Registers, register writes, and child instances belong in blocks. The checker rejects direct and indirect recursion.

For a register reset value or a loop range bound, use a constant expression that BX can evaluate during checking. For functions with type or integer parametrics, define the function in DSLX and import it with `use`.

BX emits XLS IR for runtime array operations. For an out-of-range index, IR reads the last element of a nonempty array, and `update` returns the original array. The native DSLX interpreter reports an evaluation error for either operation. The differential tests check both outcomes.

### Compile-time function parameters

Declare callable parameters between angle brackets in the order callers will supply them. Give each callback its argument and result types. As with a local function declaration, omitting the result annotation means `()`. Inside the function body, use the parameter as the callee of a direct call.

```text
fn fold4<F: fn(u16, u16) -> u16>(xs: u16[4], seed: u16) -> u16 {
    for (x, acc) in xs { F(acc, x) }(seed)
}

fn add16(a: u16, b: u16) -> u16 { a + b }

block sum(clk, xs: u16[4]) -> u16 {
    fold4<add16>(xs, u16:0)
}
```

Each callable argument must directly name a visible non-parametric BX function, a public, pure DSLX function from a `use` declaration, or a declared `extern fn`. For example, `use arithmetic;` lets a caller write `fold4<arithmetic::add16>(xs, seed)`. BX uses the declared callback argument types to resolve an imported DSLX specialization and checks its result type. An extern supplies its BX signature during checking. When lowering into a prelinked package, BX verifies the IR signature and purity of the linked function and its callees. The callback name is resolved at the call site. Names written directly in the template body follow its declaration scope.

BX checks the body against its declared signatures, then generates one function per distinct ordered set of callback identities. Only data arguments appear in the generated function's IR signature. Callable parameters cannot be assigned to a `let`, returned, or forwarded to another template. Inline lambdas remain available in `map` expressions. This function-parametric syntax is a BX extension.

### Map expressions

Use `map(array, |element| expression)` in a function or block to transform each element of a nonempty array in order. The input supplies the parameter type, and the lambda's result supplies the output element type. The output has the same length as the input.

```text
block widen(clk, words: u8[4]) -> u16[4] {
    map(words, |word| {
        let wide = word as u16;
        wide + u16:1
    })
}
```

The lambda can use its parameter, its own local bindings, and visible pure helpers. Reading a value from an enclosing scope is a capture and produces an error, even for a constant local binding. A nested `map` may use its enclosing lambda's parameter as the array to transform. Its callback gets a fresh parameter of its own.

Write `|x: u8| -> u16 { ... }` to give explicit types. Give input literals their intended width before mapping. For example, `[1, 3]` supplies two-bit elements, so `map([1, 3], |x| x + 1)` yields `[2, 0]`. For eight-bit arithmetic, write `map([u8:1, u8:3], |x| x + 1)`. A result annotation is useful for a body that returns an unsized literal, as in `|_| -> u16 { 0 }`. A constant integer range is also a valid input, for example `map(u8:0..=u8:3, |x| x * x)`.

BX emits an XLS IR `map` with a unary body function. If structural unrolling or generic block specialization creates multiple copies of the callback, those copies count toward the elaboration work limit. To emit SystemVerilog from a block that uses `map`, inline the map first.

```shell
xlsynth-driver bx2ir widen.bx > widen.ir
opt_main --passes=map_inlining widen.ir > widen.inlined.ir
xlsynth-driver block2sv widen.inlined.ir > widen.sv
```

### Counted for expressions

Use a DSLX-style `for` in a function or block to fold an array or a constant integer range into a value. The expression after the body supplies the initial accumulator. Each iteration binds the next element and the previous accumulator, then returns the next accumulator.

```text
fn sum_words(words: u16[4], bias: u32) -> u32 {
    for (word, sum): (u16, u32) in words {
        sum + (word as u32)
    }(bias)
}

block checksum(clk, words: u16[4], bias: u32) -> u32 {
    sum_words(words, bias)
}
```

The binder annotation is optional when the iterable and accumulator types can be inferred. Tuple patterns can destructure either binder. Array values can come from input ports, function calls, or other expressions. Integer ranges support both an exclusive limit (`lo..hi`) and an inclusive limit (`lo..=hi`). An empty loop returns the seed and its body still has to typecheck.

BX preserves the width of a let-bound value used as an iterable or seed. For a `u16` fold, write `let xs: u16[2] = [0, 1];` and `let seed: u16 = 0;` if those values are bound before the loop. An inline seed such as `}(0)` can take its width from the accumulator annotation. When checking range bounds and reset values, BX can evaluate constant counted loops within its compile-time work limit.

If only the body supplies the signed type of an aggregate member, give the seed or the accumulator that type explicitly. For example, with `x: s8`, write `let q = for (_, c) in u8:0..u8:1 { [x] }([s8:-1]);`. An accumulator annotation of `(u8, s8[1])` also gives the inline seed `[0xff]` the intended signed element type.

BX lowers a nonempty loop to an XLS IR `counted_for` with a body function and invariant arguments for captured values. To emit SystemVerilog with `block2sv`, run the XLS loop-unrolling pass first. Use `unroll_for!` for structural replication of child blocks.

```shell
xlsynth-driver bx2ir checksum.bx > checksum.ir
opt_main --passes=loop_unroll checksum.ir > checksum.unrolled.ir
xlsynth-driver block2sv checksum.unrolled.ir > checksum.sv
```

## Compile and link

Use `parse_module` and `check_module` to inspect BX. The checker resolves DSLX module imports and accepts signatures for prelinked externs. Pass the checked result to `lower_module`, or use `compile` for all three stages. Supply a PIR package containing your externs, or `empty_package("design")` when all calls use local definitions or imports.

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

BX also accepts concrete instances of public DSLX struct templates with `u32`
value parameters. Use a public DSLX alias to name the instance in a BX port,
register, or struct literal. For example, `float32::F32` names
`apfloat::APFloat<8, 23>` in the bundled DSLX standard library.

```text
use float32::F32;
use apfloat;

block fp_add(clk, a: F32, b: F32) -> F32 {
    apfloat::add(a, b)
}
```

For a custom port type, declare a DSLX alias such as
`pub type Half = apfloat::APFloat<5, 10>;` and import `Half`. An imported
function can return a new instance by naming the public template or a concrete
alias in its DSLX result type. BX recovers the instance's `u32` bindings from
the checked value, including parameters that do not appear in any field.

The pinned DSLX converter can emit invalid IR or dispatch to the wrong helper
when a type-valued parameter such as `T: type` is bound to a struct. When a
BX call passes or returns a concrete parametric struct, its imported DSLX
functions can use bits and arrays of bits as type-valued arguments. BX
diagnoses unsupported aggregate bindings, ambiguous
native symbols, and method dispatches whose specialization the converter API
cannot verify. DSLX `map` callbacks and indirect calls are outside this subset.
Functions such as `apfloat::add` specialize on ordinary value parameters.

If an imported function requires an implicit token, BX reports the DSLX call path and highlights the effectful operation when it can locate it in the original source. For example, a runtime `assert!(N >= u32:2, "min_width")` in a helper makes its caller effectful. When `N` is a width parametric, `const_assert!(N >= u32:2)` checks the invariant during DSLX typechecking and keeps the helper pure.

BX can follow a direct helper call in the body of a counted loop over a nonempty literal range, such as `for (_, acc) in u32:0..u32:1 { leaf::checked(acc) }(x)`. If the body branches between helpers, the range might be empty, or the initializer could introduce an earlier effect, BX reports the call path it can verify.

Lowering emits an ordinary IR `invoke` and includes the specialized function and its callees. Some functions, including `std::popcount`, contain a loop in their original IR. Consumers such as `xlsynth-codegen` that require unrolled loops need an optimization step for those bodies before RTL emission.

## Clocked blocks

- The first port is `clk`, followed by optional `rst` and typed data inputs. Pipelines conventionally put `valid_in` first among the data inputs. The final expression drives the outputs continuously. A struct result exposes its fields as output ports in declaration order, so those field names must be valid XLS IR port names. A tuple exposes `out_0`, `out_1`, and so on. Unit exposes zero outputs, and a scalar or array exposes one port named `out`. Nest a tuple or struct inside a named field to expose it as one aggregate port.
- A `reg` name in an expression reads its current value. Each register has exactly one `:=` update site. A surrounding `if` supplies the load enable. An ordinary value expression such as `q := if en { x } else { q };` supplies a mux on the data input. All registers update together at the active edge.
- A register initializer is a constant reset value and requires `rst`. Reset takes priority over load enable for that register. Registers declared without an initializer have unspecified power-up values. In a value expression, `rst` means reset is asserted. `ResetBehavior` configures polarity and timing for the hierarchy, defaulting to active-high synchronous reset.
- A module with one concrete block declaration selects it as top. With multiple concrete declarations, a single concrete `pub block` selects the top. For other configurations, select the desired block on the returned package.

### Calling a block

An unconditional `let` in a block body creates one persistent child instance. Pass the data inputs in declaration order. The child inherits its parent's clock and, if it declares `rst`, the parent's reset. Each call binding has its own registers. Reading the binding multiple times reads the same child's outputs.

Block lets accept DSLX tuple patterns. For example, `let (valid, (first, .., last)) = packet;` binds the valid bit and the first and last members of the nested tuple. Use `_` to discard one member and `..` to skip zero or more members at that tuple level. An optional annotation after the pattern gives the type of the entire initializer. Each nested pattern must match a tuple of the corresponding shape. A pattern may rebind a data value. In `let (x, previous) = (x + u8:1, x);`, both uses on the right read the previous `x`.

A pattern can also bind a child's tuple result directly, as in `let (valid_out, data_out) = pipe_reg(valid, payload);`. This instantiates the child once, and both bindings read that instance's outputs. Patterns also work in guarded register updates, value-producing `if` arms, and `unroll_for!` bodies.

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

### Generic blocks

Declare a type parameter after the block name and use it for ports, registers, and local type annotations. At each call, BX infers the type parameter from the data arguments.

```text
block pipe_reg<T>(clk, rst, valid_in: bool, data_in: T) -> (bool, T) {
    reg valid_q: bool = false;
    reg data_q: T;
    valid_q := valid_in;
    if valid_in { data_q := data_in; }
    (valid_q, data_q)
}

struct Packet { tag: u8, words: u32[2] }

block packet_stage(clk, rst, valid: bool, packet: Packet) -> (bool, Packet) {
    let stage = pipe_reg(valid, packet);
    stage
}
```

Here the type of `packet` gives `T = Packet`. The valid register clears on reset. The data register loads when `valid` is asserted and holds its last payload through bubbles. Calling `pipe_reg` with a `u32` payload creates a second block definition. Calls with the same payload type reuse one definition, and each call creates an instance with its own registers.

Use `<T, U>` to declare multiple type parameters. You can also spell a type parameter `<T: type>`. Parameters can occur within tuples or fixed-size arrays, as in `pair: (T, U[2])`. Repeated occurrences must agree on the concrete type, including signedness and struct identity. `bool` and `u1` are equivalent.

Use a call binding to supply the arguments for inference. When a payload literal is the only source for its type, give it a width, as in `let stage = pipe_reg(valid, u32:0);`. A concrete wrapper such as `packet_stage` supplies a top for compilation. BX validates each generic signature at declaration time, then checks the body at the concrete types of each instantiated specialization.

### Named callbacks on blocks

A block can declare type and callable parameters together. Supply the callback names in angle brackets at the call site. BX first infers types from data inputs, then resolves callbacks in declaration order. Each callback's input types must already be known. Its return type can supply a new type parameter or must agree with one already bound.

```text
use math;

block advance<In, Out, F: fn(In) -> Out>(clk, rst, prev: (bool, In)) -> (bool, Out) {
    let next = pipe_reg(prev.0, F(prev.1));
    next
}

block pipeline(clk, rst, packet: (bool, u8)) -> (bool, u32) {
    let widened = advance<math::widen>(packet);
    let squared = advance<math::square>(widened);
    squared
}
```

With public DSLX helpers `widen(u8) -> u16` and `square(u16) -> u32`, the first stage produces `(bool, u16)`. That pair supplies both the valid bit and the payload of the second stage. BX emits a direct call to each selected helper. Calls with identical concrete types and callback identities share a block definition. Each call still creates its own hardware instance.

The same syntax accepts local BX functions and signature-checked externs. For example, `extern fn widen(x: u8) -> u16 = "wide_ir";` lets a caller write `advance<widen>(packet)` when lowering into a package containing `wide_ir`. Callable parameters are usable only as direct callees in the block body.

### Replicating child blocks

`unroll_for!` folds a half-open integer range at compile time. The initializer
after the body supplies the first accumulator. Each iteration binds the current
index and accumulator, evaluates its local lets, and supplies the body result
to the next iteration. An empty range returns the initializer. BX must be
able to evaluate both bounds at compile time using literals, prior constant
lets, and its built-in value operations. The accumulator and the other values
used in the body can depend on input ports.

```text
block delay(clk, rst, enable: bool, x: u8) -> u8 {
    reg q: u8 = u8:0;
    if enable { q := x; }
    q
}

pub block delay_array(clk, rst, enables: bool[4], xs: u8[4]) -> u8[4] {
    let outputs = unroll_for! (i, acc): (u32, u8[4]) in u32:0..u32:4 {
        let element = delay(enables[i], xs[i]);
        update(acc, i, element)
    }(zero!<u8[4]>());
    outputs
}
```

This block elaborates four independently enabled instances named `element_0`
through `element_3`. Nested loops append one iteration ordinal per level.
BX disambiguates generated names that collide with other declarations.
An unconditional let initializer or block result can elaborate children in
its loop body. Pure loops can also appear inside other value expressions.
`zero!<T>()` supplies an all-zero value of a concrete type and
`update(array, index, value)` returns an array with one element replaced.
BX bounds total unrolling work per function or block, accounting for nested iterations
and materialized constant aggregates.

The [pipeline fixture](tests/fixtures/mul_add.bx) exercises two stages, a separately compiled DSLX multiplier, and load enables.

Hierarchy checks use the dependencies in the emitted IR. An `invoke` depends on its arguments, for example, so feedback that relies on a callee ignoring one of its arguments can be rejected. Put the register boundary on the feedback path explicitly.

To bound the size of IR types and register reset values, the checker limits each concrete type to 1,048,576 units. A bit type costs its width, with a minimum of one unit. An aggregate costs one unit plus the costs of its members. An empty array still counts one element because IR retains its element type. Expanded aggregate nesting, local function call chains, and block hierarchies are limited to 64 levels. A module can elaborate up to 4,096 distinct generic block specializations. Their bodies share an expansion budget of 1,048,576 work units. Each distinct body is checked once. Calls to blocks with callbacks also charge the calling body's budget for materializing data arguments and wiring each instance's ports.
