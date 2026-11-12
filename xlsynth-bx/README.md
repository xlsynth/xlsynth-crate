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

BX supports concrete bit types, booleans, numeric enums, tuples, named structs, and fixed-size arrays. Bit types of width 1 through 64 have shorthand names `u1` through `u64` and `s1` through `s64`. For width zero or widths above 64, use `uN[width]` or `sN[width]`, as in `uN[0]` or `sN[84]`.

The expression subset includes arithmetic, comparisons, bitwise and logical operations, shifts, concatenation, bit casts, field selection, array indexing, value-producing `if` expressions, and width slices such as `word[offset+:u8]`. See the manual’s [width slice examples](https://google.github.io/xls/dslx_reference/#width-slice).

Use `++` to concatenate arrays with the same element type. The result length is the sum of the operand lengths, and the left operand's elements come first. Each operand is typed independently. For example, `[u8:1, u8:2] ++ [u8:3]` produces a `u8[3]`.

Give an array literal its type directly, as in `u8[4]:[1, 2, 3, 4]`. A trailing `...` fills the remaining slots with the last written value, which can be computed at runtime. For example, `u8[4]:[1, x, ...]` produces `[1, x, x, x]`. An enclosing annotation can supply the length too, as in `let lanes: u8[4] = [1, x, ...];`. When passing a filled literal directly to a pure function, put the size on the argument, as in `sum(u8[4]:[1, x, ...])`.

Use `as` to unpack a bit value into a one-dimensional array of bits or to pack the array back into a bit value. The scalar width must equal the element width times the array length. Element zero holds the most significant bits. For example, `u16:0x12ab as u8[2]` gives `[u8:0x12, u8:0xab]`, and casting that array to `u16` gives `0x12ab`. Signed, unsigned, and `bool` elements preserve their bit patterns. The rule also applies to zero-width values and empty arrays.

BX uses the declared signedness in both hardware and constant expressions. For example, `([u4:8, u4:3] as s8) as s16` sign-extends to `-125`. Compiled DSLX hardware agrees. The pinned DSLX bytecode evaluator can lose the signed tag on an array cast and evaluate the same constant as `131`. BX preserves empty-array types in its IR. The pinned XLS IR interpreter and constant-folding pass can fail when they need to materialize an empty array value.

### Pure DSLX builtins

Functions and blocks can call these builtins directly.

| Operations | Builtins |
| --- | --- |
| Bits | `and_reduce`, `or_reduce`, `xor_reduce`, `rev`, `clz`, `ctz`, `bit_slice_update` |
| Encoding and selection | `encode`, `decode<T>`, `one_hot`, `one_hot_sel`, `priority_sel`, `gate!` |
| Extension and multiplication | `signex`, `widening_cast<T>`, `checked_cast<T>`, `smulp`, `umulp` |
| Arrays | `array_rev`, `array_size`, `array_slice`, `zip`, `update`, `map` |
| Typed values and shape queries | `zero!<T>()`, `all_ones!<T>()`, `bit_count<T>()`, `element_count<T>()` |
| Compile-time values | `ceillog2`, `configured_value_or` |

```text
const WORD_WIDTH = bit_count<u8>();
fn parity_of_reversed(x: uN[WORD_WIDTH]) -> bool { xor_reduce(rev(x)) }
fn set_cell(grid: u8[2][3], row: u32, col: u32, value: u8) -> u8[2][3] {
    update(grid, (row, col), value)
}
```

Use a module constant to put a shape query in a concrete signature dimension, as in `const WIDTH = bit_count<u8>();`.

The `one_hot` priority flag and `ceillog2` operand are compile-time values. When `ceillog2` is a `map` callback, the input array must be known at compile time. `configured_value_or(label, default)` takes a byte-array label and returns its compile-time default in BX. `array_slice(xs, start, witness)` takes the result length from the witness array type and repeats the last source element past the end.

`checked_cast<T>` enforces representability when evaluated as a constant and lowers to the ordinary bits cast in hardware. The two outputs of `umulp` and `smulp` are partial products whose sum is the product truncated to the operand width. Channel and token operations belong to DSLX procs.

### Numeric enums

Declare each member's encoding and refer to it by name in functions and blocks.

```text
pub enum State : u3 { Idle = 0, Busy = 5 }

block controller(clk, rst, start: bool) -> State {
    reg state: State = State::Idle;
    state := match state {
        State::Idle => if start { State::Busy } else { State::Idle },
        State::Busy => State::Idle,
    };
    state
}
```

Members can use constant expressions, including earlier constants and pure local functions. Omitting the underlying type makes BX infer a common bit type from the members. Multiple names can share an encoding. A match covers an enum when it covers every distinct declared encoding. Use a wildcard arm to specify the hardware result for other bit patterns.

Enum types retain their declaration identity in ports, registers, arrays, structs, and function signatures. Compare values of the same enum with `==` or `!=`. Cast to a bit type for arithmetic or ordering, for example `state as u3`. A runtime cast from bits to an enum uses the same bit conversion as compiled DSLX IR. A required compile-time cast from bits needs the enum's exact width and a declared encoding. `zero!<State>()` requires a zero-valued member.

Use public numeric enums from BX or DSLX libraries with `use protocol;` and `protocol::State::Idle`, or `use protocol::State;` and `State::Idle`.

In imported DSLX, cast both enum operands to their backing type before an ordered comparison, for example `(a as u3) < (b as u3)`. The DSLX typechecker reports the invalid operator when BX checks the import.

### Match expressions

Use `match` on a tuple, boolean, numeric enum, or signed or unsigned bit value
in a function or block. The first matching arm supplies the result. All arms share a result
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
module constant, enum member, or function value parametric. Patterns support literal endpoints, `|`
alternatives, and `_` for any remaining value. The arms must cover the whole
subject type. For example, `true` and `false` cover a boolean without a
wildcard.

Tuple patterns match corresponding fields. Use a scalar pattern or a nested
tuple pattern for each field. A wildcard field accepts any value in that
position. Patterns can supply widths to unsized fields in tuple literals and
conditional branches that return tuple literals. An already inferred value
used as a subject keeps its field types.

```text
fn round_to_even(guard: bool, sticky: bool, odd: bool) -> bool {
    match (guard, sticky) {
        (true, true) => true,
        (true, false) => odd,
        (false, _) => false,
    }
}
```

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
BX evaluates pure value expressions and calls to earlier local functions,
including value-parametric specializations, within its compile-time work limit.
Evaluation follows the dependencies of the result. For example, an
out-of-bounds access in an unused helper binding can
go unevaluated in BX even when DSLX reports a constexpr error. Calls to imported
DSLX functions and extern functions require runtime evaluation, including when
reached through a local helper.

Use an earlier `u32` constant as a bit width or array length in a concrete type.
Here `LANES` also bounds the traversal of the array.

```text
const WIDTH = u32:12;
const LANES = u32:4;
fn sum(xs: uN[WIDTH][LANES]) -> uN[WIDTH] {
    for (i, acc) in u32:0..LANES { acc + xs[i] }(uN[WIDTH]:0)
}
```

Widths can also appear in `bits[WIDTH]`, `sN[WIDTH]`, and
`xN[SIGNED][WIDTH]`. Use a `bool` or `u1` constant for `SIGNED`. For a
computed width or length, declare the expression as a module constant, then
use its name in the type. A public BX library constant can be qualified, as
in `u8[packet::LANES]`. Width slices and value-parametric functions also
accept dimension expressions.

## Module-level type aliases

Use `type` to give a concrete type another name. Mark an alias `pub` to export
it from a `.bx` library.

```text
// bus.bx
const LANES = u32:4;
type Lane = u8;
pub type Payload = Lane[LANES];
```

```text
// consumer.bx
use bus::Payload;
fn last(x: Payload) -> u8 { x[3] }
```

`Payload` and `u8[4]` are interchangeable. An alias for a struct or enum
preserves the original declaration's identity, including the arguments of a
concrete struct specialization. Two different structs with identical fields
remain distinct. The alias's dimensions and names resolve in the module that
declares it, so callers can use a public alias whose definition mentions
private declarations.

The same import syntax works for a public concrete alias in a `.x` library.
For example, a DSLX library can export `pub type Sample = (Hidden, bool);`
and BX can pass a `Sample` between imported functions while `Hidden` stays
private to the library.

Declare an alias before another alias or function that uses it. Under BX's
default declaration-order rules a block can use a later alias in a type
annotation. Enum member expressions such as `Mode::Idle` require `Mode` to be
in scope at the block declaration.

## Native parametric structs

A BX struct can declare compile-time value parameters of any concrete BX
value type, including booleans, signed or unsigned integers, enums, tuples,
arrays, and records. For example, a boolean can select the signedness of a
field. Constructing `Word` from an `s8` field infers `Word<true>`.

```text
struct Word<S: bool> { x: xN[S][8] }

fn is_negative(x: s8) -> bool {
    let word = Word { x };
    word.x < s8:0
}
```

Integer parameters can determine field widths, array lengths, and nested
struct arguments.

```text
struct Packet<N: u32, M: u32 = {N + u32:1}> {
    data: uN[N],
    tags: u8[M],
}

fn invert<N: u32, M: u32>(packet: Packet<N, M>) -> Packet<N, M> {
    Packet { data: !packet.data, tags: packet.tags }
}

fn make(data: u8, tags: u8[9]) -> Packet<8> {
    Packet { data: data, tags: tags }
}
```

Supply an explicit prefix of arguments in a type annotation or constructor.
Arguments can be literals, names, or braced compile-time expressions. An
untyped literal takes the parameter's declared type. Parameter types and
defaults can refer to earlier bindings. A constructor can infer omitted arguments
from its expected type or from the types of its field values. Every binding
contributes to the struct's type identity, including bindings that do not
affect its layout. Declare a template before referring to it in another type.
Declare the struct `pub` to use it from another BX module.

BX also accepts the turbofish spelling `Packet::<8>` in a type annotation
or constructor. Calls can use the same syntax, as in `invert::<8, 9>(packet)`.
Struct arguments that depend on an inferred block type parameter can appear
in the block body.

### Type parameters on structs

Declare a type parameter with `<T>` or `<T: type>`. A type application such as
`FlowOut<Packet>` supplies a type for each parameter. Type parameters and value
parametrics belong to separate struct declarations.

```text
struct FlowOut<T> { pop_valid: bool, pop_data: T }
struct Packet { tag: u8, words: u32[2] }

block flow_result<T>(clk, rst, valid: bool, data: T) -> FlowOut<T> {
    reg valid_q: bool = false;
    reg data_q: T = zero!<T>();
    valid_q := valid;
    if valid { data_q := data; }
    FlowOut { pop_valid: valid_q, pop_data: data_q }
}

pub block packet_stage(clk, rst, valid: bool, packet: Packet) -> FlowOut<Packet> {
    let stage = flow_result(valid, packet);
    stage
}
```

The call binds `T` to `Packet`. The result type supplies the type argument for
the `FlowOut` literal. A literal can also supply it explicitly, as in
`FlowOut::<T> { pop_valid: valid_q, pop_data: data_q }`. With no concrete
expected type, a constructor can infer its arguments from typed fields. Give
a numeric literal a type annotation when it supplies a type argument.

Each specialization exposes `pop_valid` and `pop_data` as named output ports.
Here `pop_data` is one port of type `Packet`. A caller can read its fields with
expressions such as `stage.pop_data.tag`.

### Struct updates

Use `..base` at the end of a struct literal to copy fields that the literal
doesn't specify. Place `..base` after any explicit fields, with no trailing
comma. The base expression is evaluated once and must have the same nominal
struct type as the literal. The base can supply omitted value parametrics or
infer a struct's type parameters when the entire argument list is omitted.
Explicit arguments must agree with its specialization.

When the literal supplies every field explicitly, BX emits
`useless_struct_splat` at `..`. To remove the splat, spell out any parametric
arguments that were inferred only from the base.

```text
fn invalidate<T>(previous: FlowOut<T>) -> FlowOut<T> {
    FlowOut { pop_valid: false, ..previous }
}
```

## Defining functions

Write `fn name(parameters) -> ResultType { ... }` or `pub fn` at module scope. Function signatures can include bits, structs, tuples, and arrays. In function type annotations, both `(u8)` and `(u8,)` mean a one-element tuple. In a value expression, write `(x,)` to construct that tuple.

Inside a function, `let` can bind a name or destructure a tuple, as in `let (data, valid) = packet;`. Braced expressions and `if` arms can introduce their own local bindings. The final expression supplies the result. A function that returns `()` may omit the result annotation and end with a semicolon or an empty body.

Give a bound integer literal its intended width before passing it to a function.
For a parameter of type `u8`, write `let n: u8 = 1; take(n)` or `take(u8:1)`.
With `let n = 1`, BX infers `u1`. The pinned DSLX typechecker can accept the
wider call, but its interpreter can retain the narrower value and IR conversion
can fail on the mismatched signature.

Define a helper before the functions that call it, as in DSLX. By default, a BX block can call a local function declared anywhere in the file. Calls in value expressions become XLS IR `invoke` nodes. Functions can also call imported DSLX functions and declared externs. Registers, register writes, and child instances belong in blocks. The checker rejects direct and indirect recursion.

### Module declaration order

Put `#![require_define_before_use]` before the first declaration to require
module names to be defined before use throughout a `.bx` file. This includes
block calls, explicit child instances, callback arguments, type annotations,
and `use` bindings. Each BX library chooses its own policy. BX resolves names
as usual, then checks the declaration order. For example, if a block's
`rev(x)` resolves to a later function named `rev`, the verifier reports that
function. A forward reference is an error even when warnings are allowed.

```text
#![require_define_before_use]

fn increment(x: u8) -> u8 { x + u8:1 }

pub block counter(clk, rst) -> u8 {
    reg q: u8 = u8:0;
    q := increment(q);
    q
}
```

Moving `increment` below `counter` makes the call an error. Inside a block,
a declared wire can still be read before its driver is connected.

For a register reset value or a loop range bound, use a constant expression that BX can evaluate during checking.

BX emits XLS IR for runtime array operations. For an out-of-range index, IR reads the last element of a nonempty array, and `update` returns the original array. The native DSLX interpreter reports an evaluation error for either operation. The differential tests check both outcomes.

### Compile-time function parameters

#### Type parameters

Pure functions use the same inferred type parameters as blocks. Both `fn id<T>(x: T) -> T { x }` and `fn id<T: type>(x: T) -> T { x }` specialize at `id(x)`. The call's data arguments bind types, including element types inside arrays and fields inside tuples. Repeated occurrences must agree on signedness and nominal record identity. `bool` and `u1` are equivalent. Give an integer a width when it is the only source for its type, as in `id(u8:1)`.

Each type parameter needs an inference source in a data argument or a callback result. BX resolves named callbacks in declaration order, using the input types known at that point. In the example below, `xs` and `ys` determine `A`, `B`, and the common array length `N`. BX checks that `join` accepts those element types and infers `R` from its return type.

```text
struct Left { x: u8 }
struct Right { y: u16 }
struct Joined { x: u8, y: u16 }

fn join(a: Left, b: Right) -> Joined { Joined { x: a.x, y: b.y } }

fn zip_with<A, B, R, F: fn(A, B) -> R, N: u32>(xs: A[N], ys: B[N]) -> R[N] {
    for (i, out_array) in u32:0..N {
        update(out_array, i, F(xs[i], ys[i]))
    }(zero!<R[N]>())
}

fn join_rows(xs: Left[3], ys: Right[3]) -> Joined[3] {
    zip_with<join>(xs, ys)
}
```

Type parameters take no call-site argument. Explicit value arguments and named callbacks retain their declaration order, so `zip_with<join, 3>(xs, ys)` also supplies the array length. BX emits a concrete IR function for each distinct combination of types, values, and callback identities. For a version of this helper that also accepts empty arrays, use `unroll_for!` in place of `for`. With zero iterations it returns the typed seed.

#### Value parametrics

Declare DSLX value parametrics between angle brackets. Each binding has a
bits, boolean, numeric enum, or struct type and is available as a constant in
the signature and body.

```text
fn resize<N: u32, M: u32 = {N + u32:1}>(x: uN[N]) -> uN[M] {
    x as uN[M]
}

fn widths(x: u8) -> (u9, u16) { (resize(x), resize<8, 16>(x)) }
```

In `resize(x)`, the input width binds `N` to 8 and the default binds `M` to 9.
BX can infer a directly named width or array length (`u32`) and signedness
(`bool`) from the data arguments, including tuple members and array elements.
For example, `xs: xN[S][N][L]` supplies the signedness, element width, and
array length. In a computed dimension such as `uN[N + u32:1]`, supply `N`
explicitly, give it a default, or infer it from another parameter.

Explicit arguments bind a prefix of the declared parameters in source order.
Use a bare name, number, boolean, or typed literal for a simple argument.
Enclose a computed argument in braces, as in `resize<8, {width + u32:4}>(x)`.
These expressions use the caller's scope and must evaluate at compile time.
Defaults use the function's declaration scope and can depend on earlier
bindings. A binding such as `K: uN[N] = {uN[N]:1}` can use an inferred width
when evaluating its default. Explicit actuals require a binding type that
can be resolved before instantiation, including widths from module constants.
The emitted IR function takes only its data parameters.

Nominal values retain their declared type. For a binding `P: Phase`, pass a
member such as `Phase::Ready`. For `C: Config`, pass a constant of that
struct type or a braced constructor such as `{Config { gain: u8:4 }}`.

#### Callable parameters

Declare callable parameters between angle brackets in the order callers will supply them. Give each callback its argument and result types. As with a local function declaration, omitting the result annotation means `()`. Inside the function body, invoke the parameter directly or pass a unary parameter as the callback in `map(xs, F)`.

```text
fn fold4<F: fn(u16, u16) -> u16>(xs: u16[4], seed: u16) -> u16 {
    for (x, acc) in xs { F(acc, x) }(seed)
}

fn add16(a: u16, b: u16) -> u16 { a + b }

fn sum(xs: u16[4]) -> u16 {
    fold4<add16>(xs, u16:0)
}
```

Each callable argument must directly name a visible non-parametric BX function, a public, pure DSLX function from a `use` declaration, or a declared `extern fn`. For example, `use arithmetic;` lets a caller write `fold4<arithmetic::add16>(xs, seed)`. BX uses the declared callback argument types to resolve an imported DSLX specialization and checks its result type. An extern supplies its BX signature during checking. When lowering into a prelinked package, BX verifies the IR signature and purity of the linked function and its callees. The callback name is resolved at the call site. Names written directly in the template body follow its declaration scope.

Callable parameters can share a declaration with value parametrics. For example,
`apply<N: u32, F: fn(uN[N]) -> uN[N]>` takes the width and the callback in
that order at `apply<8, inc>(x)`. Invoke the callback in the function body.
Defaults and type dimensions can call earlier pure BX functions, including
helpers from imported BX modules. Callable parameters cannot be assigned to a
`let`, returned, or forwarded to another template. Inline lambdas remain
available in `map` expressions. Callable parameters are a BX extension.

### Map expressions

Use `map(array, function_name)` or `map(array, |element| expression)` in a function or block to transform each element of a nonempty array in order. The callback takes one element, and its return type supplies the output element type. The output has the same length as the input.

```text
fn widen(x: u8) -> u16 { x as u16 }
fn widen_all(xs: u8[4]) -> u16[4] { map(xs, widen) }
```

A named callback can be a visible BX function, a public pure DSLX function imported with `use`, a signature-checked `extern fn`, or an enclosing compile-time function parameter such as `F`. Module-qualified names work directly, as in `map(xs, math::widen)`. BX can infer a callback's type and value parameters from the input element type, as in `map(xs, id)` for `fn id<T>(x: T) -> T { x }`. Supply explicit values on a BX or DSLX callback, as in `map(xs, resize<8, 16>)` or `map(xs, std::msbs<4>)`. Give literals their intended width when calling a concrete function, for example `map([u8:1, u8:3], widen)`.

The named builtin callbacks are `clz`, `ctz`, `and_reduce`, `or_reduce`, `xor_reduce`, `rev`, and `encode`. For example, `map(xs, xor_reduce)` computes the parity of each element.

```text
#[allow(combinational_block)]
block offsets(clk, base: u32) -> u32[4] {
    map(u32:0..u32:4, |i| base + i)
}
```

The lambda can read enclosing values, including function arguments, earlier
`let` bindings, register values, and child-block outputs. In `offsets`, every
element uses the current value of `base`. A nested lambda can also capture
the outer lambda's element. Parameters and local lets shadow enclosing names.
The callback is pure and can call visible pure helpers.

Captured values keep the width they had at the binding. For example, use
`let bias = u8:3` when adding a constant bias to each element of a `u8`
array. For compatibility with the pinned DSLX typechecker, annotate a
capturing callback that returns a tuple or array literal, as in
`map(xs, |x| -> (u8, u8) { (x, bias) })`.

Write `|x: u8| -> u16 { ... }` to give explicit types. Give input literals their intended width before mapping. For example, `[1, 3]` supplies two-bit elements, so `map([1, 3], |x| x + 1)` yields `[2, 0]`. For eight-bit arithmetic, write `map([u8:1, u8:3], |x| x + 1)`. A result annotation is useful for a body that returns an unsized literal, as in `|_| -> u16 { 0 }`. A constant integer range is also a valid input, for example `map(u8:0..=u8:3, |x| x * x)`.

BX emits an XLS IR `map` when the callback needs no enclosing wires. A named
callback maps directly to its linked function. For a
capturing callback, BX evaluates the first element directly and lowers the
remaining elements to a `counted_for` with the captures as invariant
arguments. Structural unrolling and generic block specialization
charge each copy of the callback to the elaboration work limit. Optimize the
IR to inline maps and unroll counted loops before emitting SystemVerilog.

```shell
xlsynth-driver bx2ir offsets.bx > offsets.ir
opt_main offsets.ir > offsets.opt.ir
xlsynth-driver block2sv offsets.opt.ir > offsets.sv
```

### Counted for expressions

Use a DSLX-style `for` in a function or block to fold an array or a constant integer range into a value. The expression after the body supplies the initial accumulator. Each iteration binds the next element and the previous accumulator, then returns the next accumulator.

```text
fn sum_words(words: u16[4], bias: u32) -> u32 {
    for (word, sum): (u16, u32) in words {
        sum + (word as u32)
    }(bias)
}

#[allow(combinational_block)]
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

## Unit tests

Put a `#[test] fn` beside the function it exercises. A test takes no
arguments. Its body can bind values with `let`, iterate over a table with
`for`, and check values with `assert_eq(actual, expected)` or
`assert!(condition)`.

```text
fn twice(x: u8) -> u8 { x + x }

#[test]
fn twice_wraps() {
    assert_eq(twice(u8:129), u8:2);
    assert!(twice(u8:3) == u8:6);
}
```

A `#[test] block` can drive data inputs, advance the clock, and check
outputs. Declare stimulus registers and wire child blocks with `let` at the
start of the body. These bindings form the persistent circuit. A `reg` in
the test holds stimulus. Assign it with `:=` before the edge. A `let` call
to the block under test creates one instance, so reading its output after
`tick!()` sees the state captured on that edge. Once the first drive,
tick, assertion, loop, or conditional executes, each subsequent `let`
samples its value at that statement. For example, `let before = total;`
at that point saves the current total for a comparison after another tick.
`tick!()` advances one edge with reset deasserted. `tick!(rst: true)`
asserts reset for that edge and releases it afterward. The optional `rst:`
clause belongs to `tick!`, much like `target:` in Rust’s
[`log::info!(target: "sim", "tick")`](https://docs.rs/log/latest/log/macro.info.html).

A stimulus declaration without an initializer starts at the typed zero.
DUT registers retain their usual power-up behavior. Sampling a value that
contains unknown bits fails the test.

This counter uses a table of `(Drive, u8)` pairs. The middle row disables
the update and checks that the counter holds the previous total.

```text
struct Drive { increment: u8, enable: bool }

block counter(clk, rst, drive: Drive) -> u8 {
    reg total: u8 = u8:0;
    if drive.enable { total := total + drive.increment; }
    total
}

const CASES: (Drive, u8)[3] = [
    (Drive { increment: u8:1, enable: true }, u8:1),
    (Drive { increment: u8:4, enable: false }, u8:1),
    (Drive { increment: u8:2, enable: true }, u8:3),
];

#[test]
block counter_steps() {
    reg drive: Drive;
    let total = counter(drive);
    tick!(rst: true);
    assert_eq(total, u8:0);
    for (to_drive, to_expect) in CASES {
        drive := to_drive;
        tick!();
        assert_eq(total, to_expect);
    }
}
```

Run `xlsynth-driver bx-test counter.bx` to execute the tests declared in
`counter.bx`. Each test runs independently. Changing the expected total in
the second row to `6` makes `counter_steps` fail at that assertion. The
diagnostic gives its source location, clock edge, and compared values.

Clocked `#[test]` blocks use synchronous reset and the same SystemVerilog lowering as
`block2sv`. The runner executes procedural `for` statements in a test on
the host. Value-producing expressions in the DUT or test circuit must lower
to supported Block IR. Raw `counted_for` and `map` IR nodes require expansion
before SystemVerilog code generation.

### Clocked QuickCheck properties

A private `#[quickcheck] block` takes typed stimulus parameters and ends with a
`bool`. Each trial draws its parameters once. Arrays of records can describe
a sequence of inputs. Using the counter above, this property checks its
load enable and accumulated value after each edge.

```text
#[quickcheck]
block counter_matches(trace: Drive[4]) -> bool {
    reg drive: Drive;
    reg expected: u8;
    let total = counter(drive);
    tick!(rst: true);
    for row in trace {
        drive := row;
        if row.enable { expected := expected + row.increment; }
        tick!();
        assert_eq(total, expected);
    }
    total == expected
}
```

```shell
xlsynth-driver bx-test counter.bx --seed 42 --quickcheck-trials 100 --cross-check-vastly
```

The PIR interpreter starts each trial with independently drawn bits in every
DUT register, including registers with a synchronous reset value. A reset edge
loads that value. Testbench stimulus registers use their declared initializers
or typed zero. Enum parameters draw from their declared members. PIR checks
linked hardware assertions on rising edges with the flattened block's reset
deasserted. Procedural BX assertions sample at their source statement.
A failed concrete trial reports its seed, trial number, parameters, initial
register values, and stimulus trace. If a clock edge fails, the trace includes
the attempted tick and its reset setting.

`--cross-check-vastly` first compares every sampled PIR value with generated
RTL running from the same concrete register values. It also runs the RTL from
all-X DUT register state with the same stimulus parameters. This second run
reports true, false, or indeterminate. An indeterminate trial identifies the
first procedural sample containing X or Z, including a snapshot or a branch
condition. The four-valued counts are reported separately from the concrete
property result. A failed concrete property, backend error, or concrete
mismatch makes `bx-test` exit unsuccessfully. For the optional RTL run,
maps and counted loops are expanded with a limit of 65,536 additional IR
nodes per property harness, including nested body copies. Use the PIR backend
for linked hardware that emits SystemVerilog `assert property` statements.
Vastly currently cannot parse this syntax.

### Bounded proofs for clocked properties

```shell
xlsynth-driver bx-prove-quickcheck counter.bx
```

This proves `counter_matches` for every value of its four `Drive` records
and every initial bit pattern in the DUT registers. The reset edge loads
the counter's reset value. On each of the four subsequent edges the solver
checks the assertion against the reference, including when the enable is
false. It also checks the final boolean. Testbench registers start at their
declared initializer or typed zero. Enum-valued stimulus ranges over the
declared members, including inside aggregates.

The proof checks the tick schedule syntactically. A tick can appear at the top
level or in fixed-length loops. A loop over a stimulus array qualifies because
its type fixes the number of iterations. A procedural `if` can choose values,
update testbench registers, or guard an assertion. Place every tick outside
such conditionals, including those with literal conditions. The reset argument
in `tick!(rst: drive.reset)` can itself be symbolic.

A property that exceeds the static expansion limit is reported as inconclusive.
The command continues checking the other properties in the file.
Reachable hardware assertions in linked IR, array operations marked
`assumed_in_bounds=true`, and partial-product multiplies (`umulp` and `smulp`)
are currently unsupported. A solver counterexample is replayed through the
concrete BX runner to report the failing observation, clock edge, and stimulus
trace. The proof covers the edges executed by the property.

### Clocked invariants and helper lemmas

A private `#[invariant]` block defines a Boolean predicate. Its trailing
`bool` is the observation for the current cycle. Declaring it a **clocked
invariant** claims that the predicate stays true forever, on every sampled
cycle of every legal execution. One reachable cycle where the block returns
`false` disproves the invariant.

BX creates the block's registers and child instances once. The runner
supplies the clock edges. On each cycle it drives fresh inputs, samples the
predicate and any lemmas against the current register values, then clocks
the `:=` updates together. A register read in the trailing expression sees
the current value even if an update appears earlier in the source.

All instances in the harness share one clock. To model different logical
rates, give the slower machine a clock enable. A register updated with
`if enable { q := next; }` holds its value on disabled edges. The invariant
block still produces an observation every cycle. For a property that checks
data only when valid, its trailing expression can be `!valid || data == expected`.

Use an ordinary register update to model a reference implementation alongside
the DUT. For example, these two parity machines have different state widths.

```text
#[invariant]
block parity_equivalent(clk, rst, toggle: bool) -> bool {
    reg lhs: u2 = u2:0;
    reg rhs: bool = false;
    #[lemma]
    let lhs_is_live = lhs < u2:2;

    if toggle {
        if lhs_is_live { lhs := lhs ^ u2:1; }
        rhs := !rhs;
    }
    (lhs == u2:1) == rhs
}
```

For a block declaring `rst`, the first sample follows one asserted synchronous
reset edge. The state before that edge is arbitrary. The checker executes the
reset edge with legal data inputs, so a register without a reset value follows its ordinary update or
retains its arbitrary value. Subsequent sampled edges may assert reset again.
A clock-only invariant starts sampling directly from arbitrary register state.

Mark a top-level Boolean binding with `#[lemma]` to give the prover a helper
invariant. The binding remains an ordinary value in the circuit. Here a dead
left state could appear to agree with the right machine for arbitrarily many
idle cycles. Proving `lhs_is_live` excludes that state from the induction step.
The engine checks that all helpers hold from reset and proves their conjunction
is preserved before using it in the proof of the trailing Boolean.

```shell
xlsynth-driver bx-test parity.bx --seed 42 --invariant-cycles 16
xlsynth-driver bx-prove-invariant parity.bx --mode=bmc --max-depth 16
xlsynth-driver bx-prove-invariant parity.bx --engine=bitwuzla --max-depth 8
```

Concrete trials and bounded proofs check the helpers and the trailing Boolean
on every sampled cycle. A reachable failure identifies the failing predicate
and the stimulus trace. An induction step that remains open at the depth limit
is inconclusive. The word-level proof has the same linked-IR operation limits
as bounded clocked QuickChecks.

For runnable observational equivalence proofs, see the widened accumulator,
binary versus one-hot detector, and parity machine with an unreachable state.
Each harness imports the two DSLX step functions and compares their outputs
on a shared input stream. [Observational equivalence case studies](../xlsynth-driver/examples/bx-observational-equivalence/README.md)

## Compile and link

Use `parse_module` to inspect BX syntax without reading imports. For a source file with library imports, `parse_module_with_options` uses the exported signatures to distinguish multi-callback template calls from comma-separated comparisons. Pass the parsed module and the same search paths to `check_module_with_options` to inspect its checked declarations. The checker resolves BX and DSLX module imports and accepts signatures for prelinked externs. Pass the checked result to `lower_module`, or use `compile` for all three stages. Supply a PIR package containing your externs, or `empty_package("design")` when all calls use local definitions or imports.

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

### BX warnings

`compile`, `compile_with_options`, `bx-test`, `bx2ir`, and `bx2gv` treat BX warnings as
errors by default. The `combinational_block` warning points to a block that
only computes a value. Prefer a `fn` for this kind of helper. A block with a
register or a child instance already serves a hardware role. A helper that
only reads its implicit reset can take the normalized reset as an explicit
`bool` function argument. Type-generic blocks can express signatures
unavailable on BX functions.

For an intentional combinational RTL interface, allow the warning on that
declaration. Other declarations in the same file or an imported BX library
still receive their own diagnostics.

```text
#[allow(combinational_block)]
pub block increment_port(clk, x: u8) -> u8 { x + u8:1 }
```

Use `#[allow(useless_struct_splat)]` on the declaration containing the update.
This applies to blocks, functions, constants, enums, structs, and extern
signatures, including expressions in parametric defaults and type dimensions.

To inspect all warnings and still emit output, pass `--warnings-as-errors=false`
to the driver command. Rust callers can use `check_source_with_options`
and inspect `CheckedModule::warnings()` before calling `lower_module`. Each
warning identifies its source declaration and location. Repeated calls,
specializations, and imports share one diagnostic for the same warning site.

### Using BX libraries

Place reusable declarations in a `.bx` file and mark the library interface with
`pub`. This `stages.bx` library defines a register for any payload type.

```text
pub block pipe_reg<T>(clk, rst, valid: bool, data: T) -> (bool, T) {
    reg valid_q: bool = false;
    reg data_q: T;
    valid_q := valid;
    if valid { data_q := data; }
    (valid_q, data_q)
}
```

A sibling file can import the module and call its block. The call inherits
the caller's clock and reset, and BX infers `T` from `data`.

```text
use stages;

pub block delay(clk, rst, valid: bool, data: u16) -> (bool, u16) {
    let delayed = stages::pipe_reg(valid, data);
    delayed
}
```

`use stages::pipe_reg;` binds the block directly. Module, member, and grouped
imports work for public blocks, functions, structs, numeric enums, and constants. Library
declarations can use private helpers and import other BX or DSLX modules.
Transitive imports stay in the file that declares them. The root file's blocks
determine the implicit top. BX links library blocks, functions, and externs reached
from the root declarations.

Module paths start at a search root. For example, `use ip::stages;` looks for
`ip/stages.x` or `ip/stages.bx` and binds `stages`. An existing DSLX module
takes precedence over a BX file with the same logical path. The driver also
searches the input file's directory. Use `--dslx-path` to add a library
directory. Library API callers supply directories in
`ImportOptions::additional_search_paths`.

### Using DSLX modules and members

`use arithmetic;` resolves `arithmetic.x` on DSLX's module search path. For a nested module, `use ip::arithmetic;` makes its public functions, structs, and numeric enums available as `arithmetic::name`. To bind individual members, write `use ip::arithmetic::clamp;` or group several under the same path, as in `use ip::arithmetic::{clamp, Limits};`. Each leaf binds its final name in the BX module. To migrate an aliased import, use the module’s final name or select the member directly. If two modules export the same member name, qualify it with distinct module bindings. If the module names also collide, bind one module and the other member directly. For example, `use a::math; use b::math::clamp;` gives you `math::clamp(...)` and `clamp(...)`. BX checks every `use` declaration, including unused ones.

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

An imported DSLX function accepts an explicit prefix of value parametrics.
For example, `std::lsbs<R, S, N>(x)` takes the result width first. In this
BX function, DSLX infers signedness and input width from `x` and from each
element of `xs`:

```text
use std;

fn nibbles(x: u8, xs: u8[2]) -> (u4, u4[2]) {
    (std::lsbs<4>(x), map(xs, std::msbs<4>))
}
```

A member imported with `use std::lsbs;` also accepts `lsbs<4>(x)`. BX
evaluates each supplied expression in caller scope with the declared DSLX
binding type. Omitted bindings use native DSLX inference and defaults. The selected
signature supplies the context for unsized data literals.

BX can name public, non-parametric DSLX structs in ports, registers, and expressions. For example, given `pub struct Header { tag: u8, valid: bool }` in `packet.x`:

```text
use packet::Header;

fn tag_header(h: Header) -> Header {
    Header { tag: h.tag + u8:1, valid: h.valid }
}
```

BX can read fields through a private struct nested inside a public one. To pass the nested value itself as a DSLX function argument, expose its type through a public name or type alias. A public DSLX type alias preserves the identity of its underlying struct. Imported calls accept these structs as arguments and results, alongside bits, booleans, tuples, and fixed-size arrays. Functions must be public and pure. Linked functions and their module path components use letters, digits, and underscores so their generated names are valid XLS IR identifiers.

BX accepts concrete instances of public DSLX struct templates with `u32` value
parameters. Write the application directly in a type annotation or struct
literal.
DSLX supplies defaults for omitted trailing parameters, including
all parameters in a spelling such as `Packet<>` when each has a default.

```text
use apfloat;

fn add_half(a: apfloat::APFloat<5, 10>, b: apfloat::APFloat<5, 10>)
    -> apfloat::APFloat<5, 10> {
    apfloat::add(a, b)
}
```

With `use apfloat::APFloat;`, the same type is `APFloat<5, 10>`, and a
constructor can use `APFloat<5, 10> { sign: false, bexp: u5:15, fraction: u10:0 }`.
A public DSLX alias and its direct application are interchangeable, such as
`float32::F32` and `apfloat::APFloat<8, 23>`. An imported function can
also return a new instance by naming the public template or a concrete alias
in its DSLX result type. BX recovers the instance's `u32` bindings from the
checked value, including parameters that do not appear in any field.

The pinned DSLX converter can emit invalid IR or dispatch to the wrong helper
when a type-valued parameter such as `T: type` is bound to a struct. When a
BX call passes or returns a concrete parametric struct, its imported DSLX
functions can use bits and arrays of bits as type-valued arguments. BX
diagnoses unsupported aggregate bindings, ambiguous
native symbols, and method dispatches whose specialization the converter API
cannot verify. DSLX `map` callbacks and indirect calls are outside this subset.
Functions such as `apfloat::add` specialize on ordinary value parameters.

For imports that enable DSLX generics, BX checks the native symbols of the
selected specializations. Distinct numeric enums can specialize the same
helper even when their source names match. The pinned mangler can give nested
tuples such as `(E, (u8, u8), u8)` and `(E, (u8, u8, u8))` the same symbol.
BX diagnoses that collision when both specializations occur in one conversion
batch. When the batch can specialize a numeric enum, BX also reports an
import error if the native call graph omits a method or indirect call target
inside a specialized function and BX cannot verify its symbols.

If an imported function requires an implicit token, BX reports the DSLX call path and highlights the effectful operation when it can locate it in the original source. For example, a runtime `assert!(N >= u32:2, "min_width")` in a helper makes its caller effectful. When `N` is a width parametric, `const_assert!(N >= u32:2)` checks the invariant during DSLX typechecking and keeps the helper pure.

BX can follow a direct helper call in the body of a counted loop over a nonempty literal range, such as `for (_, acc) in u32:0..u32:1 { leaf::checked(acc) }(x)`. If the body branches between helpers, the range might be empty, or the initializer could introduce an earlier effect, BX reports the call path it can verify.

Lowering emits an ordinary IR `invoke` and includes the specialized function and its callees. Some functions, including `std::popcount`, contain a loop in their original IR. Consumers such as `xlsynth-codegen` that require unrolled loops need an optimization step for those bodies before RTL emission.

## Clocked blocks

- The first port is `clk`, followed by optional `rst` and typed data inputs. Pipelines conventionally put `valid_in` first among the data inputs. The final expression drives the outputs continuously. A struct result exposes its fields as output ports in declaration order, so those field names must be valid XLS IR port names. A tuple exposes `out_0`, `out_1`, and so on. Unit exposes zero outputs, and a scalar or array exposes one port named `out`. Nest a tuple or struct inside a named field to expose it as one aggregate port.
- A `reg` name in an expression reads its current value. Each register has exactly one `:=` update site. A surrounding `if` supplies the load enable. An ordinary value expression such as `q := if en { x } else { q };` supplies a mux on the data input. All registers update together at the active edge.
- A register initializer is a constant reset value and requires `rst`. Reset takes priority over load enable for that register. Registers declared without an initializer have unspecified power-up values. In a value expression, `rst` means reset is asserted. `ResetBehavior` configures polarity and timing for the hierarchy, defaulting to active-high synchronous reset.
- A module with one concrete block declaration selects it as top. With multiple concrete declarations, a single concrete `pub block` selects the top. For other configurations, select the desired block on the returned package.

### Pipeline stage annotations

An output annotation such as `-> u8 @1` opts a block into stage checking. Data
inputs default to `@0`. An input annotation such as `r: u8 @1` says the caller
already supplies that input at stage 1.

Put `#![require_block_timing]` at the top of a `.bx` file to require an output
stage on every block defined there. A missing annotation emits
`missing_block_timing`, which fails compilation by default. Imported files set
their own policy. To exempt one block, put `#[allow(missing_block_timing)]`
above its definition.

```text
#![require_block_timing]

block add_held(clk, en: bool @0, x: u8 @0, r: u8 @1) -> u8 @1 {
    reg q: u8;
    if en { q := x; }
    q + r
}
```

The checker follows the output back to its inputs. A `let` adds zero stages,
combinational operands must align, and a register adds one stage. Each write
guard must align with the data at the write edge, including enclosing guards
in nested `if` statements. In the example, `q` and `r` are both at stage 1.
When `en` is low, `q` holds its previous value and the sum still responds to
`r`. A stage annotation checks alignment in the datapath. The block's enable
and valid signals determine which transactions actually advance.

The checker follows names, unary and binary operators, casts, `if`
expressions, and struct and tuple construction and field reads. A scalar
literal such as `u8:7` can be used at any stage. With `const A = [u8:7];`,
`A` can also be used at any stage because BX evaluates the constant before
checking timing. Returning `[u8:7]` directly leaves an array expression whose
stage BX cannot infer yet. Array indexing has the same limitation. For
`x: u8[2]`, returning `x[u1:0]` gives `cannot infer output cycle for this expression` even though the index is constant.

A uniform result such as `-> Ports @1` checks every output port at stage 1.
For different stages, use `-> Ports @(live: 0, saved: 1)` and list every
output port. A tuple's port names are `out_0`, `out_1`, and so on. A field
read such as `p.saved` follows that field's driver. Each aggregate input
supplies a single stage shared by its fields.

A whole registered struct or tuple is checked field by field. For example,
with `x: u8 @0` and `q := Payload { data: x, tag: u8:7 };`, `q.data` is
`@1`. The literal supplies no stage for `q.tag`, so forwarding all of `q`
cannot establish `@1`. If an `@0` enable guards that write, it supplies the
stage for capturing the tag and both fields are `@1`.

Each child instance uses one offset to translate its stages into the parent's
stages. For a child with `x @0`, `r @1`, and an output at `@1`, connecting
`x` at parent stage 4 sets the offset to 4. Connecting `r` at stage 5 then
passes and the output is at stage 5. Connecting `r` at stage 4 fails. A
second instance of the same child can start at a different stage.

A constant argument fits any stage. At least one argument with a known stage
must set the offset, and the checker must be able to follow the other
arguments. A child without an output annotation supplies no stage to its
caller, even if its inputs are annotated. An extern block also supplies no stage.
Register feedback such as `q := q + x;` is legal BX. Following a timed
output back through that feedback currently produces a diagnostic.

#### Related work

Dahlia and Filament showed how hardware timing constraints can be checked
at compile time. Dahlia catches resource conflicts such as two parallel
reads that need the same memory port. Filament describes when a port's value
is available relative to a component invocation, so it can catch a caller
that reads a result one cycle too early.
[Dahlia (PLDI 2020)](https://arxiv.org/abs/2004.04852) ·
[Filament (PLDI 2023)](https://arxiv.org/abs/2304.10646)

SUS's latency counting is a close precedent for BX's stage labels. SUS
assigns a latency to each wire and can insert registers to align paths that
meet at an operator. BX checks the registers the author wrote. In
`add_held` above, declaring `r` at `@0` puts the two inputs to `q + r` at
stages 1 and 0, and BX reports the mismatch at the adder.
[Latency Counting in the SUS Language (LATTE 2024)](https://capra.cs.cornell.edu/latte24/paper/4.pdf)

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

Use the same call syntax for a BX block. By default, the child may be declared later in the module. An `extern block` declaration supplies the signature of an existing block in the linked IR package. BX checks data-port names, directions and types. It identifies the physical clock and reset by their roles and checks that the reset policy matches. Inside the parent, the result has the child's declared type, so `left.valid` selects one output.

For feedback wiring, `inst cell: delay;` gives the child a name before its inputs are connected. For example, `cell.d := cell.out + step;` is valid when `cell.out` reads a register. Connect each data input once. Both forms lower to Block IR instantiations, and a same-cycle combinational loop is a lowering error.

### Deferred wire bindings

Declare a wire directly in a block body with `let name: Type;`. You can read
it after the declaration, including before its driver and when inferring a
generic child block's type parameters. Supply exactly one unconditional
driver with `name = value;`. The driver may be an `if` expression.

```text
block delay<T>(clk, rst, data: T) -> T {
    reg q: T = zero!<T>();
    q := data;
    q
}

pub block accumulate(clk, rst, step: u8) -> u8 {
    let next: u8;
    let current = delay(next);
    next = current + step;
    current
}
```

The register in `delay` breaks the feedback path across cycles. BX checks
same-cycle dependencies at Block IR node and child port granularity and
reports combinational cycles at their participating source connections.

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

### Value parametrics on blocks

A block can take required `u32` value parametrics for compile-time expressions
in its body. Supply each value explicitly in angle brackets at the call site.
BX infers type parameters from the data inputs.

```text
block stage<T>(clk, rst, data: T) -> T {
    reg q: T = zero!<T>();
    q := data;
    q
}

block delay<T, N: u32>(clk, rst, data: T) -> T {
    unroll_for! (_, prev): (u32, T) in u32:0..N {
        let next = stage(prev);
        next
    }(data)
}

pub block pipeline(clk, rst, data: u8) -> u8 {
    let delayed = delay<3>(data);
    delayed
}
```

Value arguments can be literals, constant names, or braced expressions such as
`delay<{u32:1 + u32:2}>(data)`. If a block declares both value and callable
parameters, supply their arguments in declaration order. Calls share a block
definition when the concrete types, values, and callback identities all agree.

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

The same syntax accepts local BX functions and signature-checked externs. For example, `extern fn widen(x: u8) -> u16 = "wide_ir";` lets a caller write `advance<widen>(packet)` when lowering into a package containing `wide_ir`. A block body can invoke a callable parameter directly or use it as a `map` callback.

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
