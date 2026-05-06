# Ahead-of-time (AOT) compilation

DSLX or XLS IR is compiled into object code at build time, and the resulting
object code is wrapped in generated Rust code to enable easy integration into
user code.

Advantages compared to the DSLX interpreter or the XLS JIT:

- Avoids compilation overhead at runtime.
- Exposed interface uses native Rust types (e.g. `u64`, `[u8]`, structs,
  arrays).
- High-performance interface avoids per-call allocation/deallocation, with
  minimal copying of arguments and outputs.

## AOT Design

The `xlsynth` AOT framework consists of the following pieces:

- The libxls library exposes a C API for compiling XLS IR into object code and
  metadata about how to invoke the code (buffer sizes, alignments, etc.).
- The XLS C API is wrapped in Rust code for convenience and safety:
  [`xlsynth/src/aot_lib.rs`](../src/aot_lib.rs).
- Builder code ([`xlsynth/src/aot_builder.rs`](../src/aot_builder.rs)) generates
  standalone Rust wrappers around artifact-scoped XLS object-code entrypoints.
  The generated wrappers link the reusable
  [`xlsynth-aot-runtime`](../../xlsynth-aot-runtime/) crate for buffer
  allocation, argument packing, and entrypoint invocation, so consumers that
  only execute precompiled artifacts do not need to link `xlsynth` or `libxls`
  at runtime.
- Runner code ([`xlsynth/src/aot_runner.rs`](../src/aot_runner.rs)) remains
  available as the direct library-runtime API: callers can manually construct
  an `AotEntrypointDescriptor` and pass it to `AotRunner`, while generated
  wrappers use the standalone runtime path instead.

## Using AOT compilation

- Invoke the builder API from the `build.rs` file of the project to AOT-compile
  the XLS code and create the generated Rust code at build time.
- Add `xlsynth-aot-runtime` as a normal dependency of the crate that includes
  the generated wrapper source.
- Add `mod`s in the project code (e.g. `lib.rs`) that pull in the generated Rust
  code via `include!`.
- Call the exported API to create the AOT object and invoke its `run` method.

For an example, see [`xlsynth/tests/aot-test-crate`](../tests/aot-test-crate/).

## Thread safety

The underlying object code is thread-safe, but each generated `Runner` object is
not. The runner owns internal buffers used for each invocation of the
AOT-compiled function. Each thread should create its own runner by calling
`new_runner`.

## Generated code

When AOT-compiled, the generated Rust wrapper module exposes a small typed
interface for calling the compiled code. The generated code reexports the
public runtime value types from `xlsynth-aot-runtime`, adds artifact-specific
layout and bridge logic, and exposes a runner with one constructor plus two
methods for running the compiled code:

- `run(...)`: Runs the compiled function and returns the output value. Raises an
  error if an assert failed.
- `run_with_events(...)`: Runs the compiled function and returns assertion
  messages along with the output value. Does not raise an error if an assert
  failed. The caller can instead check `assert_messages`.

Standalone generated artifacts preserve runtime `assert` support. They reject IR
containing runtime `trace` or `cover` nodes at generation time; those runtime
features are intentionally not part of the first standalone runtime artifact
contract.

Types are defined for each function input (`Input0`, `Input1`, etc) and the
function output (`Output`). Bits types of 64 bits or less are defined as aliases
of the smallest unsigned type which can hold the value (`bool`, `u8`, `u16`,
`u32`, or `u64`). Wider types are aliases of arrays of `u8` (e.g., `[u8; 16]`).
Tuples and arrays are defined as `structs` and arrays respectively.

As a simple example, consider a DSLX function that multiplies two 8-bit unsigned
values and returns a 16-bit unsigned product:

```dslx
pub fn mul8x8(a: u8, b: u8) -> u16 {
  (a as u16) * (b as u16)
}
```

The generated Rust code looks like:

```rust
// Type aliases generated from the XLS signature.
pub type Bits8 = u8;
pub type Bits16 = u16;

pub type Input0 = Bits8; // `a: bits[8]`
pub type Input1 = Bits8; // `b: bits[8]`
pub type Output = Bits16; // return: bits[16]

pub struct Runner {
    // owns standalone ABI buffers internally
}

impl Runner {
    pub fn new() -> Result<Self, AotError>;

    pub fn run(&mut self, a: &Input0, b: &Input1) -> Result<Output, AotError>;

    pub fn run_with_events(
        &mut self,
        a: &Input0,
        b: &Input1,
    ) -> Result<AotRunResult<Output>, AotError>;
}

pub fn new_runner() -> Result<Runner, AotError>;
```

Typical call pattern:

```rust
let mut runner = mul8x8_aot::new_runner()?;
let out = runner.run(&3, &5)?;
assert_eq!(out, 15);
```
