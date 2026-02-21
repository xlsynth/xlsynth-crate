# Ahead-of-time (AOT) compilation

DSLX or XLS IR is compiled into object code at build time, and the resulting object code is wrapped in generated Rust code to enable easy integration into user code.

Advantages compared to the DSLX interpreter or the XLS JIT:

- Avoids compilation overhead at runtime.
- Exposed interface uses native Rust types (e.g. `u64`, `[u8]`, structs, arrays).
- High-performance interface avoids per-call allocation/deallocation, with minimal copying of arguments and outputs.

## AOT Design

The `xlsynth` AOT framework consists of the following pieces:

- The libxls library exposes a C API for compiling XLS IR into object code and metadata about how to invoke the code (buffer sizes, alignments, etc.).
- The XLS C API is wrapped in Rust code for convenience and safety: [`xlsynth/src/aot_lib.rs`](../src/aot_lib.rs).
- Builder code ([`xlsynth/src/aot_builder.rs`](../src/aot_builder.rs)) generates Rust code that provides a native, high-performance interface to the underlying XLS object code. The generated code is encapsulated in an object that allocates the necessary buffers at creation time and exposes a simple entry point for calling the function. The generated code includes native Rust types for the interface.
- Runner code ([`xlsynth/src/aot_runner.rs`](../src/aot_runner.rs)) is a library used by the generated code for tasks like allocating buffers and packing/unpacking arguments.

## Using AOT compilation

- Invoke the builder API from the `build.rs` file of the project to AOT-compile the XLS code and create the generated Rust code at build time.
- Add `mod`s in the project code (e.g. `lib.rs`) that pull in the generated Rust code via `include!`.
- Call the exported API to create the AOT object and invoke its `run` method.

For an example, see [`xlsynth/tests/aot-test-crate`](../tests/aot-test-crate/).

## Generated code

As a simple example, consider a DSLX function that multiplies two 8-bit unsigned values and returns a 16-bit unsigned product:

```dslx
pub fn mul8x8(a: u8, b: u8) -> u16 {
  (a as u16) * (b as u16)
}
```

When AOT-compiled, the generated Rust wrapper module exposes a small typed interface for calling the compiled code. The exact contents vary based on the signature, but the caller-facing pieces look like:

```rust
// Type aliases generated from the XLS signature.
// Note: for bit widths <= 64, the generated wrapper currently uses `u64`.
pub type Bits8 = u64;
pub type Bits16 = u64;

pub type Input0 = Bits8; // `a: bits[8]`
pub type Input1 = Bits8; // `b: bits[8]`
pub type Output = Bits16; // return: bits[16]

pub struct Runner {
    // wraps `xlsynth::AotRunner` internally
}

impl Runner {
    pub fn new() -> Result<Self, xlsynth::XlsynthError>;

    pub fn run(&mut self, a: &Input0, b: &Input1) -> Result<Output, xlsynth::XlsynthError>;

    pub fn run_with_events(
        &mut self,
        a: &Input0,
        b: &Input1,
    ) -> Result<xlsynth::AotRunResult<Output>, xlsynth::XlsynthError>;
}

pub fn new_runner() -> Result<Runner, xlsynth::XlsynthError>;
```

Typical call pattern:

```rust
let mut runner = mul8x8_aot::new_runner()?;
let out = runner.run(&3, &5)?;
assert_eq!(out, 15);
```
