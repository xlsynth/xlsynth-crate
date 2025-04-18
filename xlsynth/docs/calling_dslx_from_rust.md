# Calling DSLX from Rust

When developing DSLX code code, we often want to be able to:

- easily invoke the DSLX functionality, or
- interoperate with DSLX types

from Rust project code-bases. Additionally, we want to be able to
interoperate with SystemVerilog code bases with "Single Source of Truth" type definitions.

To help enable this interop, the `xlsynth` crate offers "DSLX bridge" facilities.

## Building the bridge

In your `build.rs`, create Rust code that corresponds to your `.x` file, and export the path that was created through an environment variable so that it can be used in your crate:

```rust
//! build.rs

fn main() {
    let rs_path = xlsynth::x_path_to_rs_bridge_via_env("src/my_dslx_module.x");
    println!("cargo:rustc-env=DSLX_MY_DSLX_MODULE={}", rs_path.display());
}
```

## Including in your crate

Then, within the crate, textually include that code into your project:

```rust
//! src/lib.rs

include! {env!("DSLX_MY_DSLX_MODULE")} // generated by build.rs
```

This will expose Rust types that correspond to the DSLX types from your module. You can observe the generated file in the `build` directory:

```
$ find -name 'my_dslx_module.rs'
./target/debug/build/sample-usage-3cf482a1485c950f/out/my_dslx_module.rs
```
