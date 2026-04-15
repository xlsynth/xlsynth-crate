# Mainband NoC Route DSLX-to-Rust Bridge Prototype

This note captures the prototype work for emitting Rust types for the DSLX boundary of
`mainband_noc_route`.

## Summary

The goal was to make the four argument types to this DSLX function become Rust types via the
existing `rust_bridge_builder.rs` and `dslx_bridge.rs` machinery:

```dslx
pub fn mainband_noc_route(
    cfg: MainbandRouteCfg,
    cmd: NocCmd,
    device_to_rni: DeviceToRniEntry,
    route_lfsr: RouteLfsrSamples,
) -> RouteResult
```

The prototype does this by extending the bridge walker so builders can observe DSLX function
signatures, then teaching `RustBridgeBuilder` to emit aliases for one selected function boundary.
For `mainband_noc_route`, the emitted aliases are:

```rust
// Rust aliases for DSLX function `mainband_noc_route`.
pub type MainbandNocRouteCfg = MainbandRouteCfg;
pub type MainbandNocRouteCmd = NocCmd;
pub type MainbandNocRouteDeviceToRni = DeviceToRniEntry;
pub type MainbandNocRouteRouteLfsr = RouteLfsrSamples;
pub type MainbandNocRouteReturn = RouteResult;
```

## Branches And Inputs

- xlsynth branch: `dank/mainband-rust-types-xlsynth-20260415111738`
- xlsynth prototype commit: `b7931401955792bf6ae82bce31dc82ff7b416359`
- Chili source branch used for inspection: `dank/mainband-rust-types-prototype-20260415111738`
- Chili source base: `origin/oai-main` at `68ec43bda818f268ec16c50fd529a9bc17be5b4b`
- Original referenced Chili commit: `10d8ad0783d872c559fbb0395c3f382c31ed9648`
- Target DSLX file:
  `oai_internal/silicon/serrano/noc/logic/mainband_noc_route.x`

The Chili branch has no code changes; it was only a clean worktree from latest `origin/oai-main`.
The code changes are in the xlsynth branch.

## What Changed

`xlsynth/src/dslx_bridge.rs` now defines `FunctionParamData` and adds a default no-op
`BridgeBuilder::add_function_signature(...)` hook. During module conversion, function members are
no longer ignored; their parameters and return annotation are passed to builders through that hook.
Existing builders remain compatible because the new trait method has a default implementation.

`xlsynth/src/rust_bridge_builder.rs` now has
`RustBridgeBuilder::with_function_signature_aliases(function_name)`. When the requested function is
encountered, it emits Rust type aliases for each parameter and for the return type. The alias names
are based on the DSLX function name plus the parameter name in upper camel case.

The Rust builder also preserves useful type-reference spelling from DSLX annotations when possible.
For imported DSLX types, it emits sibling-module Rust paths such as:

```rust
pub type CmdType = super::chili_noc_dslx::CmdType;
pub local_hbm_base_addr: super::chili_dslx::GlobalLineAddr,
```

This avoids the previous broken shape where an imported alias could become a self-alias like
`pub type CmdType = CmdType;`.

`xlsynth/examples/dslx_rs_bridge.rs` is a small prototype runner. It takes a leaf DSLX file, a DSLX
search path, a target function name, and optional dependency `.x` files. It emits dependency modules
first and then the selected leaf module with signature aliases.

## Reproduction

From the xlsynth branch:

```bash
cargo run -p xlsynth --example dslx_rs_bridge -- \
  /path/to/chili-hw-monorepo/oai_internal/silicon/serrano/noc/logic/mainband_noc_route.x \
  /path/to/chili-hw-monorepo/oai_internal/silicon/serrano \
  mainband_noc_route \
  /path/to/chili-hw-monorepo/oai_internal/silicon/serrano/common/logic/chili_dslx.x \
  /path/to/chili-hw-monorepo/oai_internal/silicon/serrano/common/logic/chili_noc_dslx.x \
  /path/to/chili-hw-monorepo/oai_internal/silicon/serrano/noc/logic/noc_types_dslx.x \
  > /tmp/mainband_noc_route.rs
```

The generated file should compile when included into a Rust crate that depends on `xlsynth`.
A minimal local check used during the prototype was:

```rust
include!("/tmp/mainband_noc_route.rs");
```

compiled with:

```bash
rustc --edition=2018 --crate-type lib --emit=metadata verify_generated_bridge.rs \
  -L target/debug/deps \
  --extern xlsynth=target/debug/deps/libxlsynth-<hash>.rlib
```

## Key Generated Types

`GlobalLineAddr` is emitted in `chili_dslx`:

```rust
pub struct GlobalLineAddr {
    pub flat_view: IrUBits<35>,
}

pub type GlobalLineAddrFlat = IrUBits<35>;
```

`CmdType` is emitted in `chili_noc_dslx`:

```rust
pub enum CmdType {
    CommandInvalid = 0,
    CommandWriteRequest = 1,
    CommandReadRequest = 2,
    CommandAtomicRequest = 3,
    CommandReserved0 = 4,
    CommandWriteAsyncRequest = 5,
    CommandReserved1 = 6,
    CommandReserved2 = 7,
    CommandAtomicNoDataResponse = 8,
    CommandWriteResponse = 9,
    CommandReadResponse = 10,
    CommandAtomicDataResponse = 11,
    CommandPrefetchLine = 12,
    CommandReserved3 = 13,
    CommandReserved4 = 14,
    CommandReserved5 = 15,
}
```

The selected leaf module emits the route boundary types:

```rust
pub type RouteLfsrSamples = [IrUBits<11>; 4];

pub struct MainbandRouteCfg {
    pub local_router_port: RouterPort,
    pub local_hbm_base_addr: super::chili_dslx::GlobalLineAddr,
    pub local_hbm_addr_mask: super::chili_dslx::GlobalLineAddr,
    pub local_hbm_addr_limit: super::chili_dslx::GlobalLineAddr,
    pub local_util_base_addr: super::chili_dslx::GlobalLineAddr,
    pub local_util_addr_mask: super::chili_dslx::GlobalLineAddr,
    pub local_pcie_base_addr: super::chili_dslx::GlobalLineAddr,
    pub local_pcie_addr_mask: super::chili_dslx::GlobalLineAddr,
    pub endpoint_router_port: [RouterPortPath; 128],
    pub rni_router_port: [RouterPortPath; 64],
    pub route_errors_to_default_endpoint: IrUBits<1>,
    pub default_router_port: RouterPortPath,
    pub path_available_mask: IrUBits<32>,
}

pub struct NocCmd {
    pub cmd_type: CmdType,
    pub packet: NocPacket,
}

pub type DeviceToRniEntry = [DeviceToRniMask; 4];
```

## Validation

Commands run successfully:

```bash
cargo run -p xlsynth --example dslx_rs_bridge -- ...
cargo test -p xlsynth rust_bridge_builder
cargo clippy -p xlsynth --tests
cargo clippy -p xlsynth --examples
cargo fmt
```

The generated Rust file also compiled as an included module against the local `xlsynth` build
artifacts.

A mutation check was performed for the new signature-alias behavior: changing the generated alias
names made `test_convert_leaf_module_function_signature_aliases` fail, and restoring the intended
logic made it pass again.

The full workspace command:

```bash
cargo clippy --tests
```

was blocked by pre-existing unrelated clippy errors in `xlsynth-mcmc/src/multichain.rs`
(`too_many_arguments` and `needless_range_loop`). No changes were made to that crate for this
prototype.

## Known Prototype Limitations

- The example runner requires dependency `.x` files to be listed manually. A production CLI/API
  should walk imports and emit dependency modules automatically.
- Generated Rust uses `IrUBits<N>` and `IrSBits<N>`, matching the existing Rust bridge. It does not
  emit native integer primitives.
- Generated enums currently implement `Into<IrValue>` following existing bridge behavior. A future
  cleanup could prefer `From<Enum> for IrValue` and add common derives.
- Cross-module naming is only lightly modeled. The prototype emits `super::<module>::<type>` paths
  for imported refs, which works for the sibling-module output shape used here.
- This is not yet wired into `xlsynth-driver`; the runner is an `examples/` binary intended for
  experimentation and handoff.

## Suggested Next Steps

1. Decide whether signature aliases should be part of `x_path_to_rs_bridge` directly or exposed as a
   separate public API that takes a target function name.
2. Replace the prototype example's manual dependency list with import graph traversal.
3. Add a Chili-shaped regression test using small in-memory DSLX modules with imported struct and
   enum refs, rather than depending on the Chili repo.
4. Decide the final Rust type surface: keep `IrUBits` wrappers for interpreter interop, or add a
   native-Rust mode similar to the AOT wrapper type strategy.
