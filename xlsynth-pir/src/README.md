# XLS IR parameter and node reference conventions

This crate treats parameters as ordinary nodes so that algorithms which only
understand `NodeRef` indices can work without a special case.

## Parameter nodes

- Every parsed function begins with a sentinel node at index `0` whose payload
  is `Nil`.
- After that sentinel the parser automatically appends one `GetParam` node for
  each parameter in the function signature. The nodes are stored in signature
  order, so the first parameter lives at `NodeRef { index: 1 }`, the second at
  `{ index: 2 }`, and so on. The dense numbering is independent of the textual
  `id=` attribute that appears in the IR.
- Each `GetParam` node carries the parameter's name, type and `ParamId`. Because
  the payload's operand list is empty, these nodes behave as leaves in
  dependency walks (for example `operands` or `get_topological`). Later nodes
  simply record the `NodeRef` of the parameters they consume.
- The `params_are_dense_node_refs` unit test in
  `xlsynth-g8r/src/xls_ir/ir.rs` verifies this behaviour and shows an `add`
  instruction reading two parameters purely through their `NodeRef` operands.

## Returning parameters without extra work

- Most IR text does **not** include explicit `param(...)` instructions because
  the implicit nodes described above already make parameters addressable.
- When a function wants to return a parameter directly—without adding an
  `identity` node—the IR can emit `ret name: ty = param(name=name, id=...)` as
  its entire body. Parsing such a function produces a second `GetParam` node at
  the end of the node list; the return reference points at that node so the IR
  printer can emit the explicit `param` line.
- The `returning_parameter_uses_explicit_get_param_node` test demonstrates this
  round-trip: it checks that the parser keeps both the implicit parameter node
  and the explicit return node while `Display` prints a body containing only the
  `param(...)` line.

For more detail, consult the tests referenced above—they are a concise,
executable reference for the invariants described here.

## Extension ops

PIR extension op syntax, semantics, desugaring, FFI-wrapper metadata and
lowering references are documented in [../docs/extensions.md](../docs/extensions.md).

## Native values and the libxls boundary

`xlsynth_pir::IrValue` and `xlsynth_pir::IrBits` are the native Rust representations
used by PIR literals, parsing, evaluation, rewriting, and the Cranelift value
adapter. They live in this crate alongside `ir::Type`; no separate value crate
or compatibility type aliases are needed.

`IrBits` retains its exact width and stores canonical, least-significant-first
`u64` limbs, with one limb inline. Unused high bits are always zero. Arithmetic
has XLS bitvector semantics: addition/subtraction wrap at the operand width,
multiplication returns the sum of operand widths, and signed operations use
two's complement. General arithmetic uses Rust's `num-bigint`, not libxls.

`IrValue` represents bits, tokens, tuples, and homogeneous arrays. Aggregate
children use shared immutable storage. The `IrValue::Array(IrArray)` payload
has private fields and read-only accessors, so arrays can only be constructed
through the checked constructors. `IrValue::make_array_typed` retains the
element type of an empty array; bare `[]` cannot be parsed without an element
type. Typed parsing and formatting of bits, tokens, tuples, and nonempty arrays
use XLS syntax.

Use `as_bits()` and `as_elements()` to inspect values without cloning their
storage; `to_bits()`, `get_element()`, and `get_elements()` return owned copies.
Native byte conversion (`to_le_bytes()` / `to_bytes()`) and value formatting
(`to_string_fmt()` / `to_string_fmt_no_prefix()`) are infallible. Parsing,
checked construction, type checks, and bounds checks still return errors.

```rust
use xlsynth_pir::{IrBits, IrValue};

let value = IrValue::parse_typed("(bits[129]:0x1, [bits[8]:2, bits[8]:3])")?;
let bits = IrBits::make_ubits(129, 7)?;
assert_eq!(bits.add(&bits).get_bit_count(), 129);
# Ok::<(), xlsynth_pir::ValueError>(())
```

The Cranelift compiler's `run_ir_values` adapter accepts native `IrValue`s and
packs them into its existing native ABI layout. It checks the value's full type,
including aggregate kind, before execution. Its scalar and packed-buffer entry
points remain available for callers that want to reuse buffers.

The libxls-backed wrappers are named `xlsynth::XlsIrValue` and
`xlsynth::XlsIrBits`. At an upstream interpreter, JIT, DSLX, or other libxls
boundary, use `libxls_bridge::{value_from_libxls, value_to_libxls, bits_from_libxls, bits_to_libxls}`. Inbound value conversion requires an
`ir::Type` and validates its shape and widths. Outbound conversion rejects typed
empty arrays, which the current libxls value constructor cannot represent.
Gate simulation, netlist literals, gate serialization, and Verilog simulation
adapters use native values and bits directly. Range-analysis results are copied
to native storage once, when they leave the XLS analysis API.

This makes value operations native; it does not remove the crate's existing
libxls dependency for upstream optimization and other XLS integration.

### Native `.irvals` files

`parse_ir_values` and `parse_ir_values_file` read newline-delimited values into
native `IrValuesFile` records. Files contain either positional typed values or
named records such as `{y: bits[1]:0, x: bits[129]:0x1}`. Both parsing and
formatting use native `IrValue`s; no libxls value allocation is involved.

`IrValuesFile::into_positional_values` binds named records to the supplied
argument order and rejects duplicate, missing, or unknown names. It leaves
positional records unchanged; callers must still validate their types against
the function signature. `NamedIrValueSet::from_positional_tuple` performs the
reverse operation for corpus writers. Names with punctuation can be
JSON-quoted. Mixed record forms and blank lines are rejected by the shared
parser; callers that allow blank lines or comments can filter them first.

The `.irvals` API lives only in this crate. Consumers that invoke XLS convert
the parsed native values at the C API boundary with `libxls_bridge`.
