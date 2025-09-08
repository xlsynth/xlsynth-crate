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
