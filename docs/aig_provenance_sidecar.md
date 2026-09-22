# AIG provenance sidecar proposal

`ir-toggle-hotspots` measures IR values without lowering. `aig-toggle-hotspots`
measures an existing gate artifact without committing to one lowering pipeline.
Native g8r artifacts can already carry source IR node IDs, but AIGER does not,
and neither format carries the exact IR-output-bit-to-AIG-signal map. A separate
protobuf sidecar can supply that debug information when the producer has it.

## Identity and meaning

- Bind the sidecar to the **final serialized artifact** by hashing its exact
  bytes (SHA-256). The consumer rejects a mismatched artifact. For AIGER,
  define gate IDs in terms of the serialized AIGER variable numbers and retain
  the loader's variable-to-in-memory-node mapping; loading may renumber nodes.
- Identify the precise prepared IR function used for attribution, including a
  digest of its serialized package and the selected function name. The IR node
  table uses that function's IDs. Preprocessing may reuse an original IR ID for
  a different value, so labels from the unprepared package are not equivalent.
- Source associations mean a gate was constructed from or shared by an IR node;
  they do **not** imply equality with a specific output bit. Keep exact output
  bit matches in a separate optional table, with polarity. A producer only
  records a bit match when it can follow that signal through its final AIG
  transformations. An unknown match stays absent.
- The sidecar is optional. AIGs from tools that cannot emit it still get gate
  toggle counts. A sidecar must not change simulation or ranking.
- Validate array lengths, source offsets, node indices, bit widths, polarity
  bitmap length, and artifact identity before attaching any label.

## Compact protobuf layout (proposed)

```proto
syntax = "proto3";
package xlsynth.aig_provenance;

message IrNode {
  uint64 text_id = 1;
  uint32 name_index = 2; // Index into interned_strings.
  uint32 op_index = 3;
  uint64 flattened_bit_width = 4;
}

message OutputBitRun {
  uint32 ir_node_index = 1;
  uint64 first_bit = 2; // Flattened, least-significant bit first.
  uint64 bit_count = 3;
  uint64 first_aig_id = 4;
  // Present only when AIG IDs advance contiguously across the run.
  // Otherwise encode single-bit runs.
  bool inverted = 5;
}

message AigProvenance {
  uint32 format_version = 1;
  bytes artifact_sha256 = 2;
  bytes prepared_ir_sha256 = 3;
  string function_name = 4;
  repeated string interned_strings = 5;
  repeated IrNode ir_nodes = 6; // Only referenced nodes; sorted by text_id.
  // CSR encoding: only gates with source associations are present. Gate IDs
  // are sorted and delta encoded. ends[i] is the exclusive end offset for
  // gate i in source_ir_node_indices, so repeated strings are never stored.
  repeated uint64 gate_id_deltas = 7 [packed = true];
  repeated uint32 source_ends = 8 [packed = true];
  repeated uint32 source_ir_node_indices = 9 [packed = true];
  repeated OutputBitRun exact_output_runs = 10;
  // Irregular/single-bit matches use aligned packed columns rather than one
  // protobuf message per bit. Sort by AIG ID, IR node index, then bit index.
  repeated uint32 exact_bit_ir_node_indices = 11 [packed = true];
  repeated uint64 exact_bit_indices = 12 [packed = true];
  repeated uint64 exact_bit_aig_id_deltas = 13 [packed = true];
  bytes exact_bit_inverted_bitmap = 14; // One polarity bit per match.
}
```

For a large AIG, the sparse gate table and packed integer columns avoid a
protobuf object for every gate or gate/source pair. Interned names and ops are
stored once. Runs avoid a record per bit when a wide IR value lowers to
contiguous AIG IDs; packed columns and a polarity bitmap represent isolated
bit matches. Compression of the entire `.pb` as `.pb.zst` can be added without
changing the schema.

## Producer and consumer steps

1. Emit a sidecar from the final `ir2g8r` or other lowering result after all
   rewrites and AIG optimizations. Propagate exact bit matches through a
   transformation only when it preserves their corresponding signals.
1. Add an optional `--provenance <PATH>` to `aig-toggle-hotspots`; verify the
   artifact digest, read the sparse source table, and attach exact bit labels
   only where the exact output mappings say they are available.
1. Exercise native g8r and both AIGER encodings, an optimized/reordered AIG,
   a mismatched digest, and a wide IR value in roundtrip and command tests.

This is a format proposal; the current toggle commands do not emit or consume
the sidecar.
