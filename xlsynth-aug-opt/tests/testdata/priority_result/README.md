# Priority-result fixtures

The historical priority fixture is the exact archived std.x::next_pow2 MFFC;
provenance.textproto records action IDs and hashes. Others are synthetic controls.
shared-amount63.ir reproduces an unguarded graph logical-effort regression.
The guard rejects externally shared count/amount while retaining shared one-hot.

To reproduce current stdlib behavior, compile a DSLX wrapper calling
std::next_pow2(x) for x: u32 with dslx2ir --opt=true. Run ir2opt with
--aug-opt=true and with/without --aug-opt-fuse-priority-results=true, then map
using identical settings. Compare --aug-opt-rounds=1 and =3 after the complete
XLS/PIR/XLS sandwich.
Raw comparisons disable FRAIG, cut rewriting, reassociation and ABC.

The typed metadata schema is provenance.proto; provenance.bin is its checked-in
descriptor. The provenance test validates required fields, paths, complete
fixture coverage, and SHA-256 hashes. reflected-index32.ir uses an arbitrary
reflection constant, demonstrating an affine decoder outside next_pow2.

The pass supports decode and one-shift consumers of affine priority indices;
see the driver README for its fixed reference cost profile and sharing limits.

Regenerate the descriptor from this directory with:

```sh
protoc --include_source_info --descriptor_set_out=provenance.bin provenance.proto
```

The descriptor was generated with protoc 3.21.12.
