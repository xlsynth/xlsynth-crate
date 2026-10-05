# Priority-result fixtures

The historical priority fixture is the exact archived std.x::next_pow2 MFFC;
provenance.json records action IDs and hashes. Others are synthetic controls.
shared-amount63.ir reproduces an unguarded graph logical-effort regression.
The guard rejects externally shared count/amount while retaining shared one-hot.

To reproduce current stdlib behavior, compile a DSLX wrapper calling
std::next_pow2(x) for x: u32 with dslx2ir --opt=true. Prepare with and without
--experimental-priority-result=true, then map using identical settings.
Raw experiments disable FRAIG, cut rewriting, reassociation and ABC.
