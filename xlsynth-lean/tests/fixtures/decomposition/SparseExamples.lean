-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Sparse

open Std.Sat
open scoped AIG
open XlsynthLean.Aig XlsynthLean.Proof

namespace SparseExamples

-- Node 3 carries a high label. Nodes 4 through 6 reduce to a && b,
-- because the branch depending on node 3 is ANDed with false.
private def f := (Circuit.fromDecls 90001 2
  #[.false, .atom 100, .atom 50000, .atom 90000,
    .gate (.ofRaw 6) (.ofRaw 0), .gate (.ofRaw 2) (.ofRaw 4),
    .gate (.ofRaw 9) (.ofRaw 10), .atom 70000]
  #[.ofRaw 10, .ofRaw 12]).toOption.get (by native_decide)

private def parent : Claim 1 :=
  let one := AIG.RefVec.emptyWithCapacity 1
  ⟨f.aig, f.aig.mkConstCached true, one.push (f.outputs.get 0 (by decide)),
    one.push (f.outputs.get 1 (by decide))⟩

-- Atom 70000 is never visited. Atom 90000 is removed by folding and sweeping.
-- Four rows suffice for the two live labels, in either order.
private def leaf := (parent.sparseLeaf [50000, 100]).toOption.get (by native_decide)
example : parent.Valid := leaf.valid (by native_decide) (by native_decide)
example : leaf.child.graph.decls.size = 4 := by native_decide

-- Leaving out atom 100 cannot silently identify it with row bit zero.
private def omitted := (parent.sparseLeaf [50000]).toOption.get (by native_decide)
example : omitted.child.atomsBelow 1 = false := by native_decide

-- A dense relabeling still finds a real mismatch in the rows.
private def unequal : Claim 1 :=
  let one := AIG.RefVec.emptyWithCapacity 1
  ⟨f.aig, f.aig.mkConstCached true, one.push ⟨1, false, by native_decide⟩,
    one.push ⟨2, false, by native_decide⟩⟩
private def wrong := (unequal.sparseLeaf [50000, 100]).toOption.get (by native_decide)
example : ((wrong.child.mismatchCircuit 2).eval 1).getLsbD 0 = true := by native_decide

end SparseExamples
