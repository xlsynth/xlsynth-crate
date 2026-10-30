-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Decide

open Std.Sat
open XlsynthLean.Proof

namespace EdgeCases

/-- Input x and constant true agree exactly on the domain x = true. -/
private def guarded : Claim 1 :=
  let x := (AIG.empty : AIG Nat).mkAtom 0
  let one := x.aig.mkConstCached true
  let empty : AIG.RefVec x.aig 0 := .emptyWithCapacity 1
  { graph := x.aig, guard := x.ref, lhs := empty.push x.ref, rhs := empty.push one }

example : guarded.Valid := by aig_decide

private def vacuous : Claim 1 :=
  { guarded with guard := guarded.graph.mkConstCached false }

example : vacuous.Valid := by aig_decide

/-- Every input satisfies this guard, but there are no output bits to mismatch. -/
private def noOutputs : Claim 0 :=
  let graph : AIG Nat := AIG.empty
  { graph, guard := graph.mkConstCached true
    lhs := .emptyWithCapacity 0, rhs := .emptyWithCapacity 0 }

example : noOutputs.Valid := by aig_decide

end EdgeCases
