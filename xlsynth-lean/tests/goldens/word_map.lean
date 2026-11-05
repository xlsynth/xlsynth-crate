-- SPDX-License-Identifier: Apache-2.0
import XlsynthLean.IR

namespace «Generated».«Function»
set_option linter.unusedVariables false

def ir_widen (n1 : BitVec 4) : BitVec 8 :=
  let n2 : BitVec 8 := n1.signExtend 8
  n2

def ir_transform (n3 : Vector (BitVec 4) 3) : Vector (BitVec 8) 3 :=
  let n4 : Vector (BitVec 8) 3 := Vector.map ir_widen n3
  n4

abbrev eval := ir_transform

def evalPacked (inputs : BitVec 12) : BitVec 24 :=
  BitVec.ofNat _ (_root_.XlsynthLean.IR.Packed.pack (eval (_root_.XlsynthLean.IR.Packed.unpack (inputs.toNat >>> 0))))

end «Generated».«Function»
