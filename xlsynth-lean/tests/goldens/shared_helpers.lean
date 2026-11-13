-- SPDX-License-Identifier: Apache-2.0
import Std

namespace Example

def Helpers.«widen» (n1 : BitVec 4) : BitVec 8 :=
  let n2 : BitVec 8 := n1.signExtend 8
  n2

def Helpers.«lanes» (n5 : Vector (BitVec 4) 2) : Vector (BitVec 8) 2 :=
  let n6 : Vector (BitVec 8) 2 := Vector.map Helpers.«widen» n5
  n6

def Helpers.«scalar» (n3 : BitVec 4) : BitVec 8 :=
  let n4 : BitVec 8 := Helpers.«widen» n3
  n4

def observe (x : BitVec 4) : BitVec 8 := Helpers.«scalar» (x + 1)

#guard observe 7 == 248
#guard Helpers.«lanes» #[1, 15].toVector == #[1, 255].toVector

end Example
