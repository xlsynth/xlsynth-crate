-- SPDX-License-Identifier: Apache-2.0
import XlsynthLean.IR

local notation "L.BitVec.toNat" => _root_.BitVec.toNat

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

abbrev Rows := Vector (Vector (BitVec 8) 2) 2

def update (rows : Rows) (outer v0 : BitVec 1) (value : BitVec 8) : Rows :=
  _root_.XlsynthLean.IR.arrayModify rows (L.BitVec.toNat outer) (_root_.XlsynthLean.IR.arrayModify · (L.BitVec.toNat v0) (fun _ => value))

def rows : Rows := #[#[1, 2].toVector, #[3, 4].toVector].toVector
#guard update rows 0 0 9 == #[#[9, 2].toVector, #[3, 4].toVector].toVector
#guard update rows 0 1 9 == #[#[1, 9].toVector, #[3, 4].toVector].toVector

end Example
