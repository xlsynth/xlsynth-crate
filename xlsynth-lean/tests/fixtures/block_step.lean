-- SPDX-License-Identifier: Apache-2.0

-- Generated from XLS block swap in package swap.
-- Exact BitVec values; tuple elements end in Unit.
-- Outputs read old registers; all writes commit together.
-- Source assertions are results, never assumptions.

import XlsynthLean.IR

open _root_ renaming Repr -> L.Repr
local notation "L.And" => _root_.And
local notation "L.BitVec" => _root_.BitVec
local notation "L.BitVec.ofNat" => _root_.BitVec.ofNat
local notation "L.Bool" => _root_.Bool
local notation "L.True" => _root_.True
local notation "L.Unit" => _root_.Unit

namespace XlsBlock

structure State where
  Constructors.mk ::
  «a» : L.BitVec 2
  «b» : L.BitVec 2
  deriving L.Repr

structure Input where
  Constructors.mk ::
  «rst» : L.BitVec 1
  «en» : L.BitVec 1
  deriving L.Repr

structure Output where
  Constructors.mk ::
  «out» : L.BitVec 2
  deriving L.Repr

structure Assertions where
  Constructors.mk ::
  -- id=6; label=enable; message=enable expected
  «checked» : L.Bool
  deriving L.Repr

structure Evaluation where
  Constructors.mk ::
  next : State
  output : Output
  assertions : Assertions
  deriving L.Repr

def eval (q : State) (x : Input) : Evaluation :=
  -- node rst [id=1, op=input_port, type=bits[1]]
  let xls_node_1_rst : L.BitVec 1 := x.«rst»
  -- node en [id=2, op=input_port, type=bits[1]]
  let xls_node_2_en : L.BitVec 1 := x.«en»
  -- node old_a [id=3, op=register_read, type=bits[2]]
  let xls_node_3_old_a : L.BitVec 2 := q.«a»
  -- node old_b [id=4, op=register_read, type=bits[2]]
  let xls_node_4_old_b : L.BitVec 2 := q.«b»
  -- node tok [id=5, op=literal, type=token]
  let xls_node_5_tok : L.Unit := ()
  -- node checked [id=6, op=assert, type=token]
  let xls_node_6_checked : L.Unit := xls_node_5_tok
  -- node write_a [id=7, op=register_write, type=()]
  let xls_node_7_write_a : L.Unit := ()
  -- node write_b [id=8, op=register_write, type=()]
  let xls_node_8_write_b : L.Unit := ()
  -- node out [id=9, op=output_port, type=()]
  let xls_node_9_out : L.Unit := ()
  { next := { «a» := (if xls_node_1_rst == (L.BitVec.ofNat 1 0) then L.BitVec.ofNat 2 0x1 else (if (xls_node_2_en == (L.BitVec.ofNat 1 1)) then xls_node_4_old_b else q.«a»)), «b» := xls_node_3_old_a },
    output := { «out» := xls_node_3_old_a },
    assertions := { «checked» := (xls_node_2_en == (L.BitVec.ofNat 1 1)) } }

def step (q : State) (x : Input) : State := (eval q x).next

def observe (q : State) (x : Input) : Output := (eval q x).output

def assertions (q : State) (x : Input) : Assertions := (eval q x).assertions

def assertionsHold (q : State) (x : Input) : Prop :=
  let checks := assertions q x
  L.And (checks.«checked» = true) (L.True)

-- One clock with the declared reset port active. Other inputs and old
-- registers stay arbitrary. With no reset port this is an ordinary clock.
def reset (q : State) (x : Input) : Evaluation :=
  eval q { x with «rst» := L.BitVec.ofNat 1 0 }

end XlsBlock
