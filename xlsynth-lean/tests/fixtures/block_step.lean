-- SPDX-License-Identifier: Apache-2.0

-- Generated from XLS block swap in package swap.
-- Exact BitVec values; tuple elements end in Unit.
-- Outputs read old registers; all writes commit together.
-- Source assertions are results, never assumptions.
import Std

namespace XlsBlock

structure State where
  «a» : BitVec 2
  «b» : BitVec 2
  deriving Repr

structure Input where
  «rst» : BitVec 1
  «en» : BitVec 1
  deriving Repr

structure Output where
  «out» : BitVec 2
  deriving Repr

structure Assertions where
  -- id=6; label=enable; message=enable expected
  «checked» : Bool
  deriving Repr

structure Evaluation where
  next : State
  output : Output
  assertions : Assertions
  deriving Repr

def eval (q : State) (x : Input) : Evaluation :=
  -- node rst [id=1, op=input_port, type=bits[1]]
  let xls_node_1_rst : BitVec 1 := x.«rst»
  -- node en [id=2, op=input_port, type=bits[1]]
  let xls_node_2_en : BitVec 1 := x.«en»
  -- node old_a [id=3, op=register_read, type=bits[2]]
  let xls_node_3_old_a : BitVec 2 := q.«a»
  -- node old_b [id=4, op=register_read, type=bits[2]]
  let xls_node_4_old_b : BitVec 2 := q.«b»
  -- node tok [id=5, op=literal, type=token]
  let xls_node_5_tok : Unit := ()
  -- node checked [id=6, op=assert, type=token]
  let xls_node_6_checked : Unit := xls_node_5_tok
  -- node write_a [id=7, op=register_write, type=()]
  let xls_node_7_write_a : Unit := ()
  -- node write_b [id=8, op=register_write, type=()]
  let xls_node_8_write_b : Unit := ()
  -- node out [id=9, op=output_port, type=()]
  let xls_node_9_out : Unit := ()
  { next := { «a» := (if xls_node_1_rst == (BitVec.ofNat 1 0) then (BitVec.ofNat 2 0x1) else (if (xls_node_2_en == (BitVec.ofNat 1 1)) then xls_node_4_old_b else q.«a»)), «b» := xls_node_3_old_a },
    output := { «out» := xls_node_3_old_a },
    assertions := { «checked» := (xls_node_2_en == (BitVec.ofNat 1 1)) } }

def step (q : State) (x : Input) : State := (eval q x).next

def observe (q : State) (x : Input) : Output := (eval q x).output

def assertions (q : State) (x : Input) : Assertions := (eval q x).assertions

def assertionsHold (q : State) (x : Input) : Prop :=
  let checks := assertions q x
  And (checks.«checked» = true) (True)

-- One clock with the declared reset port active. Other inputs and old
-- registers stay arbitrary. With no reset port this is an ordinary clock.
def reset (q : State) (x : Input) : Evaluation :=
  eval q { x with «rst» := BitVec.ofNat 1 0 }

end XlsBlock
