-- SPDX-License-Identifier: Apache-2.0

import ReadableCorrespondence
open XlsReadable
private def run (mode : BitVec 2) :=
  let stimulus : Input := {rst := false, mode, pool := {status := (3, 4), slots := #v[
    {write := {valid := true, data := 11}},
    {write := {valid := false, data := 19}},
    {write := {valid := true, data := 23}}]}}
  (step {count := 7} stimulus).count
#guard run 2 == 30
#guard run 0 == 7
