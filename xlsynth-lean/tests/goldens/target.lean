-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aag

namespace Generated.Lhs

def circuit : XlsynthLean.Aig.Circuit 3 1 :=
  (XlsynthLean.Aag.parse 3 1
    (include_str "Lhs.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec 3) : BitVec 1 :=
  circuit.eval inputs

end Generated.Lhs

namespace Generated.Rhs

def circuit : XlsynthLean.Aig.Circuit 3 1 :=
  (XlsynthLean.Aag.parse 3 1
    (include_str "Rhs.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec 3) : BitVec 1 :=
  circuit.eval inputs

end Generated.Rhs

namespace Target

abbrev Equivalent : Prop :=
  XlsynthLean.Aig.Circuit.Equivalent Generated.Lhs.circuit Generated.Rhs.circuit

end Target
