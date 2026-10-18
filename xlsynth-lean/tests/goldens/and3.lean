-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aag

namespace «Generated».«And3»

def circuit : XlsynthLean.Aig.Circuit 3 1 :=
  (XlsynthLean.Aag.parse 3 1
    (include_str "And3.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec 3) : BitVec 1 :=
  circuit.eval inputs

end «Generated».«And3»
