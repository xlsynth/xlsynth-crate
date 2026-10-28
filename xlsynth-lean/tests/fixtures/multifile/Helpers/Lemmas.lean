-- SPDX-License-Identifier: Apache-2.0

import Std.Tactic.BVDecide

namespace Helpers.Lemmas

theorem distributive (x y z : BitVec 4) : x * (y + z) = x * y + x * z := by
  bv_decide

end Helpers.Lemmas
