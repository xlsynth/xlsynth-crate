-- SPDX-License-Identifier: Apache-2.0

import Generated.Lhs
import Generated.Rhs

namespace Target

abbrev Equivalent : Prop :=
  ∀ arg0 arg1, Generated.Lhs.eval arg0 arg1 = Generated.Rhs.eval arg0 arg1

end Target
