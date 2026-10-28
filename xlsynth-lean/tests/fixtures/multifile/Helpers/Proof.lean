-- SPDX-License-Identifier: Apache-2.0

import Target
import Helpers.Lemmas

namespace Helpers.Proof

theorem equivalent : Target.Equivalent := by
  have _ := Helpers.Lemmas.distributive
  unfold Target.Equivalent XlsynthLean.Aig.Circuit.Equivalent
  native_decide

end Helpers.Proof
