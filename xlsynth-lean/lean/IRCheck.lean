-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.IRProof
import Target
import UserProof

example : Target.Equivalent := UserProof.equivalent
#check_ir_axioms UserProof.equivalent
