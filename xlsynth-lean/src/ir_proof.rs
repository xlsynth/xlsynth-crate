// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_pir::ir::Package;

use crate::support::{IR_SUPPORT, write_file};
use crate::{LeanError, LeanFunction, LeanModuleName, export_ir_function};

/// Trusted word-level proof libraries supplied by the checker.
pub(crate) const FIXED_IR_PROOF_FILES: &[(&str, &[u8])] = &[
    (
        "XlsynthLean/IRProof.lean",
        include_bytes!("../lean/XlsynthLean/IRProof.lean"),
    ),
    (
        "XlsynthLean/IRDecompose.lean",
        include_bytes!("../lean/XlsynthLean/IRDecompose.lean"),
    ),
];

/// Two independently exported IR functions with the same positional XLS
/// signature.
#[derive(Debug)]
pub struct LeanIrFunctionEquivalenceTarget {
    lhs: LeanFunction,
    rhs: LeanFunction,
}

/// Selects the two entrypoints and validates their full argument and result
/// types.
pub fn prepare_ir_function_equivalence_target(
    lhs: &Package,
    lhs_top: Option<&str>,
    rhs: &Package,
    rhs_top: Option<&str>,
) -> Result<LeanIrFunctionEquivalenceTarget, LeanError> {
    let lhs = export_ir_function(lhs, lhs_top, &LeanModuleName::new("Generated.Lhs")?)?;
    let rhs = export_ir_function(rhs, rhs_top, &LeanModuleName::new("Generated.Rhs")?)?;
    if lhs.signature() != rhs.signature() {
        return Err(LeanError::InvalidIr(format!(
            "equivalence requires matching XLS signatures: lhs {:?}, rhs {:?}",
            lhs.signature(),
            rhs.signature()
        )));
    }
    Ok(LeanIrFunctionEquivalenceTarget { lhs, rhs })
}

impl LeanIrFunctionEquivalenceTarget {
    /// Prepares a proof directory with the shared IR module, both models, and
    /// their universally quantified, typed equality.
    pub fn write_to(&self, root: &Path) -> Result<(), LeanError> {
        IR_SUPPORT.write_to(root)?;
        self.lhs.write_to(root)?;
        self.rhs.write_to(root)?;
        for (path, contents) in FIXED_IR_PROOF_FILES {
            write_file(root, Path::new(path), contents)?;
        }
        let arguments = (0..self.lhs.signature().param_types.len())
            .map(|i| format!(" arg{i}"))
            .collect::<String>();
        let binders = if arguments.is_empty() {
            String::new()
        } else {
            format!("∀{arguments}, ")
        };
        let target = format!(
            r#"-- SPDX-License-Identifier: Apache-2.0

import Generated.Lhs
import Generated.Rhs

namespace Target

abbrev Equivalent : Prop :=
  {binders}Generated.Lhs.eval{arguments} = Generated.Rhs.eval{arguments}

end Target
"#
        );
        write_file(root, Path::new("Target.lean"), target.as_bytes())
    }
}
