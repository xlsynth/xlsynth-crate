// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_pir::ir::Package;

use crate::ir_fn::{Bindings, export_with_bindings};
use crate::support::write_file;
use crate::{LeanError, LeanFunction, LeanModuleName};

/// The checker supplies the audit implementation and root command itself.
pub(crate) const FIXED_IR_PROOF_FILES: &[(&str, &[u8])] = &[
    ("Check.lean", include_bytes!("../lean/IRCheck.lean")),
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
    let lhs = export_with_bindings(
        lhs,
        lhs_top,
        &LeanModuleName::new("Generated.Lhs")?,
        Bindings::NamedNodes,
    )?;
    let rhs = export_with_bindings(
        rhs,
        rhs_top,
        &LeanModuleName::new("Generated.Rhs")?,
        Bindings::NamedNodes,
    )?;
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
    /// Writes the definitions and their universally quantified, typed equality.
    pub fn write_to(&self, root: &Path) -> Result<(), LeanError> {
        self.lhs.write_to(root)?;
        self.rhs.write_module(root)?;
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
