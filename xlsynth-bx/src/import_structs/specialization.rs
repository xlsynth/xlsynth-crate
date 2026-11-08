// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth::dslx::{self, ImportData};

use super::{DeclId, ImportedStructs};
use crate::check::MAX_TYPE_SIZE;
use crate::dslx_import::{ProbeCapture, TypeWitness};

impl ImportedStructs {
    /// Infer every u32 actual from the checked value, including bindings absent
    /// from its fields. The released DSLX API exposes invocation bindings.
    pub(super) fn capture_struct_bindings(
        &mut self,
        id: &DeclId,
        def: &dslx::StructDef,
        witness: &TypeWitness,
        data: &mut ImportData,
    ) -> Result<Vec<u32>, String> {
        let count = def.get_parametric_binding_count();
        if count > MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX struct has more than {MAX_TYPE_SIZE} parametric bindings"
            ));
        }
        let names: Vec<_> = (0..count).map(|i| format!("P{i}")).collect();
        let formals: Vec<_> = names.iter().map(|name| format!("{name}: u32")).collect();
        // A runtime array index also permits a projection through an empty
        // array.
        let source = format!(
            r#"{}
import {} as bx_template;
fn capture<{}>(_value: bx_template::{}<{}>) {{}}
fn probe(_bx_array_index: u32, {}) {{ capture({}) }}
"#,
            witness.imports,
            id.subject,
            formals.join(", "),
            id.name,
            names.join(", "),
            witness.params,
            witness.expr,
        );
        let probe = format!("0bx_struct_{}", self.probe_index);
        self.probe_index += 1;
        #[cfg(test)]
        {
            self.type_witnesses += 1;
        }
        let checked = self.check_probe(&source, &probe, data).map_err(|e| {
            format!(
                "recovering u32 value bindings for DSLX struct '{}::{}': {e}",
                id.subject, id.name
            )
        })?;
        let capture = ProbeCapture::new(&checked, |_| None)?;
        let captured = capture.param_type(0)?;
        if captured.get_struct_def().map_err(|e| e.to_string())? != *def {
            return Err(format!(
                "DSLX binding capture selected a different struct than '{}::{}'",
                id.subject, id.name
            ));
        }
        let env = capture.bindings()?;
        let bindings: BTreeMap<_, _> = env.bindings().into_iter().collect();
        names
            .iter()
            .map(|name| {
                bindings
                    .get(name)
                    .ok_or_else(|| format!("missing DSLX struct value binding '{name}'"))?
                    .convert_to_ir()
                    .and_then(|v| v.to_u32())
                    .map_err(|e| e.to_string())
            })
            .collect()
    }
}
