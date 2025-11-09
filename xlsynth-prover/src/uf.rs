// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for inferring and merging uninterpreted-function signatures.

use std::collections::HashMap;

use crate::types::UfSignature;

/// Infers UF signatures for the provided package and map from functions to UF
/// names.
pub fn infer_uf_signatures(
    pkg: &xlsynth_pir::ir::Package,
    uf_map: &HashMap<String, String>,
) -> Result<HashMap<String, UfSignature>, String> {
    let mut uf_sigs: HashMap<String, UfSignature> = HashMap::new();
    for (fn_name, uf_sym) in uf_map {
        let (ir_fn, skip_implicit) = match pkg.get_fn(fn_name) {
            Some(f) => (f, false),
            None => {
                let itok_name = format!("__itok{}", fn_name);
                match pkg.get_fn(&itok_name) {
                    Some(f) => (f, true),
                    None => {
                        return Err(format!(
                            "Unknown function '{}' when inferring UF signature for symbol '{}'",
                            fn_name, uf_sym
                        ));
                    }
                }
            }
        };

        let arg_widths: Vec<usize> = if skip_implicit {
            ir_fn
                .params
                .iter()
                .skip(2)
                .map(|p| p.ty.bit_count())
                .collect()
        } else {
            ir_fn.params.iter().map(|p| p.ty.bit_count()).collect()
        };
        let ret_width = ir_fn.ret_ty.bit_count();
        let sig = UfSignature {
            arg_widths,
            ret_width,
        };

        if let Some(prev) = uf_sigs.get(uf_sym) {
            if prev != &sig {
                return Err(format!(
                    "Conflicting UF signature for symbol '{}': {:?} vs {:?}",
                    uf_sym, prev, sig
                ));
            }
        } else {
            uf_sigs.insert(uf_sym.clone(), sig);
        }
    }
    Ok(uf_sigs)
}

/// Merges UF signature maps, ensuring there are no conflicts.
pub fn merge_uf_signatures(
    mut lhs: HashMap<String, UfSignature>,
    rhs: &HashMap<String, UfSignature>,
) -> Result<HashMap<String, UfSignature>, String> {
    for (name, sig) in rhs {
        if let Some(prev) = lhs.get(name) {
            if prev != sig {
                return Err(format!(
                    "Conflicting UF signature for symbol '{}': {:?} vs {:?}",
                    name, prev, sig
                ));
            }
        } else {
            lhs.insert(name.clone(), sig.clone());
        }
    }
    Ok(lhs)
}

/// Convenience helper that infers UF signatures for both sides of an
/// equivalence proof and merges them, ensuring there are no conflicts.
pub fn infer_merged_uf_signatures(
    lhs_pkg: &xlsynth_pir::ir::Package,
    lhs_map: &HashMap<String, String>,
    rhs_pkg: &xlsynth_pir::ir::Package,
    rhs_map: &HashMap<String, String>,
) -> Result<HashMap<String, UfSignature>, String> {
    let lhs = infer_uf_signatures(lhs_pkg, lhs_map)?;
    let rhs = infer_uf_signatures(rhs_pkg, rhs_map)?;
    merge_uf_signatures(lhs, &rhs)
}
