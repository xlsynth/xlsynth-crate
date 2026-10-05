// SPDX-License-Identifier: Apache-2.0

//! Format checks that distinguish mapped GV from unmapped UGV syntax.

use std::collections::HashSet;

use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

use crate::netlist::bit_ref;
use crate::netlist::parse::{AssignExpr, Net, NetlistModule, Pos, ScanError, Span};

/// Requires every GV assignment to be pure wiring, leaving logic to cells.
pub(crate) fn validate_gv_form(
    modules: &[NetlistModule],
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<(), ScanError> {
    for module in modules {
        for assign in &module.assigns {
            if !matches!(assign.rhs, AssignExpr::Leaf(_)) {
                return Err(ScanError {
                    message: format!(
                        "GV permits only wiring in assignments; assignment to '{}' computes Boolean logic. Represent the logic with cells, or use ugv2aig for a cell-free UGV netlist",
                        bit_ref::render_net_ref(&assign.lhs, nets, interner)
                    ),
                    span: assign.span,
                });
            }
        }
    }
    Ok(())
}

/// Requires every UGV instance to refer to a module defined in the same input.
pub(crate) fn validate_ugv_form(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<(), ScanError> {
    let module_names = modules
        .iter()
        .map(|module| module.name)
        .collect::<HashSet<_>>();
    for module in modules {
        for instance in &module.instances {
            if !module_names.contains(&instance.type_name) {
                let pos = Pos {
                    lineno: instance.inst_lineno,
                    colno: instance.inst_colno,
                };
                return Err(ScanError {
                    message: format!(
                        "UGV does not permit leaf cell instance '{}' of type '{}'; use gv2aig with --liberty_proto for cell netlists",
                        interner
                            .resolve(instance.instance_name)
                            .unwrap_or("<unknown>"),
                        interner.resolve(instance.type_name).unwrap_or("<unknown>")
                    ),
                    span: Span {
                        start: pos,
                        limit: pos,
                    },
                });
            }
        }
    }
    Ok(())
}
