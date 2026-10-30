// SPDX-License-Identifier: Apache-2.0

//! Format checks that distinguish mapped GV from unmapped UGV syntax.

use std::collections::HashSet;

use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

use crate::netlist::bit_ref;
use crate::netlist::parse::{
    AssignExpr, Net, NetRef, NetlistAssignKind, NetlistModule, Pos, ScanError, Span,
};

/// Checks selects against the final declarations, including forward-declared
/// wires. A Verilog part-select cannot reverse the order of the selected bits.
fn validate_part_select_directions(
    reference: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    span: Span,
) -> Result<(), ScanError> {
    match reference {
        NetRef::PartSelect(index, left, right) if left != right => {
            let (declared_left, declared_right) = nets[index.0].width.unwrap_or((0, 0));
            if left.cmp(right) != declared_left.cmp(&declared_right) {
                return Err(ScanError {
                    message: format!(
                        "part-select '{}' does not follow its declared packed range [{declared_left}:{declared_right}]",
                        bit_ref::render_net_ref(reference, nets, interner),
                    ),
                    span,
                });
            }
        }
        NetRef::Concat(parts) => {
            for part in parts {
                validate_part_select_directions(part, nets, interner, span)?;
            }
        }
        _ => {
            // Scalar references and singleton slices have no select direction.
        }
    }
    Ok(())
}

/// Requires every GV assignment to be pure wiring, leaving logic to cells.
pub(crate) fn validate_gv_form(
    modules: &[NetlistModule],
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    check_proof_wiring: bool,
) -> Result<(), ScanError> {
    for module in modules {
        for assign in &module.assigns {
            let AssignExpr::Leaf(rhs) = &assign.rhs else {
                return Err(ScanError {
                    message: format!(
                        "GV permits only wiring in assignments; assignment to '{}' computes Boolean logic. Represent the logic with cells, or use ugv2aig for a cell-free UGV netlist",
                        bit_ref::render_net_ref(&assign.lhs, nets, interner)
                    ),
                    span: assign.span,
                });
            };
            if check_proof_wiring {
                for terminal in [&assign.lhs, rhs] {
                    validate_part_select_directions(terminal, nets, interner, assign.span)?;
                    if assign.kind == NetlistAssignKind::Tran {
                        let width = bit_ref::net_ref_width_bits(terminal, nets, interner).map_err(
                            |error| ScanError {
                                message: error.to_string(),
                                span: assign.span,
                            },
                        )?;
                        if width != 1 {
                            return Err(ScanError {
                                message: format!(
                                    "plain tran terminal '{}' has width {width}. Use one tran primitive per bit",
                                    bit_ref::render_net_ref(terminal, nets, interner),
                                ),
                                span: assign.span,
                            });
                        }
                    }
                }
            }
        }
        if check_proof_wiring {
            for instance in &module.instances {
                let pos = Pos {
                    lineno: instance.inst_lineno,
                    colno: instance.inst_colno,
                };
                let span = Span {
                    start: pos,
                    limit: pos,
                };
                for (_, terminal) in &instance.connections {
                    validate_part_select_directions(terminal, nets, interner, span)?;
                }
            }
        }
    }
    Ok(())
}

/// Requires continuous UGV assignments and instances defined in the same input.
pub(crate) fn validate_ugv_form(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<(), ScanError> {
    let module_names = modules
        .iter()
        .map(|module| module.name)
        .collect::<HashSet<_>>();
    for module in modules {
        for assign in &module.assigns {
            if assign.kind == NetlistAssignKind::Tran {
                return Err(ScanError {
                    message: "UGV does not support bidirectional tran connections; use continuous assignments for directional wiring".to_string(),
                    span: assign.span,
                });
            }
        }
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
