// SPDX-License-Identifier: Apache-2.0

//! Simplify physical cell inputs after distinct cut leaves resolve to wires.

use super::{Emitter, literal, mask, variable};
use crate::aig::gate::AigNode;
use crate::netlist::parse::{AssignExpr, NetIndex, NetRef};
use std::collections::HashMap;

/// A cell whose truth table under the actual pin connections has support <= 1.
#[derive(Debug, PartialEq, Eq)]
enum Fold {
    Constant(bool),
    Wire(NetIndex),
    Inverted(NetIndex),
}

/// Restricts the physical truth table to assignments that the pins can receive.
fn fold_tied_inputs(table: u64, inputs: &[NetRef]) -> Option<Fold> {
    let n = inputs.len();
    let mut possible = mask(n);
    for (pin, input) in inputs.iter().enumerate() {
        match input {
            NetRef::Literal(bits) => {
                let value = variable(n, pin);
                possible &= if bits.get_bit(0).expect("one-bit constant") {
                    value
                } else {
                    !value
                };
            }
            NetRef::Simple(_) => {
                if let Some(other) = inputs[..pin].iter().position(|other| other == input) {
                    possible &= !(variable(n, pin) ^ variable(n, other));
                }
            }
            _ => unreachable!("the mapper only emits scalar wires and literals"),
        }
    }
    if possible == mask(n) {
        return None;
    }
    let result = table & possible;
    if result == 0 || result == possible {
        return Some(Fold::Constant(result != 0));
    }
    for (pin, input) in inputs.iter().enumerate() {
        if let NetRef::Simple(net) = input {
            let value = variable(n, pin) & possible;
            if result == value {
                return Some(Fold::Wire(*net));
            }
            if result == possible ^ value {
                return Some(Fold::Inverted(*net));
            }
        }
    }
    None
}

impl Emitter<'_> {
    /// Uses a constant, wire, or suitable inverter in place of a degenerate
    /// cell.
    pub(super) fn fold_cell(&mut self, index: usize, inputs: &[NetRef]) -> Option<NetRef> {
        let cell = &self.cover.lib.cells[index];
        let fold = fold_tied_inputs(cell.truth_table, inputs)?;
        let kept_net = match fold {
            Fold::Constant(_) => None,
            Fold::Wire(net) | Fold::Inverted(net) => Some(net),
        };
        let value = match fold {
            Fold::Constant(value) => literal(value),
            Fold::Wire(net) => NetRef::Simple(net),
            Fold::Inverted(net) => {
                let inv = self.cover.lib.inv?;
                let inverter = &self.cover.lib.cells[inv];
                if inverter.area > cell.area
                    || (self.cover.lib.delay_first
                        && !inputs.iter().zip(&cell.pin_delays).any(|(input, delay)| {
                            *input == NetRef::Simple(net) && *delay >= inverter.pin_delays[0]
                        }))
                {
                    return None;
                }
                self.cell(inv, vec![NetRef::Simple(net)])
            }
        };
        let port_count = self.result.netlist.module.ports.len();
        self.needs_pruning |= inputs.iter().any(|input| {
            matches!(input, NetRef::Simple(net) if net.0 >= port_count && Some(*net) != kept_net)
        });
        Some(value)
    }

    /// Drops only cells whose fanout disappeared. Emission is in dependency
    /// order, with one wire per cell, following all port nets.
    pub(super) fn prune_dead_cells(&mut self) {
        let mapped = &mut self.result.netlist;
        let mut live = vec![false; mapped.nets.len()];
        live[..mapped.module.ports.len()].fill(true);
        for assign in &mapped.module.assigns {
            if let AssignExpr::Leaf(NetRef::Simple(net)) = assign.rhs {
                live[net.0] = true;
            }
        }
        for (output, inst) in mapped
            .module
            .wires
            .iter()
            .zip(&mapped.module.instances)
            .rev()
        {
            if live[output.0] {
                for (_, pin) in &inst.connections {
                    if let NetRef::Simple(net) = pin {
                        live[net.0] = true;
                    }
                }
            }
        }
        if live.iter().all(|&used| used) {
            return;
        }
        let stats = &mut self.result.stats;
        if self.cover.lib.inv.is_some() {
            stats.shared_opposite_polarities = self
                .signals
                .iter()
                .filter(|(op, signal)| {
                    let id = op.node.id;
                    self.cover.boundary[id]
                        && op.negated != self.cover.anchors[id]
                        && matches!(self.cover.aig.gates[id], AigNode::And2 { .. })
                        && matches!(signal, NetRef::Simple(net) if live[net.0])
                })
                .count();
        }
        let mut next = 0;
        let relocated: Vec<_> = live
            .iter()
            .map(|&used| {
                let index = NetIndex(next);
                next += usize::from(used);
                index
            })
            .collect();
        let mut retained_nets = live.iter();
        mapped.nets.retain(|_| *retained_nets.next().unwrap());
        mapped.module.wires.retain_mut(|net| {
            let used = live[net.0];
            *net = relocated[net.0];
            used
        });
        let cell_indices: HashMap<_, _> = self
            .cover
            .lib
            .cells
            .iter()
            .enumerate()
            .map(|(index, c)| (c.name.as_str(), index))
            .collect();
        stats.cell_area = 0.0;
        stats.cell_count = 0;
        stats.inverter_count = 0;
        stats.cells_by_type.clear();
        let mut retained_instances = live[mapped.module.ports.len()..].iter();
        mapped.module.instances.retain_mut(|inst| {
            if !retained_instances.next().unwrap() {
                return false;
            }
            for (_, pin) in &mut inst.connections {
                if let NetRef::Simple(net) = pin {
                    *net = relocated[net.0];
                }
            }
            let name = mapped.interner.resolve(inst.type_name).unwrap();
            stats.record_cell(cell_indices[name], self.cover.lib);
            true
        });
        for assign in &mut mapped.module.assigns {
            if let AssignExpr::Leaf(NetRef::Simple(net)) = &mut assign.rhs {
                *net = relocated[net.0];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricts_physical_truth_table_to_possible_assignments() {
        let x = NetRef::Simple(NetIndex(1));
        let y = NetRef::Simple(NetIndex(2));
        let (a, b, c) = (variable(3, 0), variable(3, 1), variable(3, 2));
        assert_eq!(
            fold_tied_inputs(a ^ b ^ c, &[x.clone(), y.clone(), x.clone()]),
            Some(Fold::Wire(NetIndex(2)))
        );
        assert_eq!(
            fold_tied_inputs(!(a ^ b ^ c), &[x.clone(), y.clone(), x.clone()]),
            Some(Fold::Inverted(NetIndex(2)))
        );
        assert_eq!(
            fold_tied_inputs(a & b & c, &[x.clone(), literal(false), y.clone()]),
            Some(Fold::Constant(false))
        );
        assert_eq!(
            fold_tied_inputs(a | b | c, &[x.clone(), literal(true), y.clone()]),
            Some(Fold::Constant(true))
        );
        assert_eq!(fold_tied_inputs(a & b & c, &[x.clone(), y, x]), None);
    }
}
