// SPDX-License-Identifier: Apache-2.0

//! Exact electrical wiring for a normalized netlist snapshot. Unlike a logical
//! cone, a wire alias shares physical load with its source. Expression assigns,
//! repeated destinations and cycles are recorded as issues rather than silently
//! treated as physical wires. Rebuild this view after changing connectivity.

use super::super::normalized::{
    BitExpr, BitSource, NormalizedBitIndex, NormalizedNetlistModule, find_alias_root,
};
use super::super::parse::{InstIndex, PortDirection};
use super::PinConnection;
use crate::liberty_model::PinDirection;
use std::collections::{HashMap, HashSet};

/// Why a destination, or the sources referenced by it, cannot be assumed to
/// describe one ordinary electrical wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiringIssueKind {
    RepeatedDestination,
    AliasCycle,
    Expression,
}

/// An issue and the electrical group it affects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WiringIssue {
    pub bit: NormalizedBitIndex,
    pub kind: WiringIssueKind,
}

/// Exact pin occupants of an electrical group. Unknown directions retain their
/// binding positions so an editing client can reject incomplete evidence.
#[derive(Default, Debug)]
pub struct SignalPins {
    pub drivers: Vec<PinConnection>,
    pub sinks: Vec<PinConnection>,
    pub unknown: Vec<PinConnection>,
}

/// Sparse index of the requested electrical groups, preserving netlist order.
#[derive(Default, Debug)]
pub struct PinIndex {
    signals: HashMap<NormalizedBitIndex, SignalPins>,
}
impl PinIndex {
    pub fn signal(&self, group: NormalizedBitIndex) -> Option<&SignalPins> {
        self.signals.get(&group)
    }
}

/// Alias groups, exact assign destinations and literal sources for an unchanged
/// normalized snapshot. A directed `assign b = a` groups the wire load but
/// still records a write to `b`; timing and Boolean clients must preserve that
/// direction when resolving signal values.
pub struct ExactConnectivity<'n, 'm> {
    normalized: &'n NormalizedNetlistModule<'m>,
    groups: Vec<NormalizedBitIndex>,
    pub assigned: Vec<bool>,
    pub issues: Vec<WiringIssue>,
    pub literals: Vec<(NormalizedBitIndex, bool)>,
    duplicate_instances: HashSet<InstIndex>,
    pub output_ports: HashMap<NormalizedBitIndex, usize>,
    pub input_groups: HashSet<NormalizedBitIndex>,
}
impl<'n, 'm> ExactConnectivity<'n, 'm> {
    pub fn new(normalized: &'n NormalizedNetlistModule<'m>) -> Self {
        let mut groups = normalized.canonical_bits().to_vec();
        let mut assigned = vec![false; normalized.bit_count()];
        let mut issues = Vec::new();
        let mut literals = Vec::new();
        for assign in &normalized.assigns {
            for (&dst, expr) in assign.lhs_bits.iter().zip(&assign.rhs_bits) {
                if std::mem::replace(&mut assigned[dst.index()], true) {
                    issues.push(WiringIssue {
                        bit: dst,
                        kind: WiringIssueKind::RepeatedDestination,
                    });
                }
                match expr {
                    BitExpr::Source(BitSource::Bit(src)) => {
                        let a = find_alias_root(&mut groups, dst);
                        let b = find_alias_root(&mut groups, *src);
                        if a == b {
                            issues.push(WiringIssue {
                                bit: dst,
                                kind: WiringIssueKind::AliasCycle,
                            });
                        } else {
                            groups[a.index()] = b;
                        }
                    }
                    BitExpr::Source(BitSource::Literal(value)) => literals.push((dst, *value)),
                    _ => {
                        issues.push(WiringIssue {
                            bit: dst,
                            kind: WiringIssueKind::Expression,
                        });
                        let mut sources = Vec::new();
                        expr.collect_source_bits(&mut sources);
                        issues.extend(sources.into_iter().map(|bit| WiringIssue {
                            bit,
                            kind: WiringIssueKind::Expression,
                        }));
                    }
                }
            }
        }
        for bit in 0..groups.len() {
            groups[bit] = find_alias_root(&mut groups, NormalizedBitIndex::new(bit));
        }
        for issue in &mut issues {
            issue.bit = groups[issue.bit.index()];
        }
        for (bit, _) in &mut literals {
            *bit = groups[bit.index()];
        }
        let mut output_ports = HashMap::new();
        let mut input_groups = HashSet::new();
        for port in &normalized.ports {
            for &bit in &port.bits {
                match port.direction {
                    PortDirection::Output => {
                        *output_ports.entry(groups[bit.index()]).or_default() += 1
                    }
                    PortDirection::Input | PortDirection::Inout => {
                        input_groups.insert(groups[bit.index()]);
                    }
                }
            }
        }
        let mut duplicate_instances = HashSet::new();
        let mut seen = HashSet::new();
        for inst in &normalized.instances {
            seen.clear();
            for pin in &inst.connections {
                if !seen.insert(pin.port) {
                    duplicate_instances.insert(inst.raw_index);
                }
            }
        }
        Self {
            normalized,
            groups,
            assigned,
            issues,
            literals,
            duplicate_instances,
            output_ports,
            input_groups,
        }
    }
    pub fn has_duplicate_pin(&self, instance: InstIndex) -> bool {
        self.duplicate_instances.contains(&instance)
    }

    pub fn group(&self, bit: NormalizedBitIndex) -> NormalizedBitIndex {
        self.groups[bit.index()]
    }

    /// Index only selected signals, avoiding a full adjacency allocation for
    /// transforms interested in a small driver population. `direction` must
    /// return None for unknown masters or pins; those bindings are retained.
    pub fn index_pins(
        &self,
        selected: &HashSet<NormalizedBitIndex>,
        mut direction: impl FnMut(PinConnection) -> Option<PinDirection>,
    ) -> PinIndex {
        let mut index = PinIndex::default();
        for inst in &self.normalized.instances {
            for (connection, pin) in inst.connections.iter().enumerate() {
                let id = PinConnection {
                    instance: inst.raw_index,
                    connection,
                };
                for source in &pin.bits {
                    if let BitSource::Bit(bit) = source {
                        let group = self.group(*bit);
                        if selected.contains(&group) {
                            let pins = index.signals.entry(group).or_default();
                            match direction(id) {
                                Some(PinDirection::Output) => pins.drivers.push(id),
                                Some(PinDirection::Input) => pins.sinks.push(id),
                                _ => pins.unknown.push(id),
                            }
                        }
                    }
                }
            }
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::io::{read_gv_from_str, read_ugv_from_str};
    use anyhow::Result;

    /// An alias carries both of its physical pin bindings; pin positions, not
    /// just port names, distinguish malformed duplicate bindings.
    #[test]
    fn aliases_keep_exact_driver_sink_and_unknown_bindings() -> Result<()> {
        let parsed = read_gv_from_str(
            "module top(input a, output y); wire q, alias; assign alias = q; DFF r(.D(a), .Q(q)); AND g(.A(alias), .A(q), .Y(y)); UNKNOWN u(.P(alias)); endmodule",
        )?;
        let n = NormalizedNetlistModule::new(&parsed.modules[0], &parsed.nets, &parsed.interner)?;
        let wire = ExactConnectivity::new(&n);
        let q = match n.instances[0].connections[1].bits[0] {
            BitSource::Bit(bit) => wire.group(bit),
            _ => unreachable!(),
        };
        let index = wire.index_pins(&HashSet::from([q]), |id| {
            match (id.instance.0, id.connection) {
                (0, 1) => Some(PinDirection::Output),
                (1, 0 | 1) => Some(PinDirection::Input),
                _ => None,
            }
        });
        let signal = index.signal(q).unwrap();
        assert_eq!(
            signal.drivers,
            [PinConnection {
                instance: InstIndex(0),
                connection: 1
            }]
        );
        assert_eq!(signal.sinks.len(), 2);
        assert_eq!(signal.unknown.len(), 1);
        assert_ne!(signal.sinks[0], signal.sinks[1]);
        assert!(wire.has_duplicate_pin(signal.sinks[1].instance));
        Ok(())
    }

    /// A logical expression or cycle must never quietly become a safe wire.
    #[test]
    fn cyclic_and_expression_assigns_remain_visible() -> Result<()> {
        let parsed = read_ugv_from_str(
            "module top(input a, output y); wire p, q; assign p=q; assign q=p; assign y=~a; endmodule",
        )?;
        let n = NormalizedNetlistModule::new(&parsed.modules[0], &parsed.nets, &parsed.interner)?;
        let wire = ExactConnectivity::new(&n);
        assert!(
            wire.issues
                .iter()
                .any(|i| i.kind == WiringIssueKind::AliasCycle)
        );
        assert!(
            wire.issues
                .iter()
                .any(|i| i.kind == WiringIssueKind::Expression)
        );
        Ok(())
    }
}
