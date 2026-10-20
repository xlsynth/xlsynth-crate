// SPDX-License-Identifier: Apache-2.0

//! One sparse edit plan, interpreted by both the emitted netlist and STA.

use super::*;
use crate::netlist::normalized::BitSource;
use crate::netlist::sta::incremental::{TimingEdit, TimingInstance, TimingMutation};

/// A replacement or inserted instance; raw bindings retain aliases and pin
/// order.
#[derive(Clone)]
struct Instance {
    cell: usize,
    template: Option<usize>,
    connections: Vec<(String, NetRef)>,
}

/// Each inserted instance owns one new output wire, allocated in insertion
/// order.
#[derive(Clone)]
pub(super) struct EditPlan {
    changed: BTreeMap<usize, Instance>,
    added: Vec<Instance>,
    first_net: usize,
}
impl EditPlan {
    /// Builds the fanout topology once without cloning the module or interner.
    pub(super) fn new(
        netlist: &MappedNetlist,
        graph: &Graph,
        driver: &Driver,
        kind: &FanoutEdit,
        cells: &[CellChoice],
    ) -> Result<Self> {
        let node = |i: usize| -> Result<Instance> {
            Ok(Instance {
                cell: graph.cells[i],
                template: Some(i),
                connections: netlist.module.instances[i]
                    .connections
                    .iter()
                    .map(|(p, r)| Ok((resolve_symbol(&netlist.interner, *p, "pin")?, r.clone())))
                    .collect::<Result<_>>()?,
            })
        };
        let mut plan = Self {
            changed: BTreeMap::new(),
            added: vec![],
            first_net: netlist.nets.len(),
        };
        let original = node(driver.instance)?;
        let mut insert = |mut instance: Instance, output: usize, sinks: &[Sink]| -> Result<usize> {
            let net = plan.first_net + plan.added.len();
            instance.connections[output].1 = NetRef::Simple(NetIndex(net));
            plan.added.push(instance);
            for sink in sinks {
                if !plan.changed.contains_key(&sink.instance) {
                    plan.changed.insert(sink.instance, node(sink.instance)?);
                }
                plan.changed.get_mut(&sink.instance).unwrap().connections[sink.connection].1 =
                    NetRef::Simple(NetIndex(net));
            }
            Ok(net)
        };
        match kind {
            FanoutEdit::Upsize { cell } => {
                let new = cells
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("missing replacement"))?;
                let mut replacement = original;
                replacement.cell = new.index;
                replacement.connections[driver.output_connection].0 = new.output.clone();
                plan.changed.insert(driver.instance, replacement);
            }
            FanoutEdit::Replicate { groups } => {
                for part in partition(&driver.sinks, *groups, driver.fixed_output_load)
                    .iter()
                    .skip(1)
                {
                    if part.is_empty() {
                        continue;
                    }
                    insert(original.clone(), driver.output_connection, part)?;
                }
            }
            FanoutEdit::Buffer {
                cell,
                groups,
                levels,
            } => {
                let cell = cells
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("missing buffer"))?;
                ensure!(
                    cell.inputs.len() == 1 && cell.truth == 2,
                    "buffer must preserve polarity"
                );
                let buffer = |input| Instance {
                    cell: cell.index,
                    template: None,
                    connections: vec![
                        (cell.inputs[0].clone(), NetRef::Simple(NetIndex(input))),
                        (cell.output.clone(), NetRef::Unconnected),
                    ],
                };
                let mut parents = vec![driver.net; *groups];
                if *levels == 2 {
                    for parity in 0..2 {
                        let output = insert(buffer(driver.net), 1, &[])?;
                        for k in (parity..*groups).step_by(2) {
                            parents[k] = output;
                        }
                    }
                }
                for (k, part) in partition(&driver.sinks, *groups, 0.0).iter().enumerate() {
                    if !part.is_empty() {
                        insert(buffer(parents[k]), 1, part)?;
                    }
                }
            }
        }
        Ok(plan)
    }

    /// Applies the plan using the same wire/instance naming order as
    /// construction.
    pub(super) fn apply(&self, netlist: &mut MappedNetlist, library: &Library) -> Result<()> {
        let materialize = |node: &Instance, netlist: &mut MappedNetlist, added: bool| {
            let instance_name = if added {
                fresh_symbol(netlist)
            } else {
                netlist.module.instances[node.template.expect("replacement template")].instance_name
            };
            let mut instance = node
                .template
                .map(|i| netlist.module.instances[i].clone())
                .unwrap_or(NetlistInstance {
                    type_name: instance_name,
                    instance_name,
                    connections: vec![],
                    inst_lineno: 0,
                    inst_colno: 0,
                });
            instance.instance_name = instance_name;
            instance.type_name = netlist
                .interner
                .get_or_intern(&library.cells[node.cell].name);
            instance.connections = node
                .connections
                .iter()
                .map(|(name, reference)| (netlist.interner.get_or_intern(name), reference.clone()))
                .collect();
            instance
        };
        for (&i, node) in &self.changed {
            netlist.module.instances[i] = materialize(node, netlist, false);
        }
        for node in &self.added {
            let name = fresh_symbol(netlist);
            let index = netlist.nets.len();
            netlist.nets.push(Net { name, width: None });
            netlist.module.wires.push(NetIndex(index));
            netlist.module.net_index_range.end = netlist.nets.len();
            let instance = materialize(node, netlist, true);
            netlist.module.instances.push(instance);
        }
        Ok(())
    }

    /// Normalizes only the plan's bindings for the shared timing transaction.
    pub(super) fn timing_edit(
        &self,
        graph: &Graph,
        sta: &crate::netlist::sta::incremental::IncrementalSta<'_>,
    ) -> Result<TimingEdit> {
        let connections = |node: &Instance| -> Result<Vec<(String, BitSource)>> {
            node.connections
                .iter()
                .filter(|(_, r)| !matches!(r, NetRef::Unconnected))
                .map(|(name, r)| {
                    let source = match r {
                        NetRef::Simple(n) if n.0 >= self.first_net => BitSource::Bit(n.0),
                        NetRef::Simple(n) => match graph.roots[n.0] {
                            Some(root) => BitSource::Bit(root),
                            None => sta
                                .instance(
                                    node.template
                                        .expect("literal alias belongs to original instance"),
                                )
                                .and_then(|i| i.pins.get(name))
                                .and_then(|p| p.first())
                                .copied()
                                .ok_or_else(|| anyhow!("missing literal alias source"))?,
                        },
                        NetRef::Literal(v) => BitSource::Literal(v.get_bit(0)?),
                        _ => return Err(anyhow!("unsupported scalar plan connection")),
                    };
                    Ok((name.clone(), source))
                })
                .collect()
        };
        let mut mutations = self
            .changed
            .iter()
            .map(|(&instance, node)| {
                Ok(TimingMutation::Replace {
                    instance,
                    cell: node.cell,
                    connections: connections(node)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (j, node) in self.added.iter().enumerate() {
            let mut name = format!("__timing_fanout_{}", sta.instance_count() + j);
            while sta.contains_instance_name(&name) {
                name.push('_');
            }
            mutations.push(TimingMutation::Insert(TimingInstance {
                name,
                cell: node.cell,
                connections: connections(node)?,
            }));
        }
        Ok(TimingEdit {
            new_bits: self.added.len(),
            mutations,
        })
    }

    /// Updates the deterministic per-library area histogram for this plan.
    pub(super) fn counts(&self, counts: &[usize], old_cells: &[usize]) -> Vec<usize> {
        let mut counts = counts.to_vec();
        for (&i, node) in &self.changed {
            counts[old_cells[i]] -= 1;
            counts[node.cell] += 1;
        }
        for node in &self.added {
            counts[node.cell] += 1;
        }
        counts
    }
}
