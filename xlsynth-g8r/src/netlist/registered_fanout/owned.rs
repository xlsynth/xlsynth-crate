// SPDX-License-Identifier: Apache-2.0

//! Ownership of the identity buffers inserted into one candidate module.
//! Symbols survive cell resizing and instance-vector compaction. Sink wiring
//! and cell sizes are always read from the candidate itself.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use anyhow::{Result, ensure};

use super::graph::{self, Root};
use crate::liberty_model::Library;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{NetRef, PortId};

/// Candidate-local ledger of buffer instances and the output pins they serve.
/// Clone it with a netlist, and retain the pair together when accepting an
/// edit.
#[derive(Clone, Debug, Default)]
pub struct OwnedBuffers {
    drivers: HashMap<PortId, (PortId, PortId)>,
    roots: HashSet<(PortId, PortId)>,
}

impl OwnedBuffers {
    /// Number of buffer instances in the current candidate's owned trees.
    pub fn len(&self) -> usize {
        self.drivers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.drivers.is_empty()
    }

    /// Whether the named instance output owns a tree in this candidate. Both
    /// symbols are interned in the corresponding netlist.
    pub fn contains_root(&self, root: (PortId, PortId)) -> bool {
        self.roots.contains(&root)
    }

    pub(super) fn root_of(&self, buffer: PortId) -> Option<(PortId, PortId)> {
        self.drivers.get(&buffer).copied()
    }

    /// A tree inserted at an existing branch belongs to the original driver.
    pub(super) fn record(&mut self, net: &MappedNetlist, root: &Root, added: Range<usize>) {
        if added.is_empty() {
            return;
        }
        let instance = &net.module.instances[root.instance];
        let driver = self.root_of(instance.instance_name).unwrap_or((
            instance.instance_name,
            instance.connections[root.connection].0,
        ));
        self.roots.insert(driver);
        for branch in &net.module.instances[added] {
            self.drivers.insert(branch.instance_name, driver);
        }
    }

    /// Detach complete selected trees in the trial. Skip a root whose branches
    /// have gained a port, alias, control pin, foreign driver or unsupported
    /// load. Its buffers and ownership stay intact. Other roots can proceed.
    #[cfg(test)]
    pub(super) fn flatten(
        &mut self,
        net: &mut MappedNetlist,
        lib: &Library,
        wanted: &HashSet<(PortId, PortId)>,
    ) -> Result<usize> {
        if !wanted.iter().any(|root| self.contains_root(*root)) {
            return Ok(0);
        }
        let live = graph::collect_roots(net, lib, 0.0, Some(1))?;
        self.flatten_with_graph(net, lib, &live, wanted)
    }

    /// `live` must include every movable single-pin connection in the same
    /// starting netlist, or an unmodified clone. It can be reused while
    /// evaluating independent alternatives.
    pub(super) fn flatten_with_graph(
        &mut self,
        net: &mut MappedNetlist,
        lib: &Library,
        live: &[Root],
        wanted: &HashSet<(PortId, PortId)>,
    ) -> Result<usize> {
        let counts = self
            .drivers
            .values()
            .filter(|root| wanted.contains(*root))
            .fold(HashMap::new(), |mut counts, &root| {
                *counts.entry(root).or_insert(0usize) += 1;
                counts
            });
        if counts.is_empty() {
            return Ok(0);
        }
        let instances = &net.module.instances;
        let mut sources = Vec::new();
        let mut branches = HashMap::new();
        let mut buf_pins = HashMap::new();
        let mut references = Vec::new();
        for assign in &net.module.assigns {
            assign.lhs.collect_net_indices(&mut references);
            assign.rhs.collect_net_indices(&mut references);
        }
        let aliased: HashSet<_> = references.into_iter().collect();
        for root in live {
            let inst = &instances[root.instance];
            let key = (inst.instance_name, inst.connections[root.connection].0);
            if counts.contains_key(&key) {
                sources.push((key, root));
            }
            if !self
                .root_of(inst.instance_name)
                .is_some_and(|r| counts.contains_key(&r))
                || root.fixed_sink_count != 0
                || root.output_count != 0
            {
                continue;
            }
            let NetRef::Simple(wire) = root.source else {
                continue;
            };
            if aliased.contains(&wire) {
                continue;
            }
            let pins = buf_pins.entry(root.cell).or_insert_with(|| {
                let b = graph::buffer_model(lib, root.cell)?;
                Some((net.interner.get(&b.input)?, net.interner.get(&b.output)?))
            });
            if let Some((input, output)) = *pins
                && key.1 == output
            {
                branches.insert(root.instance, (root, input, wire));
            }
        }
        // Validate connectivity and all branch uses before rewiring. Direct
        // sinks of the unchanged root (including fixed pins) retain their
        // binds.
        let mut remove = HashSet::new();
        let mut wires = HashSet::new();
        let mut rewires = Vec::new();
        let mut flattened = Vec::new();
        for (key, source) in sources {
            let mut active = vec![source];
            let mut visited = HashSet::new();
            let mut terminals = Vec::new();
            let mut complete = true;
            while let Some(parent) = active.pop() {
                for sink in &parent.sinks {
                    let endpoint = &instances[sink.instance];
                    let binding = &endpoint.connections[sink.connection];
                    ensure!(
                        binding.1 == sink.original,
                        "stale sink in buffer-replacement graph"
                    );
                    if self.root_of(endpoint.instance_name) == Some(key) {
                        let Some(&(child, input, _)) = branches.get(&sink.instance) else {
                            complete = false;
                            break;
                        };
                        if binding.0 != input || !visited.insert(sink.instance) {
                            complete = false;
                            break;
                        }
                        active.push(child);
                    } else if parent.instance != source.instance {
                        terminals.push((sink.instance, sink.connection, source.source.clone()));
                    }
                }
                if !complete {
                    break;
                }
            }
            if complete && visited.len() == counts[&key] {
                wires.extend(visited.iter().map(|i| branches[i].2));
                remove.extend(visited.into_iter().map(|i| instances[i].instance_name));
                rewires.extend(terminals);
                flattened.push(key);
            }
        }
        for (inst, pin, source) in rewires {
            net.module.instances[inst].connections[pin].1 = source;
        }
        net.module
            .instances
            .retain(|i| !remove.contains(&i.instance_name));
        // Keep the backing net indices stable for surviving ports, selects,
        // assigns and subsequent modules; drop only unused declarations.
        net.module.wires.retain(|wire| !wires.contains(wire));
        self.drivers.retain(|b, _| !remove.contains(b));
        for key in flattened {
            self.roots.remove(&key);
        }
        Ok(remove.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::parse::{AssignExpr, NetlistAssign, NetlistAssignKind};
    use crate::netlist::registered_fanout::{
        edit,
        tests::{library, read_mapped_netlist},
    };

    // The original gates include a user-authored buffer at a name deliberately
    // resembling generated names. Only instances recorded by the editor can go.
    // Flattening restores each movable sink to its original source. A buffer
    // branch used by a module output must survive.
    #[test]
    fn flatten_preserves_original_sink_pins_and_skips_observable_branches() -> Result<()> {
        let lib = library()?;
        let mut n = read_mapped_netlist(
            "module top(input clk, input d, output [2:0] y, output expose);
             wire q; wire tapped; wire spare; assign expose = q;
             DFF1 ff(.D(d),.CLK(clk),.Q(q));
             BUF4 g8r_data_fanout_buf_42(.A(q),.Y(tapped));
             INV1 a(.A(q),.Y(y[0])); INV1 b(.A(q),.Y(y[1]));
             INV1 c(.A(tapped),.Y(y[2]));
             endmodule",
        )?;
        let original = n.module.clone();
        let root = graph::roots(&n, &lib, 0.0)?
            .into_iter()
            .find(|r| r.instance == 0)
            .unwrap();
        let key = (
            n.module.instances[0].instance_name,
            n.module.instances[0].connections[root.connection].0,
        );
        let buf = graph::buffer_model(
            &lib,
            lib.cells.iter().position(|c| c.name == "BUF4").unwrap(),
        )
        .unwrap();
        let branches = [
            edit::Branch {
                parent: None,
                sinks: vec![],
            },
            edit::Branch {
                parent: Some(0),
                sinks: vec![0, 1, 2],
            },
        ];
        let before = n.module.instances.len();
        let mut owned = OwnedBuffers::default();
        edit::apply_tree_tracked(&mut n, &lib, &root, &buf, &branches, &mut owned)?;
        assert_eq!(owned.len(), 2);
        assert_eq!(
            owned.root_of(n.module.instances[before + 1].instance_name),
            Some(key)
        );
        let wanted = HashSet::from([key]);

        let mut protected = n.clone();
        let out = protected.module.instances[before + 1].connections[1]
            .1
            .clone();
        let spare = protected
            .module
            .find_net_index(protected.interner.get("spare").unwrap(), &protected.nets)
            .unwrap();
        protected.module.assigns.push(NetlistAssign {
            kind: NetlistAssignKind::Continuous,
            lhs: NetRef::Simple(spare),
            rhs: AssignExpr::Leaf(out),
            span: protected.module.assigns[0].span.clone(),
        });
        let same = protected.module.clone();
        let mut keep = owned.clone();
        assert_eq!(keep.flatten(&mut protected, &lib, &wanted)?, 0);
        assert_eq!(protected.module, same);
        assert_eq!(keep.len(), 2);

        // Only one pin is now attached directly to the register Q.
        assert_eq!(owned.flatten(&mut n, &lib, &wanted)?, 2);
        assert!(owned.is_empty());
        assert_eq!(n.module.instances, original.instances);
        assert_eq!(n.module.assigns, original.assigns);
        assert_eq!(n.module.wires, original.wires);
        Ok(())
    }
}
