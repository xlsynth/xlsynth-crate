// SPDX-License-Identifier: Apache-2.0

//! Apply a buffer tree to the exact data-pin connections selected by the graph.

use super::graph::{Buffer, Root, buffer_model};
use crate::liberty_model::Library;
use crate::netlist::bit_ref::{net_ref_lsb_targets, net_ref_width_bits};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance};
use anyhow::{Result, anyhow, ensure};
use std::collections::HashSet;

/// One new buffer. Its input comes from the original source or an earlier
/// branch's output. Sink numbers index the graph root's movable pin loads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Branch {
    pub(super) parent: Option<usize>,
    pub(super) sinks: Vec<usize>,
}

/// Validate every tree before altering the module. Each first tuple element is
/// an index into the supplied graph roots. Parent and sink indices are local
/// to one tree. Ports, aliases, fixed loads and unlisted taps retain their
/// original connections, including when two roots share a sink instance.
pub(super) fn apply_forest(
    netlist: &mut MappedNetlist,
    library: &Library,
    buffer: &Buffer,
    roots: &[Root],
    plans: &[(usize, Vec<Branch>)],
) -> Result<()> {
    if let [(index, branches)] = plans {
        let root = roots
            .get(*index)
            .ok_or_else(|| anyhow!("unknown tree root {index}"))?;
        return apply_tree(netlist, library, root, buffer, branches);
    }
    let selected = plans
        .iter()
        .map(|(index, branches)| {
            let root = roots
                .get(*index)
                .ok_or_else(|| anyhow!("unknown tree root {index}"))?;
            Ok((root, branches.as_slice()))
        })
        .collect::<Result<Vec<_>>>()?;
    apply_plans(netlist, library, buffer, &selected)
}

/// Apply a single tree without allocating a list of selected roots.
pub(super) fn apply_tree(
    netlist: &mut MappedNetlist,
    library: &Library,
    root: &Root,
    buffer: &Buffer,
    branches: &[Branch],
) -> Result<()> {
    apply_plans(netlist, library, buffer, &[(root, branches)])
}

pub(super) fn apply_tree_tracked(
    netlist: &mut MappedNetlist,
    library: &Library,
    root: &Root,
    buffer: &Buffer,
    branches: &[Branch],
    owned: &mut super::owned::OwnedBuffers,
) -> Result<()> {
    let first = netlist.module.instances.len();
    apply_tree(netlist, library, root, buffer, branches)?;
    owned.record(netlist, root, first..netlist.module.instances.len());
    Ok(())
}

/// Graph discovery certifies driver/sink roles and bit-level alias equivalence.
/// Recheck their physical bindings and tree topology before allocating names.
/// Empty branches may drive child buffers. An entirely empty forest is a no-op.
fn apply_plans(
    netlist: &mut MappedNetlist,
    library: &Library,
    buffer: &Buffer,
    trees: &[(&Root, &[Branch])],
) -> Result<()> {
    let count = trees.iter().try_fold(0_usize, |count, (_, branches)| {
        count
            .checked_add(branches.len())
            .ok_or_else(|| anyhow!("branch count overflow"))
    })?;
    let mut drivers = HashSet::with_capacity(trees.len());
    for &(root, _) in trees {
        ensure!(
            drivers.insert((root.instance, root.connection)),
            "repeated tree driver"
        );
    }
    if count == 0 {
        return Ok(());
    }
    let cell = library
        .cells
        .get(buffer.cell)
        .ok_or_else(|| anyhow!("unknown buffer cell {}", buffer.cell))?;
    let model = buffer_model(library, buffer.cell)
        .ok_or_else(|| anyhow!("cell {} is not a legal positive data buffer", buffer.cell))?;
    ensure!(
        model.input == buffer.input && model.output == buffer.output,
        "buffer pins disagree with the library"
    );
    let module = &netlist.module;
    let range = &module.net_index_range;
    ensure!(
        range.start <= range.end && range.end <= netlist.nets.len(),
        "invalid module net range"
    );
    let check_binding = |instance: usize, connection: usize, original: &NetRef| -> Result<()> {
        let (_, current) = module
            .instances
            .get(instance)
            .and_then(|cell| cell.connections.get(connection))
            .ok_or_else(|| {
                anyhow!("missing tree endpoint at instance {instance}, connection {connection}")
            })?;
        ensure!(
            current == original,
            "stale tree endpoint at instance {instance}, connection {connection}"
        );
        ensure!(
            net_ref_width_bits(current, &netlist.nets, &netlist.interner)? == 1,
            "expected a scalar endpoint"
        );
        let targets = net_ref_lsb_targets(current, &netlist.nets, &netlist.interner)?;
        ensure!(
            matches!(targets.as_slice(), [bit] if range.contains(&bit.net.0)),
            "expected a connected signal in this module"
        );
        Ok(())
    };
    let mut taps = HashSet::new();
    for &(root, branches) in trees {
        check_binding(root.instance, root.connection, &root.source)?;
        let driver = &module.instances[root.instance];
        let driver_type = library
            .cells
            .get(root.cell)
            .ok_or_else(|| anyhow!("unknown driver cell"))?;
        ensure!(
            netlist.interner.resolve(driver.instance_name) == Some(root.instance_name.as_str())
                && netlist.interner.resolve(driver.type_name) == Some(driver_type.name.as_str())
                && netlist
                    .interner
                    .resolve(driver.connections[root.connection].0)
                    == Some(root.pin_name.as_str()),
            "stale tree driver identity"
        );
        let mut used = vec![false; branches.len()];
        for (i, branch) in branches.iter().enumerate() {
            if let Some(parent) = branch.parent {
                ensure!(parent < i, "branch {i} must follow its parent");
                used[parent] = true;
            }
            for &index in &branch.sinks {
                let sink = root
                    .sinks
                    .get(index)
                    .ok_or_else(|| anyhow!("unknown sink {index}"))?;
                let tap = (sink.instance, sink.connection);
                ensure!(
                    tap != (root.instance, root.connection),
                    "cannot rewire the driver"
                );
                ensure!(taps.insert(tap), "pin appears twice in the tree");
                check_binding(sink.instance, sink.connection, &sink.original)?;
                used[i] = true;
            }
        }
        ensure!(
            used.iter().all(|&has_load| has_load),
            "tree contains a buffer without a load"
        );
    }
    ensure!(drivers.is_disjoint(&taps), "cannot rewire a forest driver");
    // New nets occupy consecutive indices at the end of this module's range.
    // A parsed file can retain unused net declarations from a later module.
    let first = range.end;
    let end = first
        .checked_add(count)
        .ok_or_else(|| anyhow!("net index overflow"))?;
    let cell_type = netlist.interner.get_or_intern(&cell.name);
    let input_pin = netlist.interner.get_or_intern(&buffer.input);
    let output_pin = netlist.interner.get_or_intern(&buffer.output);
    let mut ordinal = module.instances.len();
    let mut wires = Vec::with_capacity(count);
    netlist.module.wires.reserve(count);
    netlist.module.instances.reserve(count);
    for &(root, branches) in trees {
        let tree_first = first + wires.len();
        for (i, branch) in branches.iter().enumerate() {
            let (wire_name, cell_name) = fresh_names(netlist, &mut ordinal);
            let input = match branch.parent {
                Some(parent) => NetRef::Simple(NetIndex(tree_first + parent)),
                None => root.source.clone(),
            };
            let output = NetRef::Simple(NetIndex(tree_first + i));
            wires.push(Net {
                name: wire_name,
                width: None,
            });
            netlist.module.wires.push(NetIndex(tree_first + i));
            netlist.module.instances.push(NetlistInstance {
                instance_name: cell_name,
                type_name: cell_type,
                connections: vec![(input_pin, input), (output_pin, output.clone())],
                inst_lineno: 0,
                inst_colno: 0,
            });
            for &index in &branch.sinks {
                let sink = &root.sinks[index];
                netlist.module.instances[sink.instance].connections[sink.connection].1 =
                    output.clone();
            }
        }
    }
    netlist.nets.splice(first..first, wires);
    netlist.module.net_index_range.end = end;
    Ok(())
}

fn fresh_names(
    netlist: &mut MappedNetlist,
    ordinal: &mut usize,
) -> (crate::netlist::parse::PortId, crate::netlist::parse::PortId) {
    let interner = &mut netlist.interner;
    loop {
        let wire = format!("g8r_data_fanout_wire_{ordinal}");
        let cell = format!("g8r_data_fanout_buf_{ordinal}");
        *ordinal += 1;
        if interner.get(&wire).is_none() && interner.get(&cell).is_none() {
            return (interner.get_or_intern(wire), interner.get_or_intern(cell));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        graph::roots,
        tests::{library, read_mapped_netlist},
    };
    use super::*;

    fn example() -> Result<(MappedNetlist, Library, Root, Buffer)> {
        let lib = library()?;
        let text =
            "module top(input clk, input d, output [3:0] out, output g8r_data_fanout_wire_5);
            wire q;
            wire [3:3] alias_wire;
            assign alias_wire[3] = q;
            assign g8r_data_fanout_wire_5 = q;
            DFF1 launch(.D(d), .CLK(clk), .Q(q));
            INV1 g0(.A(alias_wire[3]), .Y(out[0]));
            INV1 g1(.A(q), .Y(out[1]));
            INV1 g2(.A(q), .Y(out[2]));
            INV1 g3(.A(q), .Y(out[3]));
        endmodule
        module sibling(input extra, output extra_alias);
            assign extra_alias = extra;
        endmodule";
        let netlist = read_mapped_netlist(text)?;
        let root = roots(&netlist, &lib, 1.0)?
            .into_iter()
            .find(|root| root.instance == 0)
            .expect("launch Q");
        assert_eq!(root.sinks.len(), 4);
        let index = lib.cells.iter().position(|c| c.name == "BUF4").unwrap();
        let buffer = buffer_model(&lib, index).expect("ordinary buffer");
        Ok((netlist, lib, root, buffer))
    }

    // Aliases share the load seen by the driver. Rewiring one selected data pin
    // must preserve other uses of the aliased net.
    #[test]
    fn branches_rewire_selected_pins_and_preserve_fixed_aliases() -> Result<()> {
        let (mut netlist, lib, root, buffer) = example()?;
        let original = netlist.clone();
        let first = original.module.net_index_range.end;
        let old_instances = original.module.instances.len();
        let alias = root
            .sinks
            .iter()
            .position(|s| s.original != root.source)
            .unwrap();
        let direct = root
            .sinks
            .iter()
            .position(|s| s.original == root.source)
            .unwrap();
        let branches = [
            Branch {
                parent: None,
                sinks: vec![],
            },
            Branch {
                parent: Some(0),
                sinks: vec![alias],
            },
            Branch {
                parent: None,
                sinks: vec![direct],
            },
        ];
        apply_tree(&mut netlist, &lib, &root, &buffer, &branches)?;
        assert_eq!(netlist.module.assigns, original.module.assigns);
        assert_eq!(netlist.module.ports, original.module.ports);
        assert_eq!(netlist.module.net_index_range.end, first + 3);
        assert_eq!(netlist.nets[first + 3..], original.nets[first..]);
        let mut expected = original.module.instances.clone();
        for (idx, branch) in [(alias, 1), (direct, 2)] {
            let sink = &root.sinks[idx];
            expected[sink.instance].connections[sink.connection].1 =
                NetRef::Simple(NetIndex(first + branch));
        }
        assert_eq!(netlist.module.instances[..old_instances], expected);
        for (i, branch) in branches.iter().enumerate() {
            let added = &netlist.module.instances[old_instances + i];
            assert_eq!(netlist.interner.resolve(added.type_name), Some("BUF4"));
            assert_eq!(netlist.interner.resolve(added.connections[0].0), Some("A"));
            assert_eq!(netlist.interner.resolve(added.connections[1].0), Some("Y"));
            assert_eq!(
                added.connections[0].1,
                branch
                    .parent
                    .map(|p| NetRef::Simple(NetIndex(first + p)))
                    .unwrap_or_else(|| root.source.clone())
            );
            assert_eq!(added.connections[1].1, NetRef::Simple(NetIndex(first + i)));
            assert_eq!(netlist.nets[first + i].width, None);
        }
        let declared = netlist.interner.get("g8r_data_fanout_wire_5").unwrap();
        assert!(
            netlist.nets[first..first + 3]
                .iter()
                .all(|wire| wire.name != declared)
        );
        Ok(())
    }

    // Two roots may feed different pins on one gate. A conflicting second claim
    // on the same pin invalidates the whole forest before it changes any
    // wiring.
    #[test]
    fn forest_can_move_different_pins_of_one_instance_and_rejects_late_conflicts() -> Result<()> {
        let lib = library()?;
        let mut netlist = read_mapped_netlist(
            "module top(input clk, input d, output y);
            wire a, b;
            DFF1 l0(.D(d), .CLK(clk), .Q(a));
            DFF1 l1(.D(d), .CLK(clk), .Q(b));
            AND1 use_both(.A(a), .B(b), .Y(y));
            endmodule",
        )?;
        let roots = roots(&netlist, &lib, 1.0)?;
        assert_eq!(roots.len(), 2);
        let buffer = buffer_model(
            &lib,
            lib.cells.iter().position(|c| c.name == "BUF4").unwrap(),
        )
        .unwrap();
        let leaf = Branch {
            parent: None,
            sinks: vec![0],
        };
        let trunk = Branch {
            parent: None,
            sinks: vec![],
        };
        let child = Branch {
            parent: Some(0),
            sinks: vec![0],
        };
        let original = netlist.clone();
        let bad_sink = Branch {
            parent: None,
            sinks: vec![99],
        };
        for bad_plan in [
            (0, vec![leaf.clone()]),
            (2, vec![leaf.clone()]),
            (1, vec![bad_sink]),
        ] {
            assert!(
                apply_forest(
                    &mut netlist,
                    &lib,
                    &buffer,
                    &roots,
                    &[(0, vec![leaf.clone()]), bad_plan]
                )
                .is_err()
            );
            assert_eq!(netlist.module, original.module);
            assert_eq!(netlist.nets, original.nets);
            assert_eq!(netlist.interner.len(), original.interner.len());
        }
        let first = netlist.module.net_index_range.end;
        apply_forest(
            &mut netlist,
            &lib,
            &buffer,
            &roots,
            &[(0, vec![leaf]), (1, vec![trunk, child])],
        )?;
        let pins = &netlist.module.instances[2].connections;
        assert_eq!(pins[0].1, NetRef::Simple(NetIndex(first)));
        assert_eq!(pins[1].1, NetRef::Simple(NetIndex(first + 2)));
        assert_eq!(
            netlist.module.instances[5].connections[0].1,
            NetRef::Simple(NetIndex(first + 1))
        );
        assert_eq!(
            netlist.module.instances[4].connections[0].1,
            roots[1].source
        );
        Ok(())
    }

    // Reject stale bindings and invalid branch references before allocating
    // even a temporary name or net. The original can then be used for another
    // trial.
    #[test]
    fn invalid_or_stale_tree_leaves_every_symbol_and_binding_unchanged() -> Result<()> {
        let (mut netlist, lib, root, buffer) = example()?;
        let original = netlist.clone();
        let head = Branch {
            parent: None,
            sinks: vec![0],
        };
        for tail in [
            Branch {
                parent: Some(1),
                sinks: vec![1],
            },
            Branch {
                parent: None,
                sinks: vec![0],
            },
            Branch {
                parent: None,
                sinks: vec![root.sinks.len()],
            },
            Branch {
                parent: Some(0),
                sinks: vec![],
            },
        ] {
            assert!(apply_tree(&mut netlist, &lib, &root, &buffer, &[head.clone(), tail]).is_err());
        }
        let mut wrong = buffer.clone();
        wrong.cell = lib.cells.iter().position(|c| c.name == "INV1").unwrap();
        assert!(
            apply_tree(
                &mut netlist,
                &lib,
                &root,
                &wrong,
                std::slice::from_ref(&head)
            )
            .is_err()
        );
        let mut stale = root;
        stale.sinks[1].original = NetRef::Unconnected;
        let tail = Branch {
            parent: Some(0),
            sinks: vec![1],
        };
        assert!(apply_tree(&mut netlist, &lib, &stale, &buffer, &[head, tail]).is_err());
        assert_eq!(netlist.module, original.module);
        assert_eq!(netlist.nets, original.nets);
        assert_eq!(netlist.interner.len(), original.interner.len());
        Ok(())
    }
}
