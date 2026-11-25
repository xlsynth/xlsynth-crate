// SPDX-License-Identifier: Apache-2.0

//! Apply a buffer tree to the exact data-pin connections selected by
//! editability analysis.

use super::buffers::{InsertableBufferCell, buffer_for_insertion};
use super::plan::{BufferTreePlan, PlannedBuffer};
use crate::liberty_model::Library;
use crate::netlist::bit_ref::{net_ref_lsb_targets, net_ref_width_bits};
use crate::netlist::connectivity::PinConnection;
use crate::netlist::editability::EditableOutput;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance};
use anyhow::{Result, anyhow, ensure};
use std::collections::HashSet;

/// Validate every tree against the netlist before changing any connections.
/// Ports, aliases and fixed loads stay connected. Only the listed input pins
/// are rewired, even when two trees drive different pins on the same cell.
pub(super) fn apply_buffer_trees(
    netlist: &mut MappedNetlist,
    library: &Library,
    buffer: &InsertableBufferCell,
    roots: &[EditableOutput],
    plans: &[BufferTreePlan],
) -> Result<()> {
    let root_for = |tree: &BufferTreePlan| {
        roots
            .get(tree.driver_index)
            .ok_or_else(|| anyhow!("unknown tree root {}", tree.driver_index))
    };
    if let [tree] = plans {
        let root = root_for(tree)?;
        return apply_plans(netlist, library, buffer, &[(root, tree.buffers.as_slice())]);
    }
    let selected = plans
        .iter()
        .map(|tree| Ok((root_for(tree)?, tree.buffers.as_slice())))
        .collect::<Result<Vec<_>>>()?;
    apply_plans(netlist, library, buffer, &selected)
}

/// Graph discovery certifies driver/sink roles and bit-level alias equivalence.
/// Recheck their physical bindings and tree topology before allocating names.
/// Empty branches may drive child buffers. An entirely empty forest is a no-op.
fn apply_plans(
    netlist: &mut MappedNetlist,
    library: &Library,
    buffer: &InsertableBufferCell,
    trees: &[(&EditableOutput, &[PlannedBuffer])],
) -> Result<()> {
    let count = trees.iter().try_fold(0_usize, |count, (_, buffers)| {
        count
            .checked_add(buffers.len())
            .ok_or_else(|| anyhow!("branch count overflow"))
    })?;
    let mut drivers = HashSet::with_capacity(trees.len());
    for &(root, _) in trees {
        ensure!(drivers.insert(root.binding), "repeated tree driver");
    }
    if count == 0 {
        return Ok(());
    }
    let cell = library
        .cells
        .get(buffer.cell_index)
        .ok_or_else(|| anyhow!("unknown buffer cell {}", buffer.cell_index))?;
    let model = buffer_for_insertion(library, buffer.cell_index).ok_or_else(|| {
        anyhow!(
            "cell {} is not a legal positive data buffer",
            buffer.cell_index
        )
    })?;
    ensure!(
        model.input_pin_name == buffer.input_pin_name
            && model.output_pin_name == buffer.output_pin_name,
        "buffer pins disagree with the library"
    );
    let module = &netlist.module;
    let range = &module.net_index_range;
    ensure!(
        range.start <= range.end && range.end <= netlist.nets.len(),
        "invalid module net range"
    );
    let check_binding = |binding: PinConnection, original: &NetRef| -> Result<()> {
        let instance = binding.instance.0;
        let connection = binding.connection;
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
    let mut reconnected_pins = HashSet::new();
    for &(root, buffers) in trees {
        check_binding(root.binding, &root.source)?;
        let driver = &module.instances[root.binding.instance.0];
        let driver_type = library
            .cells
            .get(root.cell)
            .ok_or_else(|| anyhow!("unknown driver cell"))?;
        ensure!(
            netlist.interner.resolve(driver.instance_name) == Some(root.instance_name.as_str())
                && netlist.interner.resolve(driver.type_name) == Some(driver_type.name.as_str())
                && netlist
                    .interner
                    .resolve(driver.connections[root.binding.connection].0)
                    == Some(root.pin_name.as_str()),
            "stale tree driver identity"
        );
        let mut used = vec![false; buffers.len()];
        for (i, branch) in buffers.iter().enumerate() {
            if let Some(driving_buffer_index) = branch.driving_buffer_index {
                ensure!(
                    driving_buffer_index < i,
                    "branch {i} must follow its parent"
                );
                used[driving_buffer_index] = true;
            }
            for &index in &branch.sink_indices {
                let sink = root
                    .sinks
                    .get(index)
                    .ok_or_else(|| anyhow!("unknown sink {index}"))?;
                ensure!(sink.binding != root.binding, "cannot rewire the driver");
                ensure!(
                    reconnected_pins.insert(sink.binding),
                    "pin appears twice in the tree"
                );
                check_binding(sink.binding, &sink.original)?;
                used[i] = true;
            }
        }
        ensure!(
            used.iter().all(|&has_load| has_load),
            "tree contains a buffer without a load"
        );
    }
    ensure!(
        drivers.is_disjoint(&reconnected_pins),
        "cannot rewire a forest driver"
    );
    // New nets occupy consecutive indices at the end of this module's range.
    // A parsed file can retain unused net declarations from a later module.
    let first = range.end;
    let end = first
        .checked_add(count)
        .ok_or_else(|| anyhow!("net index overflow"))?;
    let cell_type = netlist.interner.get_or_intern(&cell.name);
    let input_pin = netlist.interner.get_or_intern(&buffer.input_pin_name);
    let output_pin = netlist.interner.get_or_intern(&buffer.output_pin_name);
    let mut ordinal = module.instances.len();
    let mut wires = Vec::with_capacity(count);
    netlist.module.wires.reserve(count);
    netlist.module.instances.reserve(count);
    for &(root, buffers) in trees {
        let tree_first = first + wires.len();
        for (i, branch) in buffers.iter().enumerate() {
            let (wire_name, cell_name) = loop {
                let wire = format!("g8r_data_fanout_wire_{ordinal}");
                let cell = format!("g8r_data_fanout_buf_{ordinal}");
                ordinal += 1;
                if netlist.interner.get(&wire).is_none() && netlist.interner.get(&cell).is_none() {
                    break (
                        netlist.interner.get_or_intern(wire),
                        netlist.interner.get_or_intern(cell),
                    );
                }
            };
            let input = match branch.driving_buffer_index {
                Some(driving_buffer_index) => {
                    NetRef::Simple(NetIndex(tree_first + driving_buffer_index))
                }
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
            for &index in &branch.sink_indices {
                let sink = &root.sinks[index];
                netlist.module.instances[sink.binding.instance.0].connections
                    [sink.binding.connection]
                    .1 = output.clone();
            }
        }
    }
    netlist.nets.splice(first..first, wires);
    netlist.module.net_index_range.end = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::library;
    use super::*;
    use crate::netlist::editability::analyze_editable_outputs;
    use crate::netlist::io::read_gv_from_str;

    fn branch(driving_buffer_index: Option<usize>, sink_indices: &[usize]) -> PlannedBuffer {
        PlannedBuffer {
            driving_buffer_index,
            sink_indices: sink_indices.to_vec(),
        }
    }

    fn example() -> Result<(MappedNetlist, Library, EditableOutput, InsertableBufferCell)> {
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
        let mut parsed = read_gv_from_str(text)?;
        let netlist = MappedNetlist {
            module: parsed.modules.remove(0),
            nets: parsed.nets,
            interner: parsed.interner,
        };
        let root = analyze_editable_outputs(&netlist, &lib, 1.0, None)?
            .roots
            .into_iter()
            .find(|root| root.binding.instance.0 == 0)
            .expect("launch Q");
        assert_eq!(root.sinks.len(), 4);
        let index = lib.cells.iter().position(|c| c.name == "BUF4").unwrap();
        let buffer = buffer_for_insertion(&lib, index).expect("ordinary buffer");
        Ok((netlist, lib, root, buffer))
    }

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
        let buffers = [
            branch(None, &[]),
            branch(Some(0), &[alias]),
            branch(None, &[direct]),
        ];
        apply_plans(&mut netlist, &lib, &buffer, &[(&root, &buffers)])?;
        assert_eq!(netlist.module.assigns, original.module.assigns);
        assert_eq!(netlist.module.ports, original.module.ports);
        assert_eq!(netlist.module.net_index_range.end, first + 3);
        assert_eq!(netlist.nets[first + 3..], original.nets[first..]);
        let mut expected = original.module.instances.clone();
        for (idx, branch) in [(alias, 1), (direct, 2)] {
            let sink = &root.sinks[idx];
            expected[sink.binding.instance.0].connections[sink.binding.connection].1 =
                NetRef::Simple(NetIndex(first + branch));
        }
        assert_eq!(netlist.module.instances[..old_instances], expected);
        for (i, branch) in buffers.iter().enumerate() {
            let added = &netlist.module.instances[old_instances + i];
            assert_eq!(netlist.interner.resolve(added.type_name), Some("BUF4"));
            assert_eq!(netlist.interner.resolve(added.connections[0].0), Some("A"));
            assert_eq!(netlist.interner.resolve(added.connections[1].0), Some("Y"));
            assert_eq!(
                added.connections[0].1,
                branch
                    .driving_buffer_index
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

    #[test]
    fn buffer_trees_reconnect_different_pins_of_one_instance_and_reject_late_conflicts()
    -> Result<()> {
        let lib = library()?;
        let mut parsed = read_gv_from_str(
            "module top(input clk, input d, output y);
            wire a, b;
            DFF1 l0(.D(d), .CLK(clk), .Q(a));
            DFF1 l1(.D(d), .CLK(clk), .Q(b));
            AND1 use_both(.A(a), .B(b), .Y(y));
            endmodule",
        )?;
        let mut netlist = MappedNetlist {
            module: parsed.modules.remove(0),
            nets: parsed.nets,
            interner: parsed.interner,
        };
        let roots = analyze_editable_outputs(&netlist, &lib, 1.0, None)?.roots;
        assert_eq!(roots.len(), 2);
        let buffer = buffer_for_insertion(
            &lib,
            lib.cells.iter().position(|c| c.name == "BUF4").unwrap(),
        )
        .unwrap();
        let leaf = branch(None, &[0]);
        let trunk = branch(None, &[]);
        let child = branch(Some(0), &[0]);
        let original = netlist.clone();
        let bad_sink = branch(None, &[99]);
        let tree = |driver_index, buffers| BufferTreePlan {
            driver_index,
            buffers,
        };
        for bad_plan in [
            tree(0, vec![leaf.clone()]),
            tree(2, vec![leaf.clone()]),
            tree(1, vec![bad_sink]),
        ] {
            assert!(
                apply_buffer_trees(
                    &mut netlist,
                    &lib,
                    &buffer,
                    &roots,
                    &[tree(0, vec![leaf.clone()]), bad_plan]
                )
                .is_err()
            );
            assert_eq!(netlist.module, original.module);
            assert_eq!(netlist.nets, original.nets);
            assert_eq!(netlist.interner.len(), original.interner.len());
        }
        let first = netlist.module.net_index_range.end;
        apply_buffer_trees(
            &mut netlist,
            &lib,
            &buffer,
            &roots,
            &[tree(0, vec![leaf]), tree(1, vec![trunk, child])],
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

    #[test]
    fn invalid_or_stale_tree_leaves_every_symbol_and_binding_unchanged() -> Result<()> {
        let (mut netlist, lib, root, buffer) = example()?;
        let original = netlist.clone();
        let head = branch(None, &[0]);
        for tail in [
            branch(Some(1), &[1]),
            branch(None, &[0]),
            branch(None, &[root.sinks.len()]),
            branch(Some(0), &[]),
        ] {
            assert!(
                apply_plans(
                    &mut netlist,
                    &lib,
                    &buffer,
                    &[(&root, &[head.clone(), tail])]
                )
                .is_err()
            );
        }
        let mut wrong = buffer.clone();
        wrong.cell_index = lib.cells.iter().position(|c| c.name == "INV1").unwrap();
        assert!(
            apply_plans(
                &mut netlist,
                &lib,
                &wrong,
                &[(&root, std::slice::from_ref(&head))]
            )
            .is_err()
        );
        let mut stale = root;
        stale.sinks[1].original = NetRef::Unconnected;
        let tail = branch(Some(0), &[1]);
        assert!(apply_plans(&mut netlist, &lib, &buffer, &[(&stale, &[head, tail])]).is_err());
        assert_eq!(netlist.module, original.module);
        assert_eq!(netlist.nets, original.nets);
        assert_eq!(netlist.interner.len(), original.interner.len());
        Ok(())
    }
}
