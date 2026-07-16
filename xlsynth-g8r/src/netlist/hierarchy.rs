// SPDX-License-Identifier: Apache-2.0

//! Structural hierarchy elaboration for parsed gate-level netlists.
//!
//! A parsed netlist may contain helper modules around Liberty-backed leaf
//! cells. Downstream evaluators want one electrical graph, but they also need
//! stable hierarchy provenance for labels and reports. This module recursively
//! inlines structural module instances into one flat NetlistModule while
//! retaining each child module boundary.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

use crate::netlist::io::ParsedNetlist;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistModule, NetlistPort, PortDirection, PortId, Pos, Span,
};

/// One child-module port preserved after structural hierarchy flattening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaboratedModuleBoundaryPort {
    pub name: PortId,
    pub direction: PortDirection,
    /// Flattened net carrying this child port's signal.
    pub net: NetIndex,
}

/// One instantiated child-module boundary preserved in the flat module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaboratedModuleBoundary {
    /// Slash-separated instance path from the selected top module.
    pub instance_path: String,
    pub module_name: String,
    pub ports: Vec<ElaboratedModuleBoundaryPort>,
}

/// One selected structural module tree flattened into a single module.
pub struct ElaboratedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
    pub module_boundaries: Vec<ElaboratedModuleBoundary>,
}

/// Recursively inlines parsed submodule instances under top_module.
///
/// Instances whose type names do not match a parsed module remain leaf
/// instances for later Liberty lookup. Child port connections use normalized
/// tran aliases whenever both sides are nets, which preserves electrical
/// identity without making otherwise-dead output bundles appear used.
pub fn elaborate_hierarchy(
    parsed: &ParsedNetlist,
    top_module: &NetlistModule,
) -> Result<ElaboratedNetlist> {
    let mut modules_by_name = HashMap::new();
    for module in &parsed.modules {
        if modules_by_name.insert(module.name, module).is_some() {
            let name = resolve_symbol(&parsed.interner, module.name, "module name")?;
            return Err(anyhow!(
                "netlist contains multiple definitions of module '{}'",
                name
            ));
        }
    }

    let mut elaborator = HierarchyElaborator {
        parsed,
        modules_by_name,
        interner: parsed.interner.clone(),
        nets: Vec::new(),
        wires: Vec::new(),
        assigns: Vec::new(),
        instances: Vec::new(),
        module_boundaries: Vec::new(),
        active_modules: Vec::new(),
    };
    let top_net_map = elaborator.inline_scope(top_module, &[], /* is_top= */ true)?;
    elaborator
        .module_boundaries
        .sort_by(|lhs, rhs| lhs.instance_path.cmp(&rhs.instance_path));
    let ports = top_module
        .ports
        .iter()
        .map(|port| remap_top_port(port, top_module, &top_net_map, &parsed.nets))
        .collect::<Result<Vec<_>>>()?;
    let module = NetlistModule {
        name: top_module.name,
        net_index_range: 0..elaborator.nets.len(),
        ports,
        wires: elaborator.wires,
        assigns: elaborator.assigns,
        instances: elaborator.instances,
    };
    Ok(ElaboratedNetlist {
        module,
        nets: elaborator.nets,
        interner: elaborator.interner,
        module_boundaries: elaborator.module_boundaries,
    })
}

struct HierarchyElaborator<'a> {
    parsed: &'a ParsedNetlist,
    modules_by_name: HashMap<PortId, &'a NetlistModule>,
    interner: StringInterner<StringBackend<SymbolU32>>,
    nets: Vec<Net>,
    wires: Vec<NetIndex>,
    assigns: Vec<NetlistAssign>,
    instances: Vec<NetlistInstance>,
    module_boundaries: Vec<ElaboratedModuleBoundary>,
    active_modules: Vec<PortId>,
}

impl HierarchyElaborator<'_> {
    fn inline_scope(
        &mut self,
        module: &NetlistModule,
        path: &[String],
        is_top: bool,
    ) -> Result<HashMap<NetIndex, NetIndex>> {
        if let Some(cycle_start) = self
            .active_modules
            .iter()
            .position(|active| *active == module.name)
        {
            let mut cycle = self.active_modules[cycle_start..]
                .iter()
                .map(|name| resolve_symbol(&self.parsed.interner, *name, "module name"))
                .collect::<Result<Vec<_>>>()?;
            cycle.push(resolve_symbol(
                &self.parsed.interner,
                module.name,
                "module name",
            )?);
            return Err(anyhow!(
                "recursive module instantiation is not supported: {}",
                cycle.join(" -> ")
            ));
        }
        self.active_modules.push(module.name);
        let result = self.inline_scope_body(module, path, is_top);
        self.active_modules.pop();
        result
    }

    fn inline_scope_body(
        &mut self,
        module: &NetlistModule,
        path: &[String],
        is_top: bool,
    ) -> Result<HashMap<NetIndex, NetIndex>> {
        let net_map = self.copy_scope_nets(module, path, is_top)?;
        self.copy_scope_assigns(module, &net_map)?;

        for instance in &module.instances {
            if let Some(child_module) = self.modules_by_name.get(&instance.type_name).copied() {
                self.inline_child_instance(instance, child_module, path, &net_map)?;
            } else {
                self.copy_leaf_instance(instance, path, &net_map)?;
            }
        }
        Ok(net_map)
    }

    fn copy_scope_nets(
        &mut self,
        module: &NetlistModule,
        path: &[String],
        is_top: bool,
    ) -> Result<HashMap<NetIndex, NetIndex>> {
        let mut net_map = HashMap::new();
        for old_raw_index in module.net_index_range.clone() {
            let old_index = NetIndex(old_raw_index);
            let old_net = self
                .parsed
                .nets
                .get(old_raw_index)
                .ok_or_else(|| anyhow!("module net index {} is out of range", old_raw_index))?;
            let local_name = resolve_symbol(&self.parsed.interner, old_net.name, "net name")?;
            let flat_name = if is_top {
                local_name
            } else {
                scoped_name(path, &local_name)
            };
            let new_index = NetIndex(self.nets.len());
            self.nets.push(Net {
                name: self.interner.get_or_intern(flat_name),
                width: old_net.width,
            });
            net_map.insert(old_index, new_index);
        }

        if is_top {
            for wire in &module.wires {
                self.wires.push(remap_net_index(*wire, &net_map)?);
            }
        } else {
            // Child ports and internal nets are all ordinary internal wires
            // after inlining into the selected top module.
            for old_raw_index in module.net_index_range.clone() {
                self.wires
                    .push(remap_net_index(NetIndex(old_raw_index), &net_map)?);
            }
        }
        Ok(net_map)
    }

    fn copy_scope_assigns(
        &mut self,
        module: &NetlistModule,
        net_map: &HashMap<NetIndex, NetIndex>,
    ) -> Result<()> {
        for assign in &module.assigns {
            self.assigns.push(NetlistAssign {
                kind: assign.kind,
                lhs: remap_net_ref(&assign.lhs, net_map)?,
                rhs: remap_assign_expr(&assign.rhs, net_map)?,
                span: assign.span,
            });
        }
        Ok(())
    }

    fn inline_child_instance(
        &mut self,
        instance: &NetlistInstance,
        child_module: &NetlistModule,
        parent_path: &[String],
        parent_net_map: &HashMap<NetIndex, NetIndex>,
    ) -> Result<()> {
        let local_instance_name = resolve_symbol(
            &self.parsed.interner,
            instance.instance_name,
            "instance name",
        )?;
        let mut child_path = parent_path.to_vec();
        child_path.push(local_instance_name);
        let child_net_map =
            self.inline_scope(child_module, &child_path, /* is_top= */ false)?;
        self.module_boundaries.push(self.build_module_boundary(
            child_module,
            child_path.as_slice(),
            &child_net_map,
        )?);
        self.connect_child_ports(instance, child_module, parent_net_map, &child_net_map)
    }

    fn build_module_boundary(
        &self,
        module: &NetlistModule,
        path: &[String],
        net_map: &HashMap<NetIndex, NetIndex>,
    ) -> Result<ElaboratedModuleBoundary> {
        let module_name = resolve_symbol(&self.parsed.interner, module.name, "module name")?;
        let ports = module
            .ports
            .iter()
            .map(|port| {
                let old_net = module
                    .find_net_index(port.name, &self.parsed.nets)
                    .ok_or_else(|| {
                        anyhow!(
                            "module '{}' port '{}' has no matching net",
                            module_name,
                            resolve_symbol(&self.parsed.interner, port.name, "port name")
                                .unwrap_or_else(|_| "<unknown>".to_string())
                        )
                    })?;
                Ok(ElaboratedModuleBoundaryPort {
                    name: port.name,
                    direction: port.direction.clone(),
                    net: remap_net_index(old_net, net_map)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ElaboratedModuleBoundary {
            instance_path: path.join("/"),
            module_name,
            ports,
        })
    }

    fn connect_child_ports(
        &mut self,
        instance: &NetlistInstance,
        child_module: &NetlistModule,
        parent_net_map: &HashMap<NetIndex, NetIndex>,
        child_net_map: &HashMap<NetIndex, NetIndex>,
    ) -> Result<()> {
        let instance_name = resolve_symbol(
            &self.parsed.interner,
            instance.instance_name,
            "instance name",
        )?;
        let child_module_name =
            resolve_symbol(&self.parsed.interner, child_module.name, "module name")?;
        let mut connections_by_port = HashMap::new();
        for (port_name, connection) in &instance.connections {
            if connections_by_port.insert(*port_name, connection).is_some() {
                return Err(anyhow!(
                    "module instance '{}' connects port '{}' more than once",
                    instance_name,
                    resolve_symbol(&self.parsed.interner, *port_name, "port name")?
                ));
            }
        }
        for connected_port in connections_by_port.keys() {
            if !child_module
                .ports
                .iter()
                .any(|port| port.name == *connected_port)
            {
                return Err(anyhow!(
                    "module instance '{}' of '{}' connects unknown port '{}'",
                    instance_name,
                    child_module_name,
                    resolve_symbol(&self.parsed.interner, *connected_port, "port name")?
                ));
            }
        }

        let span = synthetic_instance_span(instance);
        for port in &child_module.ports {
            let Some(parent_connection) = connections_by_port.get(&port.name) else {
                continue;
            };
            let parent_connection = remap_net_ref(parent_connection, parent_net_map)?;
            if matches!(parent_connection, NetRef::Unconnected) {
                continue;
            }
            if port.direction == PortDirection::Inout {
                return Err(anyhow!(
                    "module instance '{}' of '{}' has inout port '{}'; hierarchical gv-eval supports only input and output module ports",
                    instance_name,
                    child_module_name,
                    resolve_symbol(&self.parsed.interner, port.name, "port name")?
                ));
            }
            let child_net = child_module
                .find_net_index(port.name, &self.parsed.nets)
                .ok_or_else(|| {
                    anyhow!(
                        "module '{}' port '{}' has no matching net",
                        child_module_name,
                        resolve_symbol(&self.parsed.interner, port.name, "port name")
                            .unwrap_or_else(|_| "<unknown>".to_string())
                    )
                })?;
            let child_connection = NetRef::Simple(remap_net_index(child_net, child_net_map)?);
            if net_ref_is_aliasable(&parent_connection) {
                self.assigns.push(NetlistAssign {
                    kind: NetlistAssignKind::Tran,
                    lhs: child_connection,
                    rhs: AssignExpr::Leaf(parent_connection),
                    span,
                });
            } else if port.direction == PortDirection::Input {
                self.assigns.push(NetlistAssign {
                    kind: NetlistAssignKind::Continuous,
                    lhs: child_connection,
                    rhs: AssignExpr::Leaf(parent_connection),
                    span,
                });
            } else {
                return Err(anyhow!(
                    "module instance '{}' output port '{}' must connect to nets or be unconnected",
                    instance_name,
                    resolve_symbol(&self.parsed.interner, port.name, "port name")?
                ));
            }
        }
        Ok(())
    }

    fn copy_leaf_instance(
        &mut self,
        instance: &NetlistInstance,
        path: &[String],
        net_map: &HashMap<NetIndex, NetIndex>,
    ) -> Result<()> {
        let local_name = resolve_symbol(
            &self.parsed.interner,
            instance.instance_name,
            "instance name",
        )?;
        let instance_name = if path.is_empty() {
            local_name
        } else {
            scoped_name(path, &local_name)
        };
        let connections = instance
            .connections
            .iter()
            .map(|(port, connection)| Ok((*port, remap_net_ref(connection, net_map)?)))
            .collect::<Result<Vec<_>>>()?;
        self.instances.push(NetlistInstance {
            type_name: instance.type_name,
            instance_name: self.interner.get_or_intern(instance_name),
            connections,
            inst_lineno: instance.inst_lineno,
            inst_colno: instance.inst_colno,
        });
        Ok(())
    }
}

fn remap_top_port(
    port: &NetlistPort,
    top_module: &NetlistModule,
    top_net_map: &HashMap<NetIndex, NetIndex>,
    original_nets: &[Net],
) -> Result<NetlistPort> {
    let old_net = top_module
        .find_net_index(port.name, original_nets)
        .ok_or_else(|| anyhow!("top module port has no matching net"))?;
    remap_net_index(old_net, top_net_map)?;
    Ok(port.clone())
}

fn remap_net_index(net: NetIndex, net_map: &HashMap<NetIndex, NetIndex>) -> Result<NetIndex> {
    net_map
        .get(&net)
        .copied()
        .ok_or_else(|| anyhow!("net index {} is outside the elaborated module scope", net.0))
}

fn remap_net_ref(net_ref: &NetRef, net_map: &HashMap<NetIndex, NetIndex>) -> Result<NetRef> {
    Ok(match net_ref {
        NetRef::Simple(net) => NetRef::Simple(remap_net_index(*net, net_map)?),
        NetRef::BitSelect(net, bit) => NetRef::BitSelect(remap_net_index(*net, net_map)?, *bit),
        NetRef::PartSelect(net, msb, lsb) => {
            NetRef::PartSelect(remap_net_index(*net, net_map)?, *msb, *lsb)
        }
        NetRef::Literal(bits) => NetRef::Literal(bits.clone()),
        NetRef::UnknownLiteral(width) => NetRef::UnknownLiteral(*width),
        NetRef::Unconnected => NetRef::Unconnected,
        NetRef::Concat(parts) => NetRef::Concat(
            parts
                .iter()
                .map(|part| remap_net_ref(part, net_map))
                .collect::<Result<Vec<_>>>()?,
        ),
    })
}

fn remap_assign_expr(
    expr: &AssignExpr,
    net_map: &HashMap<NetIndex, NetIndex>,
) -> Result<AssignExpr> {
    Ok(match expr {
        AssignExpr::Leaf(net_ref) => AssignExpr::Leaf(remap_net_ref(net_ref, net_map)?),
        AssignExpr::Not(inner) => AssignExpr::Not(Box::new(remap_assign_expr(inner, net_map)?)),
        AssignExpr::And(lhs, rhs) => AssignExpr::And(
            Box::new(remap_assign_expr(lhs, net_map)?),
            Box::new(remap_assign_expr(rhs, net_map)?),
        ),
        AssignExpr::Or(lhs, rhs) => AssignExpr::Or(
            Box::new(remap_assign_expr(lhs, net_map)?),
            Box::new(remap_assign_expr(rhs, net_map)?),
        ),
        AssignExpr::Xor(lhs, rhs) => AssignExpr::Xor(
            Box::new(remap_assign_expr(lhs, net_map)?),
            Box::new(remap_assign_expr(rhs, net_map)?),
        ),
    })
}

fn net_ref_is_aliasable(net_ref: &NetRef) -> bool {
    match net_ref {
        NetRef::Simple(_) | NetRef::BitSelect(_, _) | NetRef::PartSelect(_, _, _) => true,
        NetRef::Concat(parts) => parts.iter().all(net_ref_is_aliasable),
        NetRef::Literal(_) | NetRef::UnknownLiteral(_) | NetRef::Unconnected => false,
    }
}

fn scoped_name(path: &[String], local_name: &str) -> String {
    format!("{}/{}", path.join("/"), local_name)
}

fn synthetic_instance_span(instance: &NetlistInstance) -> Span {
    let start = Pos {
        lineno: instance.inst_lineno,
        colno: instance.inst_colno,
    };
    Span {
        start,
        limit: Pos {
            lineno: start.lineno,
            colno: start.colno.saturating_add(1),
        },
    }
}

fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    symbol: PortId,
    what: &str,
) -> Result<String> {
    interner
        .resolve(symbol)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("could not resolve {} symbol", what))
}
