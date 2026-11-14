// SPDX-License-Identifier: Apache-2.0

//! Checked block interfaces, sequential resources, and instantiations.

use std::collections::{BTreeMap, BTreeSet};

use super::{BValue, Builder, BuilderError, check_name, check_package_name, checked_flat_width};
use crate::IrValue;
use crate::ir::{
    self, Block, BlockPort, BlockReset, Instantiation, InstantiationKind, NodePayload, NodeRef,
    Package, PackageMember, Register, Type, Unop,
};
use crate::ir_rebase_ids::{package_max_emitted_node_id, rebase_block_ids_in_place};
use crate::ir_verify::{
    verify_block_structure_in_package, verify_function_signature, verify_package,
};

/// A register belonging to exactly one block builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BRegister {
    builder_id: usize,
    index: usize,
}

/// A block-owned electrical net, independent of its public ports or SSA reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BInoutNet {
    builder_id: usize,
    index: usize,
}

/// Compatibility handle for a net exposed through a bidirectional port.
pub type BInoutPort = BInoutNet;

/// An instance belonging to exactly one block builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BInstantiation {
    builder_id: usize,
    index: usize,
}

/// Timing and polarity of the block's reset input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResetBehavior {
    pub asynchronous: bool,
    pub active_low: bool,
}

/// Optional controls on a register write; both controls must be bits[1].
#[derive(Debug, Clone, Copy, Default)]
pub struct RegisterWriteOptions {
    pub load_enable: Option<BValue>,
    pub reset: Option<BValue>,
}

struct RegisterWrite {
    load_enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

struct RegisterState {
    definition: Register,
    read: Option<BValue>,
    writes: Vec<RegisterWrite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PortSignature {
    Input(String, Type),
    Output(String, Type),
    Inout(String, Type, String),
    Clock(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockSignature {
    ports: Vec<PortSignature>,
    reset: Option<(String, ResetBehavior)>,
}

#[derive(Debug, PartialEq)]
struct ExternalSignature {
    function_type: ir::FunctionType,
    parameter_names: Vec<String>,
}

#[derive(Debug, PartialEq)]
enum InstanceSignature {
    Block(BlockSignature),
    Extern(ExternalSignature),
}

/// Accesses emitted for a single privately connected child pin.
#[derive(Clone, Copy)]
struct PrivateInout {
    net: BInoutNet,
    read: Option<BValue>,
}

/// A child pin can share a parent port or have its own private wire.
enum InoutConnection {
    Bound,
    Private(PrivateInout),
}

/// Signature and connections accumulated for a single instance.
struct InstanceState {
    definition: Instantiation,
    signature: InstanceSignature,
    input_types: BTreeMap<String, Type>,
    output_types: BTreeMap<String, Type>,
    inout_types: BTreeMap<String, Type>,
    inout_connections: BTreeMap<String, InoutConnection>,
    inputs: BTreeSet<String>,
    outputs: BTreeMap<String, BValue>,
}

/// Block-specific state; use `BlockBuilder` to construct and finalize it.
#[doc(hidden)]
#[derive(Default)]
pub struct BlockState {
    ports: Vec<BlockPort>,
    inout_nets: Vec<ir::InoutNet>,
    inout_indices: BTreeMap<String, usize>,
    // Private nets created by legacy instance accessors receive printable names
    // at finalization.
    private_net_names: Vec<(usize, String)>,
    inout_drives: BTreeSet<usize>,
    port_names: BTreeMap<String, Option<NodeRef>>,
    reset: Option<BlockReset>,
    registers: Vec<RegisterState>,
    instances: Vec<InstanceState>,
    wires: BTreeSet<usize>,
}

impl super::sealed::State for BlockState {
    fn check_name_available(&self, name: &str, node: Option<NodeRef>) -> Result<(), BuilderError> {
        if let Some(owner) = self.port_names.get(name)
            && (node.is_none() || *owner != node)
        {
            return Err(BuilderError::DuplicateName(name.to_string()));
        }
        Ok(())
    }
}

/// Builds a block with checked values, ports, registers, and instances.
///
/// All common `FnBuilder` node operations are available, but blocks have no
/// parameters or return value. Ports may be interleaved with body nodes.
/// Resource handles are builder-scoped, and invalid operations leave the
/// builder usable. Completeness checks (such as a register needing a write)
/// run when building. Instances and function calls require package context.
///
/// ```
/// use xlsynth_pir::{BlockBuilder, ir::Type};
/// let mut builder = BlockBuilder::new("inverter");
/// let input = builder.input_port("x", Type::Bits(8))?;
/// let inverted = builder.not(input)?;
/// builder.output_port("y", inverted)?;
/// let block = builder.build()?;
/// assert_eq!(block.output_ports().count(), 1);
/// # Ok::<(), xlsynth_pir::BuilderError>(())
/// ```
pub type BlockBuilder = Builder<BlockState>;

impl Builder<BlockState> {
    /// Creates an empty builder; the block name is checked when building.
    pub fn new(name: &str) -> Self {
        Self::from_state(name, BlockState::default())
    }

    /// Declares a typed wire that can be read before its driver is attached.
    pub fn declare_wire(&mut self, ty: Type) -> Result<BValue, BuilderError> {
        // A parameter carries the wire's type while the graph is being built.
        // Finalization requires every placeholder to be replaced.
        let wire = self.add_node(NodePayload::Param, Some(ty))?;
        self.state.wires.insert(wire.node.index);
        Ok(wire)
    }

    /// Attaches the sole driver of a wire declared by this builder.
    pub fn drive_wire(&mut self, wire: BValue, value: BValue) -> Result<(), BuilderError> {
        let wire = self.node(wire)?;
        let driver = self.node(value)?;
        if !self.state.wires.contains(&wire.index) {
            return Err(invalid("value is not a wire"));
        }
        let expected = &self.graph.get_node(wire).ty;
        let actual = &self.graph.get_node(driver).ty;
        if !matches!(self.graph.get_node(wire).payload, NodePayload::Param) {
            return Err(invalid(format!("wire {} already has a driver", wire.index)));
        }
        if expected != actual {
            return Err(invalid(format!(
                "wire {} expects {expected}, got {actual}",
                wire.index
            )));
        }
        self.graph.get_node_mut(wire).payload = NodePayload::Unop(Unop::Identity, driver);
        Ok(())
    }

    /// Returns the block-local IR text ID before package rebasing.
    pub fn value_id(&self, value: BValue) -> Result<usize, BuilderError> {
        Ok(self.graph.get_node(self.node(value)?).text_id)
    }

    /// Adds an input port in declaration order, even after body nodes.
    pub fn input_port(&mut self, name: &str, ty: Type) -> Result<BValue, BuilderError> {
        self.check_node_name(name, None)?;
        checked_flat_width(&ty)?;
        let value = self.add_node(
            NodePayload::InputPort {
                name: name.to_string(),
                sv_type: None,
            },
            Some(ty),
        )?;
        self.graph.get_node_mut(value.node).name = Some(name.to_string());
        self.names.insert(name.to_string(), value.node);
        self.state
            .port_names
            .insert(name.to_string(), Some(value.node));
        self.state.ports.push(BlockPort::Input(value.node));
        Ok(value)
    }

    /// Declares a net with zero or more public aliases, in port declaration
    /// order.
    pub fn declare_inout_net(
        &mut self,
        name: &str,
        ty: Type,
        ports: &[&str],
    ) -> Result<BInoutNet, BuilderError> {
        check_name(name)?;
        if self.state.inout_indices.contains_key(name) {
            return Err(invalid(format!("duplicate inout net '{name}'")));
        }
        if checked_flat_width(&ty)? == 0 || ty.contains_token() {
            return Err(invalid("inout nets require a nonempty, token-free type"));
        }
        let mut seen = BTreeSet::new();
        for &port in ports {
            self.check_node_name(port, None)?;
            if !seen.insert(port) {
                return Err(BuilderError::DuplicateName(port.to_owned()));
            }
        }
        for &port in ports {
            self.state.ports.push(BlockPort::Inout(port.to_owned()));
            self.state.port_names.insert(port.to_owned(), None);
        }
        Ok(self.push_inout_net(name, ty, ports.iter().map(|p| (*p).to_owned()).collect()))
    }

    /// Records an already-checked net and returns its builder-scoped handle.
    fn push_inout_net(&mut self, name: &str, ty: Type, ports: Vec<String>) -> BInoutNet {
        let index = self.state.inout_nets.len();
        self.state.inout_nets.push(ir::InoutNet {
            name: name.to_owned(),
            ty,
            ports,
            sv_type: None,
        });
        self.state.inout_indices.insert(name.to_owned(), index);
        BInoutNet {
            builder_id: self.id,
            index,
        }
    }

    /// Declares a net exposed through one public pin with the same name.
    pub fn declare_inout_port(&mut self, name: &str, ty: Type) -> Result<BInoutPort, BuilderError> {
        self.declare_inout_net(name, ty, &[name])
    }

    /// Rejects handles from another builder before accessing a declaration.
    fn net_definition(&self, net: BInoutNet) -> Result<&ir::InoutNet, BuilderError> {
        if net.builder_id != self.id {
            return Err(invalid("inout net belongs to a different builder"));
        }
        Ok(&self.state.inout_nets[net.index])
    }

    /// Reads the whole resolved net.
    pub fn ext_inout_read(&mut self, net: BInoutNet) -> Result<BValue, BuilderError> {
        let def = self.net_definition(net)?;
        self.add_node(
            NodePayload::ExtInoutRead {
                net: def.name.clone(),
            },
            Some(def.ty.clone()),
        )
    }

    /// Adds a local driver on the net, enabled by an active-high bits[1] value.
    pub fn ext_inout_drive(
        &mut self,
        net: BInoutNet,
        data: BValue,
        output_enable: BValue,
    ) -> Result<BValue, BuilderError> {
        let drive = self.node(data)?;
        let output_enable = self.control_node(output_enable)?;
        let def = self.net_definition(net)?;
        if self.graph.get_node_ty(drive) != &def.ty {
            return Err(invalid(format!(
                "inout net '{}' requires drive type {}",
                def.name, def.ty
            )));
        }
        if self.state.inout_drives.contains(&net.index) {
            return Err(invalid(format!(
                "duplicate local inout driver for '{}'",
                def.name
            )));
        }
        let sink = self.add_node(
            NodePayload::ExtInoutDrive {
                net: def.name.clone(),
                drive,
                output_enable,
            },
            None,
        )?;
        self.state.inout_drives.insert(net.index);
        Ok(sink)
    }

    /// Compatibility shorthand for reading an exposed net.
    pub fn ext_inout_port_input(&mut self, port: BInoutPort) -> Result<BValue, BuilderError> {
        self.ext_inout_read(port)
    }

    /// Compatibility shorthand for driving an exposed net.
    pub fn ext_inout_port_output(
        &mut self,
        port: BInoutPort,
        drive: BValue,
        output_enable: BValue,
    ) -> Result<BValue, BuilderError> {
        self.ext_inout_drive(port, drive, output_enable)
    }

    /// Compatibility shorthand for a declaration, a read, and an enabled
    /// driver.
    pub fn ext_inout_port(
        &mut self,
        name: &str,
        drive: BValue,
        output_enable: BValue,
    ) -> Result<BValue, BuilderError> {
        let ty = self.graph.get_node_ty(self.node(drive)?).clone();
        if self.graph.get_node_ty(self.node(output_enable)?) != &Type::Bits(1) {
            return Err(invalid("ext_inout_port output enable must be bits[1]"));
        }
        let port = self.declare_inout_port(name, ty)?;
        let read = self.ext_inout_port_input(port)?;
        self.graph.get_node_mut(read.node).name = Some(name.to_owned());
        self.names.insert(name.to_owned(), read.node);
        self.state
            .port_names
            .insert(name.to_owned(), Some(read.node));
        self.ext_inout_port_output(port, drive, output_enable)?;
        Ok(read)
    }

    /// Returns the declared pin handle represented by an existing local read.
    fn pin_for_read(&self, node: NodeRef) -> Result<BInoutPort, BuilderError> {
        let NodePayload::ExtInoutRead { net } = &self.graph.get_node(node).payload else {
            return Err(invalid("expected a local inout port read"));
        };
        Ok(BInoutPort {
            builder_id: self.id,
            index: self.state.inout_indices[net],
        })
    }

    /// Binds a child inout using a read of a local pin (compatibility API).
    pub fn ext_instantiation_inout(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        parent: BValue,
    ) -> Result<BValue, BuilderError> {
        let port = self.pin_for_read(self.node(parent)?)?;
        self.bind_inout_port(instance, port_name, port)
    }

    /// Connects a child pin directly to a declared private or public net.
    pub fn bind_inout_net(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        parent: BInoutNet,
    ) -> Result<BValue, BuilderError> {
        let index = self.instance_index(instance)?;
        let pin = self.net_definition(parent)?;
        let state = &self.state.instances[index];
        if state.inout_types.get(port_name) != Some(&pin.ty) {
            return Err(invalid(format!(
                "missing or incompatible child inout {port_name}"
            )));
        }
        if state.inout_connections.contains_key(port_name) {
            return Err(invalid(format!(
                "child inout {port_name} is already connected"
            )));
        }
        let link = self.add_node(
            NodePayload::ExtInstantiationInout {
                instantiation: state.definition.name.clone(),
                port_name: port_name.to_owned(),
                net: pin.name.clone(),
            },
            None,
        )?;
        self.state.instances[index]
            .inout_connections
            .insert(port_name.to_owned(), InoutConnection::Bound);
        Ok(link)
    }

    /// Compatibility shorthand for binding a child pin to an exposed net.
    pub fn bind_inout_port(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        net: BInoutPort,
    ) -> Result<BValue, BuilderError> {
        self.bind_inout_net(instance, port_name, net)
    }

    /// Gets or creates the private net used by a legacy instance accessor.
    fn private_inout(
        &mut self,
        index: usize,
        port_name: &str,
    ) -> Result<PrivateInout, BuilderError> {
        let state = &self.state.instances[index];
        let ty = state
            .inout_types
            .get(port_name)
            .ok_or_else(|| invalid(format!("unknown child inout {port_name}")))?
            .clone();
        match state.inout_connections.get(port_name) {
            Some(InoutConnection::Bound) => {
                return Err(invalid(format!("child inout {port_name} is already bound")));
            }
            Some(InoutConnection::Private(access)) => return Ok(*access),
            None => { /* First access creates the wire and the child connection. */ }
        }
        let instantiation = state.definition.name.clone();
        let base = format!("{instantiation}_{port_name}");
        // The temporary key cannot be a user identifier. Allocate its final
        // name after later declarations and named SSA values are known.
        let key = format!("#{}", self.state.inout_nets.len());
        self.add_node(
            NodePayload::ExtInstantiationInout {
                instantiation,
                port_name: port_name.to_owned(),
                net: key.clone(),
            },
            None,
        )?;
        let net = self.push_inout_net(&key, ty, Vec::new());
        self.state.private_net_names.push((net.index, base));
        let access = PrivateInout { net, read: None };
        self.state.instances[index]
            .inout_connections
            .insert(port_name.to_owned(), InoutConnection::Private(access));
        Ok(access)
    }

    /// Samples a private child pin, sharing the same net for repeated accesses.
    pub fn ext_instantiation_inout_output(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
    ) -> Result<BValue, BuilderError> {
        let index = self.instance_index(instance)?;
        let mut access = self.private_inout(index, port_name)?;
        if let Some(value) = access.read {
            return Ok(value);
        }
        let value = self.ext_inout_read(access.net)?;
        access.read = Some(value);
        self.state.instances[index]
            .inout_connections
            .insert(port_name.to_owned(), InoutConnection::Private(access));
        Ok(value)
    }

    /// Drives a private child net using the legacy instance-oriented shorthand.
    pub fn ext_instantiation_inout_input(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        drive: BValue,
        output_enable: BValue,
    ) -> Result<BValue, BuilderError> {
        let index = self.instance_index(instance)?;
        let data = self.node(drive)?;
        self.control_node(output_enable)?;
        let ty = self.state.instances[index]
            .inout_types
            .get(port_name)
            .ok_or_else(|| invalid(format!("unknown child inout {port_name}")))?;
        if ty != self.graph.get_node_ty(data) {
            return Err(invalid(format!(
                "child inout {port_name} requires drive type {ty}"
            )));
        }
        let access = self.private_inout(index, port_name)?;
        self.ext_inout_drive(access.net, drive, output_enable)
    }

    /// Sets the declared SystemVerilog type of a net and its public aliases.
    pub fn set_inout_sv_type(
        &mut self,
        net: BInoutNet,
        sv_type: Option<&str>,
    ) -> Result<(), BuilderError> {
        self.net_definition(net)?;
        self.state.inout_nets[net.index].sv_type = sv_type.map(str::to_owned);
        Ok(())
    }

    /// Adds an output port and returns its unit-valued sink node.
    pub fn output_port(&mut self, name: &str, value: BValue) -> Result<BValue, BuilderError> {
        self.check_node_name(name, None)?;
        let arg = self.node(value)?;
        let output = self.add_node(
            NodePayload::OutputPort {
                name: name.to_string(),
                arg,
                sv_type: None,
            },
            None,
        )?;
        self.graph.get_node_mut(output.node).name = Some(name.to_string());
        self.names.insert(name.to_string(), output.node);
        self.state
            .port_names
            .insert(name.to_string(), Some(output.node));
        self.state.ports.push(BlockPort::Output(output.node));
        Ok(output)
    }

    /// Adds the single clock declaration, without allocating a value or node
    /// ID.
    pub fn clock_port(&mut self, name: &str) -> Result<(), BuilderError> {
        self.check_node_name(name, None)?;
        if self
            .state
            .ports
            .iter()
            .any(|port| matches!(port, BlockPort::Clock(_)))
        {
            return Err(invalid("a block may have only one clock port"));
        }
        self.state.port_names.insert(name.to_string(), None);
        self.state.ports.push(BlockPort::Clock(name.to_string()));
        Ok(())
    }

    /// Sets an input port's reset timing and polarity; the input must be
    /// bits[1].
    pub fn set_reset(&mut self, port: BValue, behavior: ResetBehavior) -> Result<(), BuilderError> {
        let node = self.node(port)?;
        if !matches!(
            self.graph.get_node(node).payload,
            NodePayload::InputPort { .. }
        ) || self.graph.get_node_ty(node) != &Type::Bits(1)
        {
            return Err(invalid("block reset must refer to a bits[1] input port"));
        }
        self.state.reset = Some(BlockReset {
            port: node,
            asynchronous: behavior.asynchronous,
            active_low: behavior.active_low,
        });
        Ok(())
    }

    /// Sets or clears a port's SystemVerilog type annotation.
    pub fn set_port_sv_type(
        &mut self,
        port: BValue,
        sv_type: Option<&str>,
    ) -> Result<(), BuilderError> {
        let node = self.node(port)?;
        if matches!(
            self.graph.get_node(node).payload,
            NodePayload::ExtInoutRead { .. }
        ) {
            return self.set_inout_sv_type(self.pin_for_read(node)?, sv_type);
        }
        match &mut self.graph.get_node_mut(node).payload {
            NodePayload::InputPort {
                sv_type: target, ..
            }
            | NodePayload::OutputPort {
                sv_type: target, ..
            } => {
                *target = sv_type.map(str::to_string);
                Ok(())
            }
            _ => Err(invalid(
                "SystemVerilog type annotations require a block port",
            )),
        }
    }

    /// Declares a register with an optional, exactly typed reset value.
    ///
    /// The clock and block reset may be declared later. A complete block must
    /// contain one read and at least one write for every declared register.
    pub fn register(
        &mut self,
        name: &str,
        ty: Type,
        reset_value: Option<IrValue>,
    ) -> Result<BRegister, BuilderError> {
        check_name(name)?;
        checked_flat_width(&ty)?;
        if self
            .state
            .registers
            .iter()
            .any(|register| register.definition.name == name)
        {
            return Err(BuilderError::DuplicateName(name.to_string()));
        }
        if let Some(value) = &reset_value
            && value.type_() != ty
        {
            return Err(invalid(format!(
                "reset value for register '{name}' must have type {ty}, got {}",
                value.type_()
            )));
        }
        let handle = BRegister {
            builder_id: self.id,
            index: self.state.registers.len(),
        };
        self.state.registers.push(RegisterState {
            definition: Register {
                name: name.to_string(),
                ty,
                reset_value,
            },
            read: None,
            writes: Vec::new(),
        });
        Ok(handle)
    }

    /// Checks ownership before accessing a register declaration.
    fn register_index(&self, register: BRegister) -> Result<usize, BuilderError> {
        if register.builder_id != self.id {
            return Err(BuilderError::ForeignRegister);
        }
        Ok(register.index)
    }

    /// Returns a register's current value, reusing its single read node.
    pub fn register_read(&mut self, register: BRegister) -> Result<BValue, BuilderError> {
        let index = self.register_index(register)?;
        if let Some(value) = self.state.registers[index].read {
            return Ok(value);
        }
        let definition = &self.state.registers[index].definition;
        let value = self.add_node(
            NodePayload::RegisterRead {
                register: definition.name.clone(),
            },
            Some(definition.ty.clone()),
        )?;
        self.state.registers[index].read = Some(value);
        Ok(value)
    }

    /// Adds a register write, checking data, enable, and reset consistency.
    ///
    /// Multiple writes are allowed when all have load enables and use the same
    /// reset operand. Their enables' mutual exclusivity is the caller's
    /// responsibility. A reset operand is required exactly when the register
    /// declaration has a reset value; it may be a derived bits[1] signal.
    pub fn register_write(
        &mut self,
        register: BRegister,
        value: BValue,
        options: RegisterWriteOptions,
    ) -> Result<BValue, BuilderError> {
        let index = self.register_index(register)?;
        let arg = self.node(value)?;
        let load_enable = options
            .load_enable
            .map(|value| self.control_node(value))
            .transpose()?;
        let reset = options
            .reset
            .map(|value| self.control_node(value))
            .transpose()?;
        let state = &self.state.registers[index];
        let definition = &state.definition;
        if self.graph.get_node_ty(arg) != &definition.ty {
            return Err(invalid(format!(
                "register '{}' requires {}, got {}",
                definition.name,
                definition.ty,
                self.graph.get_node_ty(arg)
            )));
        }
        if definition.reset_value.is_some() != reset.is_some() {
            return Err(invalid(
                "register reset value and write reset must both be present or both absent",
            ));
        }
        if !state.writes.is_empty()
            && (load_enable.is_none()
                || state
                    .writes
                    .iter()
                    .any(|write| write.load_enable.is_none() || write.reset != reset))
        {
            return Err(invalid(
                "multiple register writes require load enables and the same reset operand",
            ));
        }
        let sink = self.add_node(
            NodePayload::RegisterWrite {
                arg,
                register: definition.name.clone(),
                load_enable,
                reset,
            },
            None,
        )?;
        self.state.registers[index]
            .writes
            .push(RegisterWrite { load_enable, reset });
        Ok(sink)
    }

    /// Checks a one-bit control without mutating builder state.
    fn control_node(&self, value: BValue) -> Result<NodeRef, BuilderError> {
        let node = self.node(value)?;
        if self.graph.get_node_ty(node) != &Type::Bits(1) {
            return Err(invalid(format!(
                "control must be bits[1], got {}",
                self.graph.get_node_ty(node)
            )));
        }
        Ok(node)
    }

    /// Declares an instance using a child block's checked interface snapshot.
    ///
    /// The child must already exist in the package at finalization. Outputs
    /// may be requested before inputs are connected, allowing register
    /// feedback.
    /// All child data inputs must be connected and all child data outputs must
    /// be requested before building. A clocked child requires a parent clock.
    pub fn instantiate(
        &mut self,
        name: &str,
        block: &Block,
    ) -> Result<BInstantiation, BuilderError> {
        check_name(&block.name)?;
        let signature = block_signature(block)?;
        let mut input_types = BTreeMap::new();
        let mut output_types = BTreeMap::new();
        let mut inout_types = BTreeMap::new();
        for port in &signature.ports {
            match port {
                PortSignature::Input(name, ty) => {
                    input_types.insert(name.clone(), ty.clone());
                }
                PortSignature::Output(name, ty) => {
                    output_types.insert(name.clone(), ty.clone());
                }
                PortSignature::Inout(name, ty, _) => {
                    inout_types.insert(name.clone(), ty.clone());
                }
                PortSignature::Clock(_) => {
                    // Clocks are implicit hierarchy connections, not value
                    // nodes.
                }
            }
        }
        self.add_instance(
            name,
            &block.name,
            InstanceSignature::Block(signature),
            input_types,
            output_types,
            inout_types,
        )
    }

    /// Declares an external-function instance, including tuple component ports.
    ///
    /// Parameters use their names and outputs use `return`; tuple components
    /// are addressable as `name.0`, `name.1`, etc., as in XLS IR. The
    /// referenced function must exist with this signature in the
    /// finalization package.
    /// For cycle checking, every output conservatively depends on all connected
    /// inputs: a foreign function's IR body may only be a stub for its RTL.
    pub fn instantiate_extern(
        &mut self,
        name: &str,
        function: &ir::Fn,
    ) -> Result<BInstantiation, BuilderError> {
        check_name(&function.name)?;
        verify_function_signature(function).map_err(|error| invalid(error.to_string()))?;
        let mut input_types = BTreeMap::new();
        let mut output_types = BTreeMap::new();
        for param in function.param_nodes() {
            collect_external_ports(param.param_name(), &param.ty, &mut input_types);
        }
        collect_external_ports("return", &function.ret_ty, &mut output_types);
        self.add_instance(
            name,
            &function.name,
            InstanceSignature::Extern(external_signature(function)),
            input_types,
            output_types,
            BTreeMap::new(),
        )
    }

    /// Records a named instance only after its target contract is consistent.
    fn add_instance(
        &mut self,
        name: &str,
        target: &str,
        signature: InstanceSignature,
        input_types: BTreeMap<String, Type>,
        output_types: BTreeMap<String, Type>,
        inout_types: BTreeMap<String, Type>,
    ) -> Result<BInstantiation, BuilderError> {
        check_name(name)?;
        if target == self.graph.name {
            return Err(invalid("a block cannot instantiate itself"));
        }
        for instance in &self.state.instances {
            if instance.definition.name == name {
                return Err(BuilderError::DuplicateName(name.to_string()));
            }
            if instance.definition.block == target && instance.signature != signature {
                return Err(invalid(format!(
                    "conflicting signatures supplied for instance target '{target}'"
                )));
            }
        }
        let kind = match &signature {
            InstanceSignature::Block(_) => InstantiationKind::Block,
            InstanceSignature::Extern(_) => InstantiationKind::Extern,
        };
        let handle = BInstantiation {
            builder_id: self.id,
            index: self.state.instances.len(),
        };
        self.state.instances.push(InstanceState {
            definition: Instantiation {
                name: name.to_string(),
                block: target.to_string(),
                kind,
            },
            signature,
            input_types,
            output_types,
            inout_types,
            inout_connections: BTreeMap::new(),
            inputs: BTreeSet::new(),
            outputs: BTreeMap::new(),
        });
        Ok(handle)
    }

    /// Checks ownership before accessing an instance declaration.
    fn instance_index(&self, instance: BInstantiation) -> Result<usize, BuilderError> {
        if instance.builder_id != self.id {
            return Err(BuilderError::ForeignInstantiation);
        }
        Ok(instance.index)
    }

    /// Connects one data input exactly once and returns its unit-valued sink.
    pub fn instantiation_input(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        value: BValue,
    ) -> Result<BValue, BuilderError> {
        let index = self.instance_index(instance)?;
        let arg = self.node(value)?;
        let state = &self.state.instances[index];
        let expected = state
            .input_types
            .get(port_name)
            .ok_or_else(|| invalid(format!("unknown instance input '{port_name}'")))?;
        if state.inputs.contains(port_name) {
            return Err(invalid(format!(
                "instance input '{port_name}' is already connected"
            )));
        }
        if self.graph.get_node_ty(arg) != expected {
            return Err(invalid(format!(
                "instance input '{port_name}' requires {expected}, got {}",
                self.graph.get_node_ty(arg)
            )));
        }
        let sink = self.add_node(
            NodePayload::InstantiationInput {
                instantiation: state.definition.name.clone(),
                port_name: port_name.to_string(),
                arg,
            },
            None,
        )?;
        self.state.instances[index]
            .inputs
            .insert(port_name.to_string());
        Ok(sink)
    }

    /// Returns an instance output, reusing its node on repeated requests.
    pub fn instantiation_output(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
    ) -> Result<BValue, BuilderError> {
        let index = self.instance_index(instance)?;
        let state = &self.state.instances[index];
        if let Some(value) = state.outputs.get(port_name) {
            return Ok(*value);
        }
        let ty = state
            .output_types
            .get(port_name)
            .ok_or_else(|| invalid(format!("unknown instance output '{port_name}'")))?
            .clone();
        let value = self.add_node(
            NodePayload::InstantiationOutput {
                instantiation: state.definition.name.clone(),
                port_name: port_name.to_string(),
            },
            Some(ty),
        )?;
        self.state.instances[index]
            .outputs
            .insert(port_name.to_string(), value);
        Ok(value)
    }

    /// Moves graph and resource ownership into a block without cloning nodes.
    fn into_block(self) -> Block {
        let mut block = Block {
            graph: self.graph,
            ports: self.state.ports,
            inout_nets: self.state.inout_nets,
            reset: self.state.reset,
            registers: self
                .state
                .registers
                .into_iter()
                .map(|state| state.definition)
                .collect(),
            instantiations: self
                .state
                .instances
                .into_iter()
                .map(|state| state.definition)
                .collect(),
        };
        if !self.state.private_net_names.is_empty() {
            let mut names = crate::ir_block::InoutNetNames::new(&block);
            let mut renamed = BTreeMap::new();
            for (index, base) in self.state.private_net_names {
                let name = names.allocate(&base);
                let temporary = std::mem::replace(&mut block.inout_nets[index].name, name.clone());
                renamed.insert(temporary, name);
            }
            for node in &mut block.nodes {
                if let NodePayload::ExtInoutRead { net }
                | NodePayload::ExtInoutDrive { net, .. }
                | NodePayload::ExtInstantiationInout { net, .. } = &mut node.payload
                    && let Some(name) = renamed.get(net)
                {
                    net.clone_from(name);
                }
            }
        }
        block
    }

    /// Resolves wire ordering and verifies the completed block in its package.
    fn finish_block(self, package: &Package) -> Result<Block, BuilderError> {
        for &index in &self.state.wires {
            if matches!(self.graph.nodes[index].payload, NodePayload::Param) {
                return Err(invalid(format!("wire {index} has no driver")));
            }
        }
        let has_wires = !self.state.wires.is_empty();
        let mut block = self.into_block();
        if has_wires {
            // Preserve a cycle witness before reordering explicit operands.
            super::block_cycles::verify_block_combinational_cycles(&block, package)?;
            block.compact_and_toposort().map_err(invalid)?;
        }
        verify_block_structure_in_package(&block, package, package.members.len())
            .map_err(|error| invalid(error.to_string()))?;
        if !has_wires && (!block.instantiations.is_empty() || !block.inout_nets.is_empty()) {
            super::block_cycles::verify_block_combinational_cycles(&block, package)?;
        }
        Ok(block)
    }

    /// Consumes and verifies a standalone block with no calls or instances.
    pub fn build(self) -> Result<Block, BuilderError> {
        check_name(&self.graph.name)?;
        if !self.callees.is_empty() || !self.state.instances.is_empty() {
            return Err(invalid(
                "blocks containing calls or instances require package-aware finalization",
            ));
        }
        self.finish_block(&empty_package("builder_context"))
    }

    /// Builds a one-block package with the block selected as top.
    pub fn build_package(self, package_name: &str) -> Result<Package, BuilderError> {
        check_package_name(package_name)?;
        let block = self.build()?;
        let mut package = empty_package(package_name);
        package.top = Some((block.name.clone(), ir::MemberType::Block));
        package.members.push(PackageMember::Block(block));
        Ok(package)
    }

    /// Builds an insertion-ready block against existing, verified members.
    ///
    /// Targets must match the signatures supplied to the builder. IDs are
    /// rebased above all existing nodes; package ownership and top are
    /// unchanged.
    pub fn build_in_package(self, package: &Package) -> Result<Block, BuilderError> {
        check_name(&self.graph.name)?;
        check_package_name(&package.name)?;
        if package
            .members
            .iter()
            .any(|member| member.graph().name == self.graph.name)
        {
            return Err(BuilderError::DuplicateName(self.graph.name.clone()));
        }
        verify_package(package).map_err(|error| invalid(error.to_string()))?;
        self.check_callees(|name| package.get_fn(name))?;
        for instance in &self.state.instances {
            let target = &instance.definition.block;
            let actual = match &instance.signature {
                InstanceSignature::Block(_) => {
                    let block = package.get_block(target).ok_or_else(|| {
                        invalid(format!(
                            "instance target block '{target}' is missing from package"
                        ))
                    })?;
                    if block.clock_port_name().is_some()
                        && !self
                            .state
                            .ports
                            .iter()
                            .any(|port| matches!(port, BlockPort::Clock(_)))
                    {
                        return Err(invalid(format!(
                            "clocked instance target '{target}' requires a clock on the parent block"
                        )));
                    }
                    InstanceSignature::Block(block_signature(block)?)
                }
                InstanceSignature::Extern(_) => {
                    let function = package.get_fn(target).ok_or_else(|| {
                        invalid(format!(
                            "external function '{target}' is missing from package"
                        ))
                    })?;
                    InstanceSignature::Extern(external_signature(function))
                }
            };
            if actual != instance.signature {
                return Err(invalid(format!(
                    "instance target '{target}' in package has a different signature than the one supplied to the builder"
                )));
            }
        }
        let mut block = self.finish_block(package)?;
        rebase_block_ids_in_place(&mut block, package_max_emitted_node_id(package))
            .map_err(|error| invalid(error.to_string()))?;
        Ok(block)
    }

    /// Appends a verified block, leaving the package untouched on failure.
    pub fn build_into_package(self, package: &mut Package) -> Result<(), BuilderError> {
        let block = self.build_in_package(package)?;
        package.members.push(PackageMember::Block(block));
        Ok(())
    }
}

/// Collects an interface without trusting externally constructed node
/// references.
fn block_signature(block: &Block) -> Result<BlockSignature, BuilderError> {
    let mut ports = Vec::with_capacity(block.ports.len());
    let mut names = BTreeSet::new();
    let mut input_names = BTreeMap::new();
    let mut clock_seen = false;
    // Two interfaces with the same public names can short different ports.
    let mut public_nets = BTreeMap::new();
    for net in &block.inout_nets {
        if let Some(first) = net.ports.iter().min() {
            for alias in &net.ports {
                if public_nets.insert(alias, (&net.ty, first)).is_some() {
                    return Err(invalid(format!("duplicate public inout {alias}")));
                }
            }
        }
    }
    for port in &block.ports {
        let signature =
            match port {
                BlockPort::Clock(name) => {
                    if clock_seen {
                        return Err(invalid("instance target has multiple clocks"));
                    }
                    clock_seen = true;
                    PortSignature::Clock(name.clone())
                }
                BlockPort::Inout(pin) => {
                    let (ty, alias) = public_nets
                        .remove(pin)
                        .ok_or_else(|| invalid(format!("inout port '{pin}' has no net")))?;
                    if checked_flat_width(ty)? == 0 || ty.contains_token() {
                        return Err(invalid("invalid instance inout port type"));
                    }
                    PortSignature::Inout(pin.clone(), ty.clone(), alias.clone())
                }
                BlockPort::Input(port) => {
                    let node = block.nodes.get(port.index).ok_or_else(|| {
                        invalid("instance target input references a missing node")
                    })?;
                    let NodePayload::InputPort { name, .. } = &node.payload else {
                        return Err(invalid("instance target input is not an input_port node"));
                    };
                    checked_flat_width(&node.ty)?;
                    input_names.insert(port.index, (name.clone(), node.ty.clone()));
                    PortSignature::Input(name.clone(), node.ty.clone())
                }
                BlockPort::Output(port) => {
                    let node = block.nodes.get(port.index).ok_or_else(|| {
                        invalid("instance target output references a missing node")
                    })?;
                    let NodePayload::OutputPort { name, arg, .. } = &node.payload else {
                        return Err(invalid("instance target output is not an output_port node"));
                    };
                    if node.ty != Type::nil() {
                        return Err(invalid("instance target output node must have unit type"));
                    }
                    let value = block.nodes.get(arg.index).ok_or_else(|| {
                        invalid("instance target output references a missing value")
                    })?;
                    checked_flat_width(&value.ty)?;
                    PortSignature::Output(name.clone(), value.ty.clone())
                }
            };
        let name = match &signature {
            PortSignature::Input(name, _)
            | PortSignature::Inout(name, ..)
            | PortSignature::Output(name, _)
            | PortSignature::Clock(name) => name,
        };
        check_name(name)?;
        if !names.insert(name.clone()) {
            return Err(BuilderError::DuplicateName(name.clone()));
        }
        ports.push(signature);
    }
    if !public_nets.is_empty() {
        return Err(invalid("inout net names a missing public port"));
    }
    let reset = block
        .reset
        .as_ref()
        .map(|reset| {
            let (name, ty) = input_names
                .get(&reset.port.index)
                .ok_or_else(|| invalid("instance target reset must reference an input port"))?;
            if *ty != Type::Bits(1) {
                return Err(invalid("instance target reset must be bits[1]"));
            }
            Ok((
                name.clone(),
                ResetBehavior {
                    asynchronous: reset.asynchronous,
                    active_low: reset.active_low,
                },
            ))
        })
        .transpose()?;
    Ok(BlockSignature { ports, reset })
}

/// Captures named external inputs as well as positional types and return type.
fn external_signature(function: &ir::Fn) -> ExternalSignature {
    ExternalSignature {
        function_type: function.get_type(),
        parameter_names: function
            .param_nodes()
            .map(|node| node.param_name().to_string())
            .collect(),
    }
}

/// Mirrors XLS external port addressing for aggregate tuple components.
fn collect_external_ports(name: &str, ty: &Type, ports: &mut BTreeMap<String, Type>) {
    ports.insert(name.to_string(), ty.clone());
    if let Type::Tuple(elements) = ty {
        for (index, element) in elements.iter().enumerate() {
            collect_external_ports(&format!("{name}.{index}"), element, ports);
        }
    }
}

fn empty_package(name: &str) -> Package {
    Package {
        name: name.to_string(),
        file_table: ir::FileTable::new(),
        members: Vec::new(),
        top: None,
    }
}

fn invalid(reason: impl Into<String>) -> BuilderError {
    BuilderError::InvalidOperation(reason.into())
}
