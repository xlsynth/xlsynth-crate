// SPDX-License-Identifier: Apache-2.0

//! Checked block interfaces, sequential resources, and instantiations.

use std::collections::{BTreeMap, BTreeSet};

use super::block_cycles::BlockHierarchy;
use super::{BValue, Builder, BuilderError, check_name, check_package_name, checked_flat_width};
use crate::IrValue;
use crate::ir::{
    self, Block, BlockPort, BlockReset, Instantiation, InstantiationKind, NodePayload, NodeRef,
    OpaqueExternPort, OpaqueExternPortKind, Package, PackageMember, Register, Type, Unop,
};
use crate::ir_rebase_ids::{package_max_emitted_node_id, rebase_block_ids_in_place};
use crate::ir_utils::is_valid_identifier_name;
use crate::ir_verify::{
    CalleeLookup, ExternModelBindings, validate_block_with_targets, validate_opaque_extern_ports,
    verify_function_signature, verify_package,
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
    Opaque,
}

/// Signature and connections accumulated for a single instance.
struct InstanceState {
    definition: Instantiation,
    signature: InstanceSignature,
    input_types: BTreeMap<String, Type>,
    output_types: BTreeMap<String, Type>,
    inout_types: BTreeMap<String, Type>,
    inout_connections: BTreeSet<String>,
    inputs: BTreeSet<String>,
    outputs: BTreeMap<String, BValue>,
}

/// Block-specific state; use `BlockBuilder` to construct and finalize it.
#[doc(hidden)]
#[derive(Default)]
pub struct BlockState {
    ports: Vec<BlockPort>,
    inout_nets: Vec<ir::InoutNet>,
    inout_names: BTreeSet<String>,
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

    /// Declares a block-local resolved wire.
    pub fn declare_inout_net(&mut self, name: &str, ty: Type) -> Result<BInoutNet, BuilderError> {
        check_name(name)?;
        if checked_flat_width(&ty)? == 0 || ty.contains_token() {
            return Err(invalid("inout nets require a nonempty, token-free type"));
        }
        if !self.state.inout_names.insert(name.to_owned()) {
            return Err(invalid(format!("duplicate inout net '{name}'")));
        }
        let index = self.state.inout_nets.len();
        self.state.inout_nets.push(ir::InoutNet {
            name: name.to_owned(),
            ty,
        });
        Ok(BInoutNet {
            builder_id: self.id,
            index,
        })
    }

    /// Binds a public inout pin to a whole local net.
    pub fn ext_inout_port(
        &mut self,
        name: &str,
        port_type: Type,
        net: BInoutNet,
    ) -> Result<BValue, BuilderError> {
        self.check_node_name(name, None)?;
        if checked_flat_width(&port_type)? == 0 || port_type.contains_token() {
            return Err(invalid("inout ports require a nonempty, token-free type"));
        }
        let def = self.net_definition(net)?;
        if def.ty != port_type {
            return Err(invalid(format!(
                "inout port '{name}' has type {port_type}, net '{}' has type {}",
                def.name, def.ty
            )));
        }
        let port = self.add_node(
            NodePayload::ExtInoutPort {
                name: name.to_owned(),
                port_type,
                net: def.name.clone(),
                sv_type: None,
            },
            Some(Type::nil()),
        )?;
        self.graph.get_node_mut(port.node).name = Some(name.to_owned());
        self.names.insert(name.to_owned(), port.node);
        self.state
            .port_names
            .insert(name.to_owned(), Some(port.node));
        self.state.ports.push(BlockPort::Inout(port.node));
        Ok(port)
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
        if state.inout_connections.contains(port_name) {
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
            .insert(port_name.to_owned());
        Ok(link)
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
        match &mut self.graph.get_node_mut(node).payload {
            NodePayload::InputPort {
                sv_type: target, ..
            }
            | NodePayload::OutputPort {
                sv_type: target, ..
            }
            | NodePayload::ExtInoutPort {
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
        self.instantiate_block(name, block, InstantiationKind::Block)
    }

    /// Declares an RTL module instance backed by a block in the same package.
    ///
    /// Port types, clock, and combinational dependencies come from the model.
    /// Connect or request inputs, outputs, and inouts as for a child block.
    pub fn instantiate_extern_block(
        &mut self,
        name: &str,
        model: &Block,
        module_name: &str,
    ) -> Result<BInstantiation, BuilderError> {
        self.instantiate_extern_block_with_parameters(name, model, module_name, BTreeMap::new())
    }

    /// Declares an external module with named SystemVerilog parameter
    /// overrides.
    ///
    /// The model must already match this configuration. The overrides are
    /// emitted as expression text and do not specialize the model.
    pub fn instantiate_extern_block_with_parameters(
        &mut self,
        name: &str,
        model: &Block,
        module_name: &str,
        parameters: BTreeMap<String, String>,
    ) -> Result<BInstantiation, BuilderError> {
        if !is_valid_identifier_name(module_name) {
            return Err(BuilderError::InvalidName(module_name.to_string()));
        }
        ir::validate_extern_block_parameters(name, &parameters).map_err(invalid)?;
        self.instantiate_block(
            name,
            model,
            InstantiationKind::BehavioralExternBlock {
                module_name: module_name.to_string(),
                parameters,
                input_casts: BTreeMap::new(),
            },
        )
    }

    /// Declares an opaque RTL module with an inline, typed interface.
    ///
    /// Bind its inputs, outputs, and inouts as for a child block. If the
    /// interface declares a clock, it connects to the enclosing block's clock.
    pub fn instantiate_opaque_extern_block(
        &mut self,
        name: &str,
        module_name: &str,
        ports: Vec<OpaqueExternPort>,
    ) -> Result<BInstantiation, BuilderError> {
        self.instantiate_opaque_extern_block_with_parameters(
            name,
            module_name,
            ports,
            BTreeMap::new(),
        )
    }

    /// Declares an opaque RTL module with named SystemVerilog parameter
    /// overrides.
    ///
    /// The inline port types must match the resulting RTL configuration.
    pub fn instantiate_opaque_extern_block_with_parameters(
        &mut self,
        name: &str,
        module_name: &str,
        ports: Vec<OpaqueExternPort>,
        parameters: BTreeMap<String, String>,
    ) -> Result<BInstantiation, BuilderError> {
        if !is_valid_identifier_name(module_name) {
            return Err(BuilderError::InvalidName(module_name.to_string()));
        }
        ir::validate_extern_block_parameters(name, &parameters).map_err(invalid)?;
        validate_opaque_extern_ports(&ports).map_err(invalid)?;
        let mut input_types = BTreeMap::new();
        let mut output_types = BTreeMap::new();
        let mut inout_types = BTreeMap::new();
        for port in &ports {
            match &port.kind {
                OpaqueExternPortKind::Input(ty) => {
                    input_types.insert(port.name.clone(), ty.clone());
                }
                OpaqueExternPortKind::Output(ty) => {
                    output_types.insert(port.name.clone(), ty.clone());
                }
                OpaqueExternPortKind::Inout(ty) => {
                    inout_types.insert(port.name.clone(), ty.clone());
                }
                OpaqueExternPortKind::Clock => {
                    // The pin is bound to the parent clock by codegen.
                }
            }
        }
        self.add_instance(
            Instantiation {
                name: name.to_string(),
                block: module_name.to_string(),
                kind: InstantiationKind::OpaqueExternBlock {
                    ports,
                    parameters,
                    input_casts: BTreeMap::new(),
                },
            },
            InstanceSignature::Opaque,
            input_types,
            output_types,
            inout_types,
        )
    }

    /// Checks and records the model interface shared by both kinds of block
    /// instance.
    fn instantiate_block(
        &mut self,
        name: &str,
        block: &Block,
        kind: InstantiationKind,
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
            Instantiation {
                name: name.to_string(),
                block: block.name.clone(),
                kind,
            },
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
    /// The foreign function is opaque to combinational cycle checking.
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
            Instantiation {
                name: name.to_string(),
                block: function.name.clone(),
                kind: InstantiationKind::Extern,
            },
            InstanceSignature::Extern(external_signature(function)),
            input_types,
            output_types,
            BTreeMap::new(),
        )
    }

    /// Records a named instance only after its target contract is consistent.
    fn add_instance(
        &mut self,
        definition: Instantiation,
        signature: InstanceSignature,
        input_types: BTreeMap<String, Type>,
        output_types: BTreeMap<String, Type>,
        inout_types: BTreeMap<String, Type>,
    ) -> Result<BInstantiation, BuilderError> {
        let name = definition.name.as_str();
        let target = definition.block.as_str();
        check_name(name)?;
        let has_package_target = !matches!(signature, InstanceSignature::Opaque);
        if has_package_target && target == self.graph.name {
            return Err(invalid("a block cannot instantiate itself"));
        }
        for instance in &self.state.instances {
            if instance.definition.name == name {
                return Err(BuilderError::DuplicateName(name.to_string()));
            }
            if has_package_target
                && !matches!(instance.signature, InstanceSignature::Opaque)
                && instance.definition.block == target
                && instance.signature != signature
            {
                return Err(invalid(format!(
                    "conflicting signatures supplied for instance target '{target}'"
                )));
            }
        }
        let handle = BInstantiation {
            builder_id: self.id,
            index: self.state.instances.len(),
        };
        self.state.instances.push(InstanceState {
            definition,
            signature,
            input_types,
            output_types,
            inout_types,
            inout_connections: BTreeSet::new(),
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

    /// Assigns the SystemVerilog typedef used to cast one external block input.
    ///
    /// Accepts a simple or package-qualified type name. PIR still type-checks
    /// the connection against the behavioral model or the declared interface.
    pub fn set_instantiation_input_cast(
        &mut self,
        instance: BInstantiation,
        port_name: &str,
        sv_type: &str,
    ) -> Result<(), BuilderError> {
        let index = self.instance_index(instance)?;
        let state = &mut self.state.instances[index];
        let input_casts = match &mut state.definition.kind {
            InstantiationKind::BehavioralExternBlock { input_casts, .. }
            | InstantiationKind::OpaqueExternBlock { input_casts, .. } => input_casts,
            InstantiationKind::Block | InstantiationKind::Extern => {
                return Err(invalid("input casts require an external block instance"));
            }
        };
        let input_type = state
            .input_types
            .get(port_name)
            .ok_or_else(|| invalid(format!("unknown instance input '{port_name}'")))?;
        ir::validate_extern_block_input_cast(
            &state.definition.name,
            port_name,
            input_type,
            sv_type,
        )
        .map_err(invalid)?;
        if input_casts.contains_key(port_name) {
            return Err(invalid(format!(
                "instance input '{port_name}' already has a SystemVerilog cast"
            )));
        }
        input_casts.insert(port_name.to_string(), sv_type.to_string());
        Ok(())
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
        Block {
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
        }
    }

    /// Resolves wire ordering and verifies the completed block in its package.
    fn finish_block<'p>(
        self,
        get_block: &dyn Fn(&str) -> Option<&'p Block>,
        get_fn: &dyn Fn(&str) -> Option<&'p ir::Fn>,
        hierarchy: &mut BlockHierarchy,
    ) -> Result<Block, BuilderError> {
        for &index in &self.state.wires {
            if matches!(self.graph.nodes[index].payload, NodePayload::Param) {
                return Err(invalid(format!("wire {index} has no driver")));
            }
        }
        let has_wires = !self.state.wires.is_empty();
        let mut block = self.into_block();
        if has_wires {
            // Preserve a cycle witness before reordering explicit operands.
            hierarchy.check(&block, get_block)?;
            block.compact_and_toposort().map_err(invalid)?;
        }
        validate_block_with_targets(&block, get_block, CalleeLookup::Verified(get_fn))
            .map_err(|error| invalid(error.to_string()))?;
        if !has_wires && (!block.instantiations.is_empty() || !block.inout_nets.is_empty()) {
            hierarchy.check(&block, get_block)?;
        }
        Ok(block)
    }

    /// Consumes and verifies a standalone block without package references.
    pub fn build(self) -> Result<Block, BuilderError> {
        check_name(&self.graph.name)?;
        if !self.callees.is_empty()
            || self
                .state
                .instances
                .iter()
                .any(|instance| !matches!(instance.signature, InstanceSignature::Opaque))
        {
            return Err(invalid(
                "blocks containing calls or instances require package-aware finalization",
            ));
        }
        self.finish_block(&|_| None, &|_| None, &mut BlockHierarchy::default())
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
        let block = self.finish_for_verified_package(
            &|name| package.get_block(name),
            &|name| package.get_fn(name),
            package_max_emitted_node_id(package),
            &mut BlockHierarchy::default(),
        )?;
        if block.instantiations.iter().any(|instance| {
            matches!(
                instance.kind,
                InstantiationKind::BehavioralExternBlock { .. }
            )
        }) {
            ExternModelBindings::from_package(package)
                .and_then(|mut bindings| bindings.try_insert_block(&block))
                .map_err(|error| invalid(error.to_string()))?;
        }
        Ok(block)
    }

    /// Verifies this block and allocates IDs above those in its verified
    /// destination.
    pub(super) fn finish_for_verified_package<'p>(
        self,
        get_block: &dyn Fn(&str) -> Option<&'p Block>,
        get_fn: &dyn Fn(&str) -> Option<&'p ir::Fn>,
        max_node_id: usize,
        hierarchy: &mut BlockHierarchy,
    ) -> Result<Block, BuilderError> {
        self.check_callees(get_fn)?;
        for instance in &self.state.instances {
            let target = &instance.definition.block;
            let actual = match &instance.signature {
                InstanceSignature::Block(_) => {
                    let block = get_block(target).ok_or_else(|| {
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
                    let function = get_fn(target).ok_or_else(|| {
                        invalid(format!(
                            "external function '{target}' is missing from package"
                        ))
                    })?;
                    InstanceSignature::Extern(external_signature(function))
                }
                InstanceSignature::Opaque => {
                    // Its signature is inline, so there is no package member to
                    // compare.
                    continue;
                }
            };
            if actual != instance.signature {
                return Err(invalid(format!(
                    "instance target '{target}' in package has a different signature than the one supplied to the builder"
                )));
            }
        }
        let mut block = self.finish_block(get_block, get_fn, hierarchy)?;
        rebase_block_ids_in_place(&mut block, max_node_id)
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
    // Compare which public pins are tied, independently of internal net names.
    let mut net_representatives = BTreeMap::new();
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
                BlockPort::Inout(port) => {
                    let node = block.nodes.get(port.index).ok_or_else(|| {
                        invalid("instance target inout references a missing node")
                    })?;
                    let NodePayload::ExtInoutPort {
                        name,
                        port_type,
                        net,
                        ..
                    } = &node.payload
                    else {
                        return Err(invalid(
                            "instance target inout is not an ext_inout_port node",
                        ));
                    };
                    if node.ty != Type::nil()
                        || checked_flat_width(port_type)? == 0
                        || port_type.contains_token()
                    {
                        return Err(invalid("invalid instance inout port type"));
                    }
                    let representative = *net_representatives.entry(net).or_insert(name);
                    PortSignature::Inout(name.clone(), port_type.clone(), representative.clone())
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
