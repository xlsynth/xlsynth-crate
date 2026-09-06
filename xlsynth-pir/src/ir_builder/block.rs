// SPDX-License-Identifier: Apache-2.0

//! Checked block interfaces, sequential resources, and instantiations.

use std::collections::{BTreeMap, BTreeSet};

use super::{BValue, Builder, BuilderError, check_name, checked_flat_width};
use crate::IrValue;
use crate::ir::{
    self, Block, BlockPort, BlockReset, Instantiation, InstantiationKind, NodePayload, NodeRef,
    Package, PackageMember, Register, Type,
};
use crate::ir_rebase_ids::{package_max_emitted_node_id, rebase_block_ids_in_place};
use crate::ir_verify::{verify_block_in_package, verify_function_signature, verify_package};

/// A register belonging to exactly one block builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BRegister {
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

struct InstanceState {
    definition: Instantiation,
    signature: InstanceSignature,
    input_types: BTreeMap<String, Type>,
    output_types: BTreeMap<String, Type>,
    inputs: BTreeSet<String>,
    outputs: BTreeMap<String, BValue>,
}

/// Block-specific state; use `BlockBuilder` to construct and finalize it.
#[doc(hidden)]
#[derive(Default)]
pub struct BlockState {
    ports: Vec<BlockPort>,
    port_names: BTreeMap<String, Option<NodeRef>>,
    reset: Option<BlockReset>,
    registers: Vec<RegisterState>,
    instances: Vec<InstanceState>,
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
        for port in &signature.ports {
            match port {
                PortSignature::Input(name, ty) => {
                    input_types.insert(name.clone(), ty.clone());
                }
                PortSignature::Output(name, ty) => {
                    output_types.insert(name.clone(), ty.clone());
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
        Block {
            graph: self.graph,
            ports: self.state.ports,
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

    /// Consumes and verifies a standalone block with no calls or instances.
    pub fn build(self) -> Result<Block, BuilderError> {
        check_name(&self.graph.name)?;
        if !self.callees.is_empty() || !self.state.instances.is_empty() {
            return Err(invalid(
                "blocks containing calls or instances require package-aware finalization",
            ));
        }
        let block = self.into_block();
        let package = empty_package("builder_context");
        verify_block_in_package(&block, &package, 0).map_err(|error| invalid(error.to_string()))?;
        Ok(block)
    }

    /// Builds a one-block package with the block selected as top.
    pub fn build_package(self, package_name: &str) -> Result<Package, BuilderError> {
        check_name(package_name)?;
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
        check_name(&package.name)?;
        if package
            .members
            .iter()
            .any(|member| member.graph().name == self.graph.name)
        {
            return Err(BuilderError::DuplicateName(self.graph.name.clone()));
        }
        verify_package(package).map_err(|error| invalid(error.to_string()))?;
        self.check_callees(package)?;
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
        let mut block = self.into_block();
        verify_block_in_package(&block, package, package.members.len())
            .map_err(|error| invalid(error.to_string()))?;
        if !block.instantiations.is_empty() {
            super::block_cycles::verify_block_combinational_cycles(&block, package)?;
        }
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
