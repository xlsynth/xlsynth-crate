// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use xlsynth::IrValue;
use xlsynth::ir_value::IrFormatPreference;
use xlsynth::vast_helpers::{RegisterDefinition, RegisterScope, Reset};
use xlsynth_pir::ir::{
    Binop, BlockMetadata, Fn, InstantiationKind, MemberType, NodePayload, NodeRef, Package,
    PackageMember, Register, Type, Unop,
};
use xlsynth_pir::ir_utils::{get_topological, operands};
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_vast::{
    Expr, IndexableExpr, LiteralFormat, LogicRef, VastDataType, VastFile, VastFileType, VastModule,
};

use crate::stages::{Stage, reconstruct_stages};
use crate::{BlockCodegenError, BlockCodegenOptions, BlockCodegenOutput, Layout};

/// Holds a packed SystemVerilog value and its expression-inlining information.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Value {
    pub(crate) expr: Expr,
    pub(crate) indexable: Option<IndexableExpr>,
    pub(crate) depth: usize,
    pub(crate) bit_width: Option<usize>,
    pub(crate) static_origin: Option<(IndexableExpr, usize)>,
    pub(crate) array_rank: usize,
}

/// Collects registers whose writes use the same emitted reset expression.
struct ResettableRegisterGroup {
    signal: Expr,
    registers: Vec<RegisterDefinition>,
}

impl Value {
    /// Represents a declared, directly indexable SystemVerilog signal.
    pub(crate) fn signal(signal: LogicRef) -> Self {
        let indexable = signal.to_indexable_expr();
        Self {
            expr: signal.to_expr(),
            indexable: Some(indexable),
            depth: 0,
            bit_width: None,
            static_origin: Some((indexable, 0)),
            array_rank: 0,
        }
    }

    /// Represents an inline expression of the requested nesting depth.
    pub(crate) fn expression(expr: Expr, depth: usize) -> Self {
        Self {
            expr,
            indexable: None,
            depth,
            bit_width: None,
            static_origin: None,
            array_rank: 0,
        }
    }

    /// Preserves the original packed signal for nested, portable part-selects.
    pub(crate) fn static_slice(expr: Expr, source: Self, start: usize, width: usize) -> Self {
        Self {
            expr,
            indexable: None,
            depth: source.depth + 1,
            bit_width: Some(width),
            static_origin: source
                .static_origin
                .map(|(origin, offset)| (origin, offset + start)),
            array_rank: 0,
        }
    }

    /// Records a known packed width for scalar-index and slice legalization.
    pub(crate) fn with_width(mut self, bit_width: usize) -> Self {
        self.bit_width = Some(bit_width);
        self
    }

    /// Records the dimensions of a declared packed value, keeping bit slices
    /// distinct from array-element selections.
    pub(crate) fn with_type(mut self, mut ty: &Type) -> Self {
        self.bit_width = Some(ty.bit_count());
        self.array_rank = 0;
        while let Type::Array(array) = ty {
            self.array_rank += 1;
            ty = &array.element_type;
        }
        if self.array_rank != 0 {
            self.static_origin = None;
        }
        self
    }
}

/// Builds one module while preserving the source block's interfaces and state.
pub(crate) struct BlockEmitter<'a, 'file> {
    pub(crate) package: &'a Package,
    pub(crate) options: &'a BlockCodegenOptions,
    pub(crate) func: &'a Fn,
    pub(crate) metadata: &'a BlockMetadata,
    pub(crate) file: &'file mut VastFile,
    pub(crate) module: VastModule,
    pub(crate) values: Vec<Option<Value>>,
    pub(crate) ports: BTreeMap<String, LogicRef>,
    pub(crate) register_refs: BTreeMap<String, LogicRef>,
    pub(crate) users: Vec<Vec<NodeRef>>,
    pub(crate) names: BTreeSet<String>,
    pub(crate) genvar_names: BTreeMap<usize, String>,
    pub(crate) priority_helpers: BTreeMap<crate::priority::PriorityHelper, String>,
    pub(crate) arithmetic_helpers: BTreeMap<crate::arithmetic::ArithmeticHelper, String>,
    pub(crate) slice_helpers: BTreeMap<crate::slicing::SliceHelper, String>,
    pub(crate) current_stage: Option<usize>,
    pub(crate) instance_inputs: BTreeMap<(String, String), Expr>,
    pub(crate) instance_outputs: BTreeMap<(String, String), Expr>,
}

/// Emits a package after validating its selected block top.
pub(crate) fn emit_package(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<BlockCodegenOutput, BlockCodegenError> {
    let top = select_top(package, options.top.as_deref())?;
    verify_package(package).map_err(|error| BlockCodegenError::InvalidBlock(error.to_string()))?;

    let mut ordered = Vec::new();
    let mut visited = BTreeSet::new();
    let mut visiting = BTreeSet::new();
    collect_dependencies(package, top, &mut ordered, &mut visited, &mut visiting)?;
    for &(func, metadata) in &ordered {
        validate_register_structure(func, metadata)?;
    }
    crate::hierarchy::verify_hierarchy(&ordered)?;
    if let Some(module_name) = options.module_name.as_deref()
        && ordered
            .iter()
            .any(|(func, _)| func.name != top && func.name == module_name)
    {
        return Err(BlockCodegenError::InvalidBlock(format!(
            "top module name `{module_name}` conflicts with an instantiated child block; \
             choose a distinct module name"
        )));
    }

    let mut file = VastFile::new(VastFileType::SystemVerilog);
    for (func, metadata) in ordered {
        let module_name = if func.name == top {
            options.module_name.as_deref().unwrap_or(&func.name)
        } else {
            &func.name
        };
        validate_external_identifier(module_name, "module")?;
        let module = file.add_module(module_name);
        let mut emitter = BlockEmitter::new(package, options, func, metadata, &mut file, module);
        emitter.emit()?;
    }

    Ok(BlockCodegenOutput {
        system_verilog: file.emit(),
    })
}

/// Enforces complete register structure at the SystemVerilog emission boundary.
fn validate_register_structure(
    func: &Fn,
    metadata: &BlockMetadata,
) -> Result<(), BlockCodegenError> {
    if !metadata.registers.is_empty() && metadata.clock_port_name.is_none() {
        return Err(BlockCodegenError::InvalidBlock(format!(
            "block `{}` has registers but no clock port",
            func.name
        )));
    }

    if let Some(reset) = &metadata.reset {
        let reset_port = func
            .params
            .iter()
            .find(|parameter| parameter.name == reset.port_name)
            .ok_or_else(|| {
                BlockCodegenError::InvalidBlock(format!(
                    "block `{}` has no reset port `{}`",
                    func.name, reset.port_name
                ))
            })?;
        if reset_port.ty != Type::Bits(1) {
            return Err(BlockCodegenError::InvalidBlock(format!(
                "reset port `{}` in block `{}` must have type bits[1]",
                reset.port_name, func.name
            )));
        }
    }

    let mut access_counts = metadata
        .registers
        .iter()
        .map(|register| (register.name.as_str(), (0usize, 0usize)))
        .collect::<BTreeMap<_, _>>();
    for node in &func.nodes {
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                if let Some((reads, _)) = access_counts.get_mut(register.as_str()) {
                    *reads += 1;
                }
            }
            NodePayload::RegisterWrite { register, .. } => {
                if let Some((_, writes)) = access_counts.get_mut(register.as_str()) {
                    *writes += 1;
                }
            }
            _ => {}
        }
    }

    for register in &metadata.registers {
        let (reads, writes) = access_counts[register.name.as_str()];
        if reads != 1 {
            return Err(BlockCodegenError::InvalidBlock(format!(
                "register `{}` in block `{}` requires exactly one read, found {reads}",
                register.name, func.name
            )));
        }
        if writes == 0 {
            return Err(BlockCodegenError::InvalidBlock(format!(
                "register `{}` in block `{}` has no write",
                register.name, func.name
            )));
        }
        if writes != 1 {
            return Err(BlockCodegenError::InvalidBlock(format!(
                "register `{}` in block `{}` requires exactly one write, found {writes}",
                register.name, func.name
            )));
        }
        if register.reset_value.is_some() && metadata.reset.is_none() {
            return Err(BlockCodegenError::InvalidBlock(format!(
                "register `{}` in block `{}` has a reset value but the block has no reset port",
                register.name, func.name
            )));
        }
    }

    Ok(())
}

/// Rejects operations that must be lowered before SystemVerilog emission.
fn validate_supported_nodes(func: &Fn) -> Result<(), BlockCodegenError> {
    if func
        .nodes
        .iter()
        .any(|node| matches!(node.payload, NodePayload::CountedFor { .. }))
    {
        return Err(BlockCodegenError::Unsupported(format!(
            "counted_for is not supported in `{}`; unroll loops before block2sv code generation",
            func.name
        )));
    }
    if func
        .nodes
        .iter()
        .any(|node| matches!(node.payload, NodePayload::Invoke { .. }))
    {
        return Err(BlockCodegenError::Unsupported(format!(
            "invoke is not supported in `{}`; inline function calls before block2sv code generation",
            func.name
        )));
    }
    Ok(())
}

/// Selects the explicitly requested, marked, or uniquely available block.
fn select_top<'a>(
    package: &'a Package,
    explicit_top: Option<&str>,
) -> Result<&'a str, BlockCodegenError> {
    let blocks = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Block { func, .. } => Some(func.name.as_str()),
            PackageMember::Function(_) => None,
        })
        .collect::<Vec<_>>();

    if let Some(requested) = explicit_top {
        return blocks
            .into_iter()
            .find(|name| *name == requested)
            .ok_or_else(|| {
                BlockCodegenError::TopSelection(format!(
                    "requested top block `{requested}` does not exist"
                ))
            });
    }

    if let Some((name, MemberType::Block)) = &package.top {
        return blocks
            .into_iter()
            .find(|candidate| candidate == &name.as_str())
            .ok_or_else(|| {
                BlockCodegenError::TopSelection(format!(
                    "package top block `{name}` does not exist"
                ))
            });
    }

    match blocks.as_slice() {
        [only] => Ok(only),
        [] => Err(BlockCodegenError::TopSelection(
            "package contains no block suitable for SystemVerilog generation".to_owned(),
        )),
        _ => Err(BlockCodegenError::TopSelection(format!(
            "package contains multiple blocks ({}) without a block top; specify --top",
            blocks.join(", ")
        ))),
    }
}

/// Appends each transitive child once, before the block which instantiates it.
fn collect_dependencies<'a>(
    package: &'a Package,
    name: &'a str,
    ordered: &mut Vec<(&'a Fn, &'a BlockMetadata)>,
    visited: &mut BTreeSet<&'a str>,
    visiting: &mut BTreeSet<&'a str>,
) -> Result<(), BlockCodegenError> {
    if visited.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name) {
        return Err(BlockCodegenError::InvalidBlock(format!(
            "block instantiation cycle includes `{name}`"
        )));
    }
    let (func, metadata) = package
        .members
        .iter()
        .find_map(|member| match member {
            PackageMember::Block { func, metadata } if func.name == name => Some((func, metadata)),
            _ => None,
        })
        .ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "instantiated block `{name}` does not exist in the package"
            ))
        })?;

    for instance in &metadata.instantiations {
        if instance.kind == InstantiationKind::Block {
            collect_dependencies(package, &instance.block, ordered, visited, visiting)?;
        }
    }
    visiting.remove(name);
    visited.insert(name);
    ordered.push((func, metadata));
    Ok(())
}

impl<'a, 'file> BlockEmitter<'a, 'file> {
    /// Creates an emitter before constructing the module's public ports.
    fn new(
        package: &'a Package,
        options: &'a BlockCodegenOptions,
        func: &'a Fn,
        metadata: &'a BlockMetadata,
        file: &'file mut VastFile,
        module: VastModule,
    ) -> Self {
        let mut users = vec![Vec::new(); func.nodes.len()];
        for (index, node) in func.nodes.iter().enumerate() {
            for operand in operands(&node.payload) {
                users[operand.index].push(NodeRef { index });
            }
        }
        Self {
            package,
            options,
            func,
            metadata,
            file,
            module,
            values: vec![None; func.nodes.len()],
            ports: BTreeMap::new(),
            register_refs: BTreeMap::new(),
            users,
            names: BTreeSet::new(),
            genvar_names: BTreeMap::new(),
            priority_helpers: BTreeMap::new(),
            arithmetic_helpers: BTreeMap::new(),
            slice_helpers: BTreeMap::new(),
            current_stage: None,
            instance_inputs: BTreeMap::new(),
            instance_outputs: BTreeMap::new(),
        }
    }

    /// Emits ports, logic, state, output assignments, and child instances.
    fn emit(&mut self) -> Result<(), BlockCodegenError> {
        validate_supported_nodes(self.func)?;
        self.reserve_fixed_names()?;
        self.emit_ports()?;
        self.emit_arithmetic_helpers()?;
        self.emit_slice_helpers()?;
        self.emit_priority_helpers(self.func)?;
        match self.options.layout {
            Layout::None => self.emit_flat()?,
            Layout::Pipeline => self.emit_pipeline()?,
        }
        self.emit_outputs()?;
        self.emit_instantiations()?;
        Ok(())
    }

    /// Rejects conflicting fixed identifiers and protects them from generated
    /// declarations throughout this module, regardless of emission order.
    fn reserve_fixed_names(&mut self) -> Result<(), BlockCodegenError> {
        let mut fixed_names = BTreeMap::new();
        let mut reserve = |name: &str, kind: &'static str| {
            validate_external_identifier(name, kind)?;
            if let Some(previous) = fixed_names.insert(name.to_owned(), kind) {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "SystemVerilog name collision in block `{}`: `{name}` is used by both {previous} and {kind}",
                    self.func.name
                )));
            }
            Ok(())
        };
        if let Some(clock) = &self.metadata.clock_port_name {
            reserve(clock, "port")?;
        }
        for parameter in &self.func.params {
            if parameter.ty.bit_count() != 0 {
                reserve(&parameter.name, "port")?;
            }
        }
        for (name, node) in self.metadata.output_names.iter().zip(self.output_nodes()?) {
            if self.func.get_node_ty(node).bit_count() != 0 {
                reserve(name, "port")?;
            }
        }
        for instance in &self.metadata.instantiations {
            reserve(&instance.name, "instance")?;
        }
        for node in &self.func.nodes {
            match &node.payload {
                NodePayload::Assert { label, .. }
                    if self.options.emit_asserts && !label.is_empty() =>
                {
                    reserve(label, "assertion label")?;
                }
                NodePayload::Cover { label, .. } => reserve(label, "coverage label")?,
                _ => {
                    // Other nodes have generated names or introduce no fixed
                    // declaration in the module scope.
                }
            }
        }
        self.names.extend(fixed_names.into_keys());
        Ok(())
    }

    /// Emits representable ports in the source block's original header order.
    fn emit_ports(&mut self) -> Result<(), BlockCodegenError> {
        let outputs = self.output_nodes()?;
        let output_types = self
            .metadata
            .output_names
            .iter()
            .zip(outputs)
            .map(|(name, node)| (name.as_str(), &self.func.get_node(node).ty))
            .collect::<BTreeMap<_, _>>();

        let order = if self.metadata.port_order.is_empty() {
            self.metadata
                .clock_port_name
                .iter()
                .chain(self.func.params.iter().map(|param| &param.name))
                .chain(self.metadata.output_names.iter())
                .cloned()
                .collect::<Vec<_>>()
        } else {
            self.metadata.port_order.clone()
        };

        for name in order {
            if self.ports.contains_key(&name) {
                continue;
            }
            let is_clock = self.metadata.clock_port_name.as_deref() == Some(name.as_str());
            let input = self.func.params.iter().find(|param| param.name == name);
            let output = output_types.get(name.as_str());
            let width = if is_clock {
                1
            } else if let Some(param) = input {
                param.ty.bit_count()
            } else if let Some(ty) = output {
                ty.bit_count()
            } else {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "port `{name}` in block `{}` has no input or output declaration",
                    self.func.name
                )));
            };
            if width == 0 {
                continue;
            }
            validate_external_identifier(&name, "port")?;
            let ir_type = if let Some(param) = input {
                &param.ty
            } else if let Some(ty) = output {
                ty
            } else {
                &Type::Bits(1)
            };
            let data_type = if self.options.emit_sv_types {
                if let Some(sv_type) = self.metadata.port_sv_types.get(&name) {
                    if let Some((package, ty)) = sv_type.rsplit_once("::") {
                        self.file.make_extern_package_type(package, ty)
                    } else {
                        self.file.make_extern_type(sv_type)
                    }
                } else {
                    self.value_type(ir_type)
                }
            } else {
                self.value_type(ir_type)
            };
            let signal = if output.is_some() && !is_clock && input.is_none() {
                self.file.add_logic_output(self.module, &name, &data_type)
            } else {
                self.file.add_logic_input(self.module, &name, &data_type)
            };
            self.ports.insert(name, signal);
        }

        for (index, node) in self.func.nodes.iter().enumerate() {
            if let NodePayload::GetParam(param_id) = node.payload {
                if let Some(param) = self.func.params.iter().find(|param| param.id == param_id) {
                    if let Some(signal) = self.ports.get(&param.name).copied() {
                        let value = Value::signal(signal).with_width(node.ty.bit_count());
                        self.values[index] = Some(
                            if self.options.emit_sv_types
                                && self.metadata.port_sv_types.contains_key(&param.name)
                            {
                                value
                            } else {
                                value.with_type(&node.ty)
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Emits existing register state and ordinary dependency-ordered logic.
    fn emit_flat(&mut self) -> Result<(), BlockCodegenError> {
        let start = self.file.module_member_count(self.module);
        self.emit_input_array_views()?;
        for register in &self.metadata.registers {
            self.declare_register(register)?;
        }
        let mut writes = Vec::new();
        for node_ref in get_topological(self.func) {
            if self.metadata.output_names.len() != 1 && self.func.ret_node_ref == Some(node_ref) {
                continue;
            }
            match &self.func.get_node(node_ref).payload {
                NodePayload::Nil | NodePayload::GetParam(_) => {
                    // Reserved nodes and parameters are represented by module
                    // ports.
                }
                NodePayload::RegisterRead { register } => {
                    if let Some(signal) = self.register_refs.get(register).copied() {
                        self.values[node_ref.index] =
                            Some(Value::signal(signal).with_type(self.func.get_node_ty(node_ref)));
                    }
                }
                NodePayload::RegisterWrite { .. } => writes.push(node_ref),
                _ => self.emit_node(node_ref)?,
            }
        }
        self.emit_register_groups(&writes)?;
        self.file.hoist_module_declarations(self.module, start);
        Ok(())
    }

    /// Emits register-delimited comments without altering sequential behavior.
    fn emit_pipeline(&mut self) -> Result<(), BlockCodegenError> {
        let layout = reconstruct_stages(self.package, self.func, self.metadata)?;
        let mut emitted_visible_stage = false;
        for (stage_index, stage) in layout.stages.iter().enumerate() {
            self.current_stage = Some(stage_index);
            let visible =
                !stage.combinational_nodes.is_empty() || !stage.register_writes.is_empty();
            if visible {
                if emitted_visible_stage {
                    let blank = self.file.make_blank_line();
                    self.file.add_member_blank_line(self.module, blank);
                }
                let comment = self
                    .file
                    .make_comment(&format!("===== Pipe stage {stage_index}:"));
                self.file.add_member_comment(self.module, comment);
                emitted_visible_stage = true;
            }
            self.emit_stage(stage, stage_index)?;
        }
        self.current_stage = None;
        Ok(())
    }

    /// Emits one reconstructed stage and the registers written at its end.
    fn emit_stage(&mut self, stage: &Stage, stage_index: usize) -> Result<(), BlockCodegenError> {
        for &node_ref in &stage.register_reads {
            if let NodePayload::RegisterRead { register } = &self.func.get_node(node_ref).payload {
                if let Some(signal) = self.register_refs.get(register).copied() {
                    self.values[node_ref.index] =
                        Some(Value::signal(signal).with_type(self.func.get_node_ty(node_ref)));
                }
            }
        }
        let start = self.file.module_member_count(self.module);
        if stage_index == 0 {
            self.emit_input_array_views()?;
        }
        for &node_ref in &stage.combinational_nodes {
            self.emit_node(node_ref)?;
        }
        self.file.hoist_module_declarations(self.module, start);
        if !stage.register_writes.is_empty() {
            let blank = self.file.make_blank_line();
            self.file.add_member_blank_line(self.module, blank);
            let comment = self
                .file
                .make_comment(&format!("Registers for pipe stage {stage_index}:"));
            self.file.add_member_comment(self.module, comment);
            for &node_ref in &stage.register_writes {
                if let NodePayload::RegisterWrite { register, .. } =
                    &self.func.get_node(node_ref).payload
                {
                    let definition = self
                        .metadata
                        .registers
                        .iter()
                        .find(|definition| definition.name == *register)
                        .ok_or_else(|| {
                            BlockCodegenError::InvalidBlock(format!(
                                "register `{register}` has no declaration in block `{}`",
                                self.func.name
                            ))
                        })?;
                    self.declare_register(definition)?;
                }
            }
            self.emit_register_groups(&stage.register_writes)?;
        }
        Ok(())
    }

    /// Declares one representable packed register and remembers its signal.
    fn declare_register(&mut self, register: &Register) -> Result<(), BlockCodegenError> {
        if register.ty.bit_count() == 0 || self.register_refs.contains_key(&register.name) {
            return Ok(());
        }
        let data_type = self.value_type(&register.ty);
        let name = self.unique_name(&register.name);
        let signal = self.file.add_logic(self.module, &name, &data_type)?;
        self.register_refs.insert(register.name.clone(), signal);
        Ok(())
    }

    /// Groups resettable and non-resettable registers by their existing clock.
    fn emit_register_groups(&mut self, writes: &[NodeRef]) -> Result<(), BlockCodegenError> {
        let mut resettable = Vec::new();
        let mut ordinary = Vec::new();
        for &node_ref in writes {
            let NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                reset,
            } = &self.func.get_node(node_ref).payload
            else {
                continue;
            };
            let Some(&signal) = self.register_refs.get(register) else {
                continue;
            };
            let definition = self
                .metadata
                .registers
                .iter()
                .find(|definition| definition.name == *register)
                .expect("verified register write has a declaration");
            let next = self.required_value(*arg)?.expr;
            let enable = load_enable
                .map(|node| self.required_value(node).map(|value| value.expr))
                .transpose()?;
            let reset_signal = reset
                .map(|node| self.required_value(node).map(|value| value.expr))
                .transpose()?;
            let reset_value = if reset_signal.is_some() {
                definition
                    .reset_value
                    .as_ref()
                    .map(|value| self.literal(value, &definition.ty))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let register = RegisterDefinition {
                reg: signal.to_expr(),
                next,
                reset_value,
                enable,
            };
            if let Some(signal) = reset_signal {
                resettable.push((signal, register));
            } else {
                ordinary.push(register);
            }
        }
        if ordinary.is_empty() && resettable.is_empty() {
            return Ok(());
        }
        let clock_name = self.metadata.clock_port_name.as_ref().ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "block `{}` has registers but no clock port",
                self.func.name
            ))
        })?;
        let clock = self
            .ports
            .get(clock_name)
            .ok_or_else(|| {
                BlockCodegenError::InvalidBlock(format!(
                    "block `{}` has no declared clock port `{clock_name}`",
                    self.func.name
                ))
            })?
            .to_expr();

        if !ordinary.is_empty() {
            xlsynth::vast_helpers::add_registers(
                &clock,
                None,
                &ordinary,
                RegisterScope::Module(self.module),
                self.file,
                self.options.register_codegen_options.as_ref(),
            )?;
        }
        if !resettable.is_empty() {
            let reset_metadata = self.metadata.reset.as_ref().ok_or_else(|| {
                BlockCodegenError::InvalidBlock(format!(
                    "block `{}` has resettable registers but no reset metadata",
                    self.func.name
                ))
            })?;
            let mut groups = Vec::<ResettableRegisterGroup>::new();
            for (signal, register) in resettable {
                if let Some(group) = groups.iter_mut().find(|group| group.signal == signal) {
                    group.registers.push(register);
                } else {
                    groups.push(ResettableRegisterGroup {
                        signal,
                        registers: vec![register],
                    });
                }
            }
            for group in groups {
                let reset = Reset {
                    signal: group.signal,
                    active_low: reset_metadata.active_low,
                };
                if reset_metadata.asynchronous {
                    if let Some(options) = &self.options.register_codegen_options {
                        let requires_enable = group
                            .registers
                            .iter()
                            .any(|register| register.enable.is_some());
                        if options.reg_with_reset_template.is_some()
                            || (requires_enable
                                && options.reg_with_reset_with_en_template.is_some())
                        {
                            return Err(BlockCodegenError::Unsupported(format!(
                                concat!(
                                    "block `{}` has asynchronous reset; custom reset-register ",
                                    "templates cannot guarantee the required reset edge semantics"
                                ),
                                self.func.name
                            )));
                        }
                    }
                    self.emit_asynchronous_registers(&clock, reset, &group.registers)?;
                } else {
                    xlsynth::vast_helpers::add_registers(
                        &clock,
                        Some(reset),
                        &group.registers,
                        RegisterScope::Module(self.module),
                        self.file,
                        self.options.register_codegen_options.as_ref(),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Emits an asynchronous reset edge with reset priority over load enables.
    fn emit_asynchronous_registers(
        &mut self,
        clock: &Expr,
        reset: Reset,
        registers: &[RegisterDefinition],
    ) -> Result<(), BlockCodegenError> {
        let clock_edge = self.file.make_pos_edge(clock);
        let reset_edge = if reset.active_low {
            self.file.make_neg_edge(&reset.signal)
        } else {
            self.file.make_pos_edge(&reset.signal)
        };
        let always = self
            .file
            .add_always_ff(self.module, &[&clock_edge, &reset_edge])?;
        let body = self.file.statement_block(always);
        let condition = if reset.active_low {
            self.file.make_logical_not(&reset.signal)
        } else {
            reset.signal
        };
        let conditional = self.file.block_add_cond(body, &condition);
        let reset_body = self.file.conditional_then_block(conditional);
        let update_body = self.file.conditional_add_else(conditional);
        for register in registers {
            let reset_value = register
                .reset_value
                .expect("resettable register group requires reset values");
            self.file
                .block_add_nonblocking_assignment(reset_body, &register.reg, &reset_value);
            let next = if let Some(enable) = register.enable {
                self.file
                    .make_ternary(&enable, &register.next, &register.reg)
            } else {
                register.next
            };
            self.file
                .block_add_nonblocking_assignment(update_body, &register.reg, &next);
        }
        Ok(())
    }

    /// Returns the node driving each logical output in declared output order.
    fn output_nodes(&self) -> Result<Vec<NodeRef>, BlockCodegenError> {
        if self.metadata.output_names.is_empty() {
            return Ok(Vec::new());
        }
        let node = self.func.ret_node_ref.ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "block `{}` declares outputs but has no return node",
                self.func.name
            ))
        })?;
        if self.metadata.output_names.len() == 1 {
            return Ok(vec![node]);
        }
        match &self.func.get_node(node).payload {
            NodePayload::Tuple(outputs) if outputs.len() == self.metadata.output_names.len() => {
                Ok(outputs.clone())
            }
            _ => Err(BlockCodegenError::InvalidBlock(format!(
                "block `{}` has {} outputs but its return node is not a matching tuple",
                self.func.name,
                self.metadata.output_names.len()
            ))),
        }
    }

    /// Connects representable output ports after their driving logic exists.
    fn emit_outputs(&mut self) -> Result<(), BlockCodegenError> {
        for (name, node_ref) in self.metadata.output_names.iter().zip(self.output_nodes()?) {
            let Some(signal) = self.ports.get(name).copied() else {
                continue;
            };
            let value = self.required_value(node_ref)?;
            let assignment = self
                .file
                .make_continuous_assignment(&signal.to_expr(), &value.expr);
            self.file
                .add_member_continuous_assignment(self.module, assignment);
        }
        Ok(())
    }

    /// Emits deterministic child connections after all connection nodes exist.
    fn emit_instantiations(&mut self) -> Result<(), BlockCodegenError> {
        if self.metadata.instantiations.is_empty() {
            return Ok(());
        }
        if self.options.layout == Layout::Pipeline {
            let blank = self.file.make_blank_line();
            self.file.add_member_blank_line(self.module, blank);
            let comment = self.file.make_comment("===== Instantiations");
            self.file.add_member_comment(self.module, comment);
        }

        for instance in &self.metadata.instantiations {
            validate_external_identifier(&instance.name, "instance")?;
            let mut connections = self
                .instance_inputs
                .iter()
                .chain(self.instance_outputs.iter())
                .filter(|((name, _), _)| name == &instance.name)
                .map(|((_, port), expr)| (port.clone(), *expr))
                .collect::<Vec<_>>();

            if instance.kind == InstantiationKind::Block {
                let child = self
                    .package
                    .members
                    .iter()
                    .find_map(|member| match member {
                        PackageMember::Block { func, metadata } if func.name == instance.block => {
                            Some(metadata)
                        }
                        _ => None,
                    })
                    .expect("validated child block exists");
                if let Some(clock_name) = &child.clock_port_name {
                    if !connections.iter().any(|(name, _)| name == clock_name) {
                        if let Some(parent_clock) = self
                            .metadata
                            .clock_port_name
                            .as_ref()
                            .and_then(|name| self.ports.get(name))
                        {
                            connections.push((clock_name.clone(), parent_clock.to_expr()));
                        } else {
                            return Err(BlockCodegenError::InvalidBlock(format!(
                                "instance `{}` of block `{}` requires clock port \
                                 `{clock_name}`, but parent block `{}` has no clock",
                                instance.name, instance.block, self.func.name
                            )));
                        }
                    }
                }
                let order = &child.port_order;
                connections.sort_by_key(|(name, _)| {
                    (
                        order
                            .iter()
                            .position(|port| port == name)
                            .unwrap_or(usize::MAX),
                        name.clone(),
                    )
                });
                let names = connections
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>();
                let expressions = connections
                    .iter()
                    .map(|(_, expr)| Some(expr))
                    .collect::<Vec<_>>();
                let handle = self.file.make_instantiation(
                    &instance.block,
                    &instance.name,
                    &[],
                    &[],
                    &names,
                    &expressions,
                );
                self.file.add_member_instantiation(self.module, handle);
            } else {
                self.emit_external_instantiation(instance, &connections)?;
            }
        }
        Ok(())
    }

    /// Substitutes external-function placeholders into its public FFI template.
    fn emit_external_instantiation(
        &mut self,
        instance: &xlsynth_pir::ir::Instantiation,
        connections: &[(String, Expr)],
    ) -> Result<(), BlockCodegenError> {
        let foreign = self.package.get_fn(&instance.block).ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "external instance `{}` references missing function `{}`",
                instance.name, instance.block
            ))
        })?;
        let attribute = foreign
            .outer_attrs
            .iter()
            .find(|attribute| attribute.contains("code_template:"))
            .ok_or_else(|| {
                BlockCodegenError::Unsupported(format!(
                    "external function `{}` has no ffi_proto code_template",
                    foreign.name
                ))
            })?;
        let marker = attribute.find("code_template:").expect("template marker");
        let remainder = attribute[marker + "code_template:".len()..].trim_start();
        let start = remainder.find('"').ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "external function `{}` has an invalid ffi_proto code_template",
                foreign.name
            ))
        })?;
        let mut escaped = false;
        let mut end = None;
        for (index, character) in remainder[start + 1..].char_indices() {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                end = Some(start + 1 + index);
                break;
            }
        }
        let end = end.ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "external function `{}` has an unterminated ffi_proto code_template",
                foreign.name
            ))
        })?;
        let template = remainder[start + 1..end]
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
        let bindings = connections
            .iter()
            .map(|(name, value)| (name.as_str(), self.file.emit_expression(value)))
            .collect::<BTreeMap<_, _>>();
        let template = substitute_external_template(&template, &instance.name, &bindings)?;
        let statement = self.file.make_inline_verilog_statement(template.trim());
        self.file
            .add_member_inline_statement(self.module, statement);
        Ok(())
    }

    /// Builds the packed scalar or vector type used for one IR value.
    pub(crate) fn bits_type(&mut self, width: usize) -> VastDataType {
        self.file.make_bit_vector_type(width as i64, false)
    }

    /// Preserves outer array dimensions while packing tuple leaves into bits.
    pub(crate) fn value_type(&mut self, mut ty: &Type) -> VastDataType {
        let mut dimensions = Vec::new();
        while let Type::Array(array) = ty {
            dimensions.push(array.element_count as i64);
            ty = &array.element_type;
        }
        let element = self.bits_type(ty.bit_count());
        if dimensions.is_empty() {
            element
        } else {
            self.file.make_packed_array_type(element, &dimensions)
        }
    }

    /// Gives custom-typed array input ports shaped views without changing
    /// their explicitly requested SystemVerilog types.
    fn emit_input_array_views(&mut self) -> Result<(), BlockCodegenError> {
        for (index, node) in self.func.nodes.iter().enumerate() {
            if matches!(node.payload, NodePayload::GetParam(_))
                && matches!(node.ty, Type::Array(_))
                && let Some(value) = self.values[index]
                && value.array_rank == 0
            {
                let view = self.assign_node(NodeRef { index }, value.expr)?;
                self.values[index] = Some(view);
            }
        }
        Ok(())
    }

    /// Returns the represented expression for a node with nonzero bit width.
    pub(crate) fn required_value(&self, node: NodeRef) -> Result<Value, BlockCodegenError> {
        self.values[node.index].ok_or_else(|| {
            BlockCodegenError::InvalidBlock(format!(
                "node `{}` in block `{}` has no representable SystemVerilog value",
                self.func.get_node(node).name.clone().unwrap_or_else(|| {
                    format!(
                        "{}.{}",
                        self.func.get_node(node).payload.get_operator(),
                        self.func.get_node(node).text_id
                    )
                }),
                self.func.name
            ))
        })
    }

    /// Converts an arbitrary-width IR literal into its packed representation.
    pub(crate) fn literal(
        &mut self,
        value: &IrValue,
        ty: &Type,
    ) -> Result<Option<Expr>, BlockCodegenError> {
        if ty.bit_count() == 0 {
            return Ok(None);
        }
        match ty {
            Type::Bits(_) => {
                let formatted = value.to_string_fmt(IrFormatPreference::Hex)?;
                Ok(Some(
                    self.file.make_literal(&formatted, &LiteralFormat::Hex)?,
                ))
            }
            Type::Tuple(types) => {
                let values = value.get_elements()?;
                let mut expressions = Vec::new();
                for (value, ty) in values.iter().zip(types) {
                    if let Some(expression) = self.literal(value, ty)? {
                        expressions.push(expression);
                    }
                }
                Ok(self.concat_or_only(&expressions))
            }
            Type::Array(array) => {
                let values = value.get_elements()?;
                let mut expressions = Vec::new();
                for value in values.iter().rev() {
                    if let Some(expression) = self.literal(value, &array.element_type)? {
                        expressions.push(expression);
                    }
                }
                Ok(self.concat_or_only(&expressions))
            }
            Type::Token => Ok(None),
        }
    }

    /// Concatenates expressions while avoiding an unnecessary one-element
    /// brace.
    pub(crate) fn concat_or_only(&mut self, expressions: &[Expr]) -> Option<Expr> {
        match expressions {
            [] => None,
            [only] => Some(*only),
            many => {
                let refs = many.iter().collect::<Vec<_>>();
                Some(self.file.make_concat(&refs))
            }
        }
    }

    /// Allocates a deterministic module-local identifier without collisions.
    pub(crate) fn unique_name(&mut self, requested: &str) -> String {
        let mut base = sanitize_identifier(requested);
        if self.names.insert(base.clone()) {
            return base;
        }
        let mut index = 1usize;
        loop {
            let candidate = format!("{base}__{index}");
            if self.names.insert(candidate.clone()) {
                return candidate;
            }
            index += 1;
            if index == usize::MAX {
                base.push('_');
                index = 1;
            }
        }
    }

    /// Declares a stable, indexable signal for one materialized IR node.
    pub(crate) fn declare_node(
        &mut self,
        node_ref: NodeRef,
    ) -> Result<LogicRef, BlockCodegenError> {
        let name = self.node_name(node_ref);
        let unique = self.unique_name(&name);
        let ty = self.value_type(self.func.get_node_ty(node_ref));
        Ok(self.file.add_logic(self.module, &unique, &ty)?)
    }

    /// Builds the requested node name in the current source-layout scope.
    pub(crate) fn node_name(&self, node_ref: NodeRef) -> String {
        let node = self.func.get_node(node_ref);
        let original_name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("{}_{}", node.payload.get_operator(), node.text_id));
        if let Some(stage) = self.current_stage {
            let base = strip_pipeline_prefix(&original_name);
            format!("p{stage}_{base}_comb")
        } else {
            original_name
        }
    }

    /// Emits a stable signal assignment and returns its indexable value.
    pub(crate) fn assign_node(
        &mut self,
        node_ref: NodeRef,
        expression: Expr,
    ) -> Result<Value, BlockCodegenError> {
        let signal = self.declare_node(node_ref)?;
        let assignment = self
            .file
            .make_continuous_assignment(&signal.to_expr(), &expression);
        self.file
            .add_member_continuous_assignment(self.module, assignment);
        Ok(Value::signal(signal).with_type(self.func.get_node_ty(node_ref)))
    }

    /// Returns whether this node should be assigned a dedicated named signal.
    pub(crate) fn should_assign(&self, node_ref: NodeRef, depth: usize) -> bool {
        self.must_assign(node_ref)
            || self.func.get_node(node_ref).name.is_some()
            || self.users[node_ref.index].len() > 1
            || depth > self.options.max_inline_depth
    }

    /// Preserves fixed-width and indexability boundaries required by lowering.
    fn must_assign(&self, node_ref: NodeRef) -> bool {
        let node = self.func.get_node(node_ref);
        matches!(node.ty, Type::Array(_))
            || matches!(
                node.payload,
                NodePayload::ArrayIndex { .. }
                    | NodePayload::ExtCarryOut { .. }
                    | NodePayload::ExtNaryAdd { .. }
            )
            || matches!(
                node.payload,
                NodePayload::Binop(
                    Binop::Add
                        | Binop::Sub
                        | Binop::Smul
                        | Binop::Umul
                        | Binop::Smulp
                        | Binop::Umulp
                        | Binop::Sdiv
                        | Binop::Udiv,
                    _,
                    _
                )
            )
            || self.options.separate_lines
            || self.users[node_ref.index]
                .iter()
                .any(|user| self.operand_must_be_named_reference(*user, node_ref))
    }

    /// Identifies operand roles whose lowering requires a declared signal.
    fn operand_must_be_named_reference(&self, user: NodeRef, operand: NodeRef) -> bool {
        match &self.func.get_node(user).payload {
            NodePayload::RegisterWrite { .. } => true,
            NodePayload::BitSlice { arg, .. } => *arg == operand,
            NodePayload::TupleIndex { tuple, .. } => *tuple == operand,
            NodePayload::SignExt { arg, .. } => {
                *arg == operand && self.func.get_node_ty(operand).bit_count() > 1
            }
            NodePayload::ExtNaryAdd { terms, .. } => {
                terms.iter().any(|term| term.operand == operand)
            }
            NodePayload::OneHot { arg, .. }
            | NodePayload::Encode { arg }
            | NodePayload::ExtPrioEncode { arg, .. }
            | NodePayload::ExtClz { arg, .. }
            | NodePayload::ExtNormalizeLeft { arg, .. }
            | NodePayload::Unop(Unop::Reverse, arg) => {
                *arg == operand && self.func.get_node_ty(operand).bit_count() > 1
            }
            NodePayload::Sel {
                selector, cases, ..
            }
            | NodePayload::PrioritySel {
                selector, cases, ..
            }
            | NodePayload::OneHotSel { selector, cases } => *selector == operand && cases.len() > 1,
            NodePayload::ArraySlice {
                array,
                start,
                width,
            } => *width > 1 && (*array == operand || *start == operand),
            NodePayload::ArrayUpdate { indices, .. } => {
                // Genvar comparisons must not widen an inline index expression.
                indices.contains(&operand)
            }
            _ => false,
        }
    }
}

/// Substitutes named FFI expressions while unescaping doubled literal braces.
fn substitute_external_template(
    template: &str,
    instance_name: &str,
    bindings: &BTreeMap<&str, String>,
) -> Result<String, BlockCodegenError> {
    let bytes = template.as_bytes();
    let mut output = String::with_capacity(template.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'{' if bytes.get(index + 1) == Some(&b'{') => {
                output.push('{');
                index += 2;
            }
            b'}' if bytes.get(index + 1) == Some(&b'}') => {
                output.push('}');
                index += 2;
            }
            b'{' => {
                let start = index + 1;
                let mut nesting = 1usize;
                index = start;
                while index < bytes.len() && nesting != 0 {
                    match bytes[index] {
                        b'{' => nesting += 1,
                        b'}' => nesting -= 1,
                        _ => {
                            // Ordinary expression characters do not change
                            // brace nesting.
                        }
                    }
                    if nesting != 0 {
                        index += 1;
                    }
                }
                if index >= bytes.len() {
                    return Err(BlockCodegenError::InvalidBlock(format!(
                        "external instance `{instance_name}` has an unterminated \
                         ffi_proto placeholder"
                    )));
                }
                let name = &template[start..index];
                if name == "fn" {
                    output.push_str(instance_name);
                } else if let Some(value) = bindings.get(name) {
                    output.push_str(value);
                } else {
                    return Err(BlockCodegenError::InvalidBlock(format!(
                        "external instance `{instance_name}` leaves ffi_proto \
                         placeholder `{{{name}}}` unresolved"
                    )));
                }
                index += 1;
            }
            b'}' => {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "external instance `{instance_name}` has an unmatched \
                     closing brace in its ffi_proto template"
                )));
            }
            _ => {
                let character = template[index..]
                    .chars()
                    .next()
                    .expect("byte index is positioned at a UTF-8 character");
                output.push(character);
                index += character.len_utf8();
            }
        }
    }
    Ok(output)
}

/// Rejects public interface names that SystemVerilog cannot represent safely.
pub(crate) fn validate_external_identifier(
    name: &str,
    kind: &str,
) -> Result<(), BlockCodegenError> {
    if sanitize_identifier(name) != name {
        return Err(BlockCodegenError::InvalidBlock(format!(
            "invalid SystemVerilog {kind} identifier `{name}`: identifier is reserved, \
             malformed, or requires escaping"
        )));
    }
    Ok(())
}

/// Produces a valid, deterministic SystemVerilog identifier from an IR name.
fn sanitize_identifier(name: &str) -> String {
    let mut output = String::with_capacity(name.len() + 1);
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_alphabetic()
            || character == '_'
            || (index != 0 && (character.is_ascii_digit() || character == '$'))
        {
            output.push(character);
        } else {
            if index == 0 && character.is_ascii_digit() {
                output.push('_');
                output.push(character);
            } else {
                output.push('_');
            }
        }
    }
    if output.is_empty() {
        output.push('_');
    }
    if xlsynth_vast::is_system_verilog_keyword(&output) {
        output.push('_');
    }
    output
}

/// Removes an existing `p<digits>_` prefix before assigning a new stage.
fn strip_pipeline_prefix(name: &str) -> &str {
    let Some(rest) = name.strip_prefix('p') else {
        return name;
    };
    let digit_count = rest.chars().take_while(char::is_ascii_digit).count();
    if digit_count > 0 && rest[digit_count..].starts_with('_') {
        &rest[digit_count + 1..]
    } else {
        name
    }
}
