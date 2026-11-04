// SPDX-License-Identifier: Apache-2.0

//! BX name resolution, value type checking, and structural driver checks.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use xlsynth_pir::IrValue;

use crate::ast::{BinaryOp, Block, Expr, ExternFn, Module, Param, Stmt, StructDef, Type};
pub use crate::typed::{CheckedBlock, CheckedModule, CheckedStmt, TypedExpr, TypedExprKind};
use crate::typed::{TypedExprKind as K, bits};

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Bits { signed, width } => write!(f, "{}{width}", if *signed { "s" } else { "u" }),
            Type::Bool => write!(f, "bool"),
            Type::Named(name) => write!(f, "{name}"),
            Type::Array { element, size } => write!(f, "{element}[{size}]"),
            Type::Tuple(items) => {
                write!(f, "(")?;
                for (i, ty) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                if items.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Choose the physical output ports from the source result's outermost type.
pub(crate) fn output_ports(result: &Type, structs: &[StructDef]) -> Result<Vec<Param>, String> {
    match result {
        Type::Named(name) => structs
            .iter()
            .find(|s| s.name == *name)
            .map(|s| s.fields.clone())
            .ok_or_else(|| format!("unknown struct '{name}'")),
        Type::Tuple(items) => Ok(items
            .iter()
            .enumerate()
            .map(|(i, ty)| Param {
                name: format!("out_{i}"),
                ty: ty.clone(),
            })
            .collect()),
        _ => Ok(vec![Param {
            name: "out".into(),
            ty: result.clone(),
        }]),
    }
}

/// DSLX's bool and u1 are the same unsigned bit type, including in tuples.
fn compatible(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Tuple(x), Type::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| compatible(x, y))
        }
        (Type::Named(x), Type::Named(y)) => x == y,
        (
            Type::Array {
                element: x,
                size: n,
            },
            Type::Array {
                element: y,
                size: m,
            },
        ) => n == m && compatible(x, y),
        _ => bits(a).is_some_and(|t| Some(t) == bits(b)),
    }
}

/// Data arguments, output wires, and reset requirement of a local or linked
/// child.
struct BlockInterface<'a> {
    params: &'a [Param],
    result: &'a Type,
    outputs: Vec<Param>,
    requires_reset: bool,
}

/// Source declaration indices for lookup during checking.
struct Declarations<'a> {
    struct_list: &'a [StructDef],
    structs: BTreeMap<&'a str, &'a StructDef>,
    struct_shapes: BTreeMap<&'a str, TypeShape>,
    externs: BTreeMap<&'a str, &'a ExternFn>,
    blocks: BTreeMap<&'a str, BlockInterface<'a>>,
}

fn declare(names: &mut BTreeSet<String>, name: &str) -> Result<(), String> {
    if names.insert(name.to_owned()) {
        Ok(())
    } else {
        Err(format!("duplicate name '{name}'"))
    }
}

/// Collect named type dependencies before computing layouts in dependency
/// order.
fn type_deps<'a>(ty: &'a Type, result: &mut Vec<&'a str>) {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Named(name) => result.push(name),
            Type::Array { element, .. } => pending.push(element),
            Type::Tuple(fields) => pending.extend(fields.iter().rev()),
            Type::Bool | Type::Bits { .. } => { /* Primitive types have no dependencies. */ }
        }
    }
}

/// Bound concrete value materialization, including expanded zero-width members.
/// A bit type costs its width (at least one), and an aggregate costs one plus
/// the sum of its members' costs. This accounts for aggregate storage even
/// when its packed width is zero. An empty array still stores its element type.
const MAX_TYPE_SIZE: usize = 1 << 20;

/// Bound recursion in downstream layout construction and hierarchy elaboration.
const MAX_ELABORATION_DEPTH: usize = 64;

/// Storage cost and maximum aggregate nesting of a fully expanded source type.
#[derive(Clone, Copy)]
struct TypeShape {
    size: usize,
    depth: usize,
}

impl<'a> Declarations<'a> {
    /// Check the shared physical port ABI before any parent uses this
    /// signature.
    fn add_block(
        &mut self,
        name: &'a str,
        clock: &str,
        reset: Option<&str>,
        params: &'a [Param],
        result: &'a Type,
    ) -> Result<(), String> {
        let mut ports = BTreeSet::new();
        declare(&mut ports, clock)?;
        if let Some(reset) = reset {
            declare(&mut ports, reset)?;
        }
        for p in params {
            declare(&mut ports, &p.name)?;
            self.validate_type(&p.ty)
                .map_err(|e| format!("input '{}': {e}", p.name))?;
        }
        self.validate_type(result)
            .map_err(|e| format!("result: {e}"))?;
        let outputs = output_ports(result, self.struct_list)?;
        for output in &outputs {
            declare(&mut ports, &output.name)?;
        }
        self.blocks.insert(
            name,
            BlockInterface {
                params,
                result,
                outputs,
                requires_reset: reset.is_some(),
            },
        );
        Ok(())
    }

    fn validate_type(&self, ty: &Type) -> Result<(), String> {
        self.type_shape(ty, MAX_ELABORATION_DEPTH).map(|_| ())
    }

    /// Bound the concrete layout, including the full depth of memoized named
    /// types.
    fn type_shape(&self, ty: &Type, remaining: usize) -> Result<TypeShape, String> {
        let shape = match ty {
            Type::Bool => TypeShape { size: 1, depth: 0 },
            Type::Bits { width, .. } => TypeShape {
                size: (*width).max(1),
                depth: 0,
            },
            Type::Tuple(items) => self.aggregate_shape(items, remaining)?,
            Type::Array { element, size } => {
                let child = self.type_shape(element, child_type_budget(remaining)?)?;
                TypeShape {
                    size: child.size.saturating_mul((*size).max(1)).saturating_add(1),
                    depth: child.depth + 1,
                }
            }
            Type::Named(name) => *self
                .struct_shapes
                .get(name.as_str())
                .ok_or_else(|| format!("unknown struct '{name}'"))?,
        };
        if shape.depth > remaining {
            return Err(format!(
                "expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        if shape.size > MAX_TYPE_SIZE {
            return Err(format!(
                "type {ty} exceeds maximum concrete type size {MAX_TYPE_SIZE}"
            ));
        }
        Ok(shape)
    }

    fn aggregate_shape<'t>(
        &self,
        fields: impl IntoIterator<Item = &'t Type>,
        remaining: usize,
    ) -> Result<TypeShape, String> {
        let child_budget = child_type_budget(remaining)?;
        let mut shape = TypeShape { size: 1, depth: 1 };
        for field in fields {
            let child = self.type_shape(field, child_budget)?;
            shape.size = shape.size.saturating_add(child.size);
            shape.depth = shape.depth.max(child.depth + 1);
            if shape.size > MAX_TYPE_SIZE {
                return Err(format!(
                    "aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"
                ));
            }
        }
        Ok(shape)
    }
}

fn child_type_budget(remaining: usize) -> Result<usize, String> {
    remaining
        .checked_sub(1)
        .ok_or_else(|| format!("expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"))
}

/// A dependency in progress or its fully checked depth in the dependency graph.
#[derive(Clone, Copy)]
enum Visit {
    Unseen,
    Active,
    Complete(usize),
}

/// Visit named dependencies in source order, rejecting recursive layouts and
/// hardware.
fn dependency_order<'a>(
    kind: &str,
    graph: &[(&'a str, Vec<&'a str>)],
) -> Result<Vec<usize>, String> {
    fn visit(
        i: usize,
        kind: &str,
        graph: &[(&str, Vec<&str>)],
        indices: &BTreeMap<&str, usize>,
        status: &mut [Visit],
        order: &mut Vec<usize>,
        remaining: usize,
    ) -> Result<usize, String> {
        let too_deep = || {
            format!(
                "{kind} dependency depth exceeds {MAX_ELABORATION_DEPTH} levels at '{}'",
                graph[i].0
            )
        };
        if remaining == 0 {
            return Err(too_deep());
        }
        match status[i] {
            Visit::Complete(depth) if depth > remaining => return Err(too_deep()),
            Visit::Complete(depth) => return Ok(depth),
            Visit::Active => {
                return Err(format!(
                    "recursive {kind} dependency involving '{}'",
                    graph[i].0
                ));
            }
            Visit::Unseen => status[i] = Visit::Active,
        }
        let mut depth = 1;
        for dep in &graph[i].1 {
            let next = *indices
                .get(dep)
                .ok_or_else(|| format!("unknown {kind} '{dep}' in '{}'", graph[i].0))?;
            let child_depth = visit(next, kind, graph, indices, status, order, remaining - 1)?;
            depth = depth.max(child_depth + 1);
        }
        status[i] = Visit::Complete(depth);
        order.push(i);
        Ok(depth)
    }
    let indices = graph
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i))
        .collect();
    let mut order = Vec::with_capacity(graph.len());
    let mut status = vec![Visit::Unseen; graph.len()];
    for i in 0..graph.len() {
        visit(
            i,
            kind,
            graph,
            &indices,
            &mut status,
            &mut order,
            MAX_ELABORATION_DEPTH,
        )?;
    }
    Ok(order)
}

/// Type-check a parsed module without requiring the external functions' IR
/// bodies.
pub fn check_module(module: &Module) -> Result<CheckedModule, String> {
    let mut names = BTreeSet::new();
    let mut struct_graph = Vec::new();
    for s in &module.structs {
        declare(&mut names, &s.name)?;
        let mut fields = BTreeSet::new();
        let mut deps = Vec::new();
        for f in &s.fields {
            declare(&mut fields, &f.name).map_err(|e| format!("struct '{}': {e}", s.name))?;
            type_deps(&f.ty, &mut deps);
        }
        struct_graph.push((s.name.as_str(), deps));
    }
    let struct_order = dependency_order("struct", &struct_graph)?;
    let mut declarations = Declarations {
        struct_list: &module.structs,
        struct_shapes: BTreeMap::new(),
        structs: module
            .structs
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect(),
        externs: module
            .externs
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect(),
        blocks: BTreeMap::new(),
    };
    for index in struct_order {
        let s = &module.structs[index];
        let shape = declarations
            .aggregate_shape(s.fields.iter().map(|f| &f.ty), MAX_ELABORATION_DEPTH)
            .map_err(|e| format!("struct '{}': {e}", s.name))?;
        declarations.struct_shapes.insert(&s.name, shape);
    }
    for f in &module.externs {
        declare(&mut names, &f.name)?;
        let mut params = BTreeSet::new();
        for p in &f.params {
            declare(&mut params, &p.name).map_err(|e| format!("extern '{}': {e}", f.name))?;
            declarations
                .validate_type(&p.ty)
                .map_err(|e| format!("extern '{}', parameter '{}': {e}", f.name, p.name))?;
        }
        declarations
            .validate_type(&f.result)
            .map_err(|e| format!("extern '{}' result: {e}", f.name))?;
    }
    for b in &module.blocks {
        declare(&mut names, &b.name)?;
        declarations
            .add_block(&b.name, &b.clock, b.reset.as_deref(), &b.params, &b.result)
            .map_err(|e| format!("block '{}': {e}", b.name))?;
    }
    for b in &module.extern_blocks {
        declare(&mut names, &b.name)?;
        declarations
            .add_block(&b.name, &b.clock, b.reset.as_deref(), &b.params, &b.result)
            .map_err(|e| format!("extern block '{}': {e}", b.name))?;
    }
    let mut block_graph = Vec::new();
    for b in &module.blocks {
        let deps = b
            .statements
            .iter()
            .filter_map(|s| match s {
                Stmt::Inst { block, .. } => Some(block.as_str()),
                Stmt::Let {
                    value: Expr::Call { name, .. },
                    ..
                } if declarations.blocks.contains_key(name.as_str()) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        block_graph.push((b.name.as_str(), deps));
    }
    // Imported blocks are leaves here; their bodies belong to the linked
    // package.
    block_graph.extend(
        module
            .extern_blocks
            .iter()
            .map(|b| (b.name.as_str(), Vec::new())),
    );
    let mut blocks = Vec::new();
    for index in dependency_order("block", &block_graph)? {
        if let Some(block) = module.blocks.get(index) {
            blocks.push(
                BlockChecker::check(&declarations, block)
                    .map_err(|e| format!("block '{}': {e}", block.name))?,
            );
        }
    }
    Ok(CheckedModule {
        structs: module.structs.clone(),
        externs: module.externs.clone(),
        extern_blocks: module.extern_blocks.clone(),
        blocks,
    })
}

/// Lexically visible values, static declarations, and the drivers seen in one
/// block.
struct BlockChecker<'a, 'm> {
    decls: &'a Declarations<'m>,
    block: &'m Block,
    names: BTreeSet<String>,
    outputs: BTreeSet<String>,
    reserved_hardware_names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    constants: BTreeMap<String, IrValue>,
    regs: BTreeMap<String, Type>,
    writes: BTreeSet<String>,
    instances: BTreeMap<String, &'a BlockInterface<'m>>,
    connections: BTreeSet<(String, String)>,
    statements: Vec<CheckedStmt>,
}

impl<'a, 'm> BlockChecker<'a, 'm> {
    /// Reserve interface and instance names before checking dataflow
    /// expressions.
    fn check(decls: &'a Declarations<'m>, block: &'m Block) -> Result<CheckedBlock, String> {
        let outputs = decls.blocks[block.name.as_str()]
            .outputs
            .iter()
            .map(|p| p.name.clone())
            .collect::<BTreeSet<_>>();
        // Generated names must avoid future declarations as well as visible
        // ones.
        let mut reserved_hardware_names = outputs.clone();
        reserved_hardware_names.insert(block.clock.clone());
        reserved_hardware_names.extend(block.reset.iter().cloned());
        reserved_hardware_names.extend(block.params.iter().map(|p| p.name.clone()));
        reserved_hardware_names.extend(block.statements.iter().filter_map(|s| match s {
            Stmt::Reg { name, .. } | Stmt::Let { name, .. } | Stmt::Inst { name, .. } => {
                Some(name.clone())
            }
            _ => None,
        }));
        let mut c = Self {
            decls,
            block,
            names: BTreeSet::new(),
            outputs,
            reserved_hardware_names,
            values: BTreeMap::new(),
            constants: BTreeMap::new(),
            regs: BTreeMap::new(),
            writes: BTreeSet::new(),
            instances: BTreeMap::new(),
            connections: BTreeSet::new(),
            statements: Vec::new(),
        };
        c.hardware_name(&block.clock)?;
        if let Some(rst) = &block.reset {
            c.hardware_name(rst)?;
            c.values.insert(rst.clone(), Type::Bool);
        }
        for p in &block.params {
            c.hardware_name(&p.name)?;
            c.values.insert(p.name.clone(), p.ty.clone());
        }
        for stmt in &block.statements {
            if let Stmt::Inst { name, block: child } = stmt {
                c.hardware_name(name)?;
                let child_block = decls
                    .blocks
                    .get(child.as_str())
                    .ok_or_else(|| format!("unknown child block '{child}'"))?;
                c.check_child_reset(name, child, child_block)?;
                c.instances.insert(name.clone(), child_block);
            }
        }
        c.body(&block.statements, false)?;
        for reg in c.regs.keys() {
            if !c.writes.contains(reg) {
                return Err(format!("register '{reg}' has no update"));
            }
        }
        for (name, child) in &c.instances {
            for p in child.params {
                if !c.connections.contains(&(name.clone(), p.name.clone())) {
                    return Err(format!("instance input '{name}.{}' has no driver", p.name));
                }
            }
        }
        let output = c
            .expr(&block.output, Some(&block.result))
            .map_err(|e| format!("result: {e}"))?;
        Ok(CheckedBlock {
            name: block.name.clone(),
            public: block.public,
            clock: block.clock.clone(),
            reset: block.reset.clone(),
            params: block.params.clone(),
            result: block.result.clone(),
            statements: c.statements,
            output,
        })
    }

    fn hardware_name(&mut self, name: &str) -> Result<(), String> {
        if self.outputs.contains(name) {
            return Err(format!("name '{name}' collides with an output port"));
        }
        declare(&mut self.names, name)
    }

    fn check_child_reset(
        &self,
        name: &str,
        target: &str,
        child: &BlockInterface<'_>,
    ) -> Result<(), String> {
        if child.requires_reset && self.block.reset.is_none() {
            return Err(format!(
                "instance '{name}' of '{target}' requires a parent reset"
            ));
        }
        Ok(())
    }

    /// Keep the binding name in IR unless it already names an output port.
    fn call_instance_name(&mut self, binding: &str) -> String {
        if !self.outputs.contains(binding) {
            return binding.to_owned();
        }
        let base = format!("{binding}_inst");
        let mut name = base.clone();
        let mut suffix = 0;
        while !self.reserved_hardware_names.insert(name.clone()) {
            suffix += 1;
            name = format!("{base}_{suffix}");
        }
        name
    }

    /// Elaborate a call to static hardware, then assemble its outputs into a BX
    /// value.
    fn block_call(
        &mut self,
        binding: &str,
        target: &str,
        child: &BlockInterface<'_>,
        args: &[Expr],
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        self.check_child_reset(binding, target, child)?;
        if let Some(ty) = expected
            && !compatible(ty, child.result)
        {
            return Err(format!("expected {ty}, got {}", child.result));
        }
        if args.len() != child.params.len() {
            return Err(format!(
                "block '{target}' expects {} data arguments, got {}",
                child.params.len(),
                args.len()
            ));
        }
        let instance = self.call_instance_name(binding);
        self.statements.push(CheckedStmt::Inst {
            name: instance.clone(),
            block: target.to_owned(),
        });
        for (arg, param) in args.iter().zip(child.params) {
            let value = self
                .expr(arg, Some(&param.ty))
                .map_err(|e| format!("input '{}', call to block '{target}': {e}", param.name))?;
            self.statements.push(CheckedStmt::Connect {
                instance: instance.clone(),
                port: param.name.clone(),
                value,
            });
        }
        let mut outputs = child.outputs.iter().map(|port| TypedExpr {
            ty: port.ty.clone(),
            kind: K::InstanceOutput {
                instance: instance.clone(),
                port: port.name.clone(),
            },
        });
        match child.result {
            Type::Tuple(_) | Type::Named(_) => Ok(TypedExpr {
                ty: child.result.clone(),
                kind: K::Tuple(outputs.collect()),
            }),
            Type::Bool | Type::Bits { .. } | Type::Array { .. } => Ok(outputs
                .next()
                .expect("scalar and array interfaces have one output")),
        }
    }

    /// Check statements in lexical order, preserving a single expression per
    /// guard.
    fn body(&mut self, statements: &[Stmt], conditional: bool) -> Result<(), String> {
        let mut locals = Vec::new();
        for stmt in statements {
            let checked = match stmt {
                Stmt::Reg { name, ty, reset } => {
                    if conditional {
                        return Err(format!("conditional declaration of register '{name}'"));
                    }
                    self.hardware_name(name)?;
                    self.decls.validate_type(ty)?;
                    let reset = match reset {
                        None => None,
                        Some(expr) => {
                            if self.block.reset.is_none() {
                                return Err(format!(
                                    "initializer on register '{name}' requires a reset"
                                ));
                            }
                            let value = self
                                .expr(expr, Some(ty))
                                .map_err(|e| format!("reset of register '{name}': {e}"))?;
                            Some(value.constant(&self.constants).ok_or_else(|| {
                                format!(
                                    "reset of register '{name}' must be a compile-time constant"
                                )
                            })?)
                        }
                    };
                    self.regs.insert(name.clone(), ty.clone());
                    self.values.insert(name.clone(), ty.clone());
                    CheckedStmt::Reg {
                        name: name.clone(),
                        ty: ty.clone(),
                        reset,
                    }
                }
                Stmt::Inst { name, block } => {
                    if conditional {
                        return Err(format!("conditional declaration of instance '{name}'"));
                    }
                    CheckedStmt::Inst {
                        name: name.clone(),
                        block: block.clone(),
                    }
                }
                Stmt::Connect {
                    instance,
                    port,
                    value,
                } => {
                    if conditional {
                        return Err(format!("conditional connection of '{instance}.{port}'"));
                    }
                    let child = self
                        .instances
                        .get(instance)
                        .ok_or_else(|| format!("unknown instance '{instance}'"))?;
                    let param = child
                        .params
                        .iter()
                        .find(|p| p.name == *port)
                        .ok_or_else(|| format!("'{instance}.{port}' is not a data input"))?;
                    let value = self
                        .expr(value, Some(&param.ty))
                        .map_err(|e| format!("input '{instance}.{port}': {e}"))?;
                    if !self.connections.insert((instance.clone(), port.clone())) {
                        return Err(format!("multiple drivers for input '{instance}.{port}'"));
                    }
                    CheckedStmt::Connect {
                        instance: instance.clone(),
                        port: port.clone(),
                        value,
                    }
                }
                Stmt::Let { name, ty, value } => {
                    if let Some(t) = ty {
                        self.decls.validate_type(t)?;
                    }
                    let value = if let Expr::Call { name: target, args } = value
                        && let Some(child) = self.decls.blocks.get(target.as_str())
                    {
                        if conditional {
                            return Err(format!(
                                "call to block '{target}' requires an unconditional top-level let"
                            ));
                        }
                        self.block_call(name, target, child, args, ty.as_ref())
                    } else {
                        self.expr(value, ty.as_ref())
                    }
                    .map_err(|e| format!("let '{name}': {e}"))?;
                    declare(&mut self.names, name)?;
                    self.values.insert(name.clone(), value.ty.clone());
                    if let Some(v) = value.constant(&self.constants) {
                        self.constants.insert(name.clone(), v);
                    }
                    if conditional {
                        locals.push(name);
                    }
                    CheckedStmt::Let {
                        name: name.clone(),
                        value,
                    }
                }
                Stmt::Write { name, value } => {
                    let ty = self
                        .regs
                        .get(name)
                        .ok_or_else(|| format!("'{name}' is not a declared register"))?;
                    let value = self
                        .expr(value, Some(ty))
                        .map_err(|e| format!("update of '{name}': {e}"))?;
                    if !self.writes.insert(name.clone()) {
                        return Err(format!("multiple updates to register '{name}'"));
                    }
                    CheckedStmt::Write {
                        name: name.clone(),
                        value,
                    }
                }
                Stmt::If { condition, body } => {
                    let condition = self
                        .expr(condition, Some(&Type::Bool))
                        .map_err(|e| format!("write guard: {e}"))?;
                    let start = self.statements.len();
                    self.body(body, true)?;
                    CheckedStmt::If {
                        condition,
                        body: self.statements.split_off(start),
                    }
                }
            };
            self.statements.push(checked);
        }
        for name in locals {
            self.names.remove(name);
            self.values.remove(name);
            self.constants.remove(name);
        }
        Ok(())
    }

    /// Propagate expected types to literals, aggregate members, and pure call
    /// arguments.
    fn expr(&self, expr: &Expr, expected: Option<&Type>) -> Result<TypedExpr, String> {
        let (ty, kind) = match expr {
            Expr::Name(name) => {
                let ty = self
                    .values
                    .get(name)
                    .ok_or_else(|| format!("unknown value '{name}'"))?;
                (ty.clone(), K::Name(name.clone()))
            }
            Expr::Bool(value) => (Type::Bool, K::Literal(IrValue::bool(*value))),
            Expr::Literal { ty, text } => {
                let ty = ty
                    .as_ref()
                    .or(expected)
                    .ok_or_else(|| format!("literal '{text}' requires a concrete bit type"))?;
                let (signed, width) =
                    bits(ty).ok_or_else(|| format!("integer literal cannot have type {ty}"))?;
                self.decls.validate_type(ty)?;
                if !signed && text.starts_with('-') {
                    return Err(format!("negative literal '{text}' has unsigned type {ty}"));
                }
                let value = IrValue::parse_typed(&format!("bits[{width}]:{text}"))
                    .map_err(|e| format!("invalid literal '{ty}:{text}': {e}"))?;
                // Like DSLX, hexadecimal and binary literals can spell a signed
                // bit pattern directly, e.g. s8:0xff represents -1.
                if signed
                    && !text.starts_with('-')
                    && !text.starts_with("0x")
                    && !text.starts_with("0b")
                    && value.as_bits().map_err(|e| e.to_string())?.is_negative()
                {
                    return Err(format!("literal '{text}' does not fit in {ty}"));
                }
                (ty.clone(), K::Literal(value))
            }
            Expr::Tuple(items) => {
                let hints = match expected {
                    Some(Type::Tuple(ts)) if ts.len() == items.len() => Some(ts),
                    _ => None,
                };
                let values = items
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.expr(v, hints.map(|ts| &ts[i])))
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    Type::Tuple(values.iter().map(|v| v.ty.clone()).collect()),
                    K::Tuple(values),
                )
            }
            Expr::Array(items) => {
                let hint = match expected {
                    Some(Type::Array { element, size }) => {
                        if *size != items.len() {
                            return Err(format!(
                                "expected array length {size}, got {}",
                                items.len()
                            ));
                        }
                        Some(element.as_ref())
                    }
                    Some(other) => return Err(format!("expected {other}, got an array")),
                    None => None,
                };
                if let Some((first, rest)) = items.split_first() {
                    let first = self.expr(first, hint)?;
                    let element = first.ty.clone();
                    let mut values = vec![first];
                    for item in rest {
                        values.push(self.expr(item, Some(&element))?);
                    }
                    (
                        Type::Array {
                            element: Box::new(element),
                            size: items.len(),
                        },
                        K::Array(values),
                    )
                } else {
                    let element = hint
                        .ok_or_else(|| "empty array literal requires an array type".to_owned())?;
                    let mut layouts = crate::layout::Layouts::new(self.decls.struct_list);
                    let element_ir = layouts.lower(element).map_err(|e| e.to_string())?;
                    let empty =
                        IrValue::make_array_typed(element_ir, &[]).map_err(|e| e.to_string())?;
                    (
                        Type::Array {
                            element: Box::new(element.clone()),
                            size: 0,
                        },
                        K::Literal(empty),
                    )
                }
            }
            Expr::Index { value, index } => {
                let array = self.expr(value, None)?;
                let Type::Array { element, size } = &array.ty else {
                    return Err(format!("indexing requires an array, got {}", array.ty));
                };
                if *size == 0 {
                    return Err("cannot index an empty array".into());
                }
                let index = self.unsigned_index(index)?;
                (
                    element.as_ref().clone(),
                    K::Index {
                        value: Box::new(array),
                        index: Box::new(index),
                    },
                )
            }
            Expr::Struct { name, fields } => {
                let s = self
                    .decls
                    .structs
                    .get(name.as_str())
                    .ok_or_else(|| format!("unknown struct '{name}'"))?;
                let mut supplied = BTreeMap::new();
                for (field, value) in fields {
                    if supplied.insert(field.as_str(), value).is_some() {
                        return Err(format!("duplicate field '{name}.{field}'"));
                    }
                }
                let mut values = Vec::new();
                for p in &s.fields {
                    let v = supplied
                        .remove(p.name.as_str())
                        .ok_or_else(|| format!("missing field '{name}.{}'", p.name))?;
                    values.push(
                        self.expr(v, Some(&p.ty))
                            .map_err(|e| format!("field '{name}.{}': {e}", p.name))?,
                    );
                }
                if let Some(field) = supplied.keys().next() {
                    return Err(format!("unknown field '{name}.{field}'"));
                }
                (Type::Named(name.clone()), K::Tuple(values))
            }
            Expr::Field { value, name } => {
                if let Expr::Name(instance) = value.as_ref()
                    && let Some(child) = self.instances.get(instance)
                {
                    let p = child
                        .outputs
                        .iter()
                        .find(|p| p.name == *name)
                        .ok_or_else(|| format!("'{instance}.{name}' is not an output"))?;
                    (
                        p.ty.clone(),
                        K::InstanceOutput {
                            instance: instance.clone(),
                            port: name.clone(),
                        },
                    )
                } else {
                    let v = self.expr(value, None)?;
                    let (index, ty) = match &v.ty {
                        Type::Tuple(items) => {
                            let index: usize = name
                                .parse()
                                .map_err(|_| format!("invalid tuple field '{name}'"))?;
                            (
                                index,
                                items
                                    .get(index)
                                    .ok_or_else(|| format!("tuple field {index} out of bounds"))?
                                    .clone(),
                            )
                        }
                        Type::Named(s) => {
                            let fields = &self.decls.structs[s.as_str()].fields;
                            let (i, p) = fields
                                .iter()
                                .enumerate()
                                .find(|(_, p)| p.name == *name)
                                .ok_or_else(|| format!("unknown field '{s}.{name}'"))?;
                            (i, p.ty.clone())
                        }
                        other => return Err(format!("cannot project field '{name}' from {other}")),
                    };
                    (
                        ty,
                        K::TupleIndex {
                            value: Box::new(v),
                            index,
                        },
                    )
                }
            }
            Expr::Call { name, args } => {
                if self.decls.blocks.contains_key(name.as_str()) {
                    return Err(format!(
                        "call to block '{name}' must be the entire right-hand side of an unconditional top-level let"
                    ));
                }
                let f = self
                    .decls
                    .externs
                    .get(name.as_str())
                    .ok_or_else(|| format!("unknown extern function '{name}'"))?;
                if args.len() != f.params.len() {
                    return Err(format!(
                        "'{name}' expects {} arguments, got {}",
                        f.params.len(),
                        args.len()
                    ));
                }
                let args = args
                    .iter()
                    .zip(&f.params)
                    .map(|(a, p)| {
                        self.expr(a, Some(&p.ty))
                            .map_err(|e| format!("argument '{}', call to '{name}': {e}", p.name))
                    })
                    .collect::<Result<_, _>>()?;
                (
                    f.result.clone(),
                    K::Call {
                        name: name.clone(),
                        args,
                    },
                )
            }
            Expr::Cast { value, ty } => {
                self.decls.validate_type(ty)?;
                let v = self.expr(
                    value,
                    matches!(value.as_ref(), Expr::Literal { ty: None, .. }).then_some(ty),
                )?;
                if bits(&v.ty).is_none() || bits(ty).is_none() {
                    return Err(format!("cast from {} to {ty} requires bit types", v.ty));
                }
                (ty.clone(), K::Cast { value: Box::new(v) })
            }
            Expr::If {
                condition,
                then_value,
                else_value,
            } => {
                let cond = self.expr(condition, Some(&Type::Bool))?;
                let (t, f) = self.pair(then_value, else_value, expected)?;
                (
                    t.ty.clone(),
                    K::If {
                        condition: Box::new(cond),
                        then_value: Box::new(t),
                        else_value: Box::new(f),
                    },
                )
            }
            Expr::Unary { op, value } => {
                let v = self.expr(value, expected)?;
                if bits(&v.ty).is_none() {
                    return Err(format!("invalid unary '{op}' for {}", v.ty));
                }
                (
                    v.ty.clone(),
                    K::Unary {
                        op: *op,
                        value: Box::new(v),
                    },
                )
            }
            Expr::Binary { op, left, right } => self.binary(*op, left, right, expected)?,
        };
        if let Some(expected) = expected
            && !compatible(&ty, expected)
        {
            return Err(format!("expected {expected}, got {ty}"));
        }
        if matches!(expr, Expr::Tuple(_) | Expr::Array(_)) {
            self.decls.validate_type(&ty)?;
        }
        Ok(TypedExpr { ty, kind })
    }

    /// Infer an unannotated integer from the opposite operand when no result
    /// type is available.
    fn pair(
        &self,
        left: &Expr,
        right: &Expr,
        expected: Option<&Type>,
    ) -> Result<(TypedExpr, TypedExpr), String> {
        if expected.is_none() && matches!(left, Expr::Literal { ty: None, .. }) {
            let r = self.expr(right, None)?;
            let l = self.expr(left, Some(&r.ty))?;
            return Ok((l, r));
        }
        let l = self.expr(left, expected)?;
        let r = self.expr(right, Some(&l.ty))?;
        Ok((l, r))
    }

    /// Type-check a shift distance or array index, defaulting a bare literal to
    /// u32.
    fn unsigned_index(&self, expr: &Expr) -> Result<TypedExpr, String> {
        let hint = Type::Bits {
            signed: false,
            width: 32,
        };
        let v = self.expr(
            expr,
            matches!(expr, Expr::Literal { ty: None, .. }).then_some(&hint),
        )?;
        if !bits(&v.ty).is_some_and(|(s, _)| !s) {
            return Err(format!("expected unsigned index, got {}", v.ty));
        }
        Ok(v)
    }

    /// Select operator signatures before lowering erases signedness and nominal
    /// types.
    fn binary(
        &self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let (ty, l, r) = match op {
            BinaryOp::Shl | BinaryOp::Shr => {
                let l = self.expr(left, expected)?;
                let r = self.unsigned_index(right)?;
                if bits(&l.ty).is_none() {
                    return Err(format!("'{op}' requires a bit value"));
                }
                (l.ty.clone(), l, r)
            }
            BinaryOp::Concat => {
                let l = self.expr(left, None)?;
                let r = self.expr(right, None)?;
                let (Some((false, left_width)), Some((false, right_width))) =
                    (bits(&l.ty), bits(&r.ty))
                else {
                    return Err(format!(
                        "bit concatenation requires unsigned operands, got {} and {}",
                        l.ty, r.ty
                    ));
                };
                let width = left_width
                    .checked_add(right_width)
                    .ok_or_else(|| "concatenation width overflow".to_owned())?;
                let ty = Type::Bits {
                    signed: false,
                    width,
                };
                self.decls.validate_type(&ty)?;
                (ty, l, r)
            }
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                let (l, r) = self.pair(left, right, None)?;
                if !matches!(op, BinaryOp::Eq | BinaryOp::Ne) && bits(&l.ty).is_none() {
                    return Err(format!("comparison '{op}' requires bit operands"));
                }
                (Type::Bool, l, r)
            }
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                let (l, r) = self.pair(left, right, Some(&Type::Bool))?;
                (Type::Bool, l, r)
            }
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor => {
                let (l, r) = self.pair(left, right, expected)?;
                if bits(&l.ty).is_none() {
                    return Err(format!("'{op}' requires bit operands, got {}", l.ty));
                }
                (l.ty.clone(), l, r)
            }
        };
        Ok((
            ty,
            K::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckedStmt, MAX_ELABORATION_DEPTH, check_module};
    use crate::parser::parse_module;
    use crate::{ResetBehavior, compile, empty_package};
    use xlsynth_pir::IrValue;

    #[test]
    fn grouped_operator_chains_obey_the_expression_tree_depth_limit() {
        let check = |expression: &str, result: &str| {
            let source = format!("block b(clk, x: u8) -> {result} {{ {expression} }}");
            let module = parse_module(&source)?;
            check_module(&module)
        };
        // Each group has only 16 operators; composed, the tree has 801 levels.
        let chain = format!(
            "{}x{}",
            "(".repeat(50),
            format!("{})", " + x".repeat(16)).repeat(50)
        );
        assert!(
            check(&chain, "u8")
                .unwrap_err()
                .contains("expression tree exceeds")
        );
        check(&format!("x{}", " + x".repeat(63)), "u8").unwrap();
        let wide = format!("[{}]", "x + x,".repeat(1000));
        check(&wide, "u8[1000]").unwrap();
    }

    #[test]
    fn bit_concatenation_requires_unsigned_operands_on_both_sides() {
        for (left, right) in [("s4", "u4"), ("u4", "s4"), ("s4", "s4")] {
            let source = format!("block join(clk, x: {left}, y: {right}) -> u8 {{ x ++ y }}");
            assert_eq!(
                check_module(&parse_module(&source).unwrap()).unwrap_err(),
                format!(
                    "block 'join': result: bit concatenation requires unsigned operands, got {left} and {right}"
                )
            );
        }
        let source = "block join(clk, x: s4, y: u4) -> u8 { (x as u4) ++ y }";
        check_module(&parse_module(source).unwrap()).unwrap();
    }

    #[test]
    fn constant_aggregate_resets_preserve_field_order_and_wide_signed_values() {
        let source = r#"
struct Payload { words: u8[2], wide: sN[130] }
block seeded(clk, rst) -> Payload {
    let seed = u8:9 * u8:4;
    reg q: Payload = Payload { wide: sN[130]:-2, words: [seed, seed + 1] };
    q := q;
    q
}"#;
        let module = check_module(&parse_module(source).unwrap()).unwrap();
        let CheckedStmt::Reg { reset, .. } = &module.blocks()[0].statements[1] else {
            panic!("expected a register after seed");
        };
        assert_eq!(
            *reset,
            Some(IrValue::parse_typed("([bits[8]:36, bits[8]:37], bits[130]:-2)").unwrap())
        );
    }

    #[test]
    fn types_are_nominal_and_signed_until_linking() {
        for expression in ["x", "B { data: u8:1 }", "A { data: s8:1 }"] {
            let source = format!(
                r#"
struct A {{ data: u8 }}
struct B {{ data: u8 }}
block nominal(clk, x: B) -> A {{ {expression} }}"#
            );
            let module = parse_module(&source).unwrap();
            assert!(check_module(&module).is_err(), "{expression}");
        }
    }

    #[test]
    fn hierarchy_requires_exactly_one_unconditional_data_driver() {
        for connections in [
            "",
            "d.x := 0; d.x := 1;",
            "if en { d.x := 0; }",
            "d.out := 0;",
        ] {
            let source = format!(
                r#"
block parent(clk, en: bool) -> u8 {{ inst d: child; {connections} d.out }}
block child(clk, x: u8) -> u8 {{ x }}"#
            );
            let module = parse_module(&source).unwrap();
            assert!(check_module(&module).is_err(), "{connections}");
        }
    }

    #[test]
    fn call_instance_names_avoid_parent_output_and_future_register_names() {
        let source = r#"
extern block leaf(clk) -> u8;
block wrap(clk) -> u8 {
    let out = leaf();
    reg out_inst: u8;
    out_inst := out;
    out_inst
}"#;
        let checked = check_module(&parse_module(source).unwrap()).unwrap();
        assert_eq!(checked.extern_blocks()[0].name, "leaf");
        let CheckedStmt::Inst { name, block } = &checked.blocks()[0].statements[0] else {
            panic!("the call must declare an instance");
        };
        assert_eq!((name.as_str(), block.as_str()), ("out_inst_1", "leaf"));
    }

    #[test]
    fn only_explicit_instances_expose_ports_before_the_declaration() {
        let source = r#"
extern block leaf(clk, d: u8) -> u8;
block wrap(clk, x: u8) -> u8 {
    let before = lane.out;
    inst lane: leaf;
    lane.d := x;
    before
}"#;
        check_module(&parse_module(source).unwrap()).unwrap();
        let lexical = source.replace("inst lane: leaf;\n    lane.d := x;", "let lane = leaf(x);");
        assert!(check_module(&parse_module(&lexical).unwrap()).is_err());
    }

    #[test]
    fn extern_block_names_and_physical_ports_are_unique() {
        for source in [
            "extern block a(clk, x: u8, x: u8) -> ();",
            "extern block a(clk, out: u8) -> u8;",
            "extern block a(clk) -> (); block a(clk) -> () { () }",
            "extern fn a() -> (); extern block a(clk) -> ();",
        ] {
            assert!(
                check_module(&parse_module(source).unwrap()).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn guards_have_lexical_scope_and_reset_is_a_logical_value() {
        let source = r#"block stage(clk, rst, x: u8) -> u8 {
    reg q: u8;
    if !rst { let local = x; q := local; }
    q
}"#;
        check_module(&parse_module(source).unwrap()).unwrap();
        for bad in [
            source.replace("    q\n", "    local\n"),
            source.replace("!rst", "clk"),
        ] {
            assert!(check_module(&parse_module(&bad).unwrap()).is_err());
        }
    }

    #[test]
    fn type_sizes_include_zero_width_members_and_are_checked_before_materialization() {
        for ty in ["uN[18446744073709551615]", "u0[1048576]", "u16[65536]"] {
            let source = format!("block sized(clk, x: {ty}) -> bool {{ false }}");
            assert!(
                parse_module(&source)
                    .and_then(|module| check_module(&module))
                    .is_err(),
                "{ty}"
            );
        }
    }

    #[test]
    fn hierarchy_and_expanded_type_depth_include_already_checked_dependencies() {
        for levels in [MAX_ELABORATION_DEPTH, MAX_ELABORATION_DEPTH + 1, 512] {
            let mut blocks = vec!["block b0(clk) -> () { () }".to_owned()];
            let mut structs = vec!["struct S0 { f: u1 }".to_owned()];
            for i in 1..levels {
                blocks.push(format!(
                    "block b{i}(clk) -> () {{ inst child: b{}; () }}",
                    i - 1
                ));
                structs.push(format!("struct S{i} {{ f: S{} }}", i - 1));
            }
            for declarations in [&mut blocks, &mut structs] {
                for _ in 0..2 {
                    let module = parse_module(&declarations.join("\n")).unwrap();
                    assert_eq!(
                        check_module(&module).is_ok(),
                        levels <= MAX_ELABORATION_DEPTH
                    );
                    declarations.reverse();
                }
            }
        }
    }

    #[test]
    fn expanded_types_count_arrays_and_tuples_between_named_structs() {
        let mut definitions = vec!["struct S0 { f: u1 }".to_owned()];
        for i in 1..=MAX_ELABORATION_DEPTH / 3 {
            definitions.push(format!("struct S{i} {{ f: (S{}[1],) }}", i - 1));
        }
        // Depth 1 + 3*21 = 64, so an additional aggregate exceeds the limit.
        let name = format!("S{}", MAX_ELABORATION_DEPTH / 3);
        let valid = definitions.join("\n");
        check_module(&parse_module(&valid).unwrap()).unwrap();
        let too_deep = format!("{valid}\nextern fn f(x: {name}[1]) -> bool;");
        assert!(check_module(&parse_module(&too_deep).unwrap()).is_err());
    }

    #[test]
    fn empty_arrays_account_for_their_expanded_element_types() {
        let mut definitions = vec!["struct S0 { f: u1 }".to_owned()];
        for i in 1..=20 {
            definitions.push(format!(
                "struct S{i} {{ a: S{}[0], b: S{}[0] }}",
                i - 1,
                i - 1
            ));
        }
        let module = parse_module(&definitions.join("\n")).unwrap();
        assert!(
            check_module(&module)
                .unwrap_err()
                .contains("maximum concrete type size")
        );
    }

    #[test]
    fn aggregate_equality_accepts_matching_arrays_and_tuples_in_block_ir() {
        let source = r#"
block same(clk, a: (s8, u8[3]), b: (s8, u8[3])) -> bool { a == b }
"#;
        compile(
            source,
            empty_package("comparison"),
            ResetBehavior::default(),
        )
        .unwrap();
    }

    #[test]
    fn constant_index_clamps_and_zero_width_cast_extends_with_zero() {
        let source = r#"
block constants(clk, rst) -> (s8, u8) {
    let samples = [s8:0xff, s8:0b1000_0000];
    reg q: (s8, u8) = (samples[uN[100]:0xffffffffffffffffffff], sN[0]:0 as u8);
    q := q;
    q
}"#;
        let checked = check_module(&parse_module(source).unwrap()).unwrap();
        let CheckedStmt::Reg { reset, .. } = &checked.blocks()[0].statements[1] else {
            panic!("expected a register after samples");
        };
        assert_eq!(
            *reset,
            Some(IrValue::parse_typed("(bits[8]:-128, bits[8]:0)").unwrap())
        );
        for expression in ["s8:128", "s8:-129", "u8:-1", "s8:1[u1:0]"] {
            let source = format!("block invalid(clk) -> s8 {{ {expression} }}");
            assert!(
                check_module(&parse_module(&source).unwrap()).is_err(),
                "{expression}"
            );
        }
    }
}
