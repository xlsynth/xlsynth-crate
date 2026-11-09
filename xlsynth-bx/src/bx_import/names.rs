// SPDX-License-Identifier: Apache-2.0

//! Resolve module names before BX declarations share a checking environment.

use std::collections::{BTreeMap, BTreeSet};

use super::{BUILTIN_MAP, BUILTIN_UPDATE, Binding, Names, ParametricKind};
use crate::ast::{
    BindingPattern, Block, CallbackParam, ConstDef, EnumDef, EnumMember, Expr, ExternBlock,
    ExternFn, ForIterable, ForLoop, Function, FunctionParametricRef, LetBinding, MapExpr, MatchArm,
    MatchPattern, Module, Param, ParametricBinding, Stmt, StructDef, Type, UnrollFor,
    WidthSliceType,
};
use crate::parser::MAX_SYNTAX_DEPTH;

/// A bound data name in call position. Zero-trip bodies only require it to
/// be in scope, while elaborating the call must report that it is a value.
const LOCAL_CALLEE_PREFIX: &str = "$local_callee$";

/// Keep a call on a data binding distinct from a call on a global declaration.
pub(crate) fn mark_local_callee(name: &str) -> String {
    format!("{LOCAL_CALLEE_PREFIX}{name}")
}

/// Recover the source value name when inspecting a qualified direct call.
pub(crate) fn local_callee_name(callee: &str) -> Option<&str> {
    callee.strip_prefix(LOCAL_CALLEE_PREFIX)
}

/// Copy declarations into the shared module namespace. Copy recursive nodes
/// during the bounded walk so programmatically supplied ASTs are checked
/// before a recursive clone can exhaust the stack.
pub(super) fn qualify(module: &Module, names: &Names) -> Result<Module, String> {
    let position = |name: &str| names.positions.get(name).copied();
    let mut out = Module::default();
    for def in &module.structs {
        out.structs.push(StructDef {
            name: names.declared(&def.name),
            public: def.public,
            fields: Scope::new(names, None, None).fields(&def.fields)?,
        });
    }
    for def in &module.enums {
        let at = position(&def.name);
        let mut scope = Scope::new(names, at, at);
        out.enums.push(EnumDef {
            name: names.declared(&def.name),
            public: def.public,
            underlying: scope.optional_type(&def.underlying)?,
            members: def
                .members
                .iter()
                .map(|member| {
                    Ok(EnumMember {
                        name: member.name.clone(),
                        value: scope.expr(&member.value)?,
                    })
                })
                .collect::<Result<_, String>>()?,
        });
    }
    for def in &module.constants {
        let at = position(&def.name);
        let mut scope = Scope::new(names, at, at);
        out.constants.push(ConstDef {
            name: names.declared(&def.name),
            public: def.public,
            ty: scope.optional_type(&def.ty)?,
            value: scope.expr(&def.value)?,
        });
    }
    for def in &module.externs {
        let mut scope = Scope::new(names, None, None);
        out.externs.push(ExternFn {
            name: names.declared(&def.name),
            params: scope.params(&def.params, Local::Lexical)?,
            result: scope.ty(&def.result)?,
            symbol: def.symbol.clone(),
        });
    }
    for def in &module.extern_blocks {
        let mut scope = Scope::new(names, None, None);
        scope.timing_ports(&def.clock, def.reset.as_deref());
        out.extern_blocks.push(ExternBlock {
            name: names.declared(&def.name),
            clock: def.clock.clone(),
            reset: def.reset.clone(),
            params: scope.params(&def.params, Local::Hardware)?,
            result: scope.ty(&def.result)?,
        });
    }
    for def in &module.functions {
        let at = position(&def.name);
        let mut scope = Scope::new(names, at, at);
        let mut callback_params = Vec::with_capacity(def.callback_params.len());
        let mut value_parametrics = Vec::with_capacity(def.value_parametrics.len());
        for formal in def.parametric_bindings() {
            match formal {
                FunctionParametricRef::Value(value) => {
                    let ty = scope.ty(&value.ty)?;
                    // A binding shadows an outer constant even in its own
                    // default. Reading that value before it is bound fails.
                    scope.bind(&value.name);
                    value_parametrics.push(ParametricBinding {
                        name: value.name.clone(),
                        ty,
                        default: value.default.as_ref().map(|e| scope.expr(e)).transpose()?,
                        index: value.index,
                    });
                }
                FunctionParametricRef::Callback(callback) => {
                    callback_params.push(scope.callback(callback)?);
                }
            }
        }
        let params = scope.params(&def.params, Local::Lexical)?;
        let result = scope.ty(&def.result)?;
        out.functions.push(Function {
            name: names.declared(&def.name),
            public: def.public,
            callback_params,
            value_parametrics,
            params,
            result,
            body: scope.expr(&def.body)?,
        });
    }
    let hardware_callees = hardware_callees(module);
    for def in &module.blocks {
        out.blocks
            .push(qualify_block(def, names, &hardware_callees)?);
    }
    Ok(out)
}

fn hardware_callees(module: &Module) -> BTreeSet<&str> {
    module
        .externs
        .iter()
        .map(|d| d.name.as_str())
        .chain(module.blocks.iter().map(|d| d.name.as_str()))
        .chain(module.extern_blocks.iter().map(|d| d.name.as_str()))
        .collect()
}

/// Blocks can call blocks and functions declared later. A constant read by a
/// block must already have been introduced.
fn qualify_block(
    block: &Block,
    names: &Names,
    hardware_callees: &BTreeSet<&str>,
) -> Result<Block, String> {
    let mut scope = Scope::new(names, None, names.positions.get(&block.name).copied());
    scope.hardware_callees = Some(hardware_callees);
    scope.block(block)
}

/// Resolve local callees in an unlinked, uninstantiated block template whose
/// possible child names overlap data binders.
pub(crate) struct UnlinkedBlockScope<'m> {
    names: Names,
    hardware_callees: BTreeSet<&'m str>,
}

impl<'m> UnlinkedBlockScope<'m> {
    pub(crate) fn new(module: &'m Module) -> Result<Self, String> {
        let uses = module
            .uses
            .iter()
            .filter_map(|u| u.binding())
            .map(|name| (name.to_owned(), super::Binding::Dslx(name.to_owned())))
            .collect();
        Ok(Self {
            names: Names::new(module, None, uses)?,
            hardware_callees: hardware_callees(module),
        })
    }

    pub(crate) fn qualify(&self, block: &Block) -> Result<Block, String> {
        qualify_block(block, &self.names, &self.hardware_callees)
    }
}

/// BX data wires can share a spelling with an extern or a structural child
/// target. Lexical lets and loop binders shadow those targets as well.
#[derive(Clone, Copy, PartialEq)]
enum Local {
    Lexical,
    Hardware,
    Formal,
}

/// A direct call may be inspected in a skipped loop. Explicit instances
/// and callback lists require a callable target during name checking.
#[derive(Clone, Copy, PartialEq)]
enum CallSite {
    Direct,
    Parametric,
    Structural,
}

/// Local binders hide module names until the end of their lexical scope.
struct Scope<'a> {
    names: &'a Names,
    at: Option<usize>,
    value_at: Option<usize>,
    locals: BTreeMap<String, Local>,
    /// A declaration's type and callable formals keep module callees hidden
    /// even when a nearer data binding shadows the formal itself.
    formals: BTreeSet<String>,
    introduced: Vec<(String, Option<Local>)>,
    hardware_callees: Option<&'a BTreeSet<&'a str>>,
    expression_depth: usize,
    guard_depth: usize,
}

impl<'a> Scope<'a> {
    fn new(names: &'a Names, at: Option<usize>, value_at: Option<usize>) -> Self {
        Self {
            names,
            at,
            value_at,
            locals: BTreeMap::new(),
            formals: BTreeSet::new(),
            hardware_callees: None,
            introduced: Vec::new(),
            expression_depth: 0,
            guard_depth: 0,
        }
    }

    fn bind_as(&mut self, name: &str, binding: Local) {
        if binding == Local::Formal {
            self.formals.insert(name.to_owned());
        }
        if (name != "_" || binding == Local::Hardware) && self.locals.get(name) != Some(&binding) {
            let previous = self.locals.insert(name.to_owned(), binding);
            self.introduced.push((name.to_owned(), previous));
        }
    }

    fn bind(&mut self, name: &str) {
        self.bind_as(name, Local::Lexical);
    }

    fn bind_pattern(&mut self, pattern: &BindingPattern) {
        pattern.visit_names(&mut |name| self.bind(name));
    }

    fn nested<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let start = self.introduced.len();
        let value = body(self)?;
        for (name, previous) in self.introduced.drain(start..).rev() {
            if let Some(previous) = previous {
                self.locals.insert(name, previous);
            } else {
                self.locals.remove(&name);
            }
        }
        Ok(value)
    }

    fn reference(&self, name: &str, at: Option<usize>) -> Result<String, String> {
        if name.contains('$') {
            return Err(format!("invalid source reference '{name}'"));
        }
        if let Some((head, _)) = name.split_once("::")
            && self.locals.contains_key(head)
        {
            return Err(format!("'{head}' names a local value here"));
        }
        if self.locals.contains_key(name) {
            Ok(name.to_owned())
        } else {
            self.names.resolve(name, at)
        }
    }

    /// Preserve BX's structural call namespace for a data wire that has the
    /// same spelling as an extern function or a locally declared child block.
    fn callee(&self, name: &str, site: CallSite) -> Result<String, String> {
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if self.formals.contains(head)
            && self.locals.get(head) != Some(&Local::Formal)
            && self.names.definitions.contains(name)
        {
            // Keep the source spelling so the checker sees both the formal
            // and the nearer value when checking a skipped or executed call.
            return Ok(name.to_owned());
        }
        // A qualified extern is declared under its full name. An ordinary
        // value called `chip` does not hide the extern `chip::step`.
        if let Some((head, _)) = name.split_once("::")
            && self.names.definitions.contains(name)
            && self.locals.get(head) != Some(&Local::Formal)
        {
            return self.names.resolve(name, self.at);
        }
        match self.locals.get(name) {
            Some(Local::Hardware)
                if site != CallSite::Parametric
                    && self
                        .hardware_callees
                        .is_some_and(|known| known.contains(name)) =>
            {
                self.names.resolve(name, self.at)
            }
            Some(Local::Lexical | Local::Hardware)
                if site == CallSite::Direct
                    || site == CallSite::Parametric
                        && matches!(self.names.uses.get(name), Some(Binding::Dslx(_))) =>
            {
                // An imported function can be hidden by a data binding. DSLX
                // resolves that local in a skipped loop and diagnoses calling
                // it only when an iteration is typechecked.
                Ok(mark_local_callee(name))
            }
            Some(Local::Lexical | Local::Hardware) => {
                Err(format!("'{name}' names a local value here"))
            }
            None if site == CallSite::Direct && name == "map" => self
                .names
                .resolve_with_root_fallback(name, self.builtin_at(name), BUILTIN_MAP),
            None if site == CallSite::Direct && name == "update" => self
                .names
                .resolve_with_root_fallback(name, self.builtin_at(name), BUILTIN_UPDATE),
            Some(Local::Formal) | None => self.reference(name, self.at),
        }
    }

    /// A block sees callable declarations throughout its module. Enum names
    /// that shadow the builtin follow the order of value declarations.
    fn builtin_at(&self, name: &str) -> Option<usize> {
        if self.names.enums.contains(name)
            || matches!(self.names.uses.get(name), Some(Binding::Member(export)) if export.is_enum)
        {
            self.value_at
        } else {
            self.at
        }
    }

    /// Resolve qualified map callbacks like direct calls, including a qualified
    /// extern whose prefix happens to name a local value.
    fn map_callback(&self, name: &str) -> Result<String, String> {
        if name.contains("::") {
            self.callee(name, CallSite::Direct)
        } else {
            self.reference(name, self.at)
        }
    }

    fn ty(&mut self, ty: &Type) -> Result<Type, String> {
        self.type_at_depth(ty, 0)
    }

    fn type_at_depth(&mut self, ty: &Type, depth: usize) -> Result<Type, String> {
        if depth > MAX_SYNTAX_DEPTH {
            return Err(format!("BX import type exceeds {MAX_SYNTAX_DEPTH} levels"));
        }
        Ok(match ty {
            Type::Named(name) => Type::Named(self.reference(name, self.at)?),
            Type::Enum {
                name,
                signed,
                width,
            } => Type::Enum {
                name: self.reference(name, self.at)?,
                signed: *signed,
                width: *width,
            },
            Type::Applied { name, args } => Type::Applied {
                name: self.reference(name, self.at)?,
                args: args.clone(),
            },
            Type::Tuple(fields) => Type::Tuple(
                fields
                    .iter()
                    .map(|field| self.type_at_depth(field, depth + 1))
                    .collect::<Result<_, _>>()?,
            ),
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.type_at_depth(element, depth + 1)?),
                size: *size,
            },
            Type::ParametricArray { element, size } => Type::ParametricArray {
                element: Box::new(self.type_at_depth(element, depth + 1)?),
                size: self.dimension(size, depth)?,
            },
            Type::ParametricBits { signed, width } => Type::ParametricBits {
                signed: self.dimension(signed, depth)?,
                width: self.dimension(width, depth)?,
            },
            Type::Bits { .. } | Type::Bool => ty.clone(),
        })
    }

    /// Carry type nesting across a dimension expression that can itself
    /// contain type annotations.
    fn dimension(&mut self, expr: &Expr, depth: usize) -> Result<Box<Expr>, String> {
        self.expression_depth += depth + 1;
        let result = self.expr(expr);
        self.expression_depth -= depth + 1;
        result.map(Box::new)
    }

    fn optional_type(&mut self, ty: &Option<Type>) -> Result<Option<Type>, String> {
        ty.as_ref().map(|ty| self.ty(ty)).transpose()
    }

    /// Struct field names do not shadow other field types.
    fn fields(&mut self, fields: &[Param]) -> Result<Vec<Param>, String> {
        fields.iter().map(|p| self.param(p)).collect()
    }

    fn param(&mut self, p: &Param) -> Result<Param, String> {
        Ok(Param {
            name: p.name.clone(),
            ty: self.ty(&p.ty)?,
        })
    }

    /// A parameter enters signature scope before its own type annotation.
    fn params(&mut self, params: &[Param], local: Local) -> Result<Vec<Param>, String> {
        params
            .iter()
            .map(|p| {
                self.bind_as(&p.name, local);
                self.param(p)
            })
            .collect()
    }

    fn timing_ports(&mut self, clock: &str, reset: Option<&str>) {
        self.bind_as(clock, Local::Hardware);
        if let Some(reset) = reset {
            self.bind_as(reset, Local::Hardware);
        }
    }

    fn callback(&mut self, param: &CallbackParam) -> Result<CallbackParam, String> {
        self.bind_as(&param.name, Local::Formal);
        Ok(CallbackParam {
            name: param.name.clone(),
            params: param
                .params
                .iter()
                .map(|ty| self.ty(ty))
                .collect::<Result<_, _>>()?,
            result: self.ty(&param.result)?,
        })
    }

    fn binding(&mut self, binding: &LetBinding) -> Result<LetBinding, String> {
        let result = LetBinding {
            pattern: copy_pattern(&binding.pattern, 0)?,
            ty: self.optional_type(&binding.ty)?,
            value: self.expr(&binding.value)?,
        };
        self.bind_pattern(&result.pattern);
        Ok(result)
    }

    /// Resolve a block body with explicit instances visible before their
    /// statements.
    fn block(&mut self, def: &Block) -> Result<Block, String> {
        for name in &def.type_params {
            self.bind_as(name, Local::Formal);
        }
        let callback_params = def
            .callback_params
            .iter()
            .map(|p| self.callback(p))
            .collect::<Result<_, _>>()?;
        self.timing_ports(&def.clock, def.reset.as_deref());
        let params = self.params(&def.params, Local::Hardware)?;
        let result = self.ty(&def.result)?;
        for stmt in &def.statements {
            if let Stmt::Inst { name, .. } = stmt {
                self.bind_as(name, Local::Hardware);
            }
        }
        let statements = def
            .statements
            .iter()
            .map(|stmt| self.statement(stmt, Local::Hardware))
            .collect::<Result<_, _>>()?;
        Ok(Block {
            name: self.names.declared(&def.name),
            name_position: def.name_position,
            allowed_warnings: def.allowed_warnings.clone(),
            public: def.public,
            type_params: def.type_params.clone(),
            callback_params,
            clock: def.clock.clone(),
            reset: def.reset.clone(),
            params,
            result,
            statements,
            output: self.expr(&def.output)?,
        })
    }

    fn statement(&mut self, stmt: &Stmt, local: Local) -> Result<Stmt, String> {
        Ok(match stmt {
            Stmt::Reg { name, ty, reset } => {
                let ty = self.ty(ty)?;
                self.bind_as(name, local);
                Stmt::Reg {
                    name: name.clone(),
                    ty,
                    reset: reset.as_ref().map(|v| self.expr(v)).transpose()?,
                }
            }
            Stmt::Let { name, ty, value } => {
                let ty = self.optional_type(ty)?;
                let value = self.expr(value)?;
                // Rebinding a tuple-pattern leaf in a block creates another
                // lexical value, including when the new let has a simple name.
                if self.locals.get(name) != Some(&Local::Lexical) {
                    self.bind_as(name, local);
                }
                Stmt::Let {
                    name: name.clone(),
                    ty,
                    value,
                }
            }
            Stmt::LetPattern(binding) => Stmt::LetPattern(self.binding(binding)?),
            Stmt::Write { name, value } => Stmt::Write {
                name: name.clone(),
                value: self.expr(value)?,
            },
            Stmt::Connect {
                instance,
                port,
                value,
            } => Stmt::Connect {
                instance: instance.clone(),
                port: port.clone(),
                value: self.expr(value)?,
            },
            Stmt::If { condition, body } => {
                if self.guard_depth >= MAX_SYNTAX_DEPTH {
                    return Err(format!("BX import guard exceeds {MAX_SYNTAX_DEPTH} levels"));
                }
                self.guard_depth += 1;
                let condition = self.expr(condition)?;
                let body = self.nested(|scope| {
                    body.iter()
                        .map(|stmt| scope.statement(stmt, local))
                        .collect()
                })?;
                self.guard_depth -= 1;
                Stmt::If { condition, body }
            }
            Stmt::Inst { name, block } => Stmt::Inst {
                name: name.clone(),
                block: self.callee(block, CallSite::Structural)?,
            },
        })
    }

    fn iterable(&mut self, iterable: &ForIterable) -> Result<ForIterable, String> {
        Ok(match iterable {
            ForIterable::Range {
                start,
                limit,
                inclusive,
            } => ForIterable::Range {
                start: self.expr(start)?,
                limit: self.expr(limit)?,
                inclusive: *inclusive,
            },
            ForIterable::Array(array) => ForIterable::Array(self.expr(array)?),
        })
    }

    fn loop_annotation(
        &mut self,
        annotation: &Option<(Type, Type)>,
    ) -> Result<Option<(Type, Type)>, String> {
        annotation
            .as_ref()
            .map(|(a, b)| Ok((self.ty(a)?, self.ty(b)?)))
            .transpose()
    }

    fn expressions(&mut self, exprs: &[Expr]) -> Result<Vec<Expr>, String> {
        exprs.iter().map(|expr| self.expr(expr)).collect()
    }

    /// Rewrite free references without changing the binders or the shape of the
    /// expression.
    fn expr(&mut self, expr: &Expr) -> Result<Expr, String> {
        // The parser bounds expression height independently of its surrounding
        // statement guards. Both counters stay active across unrolled bodies.
        if self.expression_depth > MAX_SYNTAX_DEPTH {
            return Err(format!(
                "BX import exceeds the syntax nesting limit of {MAX_SYNTAX_DEPTH}"
            ));
        }
        self.expression_depth += 1;
        let result = self.expr_kind(expr);
        self.expression_depth -= 1;
        result
    }

    fn expr_kind(&mut self, expr: &Expr) -> Result<Expr, String> {
        Ok(match expr {
            Expr::Scope { bindings, result } => self.nested(|scope| {
                Ok(Expr::Scope {
                    bindings: bindings
                        .iter()
                        .map(|b| scope.binding(b))
                        .collect::<Result<_, _>>()?,
                    result: Box::new(scope.expr(result)?),
                })
            })?,
            Expr::Name(name) => {
                let reference = self.reference(name, self.value_at)?;
                if reference.contains("::") {
                    // Qualification can remove the distinction between `_::A`
                    // and `module::_::A`. Check the original value path first.
                    crate::parser::validate_enum_member_path(name)?;
                }
                Expr::Name(reference)
            }
            Expr::Bool(value) => Expr::Bool(*value),
            Expr::Literal { ty, text } => Expr::Literal {
                ty: self.optional_type(ty)?,
                text: text.clone(),
            },
            Expr::Zero(ty) => Expr::Zero(self.ty(ty)?),
            Expr::Tuple(fields) => Expr::Tuple(self.expressions(fields)?),
            Expr::Array(items) => Expr::Array(self.expressions(items)?),
            Expr::Struct { name, fields } | Expr::AppliedStruct { name, fields, .. } => {
                let name = self.reference(name, self.at)?;
                let fields = fields
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), self.expr(value)?)))
                    .collect::<Result<_, String>>()?;
                if let Expr::AppliedStruct { args, .. } = expr {
                    Expr::AppliedStruct {
                        name,
                        args: args.clone(),
                        fields,
                    }
                } else {
                    Expr::Struct { name, fields }
                }
            }
            Expr::Field { value, name } => Expr::Field {
                value: Box::new(self.expr(value)?),
                name: name.clone(),
            },
            Expr::Index { value, index } => Expr::Index {
                value: Box::new(self.expr(value)?),
                index: Box::new(self.expr(index)?),
            },
            Expr::WidthSlice {
                value,
                start,
                width,
            } => Expr::WidthSlice {
                value: Box::new(self.expr(value)?),
                start: Box::new(self.expr(start)?),
                width: match width {
                    WidthSliceType::Concrete(ty) => WidthSliceType::Concrete(self.ty(ty)?),
                    WidthSliceType::Bits { signed, bit_count } => WidthSliceType::Bits {
                        signed: Box::new(self.expr(signed)?),
                        bit_count: Box::new(self.expr(bit_count)?),
                    },
                },
            },
            Expr::Call { name, args } | Expr::MapBeforeUse { name, args, .. } => {
                let callee = self.callee(name, CallSite::Direct)?;
                let builtin_args = match expr {
                    Expr::MapBeforeUse { builtin_args, .. } => builtin_args.as_deref(),
                    _ => Some(args.as_slice()),
                };
                if callee == BUILTIN_MAP
                    && let Some(args) = builtin_args
                {
                    // A block may reference a function imported later in this
                    // file. Resolve the callback at the same position as a
                    // direct call.
                    let args = args
                        .iter()
                        .enumerate()
                        .map(|(index, arg)| match (index, arg) {
                            (1, Expr::Name(callback)) => {
                                Ok(Expr::Name(self.map_callback(callback)?))
                            }
                            _ => self.expr(arg),
                        })
                        .collect::<Result<_, String>>()?;
                    return Ok(Expr::Call { name: callee, args });
                }
                let later_dslx_map = name == "map"
                    && matches!(self.names.uses.get(name), Some(Binding::Dslx(alias)) if alias == &callee)
                    && self
                        .value_at
                        .zip(self.names.positions.get(name))
                        .is_some_and(|(at, &pos)| at < pos);
                // A later DSLX use named map can denote a function or an enum.
                // A callback visible only in callable scope requires the
                // builtin.
                if later_dslx_map && let Some([array, Expr::Name(callback)]) = builtin_args {
                    let callable = self.map_callback(callback)?;
                    if self.reference(callback, self.value_at).as_ref() != Ok(&callable) {
                        return Ok(Expr::NamedMap {
                            target: callee,
                            iterable: Box::new(ForIterable::Array(self.expr(array)?)),
                            callback: callable,
                            parametrics: Vec::new(),
                        });
                    }
                }
                let args = self.expressions(args)?;
                if matches!(expr, Expr::MapBeforeUse { .. })
                    && (later_dslx_map || callee == BUILTIN_MAP)
                {
                    Expr::MapBeforeUse {
                        name: callee,
                        args,
                        builtin_args: builtin_args
                            .map(|args| self.expressions(args))
                            .transpose()?,
                    }
                } else {
                    Expr::Call { name: callee, args }
                }
            }
            Expr::ParametricCall {
                name,
                parametrics,
                args,
            } => Expr::ParametricCall {
                name: self.callee(
                    name,
                    if parametrics.is_empty() {
                        // AST callers may represent a plain structural call
                        // this way.
                        CallSite::Structural
                    } else {
                        CallSite::Parametric
                    },
                )?,
                parametrics: parametrics
                    .iter()
                    .enumerate()
                    .map(
                        |(i, actual)| match (self.names.parametrics(name).get(i), actual) {
                            (Some(ParametricKind::Callback), Expr::Name(callback)) => {
                                Ok(Expr::Name(self.reference(callback, self.at)?))
                            }
                            _ => self.expr(actual),
                        },
                    )
                    .collect::<Result<_, _>>()?,
                args: self.expressions(args)?,
            },
            Expr::Map(map) => Expr::Map(Box::new(self.map(map)?)),
            Expr::NamedMap {
                target,
                iterable,
                callback,
                parametrics,
            } => Expr::NamedMap {
                target: self.reference(target, self.builtin_at(target))?,
                iterable: Box::new(self.iterable(iterable)?),
                callback: self.map_callback(callback)?,
                parametrics: self.expressions(parametrics)?,
            },
            Expr::Unary { op, value } => Expr::Unary {
                op: *op,
                value: Box::new(self.expr(value)?),
            },
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: Box::new(self.expr(left)?),
                right: Box::new(self.expr(right)?),
            },
            Expr::Cast { value, ty } => Expr::Cast {
                value: Box::new(self.expr(value)?),
                ty: self.ty(ty)?,
            },
            Expr::If {
                condition,
                then_value,
                else_value,
            } => Expr::If {
                condition: Box::new(self.expr(condition)?),
                then_value: Box::new(self.expr(then_value)?),
                else_value: Box::new(self.expr(else_value)?),
            },
            Expr::Match { subject, arms } => Expr::Match {
                subject: Box::new(self.expr(subject)?),
                arms: arms
                    .iter()
                    .map(|arm| {
                        let patterns = arm
                            .patterns
                            .iter()
                            .map(|p| {
                                Ok(match p {
                                    MatchPattern::Wildcard => MatchPattern::Wildcard,
                                    MatchPattern::Literal(value) => {
                                        MatchPattern::Literal(self.expr(value)?)
                                    }
                                    MatchPattern::Range {
                                        start,
                                        end,
                                        inclusive,
                                    } => MatchPattern::Range {
                                        start: self.expr(start)?,
                                        end: self.expr(end)?,
                                        inclusive: *inclusive,
                                    },
                                })
                            })
                            .collect::<Result<_, String>>()?;
                        Ok(MatchArm {
                            patterns,
                            value: self.expr(&arm.value)?,
                        })
                    })
                    .collect::<Result<_, String>>()?,
            },
            Expr::For(fold) => Expr::For(Box::new(self.fold(fold)?)),
            Expr::UnrollFor(fold) => Expr::UnrollFor(Box::new(self.unrolled_fold(fold)?)),
        })
    }

    /// The map element is visible in the result annotation and body.
    fn map(&mut self, map: &MapExpr) -> Result<MapExpr, String> {
        let target = self.reference(&map.target, self.builtin_at(&map.target))?;
        let iterable = self.iterable(&map.iterable)?;
        let parameter_type = self.optional_type(&map.parameter_type)?;
        let outer_hardware = self.hardware_callees.take();
        let (result_type, body) = self.nested(|scope| {
            scope.bind(&map.parameter);
            Ok((
                scope.optional_type(&map.result_type)?,
                scope.expr(&map.body)?,
            ))
        })?;
        self.hardware_callees = outer_hardware;
        Ok(MapExpr {
            target,
            iterable,
            parameter: map.parameter.clone(),
            parameter_type,
            result_type,
            body,
        })
    }

    /// Only the loop body sees its element and accumulator binders.
    fn fold(&mut self, fold: &ForLoop) -> Result<ForLoop, String> {
        let annotation = self.loop_annotation(&fold.annotation)?;
        let iterable = self.iterable(&fold.iterable)?;
        let seed = self.expr(&fold.seed)?;
        let element = copy_pattern(&fold.element, 0)?;
        let accumulator = copy_pattern(&fold.accumulator, 0)?;
        let body = self.nested(|scope| {
            scope.bind_pattern(&element);
            scope.bind_pattern(&accumulator);
            scope.expr(&fold.body)
        })?;
        Ok(ForLoop {
            element,
            accumulator,
            annotation,
            iterable,
            body,
            seed,
        })
    }

    /// Resolve an unrolled body's lets in iteration order before its result.
    fn unrolled_fold(&mut self, fold: &UnrollFor) -> Result<UnrollFor, String> {
        let annotation = self.loop_annotation(&fold.annotation)?;
        let start = self.expr(&fold.start)?;
        let limit = self.expr(&fold.limit)?;
        let seed = self.expr(&fold.seed)?;
        let (body, result) = self.nested(|scope| {
            scope.bind(&fold.index);
            scope.bind(&fold.accumulator);
            Ok((
                fold.body
                    .iter()
                    .map(|stmt| scope.statement(stmt, Local::Lexical))
                    .collect::<Result<_, _>>()?,
                scope.expr(&fold.result)?,
            ))
        })?;
        Ok(UnrollFor {
            index: fold.index.clone(),
            accumulator: fold.accumulator.clone(),
            annotation,
            start,
            limit,
            body,
            result,
            seed,
        })
    }
}

/// Clone one let or counted-loop pattern within the syntax nesting limit.
fn copy_pattern(pattern: &BindingPattern, depth: usize) -> Result<BindingPattern, String> {
    if depth > MAX_SYNTAX_DEPTH {
        return Err(format!(
            "BX import pattern exceeds {MAX_SYNTAX_DEPTH} levels"
        ));
    }
    Ok(match pattern {
        BindingPattern::Name(name) => BindingPattern::Name(name.clone()),
        BindingPattern::Rest => BindingPattern::Rest,
        BindingPattern::Tuple(fields) => BindingPattern::Tuple(
            fields
                .iter()
                .map(|field| copy_pattern(field, depth + 1))
                .collect::<Result<_, _>>()?,
        ),
    })
}
