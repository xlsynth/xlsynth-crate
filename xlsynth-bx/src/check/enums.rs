// SPDX-License-Identifier: Apache-2.0

//! Check numeric enum initializers in source order and retain their nominal
//! types.

use std::collections::{BTreeMap, BTreeSet};

use petgraph::{algo::kosaraju_scc, graphmap::DiGraphMap};
use xlsynth_pir::IrBits;

use super::{BodyChecker, CheckedEnum, StructTable, bits, dependency_order, type_dependencies};
use crate::ast::{EnumDef, Expr, Module, SourceNode, Type};

impl StructTable<'_> {
    /// Only BX declarations have unqualified keys. Imported enum references
    /// must resolve through the current module's use bindings.
    pub(super) fn local_enum(&self, name: &str) -> Option<&CheckedEnum> {
        if name.contains("::") {
            return None;
        }
        self.enums.get(name)
    }

    /// Check that zero initialization yields a declared value for each enum.
    pub(super) fn check_zero_members(&self, ty: &Type) -> Result<(), String> {
        self.check_fill_members(ty, /* ones= */ false)
    }

    /// Check that a bit-filled enum is declared, including inside aggregates.
    /// Visit each nominal struct at most once.
    pub(super) fn check_fill_members(&self, ty: &Type, ones: bool) -> Result<(), String> {
        if self.enums.is_empty() {
            return Ok(());
        }
        let mut seen = BTreeSet::new();
        let mut pending = vec![ty];
        while let Some(ty) = pending.pop() {
            match ty {
                Type::Enum { name, width, .. } => {
                    let fill = if ones {
                        IrBits::all_ones(*width)
                    } else {
                        IrBits::zero(*width)
                    };
                    if !self.enums.get(name).is_some_and(|e| e.contains(&fill)) {
                        let member = if ones { "all-ones" } else { "zero" };
                        return Err(format!("enum '{name}' has no {member} member"));
                    }
                }
                Type::Tuple(fields) => pending.extend(fields),
                Type::Array { element, size } if *size != 0 => pending.push(element),
                Type::Named(name) if seen.insert(name) => {
                    pending.extend(self.definition(name)?.fields.iter().map(|f| &f.ty));
                }
                _ => { /* Already visited structs and primitive leaves need no check. */ }
            }
        }
        Ok(())
    }

    /// Schedule layouts after their nested structs, enum representations,
    /// and any module constants used as dimensions.
    pub(super) fn struct_schedule(
        module: &Module,
        position: impl Fn(&str) -> usize,
    ) -> Result<Vec<(usize, usize)>, String> {
        let enum_positions: BTreeMap<_, _> = module
            .enums
            .iter()
            .map(|def| (def.name.as_str(), position(&def.name)))
            .collect();
        let constant_positions: BTreeMap<_, _> = module
            .constants
            .iter()
            .map(|c| (c.name.as_str(), position(&c.name)))
            .collect();
        let mut graph = Vec::new();
        let mut preferences = Vec::new();
        let mut ready = Vec::new();
        let local_structs: BTreeMap<_, _> = module
            .structs
            .iter()
            .map(|def| (def.name.as_str(), def))
            .collect();
        for def in &module.structs {
            let mut deps = Vec::new();
            let mut preferred = Vec::new();
            let mut dimensions = Vec::new();
            let mut has_symbolic_application = false;
            let types = def
                .value_parametrics
                .iter()
                .map(|formal| &formal.ty)
                .chain(def.fields.iter().map(|field| &field.ty));
            let mut pending = Vec::new();
            for ty in types {
                pending.push(SourceNode::Annotation(ty));
                type_dependencies(
                    ty,
                    |ty, _| {
                        // A literal with a concrete annotation can be resolved
                        // as soon as its template is available. uN[WIDTH]:8
                        // needs its scope.
                        has_symbolic_application |= match ty {
                            Type::ParametricStruct { args, .. } => !args.iter().all(|arg| {
                                matches!(
                                    arg,
                                    Expr::Bool(_)
                                        | Expr::Literal {
                                            ty: None | Some(Type::Bits { .. } | Type::Bool),
                                            ..
                                        }
                                )
                            }),
                            Type::TypeApplied { .. } => true,
                            _ => false,
                        };
                    },
                    |expr| {
                        if let Expr::Name(name) = expr {
                            dimensions.push(name.as_str());
                        }
                    },
                );
            }
            // Visit each selected default at most once for this declaration.
            // A fully explicit application has no dependency on its defaults.
            let mut default_prefixes = BTreeMap::new();
            let mut defaults = Vec::new();
            let mut in_declaration = true;
            let mut optional = false;
            while let Some(node) = pending.pop().or_else(|| {
                // Defaults use their own value template's declaration scope.
                in_declaration = false;
                defaults.pop().map(|(node, preference)| {
                    optional = preference;
                    node
                })
            }) {
                let constructor = matches!(node, SourceNode::Expression(_));
                let application = match node {
                    SourceNode::Annotation(Type::ParametricStruct { name, args })
                    | SourceNode::Expression(Expr::ParametricStruct { name, args, .. }) => {
                        Some((name, args.len()))
                    }
                    SourceNode::Annotation(Type::TypeApplied { name, args })
                    | SourceNode::Expression(Expr::TypeAppliedStruct { name, args, .. }) => {
                        Some((name, args.len()))
                    }
                    SourceNode::Annotation(Type::Named(name) | Type::Enum { name, .. })
                    | SourceNode::Expression(Expr::Struct { name, .. }) => Some((name, 0)),
                    SourceNode::Annotation(Type::Applied { name, args })
                    | SourceNode::Expression(Expr::AppliedStruct { name, args, .. }) => {
                        Some((name, args.len()))
                    }
                    _ => None,
                };
                node.push_children(&mut pending);
                if let Some((name, supplied)) = application
                    && (!in_declaration || !def.type_params.contains(name))
                {
                    if optional {
                        preferred.push(name.as_str());
                    } else {
                        deps.push(name.as_str());
                    }
                    if let Some(template) = local_structs.get(name.as_str()) {
                        let formals = &template.value_parametrics;
                        // Constructor context or a splat can supply an
                        // omitted argument. Its default is only a preference.
                        let preference = optional || constructor;
                        let first = default_prefixes
                            .entry((name.as_str(), preference))
                            .or_insert(formals.len());
                        if supplied < *first {
                            defaults.extend(
                                formals[supplied..*first]
                                    .iter()
                                    .filter_map(|formal| formal.default.as_ref())
                                    .map(|expr| (SourceNode::Expression(expr), preference)),
                            );
                            *first = supplied;
                        }
                    }
                }
            }
            let parametric_names: BTreeSet<_> = def
                .value_parametrics
                .iter()
                .map(|binding| binding.name.as_str())
                .collect();
            let mut at = if parametric_names.is_empty() && !has_symbolic_application {
                0
            } else {
                position(&def.name)
            };
            for dimension in dimensions {
                if parametric_names.contains(dimension) {
                    continue;
                }
                if let Some(&constant_at) = constant_positions.get(dimension) {
                    if constant_at > position(&def.name) {
                        return Err(format!(
                            "module constant '{dimension}' is declared after struct '{}'",
                            def.name
                        ));
                    }
                    at = at.max(constant_at);
                }
            }
            for dep in &deps {
                if local_structs
                    .get(dep)
                    .is_some_and(|def| def.is_parametric())
                    && position(dep) > position(&def.name)
                {
                    return Err(format!(
                        "struct template '{dep}' is declared after struct '{}'",
                        def.name
                    ));
                }
                if let Some(&enum_at) = enum_positions.get(dep) {
                    if enum_at > position(&def.name) {
                        return Err(format!(
                            "enum '{dep}' is declared after struct '{}'",
                            def.name
                        ));
                    }
                    at = at.max(enum_at);
                }
            }
            ready.push(at);
            deps.retain(|name| local_structs.contains_key(name));
            preferred.retain(|name| local_structs.contains_key(name));
            graph.push((def.name.as_str(), deps));
            preferences.push(preferred);
        }
        let mut ranks = BTreeMap::new();
        let mut scheduled = Vec::with_capacity(graph.len());
        for index in layout_order(&graph, &preferences)? {
            let (name, deps) = &graph[index];
            let at = deps.iter().fold(ready[index], |at, dep| at.max(ranks[dep]));
            let at = preferences[index]
                .iter()
                .filter_map(|dep| ranks.get(dep))
                .fold(at, |at, &rank| at.max(rank));
            ranks.insert(*name, at);
            scheduled.push((at, index));
        }
        Ok(scheduled)
    }
}

/// Honor constructor-default preferences between components. Within a cycle
/// involving a preference, only required dependencies determine the order.
fn layout_order(
    graph: &[(&str, Vec<&str>)],
    preferences: &[Vec<&str>],
) -> Result<Vec<usize>, String> {
    let required = dependency_order("struct", graph)?;
    if preferences.iter().all(Vec::is_empty) {
        return Ok(required);
    }
    let indices: BTreeMap<_, _> = graph
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i))
        .collect();
    let mut edges = DiGraphMap::<usize, ()>::new();
    let mut required_rank = vec![0; graph.len()];
    for (rank, &index) in required.iter().enumerate() {
        required_rank[index] = rank;
        edges.add_node(index);
        for dep in graph[index].1.iter().chain(&preferences[index]) {
            edges.add_edge(index, indices[dep], ());
        }
    }
    let mut order = Vec::with_capacity(graph.len());
    for mut component in kosaraju_scc(&edges) {
        component.sort_unstable_by_key(|&i| required_rank[i]);
        order.extend(component);
    }
    Ok(order)
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Infer a common underlying bit type and evaluate every discriminant in
    /// the scope immediately preceding the declaration.
    pub(super) fn check_enum(&mut self, def: &'m EnumDef) -> Result<CheckedEnum, String> {
        self.warning_owner = Some(&def.name);
        self.zero_binding = self.decls.zero_binding(&def.name);
        self.function_position = self.decls.positions.get(def.name.as_str()).copied();
        self.enable_constant_functions();
        let underlying = def
            .underlying
            .as_ref()
            .map(|ty| self.source_type(ty))
            .transpose()?;
        if let Some(ty) = &underlying
            && bits(ty).is_none()
        {
            return Err(format!("enum representation must be a bit type, got {ty}"));
        }
        for member in &def.members {
            if !(crate::parser::is_use_identifier(&member.name) || member.name == "self") {
                return Err(format!("invalid enum member name '{}'", member.name));
            }
        }
        let items: Vec<_> = def.members.iter().map(|m| &m.value).collect();
        CheckedEnum::validate_size(
            underlying.as_ref().and_then(bits).map_or(0, |(_, w)| w),
            items.len(),
        )?;
        let (signed, width, values) = if items.is_empty() {
            let (signed, width) = underlying
                .as_ref()
                .and_then(bits)
                .ok_or("a numeric enum with no members requires an explicit underlying bit type")?;
            (signed, width, Vec::new())
        } else {
            self.with_scoped_results(&items, |checker, items| {
                let (index, anchor) = checker.result_anchor(items, underlying.as_ref())?;
                let (signed, width) = bits(&anchor.ty).ok_or_else(|| {
                    format!("enum representation must be a bit type, got {}", anchor.ty)
                })?;
                CheckedEnum::validate_size(width, items.len())?;
                Ok((signed, width, checker.check_siblings(items, index, anchor)?))
            })?
        };
        let mut members = BTreeMap::new();
        for (member, expr) in def.members.iter().zip(values) {
            let value = self
                .checked_constant(&expr)?
                .ok_or_else(|| format!("member '{}' requires a compile-time value", member.name))?;
            if members
                .insert(
                    member.name.clone(),
                    value.to_bits().map_err(|e| e.to_string())?,
                )
                .is_some()
            {
                return Err(format!("duplicate enum member '{}'", member.name));
            }
        }
        Ok(CheckedEnum::new(
            def.name.clone(),
            def.public,
            signed,
            width,
            members,
        ))
    }

    /// Resolve an enum member name even in a skipped loop without converting
    /// any imported functions or forcing evaluation of the enclosing
    /// expression.
    pub(super) fn is_enum_member_name(&self, name: &str) -> Result<bool, String> {
        let Some((prefix, member)) = name.rsplit_once("::") else {
            return Ok(false);
        };
        if let Some(def) = self.structs.local_enum(prefix) {
            crate::parser::validate_dslx_value_path(name)?;
            self.decls.check_type_name_visibility(
                prefix,
                self.function_position.or(self.block_position),
            )?;
            if def.members.contains_key(member) {
                return Ok(true);
            }
            return Err(format!("enum '{prefix}' has no member '{member}'"));
        }
        if let Some((subject, ty)) = self.resolver.lookup_enum_prefix(prefix)? {
            crate::parser::validate_dslx_value_path(name)?;
            self.decls.check_type_name_visibility(
                prefix,
                self.function_position.or(self.block_position),
            )?;
            self.resolver
                .check_enum_member_name(&subject, &ty, member)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// A member is a compile-time value with the nominal type of its
    /// declaration.
    pub(super) fn enum_member(&mut self, name: &str) -> Result<Option<(Type, IrBits)>, String> {
        let Some((prefix, member)) = name.rsplit_once("::") else {
            return Ok(None);
        };
        let head = prefix.split("::").next().unwrap_or(prefix);
        if self.names.contains(head) || self.callback_index(head).is_some() {
            return Err(format!("'{head}' names a local value here"));
        }
        self.reject_type_parameter_value(head)?;
        self.decls
            .check_type_name_visibility(prefix, self.function_position.or(self.block_position))?;
        let canonical;
        let def = if let Some(def) = self.structs.local_enum(prefix) {
            def
        } else if let Some((subject, ty)) = self.resolver.lookup_enum_prefix(prefix)? {
            let Type::Enum { name, .. } = self.resolver.resolve_named(&subject, &ty)? else {
                return Err(format!("'{prefix}' is not a numeric enum"));
            };
            canonical = name;
            self.structs.update_imports(self.resolver)?;
            self.structs
                .enums
                .get(&canonical)
                .ok_or_else(|| format!("unknown enum '{prefix}'"))?
        } else {
            return Ok(None);
        };
        crate::parser::validate_dslx_value_path(name)?;
        let value = def
            .members
            .get(member)
            .ok_or_else(|| format!("enum '{prefix}' has no member '{member}'"))?;
        self.charge_reference_copy(super::bits_copy_work(value))?;
        Ok(Some((def.ty(), value.clone())))
    }
}
