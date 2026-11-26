// SPDX-License-Identifier: Apache-2.0

//! Check numeric enum initializers in source order and retain their nominal
//! types.

use std::collections::{BTreeMap, BTreeSet};

use petgraph::{algo::kosaraju_scc, graphmap::DiGraphMap};
use xlsynth_pir::IrBits;

use super::{
    BodyChecker, CheckedEnum, Declarations, StructTable, bits, dependency_order, type_dependencies,
};
use crate::ast::{BitTypeAttribute, EnumDef, Expr, Module, ScopedSourceNodes, SourceNode, Type};

impl StructTable<'_> {
    /// Only BX declarations have unqualified keys. Imported enum references
    /// must resolve through the current module's use bindings.
    pub(super) fn local_enum(&self, name: &str) -> Option<&CheckedEnum> {
        if name.contains("::") {
            return None;
        }
        self.enums.get(name)
    }

    /// A BX alias can qualify a member of either a local or an imported enum.
    pub(super) fn enum_subject(
        &self,
        prefix: &str,
        decls: &Declarations<'_>,
    ) -> Option<&CheckedEnum> {
        match decls.type_aliases.get(prefix).map(|alias| &alias.ty) {
            Some(Type::Enum { name, .. }) => self.enums.get(name),
            Some(_) => None,
            None => self.local_enum(prefix),
        }
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
        zero_binding: impl Fn(&str) -> String,
    ) -> Result<Vec<(usize, usize)>, String> {
        let type_positions: BTreeMap<_, _> = module
            .enums
            .iter()
            .map(|def| (def.name.as_str(), (position(&def.name), "enum")))
            .chain(
                module
                    .type_aliases
                    .iter()
                    .map(|def| (def.name.as_str(), (position(&def.name), "type alias"))),
            )
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
            let mut expression_ready_at = 0;
            let declared_at = position(&def.name);
            let types = def
                .value_parametrics
                .iter()
                .map(|formal| &formal.ty)
                .chain(def.fields.iter().map(|field| &field.ty));
            let mut pending = ScopedSourceNodes::default();
            let mut zero = None;
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
                            Type::TypeApplied { .. } | Type::MixedApplied { .. } => true,
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
            let mut type_params = def.type_params.as_slice();
            let mut optional = false;
            while let Some(node) = pending.next().or_else(|| {
                let (node, visible_types, preference) = defaults.pop()?;
                in_declaration = false;
                type_params = visible_types;
                optional = preference;
                pending.push(node);
                pending.next()
            }) {
                // A forward-used layout is ready after the values and helpers
                // referenced by its dimensions, even if the struct itself is
                // declared later. Name resolution still uses the struct scope.
                if in_declaration && let SourceNode::Expression(expr) = node {
                    let mut record = |name: &str| {
                        if !pending.is_local(name) {
                            // A linked declaration can contain "::" in its
                            // module path. Only separators in the member
                            // identify a source qualifier, such as an enum or
                            // DSLX use.
                            let member = crate::bx_import::library_item(name)
                                .map_or(name, |(_, member)| member);
                            let binding = member
                                .split_once("::")
                                .map_or(name, |(_, tail)| &name[..name.len() - tail.len() - 2]);
                            let at = position(binding);
                            if at <= declared_at {
                                expression_ready_at = expression_ready_at.max(at);
                            }
                        }
                    };
                    match expr {
                        Expr::Name(name)
                        | Expr::Call { name, .. }
                        | Expr::ParametricCall { name, .. }
                        | Expr::MapBeforeUse { name, .. } => record(name),
                        Expr::Map(map) => record(&map.target),
                        Expr::NamedMap {
                            target, callback, ..
                        } => {
                            record(target);
                            record(callback);
                        }
                        Expr::Zero(_) if !pending.is_local("zero!") => {
                            record(zero.get_or_insert_with(|| zero_binding(&def.name)));
                        }
                        _ => { /* Other expressions only reference names in their children. */ }
                    }
                }
                let constructor = matches!(node, SourceNode::Expression(_));
                let application = match node {
                    SourceNode::Annotation(Type::ParametricStruct { name, args })
                    | SourceNode::Expression(Expr::ParametricStruct { name, args, .. }) => {
                        Some((name, args.len()))
                    }
                    SourceNode::Annotation(Type::MixedApplied { name, args })
                    | SourceNode::Expression(Expr::MixedAppliedStruct { name, args, .. }) => {
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
                if let Some((name, supplied)) = application
                    && !type_params.contains(name)
                    && !pending.is_local(name)
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
                        let supplied = formals.partition_point(|f| f.index < supplied);
                        if supplied < *first {
                            for (value_index, formal) in
                                formals.iter().enumerate().take(*first).skip(supplied)
                            {
                                if let Some(expr) = &formal.default {
                                    // Earlier type formals occupy the
                                    // declaration slots before this default
                                    // that are not taken by values.
                                    let visible_types =
                                        &template.type_params[..formal.index - value_index];
                                    defaults.push((
                                        SourceNode::Expression(expr),
                                        visible_types,
                                        preference,
                                    ));
                                }
                            }
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
                expression_ready_at
            } else {
                declared_at
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
                if let Some(&(type_at, kind)) = type_positions.get(dep) {
                    // An AST without a cross-kind declaration order can index
                    // the struct as soon as its field types are available.
                    if !module.declaration_order.is_empty() && type_at > position(&def.name) {
                        return Err(format!(
                            "{kind} '{dep}' is declared after struct '{}'",
                            def.name
                        ));
                    }
                    at = at.max(type_at);
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

    /// A local alias fixes its enum identity at the declaration site.
    fn local_enum_alias(&self, prefix: &str) -> Result<Option<&CheckedEnum>, String> {
        match self.local_type_aliases.get(prefix).map(|alias| &alias.ty) {
            Some(Type::Enum { name, .. }) => Ok(self.structs.enums.get(name)),
            Some(_) => Err(format!("'{prefix}' is not a numeric enum")),
            None => Ok(None),
        }
    }

    /// Look up a fixed-width keyword or a BX alias whose checked type is bits.
    fn local_bit_attribute_type(&self, prefix: &str) -> Option<Type> {
        if let Some(width) = crate::parser::fixed_width(prefix) {
            return Some(Type::Bits {
                signed: prefix.starts_with('s'),
                width,
            });
        }
        let ty = self
            .local_type_aliases
            .get(prefix)
            .map(|alias| &alias.ty)
            .or_else(|| self.decls.type_aliases.get(prefix).map(|alias| &alias.ty))?;
        bits(ty).map(|_| ty.clone())
    }

    /// Materialize a scalar attribute at the qualifier's concrete width and
    /// signedness.
    fn bit_type_attribute(
        &self,
        name: &str,
        ty: Type,
        attribute: BitTypeAttribute,
    ) -> Result<(Type, IrBits), String> {
        crate::parser::validate_dslx_value_path(name)?;
        let (signed, width) =
            bits(&ty).ok_or_else(|| format!("'{name}' requires a bit type, got {ty}"))?;
        self.structs.validate_type(&ty)?;
        let value = match (attribute, signed) {
            (BitTypeAttribute::Min, true) => IrBits::signed_min_value(width),
            (BitTypeAttribute::Max, true) => IrBits::signed_max_value(width),
            (BitTypeAttribute::Max, false) => IrBits::all_ones(width),
            (BitTypeAttribute::Min, false) | (BitTypeAttribute::Zero, _) => IrBits::zero(width),
        };
        self.charge_reference_copy(super::bits_copy_work(&value))?;
        Ok((ty, value))
    }

    /// Resolve a type member name even in a skipped loop without converting
    /// any imported functions or forcing evaluation of the enclosing
    /// expression.
    pub(super) fn is_type_member_name(&self, name: &str) -> Result<bool, String> {
        let Some((prefix, member)) = name.rsplit_once("::") else {
            return Ok(false);
        };
        let attribute = BitTypeAttribute::parse(member);
        if attribute.is_some() && self.local_bit_attribute_type(prefix).is_some() {
            crate::parser::validate_dslx_value_path(name)?;
            if !self.local_type_aliases.contains_key(prefix) {
                self.decls.check_type_name_visibility(
                    prefix,
                    self.function_position.or(self.block_position),
                )?;
            }
            return Ok(true);
        }
        let local = self.local_enum_alias(prefix)?;
        if let Some(def) = local.or_else(|| self.structs.enum_subject(prefix, self.decls)) {
            crate::parser::validate_dslx_value_path(name)?;
            if local.is_none() {
                self.decls.check_type_name_visibility(
                    prefix,
                    self.function_position.or(self.block_position),
                )?;
            }
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
            if attribute.is_some()
                && self.resolver.has_public_type(&subject, &ty)
                && !self.resolver.is_enum_type(&subject, &ty)
            {
                return Ok(true);
            }
            self.resolver
                .check_enum_member_name(&subject, &ty, member)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Resolve a qualified value as a bit attribute or a declared enum member.
    pub(super) fn type_member(&mut self, name: &str) -> Result<Option<(Type, IrBits)>, String> {
        let Some((prefix, member)) = name.rsplit_once("::") else {
            return Ok(None);
        };
        let head = prefix.split("::").next().unwrap_or(prefix);
        if !self.local_type_aliases.contains_key(prefix) {
            if self.has_local_name(head) || self.callback_index(head).is_some() {
                return Err(format!("'{head}' names a local value here"));
            }
            self.reject_type_parameter_value(head)?;
            self.decls.check_type_name_visibility(
                prefix,
                self.function_position.or(self.block_position),
            )?;
        }
        let attribute = BitTypeAttribute::parse(member);
        if let Some(attribute) = attribute
            && let Some(ty) = self.local_bit_attribute_type(prefix)
        {
            return self.bit_type_attribute(name, ty, attribute).map(Some);
        }
        let local = self.local_enum_alias(prefix)?;
        let canonical;
        let def = if let Some(def) = local.or_else(|| self.structs.enum_subject(prefix, self.decls))
        {
            def
        } else if let Some((subject, member_type)) = self.resolver.lookup_enum_prefix(prefix)? {
            let ty = self.resolver.resolve_named(&subject, &member_type)?;
            let Type::Enum { name, .. } = ty else {
                if let Some(attribute) = attribute
                    && self.resolver.has_public_type(&subject, &member_type)
                {
                    return self.bit_type_attribute(name, ty, attribute).map(Some);
                }
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
