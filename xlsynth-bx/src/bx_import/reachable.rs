// SPDX-License-Identifier: Apache-2.0

//! Select the checked library definitions needed by the root BX module.

use std::collections::BTreeSet;

use super::library_item;
use crate::typed::{
    CheckedBlock, CheckedModule, CheckedStmt, ScalarPattern, TypedExpr, TypedExprKind as K,
    TypedIterable,
};

impl CheckedModule {
    /// Retain the root's definitions and the library bodies and externs they
    /// call. Return the DSLX specialization IDs needed by those bodies.
    pub(crate) fn retain_library_dependencies(&mut self) -> BTreeSet<usize> {
        let roots = self
            .blocks
            .iter()
            .map(|b| b.name.as_str())
            .chain(self.functions.iter().map(|f| f.name.as_str()))
            .filter(|&name| {
                library_item(name).is_none()
                    && !self.specialized_blocks.contains(name)
                    && !self.specialized_functions.contains(name)
            });
        let mut used = Dependencies {
            named: roots.map(str::to_owned).collect(),
            dslx: BTreeSet::new(),
        };
        // Both collections are already in dependency order. Visit each
        // caller before its children, then visit functions called by blocks.
        for block in self.blocks.iter().rev() {
            if used.named.contains(&block.name) {
                used.block(block);
            }
        }
        for function in self.functions.iter().rev() {
            if used.named.contains(&function.name) {
                let expressions = std::iter::once(&function.output)
                    .chain(function.bindings.iter().map(|b| &b.value))
                    .collect();
                used.expressions(expressions);
            }
        }
        self.blocks.retain(|b| used.named.contains(&b.name));
        self.functions.retain(|f| used.named.contains(&f.name));
        self.specialized_blocks
            .retain(|name| used.named.contains(name));
        self.specialized_functions
            .retain(|name| used.named.contains(name));
        let needed_extern = |name: &str| library_item(name).is_none() || used.named.contains(name);
        self.externs.retain(|f| needed_extern(&f.name));
        self.extern_blocks.retain(|b| needed_extern(&b.name));
        used.dslx
    }
}

/// Concrete BX callees, child blocks and DSLX specializations in emitted
/// bodies.
struct Dependencies {
    named: BTreeSet<String>,
    dslx: BTreeSet<usize>,
}

impl Dependencies {
    /// A block can instantiate children and call functions from its wires,
    /// enables and output expression.
    fn block(&mut self, block: &CheckedBlock) {
        let mut expressions = vec![&block.output];
        let mut statements = vec![block.statements.as_slice()];
        while let Some(body) = statements.pop() {
            for stmt in body {
                match stmt {
                    CheckedStmt::Inst { block, .. } => {
                        self.named.insert(block.clone());
                    }
                    CheckedStmt::Let { value, .. }
                    | CheckedStmt::Write { value, .. }
                    | CheckedStmt::Connect { value, .. } => expressions.push(value),
                    CheckedStmt::If { condition, body } => {
                        expressions.push(condition);
                        statements.push(body);
                    }
                    CheckedStmt::Reg { .. } => { /* A register reset is already a concrete value. */
                    }
                }
            }
        }
        self.expressions(expressions);
    }

    /// Follow checked calls, including those in callback and iteration bodies.
    fn expressions(&mut self, mut pending: Vec<&TypedExpr>) {
        while let Some(expr) = pending.pop() {
            match &expr.kind {
                K::Call { name, args } => {
                    self.named.insert(name.clone());
                    pending.extend(args);
                }
                K::ImportedCall { id, args } => {
                    self.dslx.insert(*id);
                    pending.extend(args);
                }
                K::Tuple(items) | K::Array(items) => pending.extend(items),
                K::Index { value, index }
                | K::WidthSlice {
                    value,
                    start: index,
                }
                | K::Binary {
                    left: value,
                    right: index,
                    ..
                } => {
                    pending.extend([value.as_ref(), index.as_ref()]);
                }
                K::TupleIndex { value, .. } | K::Unary { value, .. } | K::Cast { value } => {
                    pending.push(value);
                }
                K::ArrayUpdate {
                    array,
                    index,
                    replacement,
                } => {
                    pending.extend([array.as_ref(), index.as_ref(), replacement.as_ref()]);
                }
                K::Builtin { args, .. } => pending.extend(args),
                K::If {
                    condition,
                    then_value,
                    else_value,
                } => {
                    pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
                }
                K::Match { subject, arms } => {
                    pending.push(subject);
                    for arm in arms {
                        if !arm.patterns.is_empty() {
                            pending.push(&arm.value);
                        }
                        // The first exhaustive arm becomes the lowered default.
                        if matches!(arm.patterns.as_slice(), [ScalarPattern::Wildcard]) {
                            break;
                        }
                    }
                }
                K::Map(map) => {
                    pending.extend([&map.array, &map.output]);
                    pending.extend(map.bindings.iter().map(|b| &b.value));
                }
                K::For(fold) => {
                    pending.push(&fold.seed);
                    if fold.trip_count != 0 {
                        if let TypedIterable::Array(array) = &fold.iterable {
                            pending.push(array);
                        }
                        pending.push(&fold.output);
                        pending.extend(fold.bindings.iter().map(|b| &b.value));
                    }
                }
                K::Name(_)
                | K::BindingTupleIndex { .. }
                | K::InstanceOutput { .. }
                | K::Literal(_) => {
                    // These leaves use values from the surrounding body.
                }
            }
        }
    }
}
