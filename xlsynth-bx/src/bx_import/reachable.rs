// SPDX-License-Identifier: Apache-2.0

//! Select the checked library definitions needed by the root BX module.

use std::collections::BTreeSet;

use super::library_item;
use crate::typed::{
    CheckedBinding, CheckedBlock, CheckedFunction, CheckedModule, CheckedStmt, TypedExpr,
    TypedExprKind as K, TypedIterable, TypedMatchPattern,
};

impl CheckedModule {
    /// Whether these hardware roots reach a function with extra pure-test
    /// observations. Follow the same expressions and calls hardware emits.
    pub(crate) fn blocks_reach_test_discards(&self, roots: impl Iterator<Item = String>) -> bool {
        let mut used = Dependencies {
            named: roots.collect(),
            ..Default::default()
        };
        used.visit_blocks(&self.blocks);
        for function in self.functions.iter().rev() {
            if used.named.contains(&function.name) {
                if self.functions_with_test_discards.contains(&function.name) {
                    return true;
                }
                used.function(function, false);
            }
        }
        false
    }

    /// Select the functions whose discarded fields execute in a pure test.
    /// Visit callers before their callees, including calls inside those fields.
    pub(crate) fn resolve_pure_test_dependencies(&mut self, harnesses: BTreeSet<String>) {
        let mut used = Dependencies {
            named: harnesses,
            ..Default::default()
        };
        for function in self.functions.iter().rev() {
            if used.named.contains(&function.name) {
                used.has_test_discards = false;
                used.function(function, true);
                if used.has_test_discards {
                    self.functions_with_test_discards
                        .insert(function.name.clone());
                }
            }
        }
    }

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
            ..Default::default()
        };
        // Both collections are already in dependency order. Visit each
        // caller before its children, then visit functions called by blocks.
        used.visit_blocks(&self.blocks);
        for function in self.functions.iter().rev() {
            if used.named.contains(&function.name) {
                used.function(
                    function,
                    self.functions_with_test_discards.contains(&function.name),
                );
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
#[derive(Default)]
struct Dependencies {
    named: BTreeSet<String>,
    dslx: BTreeSet<usize>,
    has_test_discards: bool,
}

impl Dependencies {
    /// Checked blocks are ordered child-first. Walking backwards discovers
    /// each reachable child before its turn and gathers the functions it calls.
    fn visit_blocks(&mut self, blocks: &[CheckedBlock]) {
        for block in blocks.iter().rev() {
            if self.named.contains(&block.name) {
                self.block(block);
            }
        }
    }

    /// Follow a function's result and the bindings that its lowering emits.
    fn function(&mut self, function: &CheckedFunction, evaluate_discards: bool) {
        let mut expressions = vec![&function.output];
        self.bindings(&function.bindings, evaluate_discards, &mut expressions);
        self.expressions(expressions, evaluate_discards);
    }

    /// Queue the emitted initializers and record any observed test discard.
    fn bindings<'a>(
        &mut self,
        bindings: &'a [CheckedBinding],
        evaluate_discards: bool,
        pending: &mut Vec<&'a TypedExpr>,
    ) {
        for binding in bindings {
            if binding.is_test_discard() {
                if !evaluate_discards {
                    continue;
                }
                self.has_test_discards = true;
            }
            pending.push(&binding.value);
        }
    }

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
                    | CheckedStmt::DriveWire { value, .. }
                    | CheckedStmt::Write { value, .. }
                    | CheckedStmt::Connect { value, .. } => expressions.push(value),
                    CheckedStmt::If { condition, body } => {
                        expressions.push(condition);
                        statements.push(body);
                    }
                    CheckedStmt::Reg { .. } | CheckedStmt::Wire { .. } => {
                        // Only runtime expressions can introduce dependencies.
                    }
                }
            }
        }
        self.expressions(expressions, false);
    }

    /// Follow checked calls, including those in callback and iteration bodies.
    fn expressions(&mut self, mut pending: Vec<&TypedExpr>, evaluate_discards: bool) {
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
                        if arm.patterns.iter().any(TypedMatchPattern::is_irrefutable) {
                            break;
                        }
                    }
                }
                K::Map(map) => {
                    pending.extend([&map.array, &map.output]);
                    self.bindings(&map.bindings, evaluate_discards, &mut pending);
                }
                K::For(fold) => {
                    pending.push(&fold.seed);
                    if fold.trip_count != 0 {
                        if let TypedIterable::Array(array) = &fold.iterable {
                            pending.push(array);
                        }
                        pending.push(&fold.output);
                        self.bindings(&fold.bindings, evaluate_discards, &mut pending);
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
