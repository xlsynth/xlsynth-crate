// SPDX-License-Identifier: Apache-2.0

//! Tuple binding grammars with independently flattened reference semantics.

use crate::ast::BitsType;
use std::ops::Range;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// The source construct in which a tuple tree binds its retained leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingContext {
    Let,
    AnnotatedLet,
    Match,
}

impl BindingContext {
    pub const ALL: [Self; 3] = [Self::Let, Self::AnnotatedLet, Self::Match];
}

/// A tuple component and its pattern; input indices refer to three bits
/// operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TupleElement {
    Bind(usize),
    Wildcard(usize),
    Tuple(TuplePattern),
}

/// A nonempty tuple and an optional element range replaced by a single `..`.
/// An empty range inserts a rest pattern that skips zero elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuplePattern {
    elements: Vec<TupleElement>,
    rest: Option<Range<usize>>,
}

impl TuplePattern {
    /// Checks input references, rest bounds, and a three-level/64-node budget.
    pub fn new(elements: Vec<TupleElement>, rest: Option<Range<usize>>) -> Result<Self, String> {
        let pattern = Self { elements, rest };
        let mut nodes = 0;
        pattern.validate(0, &mut nodes)?;
        Ok(pattern)
    }

    /// Exposes the structural tuple components, including skipped components.
    pub fn elements(&self) -> &[TupleElement] {
        &self.elements
    }

    /// Returns the skipped element range; zero-length ranges are meaningful.
    pub fn rest(&self) -> Option<Range<usize>> {
        self.rest.clone()
    }

    /// Lists retained input slots in depth-first binding order.
    pub fn bound_inputs(&self) -> Vec<usize> {
        let mut inputs = Vec::new();
        self.collect_inputs(&mut inputs);
        inputs
    }

    /// Bounds the complete tree, including values hidden by a rest pattern.
    fn validate(&self, depth: usize, nodes: &mut usize) -> Result<(), String> {
        if depth >= 3 || self.elements.is_empty() {
            return Err(
                "tuple patterns require nonempty tuples at at most three levels".to_string(),
            );
        }
        if let Some(rest) = &self.rest
            && (rest.start > rest.end || rest.end > self.elements.len())
        {
            return Err("tuple rest range must lie within the tuple".to_string());
        }
        *nodes += 1;
        for element in &self.elements {
            match element {
                TupleElement::Tuple(tuple) => tuple.validate(depth + 1, nodes)?,
                TupleElement::Bind(input) | TupleElement::Wildcard(input) => {
                    if *input >= 3 {
                        return Err("tuple leaves must reference one of three inputs".to_string());
                    }
                    *nodes += 1;
                }
            }
        }
        if *nodes > 64 {
            return Err("tuple patterns allow at most 64 nodes".to_string());
        }
        Ok(())
    }

    /// Collects the semantic inputs without constructing or evaluating DSLX.
    fn collect_inputs(&self, inputs: &mut Vec<usize>) {
        for (index, element) in self.elements.iter().enumerate() {
            if self.rest.as_ref().is_some_and(|rest| rest.contains(&index)) {
                // The entire component, including nested bindings, is skipped.
                continue;
            }
            match element {
                TupleElement::Bind(input) => inputs.push(*input),
                TupleElement::Tuple(tuple) => tuple.collect_inputs(inputs),
                TupleElement::Wildcard(_) => {
                    // A wildcard consumes the value without binding it.
                }
            }
        }
    }

    /// Formats the tuple's value or type with the same nesting and arity.
    fn render_tuple(&self, leaves: [&str; 3]) -> String {
        let elements = self
            .elements
            .iter()
            .map(|element| match element {
                TupleElement::Bind(input) | TupleElement::Wildcard(input) => {
                    leaves[*input].to_string()
                }
                TupleElement::Tuple(tuple) => tuple.render_tuple(leaves),
            })
            .collect::<Vec<_>>();
        format!("({},)", elements.join(", "))
    }

    /// Prints bindings with unique names while retaining wildcards and rest.
    fn render_pattern(&self, next_binding: &mut usize) -> String {
        let mut elements = Vec::new();
        for index in 0..=self.elements.len() {
            if self.rest.as_ref().is_some_and(|rest| rest.start == index) {
                elements.push("..".to_string());
            }
            if index == self.elements.len()
                || self.rest.as_ref().is_some_and(|rest| rest.contains(&index))
            {
                // The sentinel inserts trailing rest; skipped values add no
                // names.
                continue;
            }
            elements.push(match &self.elements[index] {
                TupleElement::Bind(_) => {
                    let name = format!("tuple_bind_{}", *next_binding);
                    *next_binding += 1;
                    name
                }
                TupleElement::Wildcard(_) => "_".to_string(),
                TupleElement::Tuple(tuple) => tuple.render_pattern(next_binding),
            });
        }
        format!("({},)", elements.join(", "))
    }

    /// Makes binding order observable using a modular `3 * acc XOR value` fold.
    pub(crate) fn to_dslx(&self, context: BindingContext, ty: BitsType, args: [&str; 3]) -> String {
        let mut count = 0;
        let pattern = self.render_pattern(&mut count);
        let value = self.render_tuple(args);
        let ty = ty.to_dslx();
        let mut bindings = String::new();
        let mut acc = format!("{ty}:0");
        for index in 0..count {
            let next = format!("tuple_acc_{index}");
            bindings.push_str(&format!(
                "let {next} = ({acc} + {acc} + {acc}) ^ tuple_bind_{index}; "
            ));
            acc = next;
        }
        match context {
            BindingContext::Let | BindingContext::AnnotatedLet => {
                let annotation = if context == BindingContext::AnnotatedLet {
                    format!(": {}", self.render_tuple([&ty; 3]))
                } else {
                    String::new()
                };
                format!("{{ let {pattern}{annotation} = {value}; {bindings}{acc} }}")
            }
            BindingContext::Match => {
                format!("match {value} {{ {pattern} => {{ {bindings}{acc} }}, }}")
            }
        }
    }

    /// Folds retained inputs directly with existing PIR bits operations.
    pub(crate) fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let mut acc = builder.literal(IrValue::from_bits(&IrBits::zero(width)))?;
        for input in self.bound_inputs() {
            let twice = builder.add(acc, acc)?;
            let triple = builder.add(twice, acc)?;
            acc = builder.xor(triple, args[input])?;
        }
        Ok(acc)
    }
}
