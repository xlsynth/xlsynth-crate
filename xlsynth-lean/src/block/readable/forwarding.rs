// SPDX-License-Identifier: Apache-2.0

//! Preserve definitional reduction for source fields wired directly to a
//! result.

use super::{
    Presentation, SourceType, TypeDecl, identifier, lean, product_element, product_value,
    resolve_aliases,
};
use anyhow::{Context, Result};
use xlsynth_pir::ir::{NodePayload, NodeRef};

/// The proof checks each replacement against the actual decoded evaluation.
/// Reduction returns the source field even when a codec cannot reduce on an
/// arbitrary Lean variable. The theorem restores the ordinary decoded result.
pub(super) const LEAN_SUPPORT: &str = r#"attribute [local simp] L.Function.comp_def

private def preserveForward {α : Type} (source decoded : α)
    (_ : source = decoded) : α := source
private theorem preserveForward_eq {α : Type} (source decoded : α) (h : source = decoded) :
    preserveForward source decoded h = decoded := h

"#;

impl Presentation<'_> {
    /// Render a source projection by following only physical tuple indices to
    /// a declared input or register. Struct field order comes from that
    /// boundary declaration.
    fn interface_field(&self, mut wire: NodeRef) -> Result<Option<(String, &SourceType)>> {
        let graph = &self.block.graph;
        let mut indices = Vec::new();
        let (mut field, mut ty) = loop {
            match &graph.get_node(wire).payload {
                NodePayload::TupleIndex { tuple, index } => {
                    indices.push(*index);
                    wire = *tuple;
                }
                NodePayload::InputPort { name, .. } => {
                    break (
                        format!("x.{}", identifier(name)?),
                        &self.interface.inputs[name],
                    );
                }
                NodePayload::RegisterRead { register } => {
                    let Some(ty) = self.interface.registers.get(register) else {
                        return Ok(None);
                    };
                    let index = graph
                        .registers
                        .iter()
                        .position(|r| &r.name == register)
                        .context("checked register read")?;
                    break (format!("q.{}", self.register_fields[index]), ty);
                }
                // Computed wires retain the ordinary decoded result.
                _ => return Ok(None),
            }
        };
        for index in indices.into_iter().rev() {
            match self.unalias(ty)? {
                SourceType::Tuple { members } => {
                    field = product_element(&field, index, members.len(), true);
                    ty = &members[index];
                }
                SourceType::Struct { name } => {
                    let TypeDecl::Struct { fields, .. } = self.declaration(name)? else {
                        unreachable!("validated struct declaration")
                    };
                    let member = &fields[index];
                    field = format!("({field}).{}", identifier(&member.name)?);
                    ty = &member.source_type;
                }
                // Only structs and tuples have physical tuple projections.
                _ => return Ok(None),
            }
        }
        Ok(Some((field, ty)))
    }

    fn forwarded_field(
        &self,
        wire: NodeRef,
        target: &SourceType,
        decoded: &str,
    ) -> Result<Option<String>> {
        // Ordinary bits already reduce through the identity codec.
        if matches!(self.unalias(target)?, SourceType::Bits { .. }) {
            return Ok(None);
        }
        let Some((field, ty)) = self.interface_field(wire)? else {
            return Ok(None);
        };
        if ty != target
            && resolve_aliases(ty, self.declarations)?
                != resolve_aliases(target, self.declarations)?
        {
            return Ok(None);
        }
        let encoded = self.convert(Some(target), &field, true)?;
        let decoded_physical = self.convert(Some(target), "physical", false)?;
        Ok(Some(format!(
            "(preserveForward ({field}) ({decoded}) (by change ({field}) = (let physical := {encoded}; {decoded_physical}); simp))"
        )))
    }

    /// Render the evaluation from its decoded value and checked forwarded
    /// fields. The caller binds `decoded` to the physical evaluator's decoded
    /// result. Conditional writes stay inside that evaluator.
    pub(super) fn forwarded_evaluation(&self) -> Result<Option<String>> {
        let graph = &self.block.graph;
        let mut updates = Vec::new();
        let mut outputs = Vec::new();
        let mut any_output = false;
        for (index, port) in graph.output_ports().enumerate() {
            let decoded =
                product_element("decoded.output", index, self.interface.outputs.len(), false);
            let source = self.forwarded_field(
                graph.output_value(port),
                &self.interface.outputs[index],
                &decoded,
            )?;
            any_output |= source.is_some();
            outputs.push(source.unwrap_or(decoded));
        }
        if any_output {
            updates.push(format!("output := {}", product_value(&outputs, false)));
        }
        let mut state = Vec::new();
        for node in &graph.nodes {
            let NodePayload::RegisterWrite {
                register,
                arg,
                load_enable: None,
                reset: None,
            } = &node.payload
            else {
                continue;
            };
            if let Some(target) = self.interface.registers.get(register) {
                let index = graph
                    .registers
                    .iter()
                    .position(|r| &r.name == register)
                    .context("checked register write")?;
                let field = &self.register_fields[index];
                if let Some(source) =
                    self.forwarded_field(*arg, target, &format!("decoded.next.{field}"))?
                {
                    state.push(format!("{} := {source}", lean::storage_field(field)));
                }
            }
        }
        if !state.is_empty() {
            updates.push(format!(
                "next := {{ decoded.next with {} }}",
                state.join(", ")
            ));
        }
        Ok((!updates.is_empty()).then(|| format!("{{ decoded with {} }}", updates.join(", "))))
    }
}
