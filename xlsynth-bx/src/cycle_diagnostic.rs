// SPDX-License-Identifier: Apache-2.0

//! Project a Block IR cycle onto the BX connections that contributed it.

use std::collections::{BTreeMap, BTreeSet};

use annotate_snippets::{AnnotationKind, Group, Level, Renderer, Snippet};
use xlsynth_pir::BuilderError;

use crate::ast::SourceSpan;
use crate::typed::CheckedBlock;

/// A deferred wire's source name and the span of its driver.
pub(crate) struct WireDriver {
    pub(crate) name: String,
    pub(crate) span: SourceSpan,
}

/// Source attribution keyed by the block builder's node IDs.
#[derive(Default)]
pub(crate) struct CycleSources {
    pub(crate) wires: BTreeMap<usize, WireDriver>,
    pub(crate) inputs: BTreeMap<usize, SourceSpan>,
}

impl CycleSources {
    /// Render a cycle from this block while builder IDs still identify its
    /// source connections. A cycle inside a linked child has its own IDs.
    pub(crate) fn render(
        &self,
        error: &BuilderError,
        block: &CheckedBlock,
        ir_block: &str,
    ) -> Option<String> {
        let BuilderError::CombinationalCycle {
            block: cycle_block,
            steps,
        } = error
        else {
            return None;
        };
        if cycle_block != ir_block {
            return None;
        }
        let source = block.source.as_ref()?;

        let mut wire_sites = Vec::new();
        let mut input_sites = Vec::new();
        let mut path = Vec::new();
        for step in steps {
            if let Some(wire) = self.wires.get(&step.node_id) {
                path.push(wire.name.clone());
                wire_sites.push((&wire.span, format!("driver of `{}`", wire.name)));
            }
            if let Some(connection) = &step.connection {
                let input = format!("{}.{}", connection.instance, connection.input_port);
                let output = format!("{}.{}", connection.instance, connection.output_port);
                if let Some(span) = self.inputs.get(&connection.input_node_id) {
                    input_sites.push((span, format!("`{output}` depends on `{input}`")));
                }
                path.push(output);
                path.push(input);
            }
        }
        let first = path.first()?.clone();
        path.push(first);
        let filename = source
            .path
            .as_ref()
            .map_or_else(|| "<bx>".into(), |path| path.to_string_lossy());
        let mut snippet = Snippet::source(source.text.as_ref())
            .path(filename)
            .fold(true);
        let mut seen = BTreeSet::new();
        for (span, label) in wire_sites.into_iter().chain(input_sites) {
            if source.text.get(span.start..span.end).is_some()
                && seen.insert((span.start, span.end))
            {
                let kind = if seen.len() == 1 {
                    AnnotationKind::Primary
                } else {
                    AnnotationKind::Context
                };
                snippet = snippet.annotation(kind.span(span.start..span.end).label(label));
            }
        }
        if seen.is_empty() {
            return None;
        }
        let block_name = crate::bx_import::diagnostic(block.source_name.clone());
        let report = Group::with_title(
            Level::ERROR.primary_title(format!("combinational cycle in block '{block_name}'")),
        )
        .element(snippet)
        .element(Level::NOTE.message(format!("dependency cycle: {}", path.join(" -> "))));
        Some(Renderer::plain().render(&[report]).to_string())
    }
}
