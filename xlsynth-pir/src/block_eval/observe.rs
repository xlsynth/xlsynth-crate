// SPDX-License-Identifier: Apache-2.0

use super::prepare::{NodeOrigin, Prepared};
use crate::IrValue;
use crate::corners::{CornerEvent, FailureEvent};
use crate::ir::NodeRef;
use crate::ir_eval::{BoolNodeEvent, EvalObserver, SelectEvent};

/// Translates projected events to original node identities in one block
/// instance.
pub(super) struct BlockObserver<'a> {
    pub instance: &'a str,
    pub prepared: &'a Prepared,
    pub inner: &'a mut dyn EvalObserver,
}

impl BlockObserver<'_> {
    /// Nested function callbacks use local refs and package IDs; only projected
    /// refs match both.
    fn origin(&self, node: NodeRef, text_id: usize) -> Option<&NodeOrigin> {
        let graph = &self.prepared.function;
        if graph.nodes.get(node.index)?.text_id != text_id {
            return None;
        }
        self.prepared.origins[node.index]
            .as_ref()
            .filter(|o| &*o.instance == self.instance)
    }
}

impl EvalObserver for BlockObserver<'_> {
    fn on_select(&mut self, ev: SelectEvent) {
        if let Some(o) = self.origin(ev.node_ref, ev.node_text_id) {
            self.inner.on_select(SelectEvent {
                node_ref: o.node_ref,
                node_text_id: o.text_id,
                ..ev
            });
        }
    }

    fn on_bool_node(&mut self, ev: BoolNodeEvent) {
        if let Some(o) = self.origin(ev.node_ref, ev.node_text_id) {
            self.inner.on_bool_node(BoolNodeEvent {
                node_ref: o.node_ref,
                node_text_id: o.text_id,
                ..ev
            });
        }
    }

    fn on_node_value(&mut self, nr: NodeRef, id: usize, value: &IrValue) {
        if let Some(o) = self.origin(nr, id) {
            self.inner.on_node_value(o.node_ref, o.text_id, value);
        }
    }

    fn on_corner_event(&mut self, ev: CornerEvent) {
        if let Some(o) = self.origin(ev.node_ref, ev.node_text_id) {
            self.inner.on_corner_event(CornerEvent {
                node_ref: o.node_ref,
                node_text_id: o.text_id,
                ..ev
            });
        }
    }

    fn on_failure_event(&mut self, ev: FailureEvent) {
        if let Some(o) = self.origin(ev.node_ref, ev.node_text_id) {
            self.inner.on_failure_event(FailureEvent {
                node_ref: o.node_ref,
                node_text_id: o.text_id,
                ..ev
            });
        }
    }
}
