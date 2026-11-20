// SPDX-License-Identifier: Apache-2.0

//! Lower passive PIR connections to SystemVerilog net lvalues.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::inout::{InoutBinding, InoutNetMap, InoutPiece, PublicInoutRange};
use xlsynth_pir::ir::{NodePayload, Type};

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value};

impl BlockEmitter<'_, '_> {
    /// Forms a writable select, retaining the underlying signal for further
    /// selects.
    fn net_select(&mut self, signal: Value, start: usize, width: usize) -> Value {
        if start == 0 && signal.bit_width == Some(width) {
            return signal;
        }
        let expr = self.static_slice(signal, start, width);
        Value::static_slice(expr, signal, start, width)
    }

    /// Selects a public port range, sharing a flat view whenever electrical
    /// indexing must cross a typedef with unknown dimensions or index origin.
    fn public_inout_value(
        &mut self,
        range: PublicInoutRange,
        indexed_by_child: bool,
        flat_ports: &mut BTreeMap<usize, Value>,
    ) -> Result<Value, BlockCodegenError> {
        let NodePayload::ExtInoutPort {
            name,
            port_type,
            sv_type,
            ..
        } = &self.block.get_node(range.port).payload
        else {
            unreachable!("verified inout port")
        };
        let signal = self.ports[name];
        let width = port_type.bit_count();
        let value = if !self.options.emit_sv_types || sv_type.is_none() {
            Value::signal(signal).with_type(port_type)
        } else if range.span.width == width && !indexed_by_child {
            Value::signal(signal).with_width(width)
        } else if let Some(&value) = flat_ports.get(&range.port.index) {
            value
        } else {
            let wire_name = self.unique_name(&format!("{name}_bits"));
            let ty = self.value_type(&Type::Bits(width));
            let wire = self.file.add_wire(self.module, &wire_name, &ty);
            let tie = self.unique_name(&format!("{name}_tie"));
            self.file.add_tran(self.module, &tie, width, signal, wire)?;
            let value = Value::signal(wire).with_width(width);
            flat_ports.insert(range.port.index, value);
            value
        };
        Ok(self.net_select(value, range.span.port_start, range.span.width))
    }

    /// Represents each net by one full public range or by a declared wire
    /// tied bidirectionally to all of its public ranges.
    pub(crate) fn emit_inout_nets(&mut self) -> Result<(), BlockCodegenError> {
        let ranges = InoutNetMap::new(&self.block.inout_nets)
            .and_then(|nets| nets.public_ranges(self.block))
            .map_err(BlockCodegenError::InvalidBlock)?;
        let selected: BTreeSet<_> = self
            .block
            .nodes
            .iter()
            .filter_map(|n| match &n.payload {
                NodePayload::ExtInstantiationInout { net, .. } => Some(net.pieces()),
                _ => None,
            })
            .flatten()
            .filter_map(|p| match p {
                InoutPiece::Select { net, .. } => Some(net.as_str()),
                _ => None,
            })
            .collect();
        let mut flat_ports = BTreeMap::new();
        for (net, bindings) in self.block.inout_nets.iter().zip(ranges) {
            let value = if let [binding] = bindings.as_slice()
                && binding.span.width == net.ty.bit_count()
            {
                self.public_inout_value(
                    *binding,
                    selected.contains(net.name.as_str()),
                    &mut flat_ports,
                )?
            } else {
                let name = self.unique_name(&net.name);
                let ty = self.value_type(&net.ty);
                let wire =
                    Value::signal(self.file.add_wire(self.module, &name, &ty)).with_type(&net.ty);
                for binding in bindings {
                    let pin = self.public_inout_value(binding, false, &mut flat_ports)?;
                    let span = binding.span;
                    let local = self.net_select(wire, span.net_start, span.width);
                    let tie =
                        self.unique_name(&format!("{}_tie", self.block.port_name(binding.port)));
                    self.file.add_tran_selects(
                        self.module,
                        &tie,
                        span.width,
                        local.expr,
                        pin.expr,
                    )?;
                }
                wire
            };
            self.inout_nets.insert(&net.name, value);
        }
        Ok(())
    }

    /// Emits a child connection as an MSB-first lvalue of whole nets and
    /// selects.
    pub(crate) fn instance_inout(&mut self, binding: &InoutBinding) -> Value {
        let mut pieces = Vec::with_capacity(binding.pieces().len());
        for piece in binding.pieces() {
            pieces.push(match piece {
                InoutPiece::Whole(name) => self.inout_nets[name.as_str()],
                InoutPiece::Select { net, hi, lo } => {
                    self.net_select(self.inout_nets[net.as_str()], *lo, hi - lo + 1)
                }
            });
        }
        if let [value] = pieces.as_slice() {
            return *value;
        }
        let parts = pieces.iter().map(|v| &v.expr).collect::<Vec<_>>();
        Value::expression(self.file.make_concat(&parts), 1)
            .with_width(pieces.iter().map(|v| v.bit_width.unwrap()).sum())
    }
}
