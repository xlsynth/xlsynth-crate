// SPDX-License-Identifier: Apache-2.0

//! Lower passive PIR connections to SystemVerilog net lvalues.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::inout::{InoutBinding, InoutNetMap, InoutPiece};
use xlsynth_pir::ir::{NodePayload, Type};

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value};

/// A range of local net bits backed by a named Verilog signal or its select.
#[derive(Clone, Copy)]
pub(crate) struct NetRun {
    start: usize,
    width: usize,
    value: Value,
}

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

    /// Joins MSB-first net lvalues, preserving a single named signal as-is.
    fn net_concat(&mut self, values: &[Value]) -> Value {
        if let [value] = values {
            return *value;
        }
        let parts = values.iter().map(|v| &v.expr).collect::<Vec<_>>();
        Value::expression(self.file.make_concat(&parts), 1)
            .with_width(values.iter().map(|v| v.bit_width.unwrap()).sum())
    }

    pub(crate) fn emit_inout_nets(&mut self) -> Result<(), BlockCodegenError> {
        let nets =
            InoutNetMap::new(&self.block.inout_nets).map_err(BlockCodegenError::InvalidBlock)?;
        let layout = nets
            .public_segments(self.block)
            .map_err(BlockCodegenError::InvalidBlock)?;
        // External typedefs can use multiple packed dimensions or a different
        // index origin. Select bits on a flat wire when a typedef-backed pin
        // needs electrical indexing.
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
        let mut indexed_ports = BTreeSet::new();
        for (net, segments) in nets.nets.iter().zip(&layout) {
            for segment in segments {
                for p in &segment.ports {
                    if segment.width != self.block.port_type(p.port).bit_count()
                        || selected.contains(net.name.as_str())
                    {
                        indexed_ports.insert(p.port.index);
                    }
                }
            }
        }
        let mut port_values = BTreeMap::new();
        for p in self.block.inout_ports() {
            let NodePayload::ExtInoutPort {
                name,
                sv_type,
                port_type,
                ..
            } = &self.block.get_node(p).payload
            else {
                unreachable!("verified inout port")
            };
            let signal = self.ports[name];
            let width = port_type.bit_count();
            let value = if self.options.emit_sv_types && sv_type.is_some() {
                let flat = if indexed_ports.contains(&p.index) {
                    let wire_name = self.unique_name(&format!("{name}_bits"));
                    let ty = self.value_type(&Type::Bits(width));
                    let wire = self.file.add_wire(self.module, &wire_name, &ty);
                    let tie = self.unique_name(&format!("{name}_tie"));
                    self.file.add_tran(self.module, &tie, width, signal, wire)?;
                    wire
                } else {
                    signal
                };
                Value::signal(flat).with_width(width)
            } else {
                Value::signal(signal).with_type(port_type)
            };
            port_values.insert(p.index, value);
        }
        for (net, segments) in nets.nets.iter().zip(layout) {
            let private = if segments.iter().any(|s| s.ports.is_empty()) {
                let name = self.unique_name(&net.name);
                let ty = self.value_type(&net.ty);
                Some(Value::signal(self.file.add_wire(self.module, &name, &ty)).with_type(&net.ty))
            } else {
                None
            };
            let mut runs = Vec::new();
            for segment in segments {
                let value = if let Some((first, aliases)) = segment.ports.split_first() {
                    let value =
                        self.net_select(port_values[&first.port.index], first.start, segment.width);
                    for alias in aliases {
                        let right = self.net_select(
                            port_values[&alias.port.index],
                            alias.start,
                            segment.width,
                        );
                        let tie =
                            self.unique_name(&format!("{}_tie", self.block.port_name(alias.port)));
                        self.file.add_tran_selects(
                            self.module,
                            &tie,
                            segment.width,
                            value.expr,
                            right.expr,
                        )?;
                    }
                    value
                } else {
                    self.net_select(
                        private.expect("private range has a wire"),
                        segment.start,
                        segment.width,
                    )
                };
                runs.push(NetRun {
                    start: segment.start,
                    width: segment.width,
                    value,
                });
            }
            let msb_first: Vec<_> = runs.iter().rev().map(|r| r.value).collect();
            let value = self.net_concat(&msb_first);
            self.inout_nets.insert(&net.name, value);
            self.net_runs.insert(&net.name, runs);
        }
        Ok(())
    }

    /// Emits a child port actual as a net lvalue; no read, driver or expression
    /// temporary is introduced.
    pub(crate) fn instance_inout(&mut self, binding: &InoutBinding) -> Value {
        let mut pieces = Vec::new();
        for piece in binding.pieces() {
            match piece {
                InoutPiece::Whole(name) => pieces.push(self.inout_nets[name.as_str()]),
                InoutPiece::Select { net, hi, lo } => {
                    let runs = &self.net_runs[net.as_str()];
                    // Visit only runs overlapping this select, from MSB to LSB.
                    let end = runs.partition_point(|r| r.start <= *hi);
                    let start = runs[..end].partition_point(|r| r.start + r.width <= *lo);
                    for i in (start..end).rev() {
                        let run = self.net_runs[net.as_str()][i];
                        let a = (*lo).max(run.start);
                        let b = (hi + 1).min(run.start + run.width);
                        pieces.push(self.net_select(run.value, a - run.start, b - a));
                    }
                }
            }
        }
        self.net_concat(&pieces)
    }
}
